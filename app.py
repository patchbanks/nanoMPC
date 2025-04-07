import os
import pickle
import torch
import random
import subprocess
import re
import pretty_midi
import gradio as gr
from contextlib import nullcontext
from model import GPTConfig, GPT
from pedalboard import Pedalboard, Reverb, Compressor, Gain, Limiter
from pedalboard.io import AudioFile
import gradio as gr


in_space = os.getenv("SYSTEM") == "spaces"

temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)

init_from = 'resume'
out_dir = 'checkpoints'
ckpt_load = 'model.pt'

start = "000000000000\n"
num_samples = 1
max_new_tokens = 768

seed = random.randint(1, 100000)
torch.manual_seed(seed)
device = 'cpu' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cpu' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, ckpt_load)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

tokenizer = re.compile(r'000000000000|\d{2}|\n')

dataset = checkpoint['config']['dataset']
meta_path = os.path.join('data', dataset, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
    stoi = meta.get('stoi', None)
    itos = meta.get('itos', None)

def encode(text):
    matches = tokenizer.findall(text)
    return [stoi[c] for c in matches]

def decode(encoded):
    return ''.join([itos[i] for i in encoded])

def clear_midi(dir):
    for file in os.listdir(dir):
        if file.endswith('.mid'):
            os.remove(os.path.join(dir, file))

clear_midi(temp_dir)


def generate_midi(temperature, top_k):
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        
    midi_events = []
    seq_count = 0

    with torch.no_grad():
        for _ in range(num_samples):
            sequence = []
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            tkn_seq = decode(y[0].tolist())
            lines = tkn_seq.splitlines()

            for event in lines:
                if event.startswith(start.strip()):
                    if sequence:
                        midi_events.append(sequence)
                        sequence = []
                    seq_count += 1
                elif event.strip() == "":
                    continue
                else:
                    try:
                        p = int(event[0:2])
                        v = int(event[2:4])
                        s = int(event[4:8])
                        e = int(event[8:12])
                    except ValueError:
                        p, v, s, e = 0, 0, 0, 0
                    sequence.append({'file_name': f'nanompc_{seq_count:02d}', 'pitch': p, 'velocity': v, 'start': s, 'end': e})

            if sequence:
                midi_events.append(sequence)

    round_bars = []
    
    for sequence in midi_events:
        filtered_sequence = []
        for event in sequence:
            if event['start'] < 1536 and event['end'] <= 1536:
                filtered_sequence.append(event)
        if filtered_sequence:
            round_bars.append(filtered_sequence)

    midi_events = round_bars

    for track in midi_events:
        track.sort(key=lambda x: x['start'])
        unique_notes = []
        
        for note in track:
            if not any(abs(note['start'] - n['start']) < 12 and note['pitch'] == n['pitch'] for n in unique_notes):
                unique_notes.append(note)
        
        track[:] = unique_notes 

    return midi_events


def write_midi(midi_events, bpm):
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=96)
    midi_data.time_signature_changes.append(pretty_midi.containers.TimeSignature(4, 4, 0))
    instrument = pretty_midi.Instrument(0)
    midi_data.instruments.append(instrument)

    for sequence in midi_events:
        for event in sequence:
            pitch = event['pitch']
            velocity = event['velocity']
            start = midi_data.tick_to_time(event['start'])
            end = midi_data.tick_to_time(event['end'])
            note = pretty_midi.Note(pitch=pitch, velocity=velocity, start=start, end=end)
            instrument.notes.append(note)

    midi_path = os.path.join(temp_dir, 'output.mid')
    midi_data.write(midi_path)
    print(f"Generated: {midi_path}")


def render_wav(midi_file, uploaded_sf2=None, output_level='2.0'):
    sf2_dir = 'sf2_kits'
    audio_format = 's16'
    sample_rate = '44100'
    gain = str(output_level)

    if uploaded_sf2:
        sf2_file = uploaded_sf2
    else:
        sf2_files = [f for f in os.listdir(os.path.join(sf2_dir, dataset)) if f.endswith('.sf2')]
        if not sf2_files:
            raise ValueError("No SoundFont (.sf2) file found in directory.")
        sf2_file = os.path.join(sf2_dir, dataset, random.choice(sf2_files))

    print(f"Using SoundFont: {sf2_file}")
    output_wav = os.path.join(temp_dir, 'output.wav')

    with open(os.devnull, 'w') as devnull:
        command = [
            'fluidsynth', '-ni', sf2_file, midi_file, '-F', output_wav, '-r', str(sample_rate), 
            '-o', f'audio.file.format={audio_format}', '-g', str(gain)
        ]
        subprocess.call(command, stdout=devnull, stderr=devnull)

    return output_wav


def generate_and_return_files(bpm, temperature, top_k, uploaded_sf2=None, output_level='2.0'):
    midi_events = generate_midi(temperature, top_k)  
    if not midi_events:
        return "Error generating MIDI.", None, None
    
    write_midi(midi_events, bpm)
    
    midi_file = os.path.join(temp_dir, 'output.mid')
    wav_raw = render_wav(midi_file, uploaded_sf2, output_level)
    wav_fx = os.path.join(temp_dir, 'output_fx.wav')

    sfx_settings = [
        {
            'board': Pedalboard([
                Reverb(room_size=0.01, wet_level=random.uniform(0.005, 0.01), dry_level=0.75, width=1.0),
                Compressor(threshold_db=-3.0, ratio=8.0, attack_ms=0.0, release_ms=300.0),
            ])
        }
    ]

    for setting in sfx_settings:
        board = setting['board']

        with AudioFile(wav_raw) as f:
            with AudioFile(wav_fx, 'w', f.samplerate, f.num_channels) as o:
                while f.tell() < f.frames:
                    chunk = f.read(int(f.samplerate))
                    effected = board(chunk, f.samplerate, reset=False)
                    o.write(effected)

    return midi_file, wav_fx


custom_css = """
#container {
  max-width: 1200px !important;
  margin: 0 auto !important;
}
#generate-btn {
  font-size: 18px;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  background: linear-gradient(90deg, hsla(268, 90%, 70%, 1) 0%, hsla(260, 72%, 74%, 1) 50%, hsla(247, 73%, 65%, 1) 100%);
  transition: background 1s ease;
}
#generate-btn:hover {
  color: white;
  background: linear-gradient(90deg, hsla(268, 90%, 62%, 1) 0%, hsla(260, 70%, 70%, 1) 50%, hsla(247, 73%, 55%, 1) 100%);
}
#container .prose {
  text-align: center !important;
}
#container h1 {
  font-weight: bold;
  font-size: 40px;
  margin: 0px;
}
#container p {
  font-size: 18px;
  text-align: center;
}

"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Default(
        font=[gr.themes.GoogleFont("Roboto"), "sans-serif"],
        primary_hue="violet",
        secondary_hue="violet"
    )
) as iface:
    with gr.Column(elem_id="container"):
        gr.Markdown("<h1 style='font-weight: bold; text-align: center;'>nanoMPC</h1>")
        bpm = gr.Slider(minimum=50, maximum=200, step=1, value=120, label="BPM")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature")
        top_k = gr.Slider(minimum=4, maximum=16, step=1, value=8, label="Top-k")
        output_level = gr.Slider(minimum=0, maximum=3, step=0.10, value=2.0, label="Output Gain")
        generate_button = gr.Button("Generate", elem_id="generate-btn")
        midi_file = gr.File(label="MIDI Output")
        audio_file = gr.Audio(label="Audio Output", type="filepath")
        soundfont = gr.File(label="Optional: Upload SoundFont (preset=0, bank=0)")

        generate_button.click(
            fn=generate_and_return_files,
            inputs=[bpm, temperature, top_k, soundfont, output_level],
            outputs=[midi_file, audio_file]
        )

        gr.Markdown("<p style='font-size: 16px;'>Developed by <a href='https://www.patchbanks.com/' target='_blank'><strong>Patchbanks</strong></a></p>")

iface.launch(share=True)
