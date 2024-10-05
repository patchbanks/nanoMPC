"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
from tabulate import tabulate
import torch
import tiktoken
from model import GPTConfig, GPT
import pandas as pd
import random
import re
import pretty_midi


init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'checkpoints' # ignored if init_from is not 'resume'
ckpt_load = 'model.pt'

midi_dir = 'midi_output'
os.makedirs(midi_dir, exist_ok=True)

start = "000000000000\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 1152 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 8 # retain only the top_k most likely tokens, clamp others to have 0 probability

seed = random.randint(1, 100000)
torch.manual_seed(seed)
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, ckpt_load)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

##############################################################################################################

tokenizer = re.compile(r'000000000000|\d{1}|\n')

load_meta = True
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi = meta.get('stoi', None)
    itos = meta.get('itos', None)
    vocab_size = meta.get('vocab_size', None)

else:
    print(f"meta file not found at {meta_path}")

def encode(text):
    matches = tokenizer.findall(text)
    return [stoi[c] for c in matches]

def decode(encoded):
    return ''.join([itos[i] for i in encoded])


def clear_midi(dir):
    for file in os.listdir(dir):
        if file.endswith('.mid'):
            os.remove(os.path.join(dir, file))

clear_midi(midi_dir)

print("generating...")

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


midi_events = []
seq_count = 0


with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            sequence = []
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            tkn_seq = decode(y[0].tolist())

            if not tkn_seq.startswith(start):
                tkn_seq = start + tkn_seq

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
                        p = 0
                        v = 0
                        s = 0
                        e = 0

                    sequence.append({
                        'file_name': f'nanompc_{seq_count:04d}',
                        'pitch': p,
                        'velocity': v,
                        'start': s,
                        'end': e
                    })

            if sequence:
                midi_events.append(sequence)

midi_data = pd.DataFrame([pd.Series(event) for sequence in midi_events for event in sequence])
midi_data = midi_data[['file_name', 'pitch', 'velocity', 'start', 'end']]
midi_data = midi_data.sort_values(by=['file_name', 'start']).reset_index(drop=True)

trim_4br = midi_data[(midi_data['start'] < 1536) & (midi_data['end'] <= 1536)]
filter_bars = trim_4br.reset_index(drop=True)
start_max = filter_bars['start'].max()
end_max = filter_bars['end'].max()


def write_midi(midi_data):
    midi_events_by_file = {}

    for index, event in midi_data.iterrows():
        file_name = event['file_name']
        if file_name not in midi_events_by_file:
            midi_events_by_file[file_name] = []
        midi_events_by_file[file_name].append(event)

    for file_name, events in midi_events_by_file.items():
        
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=96)
        midi_data.time_signature_changes.append(pretty_midi.containers.TimeSignature(4, 4, 0))
        instrument = pretty_midi.Instrument(0)
        midi_data.instruments.append(instrument)
        
        for event in events:
            pitch = event['pitch']
            velocity = event['velocity']
            start = midi_data.tick_to_time(event['start'])
            end = midi_data.tick_to_time(event['end'])
            note = pretty_midi.Note(pitch=pitch, velocity=velocity, start=start, end=end)
            instrument.notes.append(note)
        
        midi_path = os.path.join(midi_dir, file_name + '.mid')
        midi_data.write(midi_path)

write_midi(midi_data)


def delete_tiny_files(output_dir):
    file_count = 0
    for root, _, files in os.walk(output_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith('.mid') and os.path.getsize(file_path) <= 300:  # 300 bytes
                os.remove(file_path)
                file_count += 1
    return file_count

deleted_files = delete_tiny_files(midi_dir)

print("completed")