import os
import pickle
import requests
import numpy as np
import re
import torch

dataset = 'nanompc'
data_dir = os.path.join('data', dataset)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

text_file = os.path.join(data_dir, 'nanompc_dataset.txt')
with open(text_file, 'r') as f:
    text_data = f.read()
print(f"dataset length: {len(text_data):,}")

tokenizer = re.compile(r'000000000000|\d{2}|\n')
matches = tokenizer.findall(text_data)
unique_pairs = sorted(set(matches), key=matches.index)

vocab_size = len(unique_pairs)
print('vocab size:', vocab_size)

def init_stoi(unique_pairs):
    return {ch: i for i, ch in enumerate(unique_pairs)}

stoi = init_stoi(unique_pairs)
itos = {i: ch for ch, i in stoi.items()}

def encode(text):
    matches = tokenizer.findall(text)
    return [stoi[c] for c in matches]

def decode(encoded):
    return ''.join([itos[i] for i in encoded])

data = torch.tensor(encode(text_data), dtype=torch.long)

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'val.bin')
train_ids.tofile(train_path)
val_ids.tofile(val_path)

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

meta_path = os.path.join(data_dir, 'meta.pkl')

with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print('dataset completed')
