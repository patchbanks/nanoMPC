"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser(description="nanoMPC")
parser.add_argument("--bpm", type=int, default=90, help="Beats per minute")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
args, unknown_args = parser.parse_known_args()  # Capture unknown args for configurator

bpm = args.bpm
num_samples = args.num_samples

for arg in unknown_args:
    if arg.startswith('--'):
        print(f"Skipping command-line argument: {arg}")
        continue

    if '=' not in arg:
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        key, val = arg.split('=')
        key = key[2:]

        if key in globals():
            if key in ['bpm', 'num_samples']:
                continue
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")


