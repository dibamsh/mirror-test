#!/usr/bin/env python3
import subprocess, sys, os, argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['nell', 'fb15k237'], required=True)
parser.add_argument('--model',   choices=['transe', 'rotate', 'complex', 'boxe'])
parser.add_argument('--mode',    choices=['replace', 'add'])
parser.add_argument('--n_iters', type=int, required=True)
args = parser.parse_args()

models = [args.model] if args.model else ['transe', 'rotate', 'complex', 'boxe']
modes  = [args.mode]  if args.mode  else ['replace', 'add']

env = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

for model in models:
    for mode in modes:
        log = f'{args.dataset}_{model}_{mode}.log'
        print(f"running {args.dataset} {model} {mode} {args.n_iters} iters ...")
        with open(log, 'w') as f:
            subprocess.run([sys.executable, 'infinity_mirror.py', model, mode, str(args.n_iters), args.dataset],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
        print(f"  done -> {log}")
