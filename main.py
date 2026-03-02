#!/usr/bin/env python3
import subprocess, sys, os, argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['nell', 'fb15k237'], required=True)
parser.add_argument('--model',   choices=['transe', 'rotate', 'complex', 'boxe'], required=True)
parser.add_argument('--mode',    choices=['replace', 'add'], required=True)
parser.add_argument('--n_iters', type=int, required=True)
args = parser.parse_args()

env = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

log = f'{args.dataset}_{args.model}_{args.mode}.log'
print(f"running {args.dataset} {args.model} {args.mode} {args.n_iters} iters ...")
with open(log, 'w') as f:
    subprocess.run([sys.executable, 'infinity_mirror.py', args.model, args.mode, str(args.n_iters), args.dataset],
                   stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
print(f"done -> {log}")
