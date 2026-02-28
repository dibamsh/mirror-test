#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

dataset = input("dataset (nell / fb15k237)? ").strip()
model   = input("model (transe / rotate / complex / boxe)? ").strip()
mode    = input("mode (replace / add)? ").strip()
iters   = input("iterations? ").strip()

env = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

log = f'{dataset}_{model}_{mode}.log'
print(f"running {dataset} {model} {mode} {iters} iters ...")
with open(log, 'w') as f:
    subprocess.run([sys.executable, 'infinity_mirror.py', model, mode, iters, dataset],
                   stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
print(f"done -> {log}")
