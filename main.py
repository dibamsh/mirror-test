#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# put your model folder paths here
NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

print("dataset  : nell / fb15k237 / all")
print("model    : transe / rotate / complex / boxe / all")
print("mode     : replace / add / all")
print("iterations: any number (e.g. 5)")
print()

ds    = input("dataset: ").strip()
model = input("model: ").strip()
mode  = input("mode: ").strip()
iters = input("iterations: ").strip()

datasets = ['nell', 'fb15k237']  if ds    == 'all' else [ds]
models   = ['transe', 'rotate', 'complex', 'boxe'] if model == 'all' else [model]
modes    = ['replace', 'add']    if mode  == 'all' else [mode]

env = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

print()
for ds in datasets:
    for m in models:
        for md in modes:
            log = f'{ds}_{m}_{md}.log'
            print(f"running {ds} {m} {md} ...")
            with open(log, 'w') as f:
                subprocess.run([sys.executable, 'infinity_mirror.py', m, md, iters, ds],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
            print(f"  saved {log}")
