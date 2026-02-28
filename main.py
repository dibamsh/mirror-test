#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# put your model folder paths here
NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

ALL_DATASETS = ['nell', 'fb15k237']
ALL_MODELS   = ['transe', 'rotate', 'complex', 'boxe']
ALL_MODES    = ['replace', 'add']

def ask(prompt, options):
    for i, o in enumerate(options + ['all']):
        print(f"  {i}: {o}")
    while True:
        v = input(f"{prompt}: ").strip()
        if v.isdigit() and int(v) <= len(options):
            return options if int(v) == len(options) else [options[int(v)]]

ENV = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

def run(cmd, log):
    with open(log, 'w') as f:
        subprocess.run([PY] + cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=ENV)
    print(f"  saved {log}")

datasets = ask("dataset", ALL_DATASETS)
models   = ask("model",   ALL_MODELS)
modes    = ask("mode",    ALL_MODES)
iters    = input("iterations (default 5): ").strip() or "5"

print()
for ds in datasets:
    for m in models:
        for mode in modes:
            print(f"running {ds} | {m} | {mode}")
            run(['infinity_mirror.py', m, mode, iters, ds], log=f'{ds}_{m}_{mode}.log')
