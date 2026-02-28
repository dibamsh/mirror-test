#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'

iters = input("how many iterations? ").strip()

env = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

for dataset in ['nell', 'fb15k237']:
    for model in ['transe', 'rotate', 'complex', 'boxe']:
        for mode in ['replace', 'add']:
            log = f'{dataset}_{model}_{mode}.log'
            print(f"running {dataset} {model} {mode} ...")
            with open(log, 'w') as f:
                subprocess.run([sys.executable, 'infinity_mirror.py', model, mode, iters, dataset],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
            print(f"  -> {log}")
