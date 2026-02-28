#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# ── Paths ──────────────────────────────────────────────
# AugmentedKGE and datasets are included in the repo (no change needed).
# Set the paths to your model folders below:
NELL_MODEL_DIR     = os.path.join(os.path.dirname(__file__), 'Model/3/')
FB15K237_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Model/2/')
# ───────────────────────────────────────────────────────

ALL_DATASETS = ['nell', 'fb15k237']
ALL_MODELS   = ['transe', 'rotate', 'complex', 'boxe']
ALL_MODES    = ['replace', 'add']

def ask(prompt, options):
    options_with_all = options + ['all']
    for i, o in enumerate(options_with_all):
        print(f"  [{i}] {o}")
    while True:
        val = input(f"{prompt} (0-{len(options_with_all)-1}): ").strip()
        if val.isdigit() and int(val) < len(options_with_all):
            chosen = options_with_all[int(val)]
            return options if chosen == 'all' else [chosen]

ENV = {**os.environ, 'NELL_MODEL_DIR': NELL_MODEL_DIR, 'FB15K237_MODEL_DIR': FB15K237_MODEL_DIR}

def run(cmd, log=None):
    if log:
        with open(log, 'w') as f:
            subprocess.run([PY] + cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=ENV)
        print(f"  -> {log}")
    else:
        subprocess.run([PY] + cmd, check=True, env=ENV)

print("\nDatasets:"); datasets = ask("Choose", ALL_DATASETS)
print("\nModels:");   models   = ask("Choose", ALL_MODELS)
print("\nModes:");    modes    = ask("Choose", ALL_MODES)
iters = input("\nIterations (default 5): ").strip() or "5"

print()
for ds in datasets:
    for m in models:
        for mode in modes:
            print(f"{ds} | {m} | {mode}")
            run(['infinity_mirror.py', m, mode, iters, ds], log=f'{ds}_{m}_{mode}.log')
