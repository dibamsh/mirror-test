#!/usr/bin/env python3
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MODELS = ['transe', 'rotate']

for m in MODELS:
    print(f"\n{'='*60}\n  PREDICTIONS: {m}\n{'='*60}")
    subprocess.run([PY, 'top5_predictions.py', m], check=True)
    print(f"\n{'='*60}\n  PAGERANK: {m}\n{'='*60}")
    with open(f'{m}.log', 'w') as f:
        subprocess.run([PY, 'compare_pagerank.py', m], stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"  -> saved {m}.log")

print(f"\n{'='*60}\n  PARSING BIAS\n{'='*60}")
subprocess.run([PY, 'parse_bias.py'] + [f'{m}.log' for m in MODELS], check=True)
