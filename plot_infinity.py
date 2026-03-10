#!/usr/bin/env python3
"""
Plot centrality bias from InfinityMirrorTest logs.
One PNG per dataset: all models across iterations, replace vs add side by side.

Usage:
  python plot_infinity.py logs/LogsIMT/
  python plot_infinity.py logs/LogsIMT/ --out plots/
"""
import sys, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['boxe', 'complex', 'hake_both', 'hole', 'quate',
          'rotate', 'rotpro', 'toruse_eL2', 'transe', 'tucker']
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
MODEL_COLOR = dict(zip(MODELS, COLORS))


def parse_logs(log_dir):
    """Returns dict: data[dataset][mode][model] = {iter: pct_higher}"""
    data = defaultdict(lambda: defaultdict(dict))

    for entry in os.scandir(log_dir):
        if not entry.name.endswith('.out'):
            continue
        cfg = {}
        bias = defaultdict(dict)
        with open(entry.path) as f:
            for line in f:
                line = line.strip()
                m = re.match(r'\[CONFIG\] dataset=(\S+) model=(\S+) mode=(\S+)', line)
                if m:
                    cfg = {'dataset': m.group(1), 'model': m.group(2), 'mode': m.group(3)}
                m = re.match(r'\[BIAS\] iteration=(\d+) type=overall\s+pct_higher=([\d.]+)%', line)
                if m:
                    bias[int(m.group(1))] = float(m.group(2))
        if cfg and bias:
            data[cfg['dataset']][cfg['mode']][cfg['model']] = dict(bias)

    return data


def make_plots(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for ds in sorted(data):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(f'{ds} — Centrality bias across iterations', fontsize=13, fontweight='bold')

        for ax, mode in zip(axes, ['replace', 'add']):
            mode_data = data[ds].get(mode, {})

            for model in MODELS:
                if model not in mode_data:
                    continue
                iters_dict = mode_data[model]
                iters = sorted(iters_dict)
                vals  = [iters_dict[i] for i in iters]
                color = MODEL_COLOR.get(model, 'black')
                ax.plot(iters, vals, 'o-', color=color, label=model,
                        linewidth=1.8, markersize=4)
                ax.annotate(f'{vals[-1]:.0f}',
                            xy=(iters[-1], vals[-1]),
                            xytext=(3, 0), textcoords='offset points',
                            fontsize=6.5, color=color)

            # 50% threshold
            ax.axhline(50, color='black', linestyle='--', linewidth=1,
                       alpha=0.5, label='50% (unbiased)')

            ax.set_title(f'mode = {mode}', fontsize=11)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('% predictions with higher PageRank than real entity')
            ax.set_xticks(range(1, 6))
            ax.set_ylim(20, 80)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc='upper left', ncol=2)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f'bias_{ds}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[SAVED] {out_path}')


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs/LogsIMT'
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else '.'
    data = parse_logs(log_dir)
    print(f'Parsed {sum(len(m) for ds in data.values() for m in ds.values())} experiments '
          f'across {len(data)} datasets.')
    make_plots(data, out_dir)


if __name__ == '__main__':
    main()
