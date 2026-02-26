#!/usr/bin/env python3
"""Parse infinity mirror logs and plot centrality bias evolution across iterations."""
import sys, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def parse(path):
    model, mode, data = None, None, defaultdict(dict)
    with open(path) as f:
        for line in f:
            m = re.search(r'\[CONFIG\] model=(\S+)', line)
            if m: model = m.group(1)
            m = re.search(r'\[MODE\] approach=(\S+)', line)
            if m: mode = m.group(1)

            m = re.search(r'\[ITER\] iteration=(\d+) new_triples=(\d+) total_triples=(\d+)', line)
            if m:
                it = int(m.group(1))
                data[it]['new_triples']    = int(m.group(2))
                data[it]['total_triples']  = int(m.group(3))

            m = re.search(r'\[BIAS\] iteration=(\d+) type=overall\s+.*?pct_higher=([\d.]+)%\s+signed_diff=([+\-\d.eE]+)\s+mean_abs=([\d.eE]+)', line)
            if m:
                it = int(m.group(1))
                data[it]['pct_higher']  = float(m.group(2))
                data[it]['signed_diff'] = float(m.group(3))
                data[it]['mean_err']    = float(m.group(4))

            m = re.search(r'\[BIAS\] iteration=(\d+) type=tail\s+.*?pct_higher=([\d.]+)%.*mean_abs=([\d.eE]+)', line)
            if m: data[int(m.group(1))]['tail_err'] = float(m.group(3))

            m = re.search(r'\[BIAS\] iteration=(\d+) type=head\s+.*?pct_higher=([\d.]+)%.*mean_abs=([\d.eE]+)', line)
            if m: data[int(m.group(1))]['head_err'] = float(m.group(3))

            m = re.search(r'\[HITS\] iteration=(\d+) tail_hits@5=([\d.]+)% head_hits@5=([\d.]+)%', line)
            if m:
                it = int(m.group(1))
                data[it]['tail_hits'] = float(m.group(2))
                data[it]['head_hits'] = float(m.group(3))

    return model or path, mode or 'unknown', dict(data)


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_infinity.py <log1> [log2]"); sys.exit(1)

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    all_data = {}
    for i, path in enumerate(sys.argv[1:]):
        model, mode, data = parse(path)
        key = f"{model}[{mode}]"
        all_data[key] = data
        iters = sorted(data.keys())
        print(f"[PARSED] model={model} mode={mode} iterations={iters}")
        for it in iters:
            d = data[it]
            print(f"  iter={it} new={d.get('new_triples','?')} total={d.get('total_triples','?')} "
                  f"pct_higher={d.get('pct_higher','?')}% signed={d.get('signed_diff','?')}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Infinity Mirror Test — KGE Centrality Bias Over Iterations\n'
        'NELL-995 · top-1 predictions · replace = facts replaced by predictions · add = known set grows',
        fontsize=12, fontweight='bold'
    )

    for idx, (model, data) in enumerate(all_data.items()):
        color = palette[idx % len(palette)]
        iters = sorted(k for k in data if k > 0)

        def vals(key, iters=iters, data=data):
            return [data[it].get(key) for it in iters]

        # 1. Centrality bias (pct_higher)
        ax = axes[0, 0]
        v = vals('pct_higher')
        ax.plot(iters, v, 'o-', color=color, label=model, linewidth=2)
        # annotate last point
        if v and v[-1] is not None:
            ax.annotate(f'{v[-1]:.1f}%', xy=(iters[-1], v[-1]),
                        xytext=(4, 2), textcoords='offset points', fontsize=7, color=color)

        # 2. Signed diff (mean PR(pred) - PR(real))
        ax = axes[0, 1]
        v = vals('signed_diff')
        ax.plot(iters, v, 'o-', color=color, label=model, linewidth=2)

        # 3. New triples per iteration
        ax = axes[1, 0]
        ax.bar([it + idx * 0.2 for it in iters], vals('new_triples'),
               width=0.2, color=color, alpha=0.8, label=model)

        # 4. Mean absolute PR error
        ax = axes[1, 1]
        ax.plot(iters, vals('mean_err'), 'o-', color=color, label=model, linewidth=2)

    # Formatting — plot 1: centrality bias %
    ax = axes[0, 0]
    ax.axhline(50, color='grey', linestyle='--', alpha=0.5, linewidth=1.2, label='unbiased (50%)')
    ax.set_title('Centrality Bias — % of predictions with PR higher than real entity',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('% predictions where PR(pred) > PR(real)')
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.05,
            'Above 50% = model prefers more central (hub) entities\n'
            'Below 50% = model avoids hubs (low-centrality bias)',
            transform=ax.transAxes, fontsize=7.5, color='dimgray',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    # Formatting — plot 2: signed diff
    ax = axes[0, 1]
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5, linewidth=1.2, label='no bias (0)')
    ax.set_title('Signed PR Difference — mean(PR(pred) − PR(real))',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('mean PR(pred) − PR(real)')
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))
    ax.text(0.02, 0.05,
            'Positive = model picks entities with higher PageRank\n'
            'Negative = model picks entities with lower PageRank',
            transform=ax.transAxes, fontsize=7.5, color='dimgray',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    # Formatting — plot 3: new triples
    ax = axes[1, 0]
    ax.set_title('New Unique Triples Added per Iteration', fontsize=10, fontweight='bold')
    ax.set_ylabel('New triples')
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            'replace: same count each iter (2268 test facts × 2 sides)\n'
            'add: shrinks if graph absorbs predictions as known facts',
            transform=ax.transAxes, fontsize=7.5, color='dimgray',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    # Formatting — plot 4: mean abs PR error
    ax = axes[1, 1]
    ax.set_title('Mean Absolute PageRank Error  |PR(pred) − PR(real)|',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('mean |PR(pred) − PR(real)|')
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))
    ax.text(0.02, 0.95,
            'Magnitude of centrality error regardless of direction.\n'
            'Lower = predictions closer to real entity centrality.',
            transform=ax.transAxes, fontsize=7.5, color='dimgray',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    out = '/Users/dm6541/mirror test/infinity_mirror_plots.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {out}")


if __name__ == '__main__':
    main()
