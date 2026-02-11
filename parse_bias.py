#!/usr/bin/env python3
import sys, re
import numpy as np


def parse(path):
    model, entries = None, {'tail': [], 'head': []}
    with open(path) as f:
        for line in f:
            m = re.search(r'^\[CONFIG\] model=(\S+)', line)
            if m: model = m.group(1)
            for tag in ['TAIL', 'HEAD']:
                if line.startswith(f'[{tag}]'):
                    kv = dict(re.findall(r'(\w+)=([\S]+)', line))
                    entries[tag.lower()].append({
                        'pr_real': float(kv['pr_real']), 'pr_pred': float(kv['pr_pred']), 'rel': kv['rel']})
    return model or path, entries


def stats(entries):
    if not entries: return None
    diffs = np.array([e['pr_pred'] - e['pr_real'] for e in entries])
    pct = np.mean(diffs > 0) * 100
    return {'n': len(diffs), 'mean_pr_real': np.mean([e['pr_real'] for e in entries]),
            'mean_pr_pred': np.mean([e['pr_pred'] for e in entries]),
            'signed_diff': np.mean(diffs), 'abs_diff': np.mean(np.abs(diffs)), 'pct_higher': pct,
            'bias': 'HIGH_CENTRALITY' if pct > 55 else ('LOW_CENTRALITY' if pct < 45 else 'NEUTRAL')}


def print_stats(label, s):
    print(f"[{label}] n={s['n']} mean_pr_real={s['mean_pr_real']:.10f} mean_pr_pred={s['mean_pr_pred']:.10f} "
          f"signed_diff={s['signed_diff']:+.10f} abs_diff={s['abs_diff']:.10f} "
          f"pct_pred_higher={s['pct_higher']:.1f}% bias={s['bias']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_bias.py <log1> [log2] ..."); sys.exit(1)

    results = {}
    for path in sys.argv[1:]:
        model, entries = parse(path)
        results[model] = {}
        print(f"\n{'='*80}\n[MODEL] {model}\n{'='*80}")
        for typ in ['tail', 'head']:
            s = stats(entries[typ])
            if s: results[model][typ] = s; print_stats(typ.upper(), s)
        s = stats(entries['tail'] + entries['head'])
        if s: results[model]['overall'] = s; print_stats('OVERALL', s)

        # per-relation bias (tail only, most interesting)
        from collections import defaultdict
        by_rel = defaultdict(list)
        for e in entries['tail']: by_rel[e['rel']].append(e['pr_pred'] - e['pr_real'])
        print(f"\n[PER-RELATION TAIL BIAS] (sorted by % predicting higher centrality)")
        for rel in sorted(by_rel, key=lambda r: np.mean(np.array(by_rel[r]) > 0), reverse=True):
            d = np.array(by_rel[rel])
            pct = np.mean(d > 0) * 100
            print(f"  rel={rel} n={len(d)} pct_higher={pct:.1f}% signed_diff={np.mean(d):+.10f}")

    if len(results) > 1:
        print(f"\n{'='*80}\n[CROSS-MODEL COMPARISON]\n{'='*80}")
        ranked = sorted(results, key=lambda m: results[m].get('overall', {}).get('pct_higher', 0), reverse=True)
        for m in ranked:
            s = results[m].get('overall', {})
            print(f"[RANK] model={m} pct_higher={s.get('pct_higher', 0):.1f}% "
                  f"signed_diff={s.get('signed_diff', 0):+.10f} bias={s.get('bias', '?')}")
        print(f"\n[VERDICT] most_centrality_biased={ranked[0]} least_centrality_biased={ranked[-1]}")
        d = results[ranked[0]]['overall']['pct_higher'] - results[ranked[-1]]['overall']['pct_higher']
        print(f"[VERDICT] difference={d:.1f}pp {'(significant)' if abs(d) > 5 else '(marginal)'}")


if __name__ == '__main__':
    main()
