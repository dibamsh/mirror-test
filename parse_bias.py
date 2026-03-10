#!/usr/bin/env python3
"""
Parse InfinityMirrorTest log files.

Usage:
  python parse_bias.py logs/LogsIMT/                      # parse whole folder
  python parse_bias.py logs/LogsIMT/file.out              # single file
  python parse_bias.py logs/LogsIMT/ --csv results.csv    # also write CSV
  python parse_bias.py logs/LogsIMT/ --scores             # also parse score dicts (slow)
"""
import sys, re, ast, csv, os
import numpy as np
from collections import defaultdict

VARIANTS = ['standard', 'weighted', 'domainrange']


def parse_file(path, parse_scores=False):
    cfg = {}
    # bias[variant][type][iteration] = pct_higher
    bias = {v: defaultdict(dict) for v in VARIANTS}
    scores = defaultdict(list)   # scores[iteration] = list of (real, tail, head)

    with open(path) as f:
        for line in f:
            line = line.strip()

            # [CONFIG] dataset=FB13 model=boxe mode=replace iterations=5
            m = re.match(r'\[CONFIG\] dataset=(\S+) model=(\S+) mode=(\S+) iterations=(\d+)', line)
            if m:
                cfg = {'dataset': m.group(1), 'model': m.group(2),
                       'mode': m.group(3), 'n_iters': int(m.group(4))}

            # [BIAS] iteration=1 type=tail pct_higher=22.9% [pagerank_variant=...]
            m = re.match(r'\[BIAS\] iteration=(\d+) type=(\S+)\s+pct_higher=([\d.]+)%', line)
            if m:
                it  = int(m.group(1))
                typ = m.group(2).strip()
                val = float(m.group(3))
                pv  = re.search(r'pagerank_variant=(\S+)', line)
                variant = pv.group(1) if pv else 'standard'
                if variant in bias:
                    bias[variant][typ][it] = val

            # score dict lines — only parsed when --scores flag is set (slow on large logs)
            if parse_scores and line.startswith("{'it':"):
                try:
                    d = ast.literal_eval(line)
                    it = d['it']
                    real  = d.get('real_score')
                    tail  = d.get('tail_score')
                    head  = d.get('head_score')
                    if real is not None and tail is not None and head is not None:
                        scores[it].append((real, tail, head))
                except Exception:
                    pass

    return cfg, bias, scores


def score_gap_summary(scores):
    """For each iteration: mean(tail_score - real_score) and mean(head_score - real_score)."""
    result = {}
    for it in sorted(scores):
        entries = scores[it]
        tail_gaps = [t - r for r, t, h in entries]
        head_gaps = [h - r for r, t, h in entries]
        result[it] = {
            'n': len(entries),
            'tail_gap': np.mean(tail_gaps),
            'head_gap': np.mean(head_gaps),
        }
    return result


def parse_all(paths, parse_scores=False):
    experiments = []
    for path in paths:
        cfg, bias, scores = parse_file(path, parse_scores=parse_scores)
        if not cfg:
            continue
        gaps = score_gap_summary(scores)
        experiments.append({'path': path, 'cfg': cfg, 'bias': bias, 'gaps': gaps})
    return experiments


def _has_variant(bias, variant):
    return any(bias[variant][t] for t in ['tail', 'head', 'overall'])


def print_experiment(exp):
    cfg  = exp['cfg']
    bias = exp['bias']
    gaps = exp['gaps']
    n_iters = cfg.get('n_iters', 5)

    print(f"\n{'='*80}")
    print(f"  {cfg['dataset']}  |  {cfg['model']}  |  {cfg['mode']}")
    print(f"{'='*80}")

    active_variants = [v for v in VARIANTS if _has_variant(bias, v)]

    for variant in active_variants:
        b = bias[variant]
        label = f'[{variant}]'
        print(f"\n  {label}")
        print(f"  {'iter':<6}", end='')
        for t in ['tail', 'head', 'overall']:
            print(f"  {t:>10}", end='')
        if variant == 'standard':
            print(f"  {'tail_gap':>12}  {'head_gap':>12}", end='')
        print()
        print(f"  {'-'*6}", end='')
        for _ in range(3): print(f"  {'-'*10}", end='')
        if variant == 'standard':
            print(f"  {'-'*12}  {'-'*12}", end='')
        print()

        for it in range(1, n_iters + 1):
            row = f"  {it:<6}"
            for t in ['tail', 'head', 'overall']:
                val = b[t].get(it)
                row += f"  {val:>9.1f}%" if val is not None else f"  {'—':>10}"
            if variant == 'standard':
                g = gaps.get(it)
                if g:
                    row += f"  {g['tail_gap']:>+12.4f}  {g['head_gap']:>+12.4f}"
            print(row)

        overall_vals = [b['overall'].get(it) for it in range(1, n_iters + 1) if b['overall'].get(it) is not None]
        if len(overall_vals) >= 2:
            trend = overall_vals[-1] - overall_vals[0]
            direction = f"↑ +{trend:.1f}pp" if trend > 1 else (f"↓ {trend:.1f}pp" if trend < -1 else f"→ {trend:+.1f}pp")
            print(f"\n  overall trend: {direction}  (iter1={overall_vals[0]:.1f}% → iter{n_iters}={overall_vals[-1]:.1f}%)")


def print_summary(experiments):
    for variant in VARIANTS:
        active = [e for e in experiments if _has_variant(e['bias'], variant)]
        if not active:
            continue

        print(f"\n\n{'='*90}")
        print(f"  SUMMARY [{variant}] — final iteration overall pct_higher")
        print(f"{'='*90}")
        print(f"  {'dataset':<12}  {'model':<14}  {'mode':<10}  {'it1':>6}  {'it3':>6}  {'it5':>6}  {'trend':>10}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")

        sorted_exp = sorted(active,
                            key=lambda e: (e['cfg']['dataset'], e['cfg']['model'], e['cfg']['mode']))

        for exp in sorted_exp:
            cfg = exp['cfg']
            b   = exp['bias'][variant]
            it1 = b['overall'].get(1)
            it3 = b['overall'].get(3)
            it5 = b['overall'].get(5)
            if it1 is not None and it5 is not None:
                trend = it5 - it1
                t_str = f"{trend:+.1f}pp"
            else:
                t_str = "—"
            print(f"  {cfg['dataset']:<12}  {cfg['model']:<14}  {cfg['mode']:<10}  "
                  f"  {it1 if it1 else 0:>5.1f}  {it3 if it3 else 0:>5.1f}  {it5 if it5 else 0:>5.1f}  {t_str:>10}")

        # Cross-model ranking per dataset
        print(f"\n\n{'='*90}")
        print(f"  MODEL RANKING [{variant}] (final iteration, overall, averaged across modes)")
        print(f"{'='*90}")

        by_dataset = defaultdict(lambda: defaultdict(list))
        for exp in active:
            cfg = exp['cfg']
            it5 = exp['bias'][variant]['overall'].get(5)
            if it5 is not None:
                by_dataset[cfg['dataset']][cfg['model']].append(it5)

        for ds in sorted(by_dataset):
            print(f"\n  {ds}:")
            ranking = [(m, np.mean(vals)) for m, vals in by_dataset[ds].items()]
            ranking.sort(key=lambda x: -x[1])
            for rank, (model, pct) in enumerate(ranking, 1):
                bar = '█' * int(pct / 5)
                print(f"    {rank:2d}. {model:<14}  {pct:5.1f}%  {bar}")


def write_csv(experiments, csv_path):
    n_iters = 5
    types   = ['tail', 'head', 'overall']

    fieldnames = ['dataset', 'model', 'mode']
    for variant in VARIANTS:
        for t in types:
            for it in range(1, n_iters + 1):
                fieldnames.append(f'{variant}_{t}_it{it}_pct')
        fieldnames += [f'{variant}_overall_trend_pp', f'{variant}_final_overall_pct']
    for it in range(1, n_iters + 1):
        fieldnames += [f'it{it}_tail_gap', f'it{it}_head_gap']

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for exp in experiments:
            cfg = exp['cfg']
            row = {'dataset': cfg['dataset'], 'model': cfg['model'], 'mode': cfg['mode']}
            for variant in VARIANTS:
                b = exp['bias'][variant]
                for t in types:
                    for it in range(1, n_iters + 1):
                        val = b[t].get(it, '')
                        row[f'{variant}_{t}_it{it}_pct'] = f'{val:.2f}' if val != '' else ''
                it1 = b['overall'].get(1)
                it5 = b['overall'].get(5)
                row[f'{variant}_overall_trend_pp']  = f'{it5 - it1:.2f}' if (it1 is not None and it5 is not None) else ''
                row[f'{variant}_final_overall_pct'] = f'{it5:.2f}' if it5 is not None else ''
            for it in range(1, n_iters + 1):
                g = exp['gaps'].get(it)
                row[f'it{it}_tail_gap'] = f'{g["tail_gap"]:.6f}' if g else ''
                row[f'it{it}_head_gap'] = f'{g["head_gap"]:.6f}' if g else ''
            w.writerow(row)

    print(f"\nCSV saved to: {csv_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    csv_path = None
    parse_scores = False
    paths_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--csv' and i + 1 < len(sys.argv):
            csv_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == '--scores':
            parse_scores = True; i += 1
        else:
            paths_args.append(sys.argv[i]); i += 1

    # Collect files
    files = []
    for p in paths_args:
        if os.path.isdir(p):
            files += sorted(f.path for f in os.scandir(p) if f.name.endswith('.out'))
        else:
            files.append(p)

    print(f"Parsing {len(files)} files...")
    experiments = parse_all(files, parse_scores=parse_scores)
    print(f"Parsed {len(experiments)} valid experiments.")

    for exp in experiments:
        print_experiment(exp)

    print_summary(experiments)

    if csv_path:
        write_csv(experiments, csv_path)
    else:
        write_csv(experiments, 'results.csv')


if __name__ == '__main__':
    main()
