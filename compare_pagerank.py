#!/usr/bin/env python3
import sys, os, pickle
from collections import defaultdict
import numpy as np, networkx as nx

sys.path.insert(0, '/Users/dm6541/Research_Experiments')
from algorithms.algorithms import PageRankAlgorithms

DATASET = '/Users/dm6541/mirror test/NELL-995/'
PRED_DIR = '/Users/dm6541/mirror test/'
PREFIX = "resplit_"


def load_triples(path):
    with open(path) as f:
        f.readline()
        return [(int(p[0]), int(p[1]), int(p[2])) for line in f for p in [line.split()] if len(p) == 3]


def load_maps(path):
    def read(f):
        m = {}
        with open(os.path.join(path, f)) as fh:
            fh.readline()
            for line in fh:
                p = line.strip().split('\t')
                if len(p) == 2: m[int(p[1])] = p[0]
        return m
    return read('entity2id.txt'), read('relation2id.txt')


def build_graph(triples, emap, rmap):
    G = nx.MultiDiGraph()
    for eid, name in emap.items(): G.add_node(eid, name=name)
    for h, t, r in triples: G.add_edge(h, t, relation_id=r, relation_name=rmap.get(r, str(r)))
    return G


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'transe'
    print(f"[CONFIG] model={name}")

    emap, rmap = load_maps(DATASET)
    train = load_triples(DATASET + PREFIX + "train2id.txt")
    test = load_triples(DATASET + PREFIX + "test2id.txt")
    valid = load_triples(DATASET + PREFIX + "valid2id.txt")
    original = train + test + valid
    print(f"[DATA] entities={len(emap)} relations={len(rmap)} "
          f"train={len(train)} test={len(test)} valid={len(valid)} total={len(original)}")

    with open(os.path.join(PRED_DIR, f'{name}_predictions.pickle'), 'rb') as f:
        preds = pickle.load(f)
    print(f"[DATA] prediction_sets={len(preds)}")

    orig_set = set(original)
    new = {(p['head'], p['tail'], p['relation'])
           for r in preds for p in r['head_predictions'] + r['tail_predictions']
           if not p['is_original'] and (p['head'], p['tail'], p['relation']) not in orig_set}

    G_o = build_graph(original, emap, rmap)
    G_a = build_graph(original + list(new), emap, rmap)
    print(f"[GRAPH] type=original nodes={G_o.number_of_nodes()} edges={G_o.number_of_edges()}")
    print(f"[GRAPH] type=augmented nodes={G_a.number_of_nodes()} edges={G_a.number_of_edges()} new={len(new)}")

    algo = PageRankAlgorithms()
    pr_o, it_o = algo.standard_pagerank(G_o)
    pr_a, it_a = algo.standard_pagerank(G_a)
    print(f"[PAGERANK] type=original iterations={it_o}")
    print(f"[PAGERANK] type=augmented iterations={it_a}")

    errs = {'tail_orig': [], 'tail_aug': [], 'head_orig': [], 'head_aug': []}
    rel_errs = defaultdict(list)

    for idx, res in enumerate(preds):
        h, r, o = res['original']['head'], res['original']['relation'], res['original']['tail']
        rn = rmap.get(r, str(r))
        for p in res['tail_predictions']:
            if p['is_original']: continue
            oi = p['tail']
            eo, ea = abs(pr_o.get(o, 0) - pr_o.get(oi, 0)), abs(pr_a.get(o, 0) - pr_a.get(oi, 0))
            errs['tail_orig'].append(eo); errs['tail_aug'].append(ea); rel_errs[r].append(eo)
            print(f"[TAIL] test={idx+1} rel={rn} real={emap.get(o, o)} pred={emap.get(oi, oi)} "
                  f"pr_real={pr_o.get(o, 0):.10f} pr_pred={pr_o.get(oi, 0):.10f} "
                  f"err_orig={eo:.10f} err_aug={ea:.10f}")
        for p in res['head_predictions']:
            if p['is_original']: continue
            si = p['head']
            eo, ea = abs(pr_o.get(h, 0) - pr_o.get(si, 0)), abs(pr_a.get(h, 0) - pr_a.get(si, 0))
            errs['head_orig'].append(eo); errs['head_aug'].append(ea)
            print(f"[HEAD] test={idx+1} rel={rn} real={emap.get(h, h)} pred={emap.get(si, si)} "
                  f"pr_real={pr_o.get(h, 0):.10f} pr_pred={pr_o.get(si, 0):.10f} "
                  f"err_orig={eo:.10f} err_aug={ea:.10f}")

    for t in ['tail', 'head']:
        for g in ['orig', 'aug']:
            a = errs[f'{t}_{g}']
            print(f"[SUMMARY] type={t} graph={'original' if g == 'orig' else 'augmented'} "
                  f"mean={np.mean(a):.10f} median={np.median(a):.10f} max={np.max(a):.10f} std={np.std(a):.10f}")
    all_o, all_a = errs['tail_orig'] + errs['head_orig'], errs['tail_aug'] + errs['head_aug']
    print(f"[SUMMARY] type=overall graph=original mean={np.mean(all_o):.10f} median={np.median(all_o):.10f}")
    print(f"[SUMMARY] type=overall graph=augmented mean={np.mean(all_a):.10f} median={np.median(all_a):.10f}")

    for r in sorted(rel_errs, key=lambda x: np.mean(rel_errs[x]), reverse=True):
        print(f"[RELATION] rel={rmap.get(r, str(r))} mean_err={np.mean(rel_errs[r]):.10f} count={len(rel_errs[r])}")


if __name__ == '__main__':
    main()
