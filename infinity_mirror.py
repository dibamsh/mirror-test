#!/usr/bin/env python3
import sys, os, time
import numpy as np, torch, networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'AugmentedKGE'))
from DataLoader.TripleManager import TripleManager
sys.path.insert(0, '/Users/dm6541/Research_Experiments')
from algorithms.algorithms import PageRankAlgorithms

DATASETS = {
    'nell': {
        'model_dir':  os.environ.get('NELL_MODEL_DIR',     os.path.join(BASE, 'Model/3/')),
        'dataset_dir': os.path.join(BASE, 'datasets/NELL-995/'),
        'prefix':     'resplit_',
        'models': {
            'transe':  'transe_resplit__67_Expl.model',
            'rotate':  'rotate_resplit__43_Expl.model',
            'complex': 'complex_resplit__11_Expl.model',
            'boxe':    'boxe_resplit__3_Expl.model',
        },
    },
    'fb15k237': {
        'model_dir':  os.environ.get('FB15K237_MODEL_DIR', os.path.join(BASE, 'Model/2/')),
        'dataset_dir': os.path.join(BASE, 'datasets/FB15K237/'),
        'prefix':     '',
        'models': {
            'transe':  'transe__66_Expl.model',
            'rotate':  'rotate__42_Expl.model',
            'complex': 'complex__10_Expl.model',
            'boxe':    'boxe__2_Expl.model',
        },
    },
}


def load_maps(dataset_dir):
    def read(p):
        m = {}
        with open(p) as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2: m[int(parts[1])] = parts[0]
        return m
    return read(dataset_dir + 'entity2id.txt'), read(dataset_dir + 'relation2id.txt')


def score_batch(h, r, t, model):
    with torch.no_grad():
        return model.predict({'batch_h': torch.LongTensor(h), 'batch_r': torch.LongTensor(r),
                              'batch_t': torch.LongTensor(t), 'mode': 'normal'}).detach().numpy()


def load_hrt(path):
    """Load 2id file as (head, relation, tail). File format is: head tail relation per line."""
    with open(path) as f:
        f.readline()
        triples = []
        for line in f:
            p = line.split()
            if len(p) == 3:
                h, t_node, r = int(p[0]), int(p[1]), int(p[2])
                triples.append((h, r, t_node))   # convert to (head, relation, tail)
        return triples


def build_graph(triples_hrt, emap, rmap):
    G = nx.MultiDiGraph()
    for eid, name in emap.items(): G.add_node(eid, name=name)
    for h, r, t in triples_hrt:
        G.add_edge(h, t, relation_id=r, relation_name=rmap.get(r, str(r)))
    return G


def top1(h, r, t, side, model, tm, extra_excl=None):
    """
    Top-1 prediction for fact (h,r,t) on given side.
    extra_excl: set of (h,r,t) triples to additionally exclude on top of TM's LCWA.
    Falls back to all entities seen in that role for the relation if TM doesn't know the entity.
    Returns predicted entity ID or None.
    """
    try:
        candidates = list(tm.get_corrupted(h, r, t, side))
    except KeyError:
        # Predicted entity is novel for this relation/side — use all known entities for that side
        if side == 'head':
            candidates = list(getattr(tm, 'headEntities', {}).get(r, []))
        else:
            candidates = list(getattr(tm, 'tailEntities', {}).get(r, []))
    # Always exclude the fact entity itself
    fact_ent = h if side == 'head' else t
    candidates = [e for e in candidates if e != fact_ent]

    if extra_excl:
        if side == 'tail':
            candidates = [e for e in candidates if (h, r, e) not in extra_excl]
        else:
            candidates = [e for e in candidates if (e, r, t) not in extra_excl]

    if not candidates: return None
    n = len(candidates)
    arr = np.array(candidates, np.int64)
    if side == 'tail':
        aH, aR, aT = np.full(n, h, np.int64), np.full(n, r, np.int64), arr
    else:
        aH, aR, aT = arr, np.full(n, r, np.int64), np.full(n, t, np.int64)
    return int(candidates[int(np.argmax(score_batch(aH, aR, aT, model)))])


def predict_all(facts_hrt, model, tm, extra_excl=None):
    """Get top-1 tail and head prediction for each fact (h,r,t)."""
    results = []
    for i, (h, r, t) in enumerate(facts_hrt):
        if (i + 1) % 500 == 0:
            print(f"[SCORING] {i+1}/{len(facts_hrt)}", flush=True)
        results.append({
            'fact': (h, r, t),
            'tail_pred': top1(h, r, t, 'tail', model, tm, extra_excl),
            'head_pred': top1(h, r, t, 'head', model, tm, extra_excl),
        })
    return results


def bias(results, pr):
    """
    For each prediction, compute signed PR diff: PR(predicted) - PR(real entity).
    Positive = model picked a more central entity than the real one.
    Negative = model picked a less central entity (low-centrality bias).
    """
    td, hd = [], []
    for res in results:
        h, r, t = res['fact']
        if res['tail_pred'] is not None:
            td.append(pr.get(res['tail_pred'], 0) - pr.get(t, 0))
        if res['head_pred'] is not None:
            hd.append(pr.get(res['head_pred'], 0) - pr.get(h, 0))
    return np.array(td), np.array(hd)


def log_iter(it, td, hd, total_triples, new_triples, pr_iters):
    all_d = np.concatenate([td, hd])
    print(f"[ITER] iteration={it} new_triples={new_triples} total_triples={total_triples} pagerank_iters={pr_iters}")
    for label, d in [('tail', td), ('head', hd), ('overall', all_d)]:
        if len(d) == 0: continue
        print(f"[BIAS] iteration={it} type={label:<7} n={len(d)} "
              f"pct_higher={np.mean(d > 0)*100:.1f}% "
              f"signed_diff={np.mean(d):+.10f} "
              f"mean_abs={np.mean(np.abs(d)):.10f} "
              f"mean_pr_real={np.mean([pr_real for pr_real in [0]]):.10f}")


def log_iter_full(it, td, hd, pr_reals_t, pr_reals_h, total_triples, new_triples, pr_iters):
    """Extended logging with mean PR of real vs predicted entities."""
    all_d = np.concatenate([td, hd])
    print(f"[ITER] iteration={it} new_triples={new_triples} total_triples={total_triples} pagerank_iters={pr_iters}")
    for label, d, pr_r in [('tail', td, pr_reals_t), ('head', hd, pr_reals_h)]:
        if len(d) == 0: continue
        pr_pred = np.array(pr_r) + d   # pr_pred = pr_real + diff
        print(f"[BIAS] iteration={it} type={label:<7} n={len(d)} "
              f"pct_higher={np.mean(d > 0)*100:.1f}% "
              f"signed_diff={np.mean(d):+.10f} "
              f"mean_abs={np.mean(np.abs(d)):.10f} "
              f"mean_pr_real={np.mean(pr_r):.10f} "
              f"mean_pr_pred={np.mean(pr_pred):.10f}")
    if len(all_d) > 0:
        print(f"[BIAS] iteration={it} type=overall n={len(all_d)} "
              f"pct_higher={np.mean(all_d > 0)*100:.1f}% "
              f"signed_diff={np.mean(all_d):+.10f} "
              f"mean_abs={np.mean(np.abs(all_d)):.10f}")


def collect_bias(results, pr):
    td, hd, pr_rt, pr_rh = [], [], [], []
    for res in results:
        h, r, t = res['fact']
        if res['tail_pred'] is not None:
            pr_t = pr.get(t, 0)
            pr_p = pr.get(res['tail_pred'], 0)
            td.append(pr_p - pr_t)
            pr_rt.append(pr_t)
        if res['head_pred'] is not None:
            pr_h = pr.get(h, 0)
            pr_p = pr.get(res['head_pred'], 0)
            hd.append(pr_p - pr_h)
            pr_rh.append(pr_h)
    return np.array(td), np.array(hd), pr_rt, pr_rh


# ─── Approach 1: Replace ──────────────────────────────────────────────────────

def run_replace(model, tm, emap, rmap, algo, original_hrt, test_hrt, n_iters):
    """
    Each iteration: current test facts → top-1 predictions → become next iteration's test facts.
    Graph = train+valid + current_predictions (test facts replaced).
    Tracks centrality of predictions relative to the current fact entity.
    """
    print(f"[MODE] approach=replace iterations={n_iters}")
    test_set = set((h, r, t) for h, r, t in test_hrt)
    base_graph_triples = [tri for tri in original_hrt if tri not in test_set]

    pr0, it0 = algo.standard_pagerank(build_graph(original_hrt, emap, rmap))
    print(f"[ITER] iteration=0 new_triples=0 total_triples={len(original_hrt)} pagerank_iters={it0}")

    current_facts = test_hrt                  # (h, r, t) triples we predict FOR
    current_graph_extra = list(test_hrt)      # what gets added to base graph

    for it in range(1, n_iters + 1):
        start = time.perf_counter()
        results = predict_all(current_facts, model, tm)
        elapsed = time.perf_counter() - start

        # Collect new predictions
        tail_preds = [(res['fact'][0], res['fact'][1], res['tail_pred'])
                      for res in results if res['tail_pred'] is not None]
        head_preds = [(res['head_pred'], res['fact'][1], res['fact'][2])
                      for res in results if res['head_pred'] is not None]

        # Graph = base (train+valid) + new predictions (replacing old test facts)
        all_graph = base_graph_triples + tail_preds + head_preds
        pr, pr_iters = algo.standard_pagerank(build_graph(all_graph, emap, rmap))

        td, hd, pr_rt, pr_rh = collect_bias(results, pr)
        log_iter_full(it, td, hd, pr_rt, pr_rh,
                      len(all_graph), len(tail_preds) + len(head_preds), pr_iters)
        print(f"[TIME] iteration={it} scoring={elapsed:.1f}s")

        # Next iteration: predict FROM the predictions (tail predictions as new facts)
        # Replace (s,p,o) → (s,p,o') so next iteration queries (s,p,o')
        next_facts = []
        for res in results:
            h, r, t = res['fact']
            if res['tail_pred'] is not None:
                next_facts.append((h, r, res['tail_pred']))   # replaced tail
        current_facts = next_facts
        current_graph_extra = tail_preds

    print(f"[DONE] approach=replace")


# ─── Approach 2: Add ──────────────────────────────────────────────────────────

def run_add(model, tm, emap, rmap, algo, original_hrt, test_hrt, n_iters):
    """
    Test facts are fixed (original test set).
    Each iteration: top-1 predictions become "known facts" (added to graph + excluded from future).
    Model must find the next-best prediction each iteration as known set grows.
    Tracks: how does centrality bias evolve as model is pushed to predict further?
    """
    print(f"[MODE] approach=add iterations={n_iters}")

    pr0, it0 = algo.standard_pagerank(build_graph(original_hrt, emap, rmap))
    print(f"[ITER] iteration=0 new_triples=0 total_triples={len(original_hrt)} pagerank_iters={it0}")

    extra_excl   = set()       # grows each iter: (h,r,t) triples already predicted
    accumulated  = set()       # all predictions so far (added to graph)

    for it in range(1, n_iters + 1):
        start = time.perf_counter()
        results = predict_all(test_hrt, model, tm, extra_excl=extra_excl)
        elapsed = time.perf_counter() - start

        new_this_iter = set()
        for res in results:
            h, r, t = res['fact']
            if res['tail_pred'] is not None:
                new_this_iter.add((h, r, res['tail_pred']))
            if res['head_pred'] is not None:
                new_this_iter.add((res['head_pred'], r, t))

        extra_excl.update(new_this_iter)
        accumulated.update(new_this_iter)

        all_graph = original_hrt + list(accumulated)
        pr, pr_iters = algo.standard_pagerank(build_graph(all_graph, emap, rmap))

        td, hd, pr_rt, pr_rh = collect_bias(results, pr)
        log_iter_full(it, td, hd, pr_rt, pr_rh,
                      len(all_graph), len(new_this_iter), pr_iters)
        print(f"[TIME] iteration={it} scoring={elapsed:.1f}s")

    print(f"[DONE] approach=add")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    name    = sys.argv[1] if len(sys.argv) > 1 else 'transe'
    mode    = sys.argv[2] if len(sys.argv) > 2 else 'replace'
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    ds_name = sys.argv[4] if len(sys.argv) > 4 else 'nell'

    assert ds_name in DATASETS, f"Unknown dataset: {ds_name}"
    cfg = DATASETS[ds_name]
    assert name in cfg['models'], f"Unknown model: {name}"
    assert mode in ('replace', 'add'), "mode must be 'replace' or 'add'"

    print(f"[CONFIG] dataset={ds_name} model={name} mode={mode} iterations={n_iters}")

    model = torch.load(cfg['model_dir'] + cfg['models'][name], map_location='cpu', weights_only=False)
    model.eval()
    emap, rmap = load_maps(cfg['dataset_dir'])

    prefix = cfg['prefix']
    splits = [f"{prefix}test", f"{prefix}valid", f"{prefix}train"]
    tm = TripleManager(cfg['dataset_dir'], splits=splits, corruption_mode="LCWA")

    train_hrt = load_hrt(cfg['dataset_dir'] + f"{prefix}train2id.txt")
    test_hrt  = load_hrt(cfg['dataset_dir'] + f"{prefix}test2id.txt")
    valid_hrt = load_hrt(cfg['dataset_dir'] + f"{prefix}valid2id.txt")
    original_hrt = train_hrt + test_hrt + valid_hrt
    print(f"[DATA] entities={len(emap)} relations={len(rmap)} "
          f"train={len(train_hrt)} test={len(test_hrt)} valid={len(valid_hrt)} total={len(original_hrt)}")

    algo = PageRankAlgorithms()

    if mode == 'replace':
        run_replace(model, tm, emap, rmap, algo, original_hrt, test_hrt, n_iters)
    else:
        run_add(model, tm, emap, rmap, algo, original_hrt, test_hrt, n_iters)


if __name__ == '__main__':
    main()
