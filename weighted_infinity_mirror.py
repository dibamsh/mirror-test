#!/usr/bin/env python3
import glob
import sys, os, time
import numpy as np, torch, networkx as nx

from DataLoader.TripleManager import TripleManager
from Utils import DatasetUtils
from pagerank import weighted_pagerank

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
    Returns (predicted entity ID, score) or (None, None).
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

    if not candidates: return None, None
    n = len(candidates)
    arr = np.array(candidates, np.int64)
    if side == 'tail':
        aH, aR, aT = np.full(n, h, np.int64), np.full(n, r, np.int64), arr
    else:
        aH, aR, aT = arr, np.full(n, r, np.int64), np.full(n, t, np.int64)
    scores = score_batch(aH, aR, aT, model)
    best = int(np.argmax(scores))
    return int(candidates[best]), float(scores[best])


def predict_all(facts_hrt, model, tm, extra_excl=None):
    """Get top-1 tail and head prediction with scores for each fact (h,r,t)."""
    results = []
    for i, (h, r, t) in enumerate(facts_hrt):
        if (i + 1) % 500 == 0:
            print(f"[SCORING] {i+1}/{len(facts_hrt)}", flush=True)
        tail_pred, tail_score = top1(h, r, t, 'tail', model, tm, extra_excl)
        head_pred, head_score = top1(h, r, t, 'head', model, tm, extra_excl)
        real_score = float(score_batch([h], [r], [t], model)[0])
        results.append({
            'fact':           (h, r, t),
            'real_score':     real_score,
            'tail_pred':      tail_pred,
            'tail_score':     tail_score,
            'head_pred':      head_pred,
            'head_score':     head_score,
        })
    return results


def save_scores(it, results, emap, rmap):
    for res in results:
        h, r, t = res['fact']

        print(
            {'it': it,
             'h': h, 'h_name': emap.get(h, ''),
             'r': r, 'r_name': rmap.get(r, ''),
             't': t, 't_name': emap.get(t, ''),
             'real_score': res['real_score'],
             'tail_pred': res['tail_pred'],
             'tail_pred_name': emap.get(res['tail_pred'], '') if res['tail_pred'] is not None else '',
             'tail_score': res['tail_score'],
             'head_pred': res['head_pred'],
             'head_pred_name': emap.get(res['head_pred'], '') if res['head_pred'] is not None else '',
             'head_score': res['head_score']})


def log_bias(it, results, pr, total_triples, new_triples, pr_iters):
    tail = [pr.get(r['tail_pred'], 0) > pr.get(r['fact'][2], 0)
            for r in results if r['tail_pred'] is not None]
    head = [pr.get(r['head_pred'], 0) > pr.get(r['fact'][0], 0)
            for r in results if r['head_pred'] is not None]
    all_ = tail + head
    print(f"[ITER] iteration={it} new_triples={new_triples} total_triples={total_triples} pagerank_iters={pr_iters}")
    if tail:  print(f"[BIAS] iteration={it} type=tail    pct_higher={np.mean(tail)*100:.1f}%")
    if head:  print(f"[BIAS] iteration={it} type=head    pct_higher={np.mean(head)*100:.1f}%")
    if all_:  print(f"[BIAS] iteration={it} type=overall pct_higher={np.mean(all_)*100:.1f}%")


# ─── Approach 1: Replace ──────────────────────────────────────────────────────

def run_replace(model, tm, emap, rmap, original_hrt, test_hrt, n_iters):
    """
    Each iteration: current test facts, top-1 predictions, become next iteration's test facts.
    Graph = train+valid + current_predictions (test facts replaced).
    Tracks centrality of predictions relative to the current fact entity.
    """
    print(f"[MODE] approach=replace iterations={n_iters}")
    test_set = set(tm.tripleList)
    base_graph_triples = [tri for tri in original_hrt if tri not in test_set]

    _, it0 = weighted_pagerank(build_graph(original_hrt, emap, rmap))
    print(f"[ITER] iteration=0 new_triples=0 total_triples={len(original_hrt)} pagerank_iters={it0}")

    current_facts = test_hrt

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
        pr, pr_iters = weighted_pagerank(build_graph(all_graph, emap, rmap))

        log_bias(it, results, pr, len(all_graph), len(tail_preds) + len(head_preds), pr_iters)
        save_scores(it, results, emap, rmap)

        print(f"[TIME] iteration={it} scoring={elapsed:.1f}s")

        # Next iteration: predict FROM the predictions (tail predictions as new facts)
        # Replace (s,p,o) with (s,p,o') so next iteration queries (s,p,o')
        next_facts = []
        for res in results:
            h, r, t = res['fact']
            if res['tail_pred'] is not None:
                next_facts.append((h, r, res['tail_pred']))   # replaced tail
        current_facts = next_facts

    print(f"[DONE] approach=replace")


# ─── Approach 2: Add ──────────────────────────────────────────────────────────

def run_add(model, tm, emap, rmap, original_hrt, test_hrt, n_iters):
    """
    Test facts are fixed (original test set).
    Each iteration: top-1 predictions become "known facts" (added to graph + excluded from future).
    Model must find the next-best prediction each iteration as known set grows.
    Tracks: how does centrality bias evolve as model is pushed to predict further?
    """
    print(f"[MODE] approach=add iterations={n_iters}")

    _, it0 = weighted_pagerank(build_graph(original_hrt, emap, rmap))
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
        pr, pr_iters = weighted_pagerank(build_graph(all_graph, emap, rmap))

        log_bias(it, results, pr, len(all_graph), len(new_this_iter), pr_iters)
        save_scores(it, results, emap, rmap)
        print(f"[TIME] iteration={it} scoring={elapsed:.1f}s")

    print(f"[DONE] approach=add")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # This is the main folder where AKGE is located.
    folder = sys.argv[1]
    # This is the number of the current experiment.
    index = int(sys.argv[2])

    def select_exp():
        i = 0

        for dataset in range(0, 10):
            for mode in ['replace', 'add']:
                for name in [
                    'boxe', 'complex', 'hake_both', 'hole', 'quate', 'rotate',
                    'rotpro', 'toruse_eL2', 'transe', 'tucker']:
                    if i == index:
                        return dataset, mode, name
                    else:
                        i += 1

        return dataset, mode, name

    n_iters = 10
    dataset, mode, name = select_exp()

    ds_name = DatasetUtils.get_dataset_name(dataset)

    print(f"[CONFIG] dataset={ds_name} model={name} mode={mode} iterations={n_iters}")

    matches = glob.glob(folder + '/Model/' + str(dataset) + '/' + name + '*.model')

    if len(matches) == 1:
        model_path = matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Expected 1 file, but found {len(matches)}: {matches}")
    else:
        raise FileNotFoundError("No matching file found.")

    dataset_path = folder + '/Datasets/' + ds_name + '/'

    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    emap, rmap = load_maps(dataset_path)

    prefix = 'resplit_' if dataset == 3 else ''

    splits = [f"{prefix}test", f"{prefix}valid", f"{prefix}train"]
    tm = TripleManager(dataset_path, splits=splits, corruption_mode="LCWA")

    # TODO This can be done as a DataLoader.
    train_hrt = load_hrt(dataset_path + f"{prefix}train2id.txt")
    test_hrt  = load_hrt(dataset_path + f"{prefix}test2id.txt")
    valid_hrt = load_hrt(dataset_path + f"{prefix}valid2id.txt")

    original_hrt = train_hrt + test_hrt + valid_hrt

    print(f"[DATA] entities={len(emap)} relations={len(rmap)} "
          f"train={len(train_hrt)} test={len(test_hrt)} valid={len(valid_hrt)} total={len(original_hrt)}")

    if mode == 'replace':
        run_replace(model, tm, emap, rmap, original_hrt, test_hrt, n_iters)
    else:
        run_add(model, tm, emap, rmap, original_hrt, test_hrt, n_iters)


if __name__ == '__main__':
    main()
