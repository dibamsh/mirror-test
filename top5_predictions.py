#!/usr/bin/env python3
import sys, os, time, pickle
import numpy as np, torch

sys.path.insert(0, '/Users/dm6541/AugmentedKGE')
from DataLoader.TripleManager import TripleManager

MODELS = {
    'transe': 'transe_resplit__67_Expl.model', 'complex': 'complex_resplit__11_Expl.model',
    'rotate': 'rotate_resplit__43_Expl.model', 'boxe': 'boxe_resplit__3_Expl.model',
    'hake': 'hake_both_resplit__19_Expl.model', 'hole': 'hole_resplit__27_Expl.model',
    'quate': 'quate_resplit__35_Expl.model', 'rotpro': 'rotpro_resplit__51_Expl.model',
    'toruse': 'toruse_eL2_resplit__59_Expl.model', 'tucker': 'tucker_resplit__75_Expl.model',
}
MODEL_DIR = '/Users/dm6541/mirror test/Model/3/'
DATASET = '/Users/dm6541/AugmentedKGE/Datasets/NELL-995/'
OUT_DIR = '/Users/dm6541/mirror test/'
K, PREFIX = 5, "resplit_"


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


def score(h, r, t, model):
    with torch.no_grad():
        return model.predict({'batch_h': torch.LongTensor(h), 'batch_r': torch.LongTensor(r),
                              'batch_t': torch.LongTensor(t), 'mode': 'normal'}).detach().numpy()


def topk(tm, tri, model):
    h, r, t = tri.h, tri.r, tri.t
    preds = {}
    for side in ['head', 'tail']:
        corrupted = list(tm.get_corrupted(h, r, t, side))
        preds[side] = []
        if not corrupted: continue
        n = 1 + len(corrupted)
        aH, aR, aT = np.full(n, h, np.int64), np.full(n, r, np.int64), np.full(n, t, np.int64)
        arr = np.array([(h if side == 'head' else t)] + corrupted, np.int64)
        if side == 'head': aH = arr
        else: aT = arr
        sc = score(aH, aR, aT, model)
        for i in np.argsort(-sc)[:K]:
            preds[side].append({'head': int(aH[i]), 'relation': r, 'tail': int(aT[i]),
                                'score': float(sc[i]), 'is_original': bool(i == 0)})
    return {'original': {'head': h, 'relation': r, 'tail': t},
            'head_predictions': preds['head'], 'tail_predictions': preds['tail']}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'transe'
    assert name in MODELS, f"Unknown: {name}. Options: {list(MODELS.keys())}"
    print(f"[CONFIG] model={name} file={MODELS[name]} k={K}")

    model = torch.load(MODEL_DIR + MODELS[name], map_location='cpu', weights_only=False)
    model.eval()
    emap, rmap = load_maps(DATASET)
    tm = TripleManager(DATASET, splits=[f"{PREFIX}test", f"{PREFIX}valid", f"{PREFIX}train"], corruption_mode="LCWA")
    triples = tm.get_triples()
    print(f"[DATA] entities={len(emap)} relations={len(rmap)} test_triples={len(triples)}")

    results, start = [], time.perf_counter()
    for i, tri in enumerate(triples):
        if (i + 1) % 500 == 0:
            e = time.perf_counter() - start
            print(f"[PROGRESS] {i+1}/{len(triples)} elapsed={e:.1f}s eta={e/(i+1)*(len(triples)-i-1):.1f}s")
        results.append(topk(tm, tri, model))

    new = {(p['head'], p['tail'], p['relation'])
           for r in results for p in r['head_predictions'] + r['tail_predictions'] if not p['is_original']}
    print(f"[DONE] triples={len(triples)} new_predictions={len(new)} time={time.perf_counter()-start:.1f}s")
    out = os.path.join(OUT_DIR, f'{name}_predictions.pickle')
    with open(out, 'wb') as f: pickle.dump(results, f)
    print(f"[SAVED] {out}")


if __name__ == '__main__':
    main()
