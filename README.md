# Mirror Test

Evaluates centrality bias in Knowledge Graph Embedding (KGE) models using an Infinity Mirror Test.

## Run

**Interactive — pick dataset, model, mode, iterations:**
```bash
python3 main.py
```
Prompts you to choose each option, with `all` available to run everything. Output saved to `{dataset}_{model}_{mode}.log`.

**Direct single run:**
```bash
python3 infinity_mirror.py <model> <mode> <iters> <dataset> > output.log
```
```bash
python3 infinity_mirror.py transe replace 5 nell > nell_transe_replace.log
python3 infinity_mirror.py rotate add 5 fb15k237 > fb_rotate_add.log
```

---

## What it does

For each test fact `(s, p, o)`, the model makes two predictions:
- **tail**: `(s, p, ?)` — compare PageRank of `?` vs `o`
- **head**: `(?, p, o)` — compare PageRank of `?` vs `s`

If predictions consistently have **lower PageRank than real entities** → low-centrality bias (model avoids hubs).

### Two approaches

**replace** — each iteration's test facts are the predictions from the previous iteration. The model scores its own outputs. Reveals whether bias compounds or flips when the model feeds on itself.

**add** — test facts stay fixed. Each iteration's predictions are added to the known graph, forcing the model to find the next-best answer. Reveals how bias evolves deeper in the model's score distribution.

---

## Datasets & Models

| Dataset   | Entities | Relations | Triples |
|-----------|----------|-----------|---------|
| NELL-995  | 75,492   | 200       | 154,213 |
| FB15K237  | 14,541   | 237       | 310,116 |

Models used: TransE, RotatE, ComplEx, BoxE (pre-trained, no training happens).

---

## Key metric

`pct_higher` — % of predictions where `PR(predicted) > PR(real entity)`.
- ~50% = unbiased
- Well below 50% = low-centrality bias (model prefers peripheral entities)
- Well above 50% = high-centrality bias (model prefers hubs)

---

## Dependencies

```bash
pip install -r requirements.txt
```

Also requires (local, not pip):
- `AugmentedKGE` at `/Users/dm6541/AugmentedKGE`
- `Research_Experiments` at `/Users/dm6541/Research_Experiments`
