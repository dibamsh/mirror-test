# Infinity Mirror Test
---

## Setup

### 1. Download the models

Download from: https://drive.google.com/drive/u/0/folders/14jzhmwUrQmreZfmMH8eC8MYocdH_YS3k

Folders **2** (FB15K237) and **3** (NELL-995) are used in this project.

Extract them so the structure looks like:
```
Model/
  2/   ← FB15K237 models
  3/   ← NELL-995 models
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Set model paths

Open `main.py` and update these two lines to point to your model folders:

```python
NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'
```

---

## How to run

```bash
python main.py
```

It will ask four questions:

```
dataset (nell / fb15k237)?
model (transe / rotate / complex / boxe)?
mode (replace / add)?
iterations?
```

**Example — NELL-995:**
```
dataset (nell / fb15k237)? nell
model (transe / rotate / complex / boxe)? transe
mode (replace / add)? replace
iterations? 5
```

**Example — FB15K237:**
```
dataset (nell / fb15k237)? fb15k237
model (transe / rotate / complex / boxe)? transe
mode (replace / add)? replace
iterations? 5
```

Results are saved to `{dataset}_{model}_{mode}.log`.

---

## What the log contains

- `[CONFIG]` — what was run
- `[DATA]` — number of entities, relations, and triples per split
- `[ITER]` — per iteration: total triples in graph, how many were added
- `[BIAS]` — **the key result**: `pct_higher` = % of predictions where the predicted entity has higher PageRank than the real entity
- `[TIME]` — scoring time per iteration

---

## Code overview

| File | What it does |
|------|--------------|
| `main.py` | Entry point. Asks for inputs, runs the experiment, saves output to a log file. |
| `infinity_mirror.py` | Core experiment. Loads model and dataset, runs the mirror test, logs bias results. |
| `pagerank.py` | Standard PageRank on a directed multigraph. |
| `AugmentedKGE/` | Provides TripleManager for candidate filtering under LCWA. Copied from [nari97/AugmentedKGE](https://github.com/nari97/AugmentedKGE). |
| `datasets/` | NELL-995 and FB15K237 triple files (train/valid/test). |

### Two modes

**replace** — each iteration, predictions *replace* the test facts. The model then predicts on its own previous output, creating a feedback loop.

**add** — test facts stay fixed, but each iteration's predictions are excluded from future candidates. The model is forced to find its next-best answer each time.

### The bias metric

After each iteration, PageRank is computed on the updated graph. For every prediction, we check: does the predicted entity have higher PageRank than the real entity? `pct_higher` is the percentage where this is true. Above 50% consistently means the model is biased toward hub entities.
