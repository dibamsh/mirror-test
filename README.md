# Infinity Mirror Test


## Setup

### 1. Download the models

Download from: https://drive.google.com/drive/u/0/folders/14jzhmwUrQmreZfmMH8eC8MYocdH_YS3k

Folders **2** (FB15K237) and **3** (NELL-995) are used in this project.

1. Create a folder called `Model` inside the project directory
2. Download folders **2** and **3** from the Drive and put them inside `Model/`

The structure should look like this:
```
mirror-test/
  Model/
    2/   ← FB15K237 models (e.g. transe__66_Expl.model)
    3/   ← NELL-995 models (e.g. transe_resplit__67_Expl.model)
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Set model paths

Open `main.py` and update these two lines to the full path of your Model folder:

```python
NELL_MODEL_DIR     = '/your/path/to/Model/3/'
FB15K237_MODEL_DIR = '/your/path/to/Model/2/'
```

If you placed `Model/` inside the project folder (as shown above), the defaults already work:

```python
NELL_MODEL_DIR     = 'Model/3/'
FB15K237_MODEL_DIR = 'Model/2/'
```

---

## Run experiments

Results are saved to `{dataset}_{model}_{mode}.log`.

**Run all models and modes for NELL-995:**
```bash
python main.py --dataset nell --n_iters 10
```

**Run all models and modes for FB15K237:**
```bash
python main.py --dataset fb15k237 --n_iters 10
```

**Run a specific model and mode:**
```bash
python main.py --dataset nell --model transe --mode replace --n_iters 10
```

---

## What the log contains

- `[CONFIG]` — what was run
- `[DATA]` — number of entities, relations, and triples per split
- `[ITER]` — per iteration: total triples in graph, how many were added
- `[BIAS]` — **the key result**: `pct_higher` = % of predictions where the predicted entity has higher PageRank than the real entity
- `[TIME]` — scoring time per iteration

---

### Two modes

**replace** — each iteration, predictions *replace* the test facts. 

**add** — test facts stay fixed, but each iteration's predictions are excluded from future candidates. 

### The bias metric

After each iteration, PageRank is computed on the updated graph. For every prediction, we check: does the predicted entity have higher PageRank than the real entity? `pct_higher` is the percentage where this is true. Above 50% consistently means the model is biased toward hub entities.
