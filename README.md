# Minimal FastForest-Style RF

Minimal implementation of a FastForest/MABSplit-style Random Forest.

- Random Forest only
- Histogram splits
- `exact` vs `mab` split search
- Profile presets for runtime/accuracy tradeoff
- Required dependency: `numpy`
- Optional dependency (for extra datasets): `scikit-learn`

Code is split into small modules:
- `mfrf/datasets.py`
- `mfrf/preprocess.py`
- `mfrf/splitters.py`
- `mfrf/models.py`
- `minimal_fastforest_rf.py` (CLI)

## Setup

```bash
python3 -m pip install numpy
```

Optional:

```bash
python3 -m pip install scikit-learn
```

## Quick run (fast)

```bash
python3 minimal_fastforest_rf.py --quick --mode both
```

This runs on a synthetic dataset by default and prints:
- train time
- histogram insertions (sample complexity proxy)
- test accuracy
- speedup summary (`exact` vs `mab`)

The script prints `engine: mfrf-v2` at startup so you can confirm you are using the new implementation.
MAB defaults now use paper-style early stopping (`consume_all_data=False`) to improve insertion reduction.

## Profiles

- `quick`: fastest, smallest sample sizes
- `balanced`: better accuracy with moderate runtime
- `quality`: strongest results, longer runtime
- `custom`: use exactly your CLI params

Examples:

```bash
python3 minimal_fastforest_rf.py --dataset mnist --profile balanced --mode both
python3 minimal_fastforest_rf.py --dataset mnist --profile quality --mode mab
```

## MNIST run

```bash
python3 minimal_fastforest_rf.py --dataset mnist --quick --mode both
```

If MNIST files are missing, the script tries to download them into `mnist/`.
If download is unavailable, it automatically falls back to toy data.

## Better MNIST Accuracy

```bash
python3 minimal_fastforest_rf.py \
  --dataset mnist \
  --profile balanced \
  --mode both
```

Higher-quality MAB-only run:

```bash
python3 minimal_fastforest_rf.py \
  --dataset mnist \
  --profile quality \
  --mode mab \
  --n_test 2000
```

Typical outcomes (same machine, seed=0):
- `quick` (6k train): ~`0.87-0.88` accuracy, ~3s/model.
- `balanced` (12k train): ~`0.92-0.93` accuracy, ~13s/model.
- `quality` (30k train): ~`0.94-0.95` accuracy, longer runtime.

## Strict benchmark parity

To compare exact vs MAB with the same model/data setup (difference only in split-selection), run:

```bash
python3 minimal_fastforest_rf.py \
  --dataset mnist \
  --profile custom \
  --mode both \
  --n_train 12000 --n_test 2000 \
  --n_estimators 12 --max_depth 12 \
  --num_bins 64 --max_features sqrt \
  --feature_var_threshold 0.0
```

If you want conservative behavior closer to full-data MAB evaluation, add `--consume_all_data`.

## MAB-only training (no baseline)

```bash
python3 minimal_fastforest_rf.py --mode mab --quick
```

## Different datasets

Single dataset:

```bash
python3 minimal_fastforest_rf.py --dataset digits --quick --mode both
```

Multiple datasets in one run:

```bash
python3 minimal_fastforest_rf.py --datasets mnist,digits,wine --quick --mode both
```

Available dataset names:
- `toy`
- `mnist`
- `digits`
- `iris`
- `wine`
- `breast_cancer`
