# Minimal FastForest-Style RF

Minimal implementation of a FastForest/MABSplit-style Random Forest with
multi-run experiment logging to CSV.

- Random Forest only
- Histogram splits
- `exact` vs `mab` split search
- Profile presets for runtime/accuracy tradeoff
- Required dependency: `numpy`
- Optional dependency (for real tabular datasets): `scikit-learn`

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
- CSV file (`result.csv` by default)

The script prints `engine: mfrf-v3` at startup so you can confirm you are using the new implementation.
MAB defaults now use paper-style early stopping (`consume_all_data=False`) to improve insertion reduction.

## CSV Experiments (multi-run averages)

Run multiple seeds and save summary rows:

```bash
python3 minimal_fastforest_rf.py \
  --dataset mnist \
  --profile balanced \
  --mode both \
  --runs 5 \
  --results_csv mnist_runs.csv
```

CSV contains:
- one compact row per dataset-size/config with mean/std over runs
- no duplicate benchmark/mab rows; exact and mab are side-by-side columns

Key comparison columns:
- `time_speedup_exact_over_mab` (ratio, >1 means MAB faster)
- `runtime_reduction_pct` (percent runtime reduction for MAB vs exact)
- `insertion_reduction_pct` (percent insertion reduction for MAB vs exact)

Useful flags:
- `--runs`: number of runs per dataset
- `--seed` and `--seed_stride`: seed schedule
- `--results_csv`: output CSV path
- `--append_results`: append to existing CSV

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

## Dataset selection options

Single dataset:

```bash
python3 minimal_fastforest_rf.py --dataset mnist --quick --mode both
```

Multiple datasets:

```bash
python3 minimal_fastforest_rf.py --datasets mnist,digits,wine --quick --mode both
```

Dataset specs with explicit sizes:

```bash
python3 minimal_fastforest_rf.py \
  --dataset_specs "mnist:12000:2000:balanced;digits:1300:400:quick;covtype:500000:80000:custom" \
  --runs 3 \
  --mode both \
  --results_csv mixed_specs.csv
```

Predefined groups:

```bash
python3 minimal_fastforest_rf.py --dataset_group small --runs 5 --mode both
python3 minimal_fastforest_rf.py --dataset_group medium --runs 3 --mode both
python3 minimal_fastforest_rf.py --dataset_group big --runs 2 --mode both
python3 minimal_fastforest_rf.py --dataset_group small_big --runs 5 --mode both
python3 minimal_fastforest_rf.py --dataset_group all --runs 2 --mode both
```

Run all configured datasets with 5 seeds and save to the default output file:

```bash
python3 minimal_fastforest_rf.py --dataset_group all --runs 5 --mode both
```

`big` uses real large datasets:
- `covtype` (500k train / 80k test)
- `kddcup99_10p` (400k train / 80k test)
- `kddcup99_full` (900k train / 100k test)

`small_big` mixes the small tabular datasets with these big datasets in one run.
For large datasets, explicit `n_train + n_test` sizes are sampled directly from the full dataset when feasible.

## MAB-only training

```bash
python3 minimal_fastforest_rf.py --mode mab --quick
```

Available dataset names:
- `toy`
- `mnist`
- `digits`
- `iris`
- `wine`
- `breast_cancer`
- `covtype`
- `kddcup99_10p`
- `kddcup99_full`
