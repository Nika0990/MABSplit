# Minimal FastForest-Style RF

Minimal implementation of a FastForest/MABSplit-style Random Forest benchmark with CSV experiment logging.

Last verified: 2026-02-20.

## What this project does

- Trains histogram-based Random Forest models.
- Compares `exact` split search vs `mab` split search.
- Supports fast/quality profiles for runtime vs accuracy tradeoffs.
- Writes aggregated benchmark summaries to CSV.

Main files:
- `minimal_fastforest_rf.py` (CLI entrypoint)
- `mfrf/datasets.py` (dataset loading)
- `mfrf/preprocess.py` (feature selection + binning)
- `mfrf/splitters.py` (exact and MAB splitters)
- `mfrf/models.py` (tree + forest models)

## Requirements

- Python 3.10+ (tested with modern Python 3).
- Required package: `numpy`.
- Optional package: `scikit-learn` (needed for `digits`, `iris`, `wine`, `breast_cancer`, `covtype`, `aps_failure`, `kddcup99_*` datasets).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install scikit-learn  # optional, for non-MNIST real datasets
```

## Run the code

### 1) Quick smoke test (recommended first run)

```bash
python3 minimal_fastforest_rf.py --dataset toy --quick --mode both
```

This should print:
- `engine: mfrf-v3`
- train time
- insertion counts
- test accuracy
- speedup summary for exact vs MAB
- `results_csv_written: result.csv (...)`

### 2) MNIST benchmark

```bash
python3 minimal_fastforest_rf.py --dataset mnist --profile balanced --mode both
```

If MNIST files are missing, the script tries to download them into `mnist/`.

### 3) Multi-run CSV experiment

```bash
python3 minimal_fastforest_rf.py \
  --dataset mnist \
  --profile balanced \
  --mode both \
  --runs 5 \
  --results_csv mnist_runs.csv
```

CSV includes one aggregated row per configuration with mean/std columns.

## Important CLI options

- `--mode {both,mab,exact}`: choose which model(s) to run.
- `--profile {quick,balanced,quality,custom}`: apply preset runtime/accuracy configs.
- `--runs N`: run multiple seeds and aggregate.
- `--seed` and `--seed_stride`: control deterministic seed schedule.
- `--results_csv PATH`: output summary CSV path.
- `--append_results`: append rows instead of overwrite.

Show all options:

```bash
python3 minimal_fastforest_rf.py --help
```

## Dataset selection

Single dataset:

```bash
python3 minimal_fastforest_rf.py --dataset mnist --quick --mode both
```

Multiple datasets:

```bash
python3 minimal_fastforest_rf.py --datasets mnist,digits,wine --quick --mode both
```

Explicit dataset specs:

```bash
python3 minimal_fastforest_rf.py \
  --dataset_specs "mnist:12000:2000:balanced;digits:1300:400:quick" \
  --runs 3 \
  --mode both \
  --results_csv mixed_specs.csv
```

Dataset groups:

```bash
python3 minimal_fastforest_rf.py --dataset_group small --runs 5 --mode both
python3 minimal_fastforest_rf.py --dataset_group medium --runs 3 --mode both
python3 minimal_fastforest_rf.py --dataset_group big --runs 2 --mode both
python3 minimal_fastforest_rf.py --dataset_group all --runs 2 --mode both
```

APS Failure (OpenML) example:

```bash
python3 minimal_fastforest_rf.py --dataset aps_failure --profile quick --mode both
```

`aps_failure` is fetched from OpenML on first use and requires network access.

## Reproducibility notes

- Exact and MAB use identical data/model settings; only split strategy differs.
- MAB treats each `(feature, threshold)` candidate as an arm during elimination.
- `time_speedup_exact_over_mab > 1` means MAB is faster.
- Add `--consume_all_data` to disable MAB early stopping.
