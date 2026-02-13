"""
Minimal FastForest-style Random Forest benchmark + CSV experiment runner.

Core behavior:
- Exact and MABSplit share identical model/data settings.
- MABSplit differs only in split-selection routine.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from statistics import mean, pstdev
from typing import Optional

import numpy as np

from mfrf.datasets import AVAILABLE_DATASETS, load_dataset
from mfrf.models import HistogramRandomForestClassifier
from mfrf.preprocess import UniformBinner, VarianceFeatureSelector
from mfrf.splitters import ExactHistogramSplitter, MABSplitHistogramSplitter

SCRIPT_VERSION = "mfrf-v3"

CSV_FIELDS = [
    "dataset",
    "dataset_requested",
    "profile",
    "seed_start",
    "seed_stride",
    "n_runs",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "n_estimators",
    "max_depth",
    "max_features",
    "num_bins",
    "confidence_scale",
    "exact_train_time_sec_mean",
    "exact_train_time_sec_std",
    "mab_train_time_sec_mean",
    "mab_train_time_sec_std",
    "time_speedup_exact_over_mab",
    "time_speedup_exact_over_mab_std",
    "runtime_reduction_pct",
    "runtime_reduction_pct_std",
    "exact_insertions_mean",
    "exact_insertions_std",
    "mab_insertions_mean",
    "mab_insertions_std",
    "insertion_reduction_pct",
    "insertion_reduction_pct_std",
    "exact_test_accuracy_mean",
    "exact_test_accuracy_std",
    "mab_test_accuracy_mean",
    "mab_test_accuracy_std",
    "accuracy_gap_mab_minus_exact",
    "accuracy_gap_mab_minus_exact_std",
    "note",
]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="toy")
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset list, e.g. mnist,digits,wine",
    )
    p.add_argument(
        "--dataset_specs",
        type=str,
        default="",
        help="Semicolon-separated specs: dataset:n_train:n_test[:profile]",
    )
    p.add_argument(
        "--dataset_group",
        choices=["none", "small", "medium", "big", "small_big", "all"],
        default="none",
        help="Predefined real-dataset groups with reasonable sizes.",
    )
    p.add_argument("--data_dir", type=str, default="mnist")
    p.add_argument("--mode", choices=["both", "mab", "exact"], default="both")

    p.add_argument(
        "--profile",
        choices=["quick", "balanced", "quality", "custom"],
        default="quick",
    )
    p.add_argument("--quick", action="store_true", help="Alias for --profile quick")

    p.add_argument("--n_train", type=int, default=6000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--toy_features", type=int, default=64)
    p.add_argument("--toy_classes", type=int, default=10)

    p.add_argument("--n_estimators", type=int, default=6)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--max_features", type=str, default="auto")
    p.add_argument("--min_impurity_decrease", type=float, default=0.0)

    p.add_argument("--num_bins", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--mab_min_samples", type=int, default=1200)
    p.add_argument("--check_every", type=int, default=4)
    p.add_argument("--confidence_scale", type=float, default=0.15)
    p.add_argument("--stop_active_features", type=int, default=3)
    p.add_argument("--min_batches_before_stop", type=int, default=2)
    p.add_argument("--consume_all_data", action="store_true")
    p.add_argument("--feature_var_threshold", type=float, default=0.0)

    p.add_argument("--runs", type=int, default=1, help="Number of seeds per dataset.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_stride", type=int, default=1)

    p.add_argument("--results_csv", type=str, default="result.csv")
    p.add_argument("--append_results", action="store_true")
    return p.parse_args()


def parse_dataset_list(args: argparse.Namespace) -> list[str]:
    if not args.datasets.strip():
        return [args.dataset]
    valid = set(AVAILABLE_DATASETS)
    out: list[str] = []
    for raw in args.datasets.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in valid:
            raise ValueError(f"unknown dataset in --datasets: {name}")
        out.append(name)
    if not out:
        raise ValueError("--datasets was provided but empty after parsing")
    return out


def parse_dataset_specs(specs: str) -> list[tuple[str, int, int, Optional[str]]]:
    valid = set(AVAILABLE_DATASETS)
    out: list[tuple[str, int, int, Optional[str]]] = []
    for raw in specs.split(";"):
        part = raw.strip()
        if not part:
            continue
        toks = [t.strip() for t in part.split(":") if t.strip()]
        if len(toks) < 3:
            raise ValueError(
                "Each --dataset_specs entry must be dataset:n_train:n_test[:profile]"
            )
        name = toks[0]
        if name not in valid:
            raise ValueError(f"unknown dataset in --dataset_specs: {name}")
        n_train = int(toks[1])
        n_test = int(toks[2])
        profile_override = toks[3] if len(toks) >= 4 else None
        out.append((name, n_train, n_test, profile_override))
    if not out:
        raise ValueError("--dataset_specs was provided but no valid specs were parsed")
    return out


def dataset_group_specs(group: str) -> list[tuple[str, int, int, Optional[str]]]:
    small = [
        ("iris", 120, 30, "quick"),
        ("wine", 130, 48, "quick"),
        ("breast_cancer", 400, 150, "quick"),
        ("digits", 1300, 400, "quick"),
    ]
    medium = [
        ("mnist", 12000, 2000, "balanced"),
    ]
    big = [
        # Keep big runs around 500k-1M combined samples for practical runtimes.
        ("covtype", 500000, 80000, "custom"),
        ("kddcup99_10p", 400000, 80000, "custom"),
        ("kddcup99_full", 900000, 100000, "custom"),
    ]
    if group == "small":
        return small
    if group == "medium":
        return medium
    if group == "big":
        return big
    if group == "small_big":
        return small + big
    if group == "all":
        return small + medium + big
    return []


def build_dataset_plan(args: argparse.Namespace) -> list[tuple[str, int, int, Optional[str]]]:
    if args.dataset_specs.strip():
        return parse_dataset_specs(args.dataset_specs)
    if args.dataset_group != "none":
        return dataset_group_specs(args.dataset_group)
    return [(name, args.n_train, args.n_test, None) for name in parse_dataset_list(args)]


def resolve_max_features(spec: str, dataset_name: str) -> str | int:
    s = spec.strip().lower()
    if s == "auto":
        return (
            "sqrt"
            if dataset_name in ("mnist", "covtype", "kddcup99_10p", "kddcup99_full")
            else "all"
        )
    if s in ("all", "sqrt"):
        return s
    if s.isdigit():
        return int(s)
    raise ValueError("max_features must be auto, all, sqrt, or an integer")


def apply_profile(args: argparse.Namespace, dataset_name: str) -> None:
    if args.quick:
        args.profile = "quick"
    if args.profile == "custom":
        return

    if args.profile == "quick":
        if dataset_name == "mnist":
            args.n_train = min(args.n_train, 6000)
            args.n_test = min(args.n_test, 2000)
            args.n_estimators = min(args.n_estimators, 6)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 48)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.mab_min_samples = max(args.mab_min_samples, 1200)
        elif dataset_name == "covtype":
            args.n_train = min(args.n_train, 200000)
            args.n_test = min(args.n_test, 50000)
            args.n_estimators = min(args.n_estimators, 4)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
        elif dataset_name == "kddcup99_10p":
            args.n_train = min(args.n_train, 300000)
            args.n_test = min(args.n_test, 80000)
            args.n_estimators = min(args.n_estimators, 5)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
        elif dataset_name == "kddcup99_full":
            args.n_train = min(args.n_train, 400000)
            args.n_test = min(args.n_test, 100000)
            args.n_estimators = min(args.n_estimators, 4)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
        elif dataset_name == "digits":
            args.n_train = min(args.n_train, 1300)
            args.n_test = min(args.n_test, 400)
            args.n_estimators = min(args.n_estimators, 10)
            args.max_depth = min(args.max_depth, 12)
            args.num_bins = min(args.num_bins, 48)
        elif dataset_name in ("iris", "wine", "breast_cancer"):
            args.n_train = min(args.n_train, 400)
            args.n_test = min(args.n_test, 200)
            args.n_estimators = min(args.n_estimators, 20)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 32)
        else:
            args.n_train = min(args.n_train, 5000)
            args.n_test = min(args.n_test, 1500)
            args.n_estimators = min(args.n_estimators, 6)
            args.max_depth = min(args.max_depth, 10)
            args.num_bins = min(args.num_bins, 48)
        return

    if args.profile == "balanced":
        if dataset_name == "mnist":
            args.n_train = max(args.n_train, 12000)
            args.n_test = max(args.n_test, 2000)
            args.n_estimators = max(args.n_estimators, 12)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 768)
            args.mab_min_samples = max(args.mab_min_samples, 1800)
        elif dataset_name == "covtype":
            args.n_train = max(args.n_train, 500000)
            args.n_test = max(args.n_test, 80000)
            args.n_estimators = max(args.n_estimators, 6)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 2400)
        elif dataset_name == "kddcup99_10p":
            args.n_train = max(args.n_train, 400000)
            args.n_test = max(args.n_test, 80000)
            args.n_estimators = max(args.n_estimators, 6)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 2400)
        elif dataset_name == "kddcup99_full":
            args.n_train = max(args.n_train, 900000)
            args.n_test = max(args.n_test, 100000)
            args.n_estimators = max(args.n_estimators, 6)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 2600)
        else:
            args.n_estimators = max(args.n_estimators, 10)
            args.max_depth = max(args.max_depth, 10)
            args.num_bins = max(args.num_bins, 48)
        return

    if args.profile == "quality":
        if dataset_name == "mnist":
            args.n_train = max(args.n_train, 30000)
            args.n_test = max(args.n_test, 2000)
            args.n_estimators = max(args.n_estimators, 20)
            args.max_depth = max(args.max_depth, 14)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 2400)
        elif dataset_name == "covtype":
            args.n_train = max(args.n_train, 550000)
            args.n_test = max(args.n_test, 80000)
            args.n_estimators = max(args.n_estimators, 10)
            args.max_depth = max(args.max_depth, 14)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 3000)
        elif dataset_name == "kddcup99_10p":
            args.n_train = max(args.n_train, 450000)
            args.n_test = max(args.n_test, 80000)
            args.n_estimators = max(args.n_estimators, 10)
            args.max_depth = max(args.max_depth, 14)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 3000)
        elif dataset_name == "kddcup99_full":
            args.n_train = max(args.n_train, 1000000)
            args.n_test = max(args.n_test, 120000)
            args.n_estimators = max(args.n_estimators, 10)
            args.max_depth = max(args.max_depth, 14)
            args.num_bins = max(args.num_bins, 64)
            args.max_features = "sqrt" if args.max_features == "auto" else args.max_features
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 3200)
        else:
            args.n_estimators = max(args.n_estimators, 14)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)


def make_group_key(row: dict[str, object]) -> tuple[object, ...]:
    fields = (
        "dataset_requested",
        "dataset",
        "profile",
        "n_train",
        "n_test",
        "n_estimators",
        "max_depth",
        "max_features",
        "num_bins",
        "confidence_scale",
    )
    return tuple(row.get(f) for f in fields)


def benchmark_model(
    name: str,
    model: HistogramRandomForestClassifier,
    Xb_tr: np.ndarray,
    y_tr: np.ndarray,
    Xb_te: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, float]:
    t0 = time.perf_counter()
    model.fit(Xb_tr, y_tr)
    train_time = time.perf_counter() - t0
    pred = model.predict(Xb_te)
    acc = accuracy(y_te, pred)
    out = {
        "name": name,
        "train_time": train_time,
        "insertions": float(model.insertions_),
        "accuracy": acc,
    }
    print(f"\n=== {name} ===")
    print(f"train_time_sec: {train_time:.3f}")
    print(f"insertions: {model.insertions_}")
    print(f"test_accuracy: {acc:.4f}")
    return out


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), pstdev(vals)


def write_rows_csv(path: str, rows: list[dict[str, object]], append: bool) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    write_header = True
    mode = "w"
    if append and os.path.exists(path) and os.path.getsize(path) > 0:
        write_header = False
        mode = "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    plan = build_dataset_plan(args)
    print(f"engine: {SCRIPT_VERSION}")
    print(f"datasets_in_plan: {len(plan)}, runs_per_dataset: {args.runs}")

    run_rows: list[dict[str, object]] = []
    skipped_by_group: dict[tuple[object, ...], dict[str, object]] = {}

    for ds_name, ds_n_train, ds_n_test, profile_override in plan:
        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx * args.seed_stride
            run_args = argparse.Namespace(**vars(args))
            run_args.n_train = ds_n_train
            run_args.n_test = ds_n_test
            run_args.seed = run_seed
            if profile_override is not None:
                run_args.profile = profile_override
            apply_profile(run_args, ds_name)

            try:
                Xtr, ytr, Xte, yte, resolved_name = load_dataset(
                    dataset_name=ds_name,
                    data_dir=run_args.data_dir,
                    n_train=run_args.n_train,
                    n_test=run_args.n_test,
                    seed=run_args.seed,
                    toy_features=run_args.toy_features,
                    toy_classes=run_args.toy_classes,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"\n[{ds_name}] run {run_idx} skipped ({type(exc).__name__}: {exc})")
                skip_row = {
                    "dataset": ds_name,
                    "dataset_requested": ds_name,
                    "profile": run_args.profile,
                    "seed_start": args.seed,
                    "seed_stride": args.seed_stride,
                    "n_runs": args.runs,
                    "n_train": run_args.n_train,
                    "n_test": run_args.n_test,
                    "n_estimators": run_args.n_estimators,
                    "max_depth": run_args.max_depth,
                    "max_features": resolve_max_features(run_args.max_features, ds_name),
                    "num_bins": run_args.num_bins,
                    "confidence_scale": run_args.confidence_scale,
                    "n_features": "",
                    "n_classes": "",
                    "note": f"skipped: {type(exc).__name__}: {exc}",
                }
                key = make_group_key(skip_row)
                if key not in skipped_by_group:
                    skipped_by_group[key] = dict(skip_row)
                    skipped_by_group[key]["_skip_count"] = 0
                skipped_by_group[key]["_skip_count"] = int(skipped_by_group[key]["_skip_count"]) + 1
                continue

            selector = VarianceFeatureSelector(run_args.feature_var_threshold).fit(Xtr)
            Xtr_sel = selector.transform(Xtr)
            Xte_sel = selector.transform(Xte)
            binner = UniformBinner(run_args.num_bins).fit(Xtr_sel)
            Xb_tr = binner.transform(Xtr_sel)
            Xb_te = binner.transform(Xte_sel)
            max_features = resolve_max_features(run_args.max_features, resolved_name)

            n_train_actual, n_features_raw = Xtr.shape
            n_test_actual = Xte.shape[0]
            n_features_selected = Xtr_sel.shape[1]
            n_classes = int(np.max(ytr)) + 1

            print(f"\ndataset: {resolved_name} (requested={ds_name}) run={run_idx + 1}/{args.runs}")
            print(f"train: {Xb_tr.shape}, test: {Xb_te.shape}, profile={run_args.profile}")
            print(
                f"n_estimators={run_args.n_estimators}, max_depth={run_args.max_depth}, "
                f"max_features={max_features}, bins={run_args.num_bins}"
            )

            def build_exact() -> HistogramRandomForestClassifier:
                return HistogramRandomForestClassifier(
                    splitter_builder=lambda: ExactHistogramSplitter(n_bins=run_args.num_bins),
                    n_estimators=run_args.n_estimators,
                    max_depth=run_args.max_depth,
                    min_samples_split=run_args.min_samples_split,
                    min_samples_leaf=run_args.min_samples_leaf,
                    max_features=max_features,
                    min_impurity_decrease=run_args.min_impurity_decrease,
                    random_state=run_args.seed,
                )

            def build_mab() -> HistogramRandomForestClassifier:
                return HistogramRandomForestClassifier(
                    splitter_builder=lambda: MABSplitHistogramSplitter(
                        n_bins=run_args.num_bins,
                        batch_size=run_args.batch_size,
                        mab_min_samples=run_args.mab_min_samples,
                        check_every=run_args.check_every,
                        confidence_scale=run_args.confidence_scale,
                        stop_active_features=run_args.stop_active_features,
                        min_batches_before_stop=run_args.min_batches_before_stop,
                        consume_all_data=run_args.consume_all_data,
                    ),
                    n_estimators=run_args.n_estimators,
                    max_depth=run_args.max_depth,
                    min_samples_split=run_args.min_samples_split,
                    min_samples_leaf=run_args.min_samples_leaf,
                    max_features=max_features,
                    min_impurity_decrease=run_args.min_impurity_decrease,
                    random_state=run_args.seed,
                )

            results: dict[str, dict[str, float]] = {}
            if run_args.mode in ("both", "exact"):
                results["exact"] = benchmark_model(
                    "Exact Histogram RF", build_exact(), Xb_tr, ytr, Xb_te, yte
                )
            if run_args.mode in ("both", "mab"):
                results["mab"] = benchmark_model(
                    "MABSplit Histogram RF", build_mab(), Xb_tr, ytr, Xb_te, yte
                )

            speed = None
            qred = None
            tred = None
            acc_gap = None

            if "exact" in results and "mab" in results:
                t_exact = results["exact"]["train_time"]
                t_mab = results["mab"]["train_time"]
                q_exact = results["exact"]["insertions"]
                q_mab = results["mab"]["insertions"]
                speed = t_exact / max(t_mab, 1e-12)
                qred = (1.0 - q_mab / max(q_exact, 1.0)) * 100.0
                tred = (1.0 - t_mab / max(t_exact, 1e-12)) * 100.0
                acc_gap = results["mab"]["accuracy"] - results["exact"]["accuracy"]

                print("\n=== Speedup Summary ===")
                print(f"time_speedup_exact_over_mab: {speed:.3f}x")
                print(f"runtime_reduction_mab_vs_exact: {tred:.2f}%")
                print(f"insertion_reduction: {qred:.2f}%")
                print(f"accuracy_gap_mab_minus_exact: {acc_gap:+.4f}")

            run_row = {
                "dataset": resolved_name,
                "dataset_requested": ds_name,
                "profile": run_args.profile,
                "seed_start": args.seed,
                "seed_stride": args.seed_stride,
                "n_runs": args.runs,
                "n_train": n_train_actual,
                "n_test": n_test_actual,
                "n_features": n_features_selected if n_features_selected > 0 else n_features_raw,
                "n_classes": n_classes,
                "n_estimators": run_args.n_estimators,
                "max_depth": run_args.max_depth,
                "max_features": max_features,
                "num_bins": run_args.num_bins,
                "confidence_scale": run_args.confidence_scale,
                "exact_train_time_sec_mean": (
                    results["exact"]["train_time"] if "exact" in results else None
                ),
                "mab_train_time_sec_mean": (
                    results["mab"]["train_time"] if "mab" in results else None
                ),
                "time_speedup_exact_over_mab": speed,
                "runtime_reduction_pct": tred,
                "exact_insertions_mean": (
                    results["exact"]["insertions"] if "exact" in results else None
                ),
                "mab_insertions_mean": (
                    results["mab"]["insertions"] if "mab" in results else None
                ),
                "insertion_reduction_pct": qred,
                "exact_test_accuracy_mean": (
                    results["exact"]["accuracy"] if "exact" in results else None
                ),
                "mab_test_accuracy_mean": (
                    results["mab"]["accuracy"] if "mab" in results else None
                ),
                "accuracy_gap_mab_minus_exact": acc_gap,
                "note": "",
            }
            run_rows.append(run_row)

    # Aggregate summaries.
    summary_rows: list[dict[str, object]] = []
    grouped_runs: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in run_rows:
        key = make_group_key(row)
        grouped_runs.setdefault(key, []).append(row)

    def agg(rows: list[dict[str, object]], col: str) -> tuple[object, object]:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        if not vals:
            return "", ""
        m, s = mean_std(vals)
        return m, s

    for key, rows in grouped_runs.items():
        row0 = rows[0]
        srow = {k: row0.get(k, "") for k in CSV_FIELDS}

        ex_t_m, ex_t_s = agg(rows, "exact_train_time_sec_mean")
        mb_t_m, mb_t_s = agg(rows, "mab_train_time_sec_mean")
        ex_q_m, ex_q_s = agg(rows, "exact_insertions_mean")
        mb_q_m, mb_q_s = agg(rows, "mab_insertions_mean")
        ex_a_m, ex_a_s = agg(rows, "exact_test_accuracy_mean")
        mb_a_m, mb_a_s = agg(rows, "mab_test_accuracy_mean")
        sp_m, sp_s = agg(rows, "time_speedup_exact_over_mab")
        tr_m, tr_s = agg(rows, "runtime_reduction_pct")
        qr_m, qr_s = agg(rows, "insertion_reduction_pct")
        gp_m, gp_s = agg(rows, "accuracy_gap_mab_minus_exact")

        srow.update(
            {
                "n_runs": len(rows),
                "exact_train_time_sec_mean": ex_t_m,
                "mab_train_time_sec_mean": mb_t_m,
                "exact_insertions_mean": ex_q_m,
                "mab_insertions_mean": mb_q_m,
                "exact_test_accuracy_mean": ex_a_m,
                "mab_test_accuracy_mean": mb_a_m,
                "time_speedup_exact_over_mab": sp_m,
                "runtime_reduction_pct": tr_m,
                "insertion_reduction_pct": qr_m,
                "accuracy_gap_mab_minus_exact": gp_m,
                "exact_train_time_sec_std": ex_t_s,
                "mab_train_time_sec_std": mb_t_s,
                "exact_insertions_std": ex_q_s,
                "mab_insertions_std": mb_q_s,
                "exact_test_accuracy_std": ex_a_s,
                "mab_test_accuracy_std": mb_a_s,
                "time_speedup_exact_over_mab_std": sp_s,
                "runtime_reduction_pct_std": tr_s,
                "insertion_reduction_pct_std": qr_s,
                "accuracy_gap_mab_minus_exact_std": gp_s,
            }
        )
        if key in skipped_by_group:
            skip_count = int(skipped_by_group[key]["_skip_count"])
            srow["note"] = f"partial_skips={skip_count}/{args.runs}"
        summary_rows.append(srow)

    # Groups where all runs were skipped.
    for key, info in skipped_by_group.items():
        if key in grouped_runs:
            continue
        row = {k: "" for k in CSV_FIELDS}
        for k in (
            "dataset",
            "dataset_requested",
            "profile",
            "seed_start",
            "seed_stride",
            "n_runs",
            "n_train",
            "n_test",
            "n_estimators",
            "max_depth",
            "max_features",
            "num_bins",
            "confidence_scale",
            "n_features",
            "n_classes",
        ):
            row[k] = info.get(k, "")
        row["n_runs"] = 0
        row["note"] = f"all_runs_skipped={int(info['_skip_count'])}/{args.runs}: {info.get('note','')}"
        summary_rows.append(row)

    all_rows = summary_rows
    if args.results_csv:
        write_rows_csv(args.results_csv, all_rows, append=args.append_results)
        print(f"\nresults_csv_written: {args.results_csv} ({len(all_rows)} rows)")

    if summary_rows:
        print("\n=== Summary Rows ===")
        for row in summary_rows:
            if row.get("time_speedup_exact_over_mab", "") in ("", None):
                print(
                    f"{row['dataset_requested']} [{row['profile']}] n={row['n_train']}/{row['n_test']}: "
                    f"{row.get('note','')}"
                )
                continue
            print(
                f"{row['dataset']} [{row['profile']}] n={row['n_train']}/{row['n_test']}: "
                f"speedup={float(row['time_speedup_exact_over_mab']):.3f}x, "
                f"runtime_reduction={float(row['runtime_reduction_pct']):.2f}%, "
                f"insertion_reduction={float(row['insertion_reduction_pct']):.2f}%, "
                f"acc_gap={float(row['accuracy_gap_mab_minus_exact']):+.4f}, "
                f"runs={row['n_runs']}"
            )


if __name__ == "__main__":
    main()
