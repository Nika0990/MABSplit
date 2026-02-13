"""
Minimal FastForest-style Random Forest benchmark.

Key goals:
- Keep setup small (numpy only; sklearn optional for extra datasets).
- Keep runtime controllable.
- Provide stronger preset profiles for better MNIST results.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mfrf.datasets import AVAILABLE_DATASETS, load_dataset
from mfrf.models import HistogramRandomForestClassifier
from mfrf.preprocess import UniformBinner, VarianceFeatureSelector
from mfrf.splitters import ExactHistogramSplitter, MABSplitHistogramSplitter

SCRIPT_VERSION = "mfrf-v2"


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="toy")
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list, e.g. mnist,digits,wine",
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
    p.add_argument(
        "--consume_all_data",
        action="store_true",
        help="Disable early-stop behavior and finish all remaining samples for active features.",
    )
    p.add_argument("--feature_var_threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
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


def resolve_max_features(spec: str, dataset_name: str) -> str | int:
    s = spec.strip().lower()
    if s == "auto":
        return "sqrt" if dataset_name == "mnist" else "all"
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
            args.min_impurity_decrease = min(args.min_impurity_decrease, 0.0)
            args.mab_min_samples = max(args.mab_min_samples, 1200)
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
            args.min_impurity_decrease = 0.0
            args.batch_size = max(args.batch_size, 768)
            args.mab_min_samples = max(args.mab_min_samples, 1800)
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
            args.min_impurity_decrease = 0.0
            args.batch_size = max(args.batch_size, 1024)
            args.mab_min_samples = max(args.mab_min_samples, 2400)
        else:
            args.n_estimators = max(args.n_estimators, 14)
            args.max_depth = max(args.max_depth, 12)
            args.num_bins = max(args.num_bins, 64)


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


def main() -> None:
    args = parse_args()
    dataset_list = parse_dataset_list(args)
    multi_summary: list[tuple[str, float, float, float, float]] = []
    print(f"engine: {SCRIPT_VERSION}")

    for ds_name in dataset_list:
        run_args = argparse.Namespace(**vars(args))
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
            if ds_name == "mnist":
                print(f"\n[{ds_name}] unavailable ({type(exc).__name__}), falling back to toy.")
                Xtr, ytr, Xte, yte, resolved_name = load_dataset(
                    dataset_name="toy",
                    data_dir=run_args.data_dir,
                    n_train=run_args.n_train,
                    n_test=run_args.n_test,
                    seed=run_args.seed,
                    toy_features=run_args.toy_features,
                    toy_classes=run_args.toy_classes,
                )
            else:
                print(f"\n[{ds_name}] skipped ({type(exc).__name__}: {exc})")
                continue

        selector = VarianceFeatureSelector(run_args.feature_var_threshold).fit(Xtr)
        Xtr_sel = selector.transform(Xtr)
        Xte_sel = selector.transform(Xte)
        binner = UniformBinner(run_args.num_bins).fit(Xtr_sel)
        Xb_tr = binner.transform(Xtr_sel)
        Xb_te = binner.transform(Xte_sel)
        max_features = resolve_max_features(run_args.max_features, resolved_name)

        print(f"\ndataset: {resolved_name}")
        print(f"train: {Xb_tr.shape}, test: {Xb_te.shape}")
        print(
            f"profile={run_args.profile}, n_estimators={run_args.n_estimators}, "
            f"max_depth={run_args.max_depth}, max_features={max_features}, bins={run_args.num_bins}"
        )
        print(
            f"min_samples_leaf={run_args.min_samples_leaf}, "
            f"selected_features={selector.n_selected_}"
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

        if "exact" in results and "mab" in results:
            t_exact = results["exact"]["train_time"]
            t_mab = results["mab"]["train_time"]
            q_exact = results["exact"]["insertions"]
            q_mab = results["mab"]["insertions"]
            print("\n=== Speedup Summary ===")
            if t_mab > 0:
                print(f"time_speedup_exact_over_mab: {t_exact / t_mab:.3f}x")
            if q_mab > 0:
                print(f"insertion_reduction: {(1.0 - q_mab / q_exact) * 100:.2f}%")
            multi_summary.append(
                (
                    resolved_name,
                    t_exact / max(t_mab, 1e-12),
                    (1.0 - q_mab / max(q_exact, 1.0)) * 100.0,
                    results["exact"]["accuracy"],
                    results["mab"]["accuracy"],
                )
            )

    if len(multi_summary) > 1:
        print("\n=== Multi-Dataset Summary ===")
        print("dataset | time_speedup | insertion_reduction(%) | exact_acc | mab_acc")
        for ds_name, spd, qred, aex, amab in multi_summary:
            print(f"{ds_name} | {spd:.3f}x | {qred:.2f} | {aex:.4f} | {amab:.4f}")


if __name__ == "__main__":
    main()
