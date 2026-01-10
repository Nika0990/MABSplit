"""
FastForest-style MNIST experiment reimplementation in a single file.

This mirrors the algorithmic structure from the MABSplit/FastForest paper:
- Histogram-based node splitting with linear bins.
- MABSplit with batched sampling + successive elimination + delta-method CIs.
- Best-first tree growth.
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------
# Defaults (aligned to FastForest)
# -----------------------------
DEFAULT_NUM_BINS = 11
DEFAULT_MIN_IMPURITY_DECREASE = 5e-3
BATCH_SIZE = 1000
CONF_MULTIPLIER = 1.0
DEFAULT_EPSILON = 0.01


# -----------------------------
# Data: MNIST downloader/loader
# -----------------------------
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_URL_ROOTS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _download(url: str, path: str) -> None:
    from urllib.request import urlopen

    with urlopen(url) as r, open(path, "wb") as f:
        f.write(r.read())


def ensure_mnist(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            continue
        last_err = None
        for root in MNIST_URL_ROOTS:
            url = root + filename
            try:
                print(f"downloading {filename} from {root}...")
                _download(url, path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic {magic} in {path}")
        data = f.read(n * rows * cols)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows * cols)
        return arr.astype(np.float32)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic {magic} in {path}")
        data = f.read(n)
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr.astype(np.int64)


def load_mnist(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensure_mnist(data_dir)
    Xtr = _read_idx_images(os.path.join(data_dir, MNIST_FILES["train_images"]))
    ytr = _read_idx_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    Xte = _read_idx_images(os.path.join(data_dir, MNIST_FILES["test_images"]))
    yte = _read_idx_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))
    return Xtr, ytr, Xte, yte


# -----------------------------
# Metrics
# -----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# -----------------------------
# Histogram binning
# -----------------------------
def linear_bin_edges(min_vals: np.ndarray, max_vals: np.ndarray, num_bins: int) -> List[Optional[np.ndarray]]:
    edges: List[Optional[np.ndarray]] = []
    for mn, mx in zip(min_vals, max_vals):
        if mn == mx:
            edges.append(None)
        else:
            edges.append(np.linspace(mn, mx, num_bins, dtype=np.float32))
    return edges


def linear_bin_indices(
    X: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    # Mirrors FastForest's linear binning (ceil-based index).
    bin_width = (max_vals - min_vals) / (num_bins - 1)
    safe_width = np.where(bin_width == 0, 1.0, bin_width)
    idx = np.ceil((X - min_vals) / safe_width)
    idx = np.clip(idx, 0, num_bins)
    return idx.astype(np.int64)


def hist3d_counts_for_features(
    x_bins: np.ndarray, y: np.ndarray, num_bins: int, n_classes: int
) -> np.ndarray:
    # x_bins: (n, m) in [0, num_bins]
    n, m = x_bins.shape
    if n == 0 or m == 0:
        return np.zeros((m, num_bins + 1, n_classes), dtype=np.int64)
    f_offsets = (np.arange(m, dtype=np.int64) * (num_bins + 1)).reshape(1, m)
    idx = (x_bins.astype(np.int64) + f_offsets) * n_classes + y[:, None].astype(np.int64)
    out = np.bincount(idx.ravel(), minlength=m * (num_bins + 1) * n_classes)
    return out.reshape(m, num_bins + 1, n_classes)


# -----------------------------
# Impurity + variance (delta method)
# -----------------------------
def gini_and_var(
    counts_vec: np.ndarray,
    pop_size: Optional[np.ndarray] = None,
    n: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Follows FastForest's delta-method variance for Gini.
    if counts_vec.ndim == 1:
        counts_vec = np.expand_dims(counts_vec, 0)
    if n is None:
        n = np.sum(counts_vec, axis=1, dtype=np.int64)
    if isinstance(n, (int, np.integer)):
        n = np.array([n], dtype=np.int64)
    np.seterr(divide="ignore", invalid="ignore")
    p = counts_vec / np.expand_dims(n, axis=1)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    g = 1.0 - np.sum(p * p, axis=1)

    V_p = p * (1.0 - p) / np.expand_dims(n, axis=1)
    if pop_size is not None:
        V_p *= np.expand_dims((pop_size - n) / (pop_size - 1), axis=1)
    dG_dp = -2.0 * p[:, :-1] + 2.0 * np.expand_dims(p[:, -1], axis=1)
    V_g = np.sum(dG_dp ** 2 * V_p[:, :-1], axis=1)
    np.seterr(all="warn")
    return g, V_g


def impurity_reductions_from_counts(
    counts: np.ndarray, pop_size: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    # counts: (B+1, K)
    total_counts = counts.sum(axis=0)
    total_n = int(total_counts.sum())
    B = counts.shape[0] - 1
    if total_n <= 0:
        return np.full(B, np.nan), np.full(B, np.nan)

    cum = np.cumsum(counts[:-1], axis=0)  # (B, K)
    left_counts = cum
    right_counts = total_counts[None, :] - left_counts
    left_n = left_counts.sum(axis=1)
    right_n = total_n - left_n

    if pop_size is None:
        left_size = None
        right_size = None
    else:
        left_size = (left_n * pop_size / total_n).astype(np.int64)
        right_size = (right_n * pop_size / total_n).astype(np.int64)

    left_g, left_var = gini_and_var(left_counts, pop_size=left_size, n=left_n)
    right_g, right_var = gini_and_var(right_counts, pop_size=right_size, n=right_n)
    curr_g, curr_var = gini_and_var(total_counts, pop_size=pop_size, n=np.array([total_n]))
    curr_g = float(curr_g[0])
    curr_var = float(curr_var[0])

    left_w = left_n / total_n
    right_w = right_n / total_n
    imp = left_w * left_g + right_w * right_g
    var = (left_w ** 2) * left_var + (right_w ** 2) * right_var + curr_var
    reduction = imp - curr_g
    return reduction, var


# -----------------------------
# MABSplit + Exact solvers
# -----------------------------
def solve_exact(
    X: np.ndarray,
    y: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    num_bins: int,
) -> Tuple[int, float, float, int] | int:
    n, m = X.shape
    n_classes = int(np.max(y)) + 1
    edges = linear_bin_edges(min_vals, max_vals, num_bins)

    estimates = np.full((m, num_bins), np.inf, dtype=np.float64)
    total_queries = 0
    for i in range(m):
        if edges[i] is None:
            continue
        x_bins = linear_bin_indices(X[:, i], min_vals[i], max_vals[i], num_bins)
        counts = hist3d_counts_for_features(x_bins[:, None], y, num_bins, n_classes)[0]
        reduction, _var = impurity_reductions_from_counts(counts, pop_size=n)
        estimates[i, :] = reduction
        total_queries += n

    best_val = np.nanmin(estimates)
    best_candidates = np.column_stack(np.where(estimates == best_val))
    if best_candidates.size == 0:
        return total_queries
    best_idx = best_candidates[0]
    best_f = int(best_idx[0])
    best_edge = int(best_idx[1])
    best_reduction = float(estimates[best_f, best_edge])
    if best_reduction < 0:
        return best_f, float(edges[best_f][best_edge]), best_reduction, total_queries
    return total_queries


def solve_mab(
    X: np.ndarray,
    y: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    num_bins: int,
    epsilon: float,
    batch_size: int,
    with_replacement: bool,
    rng: np.random.Generator,
) -> Tuple[int, float, float, int] | int:
    n, m = X.shape
    n_classes = int(np.max(y)) + 1
    edges = linear_bin_edges(min_vals, max_vals, num_bins)

    estimates = np.full((m, num_bins), np.inf, dtype=np.float64)
    lcbs = np.full((m, num_bins), -np.inf, dtype=np.float64)
    ucbs = np.full((m, num_bins), np.inf, dtype=np.float64)
    cb_delta = np.zeros((m, num_bins), dtype=np.float64)
    exact_mask = np.zeros((m, num_bins), dtype=bool)

    candidates: List[Tuple[int, int]] = []
    for i in range(m):
        if edges[i] is None:
            exact_mask[i, :] = True
            continue
        for b in range(num_bins):
            candidates.append((i, b))
    candidates = list(candidates)
    if len(candidates) == 0:
        return 0

    counts = np.zeros((m, num_bins + 1, n_classes), dtype=np.int64)
    feature_active = np.array([edges[i] is not None for i in range(m)], dtype=bool)
    population_idcs = None if with_replacement else np.arange(n, dtype=np.int64)
    total_queries = 0

    while len(candidates) > 5:
        if population_idcs is not None and len(population_idcs) == 0:
            lcbs = estimates
            ucbs = estimates
            break

        if population_idcs is None:
            if n <= batch_size:
                sample_idcs = np.arange(n, dtype=np.int64)
            else:
                sample_idcs = rng.choice(n, size=batch_size, replace=True)
        else:
            remaining = len(population_idcs)
            if remaining <= batch_size:
                sample_idcs = population_idcs
                population_idcs = np.array([], dtype=np.int64)
            else:
                pick = rng.choice(remaining, size=batch_size, replace=False)
                sample_idcs = population_idcs[pick]
                population_idcs = np.delete(population_idcs, pick)

        Xb = X[sample_idcs]
        yb = y[sample_idcs]

        active_idx = np.flatnonzero(feature_active)
        if active_idx.size > 0:
            Xb_active = Xb[:, active_idx]
            mins = min_vals[active_idx]
            maxs = max_vals[active_idx]
            x_bins = linear_bin_indices(Xb_active, mins, maxs, num_bins)
            c = hist3d_counts_for_features(x_bins, yb, num_bins, n_classes)
            counts[active_idx] += c.astype(np.int64, copy=False)
            total_queries += len(sample_idcs) * active_idx.size

        pop_size = None if with_replacement else n
        for i in active_idx:
            reduction, var = impurity_reductions_from_counts(counts[i], pop_size)
            estimates[i, :] = reduction
            cb_delta[i, :] = np.sqrt(var)

        lcbs = estimates - CONF_MULTIPLIER * cb_delta
        ucbs = estimates + CONF_MULTIPLIER * cb_delta

        if np.nanmin(lcbs) > 0:
            break

        min_idx = np.unravel_index(np.nanargmin(estimates), estimates.shape)
        tied_arms = np.zeros((m, num_bins), dtype=bool)
        tied_arms[np.where(estimates < (1 - epsilon) * estimates[min_idx])] = True
        tied_arms[min_idx] = False

        cond = (
            (~exact_mask)
            & (lcbs < np.nanmin(ucbs))
            & (lcbs < 0)
            & (~tied_arms)
        )
        cand_idx = np.column_stack(np.where(cond))
        candidates = [tuple(x) for x in cand_idx]

        if len(candidates) == 0:
            break

    best_val = np.nanmin(estimates)
    best_candidates = np.column_stack(np.where(estimates == best_val))
    if best_candidates.size == 0:
        return total_queries
    best_split = random.choice(best_candidates.tolist())
    best_f = int(best_split[0])
    best_edge = int(best_split[1])
    best_reduction = float(estimates[best_f, best_edge])
    if best_reduction < 0:
        return best_f, float(edges[best_f][best_edge]), best_reduction, total_queries
    return total_queries


# -----------------------------
# Tree + Forest
# -----------------------------
@dataclass
class TreeNode:
    is_leaf: bool
    proba: Optional[np.ndarray] = None
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    idcs: Optional[np.ndarray] = None
    depth: int = 0
    proportion: float = 1.0

    best_split_computed: bool = False
    best_reduction: Optional[float] = None
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
    num_queries: int = 0


class HistogramDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_impurity_decrease: float = DEFAULT_MIN_IMPURITY_DECREASE,
        num_bins: int = DEFAULT_NUM_BINS,
        batch_size: int = BATCH_SIZE,
        epsilon: float = DEFAULT_EPSILON,
        solver: str = "mab",
        random_state: int = 0,
        with_replacement: bool = False,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.num_bins = int(num_bins)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon)
        self.solver = solver
        self.random_state = int(random_state)
        self.with_replacement = bool(with_replacement)

        self.n_classes_: Optional[int] = None
        self.root_: Optional[TreeNode] = None
        self.rng_: Optional[np.random.Generator] = None
        self.min_vals_: Optional[np.ndarray] = None
        self.max_vals_: Optional[np.ndarray] = None

        self.num_queries_: int = 0

    def _choose_features(self, n_features: int) -> np.ndarray:
        m = int(np.ceil(np.sqrt(n_features)))
        return self.rng_.choice(n_features, size=m, replace=False)

    def fit(self, X: np.ndarray, y: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.rng_ = np.random.default_rng(self.random_state)
        random.seed(self.random_state)

        self.n_classes_ = int(np.max(y)) + 1
        self.min_vals_ = min_vals
        self.max_vals_ = max_vals
        self.num_queries_ = 0

        root_idcs = np.arange(X.shape[0], dtype=np.int64)
        self.root_ = TreeNode(is_leaf=True, idcs=root_idcs, depth=0, proportion=1.0)
        self._grow_best_first(X, y)
        return self

    def _node_proba(self, y_node: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_node, minlength=self.n_classes_)
        total = counts.sum()
        return counts / max(1, total)

    def _node_splittable(self, node: TreeNode, y: np.ndarray) -> bool:
        if node.depth >= self.max_depth:
            return False
        if node.idcs is None or len(node.idcs) < self.min_samples_split:
            return False
        y_node = y[node.idcs]
        return np.unique(y_node).size > 1

    def _compute_best_split(self, X: np.ndarray, y: np.ndarray, node: TreeNode) -> Optional[float]:
        if node.best_split_computed:
            return node.best_reduction

        feature_idcs = self._choose_features(X.shape[1])
        X_node = X[node.idcs][:, feature_idcs]
        y_node = y[node.idcs]
        min_vals = self.min_vals_[feature_idcs]
        max_vals = self.max_vals_[feature_idcs]

        if self.solver == "exact":
            result = solve_exact(X_node, y_node, min_vals, max_vals, self.num_bins)
        else:
            result = solve_mab(
                X_node,
                y_node,
                min_vals,
                max_vals,
                self.num_bins,
                self.epsilon,
                self.batch_size,
                self.with_replacement,
                self.rng_,
            )

        node.best_split_computed = True

        if isinstance(result, tuple):
            f_local, threshold, reduction, num_queries = result
            node.best_feature = int(feature_idcs[f_local])
            node.best_threshold = float(threshold)
            node.best_reduction = float(reduction) * node.proportion
            node.num_queries = int(num_queries)
            self.num_queries_ += node.num_queries
            return node.best_reduction

        node.best_reduction = None
        node.num_queries = int(result)
        self.num_queries_ += node.num_queries
        return None

    def _split_node(self, X: np.ndarray, y: np.ndarray, node: TreeNode) -> None:
        f = node.best_feature
        t = node.best_threshold
        if f is None or t is None:
            return

        idcs = node.idcs
        x_f = X[idcs, f]
        mask_left = x_f <= t
        left_idcs = idcs[mask_left]
        right_idcs = idcs[~mask_left]
        if left_idcs.size == 0 or right_idcs.size == 0:
            return

        node.is_leaf = False
        node.feature = f
        node.threshold = t
        node.proba = None
        node.left = TreeNode(
            is_leaf=True,
            idcs=left_idcs,
            depth=node.depth + 1,
            proportion=node.proportion * (left_idcs.size / idcs.size),
        )
        node.right = TreeNode(
            is_leaf=True,
            idcs=right_idcs,
            depth=node.depth + 1,
            proportion=node.proportion * (right_idcs.size / idcs.size),
        )

    def _grow_best_first(self, X: np.ndarray, y: np.ndarray) -> None:
        leaves: List[TreeNode] = [self.root_]
        while True:
            best_leaf = None
            best_reduction = None
            for leaf in leaves:
                if not self._node_splittable(leaf, y):
                    leaf.is_leaf = True
                    leaf.proba = self._node_proba(y[leaf.idcs])
                    continue

                reduction = self._compute_best_split(X, y, leaf)
                if reduction is None:
                    continue
                if best_reduction is None or reduction < best_reduction:
                    best_leaf = leaf
                    best_reduction = reduction

            if best_leaf is None:
                break
            if best_reduction is None or best_reduction >= self.min_impurity_decrease:
                break

            self._split_node(X, y, best_leaf)
            leaves.remove(best_leaf)
            leaves.extend([best_leaf.left, best_leaf.right])

        # Finalize leaf probabilities
        for leaf in leaves:
            if leaf.proba is None:
                leaf.proba = self._node_proba(y[leaf.idcs])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for i in range(X.shape[0]):
            node = self.root_
            while not node.is_leaf:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.proba
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class HistogramRandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        num_bins: int = DEFAULT_NUM_BINS,
        batch_size: int = BATCH_SIZE,
        epsilon: float = DEFAULT_EPSILON,
        min_samples_split: int = 2,
        min_impurity_decrease: float = DEFAULT_MIN_IMPURITY_DECREASE,
        solver: str = "mab",
        bootstrap: bool = True,
        random_state: int = 0,
        with_replacement: bool = False,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.num_bins = int(num_bins)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.solver = solver
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.with_replacement = bool(with_replacement)

        self.trees_: List[HistogramDecisionTreeClassifier] = []
        self.n_classes_: Optional[int] = None
        self.num_queries_: int = 0
        self.min_vals_: Optional[np.ndarray] = None
        self.max_vals_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(np.max(y)) + 1
        self.min_vals_ = X.min(axis=0)
        self.max_vals_ = X.max(axis=0)

        rng = np.random.default_rng(self.random_state)
        random.seed(self.random_state)
        self.trees_ = []
        self.num_queries_ = 0
        n = X.shape[0]
        for i in range(self.n_estimators):
            if self.bootstrap:
                idcs = rng.integers(0, n, size=n, endpoint=False)
            else:
                idcs = np.arange(n, dtype=np.int64)

            tree = HistogramDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                num_bins=self.num_bins,
                batch_size=self.batch_size,
                epsilon=self.epsilon,
                solver=self.solver,
                random_state=int(rng.integers(0, 2**31 - 1)),
                with_replacement=self.with_replacement,
            )
            tree.fit(X[idcs], y[idcs], self.min_vals_, self.max_vals_)
            self.num_queries_ += tree.num_queries_
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        proba_sum = None
        for tree in self.trees_:
            p = tree.predict_proba(X)
            proba_sum = p if proba_sum is None else proba_sum + p
        return proba_sum / max(1, len(self.trees_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# -----------------------------
# Runner
# -----------------------------
def run(args):
    Xtr, ytr, Xte, yte = load_mnist(args.data_dir)
    if args.n_train and args.n_train < Xtr.shape[0]:
        rng = np.random.default_rng(args.seed)
        idcs = rng.choice(Xtr.shape[0], size=args.n_train, replace=False)
        Xtr, ytr = Xtr[idcs], ytr[idcs]
    if args.n_test and args.n_test < Xte.shape[0]:
        rng = np.random.default_rng(args.seed + 1)
        idcs = rng.choice(Xte.shape[0], size=args.n_test, replace=False)
        Xte, yte = Xte[idcs], yte[idcs]

    print(f"train: {Xtr.shape}, test: {Xte.shape}")
    print(f"bins: {args.num_bins}, batch_size: {args.batch_size}, epsilon: {args.epsilon}")

    def train_and_eval(label: str, solver: str):
        rf = HistogramRandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            num_bins=args.num_bins,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            min_samples_split=args.min_samples_split,
            min_impurity_decrease=args.min_impurity_decrease,
            solver=solver,
            bootstrap=True,
            random_state=args.seed,
            with_replacement=args.with_replacement,
        )
        t0 = time.time()
        rf.fit(Xtr, ytr)
        t1 = time.time()
        pred = rf.predict(Xte)
        acc = accuracy(yte, pred)
        print(f"\n=== {label} ===")
        print(f"train_time_sec: {t1 - t0:.3f}")
        print(f"insertions: {rf.num_queries_}")
        print(f"test_acc: {acc:.4f}")

    if args.compare in ("mab", "both"):
        train_and_eval("MABSplit RF", "mab")
    if args.compare in ("exact", "both"):
        train_and_eval("Exact Histogram RF", "exact")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="mnist")
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--n_estimators", type=int, default=5)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--num_bins", type=int, default=DEFAULT_NUM_BINS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_impurity_decrease", type=float, default=DEFAULT_MIN_IMPURITY_DECREASE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--compare", choices=["mab", "exact", "both"], default="both")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
