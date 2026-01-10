
"""
mabsplit_rf_mnist_fast.py

Goal
-----
A *from-scratch* Random Forest classifier where the ONLY difference between
"baseline" and "MABSplit" is the split-selection routine.

Why your earlier MNIST run looked "wrong"
-----------------------------------------
MABSplit improves *sample complexity* (number of histogram insertions / queries).
In pure Python, if you run MABSplit at *every* node (including tiny nodes deep in the tree),
the overhead can dominate and wall-clock can be slower—even if insertions drop.

This file fixes that in the same spirit as the authors’ implementation:
- Histogram-based thresholds (fixed number of bins).
- Batch sampling + successive elimination (MABSplit).
- **Hybrid policy**: use MABSplit only when the node has enough samples;
  otherwise fall back to the exact histogram scan (cheap for small nodes).

This matches the practical trick used in many “theory faster” methods:
use them only where they pay off.

Usage
-----
python mabsplit_rf_mnist_fast.py --dataset mnist --n_train 42000 --n_test 10000

If MNIST download fails (no internet), it will fall back to sklearn DIGITS automatically.
"""

from __future__ import annotations
import argparse
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


# -----------------------------
# Metrics
# -----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# -----------------------------
# Histogram utilities (vectorized)
# -----------------------------
def gini_from_counts(counts: np.ndarray) -> np.ndarray:
    """
    counts: (..., K) nonnegative
    returns: (...) gini
    """
    n = counts.sum(axis=-1, keepdims=True)
    # safe division
    p = np.where(n > 0, counts / n, 0.0)
    return 1.0 - np.sum(p * p, axis=-1)


def best_threshold_from_hist(counts: np.ndarray) -> Tuple[int, float]:
    """
    counts: (B, K) counts per bin
    returns: (best_bin_edge_index, best_weighted_gini) where edge index in [0..B-2]
    Split at edge j => left bins [0..j], right bins [j+1..B-1]
    """
    B, K = counts.shape
    assert B >= 2
    cum = np.cumsum(counts, axis=0)               # (B, K)
    total = cum[-1]                               # (K,)
    n_total = total.sum()
    if n_total <= 1:
        return 0, float("inf")

    left = cum[:-1]                               # (B-1, K)
    right = total[None, :] - left                 # (B-1, K)
    nL = left.sum(axis=1)                         # (B-1,)
    nR = n_total - nL
    valid = (nL > 0) & (nR > 0)

    gL = gini_from_counts(left)                   # (B-1,)
    gR = gini_from_counts(right)                  # (B-1,)
    mu = (nL / n_total) * gL + (nR / n_total) * gR
    mu = np.where(valid, mu, np.inf)

    j = int(np.argmin(mu))
    return j, float(mu[j])


def hist2d_counts_for_feature(x_bins: np.ndarray, y: np.ndarray, n_bins: int, n_classes: int) -> np.ndarray:
    """
    Fast (n_bins x n_classes) counts for one feature using a single bincount.
    x_bins: (n,) uint16/int in [0, n_bins-1]
    y:      (n,) int in [0, n_classes-1]
    """
    idx = x_bins.astype(np.int64) * n_classes + y.astype(np.int64)
    out = np.bincount(idx, minlength=n_bins * n_classes)
    return out.reshape(n_bins, n_classes)


def hist3d_counts_for_features(Xb: np.ndarray, y: np.ndarray, n_bins: int, n_classes: int) -> np.ndarray:
    """
    Fast (m, n_bins, n_classes) counts for multiple features using a single bincount.
    Xb: (n, m) uint16/int in [0, n_bins-1]
    y:  (n,) int in [0, n_classes-1]
    """
    n, m = Xb.shape
    if n == 0 or m == 0:
        return np.zeros((m, n_bins, n_classes), dtype=np.int64)
    f_offsets = (np.arange(m, dtype=np.int64) * n_bins).reshape(1, m)
    idx = (Xb.astype(np.int64) + f_offsets) * n_classes + y[:, None].astype(np.int64)
    out = np.bincount(idx.ravel(), minlength=m * n_bins * n_classes)
    return out.reshape(m, n_bins, n_classes)


def hoeffding_radius(delta: float, n: int) -> float:
    """
    mu_hat in [0,1] => Hoeffding radius for mean estimate.
    """
    n = max(1, int(n))
    delta = min(max(delta, 1e-12), 0.5)
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# -----------------------------
# Splitter API
# -----------------------------
class SplitterBase:
    def __init__(self, n_bins: int = 64):
        self.n_bins = int(n_bins)
        self.insertions_ = 0  # instrumentation: how many (feature,value,label) "updates" were processed

    def choose_split(
        self,
        Xb_node: np.ndarray,         # (n, d) pre-binned to [0..n_bins-1]
        y_node: np.ndarray,          # (n,)
        feature_indices: np.ndarray, # (m_try,)
        n_classes: int,
        rng: np.random.Generator,
        parent_impurity: float,
        min_impurity_decrease: float,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Return (best_feature, best_edge_bin, gain) or None.

        edge_bin is in [0..n_bins-2] (split: x_bin <= edge_bin vs > edge_bin)
        """
        raise NotImplementedError


# -----------------------------
# Baseline: exact histogram scan (vectorized per feature)
# -----------------------------
class ExactHistogramSplitter(SplitterBase):
    def choose_split(self, Xb_node, y_node, feature_indices, n_classes, rng, parent_impurity, min_impurity_decrease):
        n = Xb_node.shape[0]
        best_mu = float("inf")
        best_f = None
        best_edge = None

        for f in feature_indices:
            counts = hist2d_counts_for_feature(Xb_node[:, f], y_node, self.n_bins, n_classes)
            self.insertions_ += n
            edge, mu = best_threshold_from_hist(counts)
            if mu < best_mu:
                best_mu = mu
                best_f = int(f)
                best_edge = int(edge)

        if best_f is None or not np.isfinite(best_mu):
            return None

        gain = parent_impurity - best_mu
        if gain < min_impurity_decrease:
            return None
        return best_f, best_edge, float(gain)


# -----------------------------
# MABSplit: batched sampling + successive elimination (vectorized)
# Hybrid: only use MABSplit if node has >= mab_min_samples, else exact scan.
# -----------------------------
class MABSplitHistogramSplitter(SplitterBase):
    def __init__(
        self,
        n_bins: int = 64,
        batch_size: int = 512,
        mab_min_samples: int = 5000,
        check_every: int = 1,
    ):
        super().__init__(n_bins=n_bins)
        self.batch_size = int(batch_size)
        self.mab_min_samples = int(mab_min_samples)
        self.check_every = max(1, int(check_every))

    def choose_split(self, Xb_node, y_node, feature_indices, n_classes, rng, parent_impurity, min_impurity_decrease):
        n = Xb_node.shape[0]
        m = feature_indices.shape[0]
        if n < 2 or m == 0:
            return None

        # For small nodes, exact is cheaper and usually faster in Python
        if n < self.mab_min_samples:
            # Inline exact scan so insertions are counted on *this* splitter
            best_mu = float("inf")
            best_f = None
            best_edge = None
            for f in feature_indices:
                c = hist2d_counts_for_feature(Xb_node[:, f], y_node, self.n_bins, n_classes)
                self.insertions_ += n
                edge, mu = best_threshold_from_hist(c)
                if mu < best_mu:
                    best_mu = mu
                    best_f = int(f)
                    best_edge = int(edge)
            if best_f is None or not np.isfinite(best_mu):
                return None
            gain = parent_impurity - best_mu
            if gain < min_impurity_decrease:
                return None
            return best_f, best_edge, float(gain)

        B = self.n_bins
        n_edges = B - 1

        # Per-node delta (same “make it tiny” structure as in paper)
        delta = 1.0 / (max(n, 2) ** 2 * max(m, 1) * max(n_edges, 1))
        delta = max(delta, 1e-12)

        # Active mask per feature per edge
        active = np.ones((m, n_edges), dtype=bool)
        feature_active = np.ones(m, dtype=bool)

        # Hist counts per feature: (m, B, K) maintained incrementally
        counts = np.zeros((m, B, n_classes), dtype=np.int32)

        perm = rng.permutation(n)
        ptr = 0
        n_used = 0

        # Helper to compute mu_hat for ALL (m, n_edges) from counts
        def compute_mu_all() -> np.ndarray:
            # counts: (m, B, K)
            cum = np.cumsum(counts, axis=1)              # (m, B, K)
            total = cum[:, -1, :]                        # (m, K)
            n_total = total.sum(axis=1)                  # (m,)

            left = cum[:, :-1, :]                        # (m, B-1, K)
            right = total[:, None, :] - left             # (m, B-1, K)
            nL = left.sum(axis=2)                        # (m, B-1)
            nR = n_total[:, None] - nL

            gL = gini_from_counts(left)                  # (m, B-1)
            gR = gini_from_counts(right)                 # (m, B-1)

            # Avoid division by zero
            denom = np.maximum(1.0, n_total)[:, None]
            mu = (nL / denom) * gL + (nR / denom) * gR

            invalid = (nL <= 0) | (nR <= 0)
            mu[invalid] = np.inf
            return mu

        batches_since_check = 0

        while ptr < n:
            # stop if only one arm remains
            if active.sum() <= 1:
                break

            b = min(self.batch_size, n - ptr)
            idx = perm[ptr:ptr + b]
            ptr += b
            n_used += b
            batches_since_check += 1

            yb = y_node[idx]
            Xb_batch = Xb_node[idx][:, feature_indices]  # (b, m)

            # Update counts for active features in one bincount to reduce Python overhead.
            if feature_active.any():
                feat_idx = np.flatnonzero(feature_active)
                c = hist3d_counts_for_features(Xb_batch[:, feat_idx], yb, B, n_classes)
                counts[feat_idx] += c.astype(np.int32, copy=False)
                self.insertions_ += b * feat_idx.size

            if batches_since_check < self.check_every:
                continue
            batches_since_check = 0

            mu = compute_mu_all()  # (m, B-1)

            # Confidence radius for mu_hat (Hoeffding, cheap)
            rad = hoeffding_radius(delta, n_used)

            # best UCB among active arms
            ucb = mu + rad
            ucb[~active] = np.inf
            best_ucb = float(np.min(ucb))
            if not np.isfinite(best_ucb):
                # not enough info yet
                continue

            lcb = mu - rad
            # Eliminate arms whose LCB is worse than best UCB
            eliminate = (lcb > best_ucb) & active
            if eliminate.any():
                active[eliminate] = False
                feature_active = active.any(axis=1)

            # Optional: if we’re already confident no split improves impurity, stop early
            # (gain = parent - mu). If even best UCB is not better enough, we can stop.
            # We need mu < parent - min_impurity_decrease  => mu <= parent - min_impurity_decrease
            if best_ucb >= (parent_impurity - min_impurity_decrease):
                # Might still improve later as n_used increases, so don't hard-stop here.
                # Keep it conservative.
                pass

        # Finish inserting remaining points (without replacement) for features that still have active arms.
        if ptr < n and active.any():
            idx_rem = perm[ptr:]
            y_rem = y_node[idx_rem]
            Xb_rem = Xb_node[idx_rem][:, feature_indices]  # (n_rem, m)
            n_rem = Xb_rem.shape[0]
            if feature_active.any():
                feat_idx = np.flatnonzero(feature_active)
                c = hist3d_counts_for_features(Xb_rem[:, feat_idx], y_rem, B, n_classes)
                counts[feat_idx] += c.astype(np.int32, copy=False)
                self.insertions_ += n_rem * feat_idx.size

        # Now choose best remaining arm exactly using the full histograms (no more insertions).
        mu_full = compute_mu_all()      # (m, B-1) from full counts
        mu_full[~active] = np.inf
        flat_idx = int(np.argmin(mu_full))
        if not np.isfinite(mu_full.flat[flat_idx]):
            return None
        i_best, edge_best = np.unravel_index(flat_idx, mu_full.shape)
        best_f = int(feature_indices[int(i_best)])
        best_edge = int(edge_best)
        best_mu = float(mu_full[i_best, edge_best])

        gain = parent_impurity - best_mu
        if gain < min_impurity_decrease:
            return None
        return best_f, best_edge, float(gain)


# -----------------------------
# Decision Tree (classification)
# -----------------------------
@dataclass
class TreeNode:
    is_leaf: bool
    proba: Optional[np.ndarray] = None
    feature: Optional[int] = None
    edge_bin: Optional[int] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class HistogramDecisionTreeClassifier:
    def __init__(
        self,
        splitter: SplitterBase,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: str | int | float = "sqrt",
        min_impurity_decrease: float = 0.005,
        random_state: int = 0,
    ):
        self.splitter = splitter
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = int(random_state)

        self.n_classes_: Optional[int] = None
        self.root_: Optional[TreeNode] = None
        self.rng_: Optional[np.random.Generator] = None

    def _n_features_to_consider(self, n_features: int) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(math.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(math.log2(n_features)))
            if self.max_features == "all":
                return n_features
            raise ValueError("Unknown max_features string")
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        if isinstance(self.max_features, float):
            return max(1, min(n_features, int(self.max_features * n_features)))
        raise ValueError("Invalid max_features type")

    def fit(self, Xb: np.ndarray, y: np.ndarray):
        Xb = np.asarray(Xb, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        self.rng_ = np.random.default_rng(self.random_state)
        self.n_classes_ = int(np.max(y)) + 1
        self.root_ = self._build_node(Xb, y, depth=0)
        return self

    def _build_node(self, Xb_node: np.ndarray, y_node: np.ndarray, depth: int) -> TreeNode:
        n = Xb_node.shape[0]
        counts = np.bincount(y_node, minlength=self.n_classes_)
        proba = counts / max(1, counts.sum())

        if depth >= self.max_depth or n < self.min_samples_split or np.count_nonzero(counts) <= 1:
            return TreeNode(is_leaf=True, proba=proba)

        parent_impurity = float(gini_from_counts(counts))

        d = Xb_node.shape[1]
        m_try = self._n_features_to_consider(d)
        feats = self.rng_.choice(d, size=m_try, replace=False)

        split = self.splitter.choose_split(
            Xb_node=Xb_node,
            y_node=y_node,
            feature_indices=feats,
            n_classes=self.n_classes_,
            rng=self.rng_,
            parent_impurity=parent_impurity,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        if split is None:
            return TreeNode(is_leaf=True, proba=proba)

        f, edge, _gain = split
        maskL = Xb_node[:, f] <= edge
        if maskL.sum() == 0 or maskL.sum() == n:
            return TreeNode(is_leaf=True, proba=proba)

        left = self._build_node(Xb_node[maskL], y_node[maskL], depth + 1)
        right = self._build_node(Xb_node[~maskL], y_node[~maskL], depth + 1)
        return TreeNode(is_leaf=False, feature=f, edge_bin=edge, left=left, right=right)

    def predict_proba(self, Xb: np.ndarray) -> np.ndarray:
        Xb = np.asarray(Xb, dtype=np.uint8)
        out = np.zeros((Xb.shape[0], self.n_classes_), dtype=np.float64)
        for i in range(Xb.shape[0]):
            node = self.root_
            while not node.is_leaf:
                if Xb[i, node.feature] <= node.edge_bin:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.proba
        return out

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(Xb), axis=1)


# -----------------------------
# Random Forest (classification)
# -----------------------------
class HistogramRandomForestClassifier:
    def __init__(
        self,
        splitter: SplitterBase,
        n_estimators: int = 5,
        max_depth: int = 5,
        max_features: str | int | float = "sqrt",
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.005,
        bootstrap: bool = True,
        random_state: int = 0,
    ):
        self.splitter = splitter
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)

        self.trees_: List[HistogramDecisionTreeClassifier] = []
        self.n_classes_: Optional[int] = None

    def fit(self, Xb: np.ndarray, y: np.ndarray):
        Xb = np.asarray(Xb, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        self.n_classes_ = int(np.max(y)) + 1
        self.trees_ = []

        n = Xb.shape[0]
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n, endpoint=False)
            else:
                idx = np.arange(n)

            tree = HistogramDecisionTreeClassifier(
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(Xb[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, Xb: np.ndarray) -> np.ndarray:
        Xb = np.asarray(Xb, dtype=np.uint8)
        proba_sum = None
        for tree in self.trees_:
            p = tree.predict_proba(Xb)
            proba_sum = p if proba_sum is None else proba_sum + p
        return proba_sum / max(1, len(self.trees_))

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(Xb), axis=1)


# -----------------------------
# Data: MNIST loader (with fallback)
# -----------------------------
def load_mnist_or_digits(n_train: int, n_test: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        # normalize to [0,1]
        X = X / 255.0
        # fixed split used commonly: first 60k train, last 10k test
        Xtr, Xte = X[:60000], X[60000:]
        ytr, yte = y[:60000], y[60000:]
        # optionally subsample
        if n_train < len(Xtr):
            idx = rng.choice(len(Xtr), size=n_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        if n_test < len(Xte):
            idx = rng.choice(len(Xte), size=n_test, replace=False)
            Xte, yte = Xte[idx], yte[idx]
        return Xtr, ytr, Xte, yte, "mnist"
    except Exception as e:
        # Offline fallback
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        X = data.data.astype(np.float32) / 16.0
        y = data.target.astype(np.int64)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        return Xtr, ytr, Xte, yte, f"digits(fallback: {type(e).__name__})"


def bin_features(X: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Uniform binning into [0..n_bins-1].
    Assumes X is already normalized to [0,1].
    """
    X = np.clip(X, 0.0, 1.0)
    Xb = np.floor(X * (n_bins - 1 + 1e-9)).astype(np.uint8)
    return Xb


# -----------------------------
# Main experiment runner
# -----------------------------
def run(args):
    Xtr, ytr, Xte, yte, dsname = load_mnist_or_digits(args.n_train, args.n_test, seed=args.seed)
    Xb_tr = bin_features(Xtr, args.n_bins)
    Xb_te = bin_features(Xte, args.n_bins)

    print(f"dataset: {dsname}")
    print(f"train: {Xb_tr.shape}, test: {Xb_te.shape}, n_bins={args.n_bins}")

    # Baseline
    exact_splitter = ExactHistogramSplitter(n_bins=args.n_bins)
    rf_exact = HistogramRandomForestClassifier(
        splitter=exact_splitter,
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        min_impurity_decrease=args.min_impurity_decrease,
        random_state=args.seed,
    )
    t0 = time.time()
    rf_exact.fit(Xb_tr, ytr)
    t1 = time.time()
    pred = rf_exact.predict(Xb_te)
    acc = accuracy(yte, pred)

    print("\n=== Exact Histogram RF ===")
    print(f"train_time_sec: {t1 - t0:.3f}")
    print(f"insertions: {exact_splitter.insertions_}")
    print(f"test_acc: {acc:.4f}")

    # MABSplit
    mab_splitter = MABSplitHistogramSplitter(
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        mab_min_samples=args.mab_min_samples,
        check_every=args.check_every,
    )
    rf_mab = HistogramRandomForestClassifier(
        splitter=mab_splitter,
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        min_impurity_decrease=args.min_impurity_decrease,
        random_state=args.seed,
    )
    t0 = time.time()
    rf_mab.fit(Xb_tr, ytr)
    t1 = time.time()
    pred = rf_mab.predict(Xb_te)
    acc = accuracy(yte, pred)

    print("\n=== MABSplit (hybrid) Histogram RF ===")
    print(f"train_time_sec: {t1 - t0:.3f}")
    print(f"insertions: {mab_splitter.insertions_}")
    print(f"test_acc: {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "digits"])
    p.add_argument("--n_train", type=int, default=42000)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--n_trees", type=int, default=5)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--min_impurity_decrease", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--mab_min_samples", type=int, default=5000)
    p.add_argument("--check_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
