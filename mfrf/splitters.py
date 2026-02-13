from __future__ import annotations

import math
from typing import Optional

import numpy as np


def gini_from_counts(counts: np.ndarray) -> np.ndarray:
    n = counts.sum(axis=-1, keepdims=True)
    p = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, n, out=p, where=n > 0)
    return 1.0 - np.sum(p * p, axis=-1)


def hist2d_counts_for_feature(
    x_bins: np.ndarray, y: np.ndarray, n_bins: int, n_classes: int
) -> np.ndarray:
    idx = x_bins.astype(np.int64) * n_classes + y.astype(np.int64)
    out = np.bincount(idx, minlength=n_bins * n_classes)
    return out.reshape(n_bins, n_classes)


def hist3d_counts_for_features(
    Xb: np.ndarray, y: np.ndarray, n_bins: int, n_classes: int
) -> np.ndarray:
    n, m = Xb.shape
    if n == 0 or m == 0:
        return np.zeros((m, n_bins, n_classes), dtype=np.int64)
    offsets = (np.arange(m, dtype=np.int64) * n_bins).reshape(1, m)
    idx = (Xb.astype(np.int64) + offsets) * n_classes + y[:, None].astype(np.int64)
    out = np.bincount(idx.ravel(), minlength=m * n_bins * n_classes)
    return out.reshape(m, n_bins, n_classes)


def best_threshold_from_hist(counts: np.ndarray) -> tuple[int, float]:
    B, _ = counts.shape
    if B < 2:
        return 0, float("inf")
    cum = np.cumsum(counts, axis=0)
    total = cum[-1]
    n_total = float(total.sum())
    if n_total <= 1:
        return 0, float("inf")

    left = cum[:-1]
    right = total[None, :] - left
    nL = left.sum(axis=1)
    nR = n_total - nL
    valid = (nL > 0) & (nR > 0)
    gL = gini_from_counts(left)
    gR = gini_from_counts(right)
    mu = (nL / n_total) * gL + (nR / n_total) * gR
    mu = np.where(valid, mu, np.inf)
    j = int(np.argmin(mu))
    return j, float(mu[j])


def hoeffding_radius(delta: float, n: int) -> float:
    n = max(1, int(n))
    delta = float(np.clip(delta, 1e-12, 0.5))
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


class SplitterBase:
    def __init__(self, n_bins: int):
        self.n_bins = int(n_bins)
        self.insertions_ = 0

    def choose_split(
        self,
        Xb_node: np.ndarray,
        y_node: np.ndarray,
        feature_indices: np.ndarray,
        n_classes: int,
        parent_impurity: float,
        min_impurity_decrease: float,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, int, float]]:
        raise NotImplementedError


class ExactHistogramSplitter(SplitterBase):
    def choose_split(
        self,
        Xb_node,
        y_node,
        feature_indices,
        n_classes,
        parent_impurity,
        min_impurity_decrease,
        rng,
    ):
        n = Xb_node.shape[0]
        best_mu = float("inf")
        best_f = -1
        best_edge = -1
        for f in feature_indices:
            counts = hist2d_counts_for_feature(Xb_node[:, f], y_node, self.n_bins, n_classes)
            self.insertions_ += n
            edge, mu = best_threshold_from_hist(counts)
            if mu < best_mu:
                best_mu = mu
                best_f = int(f)
                best_edge = int(edge)

        if best_f < 0 or not np.isfinite(best_mu):
            return None
        gain = parent_impurity - best_mu
        if gain < min_impurity_decrease:
            return None
        return best_f, best_edge, float(gain)


class MABSplitHistogramSplitter(SplitterBase):
    def __init__(
        self,
        n_bins: int,
        batch_size: int = 512,
        mab_min_samples: int = 1200,
        check_every: int = 4,
        confidence_scale: float = 0.15,
        stop_active_features: int = 3,
        min_batches_before_stop: int = 2,
        consume_all_data: bool = False,
    ):
        super().__init__(n_bins=n_bins)
        self.batch_size = int(batch_size)
        self.mab_min_samples = int(mab_min_samples)
        self.check_every = max(1, int(check_every))
        self.confidence_scale = float(confidence_scale)
        self.stop_active_features = max(1, int(stop_active_features))
        self.min_batches_before_stop = max(1, int(min_batches_before_stop))
        self.consume_all_data = bool(consume_all_data)

    def _exact_fallback(
        self,
        Xb_node: np.ndarray,
        y_node: np.ndarray,
        feature_indices: np.ndarray,
        n_classes: int,
        parent_impurity: float,
        min_impurity_decrease: float,
    ) -> Optional[tuple[int, int, float]]:
        n = Xb_node.shape[0]
        best_mu = float("inf")
        best_f = -1
        best_edge = -1
        for f in feature_indices:
            counts = hist2d_counts_for_feature(Xb_node[:, f], y_node, self.n_bins, n_classes)
            self.insertions_ += n
            edge, mu = best_threshold_from_hist(counts)
            if mu < best_mu:
                best_mu = mu
                best_f = int(f)
                best_edge = int(edge)

        if best_f < 0 or not np.isfinite(best_mu):
            return None
        gain = parent_impurity - best_mu
        if gain < min_impurity_decrease:
            return None
        return best_f, best_edge, float(gain)

    def choose_split(
        self,
        Xb_node,
        y_node,
        feature_indices,
        n_classes,
        parent_impurity,
        min_impurity_decrease,
        rng,
    ):
        n = Xb_node.shape[0]
        m = feature_indices.shape[0]
        if n < 2 or m == 0:
            return None
        if n < self.mab_min_samples:
            return self._exact_fallback(
                Xb_node, y_node, feature_indices, n_classes, parent_impurity, min_impurity_decrease
            )

        B = self.n_bins
        active_idx = np.arange(m, dtype=np.int64)
        counts = np.zeros((m, B, n_classes), dtype=np.int32)
        delta = 1.0 / (max(n, 2) * max(m, 1))
        delta = max(delta, 1e-12)
        # Fix the candidate feature view once; only active feature columns are sliced per batch.
        X_node_features = Xb_node[:, feature_indices]

        perm = rng.permutation(n)
        ptr = 0
        n_used = 0
        steps_since_check = 0
        stop_after = self.min_batches_before_stop * self.batch_size

        def best_per_feature(feat_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Work only on active features to reduce overhead.
            c = counts[feat_idx]
            cum = np.cumsum(c, axis=1)
            total = cum[:, -1, :]
            n_total = total.sum(axis=1).astype(np.float64)
            left = cum[:, :-1, :]
            right = total[:, None, :] - left
            nL = left.sum(axis=2).astype(np.float64)
            nR = n_total[:, None] - nL
            gL = gini_from_counts(left)
            gR = gini_from_counts(right)
            denom = np.maximum(1.0, n_total)[:, None]
            mu = (nL / denom) * gL + (nR / denom) * gR
            mu[(nL <= 0) | (nR <= 0)] = np.inf
            best_edge = np.argmin(mu, axis=1)
            best_mu = mu[np.arange(mu.shape[0]), best_edge]
            return best_mu, best_edge

        while ptr < n and active_idx.size > 0:
            if active_idx.size <= 1:
                break

            batch_n = min(self.batch_size, n - ptr)
            idx = perm[ptr : ptr + batch_n]
            ptr += batch_n
            n_used += batch_n
            steps_since_check += 1

            yb = y_node[idx]
            Xb_batch = X_node_features[idx][:, active_idx]
            c = hist3d_counts_for_features(Xb_batch, yb, B, n_classes)
            counts[active_idx] += c.astype(np.int32, copy=False)
            self.insertions_ += batch_n * active_idx.size

            if steps_since_check < self.check_every:
                continue
            steps_since_check = 0

            mu_feat, _ = best_per_feature(active_idx)
            rad = self.confidence_scale * hoeffding_radius(delta, n_used)
            ucb = mu_feat + rad
            best_ucb = float(np.min(ucb))
            if not np.isfinite(best_ucb):
                continue

            lcb = mu_feat - rad
            eliminate = lcb > best_ucb
            if eliminate.any():
                keep = ~eliminate
                if not np.any(keep):
                    keep[int(np.argmin(mu_feat))] = True
                active_idx = active_idx[keep]

            # Paper-style early stopping: once only a few contenders remain, stop querying.
            if active_idx.size <= self.stop_active_features and n_used >= stop_after:
                break

        if self.consume_all_data and ptr < n and active_idx.size > 0:
            idx = perm[ptr:]
            yb = y_node[idx]
            Xb_batch = X_node_features[idx][:, active_idx]
            c = hist3d_counts_for_features(Xb_batch, yb, B, n_classes)
            counts[active_idx] += c.astype(np.int32, copy=False)
            self.insertions_ += idx.size * active_idx.size

        if active_idx.size == 0:
            return None

        mu_active, edge_active = best_per_feature(active_idx)
        i_best_local = int(np.argmin(mu_active))
        if not np.isfinite(mu_active[i_best_local]):
            return None

        best_global_idx = int(active_idx[i_best_local])
        best_f = int(feature_indices[best_global_idx])
        best_edge = int(edge_active[i_best_local])
        gain = parent_impurity - float(mu_active[i_best_local])
        if gain < min_impurity_decrease:
            return None
        return best_f, best_edge, float(gain)
