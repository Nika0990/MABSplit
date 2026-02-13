from __future__ import annotations

from typing import Optional

import numpy as np


class VarianceFeatureSelector:
    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self.mask_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "VarianceFeatureSelector":
        X = np.asarray(X, dtype=np.float32)
        if self.threshold <= 0:
            self.mask_ = np.ones(X.shape[1], dtype=bool)
            return self

        var = X.var(axis=0)
        mask = var > self.threshold
        if not np.any(mask):
            mask = np.ones(X.shape[1], dtype=bool)
        self.mask_ = mask
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X[:, self.mask_]

    @property
    def n_selected_(self) -> int:
        if self.mask_ is None:
            return 0
        return int(np.count_nonzero(self.mask_))


class UniformBinner:
    def __init__(self, n_bins: int):
        self.n_bins = int(n_bins)
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "UniformBinner":
        X = np.asarray(X, dtype=np.float32)
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        span = mx - mn
        scale = np.zeros_like(span, dtype=np.float32)
        np.divide(1.0, span, out=scale, where=span > 0)
        self.min_ = mn
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        z = (X - self.min_) * self.scale_
        z = np.clip(z, 0.0, 1.0)
        return np.floor(z * (self.n_bins - 1 + 1e-9)).astype(np.uint8)
