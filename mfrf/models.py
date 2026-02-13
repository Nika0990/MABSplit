from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .splitters import SplitterBase, gini_from_counts


@dataclass
class TreeNode:
    is_leaf: bool
    proba: np.ndarray
    feature: Optional[int] = None
    edge_bin: Optional[int] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class HistogramDecisionTreeClassifier:
    def __init__(
        self,
        splitter: SplitterBase,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int = "sqrt",
        min_impurity_decrease: float = 0.0,
        random_state: int = 0,
    ):
        self.splitter = splitter
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = int(random_state)
        self.rng_: Optional[np.random.Generator] = None
        self.n_classes_: Optional[int] = None
        self.root_: Optional[TreeNode] = None

    def _num_features_to_try(self, d: int) -> int:
        if isinstance(self.max_features, int):
            return max(1, min(d, self.max_features))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(d)))
        if self.max_features == "all":
            return d
        raise ValueError("max_features must be int, 'sqrt', or 'all'")

    def fit(self, Xb: np.ndarray, y: np.ndarray) -> "HistogramDecisionTreeClassifier":
        Xb = np.asarray(Xb, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        self.rng_ = np.random.default_rng(self.random_state)
        self.n_classes_ = int(np.max(y)) + 1
        self.root_ = self._build(Xb, y, depth=0)
        return self

    def _build(self, Xb_node: np.ndarray, y_node: np.ndarray, depth: int) -> TreeNode:
        counts = np.bincount(y_node, minlength=self.n_classes_)
        proba = counts / max(1, counts.sum())
        n = Xb_node.shape[0]
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or np.count_nonzero(counts) <= 1
            or n <= 2 * self.min_samples_leaf
        ):
            return TreeNode(is_leaf=True, proba=proba)

        parent_impurity = float(gini_from_counts(counts))
        d = Xb_node.shape[1]
        m_try = self._num_features_to_try(d)
        feats = self.rng_.choice(d, size=m_try, replace=False)
        split = self.splitter.choose_split(
            Xb_node=Xb_node,
            y_node=y_node,
            feature_indices=feats,
            n_classes=self.n_classes_,
            parent_impurity=parent_impurity,
            min_impurity_decrease=self.min_impurity_decrease,
            rng=self.rng_,
        )
        if split is None:
            return TreeNode(is_leaf=True, proba=proba)

        f, edge, _ = split
        left_mask = Xb_node[:, f] <= edge
        n_left = int(np.count_nonzero(left_mask))
        n_right = int(n - n_left)
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return TreeNode(is_leaf=True, proba=proba)

        left = self._build(Xb_node[left_mask], y_node[left_mask], depth + 1)
        right = self._build(Xb_node[~left_mask], y_node[~left_mask], depth + 1)
        return TreeNode(
            is_leaf=False,
            proba=proba,
            feature=int(f),
            edge_bin=int(edge),
            left=left,
            right=right,
        )

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


class HistogramRandomForestClassifier:
    def __init__(
        self,
        splitter_builder: Callable[[], SplitterBase],
        n_estimators: int = 8,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int = "sqrt",
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        random_state: int = 0,
    ):
        self.splitter_builder = splitter_builder
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)

        self.trees_: list[HistogramDecisionTreeClassifier] = []
        self.insertions_: int = 0
        self.n_classes_: Optional[int] = None

    def fit(self, Xb: np.ndarray, y: np.ndarray) -> "HistogramRandomForestClassifier":
        Xb = np.asarray(Xb, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        self.trees_.clear()
        self.insertions_ = 0
        self.n_classes_ = int(np.max(y)) + 1

        n = Xb.shape[0]
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n, endpoint=False)
            else:
                idx = np.arange(n, dtype=np.int64)

            splitter = self.splitter_builder()
            tree = HistogramDecisionTreeClassifier(
                splitter=splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(Xb[idx], y[idx])
            self.insertions_ += splitter.insertions_
            self.trees_.append(tree)
        return self

    def predict_proba(self, Xb: np.ndarray) -> np.ndarray:
        Xb = np.asarray(Xb, dtype=np.uint8)
        proba_sum = np.zeros((Xb.shape[0], self.n_classes_), dtype=np.float64)
        for tree in self.trees_:
            proba_sum += tree.predict_proba(Xb)
        return proba_sum / max(1, len(self.trees_))

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(Xb), axis=1)
