from __future__ import annotations

import gzip
import os
import struct
from typing import Optional

import numpy as np

AVAILABLE_DATASETS = ("toy", "mnist", "digits", "iris", "wine", "breast_cancer")

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)


def _download(url: str, path: str) -> None:
    from urllib.request import urlopen

    with urlopen(url, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())


def ensure_mnist(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            continue
        last_error: Optional[Exception] = None
        for root in MNIST_URLS:
            try:
                print(f"downloading {filename} from {root}")
                _download(root + filename, path)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"failed to fetch {filename}") from last_error


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic: {magic}")
        data = f.read(n * rows * cols)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows * cols)
    return (arr.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic: {magic}")
        data = f.read(n)
        arr = np.frombuffer(data, dtype=np.uint8)
    return arr.astype(np.int64, copy=False)


def _subsample_xy(
    X: np.ndarray, y: np.ndarray, n_keep: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n_keep <= 0 or n_keep >= X.shape[0]:
        return X, y
    idx = rng.choice(X.shape[0], size=n_keep, replace=False)
    return X[idx], y[idx]


def load_mnist(data_dir: str, n_train: int, n_test: int, seed: int):
    ensure_mnist(data_dir)
    Xtr = _read_idx_images(os.path.join(data_dir, MNIST_FILES["train_images"]))
    ytr = _read_idx_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    Xte = _read_idx_images(os.path.join(data_dir, MNIST_FILES["test_images"]))
    yte = _read_idx_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))

    rng = np.random.default_rng(seed)
    Xtr, ytr = _subsample_xy(Xtr, ytr, n_train, rng)
    Xte, yte = _subsample_xy(Xte, yte, n_test, rng)
    return Xtr, ytr, Xte, yte


def load_sklearn_dataset(
    name: str, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn import datasets as sk_datasets  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for datasets: digits/iris/wine/breast_cancer"
        ) from exc

    if name == "digits":
        data = sk_datasets.load_digits()
    elif name == "iris":
        data = sk_datasets.load_iris()
    elif name == "wine":
        data = sk_datasets.load_wine()
    elif name == "breast_cancer":
        data = sk_datasets.load_breast_cancer()
    else:
        raise ValueError(f"unknown sklearn dataset: {name}")

    X = data.data.astype(np.float32, copy=False)
    y = data.target.astype(np.int64, copy=False)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    rng = np.random.default_rng(seed)
    Xtr, ytr = _subsample_xy(Xtr, ytr, n_train, rng)
    Xte, yte = _subsample_xy(Xte, yte, n_test, rng)
    return Xtr, ytr, Xte, yte


def make_toy_data(
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 2.0, size=(n_classes, n_features)).astype(np.float32)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_classes, size=n, endpoint=False, dtype=np.int64)
        X = centers[y] + rng.normal(0.0, 1.0, size=(n, n_features)).astype(np.float32)
        return X, y

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    return Xtr, ytr, Xte, yte


def load_dataset(
    dataset_name: str,
    data_dir: str,
    n_train: int,
    n_test: int,
    seed: int,
    toy_features: int,
    toy_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if dataset_name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist(
            data_dir=data_dir, n_train=n_train, n_test=n_test, seed=seed
        )
        return Xtr, ytr, Xte, yte, "mnist"

    if dataset_name == "toy":
        Xtr, ytr, Xte, yte = make_toy_data(
            n_train=n_train,
            n_test=n_test,
            n_features=toy_features,
            n_classes=toy_classes,
            seed=seed,
        )
        return Xtr, ytr, Xte, yte, "toy"

    Xtr, ytr, Xte, yte = load_sklearn_dataset(
        name=dataset_name, n_train=n_train, n_test=n_test, seed=seed
    )
    return Xtr, ytr, Xte, yte, dataset_name
