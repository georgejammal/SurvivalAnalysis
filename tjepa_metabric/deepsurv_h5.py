from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True)
class DeepSurvH5Split:
    X_train: np.ndarray  # float32
    t_train: np.ndarray  # float32
    e_train: np.ndarray  # bool
    X_test: np.ndarray  # float32
    t_test: np.ndarray  # float32
    e_test: np.ndarray  # bool


def load_deepsurv_metabric_h5(path: str) -> DeepSurvH5Split:
    """
    Loads the exact DeepSurv METABRIC file:
    - train/test splits only
    - X is shape (N, 9)
    - t is time in months
    - e is event indicator (1=event, 0=censored)
    """
    with h5py.File(path, "r") as f:
        X_train = f["train"]["x"][:].astype("float32")
        t_train = f["train"]["t"][:].astype("float32")
        e_train = f["train"]["e"][:].astype(bool)
        X_test = f["test"]["x"][:].astype("float32")
        t_test = f["test"]["t"][:].astype("float32")
        e_test = f["test"]["e"][:].astype(bool)

    # Match DeepSurv preprocessing: standardize by train mean/std.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # DeepSurv has some exact 0 times; avoid issues in Cox loss.
    eps = np.float32(1e-3)
    t_train = np.maximum(t_train, eps)
    t_test = np.maximum(t_test, eps)

    return DeepSurvH5Split(
        X_train=X_train,
        t_train=t_train,
        e_train=e_train,
        X_test=X_test,
        t_test=t_test,
        e_test=e_test,
    )

