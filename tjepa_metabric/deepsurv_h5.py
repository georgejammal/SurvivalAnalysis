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

    # Preserve the binary treatment/IHC columns as categorical 0/1, and
    # standardize only continuous columns (matches the mixed-type setup
    # used by T-JEPA for tabular data: normalize numeric, keep categoricals).
    #
    # Feature order:
    #   0 EGFR (num), 1 PGR (num), 2 ERBB2 (num), 3 MKI67 (num),
    #   4 HORMONE_THERAPY (cat), 5 RADIO_THERAPY (cat),
    #   6 CHEMOTHERAPY (cat), 7 ER_IHC (cat), 8 AGE_AT_DIAGNOSIS (num)
    cat_idx = np.array([4, 5, 6, 7], dtype=int)
    num_idx = np.array([0, 1, 2, 3, 8], dtype=int)

    # Clamp categoricals to {0,1} (they should already be 0/1 in the file).
    X_train[:, cat_idx] = np.clip(np.rint(X_train[:, cat_idx]), 0.0, 1.0)
    X_test[:, cat_idx] = np.clip(np.rint(X_test[:, cat_idx]), 0.0, 1.0)

    mean = X_train[:, num_idx].mean(axis=0, keepdims=True)
    std = X_train[:, num_idx].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train[:, num_idx] = (X_train[:, num_idx] - mean) / std
    X_test[:, num_idx] = (X_test[:, num_idx] - mean) / std

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
