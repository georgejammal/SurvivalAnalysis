from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from survival.metabric import build_survival_dataset


@dataclass(frozen=True)
class PCASurvivalSplit:
    X_train: np.ndarray  # float32, (N, pca_dim)
    X_val: np.ndarray  # float32, (N, pca_dim)
    X_test: np.ndarray  # float32, (N, pca_dim)
    t_train: np.ndarray  # float32, (N,)
    t_val: np.ndarray  # float32, (N,)
    t_test: np.ndarray  # float32, (N,)
    e_train: np.ndarray  # bool, (N,)
    e_val: np.ndarray  # bool, (N,)
    e_test: np.ndarray  # bool, (N,)
    feature_names: list[str]  # gene names after top-k selection
    imputer_strategy: str
    imputer_statistics: np.ndarray  # float64, (D,)
    scaler_mean: np.ndarray  # float64, (D,)
    scaler_scale: np.ndarray  # float64, (D,)
    pca_components: np.ndarray  # float64, (pca_dim, D)
    pca_mean: np.ndarray  # float64, (D,)


def build_metabric_expression_pca_split(
    *,
    endpoint: str = "OS",
    expr_path: str = "brca_metabric/data_mrna_illumina_microarray.txt",
    clinical_patient_path: str = "brca_metabric/data_clinical_patient.txt",
    top_k_genes: int = 4000,
    pca_dim: int = 256,
    impute_strategy: str = "mean",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> PCASurvivalSplit:
    """
    Builds a split of METABRIC expression features, then applies:
      - StandardScaler (fit on train only)
      - PCA (fit on train only)

    This avoids network access and lets you run JEPA on continuous PCA components.
    """
    endpoint = endpoint.upper()
    ds = build_survival_dataset(
        expr_path=expr_path,
        clinical_patient_path=clinical_patient_path,
        endpoint=endpoint,  # type: ignore[arg-type]
        top_k_genes=int(top_k_genes),
        seed=int(seed),
    )

    X = ds.X.to_numpy(dtype=np.float32)
    t = ds.y["time"].astype(np.float32)
    e = ds.y["event"].astype(bool)
    feature_names = list(ds.X.columns)

    X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
        X,
        t,
        e,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=e.astype(int),
    )
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X_train,
        t_train,
        e_train,
        test_size=float(val_size),
        random_state=int(seed),
        stratify=e_train.astype(int),
    )

    imputer = SimpleImputer(strategy=str(impute_strategy))
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_val_s = scaler.transform(X_val_i)
    X_test_s = scaler.transform(X_test_i)

    pca = PCA(n_components=int(pca_dim), random_state=int(seed))
    X_train_p = pca.fit_transform(X_train_s).astype(np.float32)
    X_val_p = pca.transform(X_val_s).astype(np.float32)
    X_test_p = pca.transform(X_test_s).astype(np.float32)

    return PCASurvivalSplit(
        X_train=X_train_p,
        X_val=X_val_p,
        X_test=X_test_p,
        t_train=t_train,
        t_val=t_val,
        t_test=t_test,
        e_train=e_train,
        e_val=e_val,
        e_test=e_test,
        feature_names=feature_names,
        imputer_strategy=str(impute_strategy),
        imputer_statistics=imputer.statistics_.copy(),
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        pca_components=pca.components_.copy(),
        pca_mean=pca.mean_.copy(),
    )
