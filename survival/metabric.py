from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Endpoint = Literal["OS", "RFS"]


@dataclass(frozen=True)
class SurvivalData:
    X: pd.DataFrame  # index: sample_id, columns: features
    y: np.ndarray  # structured array: dtype=[('event', '?'), ('time', '<f8')]


def load_clinical_patient(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if "PATIENT_ID" not in df.columns:
        raise ValueError(f"Expected PATIENT_ID column in {path}")
    df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)
    return df


def _parse_event(series: pd.Series, endpoint: Endpoint) -> pd.Series:
    s = series.astype("string")
    if endpoint == "OS":
        # "1:DECEASED" vs "0:LIVING"
        return s.str.startswith("1:", na=False)
    if endpoint == "RFS":
        # "1:Recurred" vs "0:Not Recurred"
        return s.str.startswith("1:", na=False)
    raise ValueError(f"Unknown endpoint: {endpoint}")


def make_survival_labels(
    clinical_patient: pd.DataFrame,
    endpoint: Endpoint,
    *,
    id_col: str = "PATIENT_ID",
    min_time: float = 1e-3,
) -> pd.DataFrame:
    if endpoint == "OS":
        time_col, status_col = "OS_MONTHS", "OS_STATUS"
    elif endpoint == "RFS":
        time_col, status_col = "RFS_MONTHS", "RFS_STATUS"
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    missing = [c for c in [id_col, time_col, status_col] if c not in clinical_patient]
    if missing:
        raise ValueError(f"Missing columns in clinical table: {missing}")

    out = clinical_patient[[id_col, time_col, status_col]].copy()
    out.rename(columns={id_col: "sample_id"}, inplace=True)
    out["sample_id"] = out["sample_id"].astype(str)
    out["time"] = pd.to_numeric(out[time_col], errors="coerce")
    out["event"] = _parse_event(out[status_col], endpoint=endpoint)

    out = out.drop(columns=[time_col, status_col])
    out = out.dropna(subset=["sample_id", "time"])

    # If status is missing, treat as unknown; drop to avoid inventing labels.
    out = out[out["event"].notna()]

    # Survival toolkits sometimes dislike exactly 0 durations.
    out.loc[out["time"] <= 0, "time"] = float(min_time)
    return out[["sample_id", "time", "event"]]


def load_expression(
    path: str,
    *,
    aggregate_duplicate_genes: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:
    """
    Load cBioPortal-style expression matrix.

    Returns a DataFrame indexed by Hugo symbol with columns = sample IDs.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if not {"Hugo_Symbol", "Entrez_Gene_Id"}.issubset(df.columns):
        raise ValueError("Expected Hugo_Symbol and Entrez_Gene_Id columns.")

    df = df.drop(columns=["Entrez_Gene_Id"]).set_index("Hugo_Symbol")
    df.columns.name = "sample_id"
    df = df.apply(pd.to_numeric, errors="coerce").astype(dtype)

    if aggregate_duplicate_genes and df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    return df


def build_survival_dataset(
    *,
    expr_path: str,
    clinical_patient_path: str,
    endpoint: Endpoint,
    top_k_genes: int | None = 4000,
    seed: int = 42,
) -> SurvivalData:
    """
    Produces:
      - X: samples x genes (DataFrame)
      - y: structured array for scikit-survival
    """
    expr = load_expression(expr_path)  # genes x samples
    clinical = load_clinical_patient(clinical_patient_path)
    labels = make_survival_labels(clinical, endpoint=endpoint)

    sample_ids = [s for s in expr.columns.astype(str) if s in set(labels["sample_id"])]
    if not sample_ids:
        raise ValueError("No overlap between expression columns and clinical sample IDs.")

    X = expr.loc[:, sample_ids].T  # samples x genes
    X.index.name = "sample_id"

    labels = labels.set_index("sample_id").loc[X.index]

    if top_k_genes is not None and top_k_genes < X.shape[1]:
        rng = np.random.default_rng(seed)
        # break ties deterministically 
        jitter = pd.Series(rng.normal(0, 1e-12, size=X.shape[1]), index=X.columns)
        variances = X.var(axis=0, skipna=True) + jitter
        keep = variances.nlargest(top_k_genes).index
        X = X.loc[:, keep]

    y = np.array(
        list(zip(labels["event"].astype(bool).to_numpy(), labels["time"].to_numpy())),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    return SurvivalData(X=X, y=y)

