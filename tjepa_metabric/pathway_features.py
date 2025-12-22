from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from survival.metabric import load_clinical_patient, load_expression, make_survival_labels
from tjepa_metabric.gene_sets import GeneSetLibrary


@dataclass(frozen=True)
class PathwayDataset:
    X: pd.DataFrame  # samples x features
    time: np.ndarray  # float32
    event: np.ndarray  # bool
    ihc4_feature_names: list[str]
    pathway_feature_names: list[str]


def _zscore_per_gene(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score per gene across samples.
    expr: genes x samples
    """
    mu = expr.mean(axis=1, skipna=True)
    sd = expr.std(axis=1, skipna=True).replace(0, np.nan)
    z = expr.sub(mu, axis=0).div(sd, axis=0)
    return z.astype("float32")


def build_ihc4_c_features(
    *,
    expr: pd.DataFrame,  # genes x samples
    clinical_patient: pd.DataFrame,
    sample_ids: list[str],
) -> pd.DataFrame:
    needed_genes = ["EGFR", "PGR", "ERBB2", "MKI67"]
    missing = [g for g in needed_genes if g not in expr.index]
    if missing:
        raise ValueError(f"Missing genes in expression: {missing}")

    # Align clinical
    clinical_patient = clinical_patient.copy()
    clinical_patient["PATIENT_ID"] = clinical_patient["PATIENT_ID"].astype(str)
    clinical_patient = clinical_patient.set_index("PATIENT_ID").loc[sample_ids]

    def to_bool01(series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip().str.upper()
        s = s.replace(
            {"YES": "1", "NO": "0", "POSITIVE": "1", "NEGATIVE": "0", "POSITVE": "1"}
        )
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    # gene expression (raw)
    for g in needed_genes:
        out[g] = expr.loc[g, sample_ids].to_numpy(dtype="float32")

    for col in ["HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_IHC"]:
        out[col] = to_bool01(clinical_patient[col]).astype("float32")

    out["AGE_AT_DIAGNOSIS"] = pd.to_numeric(
        clinical_patient["AGE_AT_DIAGNOSIS"], errors="coerce"
    ).astype("float32")
    return out


def build_pathway_dataset(
    *,
    expr_path: str = "brca_metabric/data_mrna_illumina_microarray.txt",
    clinical_patient_path: str = "brca_metabric/data_clinical_patient.txt",
    endpoint: str = "OS",
    gene_sets: GeneSetLibrary,
    min_genes_overlap: int = 5,
    max_pathways: int = 1000,
    seed: int = 42,
) -> PathwayDataset:
    """
    Builds samples x (pathway_scores + ihc4+c) with survival labels.
    """
    endpoint = endpoint.upper()
    expr = load_expression(expr_path)  # genes x samples (float, aggregated)
    clinical = load_clinical_patient(clinical_patient_path)
    labels = make_survival_labels(clinical, endpoint=endpoint)

    sample_ids = [s for s in expr.columns.astype(str) if s in set(labels["sample_id"])]
    if not sample_ids:
        raise ValueError("No overlap between expression and labels.")

    labels = labels.set_index("sample_id").loc[sample_ids]
    time = labels["time"].to_numpy(dtype="float32")
    event = labels["event"].to_numpy(dtype=bool)

    # Z-score genes across samples for pathway scoring
    z = _zscore_per_gene(expr.loc[:, sample_ids])

    # Score pathways
    rng = np.random.default_rng(seed)
    pathway_terms = list(gene_sets.gene_sets.keys())
    rng.shuffle(pathway_terms)

    scores: dict[str, np.ndarray] = {}
    for term in pathway_terms:
        genes = gene_sets.gene_sets[term]
        genes_present = [g for g in genes if g in z.index]
        if len(genes_present) < min_genes_overlap:
            continue
        # mean z-score across genes in set => (samples,)
        v = z.loc[genes_present].mean(axis=0, skipna=True).to_numpy(dtype="float32")
        scores[term] = v
        if len(scores) >= max_pathways:
            break

    if not scores:
        raise ValueError("No pathway scores computed; check gene symbols / gene sets.")

    X_path = pd.DataFrame(scores, index=pd.Index(sample_ids, name="sample_id"))

    # Append IHC4+C raw features to bias representation toward the DeepSurv setup
    X_ihc4 = build_ihc4_c_features(
        expr=expr, clinical_patient=clinical, sample_ids=sample_ids
    )

    # Impute minimal missing values
    X = pd.concat([X_path, X_ihc4], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(axis=0, skipna=True))

    return PathwayDataset(
        X=X,
        time=time,
        event=event,
        ihc4_feature_names=list(X_ihc4.columns),
        pathway_feature_names=list(X_path.columns),
    )

