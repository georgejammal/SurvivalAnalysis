from __future__ import annotations

import argparse

import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from survival.metabric import build_survival_dataset


def cindex(y_true, risk_scores: np.ndarray) -> float:
    event = y_true["event"]
    time = y_true["time"]
    return float(concordance_index_censored(event, time, risk_scores)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", choices=["OS", "RFS"], default="OS")
    p.add_argument(
        "--top_k_genes",
        type=int,
        default=4000,
        help="Variance filter for genes (keeps runtime reasonable).",
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json_out", type=str, default=None, help="Optional path to write metrics JSON.")

    # RSF hyperparams (paper-like defaults: many trees, logrank splitting, mtry ~= sqrt(p))
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--min_samples_split", type=int, default=10)
    p.add_argument("--min_samples_leaf", type=int, default=15)
    p.add_argument("--max_features", type=str, default="sqrt")
    p.add_argument("--n_jobs", type=int, default=-1)

    args = p.parse_args()

    data = build_survival_dataset(
        expr_path="brca_metabric/data_mrna_illumina_microarray.txt",
        clinical_patient_path="brca_metabric/data_clinical_patient.txt",
        endpoint=args.endpoint,
        top_k_genes=args.top_k_genes,
        seed=args.seed,
    )

    X = data.X.to_numpy(dtype=np.float32)
    y = data.y

    # Split stratified by event indicator (common practice)
    y_event = y["event"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_event,
    )

    # Handle rare missing expression cells (few NaNs).
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardize features for Cox; for RSF it’s optional but harmless.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # High-dimensional gene expression is ill-conditioned for unpenalized CoxPH.
    # Use elastic-net Cox (ridge by default) as a stable baseline.
    # sksurv enforces l1_ratio in (0, 1]; use a tiny L1 component to approximate ridge.
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=1e-6, alpha_min_ratio=0.01, n_alphas=50)
    coxnet.fit(X_train, y_train)
    coxnet_risk = coxnet.predict(X_test)
    print(f"Coxnet(ridge) C-index ({args.endpoint}): {cindex(y_test, coxnet_risk):.4f}")

    rsf = RandomSurvivalForest(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    rsf.fit(X_train, y_train)

    # Use predicted risk score = -expected survival time proxy via CHF integral:
    # scikit-survival returns survival functions; we approximate risk as negative mean survival time.
    surv_fns = rsf.predict_survival_function(X_test, return_array=True)
    # Integrate S(t) dt over the grid -> mean survival time approximation.
    times = rsf.unique_times_
    mean_surv = np.trapezoid(surv_fns, x=times, axis=1)
    rsf_risk = -mean_surv
    print(f"RSF C-index ({args.endpoint}): {cindex(y_test, rsf_risk):.4f}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "script": "survival.rsf_baseline",
            "argv": sys.argv,
            "endpoint": args.endpoint,
            "dataset": "brca_metabric",
            "n_features": int(X_train.shape[1]),
            "hyperparams": {
                "seed": args.seed,
                "top_k_genes": args.top_k_genes,
                "test_size": args.test_size,
                "rsf": {
                    "n_estimators": args.n_estimators,
                    "min_samples_split": args.min_samples_split,
                    "min_samples_leaf": args.min_samples_leaf,
                    "max_features": args.max_features,
                    "n_jobs": args.n_jobs,
                },
                "coxnet": {"l1_ratio": 1e-6, "alpha_min_ratio": 0.01, "n_alphas": 50},
            },
            "metrics": {
                "c_index_test_coxnet_ridge": cindex(y_test, coxnet_risk),
                "c_index_test_rsf": cindex(y_test, rsf_risk),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()
