from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
import torch
from torch import nn

from survival.metabric import load_clinical_patient, load_expression, make_survival_labels


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _to_bool01(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    s = s.replace(
        {
            "YES": "1",
            "NO": "0",
            "POSITIVE": "1",
            "NEGATIVE": "0",
            "POSITVE": "1",  # typo present in METABRIC export
        }
    )
    return pd.to_numeric(s, errors="coerce")


def build_metabric_ihc4_like_features(
    *,
    expr_path: str,
    clinical_patient_path: str,
    endpoint: str = "OS",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Reconstruct the 9-feature clinical+IHC-like METABRIC setup used in DeepSurv repo.

    DeepSurv's provided `metabric_IHC4_clinical_train_test.h5` has 9 inputs that match:
      - EGFR, PGR, ERBB2, MKI67 expression
      - HORMONE_THERAPY, RADIO_THERAPY, CHEMOTHERAPY, ER_IHC (binary)
      - AGE_AT_DIAGNOSIS (continuous)
    """
    endpoint = endpoint.upper()
    if endpoint not in {"OS", "RFS"}:
        raise ValueError("endpoint must be OS or RFS")

    expr = load_expression(expr_path)  # genes x samples
    clinical = load_clinical_patient(clinical_patient_path)
    labels = make_survival_labels(clinical, endpoint=endpoint)  # sample_id, time, event

    # Features
    needed_genes = ["EGFR", "PGR", "ERBB2", "MKI67"]
    missing = [g for g in needed_genes if g not in expr.index]
    if missing:
        raise ValueError(f"Missing genes in expression matrix: {missing}")

    df = labels.set_index("sample_id").copy()
    df = df.loc[df.index.intersection(expr.columns.astype(str))]
    if df.empty:
        raise ValueError("No overlap between clinical labels and expression sample IDs.")

    # Gene expression features
    for g in needed_genes:
        df[g] = expr.loc[g, df.index].astype("float32").to_numpy()

    # Clinical binary features (map to 0/1)
    clinical2 = clinical.set_index("PATIENT_ID")
    for col in ["HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_IHC"]:
        if col not in clinical2.columns:
            raise ValueError(f"Missing clinical column: {col}")
        df[col] = _to_bool01(clinical2.loc[df.index, col])

    # age
    if "AGE_AT_DIAGNOSIS" not in clinical2.columns:
        raise ValueError("Missing AGE_AT_DIAGNOSIS in clinical table.")
    df["AGE_AT_DIAGNOSIS"] = pd.to_numeric(
        clinical2.loc[df.index, "AGE_AT_DIAGNOSIS"], errors="coerce"
    )

    # Drop rows with missing outcomes; keep features with imputation later.
    df = df.dropna(subset=["time", "event"])

    feature_names = (
        needed_genes
        + ["HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY", "ER_IHC"]
        + ["AGE_AT_DIAGNOSIS"]
    )
    X = df[feature_names]
    y = np.array(
        list(zip(df["event"].astype(bool).to_numpy(), df["time"].to_numpy())),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    return X, y


def load_deepsurv_h5(path: str) -> SplitData:
    """
    Load the exact DeepSurv-provided METABRIC split:
    `metabric_IHC4_clinical_train_test.h5` (train/test only, 9 inputs).
    """
    with h5py.File(path, "r") as f:
        X_train = f["train"]["x"][:].astype("float32")
        t_train = f["train"]["t"][:].astype("float32")
        e_train = f["train"]["e"][:].astype("bool")
        X_test = f["test"]["x"][:].astype("float32")
        t_test = f["test"]["t"][:].astype("float32")
        e_test = f["test"]["e"][:].astype("bool")

    # DeepSurv scripts standardize using train mean/std.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train = np.array(list(zip(e_train, t_train)), dtype=[("event", "?"), ("time", "<f8")])
    y_test = np.array(list(zip(e_test, t_test)), dtype=[("event", "?"), ("time", "<f8")])

    # The original DeepSurv METABRIC .h5 provides only train/test.
    # Keep validation equal to train (so code paths stay simple) and do not early-stop.
    X_val, y_val = X_train, y_train

    feature_names = [
        "EGFR",
        "PGR",
        "ERBB2",
        "MKI67",
        "HORMONE_THERAPY",
        "RADIO_THERAPY",
        "CHEMOTHERAPY",
        "ER_IHC",
        "AGE_AT_DIAGNOSIS",
    ]

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )


def split_and_preprocess(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> SplitData:
    X_np = X.to_numpy(dtype=np.float32)
    y_event = y["event"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y, test_size=test_size, random_state=seed, stratify=y_event
    )
    y_train_event = y_train["event"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train_event
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return SplitData(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_val=X_val.astype(np.float32),
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feature_names=list(X.columns),
    )


class DeepSurvNet(nn.Module):
    def __init__(self, n_in: int, hidden_layers: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_in
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (N,)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood.

    Uses a stable logcumsumexp trick by sorting by descending time.
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cum_risk = torch.logcumsumexp(risk, dim=0)
    # sum over events only
    return -(risk[event] - log_cum_risk[event]).mean()


def cindex(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    return float(
        concordance_index_censored(y_true["event"], y_true["time"], risk_scores)[0]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", choices=["OS", "RFS"], default="OS")
    p.add_argument(
        "--deepsurv_h5",
        type=str,
        default=None,
        help="If set, load the exact DeepSurv METABRIC .h5 split (overrides endpoint/features).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.0020065103592061526)
    p.add_argument("--weight_decay", type=float, default=4.6593974609375)
    p.add_argument("--dropout", type=float, default=0.034404296875000004)
    p.add_argument("--hidden", type=str, default="42,42,42")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=50)
    args = p.parse_args()

    if args.deepsurv_h5:
        split = load_deepsurv_h5(args.deepsurv_h5)
    else:
        X, y = build_metabric_ihc4_like_features(
            expr_path="brca_metabric/data_mrna_illumina_microarray.txt",
            clinical_patient_path="brca_metabric/data_clinical_patient.txt",
            endpoint=args.endpoint,
        )
        split = split_and_preprocess(
            X, y, seed=args.seed, test_size=args.test_size, val_size=args.val_size
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = DeepSurvNet(n_in=split.X_train.shape[1], hidden_layers=hidden, dropout=args.dropout).to(device)

    # DeepSurv adds L2 regularization to the loss (not AdamW-style weight decay).
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    X_train = torch.from_numpy(split.X_train).to(device)
    t_train = torch.from_numpy(split.y_train["time"].astype(np.float32)).to(device)
    e_train = torch.from_numpy(split.y_train["event"].astype(bool)).to(device)

    X_val = torch.from_numpy(split.X_val).to(device)
    t_val = torch.from_numpy(split.y_val["time"].astype(np.float32)).to(device)
    e_val = torch.from_numpy(split.y_val["event"].astype(bool)).to(device)

    best_val = -1.0
    best_state = None
    bad = 0
    use_early_stop = not bool(args.deepsurv_h5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        risk = model(X_train)
        loss = cox_ph_loss(risk, t_train, e_train)
        if args.weight_decay and args.weight_decay > 0:
            l2 = torch.zeros((), device=device)
            for p in model.parameters():
                l2 = l2 + (p * p).sum()
            loss = loss + float(args.weight_decay) * l2 / X_train.shape[0]
        loss.backward()
        opt.step()

        if use_early_stop and (epoch % 10 == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                val_risk = model(X_val).detach().cpu().numpy()
            val_c = cindex(split.y_val, val_risk)
            if val_c > best_val + 1e-4:
                best_val = val_c
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= args.patience:
                break

    if use_early_stop and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_risk = model(torch.from_numpy(split.X_test).to(device)).detach().cpu().numpy()
        train_risk = model(torch.from_numpy(split.X_train).to(device)).detach().cpu().numpy()

    print(f"DeepSurv-like (IHC4-ish 9 features) endpoint={args.endpoint}")
    print(f"Train C-index: {cindex(split.y_train, train_risk):.4f}")
    if use_early_stop:
        print(f"Val   C-index: {best_val:.4f}")
    print(f"Test  C-index: {cindex(split.y_test, test_risk):.4f}")
    print("Features:", ", ".join(split.feature_names))


if __name__ == "__main__":
    main()
