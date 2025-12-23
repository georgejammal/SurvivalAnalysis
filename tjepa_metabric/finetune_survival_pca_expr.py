from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from tjepa_metabric.models import FeatureTokenizer, ProjectionLayer, SurvivalMLP, TokenEncoder, cox_ph_loss
from tjepa_metabric.pca_expr import build_metabric_expression_pca_split


def _ensure_tmpdir():
    tmp = os.path.join(os.getcwd(), ".tmp")
    os.makedirs(tmp, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmp)
    tempfile.tempdir = tmp
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")


def cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    y = np.array(list(zip(event.astype(bool), time)), dtype=[("event", "?"), ("time", "<f8")])
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def main():
    _ensure_tmpdir()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to JEPA PCA-expr checkpoint.")
    p.add_argument("--json_out", type=str, default=None)
    p.add_argument("--projection", choices=["linear_per_feature", "linear_flatten", "mean_pool", "max_pool"], default="linear_per_feature")
    p.add_argument("--projection_out_dim", type=int, default=None)
    p.add_argument("--head_hidden_dim", type=int, default=256)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    endpoint = str(ckpt.get("endpoint", "OS"))
    expr_path = str(ckpt.get("expr_path", "brca_metabric/data_mrna_illumina_microarray.txt"))
    clinical_patient_path = str(ckpt.get("clinical_patient_path", "brca_metabric/data_clinical_patient.txt"))
    top_k_genes = int(ckpt.get("top_k_genes", 4000))
    pca_dim = int(ckpt.get("pca_dim", 256))
    imputer_strategy = str(ckpt.get("imputer_strategy", "mean"))

    split = build_metabric_expression_pca_split(
        endpoint=endpoint,
        expr_path=expr_path,
        clinical_patient_path=clinical_patient_path,
        top_k_genes=top_k_genes,
        pca_dim=pca_dim,
        impute_strategy=imputer_strategy,
        seed=args.seed,
    )

    X_train, X_val, X_test = split.X_train, split.X_val, split.X_test
    t_train, t_val, t_test = split.t_train, split.t_val, split.t_test
    e_train, e_val, e_test = split.e_train, split.e_val, split.e_test

    n_features = int(X_train.shape[1])
    num_feature_idx = np.arange(n_features, dtype=int)
    cat_feature_idx = np.array([], dtype=int)
    cat_cardinalities: list[int] = []

    token_dim = int(ckpt.get("token_dim", 32))
    token_mlp_layers = int(ckpt.get("mlp_layers", 4))
    use_type = bool(ckpt.get("use_feature_type_embedding", True))
    use_index = bool(ckpt.get("use_feature_index_embedding", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = FeatureTokenizer(
        n_features=n_features,
        num_feature_idx=num_feature_idx,
        cat_feature_idx=cat_feature_idx,
        cat_cardinalities=cat_cardinalities,
        token_dim=token_dim,
        use_feature_type_embedding=use_type,
        use_feature_index_embedding=use_index,
    )
    encoder = TokenEncoder(
        tokenizer=tokenizer,
        token_dim=token_dim,
        n_reg_tokens=0,
        mlp_layers=token_mlp_layers,
        dropout=0.1,
    ).to(device)

    projection = ProjectionLayer(
        n_features=n_features,
        token_dim=token_dim,
        mode=args.projection,
        out_dim=args.projection_out_dim,
    ).to(device)

    head = SurvivalMLP(
        in_dim=projection.out_dim,
        hidden_dim=args.head_hidden_dim,
        n_hidden=args.head_layers,
        dropout=args.head_dropout,
    ).to(device)

    try:
        encoder.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded encoder: {args.ckpt}")
    except Exception as e:
        print(f"WARNING: failed to load encoder weights ({args.ckpt}); training from scratch.")
        print(f"  error: {e}")

    if args.freeze_encoder:
        for p_ in encoder.parameters():
            p_.requires_grad = False

    opt = torch.optim.Adam(
        list(([] if args.freeze_encoder else encoder.parameters()))
        + list(projection.parameters())
        + list(head.parameters()),
        lr=args.lr,
    )

    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    ttr = torch.from_numpy(t_train).to(device)
    etr = torch.from_numpy(e_train).to(device)

    best_val = -1.0
    best_state = None
    bad = 0
    patience = 30

    def forward_risk(x_t: torch.Tensor) -> torch.Tensor:
        tokens = encoder(x_t, use_reg_tokens=False)
        z = projection(tokens)
        return head(z)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projection.train()
        head.train()
        opt.zero_grad(set_to_none=True)

        risk = forward_risk(X_train_t)
        loss = cox_ph_loss(risk, ttr, etr)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 10 == 0:
            encoder.eval()
            projection.eval()
            head.eval()
            with torch.no_grad():
                val_risk = forward_risk(X_val_t).detach().cpu().numpy()
            val_ci = cindex(e_val, t_val, val_risk)
            if val_ci > best_val + 1e-4:
                best_val = val_ci
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "projection": {k: v.detach().cpu().clone() for k, v in projection.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
                bad = 0
            else:
                bad += 1
            print(f"epoch {epoch:03d} loss={float(loss.detach().cpu()):.4f} val_cindex={val_ci:.4f}")
            if bad >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        projection.load_state_dict(best_state["projection"])
        head.load_state_dict(best_state["head"])

    encoder.eval()
    projection.eval()
    head.eval()
    with torch.no_grad():
        test_risk = forward_risk(X_test_t).detach().cpu().numpy()
        train_risk = forward_risk(X_train_t).detach().cpu().numpy()

    metrics = {
        "c_index_train": cindex(e_train, t_train, train_risk),
        "c_index_val": best_val,
        "c_index_test": cindex(e_test, t_test, test_risk),
    }

    print("T-JEPA finetune (METABRIC expression PCA)")
    print(f"Train C-index: {metrics['c_index_train']:.4f}")
    print(f"Val   C-index: {metrics['c_index_val']:.4f}")
    print(f"Test  C-index: {metrics['c_index_test']:.4f}")
    print(f"n_features={n_features} (pca_dim={pca_dim}, top_k_genes={top_k_genes})")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "script": "tjepa_metabric.finetune_survival_pca_expr",
            "argv": sys.argv,
            "dataset": "metabric_expr_pca",
            "endpoint": endpoint,
            "n_features": int(n_features),
            "top_k_genes": int(top_k_genes),
            "pca_dim": int(pca_dim),
            "hyperparams": {
                "seed": args.seed,
                "epochs": args.epochs,
                "lr": args.lr,
                "token_dim": token_dim,
                "token_mlp_layers": token_mlp_layers,
                "projection": args.projection,
                "projection_out_dim": args.projection_out_dim,
                "head_hidden_dim": args.head_hidden_dim,
                "head_layers": args.head_layers,
                "head_dropout": args.head_dropout,
                "freeze_encoder": bool(args.freeze_encoder),
                "ckpt": args.ckpt,
            },
            "metrics": metrics,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()
