from __future__ import annotations

import argparse
import os
import tempfile
import json
import sys

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from tjepa_metabric.models import FeatureTokenizer, ProjectionLayer, SurvivalMLP, TokenEncoder, cox_ph_loss


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
    p.add_argument("--ckpt", default="tjepa_metabric/checkpoints/jepa.pt")
    p.add_argument(
        "--deepsurv_h5",
        required=True,
        help="Path to DeepSurv METABRIC H5 split (IHC4+clinical, OS).",
    )
    p.add_argument("--json_out", type=str, default=None, help="Optional path to write metrics JSON.")
    p.add_argument("--token_dim", type=int, default=32)
    p.add_argument("--token_mlp_layers", type=int, default=2)
    p.add_argument("--predictor_hidden_dim", type=int, default=256)
    p.add_argument(
        "--projection",
        choices=["mean_pool", "max_pool", "linear_flatten", "linear_per_feature"],
        default="linear_per_feature",
    )
    p.add_argument("--projection_out_dim", type=int, default=None)
    p.add_argument("--no_feature_type_emb", action="store_true")
    p.add_argument("--no_feature_index_emb", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--head_hidden_dim", type=int, default=256)
    p.add_argument("--head_layers", type=int, default=4)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="If >0, compute risk scores in micro-batches (keeps full-batch CoxPH loss).",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from tjepa_metabric.deepsurv_h5 import load_deepsurv_metabric_h5

    split = load_deepsurv_metabric_h5(args.deepsurv_h5)
    X_train, X_test = split.X_train, split.X_test
    t_train, t_test = split.t_train, split.t_test
    e_train, e_test = split.e_train, split.e_test

    # No separate validation in the original setup; we report train/test.
    X_val, t_val, e_val = X_train, t_train, e_train

    feature_count = int(X_train.shape[1])
    ihc4_count = feature_count
    pathway_count = 0

    # Fixed DeepSurv IHC4+clinical feature schema.
    cat_feature_idx = np.array([4, 5, 6, 7], dtype=int)
    cat_cardinalities = [2, 2, 2, 2]
    num_feature_idx = np.array([0, 1, 2, 3, 8], dtype=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allow checkpoint metadata to override CLI defaults so finetune matches pretraining.
    ckpt_meta = {}
    if os.path.isfile(args.ckpt):
        ckpt_meta = torch.load(args.ckpt, map_location="cpu")
        args.token_dim = int(ckpt_meta.get("token_dim", args.token_dim))
        args.token_mlp_layers = int(ckpt_meta.get("mlp_layers", args.token_mlp_layers))
        args.predictor_hidden_dim = int(ckpt_meta.get("mlp_hidden_dim", args.predictor_hidden_dim))
        if "use_feature_type_embedding" in ckpt_meta and not args.no_feature_type_emb:
            args.no_feature_type_emb = not bool(ckpt_meta["use_feature_type_embedding"])
        if "use_feature_index_embedding" in ckpt_meta and not args.no_feature_index_emb:
            args.no_feature_index_emb = not bool(ckpt_meta["use_feature_index_embedding"])

    tokenizer = FeatureTokenizer(
        n_features=feature_count,
        num_feature_idx=num_feature_idx,
        cat_feature_idx=cat_feature_idx,
        cat_cardinalities=cat_cardinalities,
        token_dim=args.token_dim,
        use_feature_type_embedding=not args.no_feature_type_emb,
        use_feature_index_embedding=not args.no_feature_index_emb,
    )

    encoder = TokenEncoder(
        tokenizer=tokenizer,
        token_dim=args.token_dim,
        n_reg_tokens=0,  # discard [REG] for downstream
        mlp_layers=args.token_mlp_layers,
        dropout=0.1,
    ).to(device)

    projection = ProjectionLayer(
        n_features=feature_count,
        token_dim=args.token_dim,
        mode=args.projection,
        out_dim=args.projection_out_dim,
    ).to(device)

    head = SurvivalMLP(
        in_dim=projection.out_dim,
        hidden_dim=args.head_hidden_dim,
        n_hidden=args.head_layers,
        dropout=args.head_dropout,
    ).to(device)

    if os.path.isfile(args.ckpt):
        try:
            encoder.load_state_dict(ckpt_meta["state_dict"], strict=False)
            print(f"loaded encoder: {args.ckpt}")
        except RuntimeError as e:
            print(f"WARNING: failed to load checkpoint weights ({args.ckpt}); training from scratch.")
            print(f"  error: {e}")
    else:
        print(f"WARNING: checkpoint not found: {args.ckpt} (training from scratch)")

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

    def forward_risk(x_t: torch.Tensor) -> torch.Tensor:
        if args.micro_batch_size and args.micro_batch_size > 0:
            risks: list[torch.Tensor] = []
            for i in range(0, x_t.shape[0], int(args.micro_batch_size)):
                xb = x_t[i : i + int(args.micro_batch_size)]
                tokens = encoder(xb, use_reg_tokens=False)  # (b, 9, H)
                z = projection(tokens)  # (b, P)
                risks.append(head(z))  # (b,)
            return torch.cat(risks, dim=0)
        tokens = encoder(x_t, use_reg_tokens=False)  # (B, 9, H)
        z = projection(tokens)  # (B, P)
        return head(z)  # (B,)

    best_val = -1.0
    best_state = None
    bad = 0
    patience = 30

    ttr = torch.from_numpy(t_train).to(device)
    tva = torch.from_numpy(t_val).to(device)
    etr = torch.from_numpy(e_train).to(device)
    eva = torch.from_numpy(e_val).to(device)

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

    print("T-JEPA finetune (DeepSurv METABRIC H5)")
    print(f"Train C-index: {cindex(e_train, t_train, train_risk):.4f}")
    print(f"Val   C-index: {best_val:.4f}")
    print(f"Test  C-index: {cindex(e_test, t_test, test_risk):.4f}")
    print(f"n_features={feature_count} (ihc4+clinical)")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "script": "tjepa_metabric.finetune_survival",
            "argv": sys.argv,
            "dataset": "deepsurv_h5",
            "n_features": int(feature_count),
            "n_pathways": int(pathway_count),
            "n_ihc4": int(ihc4_count),
            "hyperparams": {
                "seed": args.seed,
                "epochs": args.epochs,
                "lr": args.lr,
                "token_dim": args.token_dim,
                "token_mlp_layers": args.token_mlp_layers,
                "projection": args.projection,
                "projection_out_dim": args.projection_out_dim,
                "head_hidden_dim": args.head_hidden_dim,
                "head_layers": args.head_layers,
                "head_dropout": args.head_dropout,
                "freeze_encoder": bool(args.freeze_encoder),
                "micro_batch_size": int(args.micro_batch_size),
                "ckpt": args.ckpt,
            },
            "metrics": {
                "c_index_train": cindex(e_train, t_train, train_risk),
                "c_index_val": best_val,
                "c_index_test": cindex(e_test, t_test, test_risk),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()
