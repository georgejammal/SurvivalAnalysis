from __future__ import annotations

import argparse
import os
import tempfile
import json
import sys

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tjepa_metabric.gene_sets import download_enrichr_library
from tjepa_metabric.models import CoxHead, MLPEncoder, cox_ph_loss
from tjepa_metabric.pathway_features import build_pathway_dataset


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
    p.add_argument("--endpoint", choices=["OS", "RFS"], default="OS")
    p.add_argument("--ckpt", default="tjepa_metabric/checkpoints/jepa.pt")
    p.add_argument(
        "--deepsurv_h5",
        default=None,
        help="If set, fine-tune/evaluate on the exact DeepSurv METABRIC H5 split (OS only).",
    )
    p.add_argument("--json_out", type=str, default=None, help="Optional path to write metrics JSON.")
    p.add_argument("--library", default="Reactome_2022")
    p.add_argument("--max_pathways", type=int, default=1000)
    p.add_argument("--min_overlap", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.deepsurv_h5:
        from tjepa_metabric.deepsurv_h5 import load_deepsurv_metabric_h5

        split = load_deepsurv_metabric_h5(args.deepsurv_h5)
        X_train, X_test = split.X_train, split.X_test
        t_train, t_test = split.t_train, split.t_test
        e_train, e_test = split.e_train, split.e_test

        # No separate validation in the original setup; we report train/test.
        X_val, t_val, e_val = X_train, t_train, e_train
        feature_count = X_train.shape[1]
        pathway_count = 0
        ihc4_count = feature_count
    else:
        gene_sets = download_enrichr_library(args.library)
        ds = build_pathway_dataset(
            endpoint=args.endpoint,
            gene_sets=gene_sets,
            min_genes_overlap=args.min_overlap,
            max_pathways=args.max_pathways,
            seed=args.seed,
        )
        X = ds.X.to_numpy(dtype=np.float32)
        time = ds.time.astype(np.float32)
        event = ds.event.astype(bool)

        # Split with event stratification
        X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
            X,
            time,
            event,
            test_size=0.2,
            random_state=args.seed,
            stratify=event.astype(int),
        )
        X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
            X_train,
            t_train,
            e_train,
            test_size=0.2,
            random_state=args.seed,
            stratify=e_train.astype(int),
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        feature_count = X.shape[1]
        pathway_count = len(ds.pathway_feature_names)
        ihc4_count = len(ds.ihc4_feature_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MLPEncoder(
        in_dim=X_train.shape[1] * 2,  # masked+indicator; we pass full visible mask for finetune
        emb_dim=args.emb_dim,
        hidden=[512, 512],
        dropout=0.1,
    )

    # Load pretrained encoder weights (if present)
    if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        encoder.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded encoder: {args.ckpt}")
    else:
        print(f"WARNING: checkpoint not found: {args.ckpt} (training from scratch)")

    encoder = encoder.to(device)
    head = CoxHead(args.emb_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)

    def to_emb(x_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np).to(device)
        mask = torch.ones_like(x)
        x2 = torch.cat([x * mask, mask], dim=1)
        return encoder(x2)

    best_val = -1.0
    best_state = None
    bad = 0
    patience = 30

    Xtr = torch.from_numpy(X_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)
    ttr = torch.from_numpy(t_train).to(device)
    tva = torch.from_numpy(t_val).to(device)
    etr = torch.from_numpy(e_train).to(device)
    eva = torch.from_numpy(e_val).to(device)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        opt.zero_grad(set_to_none=True)

        z = to_emb(X_train)
        risk = head(z)
        loss = cox_ph_loss(risk, ttr, etr)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 10 == 0:
            encoder.eval()
            head.eval()
            with torch.no_grad():
                val_risk = head(to_emb(X_val)).detach().cpu().numpy()
            val_ci = cindex(e_val, t_val, val_risk)
            if val_ci > best_val + 1e-4:
                best_val = val_ci
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
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
        head.load_state_dict(best_state["head"])

    encoder.eval()
    head.eval()
    with torch.no_grad():
        test_risk = head(to_emb(X_test)).detach().cpu().numpy()
        train_risk = head(to_emb(X_train)).detach().cpu().numpy()

    print(f"T-JEPA finetune endpoint={args.endpoint}")
    print(f"Train C-index: {cindex(e_train, t_train, train_risk):.4f}")
    if not args.deepsurv_h5:
        print(f"Val   C-index: {best_val:.4f}")
    print(f"Test  C-index: {cindex(e_test, t_test, test_risk):.4f}")
    print(f"n_features={feature_count} (pathways={pathway_count} + ihc4={ihc4_count})")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "script": "tjepa_metabric.finetune_survival",
            "argv": sys.argv,
            "endpoint": args.endpoint,
            "dataset": "deepsurv_h5" if args.deepsurv_h5 else "brca_metabric",
            "n_features": int(feature_count),
            "n_pathways": int(pathway_count),
            "n_ihc4": int(ihc4_count),
            "hyperparams": {
                "seed": args.seed,
                "epochs": args.epochs,
                "lr": args.lr,
                "emb_dim": args.emb_dim,
                "ckpt": args.ckpt,
                **(
                    {}
                    if args.deepsurv_h5
                    else {
                        "library": args.library,
                        "max_pathways": args.max_pathways,
                        "min_overlap": args.min_overlap,
                    }
                ),
            },
            "metrics": {
                "c_index_train": cindex(e_train, t_train, train_risk),
                **({} if args.deepsurv_h5 else {"c_index_val": best_val}),
                "c_index_test": cindex(e_test, t_test, test_risk),
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote: {args.json_out}")


if __name__ == "__main__":
    main()
