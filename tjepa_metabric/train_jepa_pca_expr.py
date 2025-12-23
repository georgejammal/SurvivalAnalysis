from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tjepa_metabric.masking import MaskIndexSampler
from tjepa_metabric.models import JEPA, MaskBatch
from tjepa_metabric.pca_expr import build_metabric_expression_pca_split


def _ensure_tmpdir():
    tmp = os.path.join(os.getcwd(), ".tmp")
    os.makedirs(tmp, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmp)
    tempfile.tempdir = tmp
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")


def main():
    _ensure_tmpdir()
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", choices=["OS", "RFS"], default="OS")
    p.add_argument("--expr_path", default="brca_metabric/data_mrna_illumina_microarray.txt")
    p.add_argument("--clinical_patient_path", default="brca_metabric/data_clinical_patient.txt")
    p.add_argument("--top_k_genes", type=int, default=4000)
    p.add_argument("--pca_dim", type=int, default=256)
    p.add_argument(
        "--impute_strategy",
        type=str,
        default="mean",
        help="Missing-value strategy passed to sklearn.impute.SimpleImputer (e.g. mean, median, most_frequent).",
    )
    p.add_argument("--token_dim", type=int, default=32)
    p.add_argument("--mlp_hidden_dim", type=int, default=256)
    p.add_argument("--mlp_layers", type=int, default=4)
    p.add_argument("--n_reg_tokens", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--out", default="tjepa_metabric/checkpoints/jepa_expr_pca.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = build_metabric_expression_pca_split(
        endpoint=args.endpoint,
        expr_path=args.expr_path,
        clinical_patient_path=args.clinical_patient_path,
        top_k_genes=args.top_k_genes,
        pca_dim=args.pca_dim,
        impute_strategy=args.impute_strategy,
        seed=args.seed,
    )

    # Pretrain on train split only.
    X = split.X_train.astype(np.float32)
    n_features = int(X.shape[1])

    num_feature_idx = np.arange(n_features, dtype=int)
    cat_feature_idx = np.array([], dtype=int)
    cat_cardinalities: list[int] = []

    masker = MaskIndexSampler(
        n_features=n_features,
        min_context_share=0.6,
        max_context_share=0.8,
        min_target_share=0.2,
        max_target_share=0.4,
        allow_overlap=False,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPA(
        n_features=n_features,
        token_dim=args.token_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        n_reg_tokens=args.n_reg_tokens,
        ema=args.ema,
        num_feature_idx=num_feature_idx,
        cat_feature_idx=cat_feature_idx,
        cat_cardinalities=cat_cardinalities,
        # PCA components are all continuous => no special type/index embeddings needed, but keep them on by default.
        use_feature_type_embedding=True,
        use_feature_index_embedding=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            idx_ctx, idx_tgt = masker.sample(len(xb))
            batch = MaskBatch(x=xb, idx_ctx=idx_ctx.to(device), idx_tgt=idx_tgt.to(device))
            opt.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            loss.backward()
            opt.step()
            model.update_target_ema()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.context_encoder.state_dict(),
            "token_dim": int(args.token_dim),
            "mlp_hidden_dim": int(args.mlp_hidden_dim),
            "mlp_layers": int(args.mlp_layers),
            "n_reg_tokens": int(args.n_reg_tokens),
            "use_feature_type_embedding": True,
            "use_feature_index_embedding": True,
            "num_feature_idx": num_feature_idx.astype(int).tolist(),
            "cat_feature_idx": cat_feature_idx.astype(int).tolist(),
            "cat_cardinalities": [],
            "endpoint": args.endpoint,
            "expr_path": args.expr_path,
            "clinical_patient_path": args.clinical_patient_path,
            "top_k_genes": int(args.top_k_genes),
            "pca_dim": int(args.pca_dim),
            "imputer_strategy": split.imputer_strategy,
            "imputer_statistics": split.imputer_statistics,
            "pca_components": split.pca_components,
            "pca_mean": split.pca_mean,
            "scaler_mean": split.scaler_mean,
            "scaler_scale": split.scaler_scale,
            "feature_names": split.feature_names,
            "seed": int(args.seed),
        },
        args.out,
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
