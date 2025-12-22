from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tjepa_metabric.gene_sets import download_enrichr_library
from tjepa_metabric.masking import FeatureMasker
from tjepa_metabric.models import JEPA, MaskBatch
from tjepa_metabric.pathway_features import build_pathway_dataset


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
    p.add_argument(
        "--deepsurv_h5",
        default=None,
        help="If set, pretrain on the exact DeepSurv METABRIC H5 features (apples-to-apples).",
    )
    p.add_argument("--library", default="Reactome_2022")
    p.add_argument("--max_pathways", type=int, default=1000)
    p.add_argument("--min_overlap", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--ihc4_aux_weight", type=float, default=5.0)
    p.add_argument("--out", default="tjepa_metabric/checkpoints/jepa.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.deepsurv_h5:
        from tjepa_metabric.deepsurv_h5 import load_deepsurv_metabric_h5

        split = load_deepsurv_metabric_h5(args.deepsurv_h5)
        # Pretrain on train split only (to avoid using test distribution).
        X = split.X_train.astype(np.float32)
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
        ihc4_idx = np.arange(len(feature_names), dtype=int)
        ds_meta = {
            "feature_names": feature_names,
            "ihc4_feature_names": feature_names,
            "pathway_feature_names": [],
            "library": "deepsurv_h5",
            "max_pathways": 0,
            "min_overlap": 0,
        }
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
        ihc4_idx = np.array(
            [ds.X.columns.get_loc(c) for c in ds.ihc4_feature_names], dtype=int
        )
        ds_meta = {
            "feature_names": list(ds.X.columns),
            "ihc4_feature_names": ds.ihc4_feature_names,
            "pathway_feature_names": ds.pathway_feature_names,
            "library": gene_sets.name,
            "max_pathways": args.max_pathways,
            "min_overlap": args.min_overlap,
        }

    n_features = X.shape[1]

    # Bias masking toward IHC4+C features by sampling them more often into target/context.
    w = np.ones(n_features, dtype=float)
    if ihc4_idx.size > 0:
        w[ihc4_idx] = 8.0

    masker = FeatureMasker(
        n_features=n_features,
        min_ctx_share=0.6,
        max_ctx_share=0.8,
        min_tgt_share=0.2,
        max_tgt_share=0.4,
        allow_overlap=False,
        feature_sampling_weights=w,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPA(
        n_features=n_features,
        emb_dim=args.emb_dim,
        ema=args.ema,
        ihc4_feature_idx=ihc4_idx,
        ihc4_aux_weight=args.ihc4_aux_weight,
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
            mask_ctx, mask_tgt = masker.sample_masks(len(xb))
            batch = MaskBatch(
                x=xb,
                mask_ctx=mask_ctx.to(device),
                mask_tgt=mask_tgt.to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            loss.backward()
            opt.step()
            model.update_target_ema()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": model.context_encoder.state_dict(),
            "emb_dim": args.emb_dim,
            **ds_meta,
        },
        args.out,
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
