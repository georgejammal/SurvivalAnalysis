#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p .tmp evals tjepa_metabric/checkpoints

export TMPDIR="$ROOT/.tmp"
export TORCH_DISABLE_DYNAMO=1

PY="$ROOT/.venv/bin/python"

CKPT="tjepa_metabric/checkpoints/jepa_metabric_expr_pca.pt"
OUT="evals/tjepa_metabric_expr_pca.json"

# PCA over top-k genes (continuous-only), then JEPA pretraining
$PY -m tjepa_metabric.train_jepa_pca_expr \
  --endpoint OS \
  --top_k_genes 4000 \
  --pca_dim 256 \
  --token_dim 32 \
  --mlp_layers 6 \
  --n_reg_tokens 1 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --out "$CKPT"

# Survival fine-tuning on PCA components
$PY -m tjepa_metabric.finetune_survival_pca_expr \
  --ckpt "$CKPT" \
  --epochs 200 \
  --lr 1e-4 \
  --projection linear_per_feature \
  --json_out "$OUT"

echo "Wrote: $OUT"
