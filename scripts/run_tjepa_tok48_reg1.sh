#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p .tmp evals tjepa_metabric/checkpoints

export TMPDIR="$ROOT/.tmp"
export TORCH_DISABLE_DYNAMO=1

PY="$ROOT/.venv/bin/python"
H5="DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5"

CKPT="tjepa_metabric/checkpoints/jepa_h5_bs32_e100_tok48_reg1.pt"
OUT="evals/tjepa_h5_bs32_e100_tok48_reg1_ft.json"

# Pretrain T-JEPA with token_dim=48 and n_reg_tokens=1
$PY -m tjepa_metabric.train_jepa \
  --deepsurv_h5 "$H5" \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --token_dim 48 \
  --n_reg_tokens 1 \
  --out "$CKPT"

# Fine-tune survival downstream (uses token_dim from checkpoint metadata)
$PY -m tjepa_metabric.finetune_survival \
  --deepsurv_h5 "$H5" \
  --ckpt "$CKPT" \
  --epochs 250 \
  --lr 1e-5 \
  --projection linear_per_feature \
  --json_out "$OUT"

echo "Wrote: $OUT"
