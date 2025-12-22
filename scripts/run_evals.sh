#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p .tmp evals tjepa_metabric/checkpoints

export TMPDIR="$ROOT/.tmp"
export TORCH_DISABLE_DYNAMO=1

PY="$ROOT/.venv/bin/python"

# DeepSurv paper baseline (exact H5 split)
$PY -m survival.deepsurv_baseline \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --epochs 500 \
  --json_out evals/deepsurv_h5_os.json

# T-JEPA pretrain + finetune on exact H5 split
$PY -m tjepa_metabric.train_jepa \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --epochs 500 \
  --out tjepa_metabric/checkpoints/jepa_h5_500.pt

$PY -m tjepa_metabric.finetune_survival \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --ckpt tjepa_metabric/checkpoints/jepa_h5_500.pt \
  --epochs 50 \
  --json_out evals/tjepa_h5_os.json

echo "Wrote eval JSONs to: $ROOT/evals/"

