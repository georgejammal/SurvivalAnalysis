# METABRIC survival analysis (DeepSurv + T-JEPA)

This repo runs survival prediction experiments on the METABRIC breast cancer dataset and reports results with **C-index** (concordance index).

You can run:
- A **DeepSurv-like** baseline (same network style as the DeepSurv paper)
- A **T-JEPA** self-supervised pretraining step, then **survival fine-tuning**
- Extra baselines on full gene expression (**Coxnet** and **Random Survival Forest**)

The important thing: there are **two different “data setups”** in this repo.
1. **DeepSurv H5 (apples-to-apples)**: uses the exact DeepSurv-provided `metabric_IHC4_clinical_train_test.h5` split (9 features).
2. **Full METABRIC expression**: uses the big `brca_metabric/data_mrna_illumina_microarray.txt` + `brca_metabric/data_clinical_patient.txt`.

---

## Quick start (recommended)

These commands run the “apples-to-apples” comparison against DeepSurv on the **exact same data split**.

1) Go to repo root:
```bash
cd "/path/to/survivla analysis METABRIC"
```

2) Run the full evaluation script:
```bash
bash scripts/run_evals.sh
```

Outputs:
- Metrics JSON files in `evals/`
- Checkpoints in `tjepa_metabric/checkpoints/`

If you want the other experiment that uses the **full expression data but reduces it with PCA**:
```bash
bash scripts/run_metabric_expr_pca_jepa_deepsurv.sh
```

---

## Best run (current)

As of the saved results in `evals/` (DeepSurv H5, OS):
- DeepSurv baseline test C-index: ~0.6520 (`evals/deepsurv_h5_os.json`)
- Best T-JEPA test C-index: ~0.6532 (`evals/tjepa_h5_bs32_e100_ft.json`)

Reproduce the best T-JEPA run:
```bash
./.venv/bin/python -m tjepa_metabric.train_jepa \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --seed 42 \
  --out tjepa_metabric/checkpoints/jepa_h5_bs32_e100.pt

./.venv/bin/python -m tjepa_metabric.finetune_survival \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --ckpt tjepa_metabric/checkpoints/jepa_h5_bs32_e100.pt \
  --epochs 250 \
  --lr 1e-5 \
  --projection linear_per_feature \
  --seed 42 \
  --json_out evals/tjepa_h5_bs32_e100_ft.json
```

Run the DeepSurv baseline on the same split:
```bash
./.venv/bin/python -m survival.deepsurv_baseline \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --epochs 500 \
  --seed 42 \
  --json_out evals/deepsurv_h5_os.json
```

(Results can move a little due to randomness / hardware. Keep `--seed 42` for closer reproduction.)

---

## Folder guide (what is where)

- `brca_metabric/`
  - The METABRIC dataset export (large text files).
  - The code in this repo uses mainly:
    - `brca_metabric/data_mrna_illumina_microarray.txt` (gene expression)
    - `brca_metabric/data_clinical_patient.txt` (survival times + event labels)
- `DEEPSURV_METABRIC/`
  - DeepSurv’s provided METABRIC split (small H5 file):
    - `DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5`
- `survival/`
  - Data loading + baselines.
  - `survival/metabric.py`: loads expression + clinical tables and builds `(X, y)` for survival.
  - `survival/deepsurv_baseline.py`: DeepSurv-like neural network baseline.
  - `survival/rsf_baseline.py`: Coxnet + Random Survival Forest baselines on gene expression.
- `tjepa_metabric/`
  - T-JEPA model + training code.
  - `train_jepa.py`: JEPA pretraining (either on DeepSurv H5 or on pathway features).
  - `finetune_survival.py`: supervised survival fine-tuning on DeepSurv H5.
  - `train_jepa_pca_expr.py`: JEPA pretraining on PCA-reduced full expression.
  - `finetune_survival_pca_expr.py`: supervised survival fine-tuning on PCA components.
- `scripts/`
  - Convenience “run everything” scripts.
- `evals/`
  - Saved results as JSON (C-index numbers, arguments used).
- `.tmp/`
  - Temporary files / torch caches (created automatically).

Note: `.gitignore` excludes big data folders and outputs (`brca_metabric/`, `DEEPSURV_METABRIC/`, `evals/`, `tjepa_metabric/checkpoints/`). On a fresh clone you may need to re-download data.

---

## Data used for the DeepSurv C-index comparison (important)

When we say “compare with DeepSurv C-index”, the clean comparison is:
- Use `DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5`
- Train and test on **the exact same split** from that file

That is exactly what these do:
- `survival/deepsurv_baseline.py --deepsurv_h5 ...`
- `tjepa_metabric/train_jepa.py --deepsurv_h5 ...` (pretrain on **train split only**)
- `tjepa_metabric/finetune_survival.py --deepsurv_h5 ...` (fine-tune + report test C-index)

What is inside the DeepSurv H5:
- `X` has shape `(N, 9)` (9 input features)
- `t` is survival time in months
- `e` is the event indicator (1 = event happened, 0 = censored)
- Train/test splits are already provided in the file

The 9 features (fixed order) are:
1. `EGFR` (expression)
2. `PGR` (expression)
3. `ERBB2` (expression)
4. `MKI67` (expression)
5. `HORMONE_THERAPY` (0/1)
6. `RADIO_THERAPY` (0/1)
7. `CHEMOTHERAPY` (0/1)
8. `ER_IHC` (0/1)
9. `AGE_AT_DIAGNOSIS` (number)

So: **the DeepSurv comparison in this repo uses only those 9 features**, not the full gene expression matrix.

---

## How C-index is computed here (simple explanation)

C-index checks if the model gives higher “risk” to patients who have the event earlier.

In code, we use:
- `sksurv.metrics.concordance_index_censored(event, time, risk)`

Where:
- `event` is `True/False`
- `time` is survival time
- `risk` is a 1D score from the model (higher = higher risk)

Scripts that print and/or save C-index:
- DeepSurv baseline: `survival/deepsurv_baseline.py`
- T-JEPA fine-tuning: `tjepa_metabric/finetune_survival.py`, `tjepa_metabric/finetune_survival_pca_expr.py`
- RSF/Coxnet baselines: `survival/rsf_baseline.py`

---

## How training works (what runs in what order)

### 1) DeepSurv baseline (no JEPA)

File: `survival/deepsurv_baseline.py`

Two modes:
- **Recommended (exact DeepSurv split):**
  ```bash
  ./.venv/bin/python -m survival.deepsurv_baseline \
    --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
    --epochs 500 \
    --seed 42 \
    --json_out evals/deepsurv_h5_os.json
  ```
- **Rebuild “IHC4-like” features from the raw METABRIC export:**
  - Builds the same 9-feature table by combining:
    - expression of EGFR/PGR/ERBB2/MKI67 from `brca_metabric/data_mrna_illumina_microarray.txt`
    - binary treatments + age from `brca_metabric/data_clinical_patient.txt`
  - Then it makes its own train/val/test split with `train_test_split`.

### 2) T-JEPA pretraining on DeepSurv H5 (self-supervised)

File: `tjepa_metabric/train_jepa.py`

What it does in simple words:
- Takes the 9 input features
- Randomly hides (“masks”) some features
- Learns to predict the hidden part from the visible part
- Uses an EMA target encoder (common JEPA trick)

Important: when `--deepsurv_h5` is used, pretraining uses **only the train split** (to avoid leaking test information).

Example:
```bash
./.venv/bin/python -m tjepa_metabric.train_jepa \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --seed 42 \
  --out tjepa_metabric/checkpoints/jepa_h5_bs32_e100.pt
```

### 3) T-JEPA survival fine-tuning on DeepSurv H5 (supervised)

File: `tjepa_metabric/finetune_survival.py`

What it does:
- Loads the JEPA checkpoint
- Builds a small survival head (Cox proportional hazards loss)
- Trains to predict a risk score from the 9 features
- Reports train/test C-index

Example:
```bash
./.venv/bin/python -m tjepa_metabric.finetune_survival \
  --deepsurv_h5 DEEPSURV_METABRIC/metabric_IHC4_clinical_train_test.h5 \
  --ckpt tjepa_metabric/checkpoints/jepa_h5_bs32_e100.pt \
  --epochs 250 \
  --lr 1e-5 \
  --projection linear_per_feature \
  --seed 42 \
  --json_out evals/tjepa_h5_bs32_e100_ft.json
```

### 4) Full expression variant (PCA → JEPA → survival)

This uses the big gene expression matrix but keeps it simple by reducing it with PCA.

Pipeline:
1. `survival/metabric.py` loads full expression + clinical labels.
2. `tjepa_metabric/pca_expr.py`:
   - selects top-k genes by variance
   - splits train/val/test (stratified by event)
   - imputes missing values
   - standardizes (fit on train only)
   - runs PCA (fit on train only)
3. `tjepa_metabric/train_jepa_pca_expr.py` pretrains JEPA on PCA components (train split only).
4. `tjepa_metabric/finetune_survival_pca_expr.py` fine-tunes survival head and reports C-index.

Run script:
```bash
bash scripts/run_metabric_expr_pca_jepa_deepsurv.sh
```

### 5) RSF + Coxnet baselines on full expression

File: `survival/rsf_baseline.py`

Example:
```bash
./.venv/bin/python -m survival.rsf_baseline --endpoint OS --top_k_genes 4000
```

---

## “One command” runners

- Exact DeepSurv H5 comparison (baseline + T-JEPA):
  - `bash scripts/run_evals.sh`
- PCA-expression JEPA experiment:
  - `bash scripts/run_metabric_expr_pca_jepa_deepsurv.sh`
- A specific hyperparameter run (token_dim=48, reg_tokens=1):
  - `bash scripts/run_tjepa_tok48_reg1.sh`

---

## Output files (what to look at)

`evals/*.json` files include:
- the command line used (`argv`)
- hyperparameters (`hyperparams`)
- metrics (`metrics.c_index_*`)

Example:
- `evals/deepsurv_h5_os.json` stores the DeepSurv baseline C-index on the DeepSurv H5 split.

---

## Setup notes (if you don’t already have a working venv)

This repo’s scripts expect `./.venv/bin/python`. If you don’t have it yet:
```bash
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install numpy pandas scikit-learn scikit-survival torch h5py requests
```

If `scikit-survival` fails to install on your machine, use a Conda environment (it often installs easier that way).

---

## Optional: pathway features mode (needs internet)

If you run `tjepa_metabric/train_jepa.py` **without** `--deepsurv_h5`, it downloads gene sets from Enrichr (needs network access) via `tjepa_metabric/gene_sets.py`, then builds pathway scores in `tjepa_metabric/pathway_features.py`.
