# Image2Transcript

**Predict single-cell transcriptomic profiles directly from H&E histology.**

Image2Transcript is a multimodal deep-learning model that maps a cell-centered
H&E image crop to the cell's gene-expression profile. It couples a Vision
Transformer image encoder with a transformer-based gene encoder under a joint
objective that combines **image–gene contrastive alignment**, **embedding
alignment**, and a **zero-inflated negative binomial (ZINB)** reconstruction
head tailored to sparse, over-dispersed single-cell counts.

The reference model is trained and evaluated on a 372-gene 10x **Xenium** panel,
predicting per-cell expression from morphology alone.

---

## Table of contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference / prediction](#inference--prediction)
- [Command-line reference](#command-line-reference)
- [Citation](#citation)

---

## Highlights

- **Morphology → transcriptome.** Predicts a full gene panel per cell from a
  single H&E crop, no spatial barcoding needed at inference time.
- **Probabilistic output.** A ZINB head models zero-inflation and
  over-dispersion, giving calibrated counts rather than point estimates.
- **Contrastive + reconstruction training.** CLIP-style image–gene alignment
  regularizes the encoder while the ZINB head reconstructs counts.
- **Leakage-safe evaluation.** Slides are split by sample group so no tissue
  leaks between train / val / test.
- **Multi-GPU by default.** Distributed data-parallel (DDP) training with mixed
  precision and numerically-hardened ZINB losses.

## How it works

```
              ┌──────────────────────┐
 H&E crop ───▶│  ViT-B/16 image enc.  │──▶ image embedding ─┐
              └──────────────────────┘                      │  contrastive
                                                            ├─ alignment
              ┌──────────────────────┐                      │
 counts   ───▶│  gene transformer enc.│──▶ gene embedding ──┘
 (+neighbor   └──────────────────────┘
  logFC)                    │
                            ▼
              ┌──────────────────────┐
 image feat ─▶│  ZINB head (μ, θ, π)  │──▶ predicted expression
              └──────────────────────┘
```

- **Image encoder** — ViT-B/16 (ImageNet-pretrained by default), producing a
  pooled `[CLS]` feature.
- **Gene encoder** — a Transformer over per-gene value tokens (count +
  spatial-neighbor logFC), gene-ID embeddings, and learned positions.
- **ZINB head** — predicts mean `μ`, dispersion `θ`, and dropout logit `π` for
  every gene from the image feature.
- **Objective** — `w_contrast · L_contrastive + w_align · L_align + w_zinb · L_ZINB`.

The full model definition is in [model/model.py](model/model.py); the loss in
[model/train.py](model/train.py); the training loop in [model/main.py](model/main.py).

## Repository layout

```
Image2Transcript/
├── model/
│   ├── model.py        # Image2Transcripts: ViT + gene transformer + ZINB head
│   ├── train.py        # ZINB / MSE reconstruction + contrastive + alignment losses
│   └── main.py         # dataset, DDP training loop, CLI entry point
├── scripts/
│   ├── cell_extraction.py(.sh)   # WSI + segmentation masks -> per-cell H&E crops
│   ├── gene_extraction.py        # Xenium cell_feature_matrix -> per-cell .h5ad
│   ├── create_hdf5.py(.sh)       # pack (image, gene) embedding pairs into HDF5
│   ├── image2embedding.py(.sh)   # optional: precompute image embeddings
│   ├── eval_fixedsplit.py        # dump held-out predictions + metrics
│   ├── 4v100.sh                  # example 4-GPU training launcher
│   └── summarize_train_log.sh    # quick train_log.csv summary
├── splits/
│   └── xenium_slide_split_v1.json   # leakage-safe train/val/test slide split
├── gene_signatures/    # curated marker-gene signatures (cell typing / analysis)
├── predict.ipynb       # end-to-end inference demo
├── environment_image2transcripts.yml
└── requirements.txt
```

> The public tree ships the **core, reusable pipeline** only. Experiment
> sweeps, external benchmark suites, paper figures, run logs, and manuscript
> drafts are intentionally excluded (see `.gitignore`).

## Installation

Create the conda environment (recommended):

```bash
conda env create -f environment_image2transcripts.yml
conda activate image2transcripts
```

The environment provides Python 3.9, PyTorch (CUDA build), `timm`, `scanpy`,
`anndata`, `scikit-learn`, and the usual scientific stack. A GPU is required for
training; evaluation and small-scale inference can run on CPU (slowly).

Some optional paths need extra packages:

- **Gene-panel / analysis notebooks** — `scanpy`, `anndata` (already included).
- **Precomputing CONCH image embeddings** (`scripts/image2embedding.py`) — the
  `conch` package and a HuggingFace token.
- **Gated pathology foundation models** (Phikon2, Prov-GigaPath, UNI2-h — used by
  `feature_extraction.py` and the foundation-model benchmark) — each is gated on
  HuggingFace and requires an authenticated token.

  In every case, set **your own** token via the environment, never in code. The
  scripts read `HF_TOKEN` from the environment and raise a clear error if it is
  unset — there is no bundled fallback token:

  ```bash
  export HF_TOKEN=<your_hf_token>
  ```

## Data preparation

Image2Transcript trains on pairs of **(cell-centered H&E crop, cell expression
vector)**. Starting from a Xenium run you need two directories that share slide
IDs and cell IDs:

```
data/
├── images/<slide_id>/<cell_id>.png     # one crop per cell
└── gene_expression/<slide_id>.h5ad     # AnnData with cells × genes counts
```

**1. Extract per-cell H&E crops** from the whole-slide image and Xenium cell
segmentation masks:

```bash
python scripts/cell_extraction.py \
  --wsi   path/to/slide.ome.tiff \
  --mask  path/to/cells.zarr \
  --output data/images/<slide_id> \
  --cell_count 10          # neighboring cells to include in the crop FOV
```

**2. Export per-cell expression** from the Xenium `cell_feature_matrix` into
AnnData (`.h5ad`) keyed by cell ID (`scripts/gene_extraction.py`). Each h5ad
must carry `x_centroid` / `y_centroid` in `.obs` — the loader uses them to
compute the spatial-neighbor logFC feature.

Cell IDs in the image filenames must match `obs_names` in the h5ad; ordering is
not assumed. The dataset loader automatically intersects genes across slides and
keeps only cells that have both a crop and an expression row.

## Training

Single-GPU quick start:

```bash
python model/main.py \
  --gene_dir data/gene_expression \
  --img_dir  data/images \
  --out_dir  runs/my_run \
  --gpus 1 --batch 32 --epochs 100
```

Multi-GPU, leakage-safe split (the recommended reference recipe):

```bash
python model/main.py \
  --gene_dir data/gene_expression \
  --img_dir  data/images \
  --split_file splits/xenium_slide_split_v1.json \
  --out_dir  runs/full_seed42 --seed 42 \
  --gpus 4 --batch 128 --epochs 100 \
  --lr_backbone 1e-5 --lr_head 5e-4 \
  --w_contrast 0.1 --w_align 0.1 --w_zinb 1.0 \
  --gene_mask_ratio 0.15 --patience 15
```

`scripts/4v100.sh` is a ready-made 4×GPU launcher you can adapt.

Training writes to `--out_dir`:

- `best_model.pt` / `last_model.pt` — checkpoints (best = lowest val loss)
- `train_log.csv` — per-epoch loss, cosine similarity, Pearson, LR
- `slide_split_used.json` — the exact slide split used

Get a one-line summary of any run:

```bash
bash scripts/summarize_train_log.sh runs/full_seed42/train_log.csv
```

### Training notes

- Training uses **DDP + AMP (fp16 forward)** with the **ZINB loss computed in
  fp32** for stability. The loop collectively skips non-finite / oversized
  batches across ranks.
- If you swap in a new reconstruction objective, keep every model parameter on
  the graph (DDP aborts on unused parameters) — see the MSE-mode pattern in
  [model/train.py](model/train.py).
- If you freeze the image encoder, use a gentler head recipe (lower `--lr_head`,
  `--fixed_temperature`, tighter `--grad_clip`, longer `--warmup_ratio`) to
  avoid a val-loss runaway.

## Evaluation

Dump held-out predictions and metrics for a trained run:

```bash
python scripts/eval_fixedsplit.py --run_dir runs/full_seed42
```

This reloads the checkpoint, rebuilds the exact split, and writes predicted vs.
true expression plus per-cell / per-gene Pearson correlations under
`runs/full_seed42/eval/`.

## Inference / prediction

See [predict.ipynb](predict.ipynb) for an end-to-end example: load a checkpoint,
crop cells from a slide, and predict expression. The minimal call is:

```python
import sys, torch
sys.path.insert(0, "model")
from model import Image2Transcripts

model = Image2Transcripts(gene_dim=372)                 # panel size
model.load_state_dict(torch.load("runs/full_seed42/best_model.pt"))
model.eval()

# image: (B, 3, 224, 224) float in [0,1]; gene_input: (B, 2*gene_dim)
i_emb, g_emb, mu, theta, pi = model(image, gene_input)
# `mu` is the predicted expression (mean of the ZINB).
```

## Command-line reference

Selected `model/main.py` flags (all optional, sensible defaults):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--gene_dir` | — | Directory of per-slide `.h5ad` expression files (required) |
| `--img_dir` | — | Directory of `<slide_id>/<cell_id>.png` crops (required) |
| `--out_dir` | `output_zinb` | Where checkpoints and logs are written |
| `--split_file` | none | Explicit train/val/test slide split (leakage-safe); random 80/20 if omitted |
| `--gpus` | all visible | Number of GPUs for DDP |
| `--batch` / `--epochs` | 32 / 100 | Batch size and max epochs |
| `--lr_backbone` / `--lr_head` | 1e-5 / 5e-4 | Differential learning rates for ViT vs. heads |
| `--w_contrast` / `--w_align` / `--w_zinb` | 0.1 / 0.1 / 1.0 | Loss-term weights |
| `--loss_mode` | `zinb` | `zinb` likelihood (default) or `mse` on log1p counts (baseline) |
| `--gene_mask_ratio` | 0.15 | Fraction of genes masked during training |
| `--k_neighbors` | 5 | Spatial neighbors for the logFC feature (0 disables it) |
| `--fixed_temperature` | off | Freeze contrastive temperatures at 0.07 |
| `--patience` | 15 | Early-stopping patience (epochs) |
| `--seed` | 42 | Seed for split, sampling, and training |

Run `python model/main.py --help` for the complete list.




