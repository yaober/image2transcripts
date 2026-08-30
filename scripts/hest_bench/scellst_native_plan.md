# sCellST native HEST-Xenium benchmark — plan

sCellST is a *full pipeline*, not a frozen patch encoder, so it cannot be
slotted into HEST-Bench like Image2Transcript or GHIST. To get a fair
sCellST number on the 5 Xenium tasks (IDC, PAAD, SKCM, COAD, LUNG) we have
to run its native MIL training per fold and score on the same 50-HVG
Pearson HEST-Bench uses.

## Why this is more than an afternoon

sCellST has its own dependency stack (Python 3.10, torch 2.4.1+cu118,
torch-scatter from a private wheel index, squidpy, lightning) and its own
data layout (raw HEST slides + CellViT cell shapes), neither of which is
compatible with the `hest_bench` env or the `eval/bench_data` patches we
already have for HEST-Bench.

## Setup (one-time)

```bash
# 1) New conda env, sCellST-only.
conda create -p /work/OSPH/s439765/envs/scellst python=3.11 -y
conda activate /work/OSPH/s439765/envs/scellst

# 2) Install sCellST with poetry from the vendored copy.
pip install poetry
cd benchmark/sCellST
poetry install
# torch-scatter wheel comes from the URL in pyproject.toml; if poetry
# struggles, fall back to:
#   pip install torch==2.4.1 torchvision==0.19.1 --index-url \
#       https://download.pytorch.org/whl/cu118
#   pip install torch-scatter -f \
#       https://data.pyg.org/whl/torch-2.4.1+cu118.html
```

## Data preparation (per task)

For each task in {IDC, PAAD, SKCM, COAD, LUNG}, download the *raw* HEST
slides covered by that task (e.g. for IDC: TENX95, TENX99, NCBI783,
NCBI785) using sCellST's `download_data`. This pulls full WSIs, not just
the 224×224 spot patches we already have, and runs CellViT cell
segmentation. Storage: ~50–150 GB per task.

```python
from pathlib import Path
from scellst.submit_function import download_data

TASK_SLIDES = {
    "IDC":  ["TENX95", "TENX99", "NCBI783", "NCBI785"],
    "PAAD": ["TENX116", "TENX126", "TENX140"],
    "SKCM": ["TENX115", "TENX117"],
    "COAD": ["TENX111", "TENX147", "TENX148", "TENX149"],
    "LUNG": ["TENX118", "TENX141"],
}
for task, ids in TASK_SLIDES.items():
    download_data(Path(f"runs/hest_bench/scellst/{task}/hest_data"), None, ids)
```

## Cell embedding (per task)

Use ImageNet-rn50 (fast) — MoCo SSL would take days per task and isn't
part of v1.

```python
from scellst.submit_function import embed_cells
embed_cells(path_dataset, None, ids, tag="imagenet-rn50",
            model_name="resnet50", normalisation_type="train")
```

## MIL training + eval (per task, per fold)

HEST-Bench uses leave-one-slide-out k-fold. We replicate that: for each
fold `i` in `splits/train_{i}.csv`, train MIL on the listed slides, predict
on the held-out slide, score Pearson on the same 50-HVG panel from
`<task>/var_50genes.json`.

```python
from scellst.train import train_and_save
from scellst.predict import predict_and_save

# train
additional_kwargs = {
    "data_dir": path_dataset,
    "save_dir_tag": f"{task}_fold{i}",
    "embedding_tag": "imagenet-rn50_train",
    "genes": gene_list_csv,  # 50 HVGs from HEST-Bench var_50genes.json
    "list_training_ids": train_slide_ids,
}
train_and_save(Path("config/gene_default.yaml"), additional_kwargs)

# predict
predict_and_save(config_dir, {"predict_id": held_out_slide},
                 infer_mode="bag", compute_metrics=True, save_adata=True)
```

## Aggregation

Concatenate predictions across folds, compute Pearson per gene, then mean
± std across folds — same protocol HEST-Bench uses internally — and emit
one row per (encoder=`scellst`, task) into `runs/hest_bench/summary.csv`.

## Estimated wall clock

- Env install: ~1 h
- Per-task data download + cell extraction: 4–10 h
- Per-task per-fold training: 2–4 h (4 folds typical)
- All 5 tasks: ~3–5 days end-to-end on a single GPU; ~1 day with 5 jobs
  in parallel.

## Status

NOT YET STARTED. Image2Transcript + GHIST results land first; sCellST is
queued for the next push.
