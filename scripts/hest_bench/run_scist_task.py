"""
SciSt HEST-benchmark driver — runs one task end-to-end.

Must be invoked with the scist conda env active:
    /work/OSPH/s439765/envs/scist/bin/python

Usage:
    python scripts/hest_bench/run_scist_task.py --task IDC \
        [--data_dir /work/OSPH/s439765/scist] \
        [--skip_prep] [--skip_train] [--start_fold 0]

Writes per-task results to runs/hest_bench/summary.csv (shared with other
benchmark drivers) with encoder="scist".
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
SCIST_SRC = REPO_ROOT / "benchmark" / "SciSt" / "src"
PREP_SCRIPT = Path(__file__).parent / "prep_scist_hest.py"
GENE_DIR = REPO_ROOT / "benchmark" / "sCellST" / "data"

SUMMARY_CSV = REPO_ROOT / "runs" / "hest_bench" / "summary.csv"

# SciSt uses leave-one-out by integer index (Initiate_dataset), so we always
# generate one fold per slide.  This differs from sCellST's 2-fold COAD split,
# but yields a clean LOO aggregate that is directly comparable on every task.
TASK_IDS: dict[str, list[str]] = {
    "IDC":  ["TENX95", "TENX99", "NCBI783", "NCBI785"],
    "PAAD": ["TENX116", "TENX126", "TENX140"],
    "SKCM": ["TENX115", "TENX117"],
    "COAD": ["TENX111", "TENX147", "TENX148", "TENX149"],
    "LUNG": ["TENX118", "TENX141"],
}


def _loo_folds(sorted_ids: list[str]) -> list[dict]:
    """One fold per slide: test on that slide, train on the rest."""
    return [
        {"train": [s for s in sorted_ids if s != test], "test": [test]}
        for test in sorted_ids
    ]

NUM_GENES = 50
EPOCHS = 20
BATCH_SIZE = 32
IMG_ENCODER = "resnet34"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_IDS.keys()))
    p.add_argument("--data_dir", default="/work/OSPH/s439765/scist",
                   help="Root dir of SciSt-formatted HEST data.")
    p.add_argument("--skip_prep", action="store_true",
                   help="Skip data preparation (already done).")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; only run evaluation on existing checkpoints.")
    p.add_argument("--start_fold", type=int, default=0,
                   help="Resume from this fold index (0-based).")
    return p.parse_args()


def prep_data(task: str, data_dir: Path) -> None:
    logger.info(f"Preparing data for {task} …")
    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--task", task,
        "--out_dir", str(data_dir),
        "--gene_dir", str(GENE_DIR),
    ]
    subprocess.run(cmd, check=True)


def _wsi_ids_sorted(task_data_dir: Path) -> list[str]:
    """Return sorted WSI IDs found in 02_label/ — matches Initiate_dataset order."""
    label_dir = task_data_dir / "02_label"
    ids = [f.stem for f in sorted(label_dir.glob("*.json"))]
    return ids


def make_config(
    task_data_dir: Path,
    test_index: int,
    output_subdir: str,
) -> dict:
    """Build a SciSt config dict for one fold."""
    return {
        "patch_path": str(task_data_dir / "01_patch") + "/",
        "exp_label_path": str(task_data_dir / "02_label") + "/",
        "orig_path": str(task_data_dir / "06-noise_exp") + "/",
        "num_gene": NUM_GENES,
        "test_index": test_index,
        "data_name": "her2",   # naming convention matches our HEST patches
        "model_name": "SciSt",
        "img_encoder": IMG_ENCODER,
        "wts_path": None,
        "batch_size": BATCH_SIZE,
        "epoch": EPOCHS,
        "num_workers": 8,
        "loss_coefficient": 0.8,
        "log_step_freq": 20,
        "output_path": str(task_data_dir / output_subdir) + "/",
        "random": False,
    }


def train_fold(task_data_dir: Path, test_index: int, fold_tag: str) -> Path:
    """Run SciSt training for one fold; return model output dir."""
    cfg = make_config(task_data_dir, test_index, f"07-SciSt-{fold_tag}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as fh:
        yaml.dump(cfg, fh)
        cfg_path = fh.name

    train_script = SCIST_SRC / "scripts" / "main_scist_train.py"
    cmd = [sys.executable, str(train_script), "--cfg", cfg_path]
    logger.info(f"Training fold {fold_tag}: test_index={test_index} ...")
    # SCIST_SRC must be on PYTHONPATH so that `from train_test.SciSt_train import ...`
    # inside main_scist_train.py resolves correctly.  Python adds the *script file's*
    # directory (src/scripts/) to sys.path[0], not cwd, so we set PYTHONPATH explicitly.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCIST_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(SCIST_SRC), env=env)

    os.unlink(cfg_path)
    return Path(cfg["output_path"])


def predict_fold(
    model_output_dir: Path,
    task_data_dir: Path,
    test_index: int,
    test_ids: list[str],
) -> list[float]:
    """Load trained model and run prediction on test slides; return per-slide PCCs."""
    # Discover the test sample name from the sorted ID list
    sorted_ids = _wsi_ids_sorted(task_data_dir)
    if test_index >= len(sorted_ids):
        logger.error(f"test_index {test_index} out of range ({len(sorted_ids)} ids)")
        return []
    test_sample = sorted_ids[test_index]
    logger.info(f"Evaluating fold: test={test_sample}")

    # Add SciSt src to path so we can import its modules
    sys.path.insert(0, str(SCIST_SRC))
    from dataset.my_dataset import Initiate_dataset
    from models.sc_guide_model import SciSt
    from torch.utils.data import DataLoader

    patch_path = str(task_data_dir / "01_patch") + "/"
    label_path = str(task_data_dir / "02_label") + "/"
    orig_path = str(task_data_dir / "06-noise_exp") + "/"

    wts_path = model_output_dir / test_sample / f"{test_sample}-SciSt-best.pth"
    if not wts_path.exists():
        logger.warning(f"No checkpoint at {wts_path}, skipping.")
        return []

    model = SciSt(IMG_ENCODER, NUM_GENES)
    model.load_state_dict(torch.load(str(wts_path), map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    dataset = Initiate_dataset(
        patch_path, label_path, orig_path, "test", test_index, "her2"
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    preds_list, gt_list = [], []
    with torch.no_grad():
        for _, imgs, orig_exp, genes in loader:
            imgs = imgs.to(device)
            orig_exp = orig_exp.to(device)
            pred = model(imgs, orig_exp).cpu().numpy()
            preds_list.append(pred)
            gt_list.append(genes.numpy())

    if not preds_list:
        logger.warning("No predictions produced.")
        return []

    preds = np.concatenate(preds_list, axis=0)  # (N, 50)
    gts = np.concatenate(gt_list, axis=0)        # (N, 50)

    # Per-gene Pearson then mean across genes
    from scipy.stats import pearsonr
    gene_pccs = []
    for g in range(preds.shape[1]):
        if np.std(preds[:, g]) < 1e-6 or np.std(gts[:, g]) < 1e-6:
            continue
        r, _ = pearsonr(preds[:, g], gts[:, g])
        gene_pccs.append(r)

    if not gene_pccs:
        return [float("nan")]

    mean_pcc = float(np.mean(gene_pccs))
    logger.info(f"Fold PCC = {mean_pcc:.4f} ({len(gene_pccs)}/{NUM_GENES} valid genes)")
    return [mean_pcc]


def append_to_summary(task: str, pearson_mean: float, pearson_std: float, n_folds: int) -> None:
    row = {
        "encoder": "scist",
        "task": task,
        "pearson_mean": round(float(pearson_mean), 4),
        "pearson_std": round(float(pearson_std), 4),
        "n_folds": n_folds,
        "notes": "SciSt (resnet34, zero noise_exp — no scRNA ref)",
    }
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        mask = (df["encoder"] == "scist") & (df["task"] == task)
        if mask.any():
            for col, val in row.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)
    logger.success(f"Updated {SUMMARY_CSV}: {task} PCC={pearson_mean:.4f}±{pearson_std:.4f}")


def main() -> None:
    args = parse_args()
    task = args.task
    task_data_dir = Path(args.data_dir) / task

    logger.info(f"=== SciSt benchmark: task={task} ===")

    if not args.skip_prep:
        prep_data(task, Path(args.data_dir))
    else:
        logger.info("Skipping data prep (--skip_prep).")

    sorted_ids = _wsi_ids_sorted(task_data_dir)
    logger.info(f"WSI IDs (sorted): {sorted_ids}")
    folds = _loo_folds(sorted_ids)

    fold_pccs: list[float] = []
    for fold_idx, fold in enumerate(folds):
        if fold_idx < args.start_fold:
            logger.info(f"Skipping fold {fold_idx} (--start_fold={args.start_fold}).")
            continue

        test_id = fold["test"][0]
        test_index = sorted_ids.index(test_id)
        fold_tag = f"{task}_fold{fold_idx}"

        if not args.skip_train:
            model_out = train_fold(task_data_dir, test_index, fold_tag)
        else:
            model_out = task_data_dir / f"07-SciSt-{fold_tag}"
            logger.info(f"Skipping training (--skip_train); models expected at {model_out}")

        pccs = predict_fold(model_out, task_data_dir, test_index, fold["test"])
        if pccs:
            fold_pccs.append(float(np.mean(pccs)))
            logger.info(f"Fold {fold_idx} (test={test_id}) PCC = {fold_pccs[-1]:.4f}")

    if fold_pccs:
        mean_pcc = float(np.mean(fold_pccs))
        std_pcc = float(np.std(fold_pccs))
        logger.success(
            f"{task}: mean PCC = {mean_pcc:.4f} ± {std_pcc:.4f} over {len(fold_pccs)} folds"
        )
        append_to_summary(task, mean_pcc, std_pcc, len(fold_pccs))
    else:
        logger.warning("No folds completed — nothing written to summary.csv.")


if __name__ == "__main__":
    main()
