"""
sCellST benchmark driver — runs one HEST-Bench task end-to-end.

Must be invoked with CWD = benchmark/sCellST/ and the scellst venv active.

Usage (from benchmark/sCellST/):
    python ../../scripts/hest_bench/run_scellst_task.py --task IDC \
        [--data_dir /work/OSPH/s439765/scellst] \
        [--skip_download] [--skip_embed] [--start_fold 0]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

SCELLST_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "sCellST"
BENCH_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "hest_bench"
    / "eval"
    / "bench_data"
)
SUMMARY_CSV = (
    Path(__file__).resolve().parents[2] / "runs" / "hest_bench" / "summary.csv"
)

# Leave-one-slide-out folds per task, derived from HEST-Bench splits.
TASK_CONFIG: dict[str, dict] = {
    "IDC": {
        "all_ids": ["TENX95", "TENX99", "NCBI783", "NCBI785"],
        "folds": [
            {"train": ["TENX95", "NCBI785", "NCBI783"], "test": ["TENX99"]},
            {"train": ["TENX99", "NCBI785", "NCBI783"], "test": ["TENX95"]},
            {"train": ["TENX99", "TENX95", "NCBI783"], "test": ["NCBI785"]},
            {"train": ["TENX99", "TENX95", "NCBI785"], "test": ["NCBI783"]},
        ],
    },
    "PAAD": {
        "all_ids": ["TENX116", "TENX126", "TENX140"],
        "folds": [
            {"train": ["TENX140", "TENX126"], "test": ["TENX116"]},
            {"train": ["TENX116", "TENX126"], "test": ["TENX140"]},
            {"train": ["TENX116", "TENX140"], "test": ["TENX126"]},
        ],
    },
    "SKCM": {
        "all_ids": ["TENX115", "TENX117"],
        "folds": [
            {"train": ["TENX115"], "test": ["TENX117"]},
            {"train": ["TENX117"], "test": ["TENX115"]},
        ],
    },
    "COAD": {
        "all_ids": ["TENX111", "TENX147", "TENX148", "TENX149"],
        # Fold 0: train on TENX111, test on the three others (multi-slide test set)
        # Fold 1: train on the three, test on TENX111
        "folds": [
            {"train": ["TENX111"], "test": ["TENX149", "TENX148", "TENX147"]},
            {"train": ["TENX149", "TENX148", "TENX147"], "test": ["TENX111"]},
        ],
    },
    "LUNG": {
        "all_ids": ["TENX118", "TENX141"],
        "folds": [
            {"train": ["TENX118"], "test": ["TENX141"]},
            {"train": ["TENX141"], "test": ["TENX118"]},
        ],
    },
}

EMB_TAG = "imagenet-rn50"
NORM_TYPE = "train"
FULL_EMB_TAG = f"{EMB_TAG}_{NORM_TYPE}"
CONFIG_PATH = SCELLST_DIR / "config" / "gene_default.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    p.add_argument(
        "--data_dir",
        default="/work/OSPH/s439765/scellst",
        help="Root dir for HEST data downloads (large, use /work).",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download+convert step (data already present).",
    )
    p.add_argument(
        "--skip_embed",
        action="store_true",
        help="Skip cell embedding step (embeddings already present).",
    )
    p.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="Resume from this fold index (0-based).",
    )
    return p.parse_args()


def download_task(path_dataset: Path, all_ids: list[str]) -> None:
    from scellst.submit_function import download_data

    logger.info(f"Downloading {all_ids} → {path_dataset}")
    path_dataset.mkdir(parents=True, exist_ok=True)
    download_data(path_dataset, None, all_ids)


def embed_task(path_dataset: Path, all_ids: list[str]) -> None:
    from scellst.submit_function import embed_cells

    logger.info(f"Embedding cells with {EMB_TAG}")
    embed_cells(
        path_dataset,
        None,
        all_ids,
        tag=EMB_TAG,
        model_name="resnet50",
        normalisation_type=NORM_TYPE,
    )


def train_fold(
    path_dataset: Path, task: str, fold_idx: int, fold: dict
) -> tuple[Path, int]:
    from scellst.train import train_and_save

    internal_fold = fold_idx % len(fold["train"])
    save_dir_tag = f"{task}_fold{fold_idx}"
    kwargs = {
        "data_dir": path_dataset,
        "save_dir_tag": save_dir_tag,
        "embedding_tag": FULL_EMB_TAG,
        "genes": f"{task}_50hvg_bench",
        "list_training_ids": fold["train"],
        "fold": internal_fold,
    }
    logger.info(f"Training {task} fold {fold_idx}: train={fold['train']}")
    train_and_save(CONFIG_PATH, kwargs)

    exp_tag = f"embedding_tag={FULL_EMB_TAG};genes={task}_50hvg_bench;fold={internal_fold}"
    model_dir = Path("models") / "mil" / save_dir_tag / exp_tag
    logger.info(f"Model saved to {model_dir}")
    return model_dir, internal_fold


def predict_fold(
    model_dir: Path, path_dataset: Path, task: str, fold_idx: int,
    internal_fold: int, test_ids: list[str]
) -> list[pd.DataFrame]:
    """Run prediction on each test slide; return list of per-slide metrics DataFrames."""
    from scellst.predict import predict_and_save

    exp_tag = f"embedding_tag={FULL_EMB_TAG};genes={task}_50hvg_bench;fold={internal_fold}"
    metrics_dir = SCELLST_DIR / "reports" / "metrics" / f"{task}_fold{fold_idx}" / "mil"
    results = []

    for test_id in test_ids:
        logger.info(f"Predicting {task} fold {fold_idx}: test={test_id}")
        predict_and_save(
            model_dir,
            {"predict_id": test_id, "data_dir": path_dataset},
            infer_mode="bag",
            compute_metrics=True,
            save_adata=True,
        )
        metrics_path = metrics_dir / f"{exp_tag};test_slide={test_id};infer_mode=bag.csv"
        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_path}")
            continue
        results.append(pd.read_csv(metrics_path))

    return results


def fold_pearson(metrics_df: pd.DataFrame) -> float:
    """Mean Pearson (pcc) across the 50 genes from sCellST metrics CSV."""
    pcc_cols = [c for c in metrics_df.columns if c.startswith("pcc/")]
    if len(pcc_cols) == 0:
        logger.warning("No pcc/* columns found in metrics; check metrics format.")
        return float("nan")
    return metrics_df[pcc_cols].values.mean()


def patch_spot_in_embeddings(path_dataset: Path, all_ids: list[str]) -> None:
    """Add 'spot' assignments to embedding H5 files that have an empty spot field.

    The cell-image H5 files were generated with an older API that omitted the
    'spot' column, so the downstream embedding H5 files have spot (0, 1) rather
    than (N, 1).  This function computes Visium-spot → cell assignments via KDTree
    and writes them in-place so create_spot_cell_map can build the MIL bags.
    """
    import h5py
    import scanpy as sc
    from scipy.spatial import KDTree

    emb_dir = path_dataset / "cell_embeddings"
    st_dir = path_dataset / "st"

    for sid in all_ids:
        adata_path = st_dir / f"{sid}.h5ad"
        if not adata_path.exists():
            logger.warning(f"No adata at {adata_path}, skipping spot patch for {sid}.")
            continue

        adata = sc.read_h5ad(adata_path)
        spot_positions = adata.obsm["spatial"]   # (M, 2) full-res pixel coords
        spot_names = np.array(adata.obs_names.tolist())

        sf = {}
        for v in adata.uns.get("spatial", {}).values():
            sf = v.get("scalefactors", {})
            break
        spot_radius = sf.get("spot_diameter_fullres", 258.8) / 2.0

        tree = KDTree(spot_positions)

        if not emb_dir.exists():
            logger.warning(f"Embedding dir not found: {emb_dir}. Run embedding step first.")
            break
        for emb_path in sorted(emb_dir.glob(f"*_{sid}_*.h5")):
            with h5py.File(emb_path, "r") as f:
                if "spot" in f and f["spot"].shape[0] > 0:
                    continue  # already patched
                coords_topleft = f["coords"][:]  # (N, 2)
            # Cell image patches are 72×72 px; centroid = top-left + 36
            half_patch = 36
            centroids = coords_topleft.astype(float) + half_patch
            dists, idxs = tree.query(centroids, k=1, workers=-1)
            spot_labels = np.where(dists <= spot_radius, spot_names[idxs], "None")
            max_len = max((len(s) for s in spot_labels), default=4)
            spot_arr = spot_labels.astype(f"S{max_len}")
            with h5py.File(emb_path, "a") as f:
                if "spot" in f:
                    del f["spot"]
                f.create_dataset("spot", data=spot_arr)
            n_assigned = int((spot_labels != "None").sum())
            logger.info(
                f"Patched spot → {emb_path.name}: "
                f"{n_assigned}/{len(spot_labels)} cells assigned to spots."
            )


def append_to_summary(task: str, pearson_mean: float, pearson_std: float, n_folds: int) -> None:
    row = {
        "encoder": "scellst",
        "task": task,
        "pearson_mean": round(float(pearson_mean), 4),
        "pearson_std": round(float(pearson_std), 4),
        "n_folds": n_folds,
        "notes": "sCellST native MIL (imagenet-rn50 cell emb, 50 HVG HEST-Bench panel)",
    }
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        mask = (df["encoder"] == "scellst") & (df["task"] == task)
        if mask.any():
            df.loc[mask, list(row.keys())] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)
    logger.success(f"Updated summary.csv: {task} PCC={pearson_mean:.4f}±{pearson_std:.4f}")


def main() -> None:
    args = parse_args()
    task = args.task
    cfg = TASK_CONFIG[task]
    path_dataset = Path(args.data_dir) / task / "hest_data"

    logger.info(f"=== sCellST benchmark: task={task} ===")
    logger.info(f"HEST data dir: {path_dataset}")
    logger.info(f"Folds: {len(cfg['folds'])} (starting from {args.start_fold})")

    # 1. Download + convert
    if not args.skip_download:
        download_task(path_dataset, cfg["all_ids"])
    else:
        logger.info("Skipping download (--skip_download).")

    # 2. Cell embedding
    if not args.skip_embed:
        embed_task(path_dataset, cfg["all_ids"])
    else:
        logger.info("Skipping embedding (--skip_embed).")

    # 2b. Ensure embedding H5 files have 'spot' data (no-op if already present).
    # Older cell-image H5 files lack the 'spot' field; patch in-place via KDTree.
    patch_spot_in_embeddings(path_dataset, cfg["all_ids"])

    # 3. Train + predict each fold
    fold_pearson_values: list[float] = []
    for fold_idx, fold in enumerate(cfg["folds"]):
        if fold_idx < args.start_fold:
            logger.info(f"Skipping fold {fold_idx} (--start_fold={args.start_fold}).")
            continue
        model_dir, internal_fold = train_fold(path_dataset, task, fold_idx, fold)
        metrics_dfs = predict_fold(model_dir, path_dataset, task, fold_idx, internal_fold, fold["test"])
        if not metrics_dfs:
            logger.warning(f"Fold {fold_idx}: no metrics produced, skipping.")
            continue
        # Average Pearson across all test slides in this fold
        pcc = float(np.mean([fold_pearson(df) for df in metrics_dfs]))
        fold_pearson_values.append(pcc)
        logger.info(f"Fold {fold_idx} PCC = {pcc:.4f} (avg over {len(metrics_dfs)} test slide(s))")

    if len(fold_pearson_values) > 0:
        mean_pcc = float(np.mean(fold_pearson_values))
        std_pcc = float(np.std(fold_pearson_values))
        logger.success(
            f"{task}: mean PCC = {mean_pcc:.4f} ± {std_pcc:.4f} over {len(fold_pearson_values)} folds"
        )
        append_to_summary(task, mean_pcc, std_pcc, len(fold_pearson_values))
    else:
        logger.warning("No folds completed — nothing written to summary.csv.")


if __name__ == "__main__":
    main()
