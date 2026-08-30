"""
Prepare HEST benchmark data in SciSt format for one task.

Reads the HEST H5 patch files (already 224×224) and the matching .h5ad for
expression, then writes three directories expected by SciSt's dataset loader:

    <out_dir>/01_patch/<sample_id>_<idx>.png
    <out_dir>/02_label/<sample_id>.json      # {spot_key: [50 floats]}
    <out_dir>/06-noise_exp/<sample_id>_<idx>.npy  # zeros, shape (50,)

noise_exp is set to zeros because HEST/Visium data has no single-cell
segmentation reference; this is the honest image-only baseline.

Usage:
    python scripts/hest_bench/prep_scist_hest.py \
        --task IDC \
        --data_dir /work/OSPH/s439765/scellst \
        --out_dir  /work/OSPH/s439765/scist \
        --gene_dir /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/benchmark/sCellST/data
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from loguru import logger

TASK_IDS: dict[str, list[str]] = {
    "IDC":  ["TENX95", "TENX99", "NCBI783", "NCBI785"],
    "PAAD": ["TENX116", "TENX126", "TENX140"],
    "SKCM": ["TENX115", "TENX117"],
    "COAD": ["TENX111", "TENX147", "TENX148", "TENX149"],
    "LUNG": ["TENX118", "TENX141"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_IDS.keys()))
    p.add_argument("--data_dir", default="/work/OSPH/s439765/scellst",
                   help="Root dir of HEST downloads (from sCellST pipeline).")
    p.add_argument("--out_dir", default="/work/OSPH/s439765/scist",
                   help="Root dir to write SciSt-formatted data.")
    p.add_argument(
        "--gene_dir",
        default=(
            "/endosome/archive/DPDS/Xiao_lab/shared/jia_yao/"
            "Image2Transcript/benchmark/sCellST/data"
        ),
        help="Dir containing genes_<TASK>_50hvg_bench.csv files.",
    )
    return p.parse_args()


def load_hvgs(gene_dir: str, task: str) -> list[str]:
    path = Path(gene_dir) / f"genes_{task}_50hvg_bench.csv"
    df = pd.read_csv(path)
    return list(df["gene"])


def normalize_adata(adata):
    """Library-size normalise then log1p (standard scRNA-seq preprocessing)."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def prep_sample(
    sample_id: str,
    hest_dir: Path,
    out_dir: Path,
    hvgs: list[str],
) -> int:
    patch_h5 = hest_dir / "patches" / f"{sample_id}.h5"
    st_h5ad = hest_dir / "st" / f"{sample_id}.h5ad"

    if not patch_h5.exists():
        logger.error(f"Patches not found: {patch_h5}")
        return 0
    if not st_h5ad.exists():
        logger.error(f"ST data not found: {st_h5ad}")
        return 0

    # Load patches
    with h5py.File(patch_h5, "r") as f:
        barcodes = f["barcode"][:].flatten()
        barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in barcodes]
        imgs = f["img"][:]  # (N, 224, 224, 3) uint8

    # Load + normalise expression
    adata = sc.read_h5ad(st_h5ad)
    adata = normalize_adata(adata)
    adata = adata[:, hvgs]

    # Build barcode → expression lookup
    bc_to_expr: dict[str, list[float]] = {
        bc: adata[bc].X.flatten().tolist()
        for bc in barcodes
        if bc in adata.obs_names
    }
    matched = len(bc_to_expr)
    if matched == 0:
        logger.error(f"{sample_id}: no barcodes matched between patches and h5ad")
        return 0

    logger.info(f"{sample_id}: {matched}/{len(barcodes)} patches matched to expression")

    patch_dir = out_dir / "01_patch"
    label_dir = out_dir / "02_label"
    noise_dir = out_dir / "06-noise_exp"
    for d in (patch_dir, label_dir, noise_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_genes = len(hvgs)
    zero_noise = np.zeros(num_genes, dtype=np.float32)

    label_dict: dict[str, list[float]] = {}
    saved = 0

    for idx, (bc, img_arr) in enumerate(zip(barcodes, imgs)):
        if bc not in bc_to_expr:
            continue
        spot_key = f"{sample_id}_{idx}"
        png_path = patch_dir / f"{spot_key}.png"
        npy_path = noise_dir / f"{spot_key}.npy"

        # Save patch image (convert from uint8 HWC)
        Image.fromarray(img_arr).save(png_path)

        # Save zero noise_exp
        np.save(npy_path, zero_noise)

        label_dict[spot_key] = bc_to_expr[bc]
        saved += 1

    # Save label JSON for this sample
    label_path = label_dir / f"{sample_id}.json"
    with open(label_path, "w") as fh:
        json.dump(label_dict, fh)

    logger.success(f"{sample_id}: saved {saved} patches / labels / noise_exp")
    return saved


def main() -> None:
    args = parse_args()
    hvgs = load_hvgs(args.gene_dir, args.task)
    logger.info(f"Task={args.task} | {len(hvgs)} HVGs")

    hest_dir = Path(args.data_dir) / args.task / "hest_data"
    out_dir = Path(args.out_dir) / args.task

    total = 0
    for sample_id in TASK_IDS[args.task]:
        total += prep_sample(sample_id, hest_dir, out_dir, hvgs)

    logger.success(f"Done. Total spots saved: {total}")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
