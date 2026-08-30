"""Foundation-model + per-gene ridge baseline on the Xenium fixed split.

Loads per-slide HDF5 embeddings produced by
``scripts/extract_foundation_features_xenium.py``, joins them with the
matched Xenium transcript counts, fits a closed-form multi-output ridge on
the fixed-split train set, and writes predictions in the same layout that
``scripts/analyze_predictions.py`` consumes.

One model × one alpha × one split combination produces::

    <out_root>/<model_tag>/alpha_<a>/eval/<split>/
        mu.npy               float16  (n_cells, G)
        x_true.npy           int16    (n_cells, G)
        slide_idx.npy        int32    (n_cells,)
        slide_names.json     list[str]  -- global slide order
        metadata.json        genes / ridge params / counts

Everything else (per-cell Pearson, per-gene Pearson, per-slide breakdown)
flows through the existing analysis script without modification.

Usage::

    python scripts/baseline_foundation_predict.py \
        --features_root runs/baselines/foundation/features \
        --gene_dir data/gene_expression \
        --split_file splits/xenium_slide_split_v1.json \
        --out_root runs/baselines/foundation \
        --models phikon2 prov-gigapath uni2-h \
        --alphas 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
for extra in (MODEL_ROOT, REPO_ROOT):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from main import load_slide_split  # noqa: E402 (imports model/main.py)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features_root", type=Path,
                    default=REPO_ROOT / "runs/baselines/foundation/features")
    ap.add_argument("--gene_dir", type=Path,
                    default=REPO_ROOT / "data/gene_expression")
    ap.add_argument("--img_dir", type=Path,
                    default=REPO_ROOT / "data/images",
                    help="Used only to reproduce the 'slides with matched "
                         "images' set (same logic as XeniumCellDataset) so "
                         "the split file's completeness check passes")
    ap.add_argument("--split_file", type=Path,
                    default=REPO_ROOT / "splits/xenium_slide_split_v1.json")
    ap.add_argument("--out_root", type=Path,
                    default=REPO_ROOT / "runs/baselines/foundation")
    ap.add_argument("--models", nargs="+",
                    default=["phikon2", "prov-gigapath", "uni2-h"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--chunk_rows", type=int, default=200_000,
                    help="Rows processed per GEMM chunk during fit")
    return ap.parse_args()


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------


@dataclass
class SplitFeatures:
    mu: np.ndarray          # not filled here; this struct holds X
    cell_id: list[str]      # n_cells
    slide_idx: np.ndarray   # (n_cells,) — global slide index
    features: np.ndarray    # (n_cells, D)
    counts: np.ndarray      # (n_cells, G)


def _model_tag(name: str) -> str:
    return name.upper().replace("-", "_")


def load_split_features(
    model_name: str,
    features_root: Path,
    gene_dir: Path,
    slide_names: list[str],
    slide_ids_in_split: list[str],
    genes: list[str],
) -> SplitFeatures:
    """Assemble (features, counts, cell_ids, slide_idx) for one split's slides."""
    tag = _model_tag(model_name)

    all_feats: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_slide_idx: list[int] = []
    all_cell_ids: list[str] = []

    slide_name_to_idx = {n: i for i, n in enumerate(slide_names)}

    for slide in slide_ids_in_split:
        h5_fp = features_root / tag / f"{slide}.h5"
        if not h5_fp.exists():
            raise FileNotFoundError(
                f"Missing foundation features for slide {slide}: {h5_fp}"
            )

        with h5py.File(h5_fp, "r") as f:
            embed = f["embeddings"][:].astype(np.float32)
            cell_ids = [c.decode("ascii") for c in f["cell_ids"][:]]

        h5ad_fp = gene_dir / f"{slide}.h5ad"
        if not h5ad_fp.exists():
            raise FileNotFoundError(f"Missing expression file: {h5ad_fp}")
        adata = ad.read_h5ad(h5ad_fp)

        # Restrict to the shared gene order and to the cells that have
        # image features.
        adata_sub = adata[:, genes]
        obs_to_row = {name: i for i, name in enumerate(adata_sub.obs_names)}

        keep_mask = np.zeros(len(cell_ids), dtype=bool)
        row_idx = np.zeros(len(cell_ids), dtype=np.int64)
        for i, cid in enumerate(cell_ids):
            if cid in obs_to_row:
                keep_mask[i] = True
                row_idx[i] = obs_to_row[cid]

        if not keep_mask.any():
            print(f"[warn] no matching cells for {slide}; dropping")
            continue

        X = (adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray")
             else np.asarray(adata_sub.X))
        counts = X[row_idx[keep_mask]].astype(np.float32)
        feats = embed[keep_mask]

        slide_idx_val = slide_name_to_idx[slide]
        all_feats.append(feats)
        all_counts.append(counts)
        all_slide_idx.extend([slide_idx_val] * feats.shape[0])
        all_cell_ids.extend(cid for cid, keep in zip(cell_ids, keep_mask) if keep)

        print(f"    {slide}: {feats.shape[0]} cells x {feats.shape[1]} dims")

    feats_arr = np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 0), dtype=np.float32)
    counts_arr = np.concatenate(all_counts, axis=0) if all_counts else np.empty((0, len(genes)), dtype=np.float32)
    slide_idx_arr = np.asarray(all_slide_idx, dtype=np.int32)

    return SplitFeatures(
        mu=np.empty(0),
        cell_id=all_cell_ids,
        slide_idx=slide_idx_arr,
        features=feats_arr,
        counts=counts_arr,
    )


def derive_gene_order(gene_dir: Path) -> list[str]:
    """Replicate the Xenium dataset's shared-gene logic."""
    expr_files = sorted(p for p in gene_dir.iterdir() if p.suffix == ".h5ad")
    shared = None
    for fp in expr_files:
        adata = ad.read_h5ad(fp, backed="r")
        s = set(adata.var_names)
        shared = s if shared is None else shared & s
    if not shared:
        raise RuntimeError(f"Could not derive shared gene list from {gene_dir}")
    return sorted(shared)


# ----------------------------------------------------------------------
# Chunked ridge
# ----------------------------------------------------------------------


def chunked_normal_equations(X: np.ndarray, Y: np.ndarray,
                              chunk_rows: int,
                              ) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    n, d = X.shape
    g = Y.shape[1]

    sum_x = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(g, dtype=np.float64)
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        sum_x += X[start:stop].astype(np.float64).sum(axis=0)
        sum_y += Y[start:stop].astype(np.float64).sum(axis=0)
    Xmean = (sum_x / n).astype(np.float32)
    Ymean = (sum_y / n).astype(np.float32)

    XTX = np.zeros((d, d), dtype=np.float64)
    XTY = np.zeros((d, g), dtype=np.float64)
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        xb = X[start:stop].astype(np.float32) - Xmean
        yb = Y[start:stop].astype(np.float32) - Ymean
        XTX += xb.T.astype(np.float64) @ xb.astype(np.float64)
        XTY += xb.T.astype(np.float64) @ yb.astype(np.float64)
        print(f"    accum {stop:,}/{n:,} rows "
              f"({100.0 * stop / n:.1f}%)")
    return XTX, XTY, Xmean, Ymean


def predict_chunked(X: np.ndarray, beta: np.ndarray,
                    Xmean: np.ndarray, Ymean: np.ndarray,
                    chunk_rows: int) -> np.ndarray:
    n = X.shape[0]
    g = beta.shape[1]
    out = np.empty((n, g), dtype=np.float16)
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        xb = X[start:stop].astype(np.float32) - Xmean
        pred = xb @ beta + Ymean
        out[start:stop] = pred.astype(np.float16)
    return out


# ----------------------------------------------------------------------
# Dump writer
# ----------------------------------------------------------------------


def write_eval_dump(out_dir: Path, mu: np.ndarray, x_true: np.ndarray,
                    slide_idx: np.ndarray, unique_slides: list[str],
                    genes: list[str], split_name: str,
                    extra_metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mu.npy", mu.astype(np.float16))
    np.save(out_dir / "x_true.npy", x_true.astype(np.int16))
    np.save(out_dir / "slide_idx.npy", slide_idx.astype(np.int32))
    with open(out_dir / "slide_names.json", "w") as f:
        json.dump(unique_slides, f, indent=2)
    meta = {
        "split_name": split_name,
        "n_cells": int(mu.shape[0]),
        "gene_dim": int(mu.shape[1]),
        "genes": genes,
        "model_type": "foundation_ridge_baseline",
    }
    meta.update(extra_metadata)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_for_model(model_name: str, args: argparse.Namespace,
                   genes: list[str], slide_names: list[str],
                   split_spec: dict) -> None:
    print(f"\n=== Foundation model: {model_name} ===")
    tag = _model_tag(model_name)
    out_root = args.out_root / tag
    out_root.mkdir(parents=True, exist_ok=True)

    # -- assemble train features ------------------------------------------------
    print("  Loading train features ...")
    train = load_split_features(
        model_name, args.features_root, args.gene_dir,
        slide_names, split_spec["train"], genes,
    )
    if train.features.size == 0:
        print(f"  [error] no training features for {model_name}; skipping")
        return
    print(f"  Train: {train.features.shape[0]:,} cells, D={train.features.shape[1]}, "
          f"G={train.counts.shape[1]}")

    print("  Accumulating normal equations ...")
    XTX, XTY, Xmean, Ymean = chunked_normal_equations(
        train.features, train.counts, chunk_rows=args.chunk_rows,
    )
    d = XTX.shape[0]

    # Free the train matrix now that normal equations are built.
    train_feats_shape = train.features.shape
    train_counts_shape = train.counts.shape
    del train

    # -- one ridge solve per alpha -----------------------------------------------
    for alpha in args.alphas:
        print(f"\n  -- alpha = {alpha:g} --")
        A = XTX + alpha * np.eye(d)
        beta = np.linalg.solve(A, XTY).astype(np.float32)
        Xmean_f = Xmean.astype(np.float32)
        Ymean_f = Ymean.astype(np.float32)

        alpha_dir = out_root / f"alpha_{alpha:g}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        np.save(alpha_dir / "ridge_beta.npy", beta)
        np.save(alpha_dir / "ridge_Xmean.npy", Xmean_f)
        np.save(alpha_dir / "ridge_Ymean.npy", Ymean_f)

        for split_name in args.splits:
            slides = split_spec.get(split_name, [])
            if not slides:
                continue
            print(f"    predicting {split_name} ({len(slides)} slides) ...")
            sub = load_split_features(
                model_name, args.features_root, args.gene_dir,
                slide_names, slides, genes,
            )
            if sub.features.size == 0:
                print(f"    [warn] {split_name} has no cells; skipping")
                continue
            mu = predict_chunked(sub.features, beta, Xmean_f, Ymean_f,
                                  chunk_rows=args.chunk_rows)
            dump_dir = alpha_dir / "eval" / split_name
            write_eval_dump(
                dump_dir, mu, sub.counts.astype(np.int16), sub.slide_idx,
                unique_slides=slide_names, genes=genes,
                split_name=split_name,
                extra_metadata={
                    "model_name": model_name,
                    "alpha": alpha,
                    "split_source": str(args.split_file.resolve()),
                    "n_train_cells": int(train_feats_shape[0]),
                    "feat_dim": int(d),
                },
            )
            print(f"    wrote {dump_dir}  mu shape={mu.shape}  "
                  f"x_true shape={sub.counts.shape}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    print("Deriving gene order from Xenium training data ...")
    genes = derive_gene_order(args.gene_dir)
    gene_dim = len(genes)
    print(f"  {gene_dim} shared genes")

    # Build the global slide name index.  Mirror XeniumCellDataset: only
    # keep slides that have both an ``.h5ad`` and a matching
    # ``data/images/<slide>/`` directory.  The leakage-safe split file was
    # constructed against exactly this intersection (29 slides), so we
    # need to reproduce it here for ``load_slide_split`` to accept the
    # spec.  The prior version used every h5ad (32) which included three
    # image-less slides and caused the "Split file is missing dataset
    # slides" error at the end of Stage 1.
    h5ad_slides = {p.stem for p in args.gene_dir.glob("*.h5ad")}
    image_slides = {p.name for p in args.img_dir.iterdir() if p.is_dir()}
    all_slides = sorted(h5ad_slides & image_slides)
    print(f"  {len(all_slides)} matched slides "
          f"(h5ads={len(h5ad_slides)}, image dirs={len(image_slides)})")

    split_spec = load_slide_split(args.split_file, all_slides)
    for split_name in ("train", "val", "test"):
        n = len(split_spec.get(split_name, []))
        print(f"  split {split_name}: {n} slides")

    for model_name in args.models:
        run_for_model(model_name, args, genes, all_slides, split_spec)


if __name__ == "__main__":
    main()
