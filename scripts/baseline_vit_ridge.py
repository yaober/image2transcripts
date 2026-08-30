"""Frozen ViT + per-gene ridge baseline on the leakage-safe fixed split.

Runs a two-phase pipeline:

1. *Feature extraction.* Loads a frozen, ImageNet-pretrained
   ``vit_base_patch16_224`` backbone from timm (or a user-supplied ViT
   checkpoint), passes each cell-centered crop through it, and caches the
   CLS-token features to ``--features_dir``.  Features are stored per split
   (``train``/``val``/``test``) as ``cls.npy`` (float16) together with the
   raw gene counts ``x_true.npy`` (int16) and slide metadata so the fit
   phase can run purely offline.

2. *Ridge fit.* Reads the train features, fits a multi-output ridge via the
   normal equations solved in chunked fashion, and applies it to val/test.
   The resulting ``mu.npy`` / ``x_true.npy`` / ``slide_idx.npy`` /
   ``slide_names.json`` / ``metadata.json`` triple lives under
   ``--out_dir/eval/<split>/`` and can be consumed directly by
   ``scripts/analyze_predictions.py``.

Either phase can be run alone with ``--phase {extract,fit,all}``; ``all``
does both.  Features are reused across ``--alphas`` so a list of ridge
strengths only pays the feature-extraction cost once.

Example::

    python scripts/baseline_vit_ridge.py \
        --gene_dir data/gene_expression --img_dir data/images \
        --split_file splits/xenium_slide_split_v1.json \
        --features_dir runs/baselines/vit_ridge/features \
        --out_dir runs/baselines/vit_ridge \
        --alphas 1.0 10.0 100.0

    python scripts/analyze_predictions.py \
        --eval_dir runs/baselines/vit_ridge/alpha_1.0/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# See scripts/eval_fixedsplit.py for the sys.path justification; the
# ``model/`` directory is not a package.
MODEL_ROOT = REPO_ROOT / "model"
for extra in (MODEL_ROOT, REPO_ROOT):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T2
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
from timm.models import load_checkpoint

from main import XeniumCellDataset, load_slide_split  # noqa: E402


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_dir", type=Path, default=Path("data/gene_expression"))
    ap.add_argument("--img_dir", type=Path, default=Path("data/images"))
    ap.add_argument("--split_file", type=Path,
                    default=Path("splits/xenium_slide_split_v1.json"))
    ap.add_argument("--features_dir", type=Path,
                    default=Path("runs/baselines/vit_ridge/features"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/baselines/vit_ridge"))
    ap.add_argument("--phase", choices=["extract", "fit", "all"], default="all")

    ap.add_argument("--vit_model", default="vit_base_patch16_224")
    ap.add_argument("--vit_ckpt", default=None,
                    help="Optional local ViT checkpoint path; if unset uses timm pretrained")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true",
                    help="Run feature extraction in fp32 (default is autocast fp16)")

    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0],
                    help="Ridge regularization strengths to sweep")
    ap.add_argument("--chunk_rows", type=int, default=200_000,
                    help="Rows processed per GEMM chunk during fit")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])

    return ap.parse_args()


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------


def make_transform() -> T2.Compose:
    return T2.Compose([
        T2.Resize((224, 224), antialias=True),
        T2.ToDtype(torch.float32, scale=True),
    ])


def load_dataset(gene_dir: Path, img_dir: Path) -> XeniumCellDataset:
    print(f"Loading dataset from {gene_dir} + {img_dir}")
    return XeniumCellDataset(gene_dir, img_dir, transform=make_transform(),
                              gene_mask_ratio=0.0)


def resolve_splits(ds: XeniumCellDataset, split_file: Path,
                   requested: list[str]) -> tuple[dict, list[str], np.ndarray]:
    slide_ids_per_cell = np.asarray(ds.slide_ids_per_cell)
    unique_slides = sorted(np.unique(slide_ids_per_cell).tolist())
    spec = load_slide_split(split_file, unique_slides)

    out: dict[str, list[int]] = {}
    for name in requested:
        slides = set(spec.get(name, []))
        if not slides:
            continue
        idx = [i for i, s in enumerate(slide_ids_per_cell) if s in slides]
        if idx:
            out[name] = idx
    return out, unique_slides, slide_ids_per_cell


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------


def build_frozen_backbone(vit_model: str, vit_ckpt: str | None,
                          device: torch.device) -> nn.Module:
    pretrained = vit_ckpt is None
    model = timm.create_model(vit_model, pretrained=pretrained)
    if vit_ckpt:
        load_checkpoint(model, vit_ckpt, strict=False)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
    else:
        model.head = nn.Identity()
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_cls_features(model: nn.Module, loader: DataLoader,
                          device: torch.device, use_autocast: bool,
                          n_cells: int, feat_dim: int) -> np.ndarray:
    out = np.empty((n_cells, feat_dim), dtype=np.float16)
    cursor = 0
    t0 = time.time()
    printed_every = max(1, len(loader) // 40)

    for step, (img, _feat, _mask) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        if use_autocast:
            with autocast(device_type="cuda"):
                feats = _vit_cls_forward(model, img)
        else:
            feats = _vit_cls_forward(model, img)
        bs = img.size(0)
        out[cursor:cursor + bs] = feats.float().cpu().numpy().astype(np.float16)
        cursor += bs

        if (step + 1) % printed_every == 0 or (step + 1) == len(loader):
            elapsed = time.time() - t0
            rate = cursor / max(1e-6, elapsed)
            print(f"    step {step + 1}/{len(loader)}  "
                  f"cells {cursor:,}/{n_cells:,}  "
                  f"{rate:,.0f} cells/s  elapsed {elapsed:,.0f}s")

    assert cursor == n_cells
    return out


def _vit_cls_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    B = image.size(0)
    x = model.patch_embed(image)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = model.pos_drop(x + model.pos_embed[:, :x.size(1), :])
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x[:, 0]


def run_extract_phase(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = load_dataset(args.gene_dir, args.img_dir)
    split_idx, unique_slides, slide_ids_per_cell = resolve_splits(
        ds, args.split_file, args.splits
    )
    gene_dim = len(ds.shared_genes)
    slide_name_to_idx = {name: i for i, name in enumerate(unique_slides)}

    model = build_frozen_backbone(args.vit_model, args.vit_ckpt, device)
    feat_dim = getattr(model, "num_features",
                        getattr(model, "embed_dim"))

    args.features_dir.mkdir(parents=True, exist_ok=True)
    with open(args.features_dir / "unique_slides.json", "w") as f:
        json.dump(unique_slides, f, indent=2)
    with open(args.features_dir / "genes.json", "w") as f:
        json.dump(ds.shared_genes, f, indent=2)

    for split_name, idx in split_idx.items():
        split_dir = args.features_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        cls_fp = split_dir / "cls.npy"
        x_fp = split_dir / "x_true.npy"
        slide_fp = split_dir / "slide_idx.npy"
        meta_fp = split_dir / "metadata.json"

        if cls_fp.exists() and x_fp.exists() and slide_fp.exists():
            print(f"[skip] features already exist for {split_name}: {cls_fp}")
            continue

        print(f"\n=== Extracting features for split: {split_name} "
              f"({len(idx):,} cells) ===")
        subset = Subset(ds, idx)
        loader = DataLoader(subset, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        features = extract_cls_features(
            model, loader, device,
            use_autocast=not args.fp32,
            n_cells=len(idx), feat_dim=feat_dim,
        )

        np.save(cls_fp, features)

        x_true = np.empty((len(idx), gene_dim), dtype=np.int16)
        for pos, dataset_idx in enumerate(idx):
            x_true[pos] = ds.cell_exprs[dataset_idx].round().astype(np.int16)
        np.save(x_fp, x_true)

        slide_arr = np.asarray(
            [slide_name_to_idx[s] for s in slide_ids_per_cell[idx]],
            dtype=np.int32,
        )
        np.save(slide_fp, slide_arr)

        with open(meta_fp, "w") as f:
            json.dump({
                "split": split_name,
                "n_cells": len(idx),
                "feat_dim": int(feat_dim),
                "gene_dim": gene_dim,
                "vit_model": args.vit_model,
                "vit_ckpt": args.vit_ckpt,
                "split_source": str(args.split_file.resolve()),
            }, f, indent=2)


# ----------------------------------------------------------------------
# Ridge fit
# ----------------------------------------------------------------------


def chunked_normal_equations(X: np.ndarray, Y: np.ndarray,
                              chunk_rows: int) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """Return XTX, XTY, Xmean, Ymean from streaming ``X`` and ``Y``.

    ``X`` and ``Y`` are mmap-ed numpy arrays; chunks are cast to fp32 before
    the dot product.  Means are subtracted inside the chunk loop so the ridge
    is effectively fit to centered predictors and centered targets.
    """
    n, d = X.shape
    g = Y.shape[1]

    # Means (two passes: one accumulates sums, one centers).  We accumulate in
    # fp64 to keep numerical error low for the millions of rows we handle.
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

    return XTX.astype(np.float64), XTY.astype(np.float64), Xmean, Ymean


def predict_chunked(X: np.ndarray, beta: np.ndarray,
                    Xmean: np.ndarray, Ymean: np.ndarray,
                    chunk_rows: int) -> np.ndarray:
    n, d = X.shape
    g = beta.shape[1]
    out = np.empty((n, g), dtype=np.float16)
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        xb = X[start:stop].astype(np.float32) - Xmean
        pred = xb @ beta + Ymean
        out[start:stop] = pred.astype(np.float16)
    return out


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
        "model_type": "vit_ridge_baseline",
    }
    meta.update(extra_metadata)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_fit_phase(args: argparse.Namespace) -> None:
    feat_root = args.features_dir
    train_dir = feat_root / "train"
    required = ["cls.npy", "x_true.npy", "slide_idx.npy"]
    for split_name in args.splits:
        sd = feat_root / split_name
        for fn in required:
            if not (sd / fn).exists():
                raise FileNotFoundError(
                    f"Missing {sd / fn}; run the extract phase first"
                )

    with open(feat_root / "unique_slides.json") as f:
        unique_slides = json.load(f)
    with open(feat_root / "genes.json") as f:
        genes = json.load(f)

    print("\n=== Loading train features (memory-mapped) ===")
    X_train = np.load(train_dir / "cls.npy", mmap_mode="r")
    Y_train = np.load(train_dir / "x_true.npy", mmap_mode="r")
    print(f"  X_train shape: {X_train.shape}  Y_train shape: {Y_train.shape}")

    print("\n=== Accumulating normal equations ===")
    XTX, XTY, Xmean, Ymean = chunked_normal_equations(
        X_train, Y_train, chunk_rows=args.chunk_rows,
    )
    d = XTX.shape[0]

    for alpha in args.alphas:
        print(f"\n=== Solving ridge with alpha={alpha} ===")
        A = XTX + alpha * np.eye(d)
        beta = np.linalg.solve(A, XTY).astype(np.float32)
        Xmean_f = Xmean.astype(np.float32)
        Ymean_f = Ymean.astype(np.float32)

        alpha_dir = args.out_dir / f"alpha_{alpha:g}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        np.save(alpha_dir / "ridge_beta.npy", beta)
        np.save(alpha_dir / "ridge_Xmean.npy", Xmean_f)
        np.save(alpha_dir / "ridge_Ymean.npy", Ymean_f)

        for split_name in args.splits:
            print(f"  predicting {split_name}")
            split_dir = feat_root / split_name
            Xsplit = np.load(split_dir / "cls.npy", mmap_mode="r")
            Ysplit = np.load(split_dir / "x_true.npy", mmap_mode="r")
            slide_idx = np.load(split_dir / "slide_idx.npy")

            pred = predict_chunked(Xsplit, beta, Xmean_f, Ymean_f,
                                   chunk_rows=args.chunk_rows)

            dump_dir = alpha_dir / "eval" / split_name
            write_eval_dump(
                dump_dir, pred, Ysplit, slide_idx,
                unique_slides=unique_slides, genes=genes,
                split_name=split_name,
                extra_metadata={
                    "alpha": alpha,
                    "vit_model": args.vit_model,
                    "vit_ckpt": args.vit_ckpt,
                    "split_source": str(args.split_file.resolve()),
                    "n_train_cells": int(X_train.shape[0]),
                    "feat_dim": int(d),
                },
            )
            print(f"    wrote {dump_dir}")


def main() -> None:
    args = parse_args()

    if args.phase in ("extract", "all"):
        run_extract_phase(args)
    if args.phase in ("fit", "all"):
        run_fit_phase(args)


if __name__ == "__main__":
    main()
