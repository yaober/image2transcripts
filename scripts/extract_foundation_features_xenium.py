"""Extract pathology-foundation-model embeddings for every Xenium cell crop.

Companion to ``feature_extraction.py`` (which operates on TCGA WSIs) but
tailored to the Xenium training crops under ``data/images/<slide>/*.png``.
Each slide becomes one HDF5 file per foundation model, ordered identically
to the dataset's per-slide alphabetical scan so a downstream ridge fit can
cheaply line up features with transcript counts.

Output layout::

    <out_dir>/<MODEL_TAG>/<slide_id>.h5
        embeddings   (N_cells, D)        float16
        cell_ids     (N_cells,)          bytes (PNG stem)
        model_name, slide_id, mpp_note   metadata

A sibling ``<out_dir>/cells_per_slide.json`` records the cell order so
predictors can reconstruct the (slide, cell) index set.

Usage::

    HF_TOKEN=<hf_token> python scripts/extract_foundation_features_xenium.py \
        --img_dir data/images \
        --out_dir runs/baselines/foundation/features \
        --models phikon2 prov-gigapath uni2-h \
        --batch 128 --num_workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from foundation_models import (  # noqa: E402
    ALL_MODEL_NAMES,
    MODEL_INPUT_SIZE,
    expected_embed_dim,
    load_all_models,
    run_inference_on_patches,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img_dir", type=Path,
                    default=REPO_ROOT / "data/images")
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "runs/baselines/foundation/features")
    ap.add_argument("--models", nargs="+", default=ALL_MODEL_NAMES,
                    choices=["phikon", "phikon2", "prov-gigapath",
                              "uni", "uni2-h"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_slides", type=int, default=0,
                    help="Cap the slide count for smoke tests (0 = all)")
    ap.add_argument("--limit_cells_per_slide", type=int, default=0,
                    help="Cap cells per slide for smoke tests (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess slides whose HDF5 already exists")
    return ap.parse_args()


def _model_tag(name: str) -> str:
    return name.upper().replace("-", "_")


def load_slide_patches(
    slide_dir: Path, limit: int = 0,
) -> tuple[list[str], list[np.ndarray]]:
    """Load every PNG crop under ``slide_dir`` as a ``(H, W, 3)`` uint8 array.

    We resize to the foundation-model's 224×224 input; the Xenium crops are
    natively 188×188 so this is a small bilinear upscale.  Returns the cell
    ids in the same order as the patches so the caller can persist them.
    """
    png_files = sorted(slide_dir.glob("*.png"))
    if limit > 0:
        png_files = png_files[:limit]

    cell_ids: list[str] = []
    patches: list[np.ndarray] = []
    for fp in png_files:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as exc:
            print(f"    [warn] skip {fp.name}: {exc}")
            continue
        if img.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
            img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                              Image.BILINEAR)
        patches.append(np.asarray(img, dtype=np.uint8))
        cell_ids.append(fp.stem)
    return cell_ids, patches


def save_slide_h5(out_fp: Path, embeddings: np.ndarray,
                  cell_ids: list[str], model_name: str,
                  slide_id: str) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_fp, "w") as f:
        f.create_dataset("embeddings", data=embeddings.astype(np.float16),
                         compression="gzip", compression_opts=4)
        f.create_dataset(
            "cell_ids",
            data=np.asarray([c.encode("ascii") for c in cell_ids]),
        )
        f.attrs["model_name"] = model_name
        f.attrs["slide_id"] = slide_id
        f.attrs["n_cells"] = len(cell_ids)
        f.attrs["embed_dim"] = int(embeddings.shape[1])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    slide_dirs = sorted(p for p in args.img_dir.iterdir() if p.is_dir())
    if args.limit_slides > 0:
        slide_dirs = slide_dirs[:args.limit_slides]
    print(f"Found {len(slide_dirs)} slides under {args.img_dir}")

    # Record the intended per-slide cell ordering up front so downstream
    # analysis can reconstruct a consistent (slide, cell) → feature-row
    # mapping even if a single slide fails mid-extraction.
    cells_per_slide: dict[str, dict] = {}

    print(f"Loading foundation models: {args.models}")
    t0 = time.time()
    models = load_all_models(args.models, device)
    print(f"Model load time: {time.time() - t0:.1f}s")

    for si, slide_dir in enumerate(slide_dirs, start=1):
        slide_id = slide_dir.name

        # Decide which models still need to be run on this slide; ``pending``
        # maps model name -> output HDF5 path (for writing), while
        # ``pending_models`` maps model name -> the loaded nn.Module (for
        # inference).  Keeping the two dicts separate avoids the bug where a
        # Path accidentally reaches ``run_inference_on_patches``.
        pending: dict[str, Path] = {}
        for name in args.models:
            out_fp = args.out_dir / _model_tag(name) / f"{slide_id}.h5"
            if out_fp.exists() and not args.force:
                continue
            pending[name] = out_fp
        pending_models = {name: models[name] for name in pending}

        if not pending:
            print(f"[{si}/{len(slide_dirs)}] {slide_id}: all outputs exist, skipping")
            # Still populate cells_per_slide from the first existing file so
            # the index stays complete.
            first_model = next(iter(args.models))
            first_fp = args.out_dir / _model_tag(first_model) / f"{slide_id}.h5"
            if first_fp.exists():
                with h5py.File(first_fp, "r") as f:
                    cell_ids = [c.decode("ascii") for c in f["cell_ids"][:]]
                cells_per_slide[slide_id] = {
                    "n_cells": len(cell_ids), "model_tag_any": _model_tag(first_model),
                }
            continue

        print(f"[{si}/{len(slide_dirs)}] {slide_id}: loading cells ...")
        t1 = time.time()
        cell_ids, patches = load_slide_patches(slide_dir, args.limit_cells_per_slide)
        print(f"    {len(cell_ids)} cells loaded in {time.time() - t1:.1f}s")
        cells_per_slide[slide_id] = {"n_cells": len(cell_ids)}
        if not patches:
            print(f"    [warn] no patches in {slide_id}; skipping")
            continue

        print(f"    running inference on {len(pending_models)} model(s) ...")
        t2 = time.time()
        embs = run_inference_on_patches(
            pending_models, patches, batch_size=args.batch,
            num_workers=args.num_workers, device=device,
            show_progress=True,
            progress_desc=f"    {slide_id}",
        )
        print(f"    inference time: {time.time() - t2:.1f}s")

        for name, out_fp in pending.items():
            emb = embs[name]
            expected = expected_embed_dim(name)
            if expected is not None and emb.shape[1] != expected:
                print(f"    [warn] {name} produced D={emb.shape[1]} "
                      f"(expected {expected})")
            save_slide_h5(out_fp, emb, cell_ids, name, slide_id)
            print(f"    wrote {out_fp}  shape={emb.shape}")
        # Free the patch list before the next slide.
        del patches

    index_fp = args.out_dir / "cells_per_slide.json"
    with open(index_fp, "w") as f:
        json.dump(cells_per_slide, f, indent=2)
    print(f"\nWrote slide index: {index_fp}")


if __name__ == "__main__":
    main()
