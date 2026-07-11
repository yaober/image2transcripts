"""Single-GPU inference that dumps per-cell model outputs for val and test.

The script reuses XeniumCellDataset from ``model/main.py`` with augmentation
disabled and gene masking set to zero.  For a given ``--run_dir`` containing
``best_model.pt`` and ``slide_split_used.json`` it produces, per split
(``val`` and ``test``), a directory with:

    mu.npy          float16  (n_cells, G)   predicted ZINB mean per gene
    theta.npy       float16  (n_cells, G)   predicted ZINB dispersion
    pi_logits.npy   float16  (n_cells, G)   predicted zero-inflation logits
    i_emb.npy       float16  (n_cells, D)   L2-normalized image embedding
    g_emb.npy       float16  (n_cells, D)   L2-normalized gene embedding
    x_true.npy      int16    (n_cells, G)   observed transcript counts
    slide_idx.npy   int32    (n_cells,)     mapping into slide_names.json
    slide_names.json                       list[str] global slide order
    cell_order.npy  int32    (n_cells,)     dataset-global cell indices
    metadata.json                          gene list, run dir, split info

A companion ``predictions_index.json`` at ``<run_dir>/eval/`` keeps track of
which splits have been dumped.

Usage::

    python scripts/eval_fixedsplit.py \
        --run_dir runs/fixedsplit_v3/full_seed42 \
        --gene_dir data/gene_expression \
        --img_dir data/images \
        --split_file splits/xenium_slide_split_v1.json \
        --batch 256 --num_workers 6

Pass ``--splits val test`` (default) or ``--splits val`` to limit which
subsets are written.  ``--subset_cells N`` caps each split at N cells
(stratified by slide) which is useful for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The ``model/`` folder is not a Python package (no __init__.py); mirror the
# sibling-import convention used by ``model/main.py`` itself by adding
# ``model/`` directly to sys.path so ``import main`` and ``import model``
# resolve to ``model/main.py`` and ``model/model.py`` respectively.
MODEL_ROOT = REPO_ROOT / "model"
for extra in (MODEL_ROOT, REPO_ROOT):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import torch
import torchvision.transforms.v2 as T2
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast

from main import XeniumCellDataset, load_slide_split, build_random_slide_split  # noqa: E402
from model import Image2Transcripts  # noqa: E402  (imports model/model.py)


# Float16 on GPU is enough precision for all downstream Pearson/retrieval
# computations and cuts the dump size roughly in half.
FLOAT_DUMP_DTYPE = np.float16


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="Directory with best_model.pt (and optionally slide_split_used.json)")
    ap.add_argument("--gene_dir", type=Path, default=Path("data/gene_expression"))
    ap.add_argument("--img_dir", type=Path, default=Path("data/images"))
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Explicit JSON split file; falls back to run_dir/slide_split_used.json")
    ap.add_argument("--out_name", default="eval",
                    help="Subdirectory under run_dir where dumps are written (default: eval)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["train", "val", "test"],
                    help="Which split subsets to evaluate")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed_fallback", type=int, default=42,
                    help="Seed used when no split file is supplied and we build a random split")
    ap.add_argument("--subset_cells", type=int, default=0,
                    help="If >0, cap each split at this many cells (stratified by slide)")
    ap.add_argument("--ckpt_name", default="best_model.pt",
                    help="Checkpoint file name inside run_dir")
    ap.add_argument("--fp32", action="store_true",
                    help="Disable autocast and run the forward pass in fp32")
    return ap.parse_args()


def resolve_split_file(args: argparse.Namespace) -> Path | None:
    if args.split_file is not None:
        return args.split_file
    fallback = args.run_dir / "slide_split_used.json"
    if fallback.exists():
        return fallback
    return None


def build_split_spec(split_file: Path | None, unique_slides: list[str], seed: int):
    if split_file is None:
        return build_random_slide_split(unique_slides, seed)

    with open(split_file) as f:
        spec = json.load(f)

    # slide_split_used.json dumps the resolved split under slightly different
    # keys; normalize to what load_slide_split expects.
    if "train_slides" in spec and "train" not in spec:
        spec = {
            "name": spec.get("split_name") or split_file.stem,
            "train": spec.get("train_slides", []),
            "val": spec.get("val_slides", []),
            "test": spec.get("test_slides", []),
        }
        normalized_path = split_file.with_suffix(".normalized.json")
        with open(normalized_path, "w") as f:
            json.dump(spec, f, indent=2)
        return load_slide_split(normalized_path, unique_slides)

    return load_slide_split(split_file, unique_slides)


def stratified_subsample(indices: list[int], slide_ids: np.ndarray,
                         cap: int, rng: np.random.Generator) -> list[int]:
    if cap <= 0 or len(indices) <= cap:
        return indices

    indices = np.asarray(indices, dtype=np.int64)
    unique_slides = np.unique(slide_ids[indices])
    per_slide = max(1, cap // max(1, len(unique_slides)))

    picked = []
    for slide in unique_slides:
        slide_mask = slide_ids[indices] == slide
        slide_pool = indices[slide_mask]
        take = min(per_slide, len(slide_pool))
        picked.extend(rng.choice(slide_pool, size=take, replace=False).tolist())

    if len(picked) < cap:
        remaining = cap - len(picked)
        leftover = np.setdiff1d(indices, np.asarray(picked), assume_unique=False)
        if len(leftover):
            extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
            picked.extend(extra.tolist())

    return sorted(picked[:cap])


def load_model(run_dir: Path, ckpt_name: str, gene_dim: int,
               device: torch.device) -> Image2Transcripts:
    ckpt_fp = run_dir / ckpt_name
    if not ckpt_fp.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_fp}")

    model = Image2Transcripts(gene_dim=gene_dim, ckpt_path=None, pretrained=False)
    state = torch.load(ckpt_fp, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys when loading {ckpt_fp}: {len(missing)}")
        print("  first 5:", missing[:5])
    if unexpected:
        print(f"[warn] unexpected keys when loading {ckpt_fp}: {len(unexpected)}")
        print("  first 5:", unexpected[:5])

    model.to(device)
    model.eval()
    return model


def run_inference(model: Image2Transcripts, loader: DataLoader,
                  device: torch.device, use_autocast: bool,
                  gene_dim: int, embed_dim: int,
                  n_cells: int) -> dict[str, np.ndarray]:
    mu = np.empty((n_cells, gene_dim), dtype=FLOAT_DUMP_DTYPE)
    theta = np.empty((n_cells, gene_dim), dtype=FLOAT_DUMP_DTYPE)
    pi = np.empty((n_cells, gene_dim), dtype=FLOAT_DUMP_DTYPE)
    i_emb = np.empty((n_cells, embed_dim), dtype=FLOAT_DUMP_DTYPE)
    g_emb = np.empty((n_cells, embed_dim), dtype=FLOAT_DUMP_DTYPE)
    x_true = np.empty((n_cells, gene_dim), dtype=np.int16)

    cursor = 0
    t0 = time.time()
    printed_every = max(1, len(loader) // 40)

    with torch.no_grad():
        for step, (img, feat, _mask) in enumerate(loader):
            img = img.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)

            if use_autocast:
                with autocast(device_type="cuda"):
                    i_out, g_out, mu_out, theta_out, pi_out = model(img, feat)
            else:
                i_out, g_out, mu_out, theta_out, pi_out = model(img, feat)

            bs = img.size(0)
            slot = slice(cursor, cursor + bs)
            cursor += bs

            mu[slot] = mu_out.float().cpu().numpy().astype(FLOAT_DUMP_DTYPE)
            theta[slot] = theta_out.float().cpu().numpy().astype(FLOAT_DUMP_DTYPE)
            pi[slot] = pi_out.float().cpu().numpy().astype(FLOAT_DUMP_DTYPE)
            i_emb[slot] = i_out.float().cpu().numpy().astype(FLOAT_DUMP_DTYPE)
            g_emb[slot] = g_out.float().cpu().numpy().astype(FLOAT_DUMP_DTYPE)

            # Dataset concatenates [counts, logfc]; recover raw counts only.
            x_true[slot] = (
                feat[:, :gene_dim].float().cpu().numpy().round().astype(np.int16)
            )

            if (step + 1) % printed_every == 0 or (step + 1) == len(loader):
                elapsed = time.time() - t0
                rate = cursor / max(1e-6, elapsed)
                print(f"  step {step + 1}/{len(loader)} | cells {cursor}/{n_cells} | "
                      f"{rate:,.0f} cells/s | elapsed {elapsed:,.0f}s")

    assert cursor == n_cells, f"Wrote {cursor} cells, expected {n_cells}"
    return {
        "mu": mu, "theta": theta, "pi_logits": pi,
        "i_emb": i_emb, "g_emb": g_emb, "x_true": x_true,
    }


def dump_split(out_dir: Path, tensors: dict[str, np.ndarray],
               slide_names: list[str], slide_idx: np.ndarray,
               cell_order: np.ndarray, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "mu.npy", tensors["mu"])
    np.save(out_dir / "theta.npy", tensors["theta"])
    np.save(out_dir / "pi_logits.npy", tensors["pi_logits"])
    np.save(out_dir / "i_emb.npy", tensors["i_emb"])
    np.save(out_dir / "g_emb.npy", tensors["g_emb"])
    np.save(out_dir / "x_true.npy", tensors["x_true"])
    np.save(out_dir / "slide_idx.npy", slide_idx.astype(np.int32))
    np.save(out_dir / "cell_order.npy", cell_order.astype(np.int32))

    with open(out_dir / "slide_names.json", "w") as f:
        json.dump(slide_names, f, indent=2)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = args.run_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    eval_tfm = T2.Compose([
        T2.Resize((224, 224), antialias=True),
        T2.ToDtype(torch.float32, scale=True),
    ])

    print(f"Loading dataset from {args.gene_dir} + {args.img_dir}")
    ds = XeniumCellDataset(args.gene_dir, args.img_dir,
                            transform=eval_tfm, gene_mask_ratio=0.0)

    gene_dim = len(ds.shared_genes)
    slide_ids_per_cell = np.asarray(ds.slide_ids_per_cell)
    unique_slides = sorted(np.unique(slide_ids_per_cell).tolist())

    split_fp = resolve_split_file(args)
    if split_fp is None:
        print(f"[warn] no split file supplied; falling back to random split seed={args.seed_fallback}")
    split_spec = build_split_spec(split_fp, unique_slides, args.seed_fallback)

    rng = np.random.default_rng(args.seed_fallback)

    requested_splits = []
    for split_name in args.splits:
        slide_set = set(split_spec.get(split_name, []))
        if not slide_set:
            print(f"[warn] split '{split_name}' has no slides; skipping")
            continue
        idx = [i for i, s in enumerate(slide_ids_per_cell) if s in slide_set]
        if not idx:
            print(f"[warn] split '{split_name}' resolves to 0 cells; skipping")
            continue
        if args.subset_cells > 0:
            idx = stratified_subsample(idx, slide_ids_per_cell, args.subset_cells, rng)
        requested_splits.append((split_name, idx, sorted(slide_set)))
        print(f"Split '{split_name}': {len(slide_set)} slides, {len(idx):,} cells")

    if not requested_splits:
        raise SystemExit("No splits resolved to any cells; nothing to do")

    print(f"Loading model from {args.run_dir / args.ckpt_name}")
    model = load_model(args.run_dir, args.ckpt_name, gene_dim, device)

    embed_dim = model.image_proj[-1].out_features

    for split_name, idx, slide_list in requested_splits:
        print(f"\n=== Evaluating split: {split_name} ===")
        subset = Subset(ds, idx)
        loader = DataLoader(subset, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        tensors = run_inference(
            model, loader, device,
            use_autocast=not args.fp32,
            gene_dim=gene_dim, embed_dim=embed_dim,
            n_cells=len(idx),
        )

        slide_name_to_idx = {name: i for i, name in enumerate(unique_slides)}
        slide_idx_arr = np.array(
            [slide_name_to_idx[s] for s in slide_ids_per_cell[idx]],
            dtype=np.int32,
        )
        cell_order_arr = np.asarray(idx, dtype=np.int32)

        metadata = {
            "run_dir": str(args.run_dir.resolve()),
            "ckpt_path": str((args.run_dir / args.ckpt_name).resolve()),
            "split_name": split_name,
            "split_source": str(split_fp.resolve()) if split_fp else "random",
            "split_slides": slide_list,
            "n_cells": len(idx),
            "gene_dim": gene_dim,
            "embed_dim": embed_dim,
            "genes": ds.shared_genes,
            "autocast": not args.fp32,
            "subset_cells_cap": args.subset_cells,
        }

        out_dir = out_root / split_name
        dump_split(out_dir, tensors, unique_slides,
                    slide_idx_arr, cell_order_arr, metadata)
        print(f"Wrote predictions to {out_dir}")

    # Tiny index file so downstream tools can discover what is available.
    index_fp = out_root / "predictions_index.json"
    existing = {}
    if index_fp.exists():
        with open(index_fp) as f:
            existing = json.load(f)
    for split_name, idx, _ in requested_splits:
        existing[split_name] = {
            "dir": f"{args.out_name}/{split_name}",
            "n_cells": len(idx),
        }
    with open(index_fp, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    main()
