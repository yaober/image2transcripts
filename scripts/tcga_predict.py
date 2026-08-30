"""Stream TCGA COAD whole-slide images through the trained Image2Transcript
model and dump slide-level gene-expression predictions.

For each ``.svs`` slide under ``--slides_dir`` we:

1. Open the WSI with OpenSlide and read its native microns-per-pixel (MPP).
2. Build a coarse tissue mask from a thumbnail using HSV saturation + Otsu
   so we don't waste inference on background.
3. Enumerate non-overlapping tiles that cover the tissue at a target field
   of view of ~40 µm (matching the ~188 px / 0.21 µm/px Xenium training
   crops).  Each tile is read at level 0 and resized to 224x224 by the
   model's validation transform.
4. Sub-sample up to ``--max_tiles_per_slide`` tiles per slide for inference
   speed; tiles are chosen with a fixed seed so reruns are reproducible.
5. Batch the tiles through ``best_model.pt`` on a single GPU and collect
   per-tile ``mu`` (predicted ZINB means) and ``i_emb`` embeddings.
6. Aggregate to a single row per slide:
       mu_mean, mu_median, mu_p25, mu_p75, mu_std  (per gene)
       i_emb_mean                                  (768-dim)
       n_tiles_used, n_tiles_total, mpp, tile_px
   plus the patient barcode parsed from the TCGA file name.

Everything is written into ``--out_dir`` (default
``runs/tcga_coad/predictions/``).  Re-running the script skips any slide
for which a summary already exists unless ``--force`` is set.

Example::

    python scripts/tcga_predict.py \
        --slides_dir "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data/TCGA/COAD/slides" \
        --run_dir runs/fixedsplit_v3/full_seed42 \
        --gene_dir data/gene_expression \
        --out_dir runs/tcga_coad/predictions \
        --batch 128 --num_workers 0 \
        --max_tiles_per_slide 1500
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
for extra in (MODEL_ROOT, REPO_ROOT):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import openslide
import torch
from PIL import Image
from torch.amp import autocast

from model import Image2Transcripts  # noqa: E402 -- imports model/model.py


# Matches the Xenium training crop field of view.  Training crops are
# 188 px at ~0.2125 µm/px  ->  ~40 µm side length.  We extract TCGA tiles
# at the same physical size, then let the model's 224x224 resize adapt.
DEFAULT_TILE_UM = 40.0

# Barcode lookup in a TCGA slide filename: "TCGA-XX-YYYY-01Z-00-DXn.*.svs".
BARCODE_RE = re.compile(r"^(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})")


# ----------------------------------------------------------------------
# Clinical / gene loading helpers
# ----------------------------------------------------------------------


def load_gene_order(gene_dir: Path) -> list[str]:
    """Load the same 372-gene order the trained model expects.

    We reuse the first shared gene list from the Xenium training data so
    the column order of ``mu`` matches the main paper.
    """
    import anndata as ad  # imported lazily to avoid scanpy cost at CLI parse

    expr_files = sorted(fp for fp in gene_dir.iterdir() if fp.suffix == ".h5ad")
    shared = None
    for fp in expr_files:
        adata = ad.read_h5ad(fp, backed="r")
        var_set = set(adata.var_names)
        shared = var_set if shared is None else shared & var_set
    if not shared:
        raise RuntimeError(f"Could not derive shared gene list from {gene_dir}")
    return sorted(shared)


# ----------------------------------------------------------------------
# Tissue mask
# ----------------------------------------------------------------------


@dataclass
class SlideMeta:
    slide_fp: Path
    slide_id: str
    patient_barcode: str
    mpp: float
    level0_w: int
    level0_h: int
    tile_px: int


def open_slide(slide_fp: Path, default_mpp: float = 0.25) -> tuple[openslide.OpenSlide, float]:
    slide = openslide.OpenSlide(str(slide_fp))
    mpp_str = slide.properties.get("openslide.mpp-x")
    if mpp_str is None:
        mpp_str = slide.properties.get("openslide.mpp-y")
    try:
        mpp = float(mpp_str) if mpp_str is not None else default_mpp
    except (TypeError, ValueError):
        mpp = default_mpp
    if not math.isfinite(mpp) or mpp <= 0:
        mpp = default_mpp
    return slide, mpp


def parse_patient_barcode(slide_fp: Path) -> str:
    m = BARCODE_RE.match(slide_fp.name)
    return m.group(1) if m else slide_fp.stem[:12]


def tissue_mask_from_thumbnail(
    slide: openslide.OpenSlide,
    target_downsample: float = 32.0,
    saturation_threshold: int = 20,
) -> tuple[np.ndarray, float]:
    """Compute a boolean tissue mask at a low-resolution thumbnail.

    Returns the mask plus the actual downsample factor used (level width
    ratio to level 0).  Tissue pixels are those with saturation above a
    low threshold and luminance below an upper threshold — this cheaply
    filters out the bright white background of H&E scans.
    """
    lvl = slide.get_best_level_for_downsample(target_downsample)
    lvl_w, lvl_h = slide.level_dimensions[lvl]
    base_w, base_h = slide.level_dimensions[0]
    downsample = base_w / lvl_w

    thumb = slide.read_region((0, 0), lvl, (lvl_w, lvl_h)).convert("RGB")
    arr = np.asarray(thumb, dtype=np.uint8)

    # Simple HSV + luminance check.  We avoid importing cv2 to keep the
    # dependency surface small; PIL has an HSV conversion.
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.uint8)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    mask = (saturation > saturation_threshold) & (value < 240) & (arr.min(axis=2) < 235)
    return mask, downsample


def sample_tile_coords(
    slide_meta: SlideMeta,
    tissue_mask: np.ndarray,
    thumb_downsample: float,
    tissue_threshold: float,
    max_tiles: int,
    seed: int,
) -> np.ndarray:
    """Return an (n, 2) int array of (x, y) level-0 top-left tile coords.

    The mask is in the thumbnail coordinate system; we sweep a grid at the
    tile pitch, accept any grid cell whose thumbnail footprint is at least
    ``tissue_threshold`` fraction tissue, then randomly subsample to
    ``max_tiles`` with a fixed seed.
    """
    tile_px = slide_meta.tile_px
    grid_px = max(1, int(round(tile_px / thumb_downsample)))
    mask_h, mask_w = tissue_mask.shape

    # Iterate over mask grid; record level-0 coordinates of valid tiles.
    coords: list[tuple[int, int]] = []
    for ty in range(0, mask_h - grid_px + 1, grid_px):
        for tx in range(0, mask_w - grid_px + 1, grid_px):
            patch = tissue_mask[ty:ty + grid_px, tx:tx + grid_px]
            if patch.size == 0:
                continue
            if patch.mean() < tissue_threshold:
                continue
            lx = int(round(tx * thumb_downsample))
            ly = int(round(ty * thumb_downsample))
            if lx + tile_px > slide_meta.level0_w or ly + tile_px > slide_meta.level0_h:
                continue
            coords.append((lx, ly))

    if not coords:
        return np.empty((0, 2), dtype=np.int32)

    coords_arr = np.asarray(coords, dtype=np.int32)
    if max_tiles > 0 and len(coords_arr) > max_tiles:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(coords_arr), size=max_tiles, replace=False)
        coords_arr = coords_arr[np.sort(pick)]
    return coords_arr


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------


def load_model(run_dir: Path, gene_dim: int, ckpt_name: str,
               device: torch.device) -> Image2Transcripts:
    ckpt_fp = run_dir / ckpt_name
    if not ckpt_fp.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_fp}")

    model = Image2Transcripts(gene_dim=gene_dim, ckpt_path=None, pretrained=False)
    state = torch.load(ckpt_fp, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys in state_dict")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys in state_dict")
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------


def _read_tile(slide: openslide.OpenSlide, x: int, y: int, tile_px: int) -> np.ndarray:
    region = slide.read_region((x, y), 0, (tile_px, tile_px)).convert("RGB")
    return np.asarray(region, dtype=np.uint8)


def _preprocess_batch(tiles: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Mirror the model's val transform: resize -> ToDtype(float32, scale=True)."""
    pil_imgs = [Image.fromarray(t).resize((224, 224), Image.BILINEAR) for t in tiles]
    stacked = np.stack(
        [np.asarray(p, dtype=np.uint8) for p in pil_imgs], axis=0
    ).transpose(0, 3, 1, 2)
    x = torch.from_numpy(stacked).to(device, non_blocking=True).float() / 255.0
    return x


@torch.no_grad()
def infer_slide(
    slide_meta: SlideMeta,
    coords: np.ndarray,
    slide: openslide.OpenSlide,
    model: Image2Transcripts,
    device: torch.device,
    gene_dim: int,
    embed_dim: int,
    batch: int,
    use_autocast: bool,
    gene_input_shape: int,
) -> dict[str, np.ndarray]:
    n = len(coords)
    mu = np.empty((n, gene_dim), dtype=np.float16)
    i_emb = np.empty((n, embed_dim), dtype=np.float16)

    # The model's gene encoder wants a (B, 2G) feature vector.  At TCGA
    # inference time we don't have matched Xenium counts, so we pass a
    # zero placeholder; only the image pathway (``i_emb`` and ``mu`` via
    # the ZINB head) is used downstream.
    zero_gene = torch.zeros((batch, gene_input_shape),
                             device=device, dtype=torch.float32)

    cursor = 0
    while cursor < n:
        stop = min(n, cursor + batch)
        tile_bytes: list[np.ndarray] = []
        for cx, cy in coords[cursor:stop]:
            tile_bytes.append(_read_tile(slide, int(cx), int(cy), slide_meta.tile_px))

        x = _preprocess_batch(tile_bytes, device)
        bs = x.size(0)
        g = zero_gene[:bs] if bs == batch else torch.zeros(
            (bs, gene_input_shape), device=device, dtype=torch.float32
        )

        if use_autocast:
            with autocast(device_type="cuda"):
                i_out, _g_out, mu_out, _theta, _pi = model(x, g)
        else:
            i_out, _g_out, mu_out, _theta, _pi = model(x, g)

        mu[cursor:stop] = mu_out.float().cpu().numpy().astype(np.float16)
        i_emb[cursor:stop] = i_out.float().cpu().numpy().astype(np.float16)
        cursor = stop

    return {"mu": mu, "i_emb": i_emb}


def summarize_slide(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu = preds["mu"].astype(np.float32)
    i_emb = preds["i_emb"].astype(np.float32)

    return {
        "mu_mean": mu.mean(axis=0).astype(np.float16),
        "mu_median": np.median(mu, axis=0).astype(np.float16),
        "mu_p25": np.percentile(mu, 25, axis=0).astype(np.float16),
        "mu_p75": np.percentile(mu, 75, axis=0).astype(np.float16),
        "mu_std": mu.std(axis=0).astype(np.float16),
        "i_emb_mean": i_emb.mean(axis=0).astype(np.float16),
    }


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides_dir", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="Trained checkpoint directory (contains best_model.pt)")
    ap.add_argument("--gene_dir", type=Path, default=REPO_ROOT / "data/gene_expression",
                    help="Xenium gene-expression root used to derive the shared gene order")
    ap.add_argument("--out_dir", type=Path, default=REPO_ROOT / "runs/tcga_coad/predictions")
    ap.add_argument("--ckpt_name", default="best_model.pt")
    ap.add_argument("--tile_um", type=float, default=DEFAULT_TILE_UM)
    ap.add_argument("--max_tiles_per_slide", type=int, default=1500)
    ap.add_argument("--tissue_threshold", type=float, default=0.5,
                    help="Minimum tissue fraction for a tile to be kept")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many slides (0 = all). Useful for smoke tests")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip slides whose summary already exists (default true)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess slides even if a summary exists")
    ap.add_argument("--save_tile_preds", action="store_true",
                    help="Also save per-tile mu/embeddings (large on disk)")
    return ap.parse_args()


def process_slide(
    slide_fp: Path,
    model: Image2Transcripts,
    device: torch.device,
    gene_dim: int,
    embed_dim: int,
    args: argparse.Namespace,
) -> dict | None:
    try:
        slide, mpp = open_slide(slide_fp)
    except Exception as exc:  # pragma: no cover -- defensive for broken SVS
        print(f"[error] cannot open {slide_fp.name}: {exc}")
        return None

    try:
        W, H = slide.level_dimensions[0]
        tile_px = max(16, int(round(args.tile_um / mpp)))
        slide_meta = SlideMeta(
            slide_fp=slide_fp,
            slide_id=slide_fp.stem,
            patient_barcode=parse_patient_barcode(slide_fp),
            mpp=mpp,
            level0_w=W,
            level0_h=H,
            tile_px=tile_px,
        )
        print(f"  slide {slide_meta.slide_id}  MPP={mpp:.3f}  "
              f"level0={W}x{H}  tile_px={tile_px}")

        t0 = time.time()
        mask, down = tissue_mask_from_thumbnail(slide)
        tissue_frac = float(mask.mean())
        coords = sample_tile_coords(
            slide_meta, mask, down, args.tissue_threshold,
            args.max_tiles_per_slide, args.seed,
        )
        t_mask = time.time() - t0
        if len(coords) == 0:
            print(f"  [warn] no tissue tiles found in {slide_meta.slide_id}")
            return None

        n_total = int(mask.sum() // max(1, (tile_px / down) ** 2))  # rough upper bound
        print(f"    tissue_frac={tissue_frac:.3f}  tiles_kept={len(coords)}  "
              f"mask_time={t_mask:.1f}s")

        t0 = time.time()
        preds = infer_slide(
            slide_meta, coords, slide, model, device,
            gene_dim=gene_dim, embed_dim=embed_dim,
            batch=args.batch, use_autocast=not args.fp32,
            gene_input_shape=2 * gene_dim,
        )
        t_infer = time.time() - t0
        print(f"    inference={t_infer:.1f}s  ({len(coords) / max(1e-3, t_infer):.0f} tiles/s)")
    finally:
        slide.close()

    summary = summarize_slide(preds)
    summary.update({
        "slide_id": slide_meta.slide_id,
        "patient_barcode": slide_meta.patient_barcode,
        "mpp": float(mpp),
        "tile_px": int(tile_px),
        "n_tiles_used": int(len(coords)),
        "n_tiles_estimated": int(n_total),
        "tissue_fraction": tissue_frac,
    })

    out_fp = args.out_dir / f"{slide_meta.slide_id}.npz"
    payload = {k: v for k, v in summary.items()
               if isinstance(v, np.ndarray)}
    payload["tile_coords"] = coords.astype(np.int32)
    meta = {k: v for k, v in summary.items() if not isinstance(v, np.ndarray)}
    meta["tile_coords_shape"] = list(coords.shape)
    np.savez_compressed(out_fp, **payload, metadata_json=np.asarray(
        json.dumps(meta), dtype="S"))

    if args.save_tile_preds:
        tile_dir = args.out_dir / "tile_preds"
        tile_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tile_dir / f"{slide_meta.slide_id}.npz",
            mu=preds["mu"], i_emb=preds["i_emb"],
            tile_coords=coords.astype(np.int32),
        )

    return meta


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Deriving gene order from Xenium training data ...")
    genes = load_gene_order(args.gene_dir)
    gene_dim = len(genes)
    print(f"  {gene_dim} shared genes")

    with open(args.out_dir / "genes.json", "w") as f:
        json.dump(genes, f, indent=2)

    print(f"Loading model from {args.run_dir / args.ckpt_name}")
    model = load_model(args.run_dir, gene_dim, args.ckpt_name, device)
    embed_dim = model.image_proj[-1].out_features

    slide_files = sorted(p for p in args.slides_dir.iterdir() if p.suffix.lower() == ".svs")
    if args.limit > 0:
        slide_files = slide_files[: args.limit]
    print(f"Found {len(slide_files)} slides in {args.slides_dir}")

    index_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for si, slide_fp in enumerate(slide_files, start=1):
        out_fp = args.out_dir / f"{slide_fp.stem}.npz"
        if out_fp.exists() and not args.force:
            print(f"[{si}/{len(slide_files)}] skip existing {slide_fp.name}")
            try:
                loaded = np.load(out_fp, allow_pickle=False)
                if "metadata_json" in loaded.files:
                    meta = json.loads(loaded["metadata_json"].item().decode())
                    index_rows.append(meta)
            except Exception:
                pass
            continue

        print(f"[{si}/{len(slide_files)}] {slide_fp.name}")
        try:
            meta = process_slide(slide_fp, model, device, gene_dim, embed_dim, args)
        except Exception as exc:
            print(f"[error] {slide_fp.name}: {exc}")
            traceback.print_exc()
            failures.append((slide_fp.name, str(exc)))
            continue

        if meta is not None:
            index_rows.append(meta)

    if index_rows:
        index_csv = args.out_dir / "slide_index.csv"
        fieldnames = [
            "slide_id", "patient_barcode", "mpp", "tile_px",
            "n_tiles_used", "n_tiles_estimated", "tissue_fraction",
        ]
        with open(index_csv, "w") as f:
            f.write(",".join(fieldnames) + "\n")
            for row in index_rows:
                f.write(",".join(str(row.get(c, "")) for c in fieldnames) + "\n")
        print(f"\nWrote slide index: {index_csv}")

    if failures:
        fail_fp = args.out_dir / "failures.json"
        with open(fail_fp, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"{len(failures)} slides failed; details in {fail_fp}")


if __name__ == "__main__":
    main()
