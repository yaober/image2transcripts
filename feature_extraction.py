"""
Dual-Scale Patch Feature Extraction for TCGA Whole-Slide Images.

For each tissue location, extracts:
  - CELLULAR scale   : 80 µm field of view
  - MICROENVIRONMENT : 256 µm field of view

Runs pathology foundation models on both scales and saves
embeddings in HDF5 format organized by model / cancer type / scale.

Usage:
    python feature_extraction.py \
        --tcga_root /path/to/TCGA \
        --output_dir /path/to/output \
        [--cancer_types BLCA LUAD] \
        [--stride_um 40.0] \
        [--batch_size 256] \
        [--device cuda:0] \
        [--max_slides 2]
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import gc
import json
import logging
import os
import random
import sys
import time
import traceback
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from openslide import open_slide
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared foundation-model helpers.
# Model loading, normalisation, PatchDataset, and batched inference all
# live in scripts/foundation_models.py so the Xenium feature extractor
# (scripts/extract_foundation_features_xenium.py) and this TCGA pipeline
# share a single source of truth.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from foundation_models import (  # noqa: E402
    ALL_MODEL_NAMES,
    MODEL_INPUT_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    PatchDataset,
    ensure_hf_token,
    load_extractor,
    run_inference_on_patches,
)


# ---------------------------------------------------------------------------
# HuggingFace token
# ---------------------------------------------------------------------------
# Callers must export their own HF_TOKEN beforehand (the gated pathology
# foundation models require an authenticated HuggingFace session).  We read it
# from the environment; ensure_hf_token() raises a clear error if it is unset.
HF_TOKEN = os.environ.get("HF_TOKEN")
ensure_hf_token()

# ---------------------------------------------------------------------------
# Pipeline-specific constants (keep local — they are TCGA-WSI-only)
# ---------------------------------------------------------------------------
INNER_PATCH_UM: float = 80.0
OUTER_PATCH_UM: float = 256.0

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Runtime configuration for the extraction pipeline."""
    tcga_root: str
    output_dir: str
    cancer_types: List[str] = field(default_factory=list)
    stride_um: float = 40.0
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 42
    max_slides: Optional[int] = None
    overwrite: bool = False
    num_workers: int = 4
    matter_threshold: float = 0.50


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure root logger to write to console and a log file.

    Args:
        output_dir: Directory where logs/extraction.log will be created.

    Returns:
        Configured Logger instance.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "extraction.log")

    logger = logging.getLogger("extraction")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# MPP detection
# ---------------------------------------------------------------------------

def detect_mpp(slide, logger: logging.Logger) -> Optional[float]:
    """Detect microns-per-pixel from slide metadata.

    Tries keys in priority order:
      1. openslide.mpp-x
      2. openslide.mpp-y
      3. tiff.XResolution (centimeter units only)

    Args:
        slide: Opened OpenSlide object.
        logger: Logger instance.

    Returns:
        MPP as float, or None if not determinable / invalid.
    """
    props = slide.properties

    for key in ("openslide.mpp-x", "openslide.mpp-y"):
        val = props.get(key)
        if val is not None:
            try:
                mpp = float(val)
                if 0 < mpp <= 2.0:
                    return mpp
                else:
                    logger.warning(f"MPP from {key}={mpp} is out of valid range (0, 2.0].")
                    return None
            except ValueError:
                continue

    # Fallback: tiff.XResolution in centimeters
    xres = props.get("tiff.XResolution")
    unit = props.get("tiff.ResolutionUnit", "")
    if xres is not None and "centimeter" in str(unit).lower():
        try:
            mpp = 10000.0 / float(xres)
            if 0 < mpp <= 2.0:
                return mpp
            else:
                logger.warning(f"MPP derived from tiff.XResolution={mpp:.4f} is out of range.")
                return None
        except (ValueError, ZeroDivisionError):
            pass

    logger.warning("Could not determine MPP from slide metadata.")
    return None


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str, logger: logging.Logger) -> str:
    """Resolve a CLI device string into a torch-compatible device string.

    Supports:
      - cpu
      - cuda
      - cuda:N
      - N  (shorthand for cuda:N)
    """
    requested = str(device_arg).strip().lower()
    if requested == "cpu":
        return "cpu"

    if requested.isdigit():
        requested = f"cuda:{requested}"

    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            return "cpu"
        return "cuda"

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning(f"{requested} requested but CUDA is not available — falling back to CPU.")
            return "cpu"
        try:
            device_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid --device value '{device_arg}'. Use cpu, cuda, cuda:N, or N."
            ) from exc

        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested GPU {device_index}, but available CUDA devices are "
                f"0..{torch.cuda.device_count() - 1}."
            )
        return requested

    raise ValueError(f"Invalid --device value '{device_arg}'. Use cpu, cuda, cuda:N, or N.")


# ---------------------------------------------------------------------------
# Tissue masking
# ---------------------------------------------------------------------------

def compute_tissue_mask(
    slide,
    slide_id: str,
    mask_cache_dir: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, float]:
    """Compute or load cached binary tissue mask.

    Args:
        slide: Opened OpenSlide object.
        slide_id: Unique identifier for this slide (used in cache filenames).
        mask_cache_dir: Directory for caching mask files.
        logger: Logger instance.

    Returns:
        (mask, scale_factor) where mask is a uint8 array (1=tissue) at
        thumbnail resolution, and scale_factor maps mask coordinates to
        level-0 coordinates: level0_coord = mask_coord * scale_factor.
    """
    os.makedirs(mask_cache_dir, exist_ok=True)
    mask_path = os.path.join(mask_cache_dir, f"{slide_id}_mask.npy")
    scale_path = os.path.join(mask_cache_dir, f"{slide_id}_scale.npy")

    if os.path.exists(mask_path) and os.path.exists(scale_path):
        logger.debug(f"Mask cache HIT for {slide_id}")
        mask = np.load(mask_path)
        scale_factor = float(np.load(scale_path))
        return mask, scale_factor

    logger.debug(f"Mask cache MISS for {slide_id} — computing")

    # Choose thumbnail level
    if slide.level_count >= 3:
        thumb_level = 2
    elif slide.level_count >= 2:
        thumb_level = 1
    else:
        thumb_level = 0

    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    region = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h))
    arr = np.array(region)[..., :3]  # drop alpha

    # HSV → saturation channel → Otsu threshold
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    close_k = np.ones((15, 15), dtype=np.uint8)
    open_k = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    # Remove small connected components (area < 1000 px)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 1000:
            mask[labels == lbl] = 0

    mask = (mask > 0).astype(np.uint8)

    scale_factor = slide.level_dimensions[0][0] / thumb_w

    np.save(mask_path, mask)
    np.save(scale_path, np.array(scale_factor))

    return mask, scale_factor


# ---------------------------------------------------------------------------
# Valid coordinate generation
# ---------------------------------------------------------------------------

def get_valid_centers(
    slide,
    mask: np.ndarray,
    scale_factor: float,
    inner_pixels: int,
    outer_pixels: int,
    stride_pixels: int,
    matter_threshold: float,
    logger: logging.Logger,
) -> List[Tuple[int, int]]:
    """Generate level-0 center coordinates that pass tissue and boundary checks.

    Args:
        slide: Opened OpenSlide object.
        mask: Binary tissue mask (H, W) at thumbnail resolution.
        scale_factor: Maps mask px → level-0 px.
        inner_pixels: CELLULAR patch size in level-0 pixels.
        outer_pixels: MICROENVIRONMENT patch size in level-0 pixels.
        stride_pixels: Grid stride in level-0 pixels.
        matter_threshold: Minimum tissue coverage fraction (0–1).
        logger: Logger instance.

    Returns:
        List of (cx, cy) level-0 center coordinates.
    """
    slide_w, slide_h = slide.level_dimensions[0]
    mask_h, mask_w = mask.shape
    outer_half = outer_pixels // 2
    inner_half_mask = max(1, round(inner_pixels / 2.0 / scale_factor))

    xs = range(stride_pixels // 2, slide_w, stride_pixels)
    ys = range(stride_pixels // 2, slide_h, stride_pixels)
    valid_xs = np.array(
        [cx for cx in xs if cx - outer_half >= 0 and cx + outer_half <= slide_w],
        dtype=np.int32,
    )
    valid_ys = np.array(
        [cy for cy in ys if cy - outer_half >= 0 and cy + outer_half <= slide_h],
        dtype=np.int32,
    )

    n_border_skipped = len(range(stride_pixels // 2, slide_w, stride_pixels)) * len(
        range(stride_pixels // 2, slide_h, stride_pixels)
    ) - (len(valid_xs) * len(valid_ys))

    if len(valid_xs) == 0 or len(valid_ys) == 0:
        return []

    cx_grid, cy_grid = np.meshgrid(valid_xs, valid_ys, indexing="ij")
    mx = (cx_grid / scale_factor).astype(np.int32)
    my = (cy_grid / scale_factor).astype(np.int32)

    mx0 = np.clip(mx - inner_half_mask, 0, mask_w)
    mx1 = np.clip(mx + inner_half_mask, 0, mask_w)
    my0 = np.clip(my - inner_half_mask, 0, mask_h)
    my1 = np.clip(my + inner_half_mask, 0, mask_h)

    areas = (mx1 - mx0) * (my1 - my0)
    non_empty = areas > 0
    if not np.any(non_empty):
        return []

    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    tissue_sum = (
        integral[my1, mx1]
        - integral[my0, mx1]
        - integral[my1, mx0]
        + integral[my0, mx0]
    )
    coverage = np.zeros_like(tissue_sum, dtype=np.float32)
    coverage[non_empty] = tissue_sum[non_empty] / areas[non_empty]

    keep = coverage > matter_threshold
    kept_coords = np.stack((cx_grid[keep], cy_grid[keep]), axis=1)
    valid = [tuple(coord) for coord in kept_coords.tolist()]

    if n_border_skipped:
        logger.debug(f"Total border-skipped coordinates: {n_border_skipped}")

    return valid


# ---------------------------------------------------------------------------
# Cellular patch strategy
# ---------------------------------------------------------------------------

def get_inner_strategy(inner_pixels: int) -> str:
    """
    Decide the cellular patch preparation strategy.

    - reflect_pad: native level-0 crop is large enough to be safely padded
      to model input size
    - skip_cellular: native crop is too small for reliable cellular-scale
      embedding with the pathology foundation model
    """
    return "reflect_pad" if inner_pixels >= 112 else "skip_cellular"


# ---------------------------------------------------------------------------
# Patch read planning
# ---------------------------------------------------------------------------

def build_patch_read_plan(
    slide,
    outer_pixels: int,
    model_input_size: int,
) -> Dict[str, int]:
    """Precompute OpenSlide read parameters reused across all patch centers."""
    outer_target_downsample = max(1.0, outer_pixels / max(model_input_size, 1))
    outer_level = slide.get_best_level_for_downsample(outer_target_downsample)
    outer_level_ds = float(slide.level_downsamples[outer_level])
    outer_read_size = max(1, round(outer_pixels / outer_level_ds))

    plan = {
        "outer_level": outer_level,
        "outer_read_size": outer_read_size,
    }

    return plan


# ---------------------------------------------------------------------------
# Dual-scale patch extraction
# ---------------------------------------------------------------------------

def extract_patch_pair(
    slide,
    cx: int,
    cy: int,
    inner_pixels: int,
    outer_pixels: int,
    model_input_size: int,
    inner_strategy: str,
    read_plan: Dict[str, int],
) -> Optional[Tuple[Optional[np.ndarray], np.ndarray]]:
    """Extract a co-centered cellular and microenvironment patch.

    Args:
        slide: Opened OpenSlide object.
        cx, cy: Level-0 center coordinates.
        inner_pixels: CELLULAR patch size in level-0 pixels.
        outer_pixels: MICROENVIRONMENT patch size in level-0 pixels.
        model_input_size: Target size (pixels) fed to the model.
        inner_strategy: "reflect_pad" or "skip_cellular".

    Returns:
        `(inner_rgb, outer_rgb)` where `inner_rgb` is `None` when the
        CELLULAR branch is skipped, or None on extraction failure.
    """
    try:
        # ---- MICROENVIRONMENT patch ----
        outer_half = outer_pixels // 2
        outer_region = slide.read_region(
            (cx - outer_half, cy - outer_half),
            read_plan["outer_level"],
            (read_plan["outer_read_size"], read_plan["outer_read_size"]),
        )
        outer_arr = np.array(outer_region)[..., :3]
        outer_rgb = cv2.resize(
            outer_arr, (model_input_size, model_input_size),
            interpolation=cv2.INTER_AREA
        )

        # ---- CELLULAR patch ----
        if inner_strategy == "reflect_pad":
            inner_half = inner_pixels // 2
            inner_region = slide.read_region(
                (cx - inner_half, cy - inner_half), 0, (inner_pixels, inner_pixels)
            )
            inner_arr = np.array(inner_region)[..., :3]

            # Symmetric pad to model_input_size
            h, w = inner_arr.shape[:2]
            pad_top = (model_input_size - h) // 2
            pad_bottom = model_input_size - h - pad_top
            pad_left = (model_input_size - w) // 2
            pad_right = model_input_size - w - pad_left

            # If inner_arr is already larger (shouldn't happen if inner_pixels<model_input_size
            # but guard against edge cases)
            if pad_top < 0 or pad_bottom < 0 or pad_left < 0 or pad_right < 0:
                inner_rgb = cv2.resize(inner_arr, (model_input_size, model_input_size),
                                       interpolation=cv2.INTER_AREA)
            else:
                inner_rgb = cv2.copyMakeBorder(
                    inner_arr,
                    pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101,
                )
        elif inner_strategy == "skip_cellular":
            inner_rgb = None
        else:
            raise ValueError(f"Unknown inner strategy: {inner_strategy}")

        return inner_rgb, outer_rgb

    except Exception:
        return None


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_embeddings_h5(
    path: str,
    embeddings: np.ndarray,
    coords: np.ndarray,
    metadata: dict,
    logger: logging.Logger,
) -> None:
    """Save embeddings and metadata to a gzip-compressed HDF5 file.

    Args:
        path: Destination file path (parent dirs created if needed).
        embeddings: (N, D) float32 array.
        coords: (N, 2) int32 array of level-0 center coordinates.
        metadata: Dict of scalar/string metadata fields.
        logger: Logger instance.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=embeddings.astype(np.float32),
                         compression="gzip", compression_opts=4)
        f.create_dataset("coords", data=coords.astype(np.int32))

        for key, val in metadata.items():
            if isinstance(val, str):
                f.create_dataset(key, data=np.bytes_(val))
            elif isinstance(val, (int, np.integer)):
                f.create_dataset(key, data=np.int32(val))
            elif isinstance(val, (float, np.floating)):
                f.create_dataset(key, data=np.float32(val))
            else:
                f.create_dataset(key, data=np.bytes_(str(val)))

    n = embeddings.shape[0]
    if n == 0:
        logger.warning(f"HDF5 at {path} has 0 embeddings!")
    else:
        logger.debug(f"Saved {n} embeddings → {path}")


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    slide_path: str,
    cancer_type: str,
    models: Dict[str, torch.nn.Module],
    cfg: ExtractionConfig,
    mask_cache_dir: str,
    device: torch.device,
    logger: logging.Logger,
    is_first_in_cancer_type: bool,
) -> bool:
    """Run the full extraction pipeline for a single slide.

    Args:
        slide_path: Absolute path to the .svs file.
        cancer_type: TCGA cancer type (e.g. "BLCA").
        models: Dict mapping model_name → loaded eval-mode model.
        cfg: Pipeline configuration.
        mask_cache_dir: Directory for caching tissue masks.
        device: Torch device.
        logger: Logger instance.
        is_first_in_cancer_type: If True, log pixel sizes at INFO level.

    Returns:
        True on success, False if slide was skipped or errored.
    """
    t0 = time.time()
    slide_id = Path(slide_path).stem

    try:
        pending_outputs = {
            "CELLULAR": {},
            "MICROENVIRONMENT": {},
        }
        for model_name in models:
            model_tag = model_name.upper().replace("-", "_")
            for scale_label in pending_outputs:
                h5_path = os.path.join(
                    cfg.output_dir, model_tag, cancer_type, scale_label, f"{slide_id}.h5"
                )
                if os.path.exists(h5_path) and not cfg.overwrite:
                    logger.info(f"  [{model_name}|{scale_label}] already exists, skipping.")
                    continue
                pending_outputs[scale_label][model_name] = h5_path

        logger.info(f"Slide: {slide_path}")
        if not pending_outputs["CELLULAR"] and not pending_outputs["MICROENVIRONMENT"]:
            logger.info("  All outputs already exist, skipping slide.")
            return True

        slide = open_slide(slide_path)

        # Step 1: MPP detection
        mpp = detect_mpp(slide, logger)
        if mpp is None:
            logger.warning(f"Skipping {slide_id}: MPP could not be determined.")
            slide.close()
            return False
        logger.info(f"  MPP={mpp:.4f}")

        # Step 2: Pixel sizes
        inner_pixels = round(INNER_PATCH_UM / mpp)
        outer_pixels = round(OUTER_PATCH_UM / mpp)
        stride_pixels = round(cfg.stride_um / mpp)
        upsample_factor = MODEL_INPUT_SIZE / inner_pixels

        if is_first_in_cancer_type:
            logger.info(
                f"  Pixel sizes for {cancer_type}: "
                f"inner={inner_pixels}px ({INNER_PATCH_UM}µm), "
                f"outer={outer_pixels}px ({OUTER_PATCH_UM}µm), "
                f"stride={stride_pixels}px ({cfg.stride_um}µm)"
            )
        logger.info(
            f"  inner_pixels={inner_pixels}px -> effective scale factor to {MODEL_INPUT_SIZE}: "
            f"{upsample_factor:.2f}x"
        )

        # Step 3: Cellular strategy
        inner_strategy = get_inner_strategy(inner_pixels)
        logger.info(
            f"  Cellular strategy: {inner_strategy} "
            f"(inner_pixels={inner_pixels}px, mpp={mpp:.4f})"
        )
        if inner_strategy == "skip_cellular":
            logger.warning(
                f"  CELLULAR scale skipped for {slide_id}: "
                f"inner_pixels={inner_pixels}px at MPP={mpp:.4f} is too small for reliable "
                f"cellular-scale embedding. Only MICROENVIRONMENT embeddings will be saved."
            )
            pending_outputs["CELLULAR"] = {}
            if not pending_outputs["MICROENVIRONMENT"]:
                logger.info("  All remaining outputs already exist, skipping slide.")
                slide.close()
                return True

        # Step 4: Tissue mask
        mask, scale_factor = compute_tissue_mask(slide, slide_id, mask_cache_dir, logger)

        # Step 5: Valid coordinates
        centers = get_valid_centers(
            slide, mask, scale_factor,
            inner_pixels, outer_pixels, stride_pixels,
            cfg.matter_threshold, logger,
        )
        logger.info(f"  Valid coordinates: {len(centers)}")
        if len(centers) == 0:
            logger.warning(f"  No valid coordinates — skipping slide.")
            slide.close()
            return False

        # Step 6: Read planning
        read_plan = build_patch_read_plan(
            slide, outer_pixels, MODEL_INPUT_SIZE
        )

        # Step 7: Extract all patch pairs
        inner_patches: List[np.ndarray] = []
        outer_patches: List[np.ndarray] = []
        valid_centers: List[Tuple[int, int]] = []

        for cx, cy in tqdm(centers, desc="  extracting patches", leave=False):
            result = extract_patch_pair(
                slide, cx, cy,
                inner_pixels, outer_pixels,
                MODEL_INPUT_SIZE, inner_strategy, read_plan,
            )
            if result is None:
                logger.debug(f"  Patch extraction failed at cx={cx} cy={cy}")
                continue
            inner_rgb, outer_rgb = result
            outer_patches.append(outer_rgb)
            valid_centers.append((cx, cy))
            if inner_rgb is not None:
                inner_patches.append(inner_rgb)

        slide.close()

        if len(valid_centers) == 0:
            logger.warning(f"  All patch extractions failed — skipping slide.")
            return False

        coords_arr = np.array(valid_centers, dtype=np.int32)  # (N, 2)

        # Step 8: Shared inference + HDF5 save for each scale
        scale_patches = {
            "CELLULAR": inner_patches,
            "MICROENVIRONMENT": outer_patches,
        }
        for scale_label, patches in scale_patches.items():
            scale_models = {
                model_name: models[model_name]
                for model_name in pending_outputs[scale_label]
            }
            if not scale_models:
                continue

            logger.info(
                f"  [{scale_label}] running shared inference on {len(patches)} patches "
                f"for {len(scale_models)} model(s)"
            )
            embeddings_by_model = run_inference_on_patches(
                scale_models, patches, cfg.batch_size, cfg.num_workers, device,
            )

            for model_name, embeddings in embeddings_by_model.items():
                logger.info(f"  [{model_name}|{scale_label}] embedding shape: {embeddings.shape}")
                metadata = {
                    "slide_path": slide_path,
                    "cancer_type": cancer_type,
                    "model_name": model_name,
                    "scale": scale_label,
                    "mpp": mpp,
                    "inner_patch_um": INNER_PATCH_UM,
                    "outer_patch_um": OUTER_PATCH_UM,
                    "inner_patch_pixels": inner_pixels,
                    "outer_patch_pixels": outer_pixels,
                    "stride_um": cfg.stride_um,
                    "inner_resize_strategy": inner_strategy,
                }
                save_embeddings_h5(
                    pending_outputs[scale_label][model_name],
                    embeddings,
                    coords_arr,
                    metadata,
                    logger,
                )
                del embeddings, metadata

            del embeddings_by_model

        # Cleanup
        del inner_patches, outer_patches, valid_centers, coords_arr
        gc.collect()

        elapsed = time.time() - t0
        logger.info(f"  Slide done in {elapsed:.1f}s")
        return True

    except Exception:
        logger.error(f"Slide {slide_path} FAILED:\n{traceback.format_exc()}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Main entry point: discover TCGA slides, load models, run extraction."""
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    # Device
    try:
        device_str = resolve_device(args.device, logger)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logger.info(f"Using device: {device}")

    # Save run config
    config_path = os.path.join(args.output_dir, "run_config.json")
    run_config = vars(args).copy()
    run_config["resolved_device"] = device_str
    run_config["timestamp"] = datetime.now().isoformat()
    with open(config_path, "w") as fp:
        json.dump(run_config, fp, indent=2)
    logger.info(f"Run config saved to {config_path}")

    cfg = ExtractionConfig(
        tcga_root=args.tcga_root,
        output_dir=args.output_dir,
        cancer_types=args.cancer_types or [],
        stride_um=args.stride_um,
        batch_size=args.batch_size,
        device=device_str,
        seed=args.seed,
        max_slides=args.max_slides,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        matter_threshold=args.matter_threshold,
    )

    # Discover cancer types
    tcga_root = args.tcga_root
    all_dirs = sorted(
        d for d in os.listdir(tcga_root)
        if os.path.isdir(os.path.join(tcga_root, d))
    )
    if cfg.cancer_types:
        cancer_types = [c for c in all_dirs if c in cfg.cancer_types]
        missing = set(cfg.cancer_types) - set(cancer_types)
        if missing:
            logger.warning(f"Requested cancer types not found: {missing}")
    else:
        cancer_types = all_dirs
    logger.info(f"Cancer types to process: {cancer_types}")

    # Mask cache directory
    mask_cache_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(mask_cache_dir, exist_ok=True)
    logger.info(f"Mask cache directory: {mask_cache_dir}")

    # Load all models once
    logger.info("Loading all models …")
    models: Dict[str, torch.nn.Module] = {}
    for mname in ALL_MODEL_NAMES:
        logger.info(f"  Loading {mname}")
        m = load_extractor(mname)
        m.eval()
        m.to(device)
        models[mname] = m
    logger.info("All models loaded.")

    # Main loop
    for cancer_type in cancer_types:
        candidate_dirs = [
            os.path.join(tcga_root, cancer_type, "slides"),
            os.path.join(tcga_root, cancer_type, "slide"),
        ]

        slides_dir = None
        for d in candidate_dirs:
            if os.path.isdir(d):
                slides_dir = d
                break

        if slides_dir is None:
            logger.warning(f"No slide directory found for {cancer_type} — skipping.")
            continue
        
        svs_files = sorted(Path(slides_dir).glob("*.svs"))
        if not svs_files:
            logger.warning(f"No .svs files found in {slides_dir}")
            continue

        if cfg.max_slides is not None:
            svs_files = svs_files[: cfg.max_slides]

        logger.info(f"=== START {cancer_type}: {len(svs_files)} slides ===")
        n_ok = n_fail = 0

        for idx, svs_path in enumerate(tqdm(svs_files, desc=cancer_type)):
            ok = process_slide(
                slide_path=str(svs_path),
                cancer_type=cancer_type,
                models=models,
                cfg=cfg,
                mask_cache_dir=mask_cache_dir,
                device=device,
                logger=logger,
                is_first_in_cancer_type=(idx == 0),
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1

        logger.info(
            f"=== END {cancer_type}: {n_ok} succeeded, {n_fail} failed/skipped ==="
        )

    logger.info("Extraction complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-scale patch feature extraction for TCGA WSIs."
    )

    # New required arguments
    parser.add_argument(
        "--tcga_root", type=str, required=True,
        help="Path to TCGA root directory (contains cancer-type subdirs).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for HDF5 files, logs, and config.",
    )

    # Optional new arguments
    parser.add_argument(
        "--cancer_types", nargs="*", default=[],
        help="Cancer types to process (default: all discovered).",
    )
    parser.add_argument(
        "--stride_um", type=float, default=40.0,
        help="Stride between tile centers in microns (default: 40.0).",
    )
    parser.add_argument(
        "--max_slides", type=int, default=None,
        help="Maximum slides per cancer type (for debugging).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing HDF5 outputs.",
    )

    # Preserved arguments
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device: cpu, cuda, cuda:N, or N (for example: 0 or cuda:0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--matter_threshold", type=float, default=0.50,
        help="Minimum tissue coverage fraction per tile (default: 0.50).",
    )

    args = parser.parse_args()
    main(args)
