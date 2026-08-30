#!/usr/bin/env python3
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

for extra_path in ("/tmp/pydeps39", "/tmp/pydeps"):
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.append(extra_path)

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
import tifffile
import zarr


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate framework-figure reference panels from a real morphology slide patch and matched transcripts."
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="Slide ID without .h5ad suffix. Defaults to the first slide that has both .h5ad and image folder.",
    )
    parser.add_argument("--gene-dir", type=Path, default=Path("data/gene_expression"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/framework_figure_assets"))
    parser.add_argument(
        "--transcript-col",
        type=str,
        default="transcript_counts",
        help="Column in adata.obs used for the transcript plot.",
    )
    parser.add_argument(
        "--roi-size",
        type=float,
        default=45.0,
        help="Size of the main square ROI in transcript coordinate units.",
    )
    parser.add_argument(
        "--zoom-size",
        type=float,
        default=18.0,
        help="Size of the zoomed square ROI in transcript coordinate units.",
    )
    parser.add_argument(
        "--display-size-px",
        type=int,
        default=768,
        help="Display size in pixels for the patch and zoom panels.",
    )
    parser.add_argument(
        "--raw-slide-dir",
        type=Path,
        default=None,
        help="Optional directory containing the slide raw Xenium outputs for this slide ID.",
    )
    return parser.parse_args()


def choose_default_slide(gene_dir: Path, image_dir: Path) -> str:
    candidates = sorted(
        fp.stem for fp in gene_dir.glob("*.h5ad") if (image_dir / fp.stem).is_dir()
    )
    if not candidates:
        raise FileNotFoundError("No slide found with both .h5ad and image folder.")
    return candidates[0]


def resolve_raw_slide_dir(slide: str, provided_dir: Optional[Path]):
    candidates = []
    if provided_dir is not None:
        candidates.append(Path(provided_dir))

    candidates.extend(
        [
            Path("../xenium32/data/Xenium_Emina_Munir/Xenium_data/run2") / slide,
            Path("../xenium32/data/gene_expression/raw") / slide,
        ]
    )

    for path in candidates:
        focus_path = path / "morphology_focus" / "morphology_focus_0000.ome.tif"
        fov_path = path / "aux_outputs" / "morphology_fov_locations.json"
        if focus_path.is_file() and fov_path.is_file():
            return path

    raise FileNotFoundError(
        f"Could not find raw slide directory with morphology_focus image for slide {slide}"
    )


def load_slide_table(slide: str, gene_dir: Path, image_dir: Path, transcript_col: str):
    adata = ad.read_h5ad(gene_dir / f"{slide}.h5ad", backed="r")
    required_cols = ["x_centroid", "y_centroid", transcript_col]
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing columns in {slide}.h5ad: {missing}")

    obs = adata.obs[required_cols].copy()
    image_paths = {p.stem: p for p in (image_dir / slide).glob("*.png")}
    common_ids = obs.index.intersection(image_paths.keys())
    if common_ids.empty:
        raise FileNotFoundError(f"No matching image crops found for slide {slide}")

    obs = obs.loc[common_ids].copy()
    obs["image_path"] = [str(image_paths[cell_id]) for cell_id in obs.index]
    return obs


def choose_dense_roi(obs, size: float):
    xs = obs["x_centroid"].to_numpy(dtype=np.float32)
    ys = obs["y_centroid"].to_numpy(dtype=np.float32)

    x_bins = np.arange(xs.min(), xs.max() + size, size, dtype=np.float32)
    y_bins = np.arange(ys.min(), ys.max() + size, size, dtype=np.float32)
    if len(x_bins) < 2:
        x_bins = np.array([xs.min(), xs.max()], dtype=np.float32)
    if len(y_bins) < 2:
        y_bins = np.array([ys.min(), ys.max()], dtype=np.float32)

    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    return {
        "x0": float(x_edges[max_idx[0]]),
        "x1": float(x_edges[max_idx[0] + 1]),
        "y0": float(y_edges[max_idx[1]]),
        "y1": float(y_edges[max_idx[1] + 1]),
        "cell_count": int(hist[max_idx]),
    }


def subset_roi(obs, roi):
    keep = (
        (obs["x_centroid"] >= roi["x0"])
        & (obs["x_centroid"] <= roi["x1"])
        & (obs["y_centroid"] >= roi["y0"])
        & (obs["y_centroid"] <= roi["y1"])
    )
    return obs.loc[keep].copy()


def build_centered_roi(center_x: float, center_y: float, size: float, obs):
    half = size / 2.0
    x_min = float(obs["x_centroid"].min())
    x_max = float(obs["x_centroid"].max())
    y_min = float(obs["y_centroid"].min())
    y_max = float(obs["y_centroid"].max())

    x0 = max(x_min, center_x - half)
    x1 = min(x_max, center_x + half)
    y0 = max(y_min, center_y - half)
    y1 = min(y_max, center_y + half)

    if (x1 - x0) < size:
        if x0 <= x_min:
            x1 = min(x_max, x0 + size)
        else:
            x0 = max(x_min, x1 - size)
    if (y1 - y0) < size:
        if y0 <= y_min:
            y1 = min(y_max, y0 + size)
        else:
            y0 = max(y_min, y1 - size)

    return {
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(y0),
        "y1": float(y1),
    }


def choose_representative_cell(obs, roi):
    center_x = (roi["x0"] + roi["x1"]) / 2.0
    center_y = (roi["y0"] + roi["y1"]) / 2.0
    coords = obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    dist2 = (coords[:, 0] - center_x) ** 2 + (coords[:, 1] - center_y) ** 2
    idx = int(np.argmin(dist2))
    return obs.iloc[idx]


def load_focus_metadata(raw_slide_dir: Path):
    with open(raw_slide_dir / "aux_outputs" / "morphology_fov_locations.json") as f:
        fov_locations = json.load(f)["fov_locations"]

    x0 = min(v["x"] for v in fov_locations.values())
    y0 = min(v["y"] for v in fov_locations.values())
    x1 = max(v["x"] + v["width"] for v in fov_locations.values())
    y1 = max(v["y"] + v["height"] for v in fov_locations.values())

    focus_path = raw_slide_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
    with tifffile.TiffFile(focus_path) as tf:
        height_px, width_px = tf.series[0].shape

    return {
        "focus_path": focus_path,
        "coord_extent": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "shape_px": {"width": width_px, "height": height_px},
        "px_per_unit_x": width_px / max(x1 - x0, 1e-6),
        "px_per_unit_y": height_px / max(y1 - y0, 1e-6),
    }


def normalize_focus_crop(crop):
    lo, hi = np.percentile(crop, [1.0, 99.7])
    scaled = np.clip((crop - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def crop_focus_patch(focus_meta, center_x: float, center_y: float, size_um: float, out_size_px: int):
    focus_path = focus_meta["focus_path"]
    px_per_unit_x = focus_meta["px_per_unit_x"]
    px_per_unit_y = focus_meta["px_per_unit_y"]
    x_offset = focus_meta["coord_extent"]["x0"]
    y_offset = focus_meta["coord_extent"]["y0"]

    center_px_x = int(round((center_x - x_offset) * px_per_unit_x))
    center_px_y = int(round((center_y - y_offset) * px_per_unit_y))
    half_w = max(1, int(round(size_um * px_per_unit_x / 2.0)))
    half_h = max(1, int(round(size_um * px_per_unit_y / 2.0)))

    with tifffile.TiffFile(focus_path) as tf:
        arr = zarr.open(tf.series[0].aszarr(level=0), mode="r")
        img_h, img_w = arr.shape
        x0 = max(0, center_px_x - half_w)
        x1 = min(img_w, center_px_x + half_w)
        y0 = max(0, center_px_y - half_h)
        y1 = min(img_h, center_px_y + half_h)
        crop = np.asarray(arr[y0:y1, x0:x1])

    crop_8bit = normalize_focus_crop(crop)
    image = Image.fromarray(crop_8bit, mode="L").resize((out_size_px, out_size_px), RESAMPLE_BICUBIC)
    return np.asarray(image)


def center_zoom_rect(image_shape, zoom_fraction: float):
    height, width = image_shape[:2]
    zoom_fraction = float(np.clip(zoom_fraction, 0.1, 0.95))
    w = width * zoom_fraction
    h = height * zoom_fraction
    x = (width - w) / 2.0
    y = (height - h) / 2.0
    return x, y, w, h


def save_histology_panel(image, zoom_fraction: float, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    ax.imshow(image)
    rect = center_zoom_rect(image.shape, zoom_fraction)
    ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, lw=2.5, ec="#00c2a8"))
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def transcript_marker_size(n_points: int):
    if n_points <= 10:
        return 80
    if n_points <= 40:
        return 42
    if n_points <= 100:
        return 24
    if n_points <= 500:
        return 10
    return 3


def save_transcript_panel(obs, roi, zoom_roi, transcript_col: str, output_path: Path):
    values = obs[transcript_col].to_numpy(dtype=np.float32)
    vmax = float(np.quantile(values, 0.99)) if len(values) else 1.0
    vmax = max(vmax, 1.0)
    marker_size = transcript_marker_size(len(obs))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    scatter = ax.scatter(
        obs["x_centroid"],
        obs["y_centroid"],
        c=values,
        s=marker_size,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_xlim(roi["x0"], roi["x1"])
    ax.set_ylim(roi["y1"], roi["y0"])
    ax.set_aspect("equal")
    ax.axis("off")
    if zoom_roi is not None:
        ax.add_patch(
            Rectangle(
                (zoom_roi["x0"], zoom_roi["y0"]),
                zoom_roi["x1"] - zoom_roi["x0"],
                zoom_roi["y1"] - zoom_roi["y0"],
                fill=False,
                lw=2.5,
                ec="#00c2a8",
            )
        )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel(transcript_col, rotation=90)
    ax.set_title("Transcript Spatial Plot", fontsize=12)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_figure(patch_img, zoom_fraction, patch_roi, patch_zoom_roi, transcript_obs, transcript_col, zoom_img, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=220)

    axes[0].imshow(patch_img)
    rect = center_zoom_rect(patch_img.shape, zoom_fraction)
    axes[0].add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, lw=2.5, ec="#00c2a8"))
    axes[0].set_title("Morphology Patch", fontsize=12)
    axes[0].axis("off")

    values = transcript_obs[transcript_col].to_numpy(dtype=np.float32)
    vmax = float(np.quantile(values, 0.99)) if len(values) else 1.0
    vmax = max(vmax, 1.0)
    marker_size = transcript_marker_size(len(transcript_obs))
    scatter = axes[1].scatter(
        transcript_obs["x_centroid"],
        transcript_obs["y_centroid"],
        c=values,
        s=marker_size,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.0,
        rasterized=True,
    )
    axes[1].add_patch(
        Rectangle(
            (patch_zoom_roi["x0"], patch_zoom_roi["y0"]),
            patch_zoom_roi["x1"] - patch_zoom_roi["x0"],
            patch_zoom_roi["y1"] - patch_zoom_roi["y0"],
            fill=False,
            lw=2.5,
            ec="#00c2a8",
        )
    )
    axes[1].set_xlim(patch_roi["x0"], patch_roi["x1"])
    axes[1].set_ylim(patch_roi["y1"], patch_roi["y0"])
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].set_title("Transcript Spatial Plot", fontsize=12)
    fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.02)

    axes[2].imshow(zoom_img)
    axes[2].set_title("Zoomed Morphology", fontsize=12)
    axes[2].axis("off")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    slide = args.slide or choose_default_slide(args.gene_dir, args.image_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    obs = load_slide_table(slide, args.gene_dir, args.image_dir, args.transcript_col)
    raw_slide_dir = resolve_raw_slide_dir(slide, args.raw_slide_dir)
    focus_meta = load_focus_metadata(raw_slide_dir)

    dense_roi = choose_dense_roi(obs, max(args.roi_size, 800.0))
    center_cell = choose_representative_cell(obs, dense_roi)
    center_x = float(center_cell["x_centroid"])
    center_y = float(center_cell["y_centroid"])

    patch_roi = build_centered_roi(center_x, center_y, args.roi_size, obs)
    zoom_roi = build_centered_roi(center_x, center_y, args.zoom_size, obs)
    patch_obs = subset_roi(obs, patch_roi)
    zoom_obs = subset_roi(obs, zoom_roi)

    zoom_fraction = args.zoom_size / max(args.roi_size, 1e-6)
    patch_img = crop_focus_patch(focus_meta, center_x, center_y, args.roi_size, args.display_size_px)
    zoom_img = crop_focus_patch(focus_meta, center_x, center_y, args.zoom_size, args.display_size_px)

    patch_path = args.out_dir / f"{slide}_histology_patch.png"
    transcript_path = args.out_dir / f"{slide}_transcript_patch.png"
    zoom_path = args.out_dir / f"{slide}_histology_zoom.png"
    combined_path = args.out_dir / f"{slide}_framework_reference.png"
    meta_path = args.out_dir / f"{slide}_framework_reference.json"

    save_histology_panel(patch_img, zoom_fraction, patch_path, "Morphology Patch")
    save_transcript_panel(patch_obs, patch_roi, zoom_roi, args.transcript_col, transcript_path)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    ax.imshow(zoom_img)
    ax.set_title("Zoomed Morphology", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(zoom_path, bbox_inches="tight")
    plt.close(fig)

    save_combined_figure(
        patch_img,
        zoom_fraction,
        patch_roi,
        zoom_roi,
        patch_obs,
        args.transcript_col,
        zoom_img,
        combined_path,
    )

    metadata = {
        "slide": slide,
        "transcript_column": args.transcript_col,
        "histology_source": "raw_morphology_focus",
        "raw_slide_dir": str(raw_slide_dir),
        "raw_focus_image": str(focus_meta["focus_path"]),
        "pixel_scale": {
            "px_per_unit_x": focus_meta["px_per_unit_x"],
            "px_per_unit_y": focus_meta["px_per_unit_y"],
        },
        "selected_cell_id": str(center_cell.name),
        "selected_cell_image": str(center_cell["image_path"]),
        "matched_cells": int(len(obs)),
        "patch_cells": int(len(patch_obs)),
        "zoom_cells": int(len(zoom_obs)),
        "dense_roi_for_selection": dense_roi,
        "patch_roi": patch_roi,
        "zoom_roi": zoom_roi,
        "outputs": {
            "histology_patch": str(patch_path),
            "transcript_patch": str(transcript_path),
            "histology_zoom": str(zoom_path),
            "combined_reference": str(combined_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
