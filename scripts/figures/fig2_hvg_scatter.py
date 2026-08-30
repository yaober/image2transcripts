"""Fig 2 — Spatial expression maps and predicted-vs-observed scatter.

2 rows × 3 columns:
  col 0: spatial map of observed log1p(count) on one CA tissue slide
  col 1: spatial map of predicted log1p(mu) on the same slide
  col 2: hexbin scatter — predicted vs observed across all 640 k test cells

Genes:
  CEACAM5 — colorectal tumor marker (CEA, r=0.769, zero_rate=0.582)
  CDCA7   — proliferation / cell-cycle (Ki-67 co-marker, r=0.692)

Spatial slide: CA551C (149 938 cells, colorectal adenocarcinoma).

Usage::

    python scripts/figures/fig2_hvg_scatter.py \
        --eval_dir runs/fixedsplit_v3/full_seed42/eval/test \
        --out figs/fig2_spatial_scatter.pdf
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import anndata

# ── gene selection ──────────────────────────────────────────────────────────
GENES = [
    ("CEACAM5", "Tumor marker (CEA)",  "Reds",   "#b2182b"),
    ("CDCA7",   "Proliferation",       "Purples", "#6a3d9a"),
]

# CA551C: colorectal adenocarcinoma, 149 938 cells
SPATIAL_H5AD      = "data/gene_expression/output-XETG00248__0010315__CA551C__20240411__220137.h5ad"
SPATIAL_IMG_DIR   = "data/images/output-XETG00248__0010315__CA551C__20240411__220137"
SPATIAL_SLIDE_IDX = 4   # index in eval/test/slide_names.json


# ── helpers ─────────────────────────────────────────────────────────────────

def load_eval(eval_dir: Path):
    mu        = np.load(eval_dir / "mu.npy").astype(np.float32)
    x_true    = np.load(eval_dir / "x_true.npy").astype(np.float32)
    slide_idx = np.load(eval_dir / "slide_idx.npy")
    with open(eval_dir / "analysis" / "per_gene_pearson.csv") as f:
        rows = list(csv.DictReader(f))
    gene_names = [r["gene"] for r in rows]
    pearson    = {r["gene"]: float(r["pearson"]) for r in rows}
    return mu, x_true, slide_idx, gene_names, pearson


def load_spatial(h5ad_path: str, img_dir: str, gene_names: list[str]):
    h5 = anndata.read_h5ad(h5ad_path)
    img_root  = Path(img_dir)
    valid     = np.array([(img_root / f"{cid}.png").exists()
                          for cid in h5.obs_names])
    coords    = h5.obs[["x_centroid", "y_centroid"]].values[valid]
    h5_genes  = list(h5.var_names)
    g_idx     = [h5_genes.index(g) for g in gene_names]
    X = h5.X.toarray() if hasattr(h5.X, "toarray") else np.asarray(h5.X)
    expr = X[valid][:, g_idx].astype(np.float32)
    return coords[:, 0], coords[:, 1], expr


def draw_spatial(ax, x, y, values, cmap_name, pt_size=0.25):
    v    = np.log1p(values)
    vmax = np.percentile(v[v > 0], 97) if (v > 0).any() else 1.0
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("#f7f7f7")          # near-zero → very light grey
    sc = ax.scatter(x, -y, c=v, cmap=cmap, s=pt_size, linewidths=0,
                    vmin=1e-3, vmax=vmax, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc, vmax


def draw_scatter(ax, x_obs, y_pred, r, cmap_name, accent):
    x = np.log1p(x_obs)
    y = np.log1p(y_pred)
    lim = max(x.max(), y.max()) * 1.08

    # build a cmap that starts at a visible mid-tone (not white)
    base = plt.get_cmap(cmap_name)
    colors_cut = base(np.linspace(0.25, 1.0, 256))   # cut off the pale end
    cmap = mcolors.LinearSegmentedColormap.from_list("cut", colors_cut)

    ax.hexbin(x, y, gridsize=55, cmap=cmap, mincnt=1,
              norm=mcolors.LogNorm(vmin=1, vmax=None),
              extent=[0, lim, 0, lim], linewidths=0.0)

    ax.plot([0, lim], [0, lim], color="#555555", lw=1.0,
            linestyle="--", alpha=0.85, zorder=5)

    ax.text(0.96, 0.06, f"$r$ = {r:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=accent,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=accent, lw=0.8, alpha=0.9))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


# ── main ────────────────────────────────────────────────────────────────────

def make_figure(eval_dir: Path, out: Path,
                scatter_n: int = 100_000, seed: int = 0):

    mu, x_true, slide_idx, gene_names, pearson = load_eval(eval_dir)
    x_sp, y_sp, expr_sp = load_spatial(SPATIAL_H5AD, SPATIAL_IMG_DIR, gene_names)

    sp_mask = slide_idx == SPATIAL_SLIDE_IDX
    mu_sp   = mu[sp_mask]

    rng = np.random.default_rng(seed)
    sel = rng.choice(x_true.shape[0], size=min(scatter_n, x_true.shape[0]),
                     replace=False)
    x_sc = x_true[sel]
    m_sc = mu[sel]

    nrows, ncols = len(GENES), 3
    fig = plt.figure(figsize=(12, 5.0 * nrows))
    gs  = GridSpec(nrows, ncols, figure=fig,
                   hspace=0.12, wspace=0.22,
                   left=0.03, right=0.97, top=0.91, bottom=0.05)

    # ── column headers ──
    col_labels = [
        f"Observed  (CA551C, {sp_mask.sum():,} cells)",
        "Predicted",
        f"All test cells  ({x_true.shape[0]:,} cells)",
    ]
    for c, lbl in enumerate(col_labels):
        cx = 0.03 + (c + 0.5) * (0.94 / ncols)
        fig.text(cx, 0.935, lbl, ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#333333")

    for row, (gene, label, cmap_name, accent) in enumerate(GENES):
        if gene not in gene_names:
            print(f"[warn] {gene} not in panel"); continue
        g_idx = gene_names.index(gene)
        r     = pearson[gene]

        # col 0 — observed spatial
        ax0 = fig.add_subplot(gs[row, 0])
        sc0, vmax = draw_spatial(ax0, x_sp, y_sp, expr_sp[:, g_idx], cmap_name)
        ax0.set_title(f"{gene}  —  {label}", fontsize=11,
                      fontweight="bold", color=accent, pad=5)
        cb0 = plt.colorbar(sc0, ax=ax0, fraction=0.033, pad=0.01,
                           extend="min")
        cb0.set_label("log(1 + count)", fontsize=7.5)
        cb0.ax.tick_params(labelsize=7)

        # col 1 — predicted spatial  (same vmax as observed for direct comparison)
        ax1 = fig.add_subplot(gs[row, 1])
        cmap_obj = plt.get_cmap(cmap_name).copy()
        cmap_obj.set_under("#f7f7f7")
        v_pred = np.log1p(mu_sp[:, g_idx])
        sc1 = ax1.scatter(x_sp, -y_sp, c=v_pred, cmap=cmap_obj, s=0.25,
                          linewidths=0, vmin=1e-3, vmax=vmax, rasterized=True)
        ax1.set_aspect("equal"); ax1.axis("off")
        cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.033, pad=0.01,
                           extend="min")
        cb1.set_label("log(1 + μ)", fontsize=7.5)
        cb1.ax.tick_params(labelsize=7)

        # col 2 — scatter
        ax2 = fig.add_subplot(gs[row, 2])
        draw_scatter(ax2, x_sc[:, g_idx], m_sc[:, g_idx], r, cmap_name, accent)
        ax2.set_xlabel("log(1 + observed count)", fontsize=9)
        ax2.set_ylabel("log(1 + predicted μ)", fontsize=9)

    fig.suptitle(
        "Image2Transcript: spatial prediction and per-gene accuracy\n"
        "Spatial slide: CA551C (colorectal adenocarcinoma)  ·  "
        "Scatter: leakage-safe held-out test set",
        fontsize=10, y=0.975
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  and  {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path,
                    default=Path("runs/fixedsplit_v3/full_seed42/eval/test"))
    ap.add_argument("--out", type=Path,
                    default=Path("figs/fig2_spatial_scatter.pdf"))
    ap.add_argument("--scatter_n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_figure(args.eval_dir, args.out, args.scatter_n, args.seed)


if __name__ == "__main__":
    main()
