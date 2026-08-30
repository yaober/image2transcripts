"""Fig 3 — Main model comparison.

Panel (a): horizontal bar chart of per-cell Pearson for all 7 evaluated
           models, grouped into multimodal (ours) vs frozen-encoder baselines.
Panel (b): grouped bar chart of per-gene Pearson broken out by gene-sparsity
           tertile (low / medium / high zero rate) for 4 representative models,
           showing where ZINB vs MSE differ and where ridge falls short.

Usage::

    python scripts/figures/fig3_main_comparison.py \
        --out figs/fig3_main_comparison.pdf
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── colour palette ──────────────────────────────────────────────────────────
C_OURS     = "#2166ac"   # deep blue  — our multimodal models
C_OURS2    = "#74add1"   # light blue — ZINB+contrast (ours, no align)
C_MSE      = "#4dac26"   # green      — MSE baseline (ours, point regressor)
C_FOUND    = "#d6604d"   # salmon     — foundation + ridge
C_VIT      = "#999999"   # grey       — ViT + ridge

# ── data ────────────────────────────────────────────────────────────────────

MODELS_A = [
    # (label, r_cell, colour, hatch, group)
    ("Full model\n(ZINB+contrast+align)",   0.5368, C_OURS,  "",    "ours"),
    ("MSE baseline\n(contrast+align)",      0.5538, C_MSE,   "",    "ours"),
    ("ZINB+contrast",                       0.5273, C_OURS2, "",    "ours"),
    ("UNI2-h + ridge",                      0.4967, C_FOUND, "//",  "baseline"),
    ("Phikon2 + ridge",                     0.4650, C_FOUND, "//",  "baseline"),
    ("PG + ridge",                          0.4646, C_FOUND, "//",  "baseline"),
    ("ViT + ridge",                         0.4501, C_VIT,   "..",  "baseline"),
]

# sorted descending by r_cell for the bar chart
MODELS_A = sorted(MODELS_A, key=lambda x: x[1], reverse=True)

SPARSITY_BINS  = ["Low\n(<0.88)", "Medium\n(0.88–0.97)", "High\n(>0.97)"]
# per-gene Pearson by sparsity tertile
MODELS_B = [
    ("Full model",     [0.5058, 0.3396, 0.2023], C_OURS),
    ("MSE baseline",   [0.5000, 0.3531, 0.2318], C_MSE),
    ("UNI2-h + ridge", [0.4232, 0.2814, 0.1441], C_FOUND),
    ("ViT + ridge",    [0.3224, 0.2078, 0.0947], C_VIT),
]

# ── figure ──────────────────────────────────────────────────────────────────

def make_figure(out: Path) -> None:
    fig = plt.figure(figsize=(13, 5.2))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38,
                   left=0.02, right=0.98, top=0.88, bottom=0.16)

    # ── panel (a): per-cell Pearson horizontal bar ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    labels  = [m[0] for m in MODELS_A]
    vals    = [m[1] for m in MODELS_A]
    colors  = [m[2] for m in MODELS_A]
    hatches = [m[3] for m in MODELS_A]
    n = len(MODELS_A)
    y = np.arange(n)

    for i, (v, c, h) in enumerate(zip(vals, colors, hatches)):
        ax_a.barh(i, v, color=c, hatch=h, edgecolor="white",
                  height=0.62, linewidth=0)
        ax_a.text(v + 0.003, i, f"{v:.3f}", va="center", ha="left",
                  fontsize=8.5, color="#222222", fontweight="bold")

    # vertical guide at the ViT+ridge baseline
    ridge_val = 0.4501
    ax_a.axvline(ridge_val, color=C_VIT, lw=1.2, linestyle=":", alpha=0.7,
                 zorder=0)
    ax_a.text(ridge_val - 0.002, n - 0.1, "ViT+ridge\nbaseline",
              ha="right", va="top", fontsize=7, color=C_VIT, style="italic")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=9)
    ax_a.set_xlim(0.38, 0.60)
    ax_a.set_xlabel("Per-cell Pearson  ($r_\\mathrm{cell}$)", fontsize=10)
    ax_a.set_title("(a)  Per-cell Pearson on held-out test set\n"
                   "(640,928 cells · leakage-safe split)",
                   fontsize=10, pad=8)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.tick_params(left=False)

    # group separating line between ours / baselines
    sep_y = 2.5   # between index 2 (ZINB+contrast) and 3 (UNI2-h)
    ax_a.axhline(sep_y, color="#bbbbbb", lw=0.9, linestyle="--")
    ax_a.text(0.381, sep_y + 0.15, "Multimodal (ours)",
              fontsize=7.5, color="#555555", style="italic")
    ax_a.text(0.381, sep_y - 0.55, "Frozen-encoder baselines",
              fontsize=7.5, color="#888888", style="italic")

    # ── panel (b): per-gene Pearson by sparsity ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    n_bins   = len(SPARSITY_BINS)
    n_models = len(MODELS_B)
    width    = 0.18
    offsets  = np.linspace(-(n_models - 1) * width / 2,
                            (n_models - 1) * width / 2, n_models)
    x = np.arange(n_bins)

    for offset, (mlabel, vals_b, col) in zip(offsets, MODELS_B):
        bars = ax_b.bar(x + offset, vals_b, width=width, color=col,
                        alpha=0.88, label=mlabel,
                        edgecolor="white", linewidth=0.4)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SPARSITY_BINS, fontsize=9.5)
    ax_b.set_xlabel("Gene sparsity bin  (observed zero rate)", fontsize=10)
    ax_b.set_ylabel("Per-gene Pearson  ($r_\\mathrm{gene}$)", fontsize=10)
    ax_b.set_title("(b)  Per-gene Pearson by sparsity tertile\n"
                   "(372 genes · low = least sparse)",
                   fontsize=10, pad=8)
    ax_b.set_ylim(0, 0.60)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(fontsize=8.5, frameon=False, loc="upper right",
                bbox_to_anchor=(1.01, 1.0))

    # annotation: MSE leads on high sparsity, ZINB leads on low
    ax_b.annotate("ZINB leads\non low sparsity",
                  xy=(0 + offsets[0], 0.508), xytext=(-0.35, 0.55),
                  fontsize=7.5, color=C_OURS, style="italic",
                  arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.8))
    ax_b.annotate("MSE leads\non high sparsity",
                  xy=(2 + offsets[1], 0.232), xytext=(1.65, 0.31),
                  fontsize=7.5, color=C_MSE, style="italic",
                  arrowprops=dict(arrowstyle="->", color=C_MSE, lw=0.8))

    fig.suptitle(
        "Image2Transcript — main results on the leakage-safe held-out test set",
        fontsize=11, y=0.97, fontweight="bold"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  and  {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("figs/fig3_main_comparison.pdf"))
    args = ap.parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
