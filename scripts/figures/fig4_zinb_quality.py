"""Fig 4 — ZINB generative quality suite.

2 × 2 panels:
  (a) top-left:  Per-gene zero-rate calibration scatter
                 (predicted P(x=0) vs observed zero rate, one dot per gene)
  (b) top-right: Cell-gene reliability diagram
                 (10 uniform bins of predicted P(x=0) vs empirical zero rate)
  (c) bot-left:  Zero/non-zero AUROC by gene sparsity tertile
                 (full model vs ZINB+contrast)
  (d) bot-right: Held-out ZINB NLL by gene sparsity tertile
                 (full model vs ZINB+contrast)

All metrics computed on the frozen held-out test set (640,928 cells, 372 genes).

Usage::

    python scripts/figures/fig4_zinb_quality.py \
        --out figs/fig4_zinb_quality.pdf
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.special import gammaln

# ── palette ──────────────────────────────────────────────────────────────────
C_FULL  = "#2166ac"   # deep blue  — full model
C_ZC    = "#74add1"   # light blue — ZINB+contrast

SPARSITY_LABELS = ["Low\n(<0.88)", "Medium\n(0.88–0.97)", "High\n(>0.97)"]

# ── data helpers ─────────────────────────────────────────────────────────────

def load_zinb_arrays(eval_dir: Path):
    mu   = np.load(eval_dir / "mu.npy").astype(np.float64)
    th   = np.clip(np.load(eval_dir / "theta.npy").astype(np.float64), 1e-6, 1e6)
    pil  = np.load(eval_dir / "pi_logits.npy").astype(np.float64)
    x    = np.load(eval_dir / "x_true.npy").astype(np.float64)
    pi   = 1.0 / (1.0 + np.exp(-pil))
    nb0  = np.exp(th * (np.log(th + 1e-12) - np.log(th + mu + 1e-12)))
    p0   = pi + (1.0 - pi) * nb0
    # NLL
    log_nb = (gammaln(x + th) - gammaln(th) - gammaln(x + 1)
              + th * (np.log(th) - np.log(th + mu))
              + x  * (np.log(mu + 1e-12) - np.log(th + mu)))
    logp   = np.where(x == 0,
                      np.logaddexp(np.log(pi + 1e-12),
                                   np.log(1 - pi + 1e-12) + log_nb),
                      np.log(1 - pi + 1e-12) + log_nb)
    nll = -logp
    return x, p0, nll


def per_gene_auroc(x, p0):
    n_genes = x.shape[1]
    auroc   = np.full(n_genes, np.nan)
    for g in range(n_genes):
        lbl  = (x[:, g] == 0)
        npos = lbl.sum(); nneg = len(lbl) - npos
        if npos == 0 or nneg == 0:
            continue
        sc    = p0[:, g]
        order = np.argsort(sc)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        auroc[g] = (ranks[lbl].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return auroc


def sparsity_bins(eval_dir: Path):
    with open(eval_dir / "analysis" / "per_gene_pearson.csv") as f:
        rows = list(csv.DictReader(f))
    zr   = np.array([float(r["zero_rate_obs"]) for r in rows])
    t33, t67 = np.percentile(zr, 33), np.percentile(zr, 67)
    masks = [zr < t33, (zr >= t33) & (zr < t67), zr >= t67]
    return zr, masks


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(eval_full: Path, eval_zc: Path, out: Path) -> None:

    # --- load both models ---
    x_f, p0_f, nll_f = load_zinb_arrays(eval_full)
    x_z, p0_z, nll_z = load_zinb_arrays(eval_zc)

    obs0_gene  = (x_f == 0).mean(axis=0)
    pred0_gene = p0_f.mean(axis=0)
    r_cal      = np.corrcoef(pred0_gene, obs0_gene)[0, 1]
    mae_cal    = np.abs(pred0_gene - obs0_gene).mean()

    # reliability diagram (10 bins, cell-gene level)
    bins_edges  = np.linspace(0, 1, 11)
    flat_pred   = p0_f.reshape(-1)
    flat_obs    = (x_f == 0).reshape(-1).astype(np.float32)
    bidx        = np.clip(np.digitize(flat_pred, bins_edges) - 1, 0, 9)
    rel_centers, rel_obs, rel_counts = [], [], []
    for b in range(10):
        m = bidx == b
        if m.sum() > 0:
            rel_centers.append(bins_edges[b] + 0.05)
            rel_obs.append(float(flat_obs[m].mean()))
            rel_counts.append(int(m.sum()))

    # AUROC and NLL by sparsity
    zr, sp_masks = sparsity_bins(eval_full)
    auroc_f = per_gene_auroc(x_f, p0_f)
    auroc_z = per_gene_auroc(x_z, p0_z)
    pg_nll_f = nll_f.mean(axis=0)
    pg_nll_z = nll_z.mean(axis=0)

    auroc_f_bins = [auroc_f[m][np.isfinite(auroc_f[m])].mean() for m in sp_masks]
    auroc_z_bins = [auroc_z[m][np.isfinite(auroc_z[m])].mean() for m in sp_masks]
    nll_f_bins   = [pg_nll_f[m].mean() for m in sp_masks]
    nll_z_bins   = [pg_nll_z[m].mean() for m in sp_masks]

    # ── layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.38, wspace=0.32,
                   left=0.09, right=0.97, top=0.91, bottom=0.08)

    # ── (a) zero-rate calibration scatter ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    sc = ax_a.scatter(pred0_gene, obs0_gene, s=18, alpha=0.75,
                      c=obs0_gene, cmap="Blues", vmin=0.3, vmax=1.0,
                      edgecolors="none", zorder=3)
    ax_a.plot([0, 1], [0, 1], color="#c0392b", lw=1.1,
              linestyle="--", alpha=0.85, label="$y = x$")
    ax_a.set_xlim(0.3, 1.02); ax_a.set_ylim(0.3, 1.02)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("Predicted $P(x=0)$  (per gene)", fontsize=10)
    ax_a.set_ylabel("Observed zero rate  (per gene)", fontsize=10)
    ax_a.set_title(f"(a)  Zero-rate calibration\n"
                   f"$r$ = {r_cal:.3f}   MAE = {mae_cal:.3f}   "
                   f"$n$ = 372 genes",
                   fontsize=10, pad=6)
    cb = plt.colorbar(sc, ax=ax_a, fraction=0.04, pad=0.02)
    cb.set_label("Observed zero rate", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── (b) reliability diagram ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    rc = np.array(rel_centers)
    ro = np.array(rel_obs)
    rn = np.array(rel_counts, dtype=float)
    rn_norm = rn / rn.max()

    ax_b.plot([0, 1], [0, 1], color="#c0392b", lw=1.1,
              linestyle="--", alpha=0.85, label="Ideal", zorder=2)
    ax_b.scatter(rc, ro, s=80 + 420 * rn_norm,
                 color=C_FULL, alpha=0.82, zorder=3,
                 label="Bin  (size ∝ count)")
    ax_b.plot(rc, ro, color=C_FULL, lw=1.0, alpha=0.5, zorder=2)

    # shade calibration gap
    ax_b.fill_between(rc, rc, ro, alpha=0.08, color=C_FULL)

    ax_b.set_xlim(-0.02, 1.02); ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Predicted $P(x=0)$  (bin centre)", fontsize=10)
    ax_b.set_ylabel("Empirical zero rate in bin", fontsize=10)
    ax_b.set_title("(b)  Cell-gene reliability diagram\n"
                   f"({x_f.shape[0]:,} cells × {x_f.shape[1]} genes)",
                   fontsize=10, pad=6)
    ax_b.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_b.spines[["top", "right"]].set_visible(False)

    # ── (c) AUROC by sparsity ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(3)
    w = 0.32
    bars_f = ax_c.bar(x_pos - w/2, auroc_f_bins, width=w,
                      color=C_FULL, label="Full model", alpha=0.88,
                      edgecolor="white")
    bars_z = ax_c.bar(x_pos + w/2, auroc_z_bins, width=w,
                      color=C_ZC,   label="ZINB+contrast", alpha=0.88,
                      edgecolor="white")
    ax_c.axhline(0.5, color="#aaaaaa", lw=1.0, linestyle=":",
                 label="Chance (0.5)")
    ax_c.axhline(np.nanmean(auroc_f), color=C_FULL, lw=1.0,
                 linestyle="--", alpha=0.6, label=f"Macro avg {np.nanmean(auroc_f):.3f}")

    for bar, val in zip(list(bars_f) + list(bars_z),
                        auroc_f_bins + auroc_z_bins):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                  f"{val:.3f}", ha="center", va="bottom",
                  fontsize=7.5, color="#333333")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(SPARSITY_LABELS, fontsize=9.5)
    ax_c.set_xlabel("Gene sparsity bin  (observed zero rate)", fontsize=10)
    ax_c.set_ylabel("AUROC  (zero vs non-zero)", fontsize=10)
    ax_c.set_title("(c)  Zero/non-zero discrimination by sparsity\n"
                   "Score = predicted $P(x=0)$",
                   fontsize=10, pad=6)
    ax_c.set_ylim(0.45, 0.92)
    ax_c.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax_c.spines[["top", "right"]].set_visible(False)

    # ── (d) NLL by sparsity ───────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    bars_f2 = ax_d.bar(x_pos - w/2, nll_f_bins, width=w,
                       color=C_FULL, label="Full model", alpha=0.88,
                       edgecolor="white")
    bars_z2 = ax_d.bar(x_pos + w/2, nll_z_bins, width=w,
                       color=C_ZC,   label="ZINB+contrast", alpha=0.88,
                       edgecolor="white")

    for bar, val in zip(list(bars_f2) + list(bars_z2),
                        nll_f_bins + nll_z_bins):
        ax_d.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                  f"{val:.3f}", ha="center", va="bottom",
                  fontsize=7.5, color="#333333")

    # delta annotations (full - zc, negative = full is better)
    for i, (vf, vz) in enumerate(zip(nll_f_bins, nll_z_bins)):
        delta = vf - vz
        sign  = "−" if delta < 0 else "+"
        ax_d.text(i, max(vf, vz) + 0.04,
                  f"Δ{sign}{abs(delta):.3f}",
                  ha="center", va="bottom", fontsize=7.5,
                  color=C_FULL if delta < 0 else C_ZC,
                  fontweight="bold")

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(SPARSITY_LABELS, fontsize=9.5)
    ax_d.set_xlabel("Gene sparsity bin  (observed zero rate)", fontsize=10)
    ax_d.set_ylabel("Mean NLL  (nats per cell-gene pair)", fontsize=10)
    ax_d.set_title("(d)  Held-out ZINB NLL by sparsity\n"
                   "Lower = better generative fit",
                   fontsize=10, pad=6)
    ax_d.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Image2Transcript — ZINB generative quality on the held-out test set\n"
        "640,928 cells · 372 genes · leakage-safe slide-group split",
        fontsize=11, y=0.975, fontweight="bold"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  and  {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_full", type=Path,
                    default=Path("runs/fixedsplit_v3/full_seed42/eval/test"))
    ap.add_argument("--eval_zc", type=Path,
                    default=Path("runs/fixedsplit_v3/zinb_contrast_seed42/eval/test"))
    ap.add_argument("--out", type=Path,
                    default=Path("figs/fig4_zinb_quality.pdf"))
    args = ap.parse_args()
    make_figure(args.eval_full, args.eval_zc, args.out)


if __name__ == "__main__":
    main()
