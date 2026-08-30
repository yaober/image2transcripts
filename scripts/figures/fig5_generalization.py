"""Fig 5 — Generalization: HEST-Bench cross-cancer + TCGA-COAD clinical.

Panel (a): HEST-Bench — grouped bar chart of mean Pearson per cancer type
           (Image2Transcript vs three competing methods), with per-fold std
           as error bars.  Highest bar highlighted per cancer type.

Panel (b): TCGA-COAD — horizontal lollipop chart of the max −log10 p per
           clinical variable (p75 aggregation, K=200 permutations).
           Observed value shown as a filled circle; null 95th-percentile shown
           as a dashed vertical line per variable.  Significant variables
           (emp p < 0.05) coloured, non-significant in grey.

Usage::

    python scripts/figures/fig5_generalization.py \
        --null_csv runs/tcga_coad/analysis_aaai/permutation_null.csv \
        --out figs/fig5_generalization.pdf
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── HEST-Bench data (from manuscript Table 10) ───────────────────────────────
# (mean ± std, number of folds given in manuscript)
CANCER_TYPES = ["IDC", "PAAD", "SKCM", "COAD", "LUNG"]

HEST = {
    "GHIST\n(random init)": {
        "mean": [0.340, 0.303, 0.335, 0.126, 0.242],
        "std":  [0.054, 0.039, 0.030, 0.024, 0.133],
    },
    "sCellST": {
        "mean": [0.182, 0.238, 0.337, 0.139, 0.231],
        "std":  [0.000, 0.065, 0.090, 0.004, 0.025],
    },
    "SciSt": {
        "mean": [0.286, 0.274, 0.198, 0.385, 0.180],
        "std":  [0.041, 0.037, 0.016, 0.149, 0.134],
    },
    "Image2Transcript\n(ours)": {
        "mean": [0.436, 0.372, 0.482, 0.200, 0.510],
        "std":  [0.035, 0.070, 0.066, 0.068, 0.013],
    },
}

METHOD_COLORS = {
    "GHIST\n(random init)":  "#bbbbbb",
    "sCellST":               "#fc8d59",
    "SciSt":                 "#d9ef8b",
    "Image2Transcript\n(ours)": "#2166ac",
}

# ── TCGA variable display names (ordered for the figure) ────────────────────
VAR_LABELS = {
    "tumor_status":    "Recurrence\n(with-tumour at f/u)",
    "vital_status":    "Vital status",
    "age":             "Age",
    "overall_survival":"Overall survival",
    "msi":             "MSI status",
    "t_ordinal":       "T stage",
    "kras_mut":        "KRAS mutation",
    "n_binary":        "Lymph-node N0/N+",
    "stage_binary":    "AJCC stage (early/late)",
    "gender":          "Gender",
    "m_binary":        "Distant met M0/M1",
}
VAR_ORDER = list(VAR_LABELS.keys())   # top → bottom = most → least significant

C_SIG   = "#2166ac"    # emp p < 0.05
C_TREND = "#74add1"    # 0.05 ≤ emp p < 0.10
C_NS    = "#aaaaaa"    # not significant


# ── helpers ──────────────────────────────────────────────────────────────────

def load_null(csv_path: Path, aggregation: str = "p75") -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    data = {}
    for r in rows:
        if r["aggregation"] != aggregation:
            continue
        data[r["variable"]] = {
            "obs":     float(r["obs_max_nlp"]),
            "null_m":  float(r["null_nlp_mean"]),
            "null_p95":float(r["null_nlp_p95"]),
            "emp_p":   float(r["empirical_p"]),
        }
    return data


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(null_csv: Path, out: Path) -> None:
    null = load_null(null_csv, aggregation="p75")

    fig = plt.figure(figsize=(14, 6.5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.40,
                   left=0.04, right=0.98, top=0.88, bottom=0.12,
                   width_ratios=[1.35, 1.0])

    # ─────────────────────────────────────────────────────────────────────────
    # Panel (a): HEST-Bench
    # ─────────────────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    methods = list(HEST.keys())
    n_m     = len(methods)
    n_c     = len(CANCER_TYPES)
    w       = 0.17
    offsets = np.linspace(-(n_m - 1) * w / 2, (n_m - 1) * w / 2, n_m)
    x_pos   = np.arange(n_c)

    for offset, method in zip(offsets, methods):
        means = HEST[method]["mean"]
        stds  = HEST[method]["std"]
        color = METHOD_COLORS[method]
        ec    = "#333333" if "ours" in method else "white"
        lw    = 1.0 if "ours" in method else 0.3
        ax_a.bar(x_pos + offset, means, width=w,
                 yerr=stds, capsize=2.5,
                 color=color, label=method,
                 edgecolor=ec, linewidth=lw, alpha=0.90,
                 error_kw=dict(elinewidth=0.9, ecolor="#555555"))

    # star the best bar per cancer type
    for ci, ct in enumerate(CANCER_TYPES):
        col_means = {m: HEST[m]["mean"][ci] for m in methods}
        best_m    = max(col_means, key=col_means.get)
        best_v    = col_means[best_m]
        best_off  = offsets[methods.index(best_m)]
        ax_a.text(ci + best_off, best_v + HEST[best_m]["std"][ci] + 0.018,
                  "★", ha="center", va="bottom", fontsize=9,
                  color=METHOD_COLORS[best_m])

    # macro-average dashed lines
    for method in methods:
        macro = np.mean(HEST[method]["mean"])
        if "ours" in method:
            ax_a.axhline(macro, color=METHOD_COLORS[method],
                         lw=1.1, linestyle="--", alpha=0.55,
                         label=f"Ours macro avg {macro:.3f}")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(CANCER_TYPES, fontsize=11)
    ax_a.set_ylabel("Mean Pearson (cross-validation folds)", fontsize=10)
    ax_a.set_ylim(0, 0.62)
    ax_a.set_title("(a)  HEST-Bench cross-cancer generalization\n"
                   "Frozen Image2Transcript encoder + ridge head  ·  no fine-tuning",
                   fontsize=10, pad=7)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=7.8, frameon=False, loc="upper left",
                ncol=1, bbox_to_anchor=(0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Panel (b): TCGA lollipop
    # ─────────────────────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # order: most significant first (= as given by VAR_ORDER)
    var_keys = [v for v in VAR_ORDER if v in null]
    y_pos    = np.arange(len(var_keys))[::-1]   # top = most significant

    for yi, var in zip(y_pos, var_keys):
        d     = null[var]
        obs   = d["obs"]
        p95   = d["null_p95"]
        emp_p = d["emp_p"]

        if emp_p < 0.05:
            col = C_SIG
        elif emp_p < 0.10:
            col = C_TREND
        else:
            col = C_NS

        # stem
        ax_b.plot([0, obs], [yi, yi], color=col, lw=1.5, alpha=0.7)
        # dot
        ax_b.scatter(obs, yi, color=col, s=55, zorder=4)
        # null 95th percentile tick
        ax_b.plot([p95, p95], [yi - 0.32, yi + 0.32],
                  color="#888888", lw=1.2, alpha=0.7, zorder=3)

        # p-value annotation
        if emp_p <= 0.005:
            plabel = "emp $p$ = 0.005*"
        elif emp_p < 0.05:
            plabel = f"emp $p$ = {emp_p:.3f}*"
        elif emp_p < 0.10:
            plabel = f"emp $p$ = {emp_p:.2f}"
        else:
            plabel = "n.s."
        ax_b.text(max(obs, p95) + 0.15, yi, plabel,
                  va="center", ha="left", fontsize=7.5,
                  color=col if emp_p < 0.10 else C_NS)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([VAR_LABELS[v] for v in var_keys], fontsize=9)
    ax_b.set_xlabel("Max $-\\log_{10}\\,p$ across 372 genes", fontsize=10)
    ax_b.set_title("(b)  TCGA-COAD zero-shot clinical associations\n"
                   "p75 aggregation · K = 200 permutation null",
                   fontsize=10, pad=7)

    # null 95th percentile legend marker
    ax_b.plot([], [], color="#888888", lw=1.2, label="Null 95th pctl (per variable)")
    sig_patch  = mpatches.Patch(color=C_SIG,   label="emp $p$ < 0.05")
    ns_patch   = mpatches.Patch(color=C_NS,    label="Not significant")
    ax_b.legend(handles=[sig_patch, ns_patch,
                          plt.Line2D([0],[0], color="#888888", lw=1.2,
                                     label="Null 95th pctl")],
                fontsize=8, frameon=False, loc="lower right")

    xlim_max = max(d["obs"] for d in null.values()) * 1.08
    ax_b.set_xlim(-0.5, min(xlim_max, 50))
    ax_b.spines[["top", "right"]].set_visible(False)

    # log-scale break for recurrence (43.2 is an outlier)
    ax_b.axvline(10, color="#dddddd", lw=0.8, linestyle="--", zorder=0)
    ax_b.text(10.2, -0.6, "axis break →", fontsize=7, color="#aaaaaa", va="top")

    fig.suptitle(
        "Image2Transcript — generalization: cross-cancer benchmark and clinical transfer",
        fontsize=11, y=0.97, fontweight="bold"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  and  {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null_csv", type=Path,
                    default=Path("runs/tcga_coad/analysis_aaai/permutation_null.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("figs/fig5_generalization.pdf"))
    args = ap.parse_args()
    make_figure(args.null_csv, args.out)


if __name__ == "__main__":
    main()
