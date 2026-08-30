"""Analyse a prediction dump produced by ``scripts/eval_fixedsplit.py``.

Computes five metric families per split directory:

1. Per-cell Pearson correlation between predicted ZINB mean and observed counts.
2. Per-gene Pearson correlation across cells.
3. ZINB calibration (predicted vs observed zero rate per gene; mu calibration
   bucketed by predicted magnitude).
4. Retrieval recall@{1,5,10} between image and gene embeddings on a stratified
   subsample of cells, averaged across ``--retrieval_seeds`` resamples.
5. Per-slide and CA / NL breakdown of per-cell Pearson.

Outputs land under ``<eval_dir>/<split>/analysis/``:

    summary.json             flat dict with all the main numbers
    summary.md               a short human-readable report
    per_gene_pearson.csv     one row per gene with counts and correlations
    per_slide_pearson.csv    one row per slide (cells, mean, median, std)
    figures/
        per_gene_pearson_hist.png
        per_gene_calibration.png
        pi_reliability.png
        per_slide_pearson_bar.png

All figures are generated only when matplotlib is available; if import fails
the numeric outputs are still written and the script exits cleanly.

Usage::

    python scripts/analyze_predictions.py \
        --eval_dir runs/fixedsplit_v3/full_seed42/eval \
        --splits val test

The script can also analyse a baseline where only ``mu.npy`` and ``x_true.npy``
are present (e.g. the ridge baseline); ZINB and retrieval blocks are skipped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover - plotting is best-effort
    HAS_MPL = False


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------


@dataclass
class Dump:
    mu: np.ndarray
    theta: np.ndarray | None
    pi_logits: np.ndarray | None
    i_emb: np.ndarray | None
    g_emb: np.ndarray | None
    x_true: np.ndarray
    slide_idx: np.ndarray
    slide_names: list[str]
    metadata: dict

    @property
    def n_cells(self) -> int:
        return self.x_true.shape[0]

    @property
    def n_genes(self) -> int:
        return self.x_true.shape[1]


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def load_dump(split_dir: Path) -> Dump:
    mu = np.load(split_dir / "mu.npy")
    theta = _load_optional(split_dir / "theta.npy")
    pi_logits = _load_optional(split_dir / "pi_logits.npy")
    i_emb = _load_optional(split_dir / "i_emb.npy")
    g_emb = _load_optional(split_dir / "g_emb.npy")
    x_true = np.load(split_dir / "x_true.npy")
    slide_idx = np.load(split_dir / "slide_idx.npy")

    with open(split_dir / "slide_names.json") as f:
        slide_names = json.load(f)

    meta_fp = split_dir / "metadata.json"
    metadata = json.load(open(meta_fp)) if meta_fp.exists() else {}

    return Dump(mu=mu, theta=theta, pi_logits=pi_logits,
                i_emb=i_emb, g_emb=g_emb, x_true=x_true,
                slide_idx=slide_idx, slide_names=slide_names,
                metadata=metadata)


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------


def _pearson_axis1(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise Pearson correlation, returning NaN for zero-variance rows."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    r = (am * bm).sum(axis=1) / (denom + eps)
    r[denom < eps] = np.nan
    return r


def _pearson_column(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise Pearson correlation (same shape assumptions as ``_pearson_axis1``)."""
    return _pearson_axis1(a.T, b.T)


def per_cell_pearson(dump: Dump) -> dict:
    r = _pearson_axis1(dump.mu, dump.x_true)
    valid = np.isfinite(r)
    return {
        "n_cells_total": int(dump.n_cells),
        "n_cells_valid": int(valid.sum()),
        "mean": float(np.nanmean(r)),
        "median": float(np.nanmedian(r)),
        "std": float(np.nanstd(r)),
        "p05": float(np.nanpercentile(r, 5)),
        "p25": float(np.nanpercentile(r, 25)),
        "p75": float(np.nanpercentile(r, 75)),
        "p95": float(np.nanpercentile(r, 95)),
        "raw": r,
    }


def per_gene_pearson(dump: Dump) -> dict:
    r = _pearson_column(dump.mu, dump.x_true)
    valid = np.isfinite(r)
    return {
        "n_genes": int(len(r)),
        "n_genes_valid": int(valid.sum()),
        "mean": float(np.nanmean(r)),
        "median": float(np.nanmedian(r)),
        "std": float(np.nanstd(r)),
        "raw": r,
    }


def zinb_zero_rate_calibration(dump: Dump) -> dict | None:
    """Per-gene predicted vs observed zero rate; requires theta and pi."""
    if dump.theta is None or dump.pi_logits is None:
        return None

    mu = dump.mu.astype(np.float32)
    theta = np.clip(dump.theta.astype(np.float32), 1e-6, 1e6)
    pi = 1.0 / (1.0 + np.exp(-dump.pi_logits.astype(np.float32)))

    # P(x=0) under ZINB = pi + (1 - pi) * (theta / (theta + mu)) ** theta
    nb_zero = np.exp(theta * (np.log(theta + 1e-12) - np.log(theta + mu + 1e-12)))
    p_zero = pi + (1.0 - pi) * nb_zero

    predicted_per_gene = p_zero.mean(axis=0)
    observed_per_gene = (dump.x_true == 0).mean(axis=0).astype(np.float32)

    # Reliability diagram bins the cell-gene pair-level predictions against
    # the binary outcome (x == 0) and reports mean observed rate per bin.
    bins = np.linspace(0.0, 1.0, 11)  # 10 uniform bins
    flat_pred = p_zero.reshape(-1)
    flat_obs = (dump.x_true == 0).reshape(-1).astype(np.float32)
    bin_idx = np.clip(np.digitize(flat_pred, bins) - 1, 0, len(bins) - 2)
    reliability_bins = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            reliability_bins.append({
                "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
                "count": 0, "mean_predicted": float("nan"),
                "mean_observed": float("nan"),
            })
            continue
        reliability_bins.append({
            "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
            "count": count,
            "mean_predicted": float(flat_pred[mask].mean()),
            "mean_observed": float(flat_obs[mask].mean()),
        })

    diff = predicted_per_gene - observed_per_gene
    return {
        "per_gene_predicted": predicted_per_gene,
        "per_gene_observed": observed_per_gene,
        "per_gene_signed_error_mean": float(diff.mean()),
        "per_gene_abs_error_mean": float(np.abs(diff).mean()),
        "pearson_pred_vs_obs_per_gene": float(np.corrcoef(
            predicted_per_gene, observed_per_gene)[0, 1]),
        "reliability_bins": reliability_bins,
    }


def zero_nonzero_auroc_by_sparsity(dump: Dump, n_sparsity_bins: int = 4) -> dict | None:
    """Per-gene AUROC for zero/non-zero classification, grouped by gene sparsity.

    Score = P(x=0) under ZINB.  Label = (x_true == 0).  Genes are bucketed
    into ``n_sparsity_bins`` equal-width bins of observed zero rate.  Returns
    per-bin mean ± std AUROC plus the macro-average across all valid genes.
    """
    if dump.theta is None or dump.pi_logits is None:
        return None

    mu = dump.mu.astype(np.float32)
    theta = np.clip(dump.theta.astype(np.float32), 1e-6, 1e6)
    pi = 1.0 / (1.0 + np.exp(-dump.pi_logits.astype(np.float32)))
    nb_zero = np.exp(theta * (np.log(theta + 1e-12) - np.log(theta + mu + 1e-12)))
    p_zero = pi + (1.0 - pi) * nb_zero  # (n_cells, n_genes)

    obs_zero_rate = (dump.x_true == 0).mean(axis=0)  # (n_genes,)

    # Per-gene AUROC via the Wilcoxon-Mann-Whitney U statistic (no sklearn needed).
    per_gene_auroc = np.full(dump.n_genes, np.nan)
    for g in range(dump.n_genes):
        labels = (dump.x_true[:, g] == 0)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        scores = p_zero[:, g]
        # U = sum of ranks of positives minus n_pos*(n_pos+1)/2
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1)
        u_pos = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
        per_gene_auroc[g] = u_pos / (n_pos * n_neg)

    # Bucket by observed zero rate.
    edges = np.linspace(0.0, 1.0, n_sparsity_bins + 1)
    bin_labels = []
    bins_out = []
    for b in range(n_sparsity_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (obs_zero_rate >= lo) & (obs_zero_rate < hi + 1e-9 * (b == n_sparsity_bins - 1))
        auc_vals = per_gene_auroc[in_bin & np.isfinite(per_gene_auroc)]
        label = f"zero_rate [{lo:.2f}, {hi:.2f})"
        bin_labels.append(label)
        if len(auc_vals) == 0:
            bins_out.append({"sparsity_bin": label, "n_genes": 0,
                             "mean_auroc": float("nan"), "std_auroc": float("nan")})
        else:
            bins_out.append({"sparsity_bin": label,
                             "n_genes": int(len(auc_vals)),
                             "mean_auroc": float(auc_vals.mean()),
                             "std_auroc": float(auc_vals.std(ddof=0))})

    valid = per_gene_auroc[np.isfinite(per_gene_auroc)]
    return {
        "macro_mean_auroc": float(valid.mean()) if len(valid) else float("nan"),
        "macro_std_auroc": float(valid.std(ddof=0)) if len(valid) else float("nan"),
        "n_genes_valid": int(len(valid)),
        "per_gene_auroc": per_gene_auroc,
        "per_gene_obs_zero_rate": obs_zero_rate,
        "sparsity_bins": bins_out,
    }


def mu_magnitude_calibration(dump: Dump, n_bins: int = 15) -> dict:
    """Bucket cell-gene pairs by predicted mu magnitude and compare observed counts."""
    mu_flat = dump.mu.astype(np.float32).reshape(-1)
    x_flat = dump.x_true.astype(np.float32).reshape(-1)

    # Use log(1 + mu) bins because mu is heavy-tailed.
    lm = np.log1p(mu_flat)
    edges = np.quantile(lm, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = edges[-1] + 1e-6  # ensure last bin is inclusive
    idx = np.clip(np.digitize(lm, edges) - 1, 0, n_bins - 1)

    out = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            out.append({"bin": b, "count": 0})
            continue
        out.append({
            "bin": b,
            "count": count,
            "mu_lower": float(np.expm1(edges[b])),
            "mu_upper": float(np.expm1(edges[b + 1])),
            "mean_mu": float(mu_flat[mask].mean()),
            "mean_x": float(x_flat[mask].mean()),
        })
    return {"bins": out}


def retrieval_metrics(dump: Dump, n_query: int = 4000,
                      seeds: list[int] | None = None,
                      topks: tuple[int, ...] = (1, 5, 10, 50),
                      stratified: bool = True) -> dict | None:
    if dump.i_emb is None or dump.g_emb is None:
        return None

    seeds = seeds or [0, 1, 2]
    i = dump.i_emb.astype(np.float32)
    g = dump.g_emb.astype(np.float32)

    # Re-normalize in case rounding during fp16 dump perturbed the norm.
    i /= np.linalg.norm(i, axis=1, keepdims=True) + 1e-8
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-8

    slide_idx = dump.slide_idx
    results_per_seed = []

    for s in seeds:
        rng = np.random.default_rng(s)

        if stratified:
            unique_slides = np.unique(slide_idx)
            per_slide = max(1, n_query // len(unique_slides))
            picked: list[int] = []
            for sl in unique_slides:
                pool = np.where(slide_idx == sl)[0]
                take = min(per_slide, len(pool))
                picked.extend(rng.choice(pool, size=take, replace=False).tolist())
            if len(picked) < n_query:
                leftover = np.setdiff1d(np.arange(dump.n_cells), np.asarray(picked))
                if len(leftover):
                    extra = rng.choice(leftover,
                                       size=min(n_query - len(picked), len(leftover)),
                                       replace=False)
                    picked.extend(extra.tolist())
            sel = np.asarray(picked[:n_query])
        else:
            sel = rng.choice(dump.n_cells,
                             size=min(n_query, dump.n_cells), replace=False)

        Ie, Ge = i[sel], g[sel]
        sims_ig = Ie @ Ge.T  # (K, K) cosine because both are L2-normalized
        sims_gi = sims_ig.T

        labels = np.arange(len(sel))

        seed_result = {"seed": int(s), "n_query": int(len(sel))}
        for direction, sims in (("image_to_gene", sims_ig),
                                  ("gene_to_image", sims_gi)):
            for k in topks:
                k_eff = min(k, sims.shape[1])
                topk = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
                hits = (topk == labels[:, None]).any(axis=1)
                seed_result[f"{direction}_recall@{k}"] = float(hits.mean())
        results_per_seed.append(seed_result)

    agg: dict[str, float] = {}
    keys = [k for k in results_per_seed[0].keys() if k.startswith(("image_to_gene", "gene_to_image"))]
    for k in keys:
        vals = np.asarray([r[k] for r in results_per_seed])
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=0))

    return {
        "per_seed": results_per_seed,
        "aggregate": agg,
        "n_query": n_query,
        "stratified": stratified,
    }


def per_slide_breakdown(dump: Dump) -> dict:
    r = _pearson_axis1(dump.mu, dump.x_true)
    rows = []
    for idx, name in enumerate(dump.slide_names):
        mask = dump.slide_idx == idx
        if not mask.any():
            continue
        vals = r[mask]
        valid = np.isfinite(vals)
        rows.append({
            "slide_idx": int(idx),
            "slide": name,
            "n_cells": int(mask.sum()),
            "n_valid": int(valid.sum()),
            "mean_pearson": float(np.nanmean(vals)),
            "median_pearson": float(np.nanmedian(vals)),
            "std_pearson": float(np.nanstd(vals)),
            "ca_or_nl": "CA" if "__CA" in name else ("NL" if "__NL" in name else "other"),
        })

    rows.sort(key=lambda x: -x["mean_pearson"])

    by_group: dict[str, list[float]] = {"CA": [], "NL": []}
    for row in rows:
        if row["ca_or_nl"] in by_group:
            by_group[row["ca_or_nl"]].append(row["mean_pearson"])

    group_summary = {
        grp: {
            "n_slides": len(vals),
            "mean_of_means": float(np.mean(vals)) if vals else float("nan"),
            "std_of_means": float(np.std(vals)) if vals else float("nan"),
        }
        for grp, vals in by_group.items()
    }

    return {"per_slide": rows, "group_summary": group_summary}


# ----------------------------------------------------------------------
# Figure helpers (no-ops when matplotlib is missing)
# ----------------------------------------------------------------------


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_per_gene_hist(per_gene: dict, out_fp: Path) -> None:
    if not HAS_MPL:
        return
    r = per_gene["raw"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r[np.isfinite(r)], bins=40, color="#3a6ea5", alpha=0.85)
    ax.set_xlabel("Per-gene Pearson (across cells)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Per-gene Pearson  (mean {per_gene['mean']:.3f}, n={per_gene['n_genes_valid']})")
    ax.axvline(per_gene["mean"], color="#c1434a", linestyle="--", lw=1.2, label="mean")
    ax.axvline(per_gene["median"], color="#2f8f3f", linestyle=":", lw=1.2, label="median")
    ax.legend(frameon=False)
    _save_fig(fig, out_fp)


def fig_per_gene_zero_calibration(zinb: dict, out_fp: Path) -> None:
    if not HAS_MPL or zinb is None:
        return
    pred = zinb["per_gene_predicted"]
    obs = zinb["per_gene_observed"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pred, obs, s=10, alpha=0.6, color="#3a6ea5")
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color="#c1434a", lw=1.0, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted P(x=0), per gene")
    ax.set_ylabel("Observed zero rate, per gene")
    ax.set_title(f"ZINB zero-rate calibration  (r={zinb['pearson_pred_vs_obs_per_gene']:.3f})")
    ax.set_aspect("equal")
    _save_fig(fig, out_fp)


def fig_pi_reliability(zinb: dict, out_fp: Path) -> None:
    if not HAS_MPL or zinb is None:
        return
    rows = [b for b in zinb["reliability_bins"] if b["count"] > 0]
    if not rows:
        return
    centers = np.array([0.5 * (b["bin_lo"] + b["bin_hi"]) for b in rows])
    observed = np.array([b["mean_observed"] for b in rows])
    counts = np.array([b["count"] for b in rows], dtype=float)
    counts /= counts.max()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#c1434a", lw=1.0, linestyle="--", label="ideal")
    ax.scatter(centers, observed, s=40 + 160 * counts,
                alpha=0.75, color="#3a6ea5", label="bin (size ~ count)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(x=0), bin center")
    ax.set_ylabel("Empirical zero rate in bin")
    ax.set_title("ZINB cell-gene reliability diagram")
    ax.legend(frameon=False)
    ax.set_aspect("equal")
    _save_fig(fig, out_fp)


def fig_auroc_by_sparsity(auroc_res: dict, out_fp: Path) -> None:
    if not HAS_MPL or auroc_res is None:
        return
    bins = [b for b in auroc_res["sparsity_bins"] if b["n_genes"] > 0]
    if not bins:
        return
    labels = [b["sparsity_bin"].replace("zero_rate ", "") for b in bins]
    means = np.array([b["mean_auroc"] for b in bins])
    stds = np.array([b["std_auroc"] for b in bins])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bins))
    ax.bar(x, means, yerr=stds, color="#3a6ea5", alpha=0.85, capsize=4)
    ax.axhline(0.5, color="#c1434a", linestyle="--", lw=1.0, label="chance")
    ax.axhline(auroc_res["macro_mean_auroc"], color="#2f8f3f",
               linestyle=":", lw=1.2, label=f"macro avg {auroc_res['macro_mean_auroc']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("AUROC (zero vs non-zero classification)")
    ax.set_title("Zero/non-zero AUROC by gene sparsity  (score = predicted P(x=0))")
    ax.set_ylim(0.45, 1.05)
    ax.legend(frameon=False)
    _save_fig(fig, out_fp)


def fig_per_slide_bar(per_slide: dict, out_fp: Path) -> None:
    if not HAS_MPL:
        return
    rows = per_slide["per_slide"]
    if not rows:
        return
    labels = [r["slide"].split("__")[2] for r in rows]  # e.g. CA425 / NL441
    means = [r["mean_pearson"] for r in rows]
    colors = ["#c1434a" if r["ca_or_nl"] == "CA" else "#3a6ea5" for r in rows]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(rows)), 4))
    ax.bar(range(len(rows)), means, color=colors, alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Per-cell Pearson (slide mean)")
    ax.set_title("Per-slide mean per-cell Pearson  (red=CA, blue=NL)")
    ax.set_ylim(0, max(means) * 1.1)
    _save_fig(fig, out_fp)


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------


def analyse_split(split_dir: Path, retrieval_n: int,
                  retrieval_seeds: list[int]) -> dict:
    print(f"\n=== Analysing {split_dir} ===")
    dump = load_dump(split_dir)
    print(f"  n_cells={dump.n_cells}  n_genes={dump.n_genes}")

    out_dir = split_dir / "analysis"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell = per_cell_pearson(dump)
    print(f"  per-cell Pearson  mean={per_cell['mean']:.4f}  "
          f"median={per_cell['median']:.4f}  std={per_cell['std']:.4f}")

    per_gene = per_gene_pearson(dump)
    print(f"  per-gene Pearson  mean={per_gene['mean']:.4f}  "
          f"median={per_gene['median']:.4f}")

    zinb = zinb_zero_rate_calibration(dump)
    if zinb is not None:
        print(f"  ZINB gene-zero calibration  r={zinb['pearson_pred_vs_obs_per_gene']:.3f}  "
              f"MAE={zinb['per_gene_abs_error_mean']:.3f}")

    auroc_res = zero_nonzero_auroc_by_sparsity(dump)
    if auroc_res is not None:
        print(f"  Zero/non-zero AUROC  macro={auroc_res['macro_mean_auroc']:.3f}  "
              f"(n_genes={auroc_res['n_genes_valid']})")
        for b in auroc_res["sparsity_bins"]:
            if b["n_genes"] > 0:
                print(f"    {b['sparsity_bin']}  n={b['n_genes']}  "
                      f"AUROC={b['mean_auroc']:.3f} ± {b['std_auroc']:.3f}")

    mu_cal = mu_magnitude_calibration(dump)

    retrieval = retrieval_metrics(dump, n_query=retrieval_n, seeds=retrieval_seeds)
    if retrieval is not None:
        agg = retrieval["aggregate"]
        print("  retrieval@k (mean over {} seeds, K={}):".format(
            len(retrieval_seeds), retrieval["n_query"]))
        for direction in ("image_to_gene", "gene_to_image"):
            for k in (1, 5, 10, 50):
                key = f"{direction}_recall@{k}_mean"
                if key in agg:
                    print(f"    {direction} recall@{k}: {agg[key]:.3f}")

    per_slide = per_slide_breakdown(dump)
    print("  per-slide Pearson:")
    for row in per_slide["per_slide"]:
        print(f"    {row['slide']}  cells={row['n_cells']}  "
              f"mean={row['mean_pearson']:.4f}")

    # --- Dump tabular outputs --------------------------------------------------
    gene_names = dump.metadata.get("genes", [str(i) for i in range(dump.n_genes)])
    with open(out_dir / "per_gene_pearson.csv", "w") as f:
        f.write("gene_idx,gene,pearson,mean_pred,mean_obs,zero_rate_pred,zero_rate_obs\n")
        mean_pred = dump.mu.astype(np.float32).mean(axis=0)
        mean_obs = dump.x_true.astype(np.float32).mean(axis=0)
        zero_pred_gene = (zinb["per_gene_predicted"]
                          if zinb is not None else np.full(dump.n_genes, np.nan))
        zero_obs_gene = (zinb["per_gene_observed"]
                         if zinb is not None else (dump.x_true == 0).mean(axis=0).astype(np.float32))
        for i_gene, gene in enumerate(gene_names):
            f.write(f"{i_gene},{gene},{per_gene['raw'][i_gene]:.6f},"
                    f"{mean_pred[i_gene]:.6f},{mean_obs[i_gene]:.6f},"
                    f"{zero_pred_gene[i_gene]:.6f},{zero_obs_gene[i_gene]:.6f}\n")

    with open(out_dir / "per_slide_pearson.csv", "w") as f:
        f.write("slide,cells,mean_pearson,median_pearson,std_pearson,ca_or_nl\n")
        for row in per_slide["per_slide"]:
            f.write(f"{row['slide']},{row['n_cells']},"
                    f"{row['mean_pearson']:.6f},{row['median_pearson']:.6f},"
                    f"{row['std_pearson']:.6f},{row['ca_or_nl']}\n")

    # --- Figures ---------------------------------------------------------------
    fig_per_gene_hist(per_gene, fig_dir / "per_gene_pearson_hist.png")
    fig_per_gene_zero_calibration(zinb, fig_dir / "per_gene_calibration.png")
    fig_pi_reliability(zinb, fig_dir / "pi_reliability.png")
    fig_auroc_by_sparsity(auroc_res, fig_dir / "auroc_by_sparsity.png")
    fig_per_slide_bar(per_slide, fig_dir / "per_slide_pearson_bar.png")

    # --- Summary json ----------------------------------------------------------
    summary = {
        "split_dir": str(split_dir),
        "n_cells": dump.n_cells,
        "n_genes": dump.n_genes,
        "per_cell_pearson": {k: v for k, v in per_cell.items() if k != "raw"},
        "per_gene_pearson": {k: v for k, v in per_gene.items() if k != "raw"},
        "mu_magnitude_calibration": mu_cal,
        "per_slide_breakdown": per_slide,
    }
    if zinb is not None:
        summary["zinb_zero_calibration"] = {
            k: v for k, v in zinb.items()
            if k not in ("per_gene_predicted", "per_gene_observed")
        }
    if auroc_res is not None:
        summary["zero_nonzero_auroc"] = {
            k: v for k, v in auroc_res.items()
            if k not in ("per_gene_auroc", "per_gene_obs_zero_rate")
        }
    if retrieval is not None:
        summary["retrieval"] = retrieval

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _write_summary_md(out_dir / "summary.md", summary, dump, per_cell,
                      per_gene, zinb, retrieval, per_slide, auroc_res)
    return summary


def _write_summary_md(path: Path, summary: dict, dump: Dump,
                       per_cell: dict, per_gene: dict, zinb: dict | None,
                       retrieval: dict | None, per_slide: dict,
                       auroc_res: dict | None = None) -> None:
    lines = [
        f"# Analysis summary — {path.parent.parent.name}",
        "",
        f"- Split dir: `{summary['split_dir']}`",
        f"- Checkpoint: `{dump.metadata.get('ckpt_path', 'unknown')}`",
        f"- Cells: {dump.n_cells:,}",
        f"- Genes: {dump.n_genes}",
        f"- Slides: {len(per_slide['per_slide'])}",
        "",
        "## Per-cell Pearson",
        f"- mean: **{per_cell['mean']:.4f}**",
        f"- median: {per_cell['median']:.4f}",
        f"- std: {per_cell['std']:.4f}",
        f"- [p5, p25, p75, p95] = "
        f"[{per_cell['p05']:.3f}, {per_cell['p25']:.3f}, "
        f"{per_cell['p75']:.3f}, {per_cell['p95']:.3f}]",
        "",
        "## Per-gene Pearson",
        f"- mean: **{per_gene['mean']:.4f}**",
        f"- median: {per_gene['median']:.4f}",
        f"- std: {per_gene['std']:.4f}",
        f"- valid genes: {per_gene['n_genes_valid']}/{per_gene['n_genes']}",
        "",
    ]

    if zinb is not None:
        lines += [
            "## ZINB zero-rate calibration",
            f"- Pearson(predicted zero rate vs observed) per gene: "
            f"**{zinb['pearson_pred_vs_obs_per_gene']:.3f}**",
            f"- Mean absolute error on per-gene zero rate: "
            f"{zinb['per_gene_abs_error_mean']:.3f}",
            f"- Signed bias (pred - obs) on per-gene zero rate: "
            f"{zinb['per_gene_signed_error_mean']:+.3f}",
            "",
        ]

    if auroc_res is not None:
        lines += [
            "## Zero/non-zero AUROC by gene sparsity",
            f"- Macro-average AUROC: **{auroc_res['macro_mean_auroc']:.3f}** "
            f"± {auroc_res['macro_std_auroc']:.3f}  (n={auroc_res['n_genes_valid']} genes)",
            "",
            "| Sparsity bin | N genes | Mean AUROC | Std |",
            "| --- | ---: | ---: | ---: |",
        ]
        for b in auroc_res["sparsity_bins"]:
            lines.append(
                f"| {b['sparsity_bin']} | {b['n_genes']} | "
                f"{b['mean_auroc']:.3f} | {b['std_auroc']:.3f} |"
            )
        lines.append("")

    if retrieval is not None:
        agg = retrieval["aggregate"]
        lines += [
            "## Retrieval",
            f"- Query pool: {retrieval['n_query']:,} cells (stratified by slide), "
            f"averaged over {len(retrieval['per_seed'])} seeds",
            "",
            "| Direction | R@1 | R@5 | R@10 | R@50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for direction in ("image_to_gene", "gene_to_image"):
            row = [direction]
            for k in (1, 5, 10, 50):
                mean_key = f"{direction}_recall@{k}_mean"
                std_key = f"{direction}_recall@{k}_std"
                if mean_key in agg:
                    row.append(f"{agg[mean_key]:.3f} ± {agg[std_key]:.3f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "## Per-slide breakdown",
        "| Slide | Cells | Mean Pearson | Median | Group |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in per_slide["per_slide"]:
        short = row["slide"].split("__")[2] if len(row["slide"].split("__")) > 2 else row["slide"]
        lines.append(
            f"| {short} | {row['n_cells']:,} | "
            f"{row['mean_pearson']:.4f} | {row['median_pearson']:.4f} | "
            f"{row['ca_or_nl']} |"
        )

    grp = per_slide["group_summary"]
    lines += ["", "### Group summary"]
    for g in ("CA", "NL"):
        info = grp[g]
        if info["n_slides"]:
            lines.append(
                f"- {g}: {info['n_slides']} slides, "
                f"mean of per-slide means = {info['mean_of_means']:.4f} "
                f"± {info['std_of_means']:.4f}"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True,
                    help="Directory produced by eval_fixedsplit.py (contains per-split subdirs)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--retrieval_n", type=int, default=4000)
    ap.add_argument("--retrieval_seeds", type=int, nargs="+",
                    default=[0, 1, 2])
    args = ap.parse_args()

    for split in args.splits:
        split_dir = args.eval_dir / split
        if not split_dir.exists():
            print(f"[warn] missing split dir: {split_dir}")
            continue
        analyse_split(split_dir, args.retrieval_n, args.retrieval_seeds)


if __name__ == "__main__":
    main()
