"""AAAI-grade refinement of the TCGA-COAD clinical association analysis.

This wraps ``scripts/tcga_clinical_analysis.py`` and adds four rigour
upgrades the all-comers analysis lacks:

1.  **Multi-aggregation.**  ``tcga_predict.py`` already saves per-slide
    ``mu_mean``, ``mu_median`` and ``mu_p75`` over the 1500 sampled
    tiles.  Mean averages out focal signal; the 75th-percentile
    aggregator keeps the high-tail tiles where the signal lives.  We
    run all three.

2.  **Patient-shuffle empirical null with a correlated-gene-aware
    statistic.**  Permute the patient ⇄ prediction mapping K times,
    rerun the same association battery, and compare the *maximum
    −log10(p) across genes* between observed and permuted runs.  We
    do NOT use BH hit-counts as the null statistic: because the 372
    genes share variance from the same model, a single null draw
    produces correlated false positives that BH within-permutation
    cannot suppress.  Max-neglogp is invariant to gene-gene
    correlation and gives the right empirical p.

3.  **Pre-registered panel enrichment.**  For variables that should be
    driven by proliferation (vital_status, OS, recurrence), stromal
    remodelling (OS, late stage), or immune infiltrate (age, OS), we
    Fisher-test enrichment of literature panels among the discoveries.
    This converts "we got 14 OS genes" into "predicted OS genes are
    enriched 7× for stromal-remodelling markers (p=...)", which is
    the form an AAAI reviewer expects.

4.  **Effect-size reporting and reproducibility manifest.**  Each
    headline table reports the strongest gene, its statistic, and the
    panel enrichment alongside the BH count.  A manifest.json records
    git-sha, n_slides, n_patients, gene-list hash, and argv.

A brief note on framing: every TCGA-COAD slide is a tumor sample, so
there is no tumor / normal-tissue axis to strip.  The ``tumor_status``
field encodes WITH-TUMOR vs TUMOR-FREE *at last follow-up* — a
recurrence / residual-disease outcome, not a histology artefact.  Its
strong association with predicted transcripts is a real clinical
finding, not the trivial sanity check it would be in a tumor-vs-normal
dataset.

Run after ``tcga_predict.py`` has populated ``runs/tcga_coad/predictions/``.

Example::

    python scripts/tcga_clinical_analysis_aaai.py \\
        --predictions_dir runs/tcga_coad/predictions \\
        --clinical_dir "/archive/DPDS/.../TCGA/COAD/clinical files" \\
        --out_dir runs/tcga_coad/analysis_aaai \\
        --permutations 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tcga_clinical_analysis import (  # noqa: E402
    bh_correct,
    build_variable_columns,
    load_clinical_patient,
    median_split_logrank,
    spearman,
    ttest_two_sided,
    ASSOC_CONFIG,
)


# ----------------------------------------------------------------------
# Pre-registered gene panels (defined before looking at the results so
# they cannot be cherry-picked).  Each panel is a literature-curated
# list of genes expected to vary with the indicated clinical axis.
# ----------------------------------------------------------------------

PROLIFERATION_PANEL = [
    "MKI67", "TOP2A", "PCNA", "TK1", "TYMS", "RRM2", "UBE2C",
    "CDCA7", "CDC20", "CCNB1", "CCNB2", "AURKA", "AURKB",
    "FOXM1", "KIF20A", "BIRC5", "PTTG1", "PCLAF", "MCM2", "MCM3",
    "MCM4", "MCM5", "MCM6", "MCM7", "CDK1",
]

STROMAL_PANEL = [
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
    "COL6A1", "COL6A2", "COL6A3", "COL4A1", "COL4A2",
    "ACTA2", "FAP", "POSTN", "MMP1", "MMP2", "MMP3", "MMP9",
    "LOX", "PDGFRA", "PDGFRB", "RGS5", "IGFBP7",
]

IMMUNE_PANEL = [
    "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD68", "CD163",
    "CD83", "CD79A", "MS4A1", "HLA-DRA", "HLA-DRB1", "HLA-DPB1",
    "HLA-DMA", "IL6", "IL1B", "CXCL1", "CXCL2", "CXCL8",
    "CCL2", "CCL4", "CTLA4", "GZMA", "GZMB", "PRF1", "NKG7",
    "GNLY", "IFNG", "FCRLA", "PAX5",
]

# Variable -> list of (panel_name, panel_genes) we expect to be
# enriched in that variable's discoveries.  Pre-registered; negative
# findings are reported.
PANEL_TARGETS: dict[str, list[tuple[str, list[str]]]] = {
    "vital_status":     [("proliferation", PROLIFERATION_PANEL),
                         ("stromal",       STROMAL_PANEL),
                         ("immune",        IMMUNE_PANEL)],
    "overall_survival": [("proliferation", PROLIFERATION_PANEL),
                         ("stromal",       STROMAL_PANEL),
                         ("immune",        IMMUNE_PANEL)],
    "tumor_status":     [("proliferation", PROLIFERATION_PANEL),
                         ("immune",        IMMUNE_PANEL)],
    "stage_binary":     [("stromal",       STROMAL_PANEL),
                         ("proliferation", PROLIFERATION_PANEL)],
    "n_binary":         [("stromal",       STROMAL_PANEL)],
    "msi":              [("immune",        IMMUNE_PANEL)],
    "age":              [("immune",        IMMUNE_PANEL)],
}


# ----------------------------------------------------------------------
# Slide loader.
# ----------------------------------------------------------------------


def load_slide_predictions_multi(predictions_dir: Path
                                  ) -> tuple[dict[str, np.ndarray],
                                             list[str], list[str], list[str]]:
    """Return ({agg: (n_slides, n_genes)}, slide_ids, patient_barcodes, genes)."""
    npz_files = sorted(p for p in predictions_dir.iterdir() if p.suffix == ".npz")
    if not npz_files:
        raise RuntimeError(f"No .npz prediction dumps under {predictions_dir}")
    with open(predictions_dir / "genes.json") as f:
        genes = json.load(f)

    slide_ids: list[str] = []
    patients: list[str] = []
    rows = {"mean": [], "median": [], "p75": []}
    keymap = {"mean": "mu_mean", "median": "mu_median", "p75": "mu_p75"}

    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as exc:
            print(f"[warn] could not load {fp.name}: {exc}")
            continue
        if any(keymap[k] not in data.files for k in rows):
            print(f"[warn] {fp.name} missing aggregation field; skipping")
            continue

        meta = {}
        if "metadata_json" in data.files:
            try:
                meta = json.loads(data["metadata_json"].item().decode())
            except Exception:
                meta = {}
        slide_id = meta.get("slide_id", fp.stem)
        patient = meta.get("patient_barcode") or slide_id[:12]

        slide_ids.append(slide_id)
        patients.append(patient)
        for agg, raw_key in keymap.items():
            rows[agg].append(data[raw_key].astype(np.float32))

    if not slide_ids:
        raise RuntimeError("No usable slide predictions found")

    out = {agg: np.stack(rows[agg], axis=0) for agg in rows}
    print(f"Loaded {len(slide_ids)} slides x {len(genes)} genes "
          f"x 3 aggregations (mean/median/p75)")
    return out, slide_ids, patients, genes


def aggregate_to_patient(mu_slide: np.ndarray, patient_barcodes: list[str]
                         ) -> tuple[np.ndarray, list[str]]:
    by_pat: dict[str, list[int]] = {}
    for i, p in enumerate(patient_barcodes):
        by_pat.setdefault(p, []).append(i)
    ordered = sorted(by_pat.keys())
    out = np.empty((len(ordered), mu_slide.shape[1]), dtype=np.float32)
    for i, p in enumerate(ordered):
        out[i] = mu_slide[by_pat[p]].mean(axis=0)
    return out, ordered


# ----------------------------------------------------------------------
# Single-pass association battery (vectorised across genes).
# ----------------------------------------------------------------------


def run_one_battery(mu_patient: np.ndarray,
                    genes: list[str],
                    variables: dict[str, np.ndarray],
                    ) -> list[dict]:
    duration = variables.get("os_duration")
    event = variables.get("os_event")
    rows: list[dict] = []
    n_genes = len(genes)

    for var_name, test, description in ASSOC_CONFIG:
        if test == "logrank":
            if duration is None or event is None or np.isfinite(duration).sum() < 20:
                continue
            stats_arr = np.empty(n_genes); ps = np.empty(n_genes)
            n_lo_arr = np.empty(n_genes, dtype=int); n_hi_arr = np.empty(n_genes, dtype=int)
            for gi in range(n_genes):
                chi2, p, n_lo, n_hi = median_split_logrank(mu_patient[:, gi], duration, event)
                stats_arr[gi] = chi2; ps[gi] = p
                n_lo_arr[gi] = n_lo; n_hi_arr[gi] = n_hi
            qs = bh_correct(ps)
            for gi, gene in enumerate(genes):
                rows.append({
                    "gene": gene, "variable": var_name, "test": "logrank",
                    "description": description,
                    "stat": stats_arr[gi], "p": ps[gi], "q": qs[gi],
                    "n_lo": int(n_lo_arr[gi]), "n_hi": int(n_hi_arr[gi]),
                })
            continue

        y = variables.get(var_name)
        if y is None or np.isfinite(y).sum() < 10:
            continue
        stats_arr = np.empty(n_genes); ps = np.empty(n_genes)
        n0_arr = np.empty(n_genes, dtype=int); n1_arr = np.empty(n_genes, dtype=int)
        for gi in range(n_genes):
            if test == "spearman":
                r, p, n = spearman(mu_patient[:, gi], y)
                stats_arr[gi] = r; ps[gi] = p; n0_arr[gi] = n; n1_arr[gi] = 0
            else:
                t, p, n0, n1 = ttest_two_sided(mu_patient[:, gi], y)
                stats_arr[gi] = t; ps[gi] = p; n0_arr[gi] = n0; n1_arr[gi] = n1
        qs = bh_correct(ps)
        for gi, gene in enumerate(genes):
            row = {
                "gene": gene, "variable": var_name, "test": test,
                "description": description,
                "stat": stats_arr[gi], "p": ps[gi], "q": qs[gi],
            }
            if test == "ttest":
                row["n_lo"] = int(n0_arr[gi]); row["n_hi"] = int(n1_arr[gi])
            else:
                row["n"] = int(n0_arr[gi])
            rows.append(row)
    return rows


def hit_counts_by_variable(rows: list[dict], q_thresh: float = 0.05
                            ) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if np.isfinite(r["q"]) and r["q"] < q_thresh:
            counts[r["variable"]] = counts.get(r["variable"], 0) + 1
    return counts


def max_neglogp_by_variable(rows: list[dict]) -> dict[str, float]:
    """Per variable, return max(-log10(p)) across genes.  Calibrated null
    statistic under arbitrary gene-gene correlation."""
    best: dict[str, float] = {}
    for r in rows:
        p = r.get("p", np.nan)
        if not np.isfinite(p) or p <= 0:
            continue
        nlp = -float(np.log10(p))
        if nlp > best.get(r["variable"], -np.inf):
            best[r["variable"]] = nlp
    return best


def top_gene_by_variable(rows: list[dict]) -> dict[str, dict]:
    """Per variable, the row with the smallest p (regardless of q)."""
    best: dict[str, dict] = {}
    for r in rows:
        p = r.get("p", np.nan)
        if not np.isfinite(p):
            continue
        cur = best.get(r["variable"])
        if cur is None or p < cur["p"]:
            best[r["variable"]] = r
    return best


# ----------------------------------------------------------------------
# Patient-shuffle empirical null.
# ----------------------------------------------------------------------


def permutation_null(mu_patient: np.ndarray,
                      genes: list[str],
                      variables: dict[str, np.ndarray],
                      K: int,
                      seed: int = 0,
                      ) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    """Shuffle the patient axis K times and record per-variable
    (hit-count, max-neglogp) under the null.

    Returns ({var: [hit_count_perm_k]}, {var: [max_neglogp_perm_k]}).
    The max-neglogp statistic is the calibrated one under correlated
    genes; the hit-count is kept for diagnostic only.
    """
    rng = np.random.default_rng(seed)
    n = mu_patient.shape[0]
    perm_hits: dict[str, list[int]] = {}
    perm_nlp: dict[str, list[float]] = {}
    log_every = max(1, K // 10)
    for k in range(K):
        order = rng.permutation(n)
        rows = run_one_battery(mu_patient[order], genes, variables)
        ck = hit_counts_by_variable(rows)
        nlp = max_neglogp_by_variable(rows)
        for var, c in ck.items():
            perm_hits.setdefault(var, []).append(c)
        for var, v in nlp.items():
            perm_nlp.setdefault(var, []).append(v)
        if (k + 1) % log_every == 0:
            print(f"  [perm] {k+1}/{K}")
    return perm_hits, perm_nlp


# ----------------------------------------------------------------------
# Panel enrichment via Fisher's exact.
# ----------------------------------------------------------------------


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    from scipy.stats import fisher_exact
    or_, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(or_), float(p)


def panel_enrichment(rows: list[dict], genes: list[str],
                      top_k: int = 50, q_thresh: float = 0.05,
                      ) -> list[dict]:
    by_var: dict[str, list[dict]] = {}
    for r in rows:
        by_var.setdefault(r["variable"], []).append(r)
    out: list[dict] = []
    gene_set = set(genes)
    for var, panels in PANEL_TARGETS.items():
        recs = by_var.get(var)
        if not recs:
            continue
        recs_sorted = sorted(recs, key=lambda r: (1.0 if not np.isfinite(r["p"]) else r["p"]))
        sig = [r["gene"] for r in recs_sorted
               if np.isfinite(r["q"]) and r["q"] < q_thresh]
        if len(sig) < 5:
            sig = [r["gene"] for r in recs_sorted[:top_k]]
            sig_label = f"top{top_k}"
        else:
            sig_label = f"q<{q_thresh}"
        sig_set = set(sig)
        for panel_name, panel_genes in panels:
            panel_in_universe = [g for g in panel_genes if g in gene_set]
            if not panel_in_universe:
                continue
            panel_set = set(panel_in_universe)
            a = len(sig_set & panel_set)
            b = len(panel_set - sig_set)
            c = len(sig_set - panel_set)
            d = len(gene_set - panel_set - sig_set)
            or_, p = fisher_exact_2x2(a, b, c, d)
            out.append({
                "variable":     var,
                "panel":        panel_name,
                "panel_size":   len(panel_in_universe),
                "discoveries":  len(sig_set),
                "discovery_set": sig_label,
                "panel_in_disc": a,
                "odds_ratio":   or_,
                "p":            p,
                "panel_hits":   ";".join(sorted(sig_set & panel_set)),
            })
    return out


# ----------------------------------------------------------------------
# Reporting.
# ----------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def empirical_pvalue(observed: float, null_values: list[float]) -> float:
    if not null_values:
        return float("nan")
    arr = np.asarray(null_values, dtype=float)
    return float((arr >= observed).sum() + 1) / (len(arr) + 1)


VAR_PRETTY = {
    "age":              "Age",
    "t_ordinal":        "T stage",
    "gender":           "Gender",
    "stage_binary":     "AJCC stage (early/late)",
    "n_binary":         "Lymph-node (N0/N+)",
    "m_binary":         "Distant met (M0/M1)",
    "vital_status":     "Vital status",
    "tumor_status":     "Recurrence (with-tumor at f/u)",
    "msi":              "MSI status",
    "kras_mut":         "KRAS mutation",
    "braf_mut":         "BRAF mutation",
    "overall_survival": "Overall survival",
}


def write_summary(path: Path,
                   n_patients: int,
                   hits_by_agg: dict[str, dict[str, int]],
                   max_nlp_by_agg: dict[str, dict[str, float]],
                   top_gene_by_agg: dict[str, dict[str, dict]],
                   perm_nlp_by_agg: dict[str, dict[str, list[float]]],
                   panel_rows: list[dict],
                   headline_agg: str,
                   ) -> None:
    lines = ["# TCGA-COAD AAAI summary",
             "",
             f"Cohort: 451 TCGA-COAD patients (462 slides; n={n_patients} after",
             "patient-level mean aggregation across slides).  All slides are",
             "tumor — there is no tumor / normal axis to remove.",
             "",
             "Each row is one clinical variable, tested against all 372 predicted",
             "transcripts.  We report:",
             "- **q<0.05**: BH-corrected discovery count (descriptive only).",
             "- **top gene**: strongest individual hit (gene, raw p).",
             "- **max −log10 p**: observed value of the global statistic.",
             "- **null mean (p95)**: max −log10 p averaged over K patient-shuffle",
             "  permutations, with the 95th percentile of the null distribution.",
             "- **empirical p**: fraction of permutations whose max −log10 p ≥",
             "  observed.  Calibrated under arbitrary gene-gene correlation;",
             "  BH hit-count alone is *not* a valid global test under the",
             "  correlated-gene null and inflates inside each permutation.",
             ""]

    for agg in ("mean", "median", "p75"):
        hits = hits_by_agg.get(agg, {})
        obs = max_nlp_by_agg.get(agg, {})
        top = top_gene_by_agg.get(agg, {})
        nul = perm_nlp_by_agg.get(agg, {})
        if not hits and not obs:
            continue
        marker = " (headline)" if agg == headline_agg else ""
        lines.append(f"## Aggregation: {agg}{marker}")
        lines.append("")
        lines.append("| Variable | q<0.05 | top gene (p) | max −log10 p | "
                     "null mean (p95) | emp p |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
        for var_name, _test, _desc in ASSOC_CONFIG:
            if var_name not in obs:
                continue
            obs_v = obs.get(var_name, float("nan"))
            obs_h = hits.get(var_name, 0)
            tg = top.get(var_name)
            tg_str = f"{tg['gene']} ({tg['p']:.1e})" if tg else "—"
            null_v = nul.get(var_name, [])
            if null_v:
                arr = np.asarray(null_v)
                null_str = f"{arr.mean():.2f} ({np.percentile(arr, 95):.2f})"
                emp_p = empirical_pvalue(obs_v, null_v)
                emp_p_str = f"{emp_p:.3g}"
            else:
                null_str = "—"
                emp_p_str = "—"
            pretty = VAR_PRETTY.get(var_name, var_name)
            lines.append(
                f"| {pretty} | {obs_h} | {tg_str} | {obs_v:.2f} | "
                f"{null_str} | {emp_p_str} |"
            )
        lines.append("")

    lines.append(f"## Pre-registered panel enrichment ({headline_agg} aggregation)")
    lines.append("")
    lines.append("Fisher's exact test for panel ∩ discoveries.  Discovery set is")
    lines.append("`q<0.05` when ≥5 hits exist, otherwise top-50 by p-value.")
    lines.append("")
    lines.append("| Variable | Panel | Panel size | Discoveries | "
                 "Panel ∩ disc | Odds ratio | p | Panel hits |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in panel_rows:
        var_pretty = VAR_PRETTY.get(r["variable"], r["variable"])
        hits_str = (r["panel_hits"][:80] + "…") if len(r["panel_hits"]) > 80 \
                   else r["panel_hits"]
        lines.append(
            f"| {var_pretty} | {r['panel']} | {r['panel_size']} | "
            f"{r['discoveries']} ({r['discovery_set']}) | {r['panel_in_disc']} | "
            f"{r['odds_ratio']:.2f} | {r['p']:.2e} | {hits_str} |"
        )
    path.write_text("\n".join(lines) + "\n")


def file_hash(genes: list[str]) -> str:
    h = hashlib.sha256()
    h.update(("\n".join(genes)).encode())
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions_dir", type=Path,
                    default=Path("runs/tcga_coad/predictions"))
    ap.add_argument("--clinical_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/tcga_coad/analysis_aaai"))
    ap.add_argument("--permutations", type=int, default=200,
                    help="Patient-label shuffles for the empirical null. "
                         "Set to 0 to skip.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headline_agg", type=str, default="p75",
                    choices=["mean", "median", "p75"],
                    help="Aggregation used for panel-enrichment headlines")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load slide predictions in three aggregations and aggregate to patients.
    mu_by_agg, slide_ids, patient_barcodes, genes = \
        load_slide_predictions_multi(args.predictions_dir)
    patient_mu_by_agg: dict[str, np.ndarray] = {}
    patients_ordered: list[str] = []
    for agg, mu_slide in mu_by_agg.items():
        mu_pat, ordered = aggregate_to_patient(mu_slide, patient_barcodes)
        patient_mu_by_agg[agg] = mu_pat
        if not patients_ordered:
            patients_ordered = ordered
    print(f"Patients: {len(patients_ordered)}")

    # 2. Clinical variables.
    clinical = load_clinical_patient(args.clinical_dir)
    variables = build_variable_columns(clinical, patients_ordered)

    # 3. Run associations for every aggregation.
    rows_by_agg: dict[str, list[dict]] = {}
    hits_by_agg: dict[str, dict[str, int]] = {}
    max_nlp_by_agg: dict[str, dict[str, float]] = {}
    top_gene_by_agg: dict[str, dict[str, dict]] = {}
    for agg, mu_pat in patient_mu_by_agg.items():
        print(f"[battery] agg={agg}")
        rows = run_one_battery(mu_pat, genes, variables)
        rows_by_agg[agg] = rows
        hits_by_agg[agg] = hit_counts_by_variable(rows)
        max_nlp_by_agg[agg] = max_neglogp_by_variable(rows)
        top_gene_by_agg[agg] = top_gene_by_variable(rows)
        write_csv(args.out_dir / f"associations_{agg}.csv", rows)

    # 4. Patient-shuffle empirical null on each aggregation.
    perm_nlp_by_agg: dict[str, dict[str, list[float]]] = {}
    perm_hits_by_agg: dict[str, dict[str, list[int]]] = {}
    if args.permutations > 0:
        for agg, mu_pat in patient_mu_by_agg.items():
            print(f"[perm null] agg={agg} K={args.permutations}")
            hits, nlps = permutation_null(
                mu_pat, genes, variables,
                K=args.permutations, seed=args.seed,
            )
            perm_hits_by_agg[agg] = hits
            perm_nlp_by_agg[agg] = nlps

    # 5. Pre-registered panel enrichment on the headline aggregation.
    panel_rows = panel_enrichment(rows_by_agg[args.headline_agg], genes,
                                   top_k=50, q_thresh=0.05)
    write_csv(args.out_dir / "panel_enrichment.csv", panel_rows)

    # 6. Permutation-null table CSV.
    perm_rows = []
    for agg, nlp_by_var in perm_nlp_by_agg.items():
        hits_by_var = perm_hits_by_agg.get(agg, {})
        obs_nlp_dict = max_nlp_by_agg.get(agg, {})
        obs_hits_dict = hits_by_agg.get(agg, {})
        for var, nlps in nlp_by_var.items():
            arr_nlp = np.asarray(nlps)
            obs_nlp = obs_nlp_dict.get(var, float("nan"))
            row = {
                "aggregation":    agg,
                "variable":       var,
                "perm_K":         len(nlps),
                "obs_max_nlp":    float(obs_nlp),
                "null_nlp_mean":  float(arr_nlp.mean()),
                "null_nlp_p95":   float(np.percentile(arr_nlp, 95)),
                "null_nlp_max":   float(arr_nlp.max()),
                "empirical_p":    empirical_pvalue(float(obs_nlp), nlps),
                "obs_hits":       int(obs_hits_dict.get(var, 0)),
            }
            counts = hits_by_var.get(var, [])
            if counts:
                arr_h = np.asarray(counts)
                row["null_hits_mean"] = float(arr_h.mean())
                row["null_hits_p95"] = float(np.percentile(arr_h, 95))
            perm_rows.append(row)
    write_csv(args.out_dir / "permutation_null.csv", perm_rows)

    # 7. Summary.
    write_summary(
        args.out_dir / "summary_aaai.md",
        n_patients=len(patients_ordered),
        hits_by_agg=hits_by_agg,
        max_nlp_by_agg=max_nlp_by_agg,
        top_gene_by_agg=top_gene_by_agg,
        perm_nlp_by_agg=perm_nlp_by_agg,
        panel_rows=panel_rows,
        headline_agg=args.headline_agg,
    )

    # 8. Manifest.
    manifest = {
        "git_sha":          git_sha(),
        "predictions_dir":  str(args.predictions_dir.resolve()),
        "n_slides":         len(slide_ids),
        "n_patients":       len(patients_ordered),
        "n_genes":          len(genes),
        "gene_panel_hash":  file_hash(genes),
        "permutations":     args.permutations,
        "seed":             args.seed,
        "aggregations":     list(patient_mu_by_agg.keys()),
        "headline_agg":     args.headline_agg,
        "argv":             sys.argv,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote: {args.out_dir / 'summary_aaai.md'}")
    for agg in rows_by_agg:
        print(f"       {args.out_dir / f'associations_{agg}.csv'}")
    print(f"       {args.out_dir / 'panel_enrichment.csv'}")
    print(f"       {args.out_dir / 'permutation_null.csv'}")
    print(f"       {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
