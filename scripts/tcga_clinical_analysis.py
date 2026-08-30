"""Correlate Image2Transcript TCGA COAD predictions with clinical variables.

Loads the slide-level prediction NPZs produced by ``scripts/tcga_predict.py``
plus the TCGA-COAD clinical files under ``clinical files/``, then:

1. Aggregates multiple slides per patient into a single per-patient row
   (mean of slide ``mu_mean`` across that patient's slides).
2. Parses the clinical variables we care about from
   ``nationwidechildrens.org_clinical_patient_coad.txt`` and augments them
   with overall-survival duration/event from the same file.
3. For each of the 372 genes, runs a battery of association tests against
   the clinical variables.  The test chosen per variable type:

       age, T-stage (1-4), N+ count   ->  Spearman's rho
       gender, KRAS/BRAF mutation,    ->  independent-samples t-test
       MSI (H vs MSS)
       AJCC stage (I/II vs III/IV)    ->  t-test on the collapsed binary
       vital status, tumor status     ->  t-test
       overall survival               ->  median-split log-rank (chi-square)

4. Applies Benjamini–Hochberg FDR correction separately per clinical
   variable, since we test 372 genes against each one independently.
5. Writes four outputs under ``--out_dir`` (default
   ``runs/tcga_coad/analysis/``):

       patient_gene_predictions.csv    (n_patients, 372)
       clinical_patient_parsed.csv     parsed clinical table
       associations.csv                one row per (gene, variable)
       top_hits.md                     top-20 genes per variable, by q

The script is intentionally dependency-light: scipy is required;
statsmodels / lifelines are optional — when absent we fall back to our own
BH and a hand-rolled median-split log-rank.

Example::

    python scripts/tcga_clinical_analysis.py \
        --predictions_dir runs/tcga_coad/predictions \
        --clinical_dir "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data/TCGA/COAD/clinical files" \
        --out_dir runs/tcga_coad/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test as lifelines_logrank
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------


def load_slide_predictions(predictions_dir: Path
                            ) -> tuple[np.ndarray, list[str], list[str], list[str], dict]:
    """Return (mu_mean_slide, slide_ids, patient_barcodes, gene_list, meta_by_slide)."""
    npz_files = sorted(p for p in predictions_dir.iterdir() if p.suffix == ".npz")
    if not npz_files:
        raise RuntimeError(f"No .npz prediction dumps found under {predictions_dir}")

    # Gene list lives next to the NPZs as a single JSON file.
    with open(predictions_dir / "genes.json") as f:
        genes = json.load(f)

    slide_ids: list[str] = []
    patient_barcodes: list[str] = []
    meta_by_slide: dict[str, dict] = {}
    mu_rows = []

    for fp in npz_files:
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as exc:
            print(f"[warn] could not load {fp.name}: {exc}")
            continue
        if "mu_mean" not in data.files:
            print(f"[warn] {fp.name} missing mu_mean; skipping")
            continue
        mu_mean = data["mu_mean"].astype(np.float32)
        if mu_mean.shape[0] != len(genes):
            print(f"[warn] {fp.name} gene count {mu_mean.shape[0]} != {len(genes)}; skipping")
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
        patient_barcodes.append(patient)
        meta_by_slide[slide_id] = meta
        mu_rows.append(mu_mean)

    if not mu_rows:
        raise RuntimeError("No usable slide predictions found")

    mu_arr = np.stack(mu_rows, axis=0)
    print(f"Loaded slide predictions: {mu_arr.shape[0]} slides, {mu_arr.shape[1]} genes")
    return mu_arr, slide_ids, patient_barcodes, genes, meta_by_slide


def aggregate_patient_level(mu_slide: np.ndarray,
                             slide_ids: list[str],
                             patient_barcodes: list[str]
                             ) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Mean of slide-level ``mu_mean`` across slides belonging to each patient."""
    patients: dict[str, list[int]] = {}
    for idx, p in enumerate(patient_barcodes):
        patients.setdefault(p, []).append(idx)

    ordered_patients = sorted(patients.keys())
    n, g = len(ordered_patients), mu_slide.shape[1]
    out = np.empty((n, g), dtype=np.float32)
    slides_per_patient: dict[str, list[str]] = {}
    for i, pat in enumerate(ordered_patients):
        idxs = patients[pat]
        out[i] = mu_slide[idxs].mean(axis=0)
        slides_per_patient[pat] = [slide_ids[j] for j in idxs]
    print(f"Aggregated to {n} patients (mean of slides per patient)")
    return out, ordered_patients, slides_per_patient


# ----------------------------------------------------------------------
# Clinical parsing
# ----------------------------------------------------------------------


DEFAULT_CLINICAL_COLS = [
    "bcr_patient_barcode",
    "gender",
    "age_at_initial_pathologic_diagnosis",
    "ajcc_pathologic_tumor_stage",
    "ajcc_tumor_pathologic_pt",
    "ajcc_nodes_pathologic_pn",
    "ajcc_metastasis_pathologic_pm",
    "vital_status",
    "last_contact_days_to",
    "death_days_to",
    "tumor_status",
    "microsatellite_instability",
    "kras_mutation_found",
    "braf_gene_analysis_result",
    "histologic_diagnosis",
    "residual_tumor",
    "lymphovascular_invasion_indicator",
    "perineural_invasion",
    "history_colon_polyps",
]


def load_clinical_patient(clinical_dir: Path) -> dict[str, dict]:
    fp = clinical_dir / "nationwidechildrens.org_clinical_patient_coad.txt"
    with open(fp, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    # First two rows under the header are metadata lines (CDE IDs, field
    # descriptions).  Drop rows whose patient barcode does not look like a
    # TCGA barcode; this handles those cleanly.
    parsed: dict[str, dict] = {}
    for row in rows:
        barcode = row.get("bcr_patient_barcode", "")
        if not barcode.startswith("TCGA-"):
            continue
        parsed[barcode] = {k: row.get(k, "") for k in DEFAULT_CLINICAL_COLS}
    print(f"Parsed clinical records for {len(parsed)} patients")
    return parsed


def _to_float(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "na", "n/a", "[not available]",
                               "[not applicable]", "[not evaluated]",
                               "[unknown]", "[discrepancy]"}:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _norm_stage(s: str | None) -> str | None:
    """Collapse free-form AJCC stage strings to I / II / III / IV."""
    if not s:
        return None
    up = s.upper().strip()
    if "STAGE IV" in up:
        return "IV"
    if "STAGE III" in up:
        return "III"
    if "STAGE II" in up:
        return "II"
    if "STAGE I" in up:
        return "I"
    if "STAGE 0" in up:
        return "0"
    return None


def _binary_from_stage(stage: str | None) -> int | None:
    if stage in {"I", "II", "0"}:
        return 0
    if stage in {"III", "IV"}:
        return 1
    return None


def _t_order(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up.startswith(("T1", "PT1")):
        return 1
    if up.startswith(("T2", "PT2")):
        return 2
    if up.startswith(("T3", "PT3")):
        return 3
    if up.startswith(("T4", "PT4")):
        return 4
    return None


def _n_binary(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up.startswith(("N0", "PN0")):
        return 0
    if up.startswith(("N", "PN")):  # N1, N2, ...
        return 1
    return None


def _m_binary(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up.startswith(("M0", "PM0")):
        return 0
    if up.startswith(("M1", "PM1")):
        return 1
    return None


def _binary_from_yes_no(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up in {"YES", "POSITIVE", "MUTATION FOUND", "ABNORMAL"}:
        return 1
    if up in {"NO", "NEGATIVE", "WILD TYPE", "WT", "WILDTYPE",
              "NO MUTATION FOUND", "NORMAL"}:
        return 0
    return None


def _msi_binary(s: str | None) -> int | None:
    """TCGA-COAD stores an indicator: YES = MSI-H, NO = MSS."""
    if not s:
        return None
    up = s.upper().strip()
    if "MSI-H" in up or up in {"MSI", "YES"}:
        return 1
    if "MSS" in up or "MSI-L" in up or "STABLE" in up or up == "NO":
        return 0
    return None


def _gender_binary(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up.startswith("MALE"):
        return 0
    if up.startswith("FEMALE"):
        return 1
    return None


def _vital_binary(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up in {"DEAD", "DECEASED"}:
        return 1
    if up in {"ALIVE", "LIVING"}:
        return 0
    return None


def _tumor_status_binary(s: str | None) -> int | None:
    if not s:
        return None
    up = s.upper().strip()
    if up.startswith("WITH TUMOR"):
        return 1
    if up.startswith("TUMOR FREE"):
        return 0
    return None


def build_variable_columns(clinical_by_patient: dict[str, dict],
                            patients: list[str]) -> dict[str, np.ndarray]:
    """Return a dict mapping variable-name -> (n_patients,) float array.

    Float arrays use ``np.nan`` for missing values.  Variables encode:

        age                         numeric
        gender                      0 male / 1 female
        stage_binary                0 early (I/II) / 1 late (III/IV)
        t_ordinal                   1..4
        n_binary                    0 N0 / 1 N+
        m_binary                    0 M0 / 1 M1
        vital_status                0 alive / 1 dead
        tumor_status                0 tumor-free / 1 with-tumor
        msi                         0 MSS / 1 MSI-H
        kras_mut                    0 wt / 1 mut
        braf_mut                    0 wt / 1 mut
        os_duration                 float, days
        os_event                    0 censored / 1 death
    """
    n = len(patients)
    def fill(fn, col):
        arr = np.full(n, np.nan, dtype=np.float32)
        for i, pat in enumerate(patients):
            rec = clinical_by_patient.get(pat)
            if rec is None:
                continue
            v = fn(rec.get(col))
            if v is not None:
                arr[i] = float(v)
        return arr

    out = {
        "age": fill(_to_float, "age_at_initial_pathologic_diagnosis"),
        "gender": fill(_gender_binary, "gender"),
        "stage_binary": np.array(
            [_binary_from_stage(_norm_stage(clinical_by_patient.get(pat, {}).get(
                "ajcc_pathologic_tumor_stage"))) for pat in patients],
            dtype=float,
        ),
        "t_ordinal": np.array(
            [_t_order(clinical_by_patient.get(pat, {}).get(
                "ajcc_tumor_pathologic_pt")) for pat in patients],
            dtype=float,
        ),
        "n_binary": np.array(
            [_n_binary(clinical_by_patient.get(pat, {}).get(
                "ajcc_nodes_pathologic_pn")) for pat in patients],
            dtype=float,
        ),
        "m_binary": np.array(
            [_m_binary(clinical_by_patient.get(pat, {}).get(
                "ajcc_metastasis_pathologic_pm")) for pat in patients],
            dtype=float,
        ),
        "vital_status": fill(_vital_binary, "vital_status"),
        "tumor_status": fill(_tumor_status_binary, "tumor_status"),
        "msi": fill(_msi_binary, "microsatellite_instability"),
        "kras_mut": fill(_binary_from_yes_no, "kras_mutation_found"),
        "braf_mut": fill(_binary_from_yes_no, "braf_gene_analysis_result"),
    }

    # Overall survival: duration = max(last_contact, death); event = vital=dead.
    last_contact = fill(_to_float, "last_contact_days_to")
    death_days = fill(_to_float, "death_days_to")
    vital = out["vital_status"]
    duration = np.full(n, np.nan, dtype=np.float32)
    event = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if vital[i] == 1 and not np.isnan(death_days[i]):
            duration[i] = death_days[i]
            event[i] = 1
        elif vital[i] == 0 and not np.isnan(last_contact[i]):
            duration[i] = last_contact[i]
            event[i] = 0
        elif not np.isnan(death_days[i]):
            duration[i] = death_days[i]
            event[i] = 1 if vital[i] == 1 else 0
        elif not np.isnan(last_contact[i]):
            duration[i] = last_contact[i]
            event[i] = 0
    out["os_duration"] = duration
    out["os_event"] = event

    return out


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def bh_correct(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values.  Falls back to a hand-rolled impl."""
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not mask.any():
        return q
    if HAS_STATSMODELS:
        _, q_valid, _, _ = multipletests(p[mask], method="fdr_bh")
        q[mask] = q_valid
        return q
    # Hand-rolled BH: sort p, compute q, un-sort.
    p_valid = p[mask]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_monotone = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_sorted = np.minimum(q_monotone, 1.0)
    q_valid = np.empty_like(p_valid)
    q_valid[order] = q_sorted
    q[mask] = q_valid
    return q


def spearman(gene: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(gene) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.spearmanr(gene[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def ttest_two_sided(gene: np.ndarray, binary: np.ndarray) -> tuple[float, float, int, int]:
    mask = np.isfinite(gene) & np.isfinite(binary)
    if mask.sum() < 10:
        return np.nan, np.nan, 0, 0
    g0 = gene[mask & (binary == 0)]
    g1 = gene[mask & (binary == 1)]
    if len(g0) < 3 or len(g1) < 3:
        return np.nan, np.nan, len(g0), len(g1)
    stat, p = stats.ttest_ind(g0, g1, equal_var=False)
    return float(stat), float(p), int(len(g0)), int(len(g1))


def median_split_logrank(gene: np.ndarray, duration: np.ndarray,
                          event: np.ndarray) -> tuple[float, float, int, int]:
    """Median-split log-rank comparing high- vs low-expression patients."""
    mask = np.isfinite(gene) & np.isfinite(duration) & np.isfinite(event)
    if mask.sum() < 20:
        return np.nan, np.nan, 0, 0
    g = gene[mask]
    d = duration[mask]
    e = event[mask].astype(bool)

    median = np.median(g)
    group = (g >= median).astype(int)
    n_hi = int(group.sum())
    n_lo = int(len(group) - n_hi)
    if n_hi < 5 or n_lo < 5:
        return np.nan, np.nan, n_lo, n_hi

    if HAS_LIFELINES:
        res = lifelines_logrank(
            durations_A=d[group == 1], durations_B=d[group == 0],
            event_observed_A=e[group == 1], event_observed_B=e[group == 0],
        )
        return float(res.test_statistic), float(res.p_value), n_lo, n_hi

    # Hand-rolled log-rank using the per-event-time expected counts.
    event_times = np.sort(np.unique(d[e]))
    chi_num = 0.0
    var = 0.0
    for t in event_times:
        at_risk_lo = int(((d >= t) & (group == 0)).sum())
        at_risk_hi = int(((d >= t) & (group == 1)).sum())
        at_risk = at_risk_lo + at_risk_hi
        d_total = int(((d == t) & e).sum())
        d_hi = int(((d == t) & e & (group == 1)).sum())
        if at_risk <= 1 or d_total == 0:
            continue
        exp_hi = d_total * at_risk_hi / at_risk
        chi_num += d_hi - exp_hi
        var += (d_total * at_risk_lo * at_risk_hi * (at_risk - d_total)
                / (at_risk * at_risk * max(1, at_risk - 1)))
    if var <= 0:
        return np.nan, np.nan, n_lo, n_hi
    chi2 = (chi_num ** 2) / var
    p = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return float(chi2), p, n_lo, n_hi


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


ASSOC_CONFIG = [
    # (variable_name, test, description)
    ("age", "spearman", "Age at diagnosis"),
    ("t_ordinal", "spearman", "T stage (1-4)"),
    ("gender", "ttest", "Male (0) vs Female (1)"),
    ("stage_binary", "ttest", "Early (0) vs Late (1) AJCC stage"),
    ("n_binary", "ttest", "N0 (0) vs N+ (1)"),
    ("m_binary", "ttest", "M0 (0) vs M1 (1)"),
    ("vital_status", "ttest", "Alive (0) vs Dead (1)"),
    ("tumor_status", "ttest", "Tumor-free (0) vs With-tumor (1)"),
    ("msi", "ttest", "MSS (0) vs MSI-H (1)"),
    ("kras_mut", "ttest", "KRAS WT (0) vs mutant (1)"),
    ("braf_mut", "ttest", "BRAF WT (0) vs mutant (1)"),
    ("overall_survival", "logrank", "OS: median-split log-rank"),
]


def run_associations(mu_patient: np.ndarray,
                      genes: list[str],
                      variables: dict[str, np.ndarray]) -> list[dict]:
    n_genes = len(genes)
    rows: list[dict] = []

    duration = variables.get("os_duration")
    event = variables.get("os_event")

    for var_name, test, description in ASSOC_CONFIG:
        if test == "logrank":
            if duration is None or event is None:
                continue
            if np.isfinite(duration).sum() < 20:
                continue
            print(f"[logrank] {var_name}: {int(np.isfinite(duration).sum())} patients with OS")
            stats_per_gene = np.empty(n_genes, dtype=float)
            ps_per_gene = np.empty(n_genes, dtype=float)
            n_lo_arr = np.empty(n_genes, dtype=int)
            n_hi_arr = np.empty(n_genes, dtype=int)
            for gi in range(n_genes):
                chi2, p, n_lo, n_hi = median_split_logrank(
                    mu_patient[:, gi], duration, event)
                stats_per_gene[gi] = chi2
                ps_per_gene[gi] = p
                n_lo_arr[gi] = n_lo
                n_hi_arr[gi] = n_hi
            qs = bh_correct(ps_per_gene)
            for gi, gene in enumerate(genes):
                rows.append({
                    "gene": gene, "variable": var_name, "test": "logrank",
                    "description": description,
                    "stat": stats_per_gene[gi], "p": ps_per_gene[gi],
                    "q": qs[gi], "n": int(n_lo_arr[gi] + n_hi_arr[gi]),
                    "n_lo": int(n_lo_arr[gi]), "n_hi": int(n_hi_arr[gi]),
                })
            continue

        y = variables.get(var_name)
        if y is None or np.isfinite(y).sum() < 10:
            continue
        print(f"[{test}] {var_name}: {int(np.isfinite(y).sum())} patients")

        stats_per_gene = np.empty(n_genes, dtype=float)
        ps_per_gene = np.empty(n_genes, dtype=float)
        n_arr = np.empty(n_genes, dtype=int)
        n_lo_arr = np.empty(n_genes, dtype=int) if test == "ttest" else None
        n_hi_arr = np.empty(n_genes, dtype=int) if test == "ttest" else None

        for gi in range(n_genes):
            if test == "spearman":
                r, p, n = spearman(mu_patient[:, gi], y)
                stats_per_gene[gi] = r
                ps_per_gene[gi] = p
                n_arr[gi] = n
            elif test == "ttest":
                t, p, n0, n1 = ttest_two_sided(mu_patient[:, gi], y)
                stats_per_gene[gi] = t
                ps_per_gene[gi] = p
                n_arr[gi] = n0 + n1
                n_lo_arr[gi] = n0
                n_hi_arr[gi] = n1
            else:
                raise ValueError(f"Unknown test: {test}")

        qs = bh_correct(ps_per_gene)
        for gi, gene in enumerate(genes):
            row = {
                "gene": gene, "variable": var_name, "test": test,
                "description": description,
                "stat": stats_per_gene[gi], "p": ps_per_gene[gi],
                "q": qs[gi], "n": int(n_arr[gi]),
            }
            if test == "ttest":
                row["n_lo"] = int(n_lo_arr[gi])
                row["n_hi"] = int(n_hi_arr[gi])
            rows.append(row)

    return rows


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_top_hits_md(path: Path, rows: list[dict], top_k: int = 20) -> None:
    by_var: dict[str, list[dict]] = {}
    for r in rows:
        by_var.setdefault(r["variable"], []).append(r)

    lines = ["# TCGA-COAD association top hits",
             "",
             f"Top {top_k} genes per clinical variable, ranked by BH-adjusted q-value.",
             "Significant q < 0.05 hits are bolded.",
             ""]
    for var_name, group in by_var.items():
        finite = [r for r in group if np.isfinite(r.get("p", np.nan))]
        if not finite:
            continue
        finite.sort(key=lambda r: (r["q"] if np.isfinite(r["q"]) else 1.0,
                                    r["p"] if np.isfinite(r["p"]) else 1.0))
        head = finite[:top_k]
        desc = head[0]["description"]
        test = head[0]["test"]
        stat_name = {"spearman": "rho", "ttest": "t", "logrank": "chi2"}.get(test, "stat")
        lines.append(f"## {var_name} — {desc}  (test: {test})")
        lines.append("")
        lines.append(f"| # | gene | {stat_name} | p | q | n |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for i, r in enumerate(head, start=1):
            signif = np.isfinite(r["q"]) and r["q"] < 0.05
            gene = f"**{r['gene']}**" if signif else r["gene"]
            n_label = r.get("n", "")
            if "n_lo" in r and "n_hi" in r:
                n_label = f"{r['n_lo']}/{r['n_hi']}"
            lines.append(
                f"| {i} | {gene} | {r['stat']:.3f} | {r['p']:.2e} | "
                f"{r['q']:.2e} | {n_label} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions_dir", type=Path,
                    default=Path("runs/tcga_coad/predictions"))
    ap.add_argument("--clinical_dir", type=Path, required=True,
                    help="Directory with nationwidechildrens.org_clinical_*_coad.txt files")
    ap.add_argument("--out_dir", type=Path, default=Path("runs/tcga_coad/analysis"))
    ap.add_argument("--save_patient_gene_table", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mu_slide, slide_ids, patient_barcodes, genes, meta_by_slide = \
        load_slide_predictions(args.predictions_dir)
    mu_patient, patients, slides_per_patient = aggregate_patient_level(
        mu_slide, slide_ids, patient_barcodes)

    # Persist the patient x gene table for external reuse.
    if args.save_patient_gene_table:
        out_fp = args.out_dir / "patient_gene_predictions.csv"
        with open(out_fp, "w") as f:
            f.write("patient," + ",".join(genes) + "\n")
            for i, pat in enumerate(patients):
                f.write(pat + "," +
                        ",".join(f"{v:.4f}" for v in mu_patient[i].astype(np.float32))
                        + "\n")
        print(f"Wrote patient x gene table: {out_fp}")

    clinical = load_clinical_patient(args.clinical_dir)

    # Persist a flat parsed clinical table restricted to the patients we have.
    parsed_fp = args.out_dir / "clinical_patient_parsed.csv"
    with open(parsed_fp, "w") as f:
        f.write("patient,n_slides," +
                ",".join(DEFAULT_CLINICAL_COLS[1:]) + "\n")
        for pat in patients:
            rec = clinical.get(pat, {})
            f.write(pat + "," + str(len(slides_per_patient.get(pat, []))))
            for col in DEFAULT_CLINICAL_COLS[1:]:
                v = rec.get(col, "").replace(",", ";")
                f.write("," + v)
            f.write("\n")
    print(f"Wrote parsed clinical table: {parsed_fp}")

    variables = build_variable_columns(clinical, patients)

    print("Variable coverage:")
    for name, arr in variables.items():
        print(f"  {name:20s}  n={int(np.isfinite(arr).sum())}/{len(arr)}")

    rows = run_associations(mu_patient, genes, variables)

    assoc_fp = args.out_dir / "associations.csv"
    write_csv(assoc_fp, rows)
    print(f"Wrote associations: {assoc_fp}")

    top_fp = args.out_dir / "top_hits.md"
    write_top_hits_md(top_fp, rows, top_k=20)
    print(f"Wrote top hits: {top_fp}")

    # A compact overview: how many genes passed q < 0.05 per variable.
    hit_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for r in rows:
        total_counts[r["variable"]] = total_counts.get(r["variable"], 0) + 1
        if np.isfinite(r["q"]) and r["q"] < 0.05:
            hit_counts[r["variable"]] = hit_counts.get(r["variable"], 0) + 1

    summary = ["# TCGA-COAD downstream summary", ""]
    summary.append(f"- Slides analysed: {len(slide_ids)}")
    summary.append(f"- Patients after aggregation: {len(patients)}")
    summary.append(f"- Genes: {len(genes)}")
    summary.append("")
    summary.append("## Significant associations (BH q < 0.05)")
    summary.append("")
    summary.append("| Variable | Genes tested | Genes with q < 0.05 |")
    summary.append("| --- | ---: | ---: |")
    for var in sorted(total_counts.keys()):
        summary.append(
            f"| {var} | {total_counts[var]} | {hit_counts.get(var, 0)} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"Wrote summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
