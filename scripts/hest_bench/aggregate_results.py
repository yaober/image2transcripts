"""Aggregate HEST-Bench Pearson results across encoders + Xenium tasks.

Reads the per-task ``results_kfold.json`` files HEST writes under
``runs/hest_bench/results/<encoder>/<exp_dir>/<task>/custom_encoder/`` and
emits a single CSV at ``runs/hest_bench/summary.csv`` with one row per
(encoder, task) pair.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "runs/hest_bench/results"
TASKS = ["IDC", "PAAD", "SKCM", "COAD", "LUNG"]

NOTES = {
    "image2transcript": "ViT-B/16 i_raw (768-d) from runs/fixedsplit_v3/full_seed42",
    "ghist": "GHIST UNet3+ encoder bottleneck (1024-d, RANDOM init — no pretrained ckpt)",
}


def latest_exp_dir(encoder_dir: Path) -> Path | None:
    # benchmark() creates ``<encoder>::<timestamp>`` subdirs; pick the newest.
    candidates = sorted(
        [p for p in encoder_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main() -> None:
    rows = []
    for encoder_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        encoder = encoder_dir.name
        exp_dir = latest_exp_dir(encoder_dir)
        if exp_dir is None:
            print(f"[skip] no exp dir under {encoder_dir}")
            continue
        for task in TASKS:
            kfold_path = exp_dir / task / "custom_encoder" / "results_kfold.json"
            if not kfold_path.is_file():
                print(f"[skip] {kfold_path} missing")
                rows.append({
                    "encoder": encoder,
                    "task": task,
                    "pearson_mean": "NA",
                    "pearson_std": "NA",
                    "n_folds": "NA",
                    "notes": NOTES.get(encoder, ""),
                })
                continue
            with open(kfold_path) as f:
                blob = json.load(f)
            rows.append({
                "encoder": encoder,
                "task": task,
                "pearson_mean": blob["pearson_mean"],
                "pearson_std": blob["pearson_std"],
                "n_folds": len(blob.get("mean_per_split", [])),
                "notes": NOTES.get(encoder, ""),
            })

    out = REPO_ROOT / "runs/hest_bench/summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["encoder", "task", "pearson_mean", "pearson_std", "n_folds", "notes"],
        )
        w.writeheader()
        w.writerows(rows)

    # Wide table: encoder × task pearson_mean
    encoders = sorted({r["encoder"] for r in rows})
    wide_path = REPO_ROOT / "runs/hest_bench/summary_wide.csv"
    with open(wide_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder"] + TASKS + ["mean"])
        for enc in encoders:
            cells = []
            for t in TASKS:
                hit = next((r for r in rows if r["encoder"] == enc and r["task"] == t), None)
                cells.append(hit["pearson_mean"] if hit else "NA")
            try:
                avg = round(
                    sum(c for c in cells if isinstance(c, (int, float))) / len(cells), 4
                )
            except (TypeError, ZeroDivisionError):
                avg = "NA"
            w.writerow([enc, *cells, avg])

    print(f"wrote {out}")
    print(f"wrote {wide_path}")


if __name__ == "__main__":
    main()
