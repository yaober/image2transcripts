"""Run HEST-Bench on the 5 Xenium tasks for one custom encoder.

Usage (from the repo root, inside the ``hest_bench`` conda env):

    python scripts/hest_bench/run_bench.py --encoder image2transcript
    python scripts/hest_bench/run_bench.py --encoder ghist
    # (sCellST is run via its native pipeline, not through this driver.)

Outputs land under ``runs/hest_bench/<encoder>/<dataset>/...`` per HEST's
``benchmark`` convention; the per-task aggregate is in
``runs/hest_bench/<encoder>/dataset_results.{csv,json}``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# HF token must be set before any HEST/Trident import that might fetch the
# bench dataset metadata.
# Replace YOUR_HF_TOKEN_HERE with your own HuggingFace token, or export
# HF_TOKEN before running. The gated pathology foundation models
# (Phikon2, Prov-GigaPath, UNI2-h) cannot be downloaded without one.
os.environ.setdefault("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

from hest.bench import benchmark  # noqa: E402

from hest_bench.encoders import GHISTEncoder, Image2TranscriptEncoder  # noqa: E402

XENIUM_TASKS = ["IDC", "PAAD", "SKCM", "COAD", "LUNG"]


def build_encoder(name: str, ckpt_path: str | None) -> torch.nn.Module:
    if name == "image2transcript":
        if ckpt_path is None:
            ckpt_path = str(
                REPO_ROOT / "runs/fixedsplit_v3/full_seed42/best_model.pt"
            )
        return Image2TranscriptEncoder(ckpt_path=ckpt_path)
    if name == "ghist":
        return GHISTEncoder(ckpt_path=ckpt_path)
    raise ValueError(f"unknown encoder {name!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--encoder",
        required=True,
        choices=["image2transcript", "ghist"],
        help="which custom encoder to benchmark",
    )
    p.add_argument(
        "--ckpt_path",
        default=None,
        help="optional checkpoint to load into the encoder",
    )
    p.add_argument(
        "--bench_data_root",
        default=str(REPO_ROOT / "runs/hest_bench/eval/bench_data"),
    )
    p.add_argument(
        "--results_dir",
        default=None,
        help="defaults to runs/hest_bench/results/<encoder>",
    )
    p.add_argument(
        "--embed_dataroot",
        default=None,
        help="defaults to runs/hest_bench/embeddings/<encoder>",
    )
    p.add_argument("--datasets", nargs="+", default=XENIUM_TASKS)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="embedding device; auto avoids CUDA when PyTorch lacks kernels for the visible GPU",
    )
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    encoder = build_encoder(args.encoder, args.ckpt_path)

    results_dir = (
        args.results_dir
        or str(REPO_ROOT / f"runs/hest_bench/results/{args.encoder}")
    )
    embed_dataroot = (
        args.embed_dataroot
        or str(REPO_ROOT / f"runs/hest_bench/embeddings/{args.encoder}")
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(embed_dataroot, exist_ok=True)

    # The HEST benchmark always runs ``snapshot_download`` to ensure the bench
    # data exists; we already pre-downloaded the Xenium tasks, so the call is
    # a no-op for those folders.
    dataset_perfs, perf_per_enc = benchmark(
        encoder,
        None,
        None,
        bench_data_root=args.bench_data_root,
        results_dir=results_dir,
        embed_dataroot=embed_dataroot,
        datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        encoders=[],  # disable any default TRIDENT encoders
        exp_code=args.encoder,
    )
    print("dataset_perfs:", dataset_perfs)
    print("perf_per_enc:", perf_per_enc)


if __name__ == "__main__":
    main()
