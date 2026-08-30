#!/usr/bin/env bash
# SciSt benchmark — LUNG only, GPU 3
# Downloads HEST LUNG data via sCellST first (skip_embed), then runs SciSt prep+train.
# Usage: nohup bash scripts/hest_bench/run_scist_LUNG.sh > scripts/hest_bench/run_scist_LUNG.log 2>&1 &
set -euo pipefail
cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

export CUDA_VISIBLE_DEVICES=3

# HEST data already downloaded to /work/OSPH/s439765/scellst/LUNG/hest_data/
# Step 1 (download) not needed. Run SciSt prep + train directly:
SCIST_DATASETS="LUNG" bash scripts/hest_bench/run_local_scist.sh
