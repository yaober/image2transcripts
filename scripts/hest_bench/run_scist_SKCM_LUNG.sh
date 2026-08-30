#!/usr/bin/env bash
# SciSt benchmark — SKCM only, GPU 3 (LUNG skipped: data not yet downloaded)
# Usage: nohup bash scripts/hest_bench/run_scist_SKCM_LUNG.sh > scripts/hest_bench/run_scist_SKCM_LUNG.log 2>&1 &
set -euo pipefail
cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

export CUDA_VISIBLE_DEVICES=3
SCIST_DATASETS="SKCM" bash scripts/hest_bench/run_local_scist.sh
