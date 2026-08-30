#!/usr/bin/env bash
# SciSt benchmark — PAAD only, GPU 1
# Usage: nohup bash scripts/hest_bench/run_scist_PAAD.sh > scripts/hest_bench/run_scist_PAAD.log 2>&1 &
set -euo pipefail
cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

export CUDA_VISIBLE_DEVICES=1
SCIST_DATASETS="PAAD" bash scripts/hest_bench/run_local_scist.sh
