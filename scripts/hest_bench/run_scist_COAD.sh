#!/usr/bin/env bash
# SciSt benchmark — COAD only, GPU 2
# Usage: nohup bash scripts/hest_bench/run_scist_COAD.sh > scripts/hest_bench/run_scist_COAD.log 2>&1 &
set -euo pipefail
cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

export CUDA_VISIBLE_DEVICES=2
SCIST_DATASETS="COAD" bash scripts/hest_bench/run_local_scist.sh
