#!/usr/bin/env bash
# SciSt benchmark — IDC only, GPU 0
# Usage: nohup bash scripts/hest_bench/run_scist_IDC.sh > scripts/hest_bench/run_scist_IDC.log 2>&1 &
set -euo pipefail
cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

export CUDA_VISIBLE_DEVICES=0
SCIST_DATASETS="IDC" SKIP_PREP=1 bash scripts/hest_bench/run_local_scist.sh
