#!/bin/bash
#SBATCH --job-name hest_bench
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -o job_%j_hest_bench.out
#SBATCH -e job_%j_hest_bench.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

# Run HEST-Bench on the 5 Xenium tasks for Image2Transcript and GHIST
# encoders. sCellST is run separately via its native pipeline.

set -euo pipefail

cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

# Replace YOUR_HF_TOKEN_HERE with your own HuggingFace token, or export
# HF_TOKEN before running. The gated pathology foundation models
# (Phikon2, Prov-GigaPath, UNI2-h) cannot be downloaded without one.
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

PY=/work/OSPH/s439765/envs/hest_bench/bin/python

DATASETS="IDC PAAD SKCM COAD LUNG"
DEVICE="${HEST_BENCH_DEVICE:-auto}"

echo "=== Image2Transcript ==="
$PY scripts/hest_bench/run_bench.py \
    --encoder image2transcript \
    --datasets $DATASETS \
    --batch_size 128 \
    --num_workers 4 \
    --device "$DEVICE"

echo "=== GHIST (random-init backbone) ==="
$PY scripts/hest_bench/run_bench.py \
    --encoder ghist \
    --datasets $DATASETS \
    --batch_size 128 \
    --num_workers 4 \
    --device "$DEVICE"

echo "=== Aggregating ==="
$PY scripts/hest_bench/aggregate_results.py
