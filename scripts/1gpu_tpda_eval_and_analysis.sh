#!/bin/bash
# Run the tPdA rebuttal analysis (single-cell resolution + crop-size sensitivity).
#
# Step 1: Run eval_fixedsplit.py for the 94×94 model (single-GPU inference, ~30 min).
# Step 2: Run the full analysis script.
#
# Submit with:
#   sbatch scripts/1gpu_tpda_eval_and_analysis.sh
#
# Or run interactively on a GPU node:
#   bash scripts/1gpu_tpda_eval_and_analysis.sh

#SBATCH --job-name tpda_analysis
#SBATCH -p GPUA100
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o job_%j_tpda_analysis.out
#SBATCH -e job_%j_tpda_analysis.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts

module load gpu_prepare 2>/dev/null || true
source activate "${ENV_ROOT}"

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

# ------------------------------------------------------------------ #
#  Step 1: run 94px model inference (skip if eval already exists)    #
# ------------------------------------------------------------------ #
EVAL_94="${REPO_ROOT}/output_94/eval/test"

if [ -f "${EVAL_94}/mu.npy" ]; then
    echo "[Step 1] 94px eval already exists at ${EVAL_94} — skipping inference."
else
    echo "[Step 1] Running 94×94 model inference ..."
    python scripts/eval_fixedsplit.py \
        --run_dir    output_94 \
        --gene_dir   data/gene_expression \
        --img_dir    data/images_94 \
        --split_file splits/xenium_slide_split_v1.json \
        --splits     test \
        --batch      512 \
        --num_workers 6
    echo "[Step 1] Done. Predictions at ${EVAL_94}"
fi

# ------------------------------------------------------------------ #
#  Step 1.5: ensure pyarrow is available (needed to read parquet)    #
# ------------------------------------------------------------------ #
python -c "import pyarrow" 2>/dev/null || {
    echo "[Step 1.5] pyarrow not found — installing ..."
    pip install --quiet pyarrow
}

# ------------------------------------------------------------------ #
#  Step 2: full analysis                                              #
# ------------------------------------------------------------------ #
echo "[Step 2] Running tPdA analysis ..."

python scripts/tpda_single_cell_analysis.py \
    --pred_188   runs/fixedsplit_v3/full_seed42/eval/test \
    --pred_94    output_94/eval/test \
    --ann_parquet /archive/DPDS/Xiao_lab/shared/jia_yao/xenium32/STORM/analysis/cell_type_annotation/cell_typing_results/merged_cell_type_annotations.parquet \
    --gene_dir   data/gene_expression \
    --img_dir    data/images \
    --out_dir    runs/tpda_single_cell_analysis \
    --n_boot     1000

echo "[Done] Results in runs/tpda_single_cell_analysis/"
