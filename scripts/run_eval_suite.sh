#!/bin/bash
#SBATCH --job-name stai_eval_suite
#SBATCH -p GPUv100s
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -o job_%j_eval_suite.out
#SBATCH -e job_%j_eval_suite.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

# Runs the single-GPU eval dump + metrics analysis on every trained
# checkpoint we care about for the STAI-X draft.  Intended for single-node
# use; the eval loop runs on one GPU (the others stay idle) since a single
# V100 takes roughly 30-60 min per split for ~600k cells.

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts
SPLIT_FILE="${REPO_ROOT}/splits/xenium_slide_split_v1.json"

module load gpu_prepare
module load python/3.8.x-anaconda || echo "python/3.8.x-anaconda module not found; continuing with conda env"

source activate "${ENV_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${REPO_ROOT}"

# Knobs
SPLITS="${SPLITS:-val test}"            # "val", "test" or "val test"
BATCH="${BATCH:-256}"
NUM_WORKERS="${NUM_WORKERS:-6}"
RETRIEVAL_N="${RETRIEVAL_N:-4000}"
RUN_ONLY="${RUN_ONLY:-}"
SKIP_RIDGE="${SKIP_RIDGE:-1}"            # 1 => skip the expensive feature extraction; set 0 to run it

# Every trained fixed-split checkpoint we currently have.  Adding new
# seed-43/seed-44 runs later only requires appending a path to this list.
RUN_DIRS=(
  runs/fixedsplit_v3/full_seed42
  runs/fixedsplit_v3/zinb_contrast_seed42
  runs/fixedsplit_v3/full_seed43
  runs/fixedsplit_v3/full_seed44
  runs/fixedsplit_mse/full_mse_seed42
  runs/fixedsplit_mse/zinb_contrast_mse_seed42
  runs/fixedsplit_foundation/PHIKON2_zinb_seed42
  runs/fixedsplit_foundation/PROV_GIGAPATH_zinb_seed42
  runs/fixedsplit_foundation/UNI2_H_zinb_seed42
)

print_banner() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

eval_and_analyze() {
  local run_dir="$1"

  if [[ ! -f "${run_dir}/best_model.pt" ]]; then
    echo "[skip] ${run_dir} missing best_model.pt"
    return 0
  fi

  if [[ -n "${RUN_ONLY}" && "${RUN_ONLY}" != "$(basename "${run_dir}")" ]]; then
    echo "[skip] ${run_dir} (RUN_ONLY=${RUN_ONLY})"
    return 0
  fi

  print_banner "Eval + analysis: ${run_dir}"
  date

  python scripts/eval_fixedsplit.py \
    --run_dir "${run_dir}" \
    --gene_dir data/gene_expression \
    --img_dir data/images \
    --split_file "${SPLIT_FILE}" \
    --splits ${SPLITS} \
    --batch "${BATCH}" \
    --num_workers "${NUM_WORKERS}"

  python scripts/analyze_predictions.py \
    --eval_dir "${run_dir}/eval" \
    --splits ${SPLITS} \
    --retrieval_n "${RETRIEVAL_N}"

  date
}

for rd in "${RUN_DIRS[@]}"; do
  eval_and_analyze "${rd}"
done

# Optional: frozen ViT + ridge baseline.  Feature extraction is expensive
# (~hours on a single V100 for 3.9M cells) so we gate it behind SKIP_RIDGE.
if [[ "${SKIP_RIDGE}" == "0" ]]; then
  print_banner "Frozen ViT + ridge baseline"
  date

  python scripts/baseline_vit_ridge.py \
    --gene_dir data/gene_expression \
    --img_dir data/images \
    --split_file "${SPLIT_FILE}" \
    --features_dir runs/baselines/vit_ridge/features \
    --out_dir runs/baselines/vit_ridge \
    --batch "${BATCH}" \
    --num_workers "${NUM_WORKERS}" \
    --alphas 1.0 10.0 100.0

  for alpha_dir in runs/baselines/vit_ridge/alpha_*; do
    [[ -d "${alpha_dir}/eval" ]] || continue
    python scripts/analyze_predictions.py \
      --eval_dir "${alpha_dir}/eval" \
      --splits ${SPLITS} \
      --retrieval_n "${RETRIEVAL_N}"
  done

  date
fi

print_banner "Eval suite finished"
