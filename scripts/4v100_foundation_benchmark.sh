#!/bin/bash
#SBATCH --job-name stai_fm_benchmark
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -o job_%j_foundation.out
#SBATCH -e job_%j_foundation.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

# Foundation-model benchmark: run Phikon2 / Prov-GigaPath / UNI2-h on every
# Xenium cell crop, fit a per-model multi-output ridge on the leakage-safe
# fixed-split train set, and produce eval-compatible prediction dumps that
# feed scripts/analyze_predictions.py for per-cell / per-gene / per-slide
# metrics.
#
# Stage 1 (GPU) is ImageNet-normalised inference over ~3.89M cell crops with
# three foundation models; expect ~4-8 h depending on GPU.
# Stage 2 (CPU) is three closed-form ridge fits and the eval analysis run.

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts
SPLIT_FILE="${REPO_ROOT}/splits/xenium_slide_split_v1.json"

module load gpu_prepare
module load python/3.8.x-anaconda || echo "python/3.8.x-anaconda module not found; continuing with conda env"

source activate "${ENV_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

# Replace YOUR_HF_TOKEN_HERE with your own HuggingFace token, or export
# HF_TOKEN before running. The gated pathology foundation models
# (Phikon2, Prov-GigaPath, UNI2-h) cannot be downloaded without one.
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"

export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID:-manual}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${SLURM_JOB_ID:-manual}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}"

cd "${REPO_ROOT}"

MODELS="${MODELS:-phikon2 prov-gigapath uni2-h}"
ALPHAS="${ALPHAS:-1.0 10.0 100.0}"
BATCH="${BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
FEATURES_ROOT="${FEATURES_ROOT:-${REPO_ROOT}/runs/baselines/foundation/features}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/baselines/foundation}"

SKIP_EXTRACT="${SKIP_EXTRACT:-0}"     # 1 = reuse existing features, only fit + eval
SKIP_FIT="${SKIP_FIT:-0}"             # 1 = reuse existing predictions, only analyse
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"   # 1 = reprocess slides even if h5 exists

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

# ----------------------------------------------------------------------
# Stage 1: foundation-model inference on Xenium cells
# ----------------------------------------------------------------------

if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  banner "Stage 1: foundation-model inference"
  date
  echo "  models:          ${MODELS}"
  echo "  features_root:   ${FEATURES_ROOT}"
  echo "  batch=${BATCH}   num_workers=${NUM_WORKERS}"

  FORCE_FLAG=""
  if [[ "${FORCE_EXTRACT}" == "1" ]]; then
    FORCE_FLAG="--force"
  fi

  python scripts/extract_foundation_features_xenium.py \
    --img_dir data/images \
    --out_dir "${FEATURES_ROOT}" \
    --models ${MODELS} \
    --batch "${BATCH}" --num_workers "${NUM_WORKERS}" \
    ${FORCE_FLAG}
  date
else
  echo "Stage 1 skipped because SKIP_EXTRACT=1"
fi

# ----------------------------------------------------------------------
# Stage 2: ridge fit + evaluation
# ----------------------------------------------------------------------

if [[ "${SKIP_FIT}" != "1" ]]; then
  banner "Stage 2: ridge fit on foundation features"
  date
  python scripts/baseline_foundation_predict.py \
    --features_root "${FEATURES_ROOT}" \
    --gene_dir data/gene_expression \
    --split_file "${SPLIT_FILE}" \
    --out_root "${OUT_ROOT}" \
    --models ${MODELS} \
    --alphas ${ALPHAS} \
    --splits val test
  date
else
  echo "Stage 2 skipped because SKIP_FIT=1"
fi

# ----------------------------------------------------------------------
# Stage 3: per-cell / per-gene / per-slide analysis for every (model, alpha)
# ----------------------------------------------------------------------

banner "Stage 3: analysis"
date
for model in ${MODELS}; do
  model_tag=$(echo "${model}" | tr '[:lower:]-' '[:upper:]_')
  for alpha in ${ALPHAS}; do
    alpha_dir="${OUT_ROOT}/${model_tag}/alpha_${alpha}"
    eval_dir="${alpha_dir}/eval"
    if [[ ! -d "${eval_dir}" ]]; then
      echo "  [skip] ${eval_dir} not found"
      continue
    fi
    echo "  analysing ${eval_dir}"
    python scripts/analyze_predictions.py \
      --eval_dir "${eval_dir}" \
      --splits val test
  done
done
date

banner "Foundation-model benchmark finished"
echo "  features:       ${FEATURES_ROOT}"
echo "  predictions:    ${OUT_ROOT}"
echo "  headline rows:"
echo "    grep 'mean' ${OUT_ROOT}/*/alpha_*/eval/test/analysis/summary.md"
