#!/bin/bash
#SBATCH --job-name stai_tcga_coad
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -o job_%j_tcga.out
#SBATCH -e job_%j_tcga.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

# Downstream analysis: run the best trained Image2Transcript checkpoint on
# TCGA-COAD whole-slide images, then correlate slide/patient-level gene
# predictions with the TCGA clinical variables.
#
# Stage 1 is single-GPU inference over ~462 SVS slides (one SVS at a time,
# batched tile inference).  Stage 2 is CPU-only association testing.

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts

TCGA_ROOT="/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data/TCGA/COAD"
SLIDES_DIR="${TCGA_ROOT}/slides"
CLINICAL_DIR="${TCGA_ROOT}/clinical files"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/fixedsplit_v3/full_seed42}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/tcga_coad}"
PRED_DIR="${OUT_ROOT}/predictions"
ANALYSIS_DIR="${OUT_ROOT}/analysis"

mkdir -p "${PRED_DIR}" "${ANALYSIS_DIR}"

module load gpu_prepare
module load python/3.8.x-anaconda || echo "python/3.8.x-anaconda module not found; continuing with conda env"

source activate "${ENV_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID:-manual}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

# OpenSlide on some compute nodes needs the system libopenslide; the conda
# env ships it under envs/.../lib, so make it visible.
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH}"

cd "${REPO_ROOT}"

BATCH="${BATCH:-128}"
MAX_TILES="${MAX_TILES:-1500}"
LIMIT="${LIMIT:-0}"                 # 0 = all slides; set to e.g. 5 for a smoke test
SKIP_PREDICT="${SKIP_PREDICT:-0}"   # 1 = reuse existing predictions, only run analysis
FORCE="${FORCE:-0}"                 # 1 = reprocess slides even if dump exists

print_banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

if [[ "${SKIP_PREDICT}" != "1" ]]; then
  print_banner "Stage 1: tile extraction + gene-expression inference"
  date
  echo "  checkpoint: ${RUN_DIR}/best_model.pt"
  echo "  slides:     ${SLIDES_DIR}"
  echo "  out:        ${PRED_DIR}"
  echo "  batch=${BATCH}  max_tiles_per_slide=${MAX_TILES}  limit=${LIMIT}"

  FORCE_FLAG=""
  if [[ "${FORCE}" == "1" ]]; then
    FORCE_FLAG="--force"
  fi

  python scripts/tcga_predict.py \
    --slides_dir "${SLIDES_DIR}" \
    --run_dir "${RUN_DIR}" \
    --gene_dir data/gene_expression \
    --out_dir "${PRED_DIR}" \
    --batch "${BATCH}" \
    --max_tiles_per_slide "${MAX_TILES}" \
    --limit "${LIMIT}" \
    ${FORCE_FLAG}

  date
else
  echo "Skipping prediction stage because SKIP_PREDICT=1"
fi

print_banner "Stage 2: gene <-> clinical associations"
date
python scripts/tcga_clinical_analysis.py \
  --predictions_dir "${PRED_DIR}" \
  --clinical_dir "${CLINICAL_DIR}" \
  --out_dir "${ANALYSIS_DIR}"
date

print_banner "TCGA downstream done"
echo "  slide predictions : ${PRED_DIR}"
echo "  analysis outputs  : ${ANALYSIS_DIR}"
echo "  top hits report   : ${ANALYSIS_DIR}/top_hits.md"
