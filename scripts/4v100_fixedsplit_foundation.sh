#!/bin/bash
#SBATCH --job-name stai_fm_zinb
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -o job_%j_fm_zinb.out
#SBATCH -e job_%j_fm_zinb.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

# Belt-and-braces with the in-Python init_process_group(timeout=30min) bump:
# also raise the NCCL watchdog via env so any other collective is covered.
# Job 10476797 hit the default 10-min watchdog on the val_stats all_reduce
# during prov-gigapath ep1 validation.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# Fair foundation-model comparison: swap each pathology foundation model
# (Phikon2 / Prov-GigaPath / UNI2-h) in as the *image encoder* of our full
# Image2Transcript architecture, freeze it, and train the ZINB + contrast
# + align heads on the leakage-safe fixed slide-group split.
#
# This complements scripts/4v100_foundation_benchmark.sh (which fits a
# plain ridge on the same frozen features): now the predictor head has
# the same structure and objective as our own model, so any remaining gap
# can only be attributed to the image encoder, not to the head architecture
# or the loss function.

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts
SPLIT_FILE="${REPO_ROOT}/splits/xenium_slide_split_v1.json"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/fixedsplit_foundation}"
SUMMARY_ROOT="${RUN_ROOT}/summaries"
MASTER_SUMMARY="${RUN_ROOT}/train_log_summaries.csv"

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"

module load gpu_prepare
module load python/3.8.x-anaconda || echo "python/3.8.x-anaconda module not found; continuing with conda env"

source activate "${ENV_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID:-manual}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${SLURM_JOB_ID:-manual}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}"

# Replace YOUR_HF_TOKEN_HERE with your own HuggingFace token, or export
# HF_TOKEN before running. The gated pathology foundation models
# (Phikon2, Prov-GigaPath, UNI2-h) cannot be downloaded without one.
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"

cd "${REPO_ROOT}"

# Which FMs to run; default to all three wired ones.  Override via
# FOUNDATION_MODELS="phikon2 uni2-h" etc.
FOUNDATION_MODELS="${FOUNDATION_MODELS:-phikon2 prov-gigapath uni2-h}"
SEED="${SEED:-42}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_LOSS="${MAX_LOSS:-10000}"

# Per-backbone batch size.  UNI2-h is a ViT-H with SwiGLU + register tokens,
# Prov-GigaPath is ViT-L — neither fits at batch 128 on a V100 even with the
# backbone frozen (activations alone don't free since forward under no_grad
# still allocates).  Override via BATCH_<BACKBONE>.
BATCH_PHIKON2="${BATCH_PHIKON2:-128}"
BATCH_PROV_GIGAPATH="${BATCH_PROV_GIGAPATH:-64}"
BATCH_UNI2_H="${BATCH_UNI2_H:-32}"
BATCH_DEFAULT="${BATCH_DEFAULT:-64}"

# Frozen-backbone stability recipe (CLAUDE.md §3.3).  Default (lr_head=5e-4,
# grad_clip=1.0, warmup=0.05, learned T) regressed val by ep3 and triggered
# the all-batches-skipped cascade at ep4 (job 10470477, PHIKON2 only).
COMMON_ARGS=(
  --gene_dir data/gene_expression
  --img_dir data/images
  --split_file "${SPLIT_FILE}"
  --epochs 60
  --gpus 4
  --lr_backbone 0.0
  --lr_head 2e-4
  --weight_decay 0.05
  --warmup_ratio 0.10
  --grad_clip 0.5
  --fixed_temperature
  --max_loss "${MAX_LOSS}"
  --w_contrast 0.1
  --w_align 0.1
  --w_zinb 1.0
  --gene_mask_ratio 0.15
  --patience 10
  --freeze_backbone
)

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

backbone_batch() {
  local bb="$1"
  case "${bb}" in
    phikon2)        echo "${BATCH_PHIKON2}" ;;
    prov-gigapath)  echo "${BATCH_PROV_GIGAPATH}" ;;
    uni2-h)         echo "${BATCH_UNI2_H}" ;;
    *)              echo "${BATCH_DEFAULT}" ;;
  esac
}

run_backbone() {
  local backbone="$1"
  local run_name
  run_name="$(echo "${backbone}" | tr '[:lower:]-' '[:upper:]_')"_zinb_seed${SEED}
  local out_dir="${RUN_ROOT}/${run_name}"
  local batch
  batch=$(backbone_batch "${backbone}")

  if [[ "${FORCE_RERUN}" != "1" && -f "${out_dir}/last_model.pt" && -f "${out_dir}/train_log.csv" ]]; then
    banner "Skipping existing run ${run_name}"
    return 0
  fi

  banner "Training ${run_name} (backbone=${backbone}, batch=${batch})"
  date

  python model/main.py "${COMMON_ARGS[@]}" \
    --out_dir "${out_dir}" \
    --seed "${SEED}" \
    --image_backbone "${backbone}" \
    --batch "${batch}"

  date

  if [[ -f "${out_dir}/train_log.csv" ]]; then
    bash scripts/summarize_train_log.sh --header "${out_dir}/train_log.csv" \
      > "${SUMMARY_ROOT}/${run_name}_summary.csv"
  fi
}

banner "Foundation-model ZINB benchmark"
echo "Backbones:  ${FOUNDATION_MODELS}"
echo "Seed:       ${SEED}"
echo "Run root:   ${RUN_ROOT}"
echo "Freeze backbone: yes"
echo

for bb in ${FOUNDATION_MODELS}; do
  run_backbone "${bb}"
done

# Rebuild the master summary from every run directory that produced a log.
rm -f "${MASTER_SUMMARY}"
wrote_header=0
for dir in "${RUN_ROOT}"/*/; do
  log="${dir%/}/train_log.csv"
  [[ -f "${log}" ]] || continue
  if [[ "${wrote_header}" -eq 0 ]]; then
    bash scripts/summarize_train_log.sh --header "${log}" > "${MASTER_SUMMARY}"
    wrote_header=1
  else
    bash scripts/summarize_train_log.sh "${log}" >> "${MASTER_SUMMARY}"
  fi
done

banner "Foundation-model ZINB training finished"
echo "  run dirs:       ${RUN_ROOT}/*"
echo "  master summary: ${MASTER_SUMMARY}"
echo
echo "To produce headline per-cell and per-gene Pearson for every trained"
echo "backbone, submit scripts/run_eval_suite.sh with its RUN_DIRS edited"
echo "to point at ${RUN_ROOT}/*, or run this one-liner inline:"
echo
echo "    for d in ${RUN_ROOT}/*/; do"
echo "      [[ -f \"\${d}/best_model.pt\" ]] || continue"
echo "      python scripts/eval_fixedsplit.py --run_dir \"\${d%/}\" \\"
echo "        --gene_dir data/gene_expression --img_dir data/images \\"
echo "        --split_file ${SPLIT_FILE} --splits val test"
echo "      python scripts/analyze_predictions.py --eval_dir \"\${d%/}/eval\" --splits val test"
echo "    done"
