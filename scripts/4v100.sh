#!/bin/bash
#SBATCH --job-name aaai_img2trans
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 100-23:0:00
#SBATCH -o job_%j_aaai.out
#SBATCH -e job_%j_aaai.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

set -euo pipefail

REPO_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
ENV_ROOT=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts
RUN_ROOT="${REPO_ROOT}/runs/aaai"
SUMMARY_ROOT="${RUN_ROOT}/summaries"
MASTER_SUMMARY="${RUN_ROOT}/train_log_summaries.csv"

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"

module load gpu_prepare
module load python/3.8.x-anaconda || echo "python/3.8.x-anaconda module not found; continuing with conda env"

source activate "${ENV_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

NVRTC_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=${ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

# Writable caches help scanpy/numba and matplotlib behave cleanly inside jobs.
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="/tmp/mpl_config_${SLURM_JOB_ID:-manual}"

cd "${REPO_ROOT}"

COMMON_ARGS=(
  --gene_dir data/gene_expression
  --img_dir data/images
  --batch 128
  --epochs 100
  --gpus 4
  --lr_backbone 1e-5
  --lr_head 5e-4
  --weight_decay 0.05
  --warmup_ratio 0.05
  --grad_clip 1.0
  --w_contrast 0.1
  --w_align 0.1
  --w_zinb 1.0
  --gene_mask_ratio 0.15
  --patience 15
)

RUN_ONLY="${RUN_ONLY:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SUMMARY_RUN_NAMES=(
  full_seed42
  full_seed43
  full_seed44
  zinb_only_seed42
  zinb_contrast_seed42
  zinb_align_seed42
  full_seed42_mask0
  full_seed42_mask30
  full_seed42_backbone1e6
)

print_banner() {
  local msg="$1"
  echo
  echo "============================================================"
  echo "${msg}"
  echo "============================================================"
}

write_summary() {
  local out_dir="$1"
  local run_name
  local log_fp
  local summary_fp

  run_name="$(basename "${out_dir}")"
  log_fp="${out_dir}/train_log.csv"
  summary_fp="${SUMMARY_ROOT}/${run_name}_summary.csv"

  if [[ ! -f "${log_fp}" ]]; then
    echo "Missing train log for ${run_name}: ${log_fp}" >&2
    return 1
  fi

  bash scripts/summarize_train_log.sh --header "${log_fp}" > "${summary_fp}"

  echo "Summary written to ${summary_fp}"
}

refresh_master_summary() {
  local run_name
  local log_fp
  local wrote_any=0

  rm -f "${MASTER_SUMMARY}"

  for run_name in "${SUMMARY_RUN_NAMES[@]}"; do
    log_fp="${RUN_ROOT}/${run_name}/train_log.csv"
    if [[ ! -f "${log_fp}" ]]; then
      continue
    fi

    if [[ "${wrote_any}" -eq 0 ]]; then
      bash scripts/summarize_train_log.sh --header "${log_fp}" > "${MASTER_SUMMARY}"
      wrote_any=1
    else
      bash scripts/summarize_train_log.sh "${log_fp}" >> "${MASTER_SUMMARY}"
    fi
  done
}

run_experiment() {
  local phase="$1"
  local run_name="$2"
  local seed="$3"
  local lr_backbone="$4"
  local w_contrast="$5"
  local w_align="$6"
  local gene_mask_ratio="$7"

  local out_dir="${RUN_ROOT}/${run_name}"
  local extra_args=(
    --out_dir "${out_dir}"
    --seed "${seed}"
    --lr_backbone "${lr_backbone}"
    --w_contrast "${w_contrast}"
    --w_align "${w_align}"
    --gene_mask_ratio "${gene_mask_ratio}"
  )

  if [[ -n "${RUN_ONLY}" && "${RUN_ONLY}" != "${run_name}" ]]; then
    echo "Skipping ${run_name} because RUN_ONLY=${RUN_ONLY}"
    return 0
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${out_dir}/last_model.pt" && -f "${out_dir}/train_log.csv" ]]; then
    print_banner "Skipping existing run ${run_name}"
    write_summary "${out_dir}"
    refresh_master_summary
    return 0
  fi

  print_banner "Starting ${phase}: ${run_name}"
  echo "Seed=${seed} lr_backbone=${lr_backbone} w_contrast=${w_contrast} w_align=${w_align} gene_mask_ratio=${gene_mask_ratio}"
  echo "Output directory: ${out_dir}"
  date

  python model/main.py "${COMMON_ARGS[@]}" "${extra_args[@]}"

  date
  write_summary "${out_dir}"
  refresh_master_summary
}

print_banner "AAAI experiment sweep"
echo "Repo root: ${REPO_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Master summary: ${MASTER_SUMMARY}"
echo "RUN_ONLY=${RUN_ONLY:-<all>}"
echo "FORCE_RERUN=${FORCE_RERUN}"

# Phase 1: stability of the full model across seeds.
run_experiment "Phase 1" "full_seed42" 42 1e-5 0.1 0.1 0.15
run_experiment "Phase 1" "full_seed43" 43 1e-5 0.1 0.1 0.15
run_experiment "Phase 1" "full_seed44" 44 1e-5 0.1 0.1 0.15

# Phase 2: objective ablations.
run_experiment "Phase 2" "zinb_only_seed42" 42 1e-5 0.0 0.0 0.15
run_experiment "Phase 2" "zinb_contrast_seed42" 42 1e-5 0.1 0.0 0.15
run_experiment "Phase 2" "zinb_align_seed42" 42 1e-5 0.0 0.1 0.15

# Phase 3: sensitivity checks.
run_experiment "Phase 3" "full_seed42_mask0" 42 1e-5 0.1 0.1 0.0
run_experiment "Phase 3" "full_seed42_mask30" 42 1e-5 0.1 0.1 0.30
run_experiment "Phase 3" "full_seed42_backbone1e6" 42 1e-6 0.1 0.1 0.15

refresh_master_summary

print_banner "Sweep finished"
echo "Per-run summaries are in ${SUMMARY_ROOT}"
echo "Combined summary is in ${MASTER_SUMMARY}"
