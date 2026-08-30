#!/usr/bin/env bash
# Run SciSt benchmark on HEST tasks (IDC, PAAD, SKCM, COAD, LUNG).
#
# Mirrors run_local_scellst.sh but uses the scist conda env.
# Data is written to /work/OSPH/s439765/scist/<TASK>/.
# Results land in runs/hest_bench/summary.csv (encoder="scist").
#
# Environment flags:
#   SCIST_DATASETS  space-separated task list (default: IDC PAAD SKCM COAD LUNG)
#   SKIP_PREP       set to 1 to skip data preparation
#   SKIP_TRAIN      set to 1 to skip training (evaluate existing checkpoints only)
#   START_FOLD      fold index to start from (default: 0)
#
# Usage:
#   nohup bash scripts/hest_bench/run_local_scist.sh \
#       > scripts/hest_bench/run_local_scist.log 2>&1 &

set -euo pipefail

cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript

PY=/work/OSPH/s439765/envs/scist/bin/python
DRIVER=/endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts/hest_bench/run_scist_task.py
DATA_DIR=/work/OSPH/s439765/scist

DATASETS="${SCIST_DATASETS:-IDC PAAD SKCM COAD LUNG}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
START_FOLD="${START_FOLD:-0}"

EXTRA_FLAGS=""
[[ "${SKIP_PREP}"  == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_prep"
[[ "${SKIP_TRAIN}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_train"

for TASK in $DATASETS; do
    echo "=== SciSt: $TASK ==="
    $PY "$DRIVER" \
        --task "$TASK" \
        --data_dir "$DATA_DIR" \
        --start_fold "$START_FOLD" \
        $EXTRA_FLAGS
done

echo "=== All SciSt tasks complete ==="
