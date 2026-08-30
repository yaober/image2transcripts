set -euo pipefail

cd /endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/benchmark/sCellST

# Replace YOUR_HF_TOKEN_HERE with your own HuggingFace token, or export
# HF_TOKEN before running. The gated pathology foundation models
# (Phikon2, Prov-GigaPath, UNI2-h) cannot be downloaded without one.
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

PY=/work/OSPH/s439765/envs/scellst/bin/python
DRIVER=/endosome/archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts/hest_bench/run_scellst_task.py
DATA_DIR=/work/OSPH/s439765/scellst

DATASETS="${SCELLST_DATASETS:-IDC PAAD SKCM COAD LUNG}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EMBED="${SKIP_EMBED:-0}"
START_FOLD="${START_FOLD:-0}"

EXTRA_FLAGS=""
[[ "${SKIP_DOWNLOAD}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_download"
[[ "${SKIP_EMBED}"    == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_embed"

for TASK in $DATASETS; do
    echo "=== sCellST: $TASK ==="
    $PY "$DRIVER" \
        --task "$TASK" \
        --data_dir "$DATA_DIR" \
        --start_fold "$START_FOLD" \
        $EXTRA_FLAGS
done

echo "=== All sCellST tasks complete ==="
