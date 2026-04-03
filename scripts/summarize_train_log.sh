#!/usr/bin/env bash
set -euo pipefail

show_header=0
if [[ "${1:-}" == "--header" ]]; then
  show_header=1
  shift
fi

log_fp="${1:-}"
if [[ -z "$log_fp" ]]; then
  echo "Usage: bash scripts/summarize_train_log.sh [--header] path/to/train_log.csv" >&2
  exit 1
fi

if [[ ! -f "$log_fp" ]]; then
  echo "Missing log file: $log_fp" >&2
  exit 1
fi

run_name="$(basename "$(dirname "$log_fp")")"

if [[ "$show_header" -eq 1 ]]; then
  echo "run_name,log_path,best_val_epoch,best_val_loss,best_cos_epoch,best_cos_val,best_pearson_epoch,best_pearson_val,last_epoch,last_train_loss,last_val_loss,last_cos_val,last_pearson_val"
fi

awk -F, -v run_name="$run_name" -v log_fp="$log_fp" '
NR == 1 { next }
NR == 2 {
    best_val_loss = $3
    best_val_epoch = $1
    best_cos = $4
    best_cos_epoch = $1
    best_pearson = $5
    best_pearson_epoch = $1
}
NR >= 2 {
    if ($3 < best_val_loss) {
        best_val_loss = $3
        best_val_epoch = $1
    }
    if ($4 > best_cos) {
        best_cos = $4
        best_cos_epoch = $1
    }
    if ($5 > best_pearson) {
        best_pearson = $5
        best_pearson_epoch = $1
    }
    last_epoch = $1
    last_train_loss = $2
    last_val_loss = $3
    last_cos = $4
    last_pearson = $5
}
END {
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        run_name,
        log_fp,
        best_val_epoch,
        best_val_loss,
        best_cos_epoch,
        best_cos,
        best_pearson_epoch,
        best_pearson,
        last_epoch,
        last_train_loss,
        last_val_loss,
        last_cos,
        last_pearson
}
' "$log_fp"
