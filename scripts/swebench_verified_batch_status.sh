#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

sanitize_name() {
  local value="$1"
  value="${value//:/__}"
  value="${value//\//__}"
  printf '%s\n' "$value"
}

BATCH_SLUG="${SWEBENCH_BATCH_SLUG:-swebench-verified-50-qwen36-nvfp4-v1}"
MODEL_NAME="${SWEBENCH_MODEL_NAME:-bench-qwen36-35b-a3b-nvfp4-mtp}"
MODEL_SLUG="$(sanitize_name "$MODEL_NAME")"
OUTPUT_ROOT="${SWEBENCH_BATCH_OUTPUT_ROOT:-results/swebench-batches}"
LOG_DIR="${SWEBENCH_LOG_DIR:-logs}"
JOB_NAME="${SWEBENCH_JOB_NAME:-swebench-batch-${BATCH_SLUG}-${MODEL_SLUG}}"
PID_FILE="$LOG_DIR/$JOB_NAME.pid"
LOG_FILE="$LOG_DIR/$JOB_NAME.log"
BATCH_ROOT="$OUTPUT_ROOT/$BATCH_SLUG"

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Status: running, PID $pid"
  else
    echo "Status: not running, last PID $pid"
  fi
else
  echo "Status: no PID file"
fi

echo
echo "Latest batch summary:"
python3 scripts/report_swebench_batch.py \
  --root "$OUTPUT_ROOT" \
  --batch "$BATCH_SLUG" || true

if [[ -f "$LOG_FILE" ]]; then
  echo
  echo "Last 100 job log lines:"
  tail -n 100 "$LOG_FILE"
fi

echo
echo "Batch artifacts:"
find "$BATCH_ROOT" -maxdepth 3 -type f 2>/dev/null | sort || true
