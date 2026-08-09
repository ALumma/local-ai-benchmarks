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

RUN_SLUG="${SWEBENCH_RUN_SLUG:-swebench-verified-qwen36-nvfp4-comparison-v1}"
MODEL_NAME="${SWEBENCH_MODEL_NAME:-bench-qwen36-35b-a3b-nvfp4-mtp}"
MODEL_SLUG="$(sanitize_name "$MODEL_NAME")"
LOG_DIR="${SWEBENCH_LOG_DIR:-logs}"
JOB_NAME="${SWEBENCH_JOB_NAME:-swebench-${RUN_SLUG}-${MODEL_SLUG}}"
PID_FILE="$LOG_DIR/$JOB_NAME.pid"
LOG_FILE="$LOG_DIR/$JOB_NAME.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "SWE-bench job is already running with PID $old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

nohup "$ROOT_DIR/scripts/run_swebench_verified_one_vllm.sh" >>"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "Started SWE-bench Verified job with PID $pid"
echo "Log: $LOG_FILE"
echo "Status: SWEBENCH_RUN_SLUG=$RUN_SLUG SWEBENCH_MODEL_NAME=$MODEL_NAME scripts/swebench_verified_status.sh"
