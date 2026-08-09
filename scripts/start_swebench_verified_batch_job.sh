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

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Ignoring invalid PYTHON_BIN=$PYTHON_BIN" >&2
    unset PYTHON_BIN
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python interpreter not found. Create .venv or install python3." >&2
    exit 1
  fi
fi
if [[ "$PYTHON_BIN" == */* && "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$ROOT_DIR/$PYTHON_BIN"
fi

BATCH_SLUG="${SWEBENCH_BATCH_SLUG:-swebench-verified-50-qwen36-nvfp4-v1}"
MODEL_NAME="${SWEBENCH_MODEL_NAME:-bench-qwen36-35b-a3b-nvfp4-mtp}"
MODEL_SLUG="$(sanitize_name "$MODEL_NAME")"
LOG_DIR="${SWEBENCH_LOG_DIR:-logs}"
JOB_NAME="${SWEBENCH_JOB_NAME:-swebench-batch-${BATCH_SLUG}-${MODEL_SLUG}}"
PID_FILE="$LOG_DIR/$JOB_NAME.pid"
LOG_FILE="$LOG_DIR/$JOB_NAME.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "SWE-bench batch is already running with PID $old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

export PYTHONUNBUFFERED=1
nohup "$PYTHON_BIN" "$ROOT_DIR/scripts/run_swebench_verified_batch_vllm.py" \
  >>"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "Started SWE-bench Verified batch with PID $pid"
echo "Log: $LOG_FILE"
echo "Status: SWEBENCH_BATCH_SLUG=$BATCH_SLUG SWEBENCH_MODEL_NAME=$MODEL_NAME scripts/swebench_verified_batch_status.sh"
