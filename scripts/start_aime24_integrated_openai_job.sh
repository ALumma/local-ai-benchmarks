#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SLUG="${AIME_RUN_SLUG:-aime24-qwen36-nvfp4-mtp-integrated-v1}"
LOG_DIR="${AIME_LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/$RUN_SLUG.pid"
LOG_FILE="$LOG_DIR/$RUN_SLUG.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "OpenAI-compatible integrated AIME job is already running with PID $old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

nohup "$ROOT_DIR/scripts/run_aime24_integrated_openai.sh" >>"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" >"$PID_FILE"

echo "Started OpenAI-compatible integrated AIME job with PID $pid"
echo "Log: $LOG_FILE"
echo "Status: AIME_RUN_SLUG=$RUN_SLUG scripts/aime24_integrated_status.sh"
