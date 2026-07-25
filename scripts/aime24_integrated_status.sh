#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SLUG="${AIME_RUN_SLUG:-aime24-integrated-answer-format-v1}"
OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-results/aime}"
LOG_DIR="${AIME_LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/$RUN_SLUG.pid"
LOG_FILE="$LOG_DIR/$RUN_SLUG.log"

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
echo "Latest integrated summary:"
python3 scripts/summarize_aime_results.py "$OUTPUT_ROOT/$RUN_SLUG" || true

if [[ -f "$LOG_FILE" ]]; then
  echo
  echo "Last 40 job log lines:"
  tail -n 40 "$LOG_FILE"
fi

echo
echo "Performance log files:"
find "$OUTPUT_ROOT/$RUN_SLUG" -maxdepth 3 -type f -name 'perf_integrated*.json*' 2>/dev/null | sort || true
