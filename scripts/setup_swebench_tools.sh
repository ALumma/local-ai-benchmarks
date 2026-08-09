#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
"$PYTHON_BIN" -m pip install mini-swe-agent datasets docker tqdm

mkdir -p tools
if [[ ! -d "tools/SWE-bench/.git" ]]; then
  git clone https://github.com/princeton-nlp/SWE-bench.git tools/SWE-bench
fi

"$PYTHON_BIN" -m pip install -e tools/SWE-bench

echo "SWE-bench tools are installed for $PYTHON_BIN"
echo "Next: run a gold-patch harness check before using model predictions:"
echo "  $PYTHON_BIN -m swebench.harness.run_evaluation --predictions_path gold --max_workers 1 --instance_ids sympy__sympy-20590 --run_id validate-gold"
