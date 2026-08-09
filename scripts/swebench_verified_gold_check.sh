#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    :
  else
    echo "Ignoring invalid PYTHON_BIN=$PYTHON_BIN" >&2
    unset PYTHON_BIN
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python interpreter not found. Create .venv or install python3." >&2
    exit 1
  fi
fi

: "${SWEBENCH_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWEBENCH_SPLIT:=test}"
: "${SWEBENCH_INSTANCE_ID:=sympy__sympy-20590}"
: "${SWEBENCH_EVAL_WORKERS:=1}"
: "${SWEBENCH_EVAL_TIMEOUT:=1800}"
: "${SWEBENCH_EVAL_CACHE_LEVEL:=env}"
: "${SWEBENCH_EVAL_CLEAN:=False}"
: "${SWEBENCH_REWRITE_EVAL_REPORTS:=False}"
: "${SWEBENCH_GOLD_RUN_ID:=validate-gold}"
: "${SWEBENCH_OUTPUT_ROOT:=results/swebench}"

if [[ -z "${SWEBENCH_EVAL_NAMESPACE+x}" ]]; then
  case "$(uname -m)" in
    arm64|aarch64) SWEBENCH_EVAL_NAMESPACE="none" ;;
    *) SWEBENCH_EVAL_NAMESPACE="swebench" ;;
  esac
fi
if [[ -z "${SWEBENCH_EVAL_ARCH+x}" ]]; then
  case "$(uname -m)" in
    arm64|aarch64) SWEBENCH_EVAL_ARCH="arm64" ;;
    *) SWEBENCH_EVAL_ARCH="x86_64" ;;
  esac
fi

RUN_DIR="$ROOT_DIR/$SWEBENCH_OUTPUT_ROOT/gold/$SWEBENCH_GOLD_RUN_ID"
mkdir -p "$RUN_DIR"

echo "==> SWE-bench gold-patch check"
echo "    instance: $SWEBENCH_INSTANCE_ID"
echo "    docker arch: $SWEBENCH_EVAL_ARCH"
echo "    eval namespace: $SWEBENCH_EVAL_NAMESPACE"
echo "    output: $RUN_DIR"

(
  cd "$RUN_DIR" || exit 1
  "$PYTHON_BIN" "$ROOT_DIR/scripts/swebench_run_evaluation_arch.py" \
    --dataset_name "$SWEBENCH_DATASET" \
    --split "$SWEBENCH_SPLIT" \
    --predictions_path gold \
    --max_workers "$SWEBENCH_EVAL_WORKERS" \
    --instance_ids "$SWEBENCH_INSTANCE_ID" \
    --run_id "$SWEBENCH_GOLD_RUN_ID" \
    --timeout "$SWEBENCH_EVAL_TIMEOUT" \
    --cache_level "$SWEBENCH_EVAL_CACHE_LEVEL" \
    --clean "$SWEBENCH_EVAL_CLEAN" \
    --namespace "$SWEBENCH_EVAL_NAMESPACE" \
    --arch "$SWEBENCH_EVAL_ARCH" \
    --rewrite_reports "$SWEBENCH_REWRITE_EVAL_REPORTS"
)
