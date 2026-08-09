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

sanitize_name() {
  local value="$1"
  value="${value//:/__}"
  value="${value//\//__}"
  printf '%s\n' "$value"
}

bool_json() {
  local lowered
  lowered="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    1|true|yes|on) printf 'true\n' ;;
    0|false|no|off) printf 'false\n' ;;
    *)
      printf 'Expected boolean value, got: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1 && [[ ! -x "$command_name" ]]; then
    printf 'Required command is missing: %s\n' "$command_name" >&2
    return 1
  fi
}

: "${SWEBENCH_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWEBENCH_SPLIT:=test}"
: "${SWEBENCH_INSTANCE_ID:=django__django-11099}"
: "${SWEBENCH_MODEL_NAME:=bench-qwen36-35b-a3b-nvfp4-mtp}"
: "${SWEBENCH_MODEL_PROFILE:=}"
: "${SWEBENCH_BASE_URL:=http://127.0.0.1:8000/v1}"
: "${SWEBENCH_OUTPUT_ROOT:=results/swebench}"
: "${SWEBENCH_RUN_SLUG:=swebench-verified-qwen36-nvfp4-comparison-v1}"
: "${SWEBENCH_TEMPERATURE:=0}"
: "${SWEBENCH_MAX_TOKENS:=8192}"
: "${SWEBENCH_THINKING:=false}"
: "${SWEBENCH_MINI_WORKERS:=1}"
: "${SWEBENCH_MINI_ENVIRONMENT_CLASS:=docker}"
: "${SWEBENCH_EVAL_WORKERS:=1}"
: "${SWEBENCH_EVAL_TIMEOUT:=1800}"
: "${SWEBENCH_EVAL_CACHE_LEVEL:=env}"
: "${SWEBENCH_EVAL_CLEAN:=False}"
: "${SWEBENCH_SKIP_AGENT:=0}"
: "${SWEBENCH_SKIP_EVAL:=0}"
: "${SWEBENCH_REWRITE_EVAL_REPORTS:=False}"
: "${SWEBENCH_DOCKER_PULL_TIMEOUT:=1800}"
: "${SWEBENCH_ENV_COMMAND_TIMEOUT:=60}"
: "${SWEBENCH_FORCE_REBUILD_IMAGES:=0}"

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

SWEBENCH_API_KEY="${SWEBENCH_API_KEY:-${VLLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}}"
THINKING_JSON="$(bool_json "$SWEBENCH_THINKING")" || exit 2
MODEL_SLUG="$(sanitize_name "$SWEBENCH_MODEL_NAME")"
RUN_DIR="$ROOT_DIR/$SWEBENCH_OUTPUT_ROOT/$SWEBENCH_RUN_SLUG"
MODEL_DIR="$RUN_DIR/$MODEL_SLUG"
AGENT_DIR="$MODEL_DIR/agent"
EVAL_DIR="$MODEL_DIR/evaluation"
CONFIG_PATH="$MODEL_DIR/minisweagent_vllm_config.yaml"
REGISTRY_PATH="$MODEL_DIR/litellm_registry.json"
METADATA_PATH="$MODEL_DIR/run_metadata.json"
PREDICTIONS_PATH="$AGENT_DIR/preds.json"
AGENT_LOG="$MODEL_DIR/minisweagent_stdout.log"
EVAL_LOG="$MODEL_DIR/swebench_eval_stdout.log"
MODELS_JSON="$MODEL_DIR/vllm_models.json"
EVAL_RUN_ID="${SWEBENCH_RUN_SLUG}__${MODEL_SLUG}"
STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
START_EPOCH="$(date +%s)"
AGENT_SECONDS=""
EVAL_SECONDS=""

mkdir -p "$AGENT_DIR" "$EVAL_DIR"

require_command "$PYTHON_BIN" || exit 1
require_command curl || exit 1
if [[ "$SWEBENCH_SKIP_EVAL" != "1" || "$SWEBENCH_SKIP_AGENT" != "1" ]]; then
  require_command docker || exit 1
fi

"$PYTHON_BIN" - <<'PY' || {
import importlib.util
missing = [
    name
    for name in ("minisweagent", "datasets", "swebench", "docker")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY
  echo "Run scripts/setup_swebench_tools.sh first." >&2
  exit 1
}

if [[ "$SWEBENCH_SKIP_EVAL" != "1" ]]; then
  "$PYTHON_BIN" - <<'PY' || {
import importlib.util
if importlib.util.find_spec("swebench") is None:
    raise SystemExit("Missing Python package: swebench")
PY
    echo "Run scripts/setup_swebench_tools.sh first." >&2
    exit 1
  }
fi

curl_headers=(-H "Accept: application/json")
if [[ -n "$SWEBENCH_API_KEY" && "$SWEBENCH_API_KEY" != "EMPTY" ]]; then
  curl_headers+=(-H "Authorization: Bearer $SWEBENCH_API_KEY")
fi

if ! curl -fsS "${curl_headers[@]}" "$SWEBENCH_BASE_URL/models" >"$MODELS_JSON"; then
  echo "Could not reach vLLM model list at $SWEBENCH_BASE_URL/models" >&2
  exit 1
fi

SWEBENCH_MODEL_NAME="$SWEBENCH_MODEL_NAME" \
"$PYTHON_BIN" - "$MODELS_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
served_model = os.environ["SWEBENCH_MODEL_NAME"]
payload = json.loads(path.read_text(encoding="utf-8"))
ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
if ids and served_model not in ids:
    print(
        f"Warning: requested model {served_model!r} was not listed by vLLM. "
        f"Available ids: {', '.join(str(i) for i in ids)}",
        file=sys.stderr,
    )
PY

SWEBENCH_MODEL_NAME="$SWEBENCH_MODEL_NAME" \
SWEBENCH_BASE_URL="$SWEBENCH_BASE_URL" \
SWEBENCH_API_KEY="$SWEBENCH_API_KEY" \
SWEBENCH_TEMPERATURE="$SWEBENCH_TEMPERATURE" \
SWEBENCH_MAX_TOKENS="$SWEBENCH_MAX_TOKENS" \
SWEBENCH_THINKING_JSON="$THINKING_JSON" \
SWEBENCH_DOCKER_PULL_TIMEOUT="$SWEBENCH_DOCKER_PULL_TIMEOUT" \
SWEBENCH_ENV_COMMAND_TIMEOUT="$SWEBENCH_ENV_COMMAND_TIMEOUT" \
"$PYTHON_BIN" - "$REGISTRY_PATH" "$CONFIG_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

registry_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
served_model = os.environ["SWEBENCH_MODEL_NAME"]
hosted_model = f"hosted_vllm/{served_model}"
base_entry = {
    "max_tokens": 262144,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat",
}
registry = {
    served_model: base_entry,
    hosted_model: base_entry,
}
config = {
    "environment": {
        "timeout": int(os.environ["SWEBENCH_ENV_COMMAND_TIMEOUT"]),
        "pull_timeout": int(os.environ["SWEBENCH_DOCKER_PULL_TIMEOUT"]),
    },
    "model": {
        "model_name": hosted_model,
        "litellm_model_registry": str(registry_path),
        "cost_tracking": "ignore_errors",
        "model_kwargs": {
            "api_base": os.environ["SWEBENCH_BASE_URL"].rstrip("/"),
            "api_key": os.environ["SWEBENCH_API_KEY"],
            "temperature": float(os.environ["SWEBENCH_TEMPERATURE"]),
            "max_tokens": int(os.environ["SWEBENCH_MAX_TOKENS"]),
            "drop_params": True,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": json.loads(
                        os.environ["SWEBENCH_THINKING_JSON"]
                    )
                }
            },
        },
    },
}
registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

write_metadata() {
  local status="$1"
  local exit_code="$2"
  local finished_at=""
  local elapsed_seconds=""
  if [[ "$status" != "running" ]]; then
    finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    elapsed_seconds="$(( $(date +%s) - START_EPOCH ))"
  fi
  STATUS="$status" \
  EXIT_CODE="$exit_code" \
  STARTED_AT="$STARTED_AT" \
  FINISHED_AT="$finished_at" \
  ELAPSED_SECONDS="$elapsed_seconds" \
  SWEBENCH_DATASET="$SWEBENCH_DATASET" \
  SWEBENCH_SPLIT="$SWEBENCH_SPLIT" \
  SWEBENCH_INSTANCE_ID="$SWEBENCH_INSTANCE_ID" \
  SWEBENCH_MODEL_NAME="$SWEBENCH_MODEL_NAME" \
  SWEBENCH_MODEL_PROFILE="$SWEBENCH_MODEL_PROFILE" \
  SWEBENCH_BASE_URL="$SWEBENCH_BASE_URL" \
  SWEBENCH_RUN_SLUG="$SWEBENCH_RUN_SLUG" \
  SWEBENCH_TEMPERATURE="$SWEBENCH_TEMPERATURE" \
  SWEBENCH_MAX_TOKENS="$SWEBENCH_MAX_TOKENS" \
  SWEBENCH_THINKING="$SWEBENCH_THINKING" \
  SWEBENCH_MINI_ENVIRONMENT_CLASS="$SWEBENCH_MINI_ENVIRONMENT_CLASS" \
  SWEBENCH_EVAL_NAMESPACE="$SWEBENCH_EVAL_NAMESPACE" \
  SWEBENCH_EVAL_ARCH="$SWEBENCH_EVAL_ARCH" \
  SWEBENCH_EVAL_CACHE_LEVEL="$SWEBENCH_EVAL_CACHE_LEVEL" \
  SWEBENCH_EVAL_TIMEOUT="$SWEBENCH_EVAL_TIMEOUT" \
  AGENT_SECONDS="$AGENT_SECONDS" \
  EVAL_SECONDS="$EVAL_SECONDS" \
  AGENT_DIR="$AGENT_DIR" \
  EVAL_DIR="$EVAL_DIR" \
  PREDICTIONS_PATH="$PREDICTIONS_PATH" \
  CONFIG_PATH="$CONFIG_PATH" \
  REGISTRY_PATH="$REGISTRY_PATH" \
  MODELS_JSON="$MODELS_JSON" \
  "$PYTHON_BIN" - "$METADATA_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])

def maybe_float(name):
    value = os.environ.get(name, "")
    return float(value) if value else None

def maybe_int(name):
    value = os.environ.get(name, "")
    return int(value) if value else None

payload = {
    "schema_version": 1,
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["EXIT_CODE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ.get("FINISHED_AT") or None,
    "elapsed_seconds": maybe_int("ELAPSED_SECONDS"),
    "dataset": os.environ["SWEBENCH_DATASET"],
    "split": os.environ["SWEBENCH_SPLIT"],
    "instance_id": os.environ["SWEBENCH_INSTANCE_ID"],
    "model": os.environ["SWEBENCH_MODEL_NAME"],
    "model_profile": os.environ["SWEBENCH_MODEL_PROFILE"] or None,
    "base_url": os.environ["SWEBENCH_BASE_URL"],
    "run_slug": os.environ["SWEBENCH_RUN_SLUG"],
    "settings": {
        "temperature": float(os.environ["SWEBENCH_TEMPERATURE"]),
        "max_tokens": int(os.environ["SWEBENCH_MAX_TOKENS"]),
        "thinking": os.environ["SWEBENCH_THINKING"],
        "mini_environment_class": os.environ["SWEBENCH_MINI_ENVIRONMENT_CLASS"],
        "eval_namespace": os.environ["SWEBENCH_EVAL_NAMESPACE"],
        "eval_arch": os.environ["SWEBENCH_EVAL_ARCH"],
        "eval_cache_level": os.environ["SWEBENCH_EVAL_CACHE_LEVEL"],
        "eval_timeout": int(os.environ["SWEBENCH_EVAL_TIMEOUT"]),
    },
    "durations": {
        "agent_seconds": maybe_float("AGENT_SECONDS"),
        "evaluation_seconds": maybe_float("EVAL_SECONDS"),
    },
    "paths": {
        "agent_dir": os.environ["AGENT_DIR"],
        "evaluation_dir": os.environ["EVAL_DIR"],
        "predictions": os.environ["PREDICTIONS_PATH"],
        "mini_config": os.environ["CONFIG_PATH"],
        "litellm_registry": os.environ["REGISTRY_PATH"],
        "vllm_models": os.environ["MODELS_JSON"],
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_metadata "running" 0

echo "==> SWE-bench Verified agent run"
echo "    model: $SWEBENCH_MODEL_NAME"
echo "    instance: $SWEBENCH_INSTANCE_ID"
echo "    output: $MODEL_DIR"
echo "    docker arch: $SWEBENCH_EVAL_ARCH"
echo "    eval namespace: $SWEBENCH_EVAL_NAMESPACE"

run_status=0

if [[ "$SWEBENCH_SKIP_AGENT" == "1" ]]; then
  echo "==> Skipping agent phase; using existing predictions at $PREDICTIONS_PATH"
else
  agent_args=(
    scripts/run_swebench_agent_vllm_one.py
    --dataset "$SWEBENCH_DATASET"
    --split "$SWEBENCH_SPLIT"
    --instance-id "$SWEBENCH_INSTANCE_ID"
    --output "$AGENT_DIR"
    --config "$CONFIG_PATH"
    --served-model-name "$SWEBENCH_MODEL_NAME"
    --arch "$SWEBENCH_EVAL_ARCH"
    --environment-class "$SWEBENCH_MINI_ENVIRONMENT_CLASS"
  )
  if [[ "$SWEBENCH_FORCE_REBUILD_IMAGES" == "1" ]]; then
    agent_args+=(--force-rebuild-images)
  fi

  phase_start="$(date +%s)"
  set +e
  "$PYTHON_BIN" "${agent_args[@]}" 2>&1 | tee "$AGENT_LOG"
  agent_code="${PIPESTATUS[0]}"
  set -e
  AGENT_SECONDS="$(( $(date +%s) - phase_start ))"
  if [[ "$agent_code" != "0" ]]; then
    run_status="$agent_code"
  fi
fi

if [[ "$run_status" == "0" ]]; then
  if [[ ! -f "$PREDICTIONS_PATH" ]]; then
    echo "Prediction file was not produced: $PREDICTIONS_PATH" >&2
    run_status=1
  else
    set +e
    SWEBENCH_INSTANCE_ID="$SWEBENCH_INSTANCE_ID" \
    "$PYTHON_BIN" - "$PREDICTIONS_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

instance_id = os.environ["SWEBENCH_INSTANCE_ID"]
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(payload, dict):
    predictions = payload
else:
    predictions = {row.get("instance_id"): row for row in payload}
if instance_id not in predictions:
    raise SystemExit(f"Prediction file does not contain {instance_id}")
patch = predictions[instance_id].get("model_patch") or ""
print(f"Prediction patch bytes: {len(patch.encode('utf-8'))}")
PY
    verify_code="$?"
    set -e
    if [[ "$verify_code" != "0" ]]; then
      run_status="$verify_code"
    fi
  fi
fi

if [[ "$run_status" == "0" && "$SWEBENCH_SKIP_EVAL" == "1" ]]; then
  echo "==> Skipping official SWE-bench evaluation."
elif [[ "$run_status" == "0" ]]; then
  echo "==> Official SWE-bench evaluation"
  eval_args=(
    scripts/swebench_run_evaluation_arch.py
    --dataset_name "$SWEBENCH_DATASET"
    --split "$SWEBENCH_SPLIT"
    --predictions_path "$PREDICTIONS_PATH"
    --max_workers "$SWEBENCH_EVAL_WORKERS"
    --instance_ids "$SWEBENCH_INSTANCE_ID"
    --run_id "$EVAL_RUN_ID"
    --timeout "$SWEBENCH_EVAL_TIMEOUT"
    --cache_level "$SWEBENCH_EVAL_CACHE_LEVEL"
    --clean "$SWEBENCH_EVAL_CLEAN"
    --namespace "$SWEBENCH_EVAL_NAMESPACE"
    --arch "$SWEBENCH_EVAL_ARCH"
    --rewrite_reports "$SWEBENCH_REWRITE_EVAL_REPORTS"
  )
  eval_args[0]="$ROOT_DIR/${eval_args[0]}"

  phase_start="$(date +%s)"
  set +e
  (
    cd "$EVAL_DIR" || exit 1
    "$PYTHON_BIN" "${eval_args[@]}"
  ) 2>&1 | tee "$EVAL_LOG"
  eval_code="${PIPESTATUS[0]}"
  set -e
  EVAL_SECONDS="$(( $(date +%s) - phase_start ))"
  if [[ "$eval_code" != "0" ]]; then
    run_status="$eval_code"
  fi
fi

if [[ "$run_status" == "0" ]]; then
  write_metadata "completed" 0
  echo "==> Done. Summary:"
  "$PYTHON_BIN" scripts/report_swebench_results.py --run "$SWEBENCH_RUN_SLUG" --model "$SWEBENCH_MODEL_NAME"
else
  write_metadata "failed" "$run_status"
  echo "==> Failed with exit code $run_status" >&2
fi

exit "$run_status"
