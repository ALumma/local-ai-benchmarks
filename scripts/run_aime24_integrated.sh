#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${AIME_TASK:-aime24}"
RUN_SLUG="${AIME_RUN_SLUG:-aime24-integrated-answer-format-v1}"
OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-results/aime}"
PROMPT_FILE="${AIME_SYSTEM_PROMPT_FILE:-prompts/aime-system.txt}"
BASE_URL="${OLLAMA_CHAT_BASE_URL:-http://127.0.0.1:11434/api/chat}"
TEMPERATURE="${AIME_TEMPERATURE:-0}"
MAX_GEN_TOKS="${AIME_MAX_GEN_TOKS:-4096}"
NUM_CONCURRENT="${AIME_NUM_CONCURRENT:-1}"
MAX_RETRIES="${AIME_MAX_RETRIES:-3}"
BATCH_SIZE="${AIME_BATCH_SIZE:-1}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "System prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ -n "${AIME_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=($AIME_MODELS)
else
  MODELS=(
    "bench-gemma31:q4"
    "bench-gemma26:q4"
    "bench-qwen-next:q4"
    "bench-devstral:q4"
    "bench-glm47:q4"
  )
fi

SYSTEM_INSTRUCTION="$(<"$PROMPT_FILE")"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"

limit_args=()
if [[ -n "${AIME_LIMIT:-}" ]]; then
  limit_args=(--limit "$AIME_LIMIT")
fi

mkdir -p "$OUTPUT_ROOT/$RUN_SLUG"

for model in "${MODELS[@]}"; do
  sanitized="${model//:/__}"
  sanitized="${sanitized//\//__}"
  model_output_dir="$OUTPUT_ROOT/$RUN_SLUG/$sanitized"
  perf_log_path="$model_output_dir/perf_integrated.jsonl"
  perf_summary_path="$model_output_dir/perf_integrated_summary.json"
  mkdir -p "$model_output_dir"
  : >"$perf_log_path"
  rm -f "$perf_summary_path"

  echo "==> Running integrated $TASK accuracy+performance for $model"
  "$PYTHON_BIN" scripts/lm_eval_ollama_timed.py run \
    --model ollama-chat-timed \
    --model_args "model=$model,base_url=$BASE_URL,num_concurrent=$NUM_CONCURRENT,max_retries=$MAX_RETRIES,perf_log_path=$perf_log_path,perf_summary_path=$perf_summary_path" \
    --tasks "$TASK" \
    --apply_chat_template \
    --system_instruction "$SYSTEM_INSTRUCTION" \
    --batch_size "$BATCH_SIZE" \
    --gen_kwargs "temperature=$TEMPERATURE,max_gen_toks=$MAX_GEN_TOKS" \
    --output_path "$OUTPUT_ROOT/$RUN_SLUG" \
    --log_samples \
    "${limit_args[@]}"
done
