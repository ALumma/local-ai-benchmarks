#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${AIME_TASK:-aime24}"
RUN_SLUG="${AIME_RUN_SLUG:-aime24-system-prompt-v1}"
OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-results/aime}"
CACHE_ROOT="${AIME_CACHE_ROOT:-$OUTPUT_ROOT/$RUN_SLUG-cache}"
PROMPT_FILE="${AIME_SYSTEM_PROMPT_FILE:-prompts/aime-system.txt}"
BASE_URL="${OLLAMA_OPENAI_BASE_URL:-http://127.0.0.1:11434/v1/chat/completions}"
TEMPERATURE="${AIME_TEMPERATURE:-0}"
MAX_GEN_TOKS="${AIME_MAX_GEN_TOKS:-4096}"
NUM_CONCURRENT="${AIME_NUM_CONCURRENT:-1}"
MAX_RETRIES="${AIME_MAX_RETRIES:-3}"
BATCH_SIZE="${AIME_BATCH_SIZE:-1}"

if [[ -n "${LM_EVAL_BIN:-}" ]]; then
  # shellcheck disable=SC2206
  LM_EVAL_CMD=($LM_EVAL_BIN)
elif [[ -x ".venv/bin/python" ]]; then
  LM_EVAL_CMD=(.venv/bin/python -m lm_eval)
else
  LM_EVAL_CMD=(lm-eval)
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

mkdir -p "$OUTPUT_ROOT/$RUN_SLUG" "$CACHE_ROOT"

for model in "${MODELS[@]}"; do
  sanitized="${model//:/__}"
  sanitized="${sanitized//\//__}"
  cache_path="$CACHE_ROOT/$sanitized/cache"
  mkdir -p "$(dirname "$cache_path")"

  echo "==> Running $TASK for $model"
  "${LM_EVAL_CMD[@]}" run \
    --model local-chat-completions \
    --model_args "model=$model,base_url=$BASE_URL,num_concurrent=$NUM_CONCURRENT,max_retries=$MAX_RETRIES" \
    --tasks "$TASK" \
    --apply_chat_template \
    --system_instruction "$SYSTEM_INSTRUCTION" \
    --batch_size "$BATCH_SIZE" \
    --gen_kwargs "temperature=$TEMPERATURE,max_gen_toks=$MAX_GEN_TOKS" \
    --use_cache "$cache_path" \
    --output_path "$OUTPUT_ROOT/$RUN_SLUG" \
    --log_samples \
    "${limit_args[@]}"
done
