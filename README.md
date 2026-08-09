# Local AI Benchmarks

Benchmark helpers for evaluating local Ollama models on a DGX Spark.

## AIME 2024 With System Prompt

Edit the system prompt here:

```text
prompts/aime-system.txt
```

Preferred path: run correctness and performance in the same `lm-eval` pass.

Run a one-question integrated smoke test:

```bash
AIME_MODELS="bench-qwen-next:q4" \
AIME_LIMIT=1 \
AIME_RUN_SLUG=aime24-qwen-integrated-smoke \
scripts/start_aime24_integrated_job.sh
```

Run full integrated AIME 24 for one model:

```bash
AIME_MODELS="bench-qwen-next:q4" \
AIME_RUN_SLUG=aime24-qwen-integrated-v1 \
scripts/start_aime24_integrated_job.sh
```

Check integrated progress/results:

```bash
AIME_RUN_SLUG=aime24-qwen-integrated-v1 scripts/aime24_integrated_status.sh
```

Display the latest full AIME result for all five models:

```bash
python3 scripts/report_aime24_models.py
```

Include artifact paths or export CSV:

```bash
python3 scripts/report_aime24_models.py --paths
python3 scripts/report_aime24_models.py --csv results/aime/aime24-model-report.csv
```

Run the integrated benchmark for all five models:

```bash
scripts/start_aime24_integrated_job.sh
```

## Qwen3.6 35B A3B NVFP4/MTP

Qwen3.6-35B-A3B is a 35B total / 3B active MoE model. The NVIDIA NVFP4
checkpoint is intended for vLLM, not Ollama. Use the stable served model name
below so reports group the result with the other benchmark aliases:

```text
bench-qwen36-35b-a3b-nvfp4-mtp
```

Serve the NVIDIA NVFP4 checkpoint on the Spark with vLLM:

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name bench-qwen36-35b-a3b-nvfp4-mtp \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

If vLLM rejects `--served-model-name`, remove that flag and run the benchmark
with:

```bash
AIME_MODELS="nvidia/Qwen3.6-35B-A3B-NVFP4"
```

Smoke test through the OpenAI-compatible vLLM endpoint:

```bash
AIME_LIMIT=1 \
AIME_RUN_SLUG=aime24-qwen36-nvfp4-mtp-smoke \
scripts/start_aime24_integrated_openai_job.sh
```

Full AIME 24:

```bash
AIME_RUN_SLUG=aime24-qwen36-nvfp4-mtp-integrated-v1 \
scripts/start_aime24_integrated_openai_job.sh
```

Check status with the same integrated status script:

```bash
AIME_RUN_SLUG=aime24-qwen36-nvfp4-mtp-integrated-v1 scripts/aime24_integrated_status.sh
```

Integrated output is stored together by run/model:

```text
results/aime/<run-slug>/<model>__<tag>/results_*.json
results/aime/<run-slug>/<model>__<tag>/samples_aime24_*.jsonl
results/aime/<run-slug>/<model>__<tag>/perf_integrated.jsonl
results/aime/<run-slug>/<model>__<tag>/perf_integrated_summary.json
logs/<run-slug>.log
```

The default integrated run slug is:

```text
aime24-integrated-answer-format-v1
```

For thinking-capable Ollama models, control thinking with top-level `think` via:

```bash
OLLAMA_THINK=false
```

Example Qwen3.6 Ollama MTP Q4 run:

```bash
AIME_MODELS="bench-qwen36-35b-a3b-mtp-q4:latest" \
AIME_RUN_SLUG=aime24-qwen36-mtp-q4-integrated-v1 \
OLLAMA_THINK=false \
scripts/start_aime24_integrated_job.sh
```

If a previous run used default thinking and produced empty answers, delete it
before rerunning:

```bash
rm -rf results/aime/aime24-qwen36-mtp-q4-integrated-v1
rm -f logs/aime24-qwen36-mtp-q4-integrated-v1.log
rm -f logs/aime24-qwen36-mtp-q4-integrated-v1.pid
```

Legacy two-pass path: run correctness first, then replay prompts for performance.

Run a one-question correctness-only smoke test:

```bash
AIME_LIMIT=1 AIME_RUN_SLUG=aime24-answer-format-smoke scripts/start_aime24_system_prompt_job.sh
```

Run the full five-model correctness-only benchmark:

```bash
scripts/start_aime24_system_prompt_job.sh
```

Check correctness-only progress or summarize completed models:

```bash
scripts/aime24_system_prompt_status.sh
```

Measure Qwen streaming performance by replaying saved AIME prompts:

```bash
python3 scripts/measure_ollama_chat_perf.py \
  --model bench-qwen-next:q4 \
  --run-dir results/aime/aime24-qwen-answer-format-v2 \
  --limit 3
```

Remove `--limit 3` to replay all 30 prompts. This reports client-measured time
to first token and continuing tokens per second, plus Ollama's server-side eval
tokens per second.

The default run slug is:

```text
aime24-answer-format-v2
```

The summary includes an automatic scan for likely answer-format false negatives
in the saved sample logs.

## SWE-bench Verified With vLLM

SWE-bench support is split into the two official phases:

1. `mini-swe-agent` generates a patch prediction through the local
   OpenAI-compatible vLLM endpoint.
2. The SWE-bench harness applies that patch in Docker and records whether the
   Verified task is resolved.

Install the SWE-bench tools on the machine that will run Docker:

```bash
scripts/setup_swebench_tools.sh
```

Run the gold-patch smoke test once before running model predictions:

```bash
scripts/swebench_verified_gold_check.sh
```

On ARM64 hosts, the gold check and one-task runner force SWE-bench image
generation to `arm64` and use `--namespace none`, which makes SWE-bench build
local images instead of pulling x86_64 images. Cached Conda environment files
contain x86_64 build hashes; the runners retain package versions while removing
those build hashes and selecting the ARM64 native linker package. The adaptation
is recorded in `run_metadata.json`. This keeps runs on the Spark comparable with
each other, but they are not byte-identical to official x86_64 leaderboard
environments. Override with `SWEBENCH_EVAL_ARCH=x86_64` only when you
intentionally want Docker emulation.

The default comparison task is the mini-SWE-agent documented Verified example:

```text
django__django-11099
```

Run NVIDIA first with the matching serving profile active:

```bash
cd ~/Desktop/local-ai-serving
MODEL_PROFILE=nvidia-nvfp4 ./scripts/serve_qwen36_nvfp4_mtp.sh
MODEL_PROFILE=nvidia-nvfp4 ./scripts/wait_for_server.sh

cd ~/Desktop/local-ai-benchmarks
SWEBENCH_MODEL_PROFILE=nvidia-nvfp4 \
SWEBENCH_MODEL_NAME=bench-qwen36-35b-a3b-nvfp4-mtp \
SWEBENCH_INSTANCE_ID=django__django-11099 \
SWEBENCH_RUN_SLUG=swebench-verified-qwen36-nvfp4-comparison-v1 \
scripts/start_swebench_verified_one_job.sh
```

Check progress:

```bash
SWEBENCH_MODEL_NAME=bench-qwen36-35b-a3b-nvfp4-mtp \
SWEBENCH_RUN_SLUG=swebench-verified-qwen36-nvfp4-comparison-v1 \
scripts/swebench_verified_status.sh
```

Then stop NVIDIA, start Unsloth, and run the identical task and run slug:

```bash
cd ~/Desktop/local-ai-serving
MODEL_PROFILE=nvidia-nvfp4 ./scripts/stop_server.sh
MODEL_PROFILE=unsloth-dynamic-nvfp4 ./scripts/serve_qwen36_nvfp4_mtp.sh
MODEL_PROFILE=unsloth-dynamic-nvfp4 ./scripts/wait_for_server.sh

cd ~/Desktop/local-ai-benchmarks
SWEBENCH_MODEL_PROFILE=unsloth-dynamic-nvfp4 \
SWEBENCH_MODEL_NAME=bench-qwen36-35b-a3b-unsloth-dynamic-nvfp4-mtp \
SWEBENCH_INSTANCE_ID=django__django-11099 \
SWEBENCH_RUN_SLUG=swebench-verified-qwen36-nvfp4-comparison-v1 \
scripts/start_swebench_verified_one_job.sh
```

Display both rows:

```bash
python3 scripts/report_swebench_results.py \
  --run swebench-verified-qwen36-nvfp4-comparison-v1 \
  --paths
```

Artifacts are kept under:

```text
results/swebench/<run-slug>/<model>/
results/swebench/<run-slug>/<model>/agent/preds.json
results/swebench/<run-slug>/<model>/agent/<instance>/<instance>.traj.json
results/swebench/<run-slug>/<model>/evaluation/
results/swebench/<run-slug>/<model>/run_metadata.json
logs/swebench-<run-slug>-<model>.log
```

The runner sends `chat_template_kwargs.enable_thinking=false` through LiteLLM's
OpenAI-compatible vLLM path by default. Override it with
`SWEBENCH_THINKING=true` only when you intentionally want to compare thinking
mode behavior.

## Expected Spark Setup

From `~/Desktop/local-ai-benchmarks` on the Spark:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
git clone https://github.com/EleutherAI/lm-evaluation-harness.git tools/lm-evaluation-harness
python -m pip install -e "tools/lm-evaluation-harness[api]"
scripts/setup_swebench_tools.sh
```

Ollama should be running locally on the Spark at:

```text
http://127.0.0.1:11434
```

The runner uses the benchmark aliases:

```text
bench-gemma31:q4
bench-gemma26:q4
bench-qwen-next:q4
bench-devstral:q4
bench-glm47:q4
bench-qwen36-35b-a3b-nvfp4-mtp
bench-qwen36-35b-a3b-mtp-q4:latest
```

## GitHub Setup

Create an empty GitHub repo in the browser, then from this directory:

```bash
git remote add origin git@github.com:ALumma/local-ai-benchmarks.git
git branch -M main
git push -u origin main
```

On the Spark, clone fresh:

```bash
cd ~/Desktop
git clone https://github.com/ALumma/local-ai-benchmarks.git local-ai-benchmarks
```

If `~/Desktop/local-ai-benchmarks` already exists on the Spark, keep it as a
runtime backup and copy the ignored local state into the fresh clone:

```bash
cd ~/Desktop
mv local-ai-benchmarks local-ai-benchmarks.local
git clone https://github.com/ALumma/local-ai-benchmarks.git local-ai-benchmarks
cp -a local-ai-benchmarks.local/.venv local-ai-benchmarks/ 2>/dev/null || true
cp -a local-ai-benchmarks.local/tools local-ai-benchmarks/ 2>/dev/null || true
cp -a local-ai-benchmarks.local/results local-ai-benchmarks/ 2>/dev/null || true
cp -a local-ai-benchmarks.local/logs local-ai-benchmarks/ 2>/dev/null || true
cd ~/Desktop/local-ai-benchmarks
```

After that, normal updates are:

```bash
git pull
```
