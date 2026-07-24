# Local AI Benchmarks

Benchmark helpers for evaluating local Ollama models on a DGX Spark.

## AIME 2024 With System Prompt

Edit the system prompt here:

```text
prompts/aime-system.txt
```

Run a one-question smoke test:

```bash
AIME_LIMIT=1 AIME_RUN_SLUG=aime24-answer-format-smoke scripts/start_aime24_system_prompt_job.sh
```

Run the full five-model benchmark:

```bash
scripts/start_aime24_system_prompt_job.sh
```

Check progress or summarize completed models:

```bash
scripts/aime24_system_prompt_status.sh
```

The default run slug is:

```text
aime24-answer-format-v2
```

The summary includes an automatic scan for likely answer-format false negatives
in the saved sample logs.

## Expected Spark Setup

From `~/Desktop/local-ai-benchmarks` on the Spark:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
git clone https://github.com/EleutherAI/lm-evaluation-harness.git tools/lm-evaluation-harness
python -m pip install -e "tools/lm-evaluation-harness[api]"
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
