#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL = "bench-qwen-next:q4"
DEFAULT_RUN_DIR = "results/aime/aime24-qwen-answer-format-v2"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_SYSTEM_PROMPT = "prompts/aime-system.txt"


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "__")


def latest_samples(run_dir: Path, model: str) -> Path:
    model_dir = run_dir / sanitize_model_name(model)
    samples = sorted(model_dir.glob("samples_*.jsonl"))
    if not samples:
        raise SystemExit(f"No samples_*.jsonl found in {model_dir}")
    return samples[-1]


def load_sample_messages(samples_path: Path, limit: int | None) -> list[dict]:
    rows = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            raw_messages = sample["arguments"]["gen_args_0"]["arg_0"][0]
            messages = json.loads(raw_messages)
            rows.append(
                {
                    "doc_id": sample["doc_id"],
                    "target": sample.get("target"),
                    "messages": messages,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def ns_to_seconds(value: int | float | None) -> float | None:
    if value in (None, 0):
        return None
    return float(value) / 1_000_000_000


def rate(count: int | None, seconds: float | None) -> float | None:
    if not count or not seconds or seconds <= 0:
        return None
    return count / seconds


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def stream_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    sample: dict,
    max_gen_toks: int,
    temperature: float,
    seed: int,
    timeout_seconds: int,
) -> dict:
    messages = [{"role": "system", "content": system_prompt}, *sample["messages"]]
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_gen_toks,
            "seed": seed,
            "stop": ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"],
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    first_chunk_at = None
    first_content_at = None
    content_parts = []
    final = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            for line in response:
                now = time.monotonic()
                if not line.strip():
                    continue
                if first_chunk_at is None:
                    first_chunk_at = now
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content") or ""
                if content:
                    if first_content_at is None:
                        first_content_at = now
                    content_parts.append(content)
                if chunk.get("done"):
                    final = chunk
                    break
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request to Ollama failed: {exc}") from exc

    end = time.monotonic()
    eval_count = final.get("eval_count")
    eval_duration_s = ns_to_seconds(final.get("eval_duration"))
    prompt_eval_count = final.get("prompt_eval_count")
    prompt_eval_duration_s = ns_to_seconds(final.get("prompt_eval_duration"))
    load_duration_s = ns_to_seconds(final.get("load_duration"))
    total_duration_s = ns_to_seconds(final.get("total_duration"))
    ttft_s = first_content_at - start if first_content_at else None
    first_chunk_s = first_chunk_at - start if first_chunk_at else None
    wall_s = end - start
    continuing_s = wall_s - ttft_s if ttft_s is not None else None
    client_continuing_tps = rate(
        max(eval_count - 1, 0) if isinstance(eval_count, int) else None,
        continuing_s,
    )

    return {
        "doc_id": sample["doc_id"],
        "target": sample["target"],
        "model": model,
        "wall_seconds": wall_s,
        "time_to_first_chunk_seconds": first_chunk_s,
        "time_to_first_token_seconds": ttft_s,
        "client_continuing_tokens_per_second": client_continuing_tps,
        "ollama_eval_tokens_per_second": rate(eval_count, eval_duration_s),
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration_s,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_seconds": prompt_eval_duration_s,
        "prompt_eval_tokens_per_second": rate(prompt_eval_count, prompt_eval_duration_s),
        "load_duration_seconds": load_duration_s,
        "total_duration_seconds": total_duration_s,
        "response_chars": len("".join(content_parts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Ollama streaming TTFT and generation rate on saved AIME prompts."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--samples")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--system-prompt-file", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-gen-toks", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", default="results/perf")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    samples_path = Path(args.samples) if args.samples else latest_samples(run_dir, args.model)
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    samples = load_sample_messages(samples_path, args.limit)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = Path(args.output_dir) / sanitize_model_name(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"ollama_chat_perf_{timestamp}.jsonl"
    summary_path = output_dir / f"ollama_chat_perf_summary_{timestamp}.json"

    rows = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] doc_id={sample['doc_id']}", flush=True)
            row = stream_chat(
                base_url=args.base_url,
                model=args.model,
                system_prompt=system_prompt,
                sample=sample,
                max_gen_toks=args.max_gen_toks,
                temperature=args.temperature,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
            )
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                "  "
                f"ttft={fmt(row['time_to_first_token_seconds'])}s "
                f"client_tps={fmt(row['client_continuing_tokens_per_second'])} "
                f"ollama_tps={fmt(row['ollama_eval_tokens_per_second'])}",
                flush=True,
            )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "samples_path": str(samples_path),
        "sample_count": len(rows),
        "avg_time_to_first_token_seconds": mean(
            [r["time_to_first_token_seconds"] for r in rows]
        ),
        "avg_client_continuing_tokens_per_second": mean(
            [r["client_continuing_tokens_per_second"] for r in rows]
        ),
        "avg_ollama_eval_tokens_per_second": mean(
            [r["ollama_eval_tokens_per_second"] for r in rows]
        ),
        "avg_prompt_eval_tokens_per_second": mean(
            [r["prompt_eval_tokens_per_second"] for r in rows]
        ),
        "avg_load_duration_seconds": mean([r["load_duration_seconds"] for r in rows]),
        "avg_wall_seconds": mean([r["wall_seconds"] for r in rows]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Samples: {len(rows)}")
    print(f"Average TTFT: {fmt(summary['avg_time_to_first_token_seconds'])}s")
    print(
        "Average continuing tokens/sec, client-side: "
        f"{fmt(summary['avg_client_continuing_tokens_per_second'])}"
    )
    print(
        "Average eval tokens/sec, Ollama server-side: "
        f"{fmt(summary['avg_ollama_eval_tokens_per_second'])}"
    )
    print(f"JSONL: {jsonl_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
