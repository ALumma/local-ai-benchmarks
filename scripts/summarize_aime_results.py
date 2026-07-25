#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODELS = [
    "bench-gemma31__q4",
    "bench-gemma26__q4",
    "bench-qwen-next__q4",
    "bench-devstral__q4",
    "bench-glm47__q4",
    "bench-qwen36-35b-a3b-nvfp4-mtp",
]


def latest_result(model_dir: Path) -> Path | None:
    results = sorted(model_dir.glob("results_*.json"))
    return results[-1] if results else None


def latest_samples(model_dir: Path) -> Path | None:
    samples = sorted(model_dir.glob("samples_*.jsonl"))
    return samples[-1] if samples else None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_perf_summary(model_dir: Path) -> dict:
    path = model_dir / "perf_integrated_summary.json"
    if not path.exists():
        return {}
    return load_json(path)


def perf_gen_tps(perf: dict) -> object:
    return (
        perf.get("avg_ollama_eval_tokens_per_second")
        or perf.get("avg_openai_completion_tokens_per_second")
        or perf.get("avg_client_continuing_tokens_per_second")
    )


def normalize_int(text: str | int | None) -> str | None:
    if text is None:
        return None
    match = re.search(r"-?\d+", str(text).strip())
    if not match:
        return None
    return str(int(match.group(0)))


def boxed_values(text: str) -> list[str]:
    return re.findall(r"\\boxed\{([^{}]+)\}", text)


def likely_format_false_negatives(samples_path: Path | None) -> str:
    if samples_path is None:
        return "no samples"

    boxed_target_but_wrong = 0
    unboxed_final_target = 0
    wrong = 0

    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if int(sample.get("exact_match", 0)) == 1:
                continue

            wrong += 1
            target = normalize_int(sample.get("target"))
            response = (sample.get("filtered_resps") or [""])[0]
            boxes = boxed_values(response)

            if boxes and normalize_int(boxes[-1]) == target:
                boxed_target_but_wrong += 1
                continue

            final_ints = re.findall(r"-?\d+", response[-300:])
            if final_ints and normalize_int(final_ints[-1]) == target:
                unboxed_final_target += 1

    if wrong == 0:
        return "0"

    possible = boxed_target_but_wrong + unboxed_final_target
    if possible == 0:
        return "0 possible"

    details = []
    if boxed_target_but_wrong:
        details.append(f"{boxed_target_but_wrong} boxed")
    if unboxed_final_target:
        details.append(f"{unboxed_final_target} unboxed")
    return f"{possible} possible ({', '.join(details)})"


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="results/aime/aime24-answer-format-v2",
        help="Directory containing per-model lm-eval result folders.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = []
    for model in MODELS:
        result_path = latest_result(run_dir / model)
        if result_path is None:
            perf = load_perf_summary(run_dir / model)
            rows.append(
                [
                    model,
                    "",
                    "",
                    "",
                    "",
                    fmt_float(perf.get("avg_time_to_first_token_seconds"), 2),
                    fmt_float(perf_gen_tps(perf), 2),
                    fmt_float(perf.get("avg_prompt_eval_tokens_per_second"), 2),
                    "missing",
                ]
            )
            continue

        data = load_json(result_path)
        perf = load_perf_summary(run_dir / model)
        task_name = next(iter(data["results"]))
        result = data["results"][task_name]
        sample_len = int(result.get("sample_len") or data["n-samples"][task_name]["effective"])
        accuracy = float(result["exact_match,none"])
        correct = round(accuracy * sample_len)
        total_seconds = float(data.get("total_evaluation_time_seconds") or 0)
        avg_seconds = total_seconds / sample_len if sample_len else 0
        rows.append(
            [
                data.get("model_name", model),
                f"{correct}/{sample_len}",
                fmt_float(accuracy),
                fmt_float(result.get("exact_match_stderr,none")),
                fmt_float(avg_seconds, 2),
                fmt_float(perf.get("avg_time_to_first_token_seconds"), 2),
                fmt_float(perf_gen_tps(perf), 2),
                fmt_float(perf.get("avg_prompt_eval_tokens_per_second"), 2),
                likely_format_false_negatives(latest_samples(run_dir / model)),
            ]
        )

    headers = [
        "Model",
        "Correct",
        "Exact match",
        "Std err",
        "Avg sec/request",
        "Avg TTFT",
        "Avg gen tok/s",
        "Avg prompt tok/s",
        "Parsing failures",
    ]
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
