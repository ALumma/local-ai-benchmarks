#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = [
    "bench-gemma31__q4",
    "bench-gemma26__q4",
    "bench-qwen-next__q4",
    "bench-devstral__q4",
    "bench-glm47__q4",
]


def latest_result(model_dir: Path) -> Path | None:
    results = sorted(model_dir.glob("results_*.json"))
    return results[-1] if results else None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="results/aime/aime24-system-prompt-v1",
        help="Directory containing per-model lm-eval result folders.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = []
    for model in MODELS:
        result_path = latest_result(run_dir / model)
        if result_path is None:
            rows.append([model, "", "", "", "", "missing"])
            continue

        data = load_json(result_path)
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
                "manual review not recorded",
            ]
        )

    headers = [
        "Model",
        "Correct",
        "Exact match",
        "Std err",
        "Avg sec/request",
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
