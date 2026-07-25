#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


KNOWN_MODELS = [
    ("bench-qwen-next__q4", "bench-qwen-next:q4"),
    ("bench-gemma31__q4", "bench-gemma31:q4"),
    ("bench-gemma26__q4", "bench-gemma26:q4"),
    ("bench-devstral__q4", "bench-devstral:q4"),
    ("bench-glm47__q4", "bench-glm47:q4"),
    (
        "bench-qwen36-35b-a3b-nvfp4-mtp",
        "bench-qwen36-35b-a3b-nvfp4-mtp",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def perf_gen_tps(perf: dict[str, Any]) -> Any:
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


def format_audit(samples_path: Path | None) -> dict[str, Any]:
    if samples_path is None:
        return {
            "text": "no samples",
            "possible": None,
            "boxed": None,
            "unboxed": None,
            "wrong": None,
        }

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

    possible = boxed_target_but_wrong + unboxed_final_target
    if wrong == 0:
        text = "0"
    elif possible == 0:
        text = "0 possible"
    else:
        details = []
        if boxed_target_but_wrong:
            details.append(f"{boxed_target_but_wrong} boxed")
        if unboxed_final_target:
            details.append(f"{unboxed_final_target} unboxed")
        text = f"{possible} possible ({', '.join(details)})"

    return {
        "text": text,
        "possible": possible,
        "boxed": boxed_target_but_wrong,
        "unboxed": unboxed_final_target,
        "wrong": wrong,
    }


def load_integrated_perf(model_dir: Path) -> tuple[dict[str, Any], str]:
    path = model_dir / "perf_integrated_summary.json"
    if not path.exists():
        return {}, ""
    return load_json(path), str(path)


def load_replay_perf(perf_root: Path, model_dir_name: str) -> tuple[dict[str, Any], str]:
    summary = latest_file(perf_root / model_dir_name, "ollama_chat_perf_summary_*.json")
    if summary is None:
        return {}, ""
    data = load_json(summary)
    return data, str(summary)


def iter_result_records(
    *,
    aime_root: Path,
    perf_root: Path,
    task: str,
    include_replay_perf: bool,
) -> list[dict[str, Any]]:
    records = []
    for result_path in sorted(aime_root.glob("*/*/results_*.json")):
        model_dir = result_path.parent
        run_dir = model_dir.parent
        try:
            data = load_json(result_path)
        except json.JSONDecodeError:
            continue

        task_name = task if task in data.get("results", {}) else None
        if task_name is None:
            if len(data.get("results", {})) != 1:
                continue
            task_name = next(iter(data["results"]))

        result = data["results"][task_name]
        sample_len = int(
            result.get("sample_len")
            or data.get("n-samples", {}).get(task_name, {}).get("effective")
            or 0
        )
        accuracy = float(result.get("exact_match,none") or 0)
        correct = round(accuracy * sample_len)
        total_seconds = float(data.get("total_evaluation_time_seconds") or 0)
        avg_request_seconds = total_seconds / sample_len if sample_len else None

        perf, perf_path = load_integrated_perf(model_dir)
        perf_source = "integrated" if perf else ""
        if not perf and include_replay_perf:
            perf, perf_path = load_replay_perf(perf_root, model_dir.name)
            perf_source = "replay" if perf else ""

        samples_path = latest_file(model_dir, f"samples_{task_name}_*.jsonl")
        audit = format_audit(samples_path)

        records.append(
            {
                "model": data.get("model_name") or model_dir.name,
                "model_dir": model_dir.name,
                "run": run_dir.name,
                "task": task_name,
                "correct": correct,
                "samples": sample_len,
                "correct_display": f"{correct}/{sample_len}" if sample_len else "",
                "exact_match": accuracy,
                "stderr": result.get("exact_match_stderr,none"),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60 if total_seconds else None,
                "avg_request_seconds": avg_request_seconds,
                "avg_ttft_seconds": perf.get("avg_time_to_first_token_seconds"),
                "avg_client_continuing_tokens_per_second": perf.get(
                    "avg_client_continuing_tokens_per_second"
                ),
                "avg_ollama_eval_tokens_per_second": perf.get(
                    "avg_ollama_eval_tokens_per_second"
                ),
                "avg_openai_completion_tokens_per_second": perf.get(
                    "avg_openai_completion_tokens_per_second"
                ),
                "avg_generation_tokens_per_second": perf_gen_tps(perf),
                "avg_prompt_eval_tokens_per_second": perf.get(
                    "avg_prompt_eval_tokens_per_second"
                ),
                "avg_load_seconds": perf.get("avg_load_duration_seconds"),
                "avg_wall_seconds": perf.get("avg_wall_seconds"),
                "perf_samples": perf.get("sample_count"),
                "perf_source": perf_source,
                "parsing_failures": audit["text"],
                "parsing_possible": audit["possible"],
                "parsing_boxed": audit["boxed"],
                "parsing_unboxed": audit["unboxed"],
                "wrong": audit["wrong"],
                "result_path": str(result_path),
                "samples_path": str(samples_path) if samples_path else "",
                "perf_path": perf_path,
                "model_dir_path": str(model_dir),
                "mtime": result_path.stat().st_mtime,
            }
        )
    return records


def latest_records_by_model(
    records: list[dict[str, Any]],
    *,
    min_samples: int,
    include_extra_models: bool,
) -> list[dict[str, Any]]:
    known_dirs = [model_dir for model_dir, _ in KNOWN_MODELS]
    by_model = {model_dir: [] for model_dir in known_dirs}
    extras: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        target = by_model if record["model_dir"] in by_model else extras
        target.setdefault(record["model_dir"], []).append(record)

    rows = []
    for model_dir, model_name in KNOWN_MODELS:
        candidates = by_model.get(model_dir, [])
        eligible = [r for r in candidates if r["samples"] >= min_samples]
        if eligible:
            rows.append(max(eligible, key=lambda r: r["mtime"]))
        else:
            rows.append(
                {
                    "model": model_name,
                    "model_dir": model_dir,
                    "run": "missing",
                    "correct_display": "",
                    "exact_match": None,
                    "stderr": None,
                    "total_minutes": None,
                    "avg_request_seconds": None,
                    "avg_ttft_seconds": None,
                    "avg_ollama_eval_tokens_per_second": None,
                    "avg_openai_completion_tokens_per_second": None,
                    "avg_generation_tokens_per_second": None,
                    "avg_prompt_eval_tokens_per_second": None,
                    "avg_load_seconds": None,
                    "avg_wall_seconds": None,
                    "perf_samples": None,
                    "perf_source": "",
                    "parsing_failures": "missing",
                    "result_path": "",
                    "samples_path": "",
                    "perf_path": "",
                    "model_dir_path": "",
                }
            )

    if include_extra_models:
        for model_dir, candidates in sorted(extras.items()):
            eligible = [r for r in candidates if r["samples"] >= min_samples]
            if eligible:
                rows.append(max(eligible, key=lambda r: r["mtime"]))

    return rows


def sort_records(records: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "model":
        return records
    if sort_key == "run":
        return sorted(records, key=lambda r: (r.get("run") or "", r.get("model") or ""))
    if sort_key == "speed":
        return sorted(
            records,
            key=lambda r: r.get("avg_generation_tokens_per_second") or -1,
            reverse=True,
        )
    return sorted(records, key=lambda r: r.get("exact_match") or -1, reverse=True)


def table_rows(records: list[dict[str, Any]], show_paths: bool) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Model",
        "Run",
        "Correct",
        "Acc",
        "Std err",
        "Total min",
        "Avg req s",
        "TTFT s",
        "Gen tok/s",
        "Prompt tok/s",
        "Load s",
        "Perf n",
        "Perf src",
        "Parsing",
    ]
    if show_paths:
        headers += ["Result path", "Perf path"]

    rows = []
    for record in records:
        row = [
            str(record.get("model") or ""),
            str(record.get("run") or ""),
            str(record.get("correct_display") or ""),
            fmt_float(record.get("exact_match"), 4),
            fmt_float(record.get("stderr"), 4),
            fmt_float(record.get("total_minutes"), 1),
            fmt_float(record.get("avg_request_seconds"), 2),
            fmt_float(record.get("avg_ttft_seconds"), 2),
            fmt_float(record.get("avg_generation_tokens_per_second"), 2),
            fmt_float(record.get("avg_prompt_eval_tokens_per_second"), 2),
            fmt_float(record.get("avg_load_seconds"), 2),
            str(record.get("perf_samples") or ""),
            str(record.get("perf_source") or ""),
            str(record.get("parsing_failures") or ""),
        ]
        if show_paths:
            row += [str(record.get("result_path") or ""), str(record.get("perf_path") or "")]
        rows.append(row)
    return headers, rows


def print_table(records: list[dict[str, Any]], show_paths: bool) -> None:
    headers, rows = table_rows(records, show_paths)
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    headers, rows = table_rows(records, show_paths=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display AIME 24 accuracy and performance for all benchmarked models."
    )
    parser.add_argument("--aime-root", default="results/aime")
    parser.add_argument("--perf-root", default="results/perf")
    parser.add_argument("--task", default="aime24")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Default filters out smoke tests. Use 1 to include smoke/partial runs.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Show every matching run instead of the latest full run per model.",
    )
    parser.add_argument(
        "--include-extra-models",
        action="store_true",
        help="Include result directories outside the five known benchmark aliases.",
    )
    parser.add_argument(
        "--include-replay-perf",
        action="store_true",
        help="Use results/perf replay summaries when integrated perf is missing.",
    )
    parser.add_argument(
        "--sort",
        choices=["accuracy", "speed", "model", "run"],
        default="accuracy",
    )
    parser.add_argument("--paths", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--csv", type=Path, help="Also write the displayed table to CSV.")
    args = parser.parse_args()

    aime_root = Path(args.aime_root)
    if not aime_root.exists():
        print(f"No AIME result root found: {aime_root}", file=sys.stderr)
        return 1

    records = iter_result_records(
        aime_root=aime_root,
        perf_root=Path(args.perf_root),
        task=args.task,
        include_replay_perf=args.include_replay_perf,
    )
    records = [r for r in records if args.all_runs or r["samples"] >= args.min_samples]

    if args.all_runs:
        rows = records
    else:
        rows = latest_records_by_model(
            records,
            min_samples=args.min_samples,
            include_extra_models=args.include_extra_models,
        )
    rows = sort_records(rows, args.sort)

    if args.json_output:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_table(rows, show_paths=args.paths)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
