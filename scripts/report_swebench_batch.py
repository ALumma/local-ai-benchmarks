#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from report_swebench_results import collect_record, fmt_duration


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def median_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.median(values) if values else None


def table_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    return [
        max(len(value) for value in [headers[index], *[row[index] for row in rows]])
        for index in range(len(headers))
    ]


def collect_batch(batch_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = batch_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    selected_ids = [row["instance_id"] for row in manifest.get("instances", [])]
    selected_set = set(selected_ids)

    records = []
    for metadata_path in sorted((batch_root / "runs").glob("*/*/run_metadata.json")):
        try:
            record = collect_record(metadata_path)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if record.get("instance_id") in selected_set:
            record["selection_index"] = selected_ids.index(record["instance_id"]) + 1
            records.append(record)
    return manifest, records


def aggregate_models(
    batch_root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    total = int(manifest.get("count") or len(manifest.get("instances", [])))
    by_model: dict[str, list[dict[str, Any]]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}

    for path in sorted((batch_root / "models").glob("*/batch_metadata.json")):
        try:
            metadata = load_json(path)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        model = str(metadata.get("model") or path.parent.name)
        model_metadata[model] = metadata
        by_model.setdefault(model, [])
    for record in records:
        by_model.setdefault(str(record.get("model") or ""), []).append(record)

    summaries = []
    for model, model_records in by_model.items():
        metadata = model_metadata.get(model, {})
        completed = [record for record in model_records if record.get("status") == "completed"]
        errors = [record for record in model_records if record.get("status") == "failed"]
        resolved = [record for record in completed if record.get("resolved") is True]
        unresolved = [record for record in completed if record.get("resolved") is False]
        empty = [record for record in completed if not record.get("patch_bytes")]
        pending = max(0, total - len(completed) - len(errors))
        summaries.append(
            {
                "model": model,
                "profile": metadata.get("model_profile")
                or next((record.get("model_profile") for record in model_records), ""),
                "status": metadata.get("status") or "",
                "selected": total,
                "completed": len(completed),
                "resolved": len(resolved),
                "unresolved": len(unresolved),
                "errors": len(errors),
                "pending": pending,
                "empty_patches": len(empty),
                "accuracy": len(resolved) / total if total else 0.0,
                "median_agent_seconds": median_value(completed, "agent_seconds"),
                "median_eval_seconds": median_value(completed, "evaluation_seconds"),
                "current_instance": metadata.get("current_instance") or "",
                "attempt": metadata.get("attempt"),
            }
        )
    summaries.sort(key=lambda row: row["model"])
    return summaries


def print_summary(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Profile",
        "Status",
        "Done",
        "Resolved",
        "Accuracy",
        "Errors",
        "Pending",
        "Empty",
        "Median agent",
        "Median eval",
        "Current",
    ]
    rows = [
        [
            str(row["model"]),
            str(row["profile"] or ""),
            str(row["status"]),
            f"{row['completed']}/{row['selected']}",
            f"{row['resolved']}/{row['selected']}",
            f"{row['accuracy']:.3f}",
            str(row["errors"]),
            str(row["pending"]),
            str(row["empty_patches"]),
            fmt_duration(row["median_agent_seconds"]),
            fmt_duration(row["median_eval_seconds"]),
            str(row["current_instance"]),
        ]
        for row in summaries
    ]
    if not rows:
        print("No models have started this batch.")
        return
    widths = table_widths(headers, rows)
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_instances(records: list[dict[str, Any]]) -> None:
    headers = [
        "#",
        "Model",
        "Instance",
        "Status",
        "Resolved",
        "Patch bytes",
        "Agent",
        "Eval",
    ]
    rows = []
    records.sort(
        key=lambda row: (row.get("model") or "", row.get("selection_index") or 0)
    )
    for record in records:
        if record.get("resolved") is True:
            resolved = "yes"
        elif record.get("resolved") is False:
            resolved = "no"
        else:
            resolved = ""
        rows.append(
            [
                str(record.get("selection_index") or ""),
                str(record.get("model") or ""),
                str(record.get("instance_id") or ""),
                str(record.get("status") or ""),
                resolved,
                str(record.get("patch_bytes") or 0),
                fmt_duration(record.get("agent_seconds")),
                fmt_duration(record.get("evaluation_seconds")),
            ]
        )
    if not rows:
        print("No task results found.")
        return
    widths = table_widths(headers, rows)
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0]) if summaries else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report a fixed SWE-bench batch.")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--root", default="results/swebench-batches")
    parser.add_argument("--instances", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_root = Path(args.root) / args.batch
    try:
        manifest, records = collect_batch(batch_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    summaries = aggregate_models(batch_root, manifest, records)

    if args.json_output:
        print(
            json.dumps(
                {"manifest": manifest, "models": summaries, "records": records},
                indent=2,
            )
        )
    elif args.instances:
        print_instances(records)
    else:
        print_summary(summaries)

    if args.write:
        summary_path = batch_root / "batch_summary.json"
        csv_path = batch_root / "batch_summary.csv"
        summary_path.write_text(
            json.dumps({"manifest": manifest, "models": summaries}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(csv_path, summaries)
        print(f"\nSummary JSON: {summary_path}")
        print(f"Summary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
