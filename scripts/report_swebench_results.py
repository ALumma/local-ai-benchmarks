#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_duration(value: Any) -> str:
    if value in (None, ""):
        return ""
    seconds = float(value)
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_prediction(preds_path: Path, instance_id: str) -> dict[str, Any]:
    if not preds_path.exists():
        return {}
    payload = load_json(preds_path)
    if isinstance(payload, dict):
        row = payload.get(instance_id)
        if isinstance(row, dict):
            return row
        return {}
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("instance_id") == instance_id:
                return row
    return {}


def load_trajectory(model_dir: Path, instance_id: str) -> dict[str, Any]:
    path = latest_file(model_dir / "agent" / instance_id, f"{instance_id}.traj.json")
    if path is None:
        paths = sorted((model_dir / "agent").glob(f"**/{instance_id}.traj.json"))
        path = paths[-1] if paths else None
    if path is None:
        return {"path": ""}
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return {"path": str(path)}
    info = payload.get("info", {}) if isinstance(payload, dict) else {}
    return {
        "path": str(path),
        "exit_status": info.get("exit_status"),
        "submission": info.get("submission"),
    }


def load_run_report(eval_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in eval_dir.glob("*.json"):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == 2:
            candidates.append((path, payload))
    if not candidates:
        return {"path": ""}
    path, payload = max(candidates, key=lambda item: item[0].stat().st_mtime)
    return {"path": str(path), **payload}


def load_instance_report(eval_dir: Path, instance_id: str) -> dict[str, Any]:
    candidates = []
    for path in eval_dir.rglob("report.json"):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and instance_id in payload:
            candidates.append((path, payload[instance_id]))
    if not candidates:
        return {"path": ""}
    path, payload = max(candidates, key=lambda item: item[0].stat().st_mtime)
    return {"path": str(path), **payload}


def collect_record(metadata_path: Path) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    model_dir = metadata_path.parent
    instance_id = metadata.get("instance_id") or ""
    preds_path = model_dir / "agent" / "preds.json"
    pred = load_prediction(preds_path, instance_id)
    patch = pred.get("model_patch") or ""
    trajectory = load_trajectory(model_dir, instance_id)
    run_report = load_run_report(model_dir / "evaluation")
    instance_report = load_instance_report(model_dir / "evaluation", instance_id)
    resolved_ids = set(run_report.get("resolved_ids") or [])
    completed_ids = set(run_report.get("completed_ids") or [])

    resolved = instance_report.get("resolved")
    if resolved is None and instance_id:
        resolved = instance_id in resolved_ids if run_report.get("path") else None

    completed = instance_id in completed_ids if run_report.get("path") else None
    if completed is None and instance_report.get("path"):
        completed = True

    return {
        "run": metadata.get("run_slug") or model_dir.parent.name,
        "model": metadata.get("model") or model_dir.name,
        "model_profile": metadata.get("model_profile") or "",
        "instance_id": instance_id,
        "status": metadata.get("status") or "",
        "exit_code": metadata.get("exit_code"),
        "agent_exit_status": trajectory.get("exit_status") or "",
        "patch_bytes": len(patch.encode("utf-8")) if patch else 0,
        "eval_completed": completed,
        "resolved": resolved,
        "agent_seconds": (metadata.get("durations") or {}).get("agent_seconds"),
        "evaluation_seconds": (metadata.get("durations") or {}).get(
            "evaluation_seconds"
        ),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "model_dir": str(model_dir),
        "metadata_path": str(metadata_path),
        "predictions_path": str(preds_path) if preds_path.exists() else "",
        "trajectory_path": trajectory.get("path") or "",
        "run_report_path": run_report.get("path") or "",
        "instance_report_path": instance_report.get("path") or "",
    }


def iter_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for metadata_path in sorted(root.glob("*/*/run_metadata.json")):
        try:
            records.append(collect_record(metadata_path))
        except json.JSONDecodeError:
            continue
    return records


def table_rows(records: list[dict[str, Any]], show_paths: bool) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Model",
        "Profile",
        "Instance",
        "Status",
        "Agent",
        "Patch bytes",
        "Eval",
        "Resolved",
        "Agent time",
        "Eval time",
        "Total",
    ]
    if show_paths:
        headers += ["Model dir", "Run report", "Instance report", "Trajectory"]

    rows = []
    for record in records:
        row = [
            str(record.get("model") or ""),
            str(record.get("model_profile") or ""),
            str(record.get("instance_id") or ""),
            str(record.get("status") or ""),
            str(record.get("agent_exit_status") or ""),
            str(record.get("patch_bytes") or 0),
            fmt_bool(record.get("eval_completed")),
            fmt_bool(record.get("resolved")),
            fmt_duration(record.get("agent_seconds")),
            fmt_duration(record.get("evaluation_seconds")),
            fmt_duration(record.get("elapsed_seconds")),
        ]
        if show_paths:
            row += [
                str(record.get("model_dir") or ""),
                str(record.get("run_report_path") or ""),
                str(record.get("instance_report_path") or ""),
                str(record.get("trajectory_path") or ""),
            ]
        rows.append(row)
    return headers, rows


def print_table(records: list[dict[str, Any]], show_paths: bool) -> None:
    headers, rows = table_rows(records, show_paths)
    if not rows:
        print("No SWE-bench result metadata found.")
        return
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
        description="Display local SWE-bench Verified agent/evaluation results."
    )
    parser.add_argument("--root", default="results/swebench")
    parser.add_argument("--run", help="Only show one run slug.")
    parser.add_argument("--model", help="Only show one served model name.")
    parser.add_argument("--paths", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--csv", type=Path, help="Also write the displayed table to CSV.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"No SWE-bench result root found: {root}", file=sys.stderr)
        return 1

    records = iter_records(root)
    if args.run:
        records = [record for record in records if record.get("run") == args.run]
    if args.model:
        records = [record for record in records if record.get("model") == args.model]
    records.sort(key=lambda r: (r.get("run") or "", r.get("model") or ""))

    if args.json_output:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        print_table(records, show_paths=args.paths)

    if args.csv:
        write_csv(records, args.csv)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
