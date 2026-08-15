#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_SLUG = re.compile(r"^[A-Za-z0-9._-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_name(value: str) -> str:
    return value.replace(":", "__").replace("/", "__")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def derive_manifest(
    source: dict[str, Any],
    *,
    source_slug: str,
    count: int,
    selected_instances: list[dict[str, Any]] | None = None,
    completed_model: str | None = None,
) -> dict[str, Any]:
    source_instances = source.get("instances") or []
    if count <= 0:
        raise ValueError("Count must be positive")
    if count >= len(source_instances):
        raise ValueError(
            f"Count must be smaller than the source batch size ({len(source_instances)})"
        )

    instances = (
        selected_instances
        if selected_instances is not None
        else source_instances[:count]
    )
    if len(instances) != count:
        raise ValueError(f"Expected {count} selected instances, got {len(instances)}")

    required_instances = source.get("required_instances") or []
    selected_ids = {str(row["instance_id"]) for row in instances}
    missing_required = [
        instance_id
        for instance_id in required_instances
        if instance_id not in selected_ids
    ]
    if missing_required:
        raise ValueError(
            "Derived selection omitted required instances: "
            + ", ".join(missing_required)
        )

    manifest = {
        key: value
        for key, value in source.items()
        if key not in {"created_at", "count", "derived_from", "instances"}
    }
    derived_from: dict[str, Any] = {
        "batch_slug": source_slug,
        "count": int(source.get("count") or len(source_instances)),
    }
    if completed_model:
        derived_from["filter"] = {
            "model": completed_model,
            "run_status": "completed",
        }
        manifest["selection_method"] = "source_order_filtered_by_completed_runs"
    manifest.update(
        {
            "created_at": utc_now(),
            "count": count,
            "derived_from": derived_from,
            "instances": instances,
        }
    )
    return manifest


def select_completed_instances(
    *,
    source_root: Path,
    source_instances: list[dict[str, Any]],
    model_slug: str,
    count: int,
) -> list[dict[str, Any]]:
    selected = []
    for row in source_instances:
        metadata_path = (
            source_root
            / "runs"
            / str(row["instance_id"])
            / model_slug
            / "run_metadata.json"
        )
        try:
            status = load_json(metadata_path).get("status")
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            continue
        if status == "completed":
            selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Source contains only {len(selected)} completed runs for this model; "
        f"cannot derive {count} tasks"
    )


def validate_existing_manifest(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    keys = [
        "dataset",
        "split",
        "count",
        "seed",
        "selection_method",
        "required_instances",
        "derived_from",
        "instances",
    ]
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            "Existing target manifest does not match the requested derivation: "
            + ", ".join(mismatches)
        )


def copy_model_runs(
    *,
    source_root: Path,
    target_root: Path,
    instances: list[dict[str, Any]],
    model_slug: str,
) -> dict[str, int]:
    counts = {"copied": 0, "existing": 0, "missing": 0}
    for row in instances:
        instance_id = str(row["instance_id"])
        source = source_root / "runs" / instance_id / model_slug
        target = target_root / "runs" / instance_id / model_slug
        if target.exists():
            counts["existing"] += 1
            continue
        if not source.is_dir():
            counts["missing"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, copy_function=shutil.copy2)
        counts["copied"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a smaller fixed SWE-bench batch and preserve prior runs."
    )
    parser.add_argument("--source", required=True, help="Existing source batch slug.")
    parser.add_argument("--target", required=True, help="New smaller batch slug.")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name whose selected task artifacts should be copied.",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help=(
            "Select the first N source-manifest tasks whose model run status is "
            "completed instead of selecting the first N tasks unconditionally."
        ),
    )
    parser.add_argument("--root", default="results/swebench-batches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SAFE_SLUG.fullmatch(args.source) or not SAFE_SLUG.fullmatch(args.target):
        print("Batch slugs may contain only letters, numbers, '.', '_' and '-'.", file=sys.stderr)
        return 2
    if args.source == args.target:
        print("Source and target batch slugs must differ.", file=sys.stderr)
        return 2

    root = Path(args.root)
    if root.is_absolute() or ".." in root.parts:
        print("Root must be a relative path inside the repository.", file=sys.stderr)
        return 2
    source_root = root / args.source
    target_root = root / args.target
    source_manifest_path = source_root / "manifest.json"
    target_manifest_path = target_root / "manifest.json"
    model_slug = sanitize_name(args.model)

    try:
        source_manifest = load_json(source_manifest_path)
        selected_instances = None
        if args.completed_only:
            selected_instances = select_completed_instances(
                source_root=source_root,
                source_instances=source_manifest.get("instances") or [],
                model_slug=model_slug,
                count=args.count,
            )
        expected_manifest = derive_manifest(
            source_manifest,
            source_slug=args.source,
            count=args.count,
            selected_instances=selected_instances,
            completed_model=args.model if args.completed_only else None,
        )
        if target_manifest_path.exists():
            validate_existing_manifest(
                load_json(target_manifest_path), expected_manifest
            )
            print(f"Using existing derived manifest: {target_manifest_path}")
        else:
            atomic_write_json(target_manifest_path, expected_manifest)
            print(f"Wrote derived {args.count}-task manifest: {target_manifest_path}")

        copy_counts = copy_model_runs(
            source_root=source_root,
            target_root=target_root,
            instances=expected_manifest["instances"],
            model_slug=model_slug,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Model: {args.model}")
    print(f"Copied task runs: {copy_counts['copied']}")
    print(f"Already present: {copy_counts['existing']}")
    print(f"No source run: {copy_counts['missing']}")
    if args.completed_only:
        print("Selection filter: source run status completed")
    print(
        "Run the target batch normally; completed imports will be skipped and "
        "failed or missing tasks will run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
