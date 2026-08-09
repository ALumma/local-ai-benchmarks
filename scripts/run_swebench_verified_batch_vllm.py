#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_INSTANCE = "django__django-11099"
SAFE_SLUG = re.compile(r"^[A-Za-z0-9._-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_name(value: str) -> str:
    return value.replace(":", "__").replace("/", "__")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def select_instances(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    required_instance_ids: list[str],
) -> list[dict[str, Any]]:
    by_id = {str(row["instance_id"]): row for row in rows}
    missing = [instance_id for instance_id in required_instance_ids if instance_id not in by_id]
    if missing:
        raise ValueError(f"Required instances not found: {', '.join(missing)}")
    if count < len(required_instance_ids):
        raise ValueError("Count is smaller than the number of required instances")
    if count > len(by_id):
        raise ValueError(f"Requested {count} instances, but dataset contains {len(by_id)}")

    required_set = set(required_instance_ids)
    ranked_ids = sorted(
        (instance_id for instance_id in by_id if instance_id not in required_set),
        key=lambda instance_id: hashlib.sha256(
            f"{seed}\0{instance_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected_ids = [*required_instance_ids, *ranked_ids[: count - len(required_instance_ids)]]
    return [
        {
            "instance_id": instance_id,
            "repo": by_id[instance_id].get("repo"),
            "version": by_id[instance_id].get("version"),
        }
        for instance_id in selected_ids
    ]


def create_manifest(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    from datasets import load_dataset

    print(f"Loading {args.dataset}:{args.split} to create the batch manifest.", flush=True)
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    instances = select_instances(
        rows,
        count=args.count,
        seed=args.seed,
        required_instance_ids=args.required_instance,
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": args.dataset,
        "split": args.split,
        "count": args.count,
        "seed": args.seed,
        "selection_method": "required_then_sha256_rank",
        "required_instances": args.required_instance,
        "instances": instances,
    }
    atomic_write_json(path, manifest)
    print(f"Wrote fixed {args.count}-task manifest: {path}", flush=True)
    return manifest


def validate_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "dataset": args.dataset,
        "split": args.split,
        "count": args.count,
        "seed": args.seed,
        "required_instances": args.required_instance,
    }
    mismatches = [
        f"{key}: manifest={manifest.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    instance_ids = [row.get("instance_id") for row in manifest.get("instances", [])]
    if len(instance_ids) != args.count or len(set(instance_ids)) != args.count:
        mismatches.append("manifest instance list is missing entries or contains duplicates")
    if mismatches:
        raise ValueError("Existing manifest does not match this batch:\n" + "\n".join(mismatches))


def task_metadata_path(
    batch_root: Path, instance_id: str, model_slug: str
) -> Path:
    return batch_root / "runs" / instance_id / model_slug / "run_metadata.json"


def task_state(path: Path) -> str:
    if not path.exists():
        return "pending"
    try:
        status = load_json(path).get("status")
    except (json.JSONDecodeError, OSError, ValueError):
        return "failed"
    if status == "completed":
        return "completed"
    if status == "running":
        return "running"
    if status == "failed":
        return "failed"
    return "pending"


def state_counts(
    batch_root: Path, instances: list[dict[str, Any]], model_slug: str
) -> dict[str, int]:
    counts = {"completed": 0, "failed": 0, "running": 0, "pending": 0}
    for row in instances:
        state = task_state(task_metadata_path(batch_root, row["instance_id"], model_slug))
        counts[state] += 1
    return counts


def write_batch_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    batch_root: Path,
    model_slug: str,
    status: str,
    started_at: str,
    attempt: int,
    current_instance: str | None,
    current_index: int | None,
    exit_code: int | None,
) -> None:
    instances = load_json(manifest_path)["instances"]
    payload = {
        "schema_version": 1,
        "status": status,
        "exit_code": exit_code,
        "started_at": started_at,
        "updated_at": utc_now(),
        "finished_at": utc_now() if status not in {"running", "starting"} else None,
        "attempt": attempt,
        "batch_slug": args.batch_slug,
        "dataset": args.dataset,
        "split": args.split,
        "count": args.count,
        "seed": args.seed,
        "model": args.model_name,
        "model_slug": model_slug,
        "model_profile": args.model_profile or None,
        "current_instance": current_instance,
        "current_index": current_index,
        "manifest_path": str(manifest_path),
        "batch_root": str(batch_root),
        "counts": state_counts(batch_root, instances, model_slug),
    }
    atomic_write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed, resumable SWE-bench Verified batch through vLLM."
    )
    parser.add_argument(
        "--batch-slug",
        default=os.environ.get(
            "SWEBENCH_BATCH_SLUG", "swebench-verified-50-qwen36-nvfp4-v1"
        ),
    )
    parser.add_argument(
        "--count", type=int, default=int(os.environ.get("SWEBENCH_BATCH_COUNT", "50"))
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("SWEBENCH_BATCH_SEED", "20260809"))
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get(
            "SWEBENCH_DATASET", "princeton-nlp/SWE-bench_Verified"
        ),
    )
    parser.add_argument("--split", default=os.environ.get("SWEBENCH_SPLIT", "test"))
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "SWEBENCH_MODEL_NAME", "bench-qwen36-35b-a3b-nvfp4-mtp"
        ),
    )
    parser.add_argument(
        "--model-profile", default=os.environ.get("SWEBENCH_MODEL_PROFILE", "")
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("SWEBENCH_BATCH_OUTPUT_ROOT", "results/swebench-batches"),
    )
    parser.add_argument(
        "--required-instance",
        action="append",
        default=[DEFAULT_REQUIRED_INSTANCE],
        help="Always include this instance before selecting the remaining seeded subset.",
    )
    parser.add_argument(
        "--one-runner",
        type=Path,
        default=Path("scripts/run_swebench_verified_one_vllm.sh"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SAFE_SLUG.fullmatch(args.batch_slug):
        raise SystemExit("Batch slug may contain only letters, numbers, '.', '_' and '-'.")
    if args.count <= 0:
        raise SystemExit("Batch count must be positive.")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if output_root.is_absolute() or ".." in output_root.parts:
        raise SystemExit("Batch output root must be a relative path inside the repository.")
    batch_root = repo_root / output_root / args.batch_slug
    manifest_path = batch_root / "manifest.json"
    model_slug = sanitize_name(args.model_name)
    model_metadata_path = batch_root / "models" / model_slug / "batch_metadata.json"
    runner = args.one_runner if args.one_runner.is_absolute() else repo_root / args.one_runner
    if not runner.is_file():
        raise SystemExit(f"One-task runner not found: {runner}")

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_manifest(manifest, args)
        print(f"Using existing fixed manifest: {manifest_path}", flush=True)
    else:
        manifest = create_manifest(args, manifest_path)

    existing_metadata = load_json(model_metadata_path) if model_metadata_path.exists() else {}
    started_at = existing_metadata.get("started_at") or utc_now()
    attempt = int(existing_metadata.get("attempt") or 0) + 1
    instances = manifest["instances"]
    write_batch_metadata(
        model_metadata_path,
        args=args,
        manifest_path=manifest_path,
        batch_root=batch_root,
        model_slug=model_slug,
        status="starting",
        started_at=started_at,
        attempt=attempt,
        current_instance=None,
        current_index=None,
        exit_code=None,
    )

    print("==> SWE-bench Verified batch", flush=True)
    print(f"    batch: {args.batch_slug}", flush=True)
    print(f"    model: {args.model_name}", flush=True)
    print(f"    profile: {args.model_profile or '-'}", flush=True)
    print(f"    tasks: {len(instances)}", flush=True)
    print(f"    output: {batch_root}", flush=True)

    for index, row in enumerate(instances, start=1):
        instance_id = row["instance_id"]
        metadata_path = task_metadata_path(batch_root, instance_id, model_slug)
        if task_state(metadata_path) == "completed":
            print(
                f"[{index}/{args.count}] {instance_id}: already completed, skipping.",
                flush=True,
            )
            continue

        write_batch_metadata(
            model_metadata_path,
            args=args,
            manifest_path=manifest_path,
            batch_root=batch_root,
            model_slug=model_slug,
            status="running",
            started_at=started_at,
            attempt=attempt,
            current_instance=instance_id,
            current_index=index,
            exit_code=None,
        )
        print(f"\n[{index}/{args.count}] Starting {instance_id}", flush=True)

        task_env = os.environ.copy()
        task_env.update(
            {
                "SWEBENCH_DATASET": args.dataset,
                "SWEBENCH_SPLIT": args.split,
                "SWEBENCH_INSTANCE_ID": instance_id,
                "SWEBENCH_MODEL_NAME": args.model_name,
                "SWEBENCH_MODEL_PROFILE": args.model_profile,
                "SWEBENCH_OUTPUT_ROOT": str(output_root / args.batch_slug / "runs"),
                "SWEBENCH_RUN_SLUG": instance_id,
            }
        )
        result = subprocess.run([str(runner)], cwd=repo_root, env=task_env, check=False)
        if result.returncode:
            print(
                f"[{index}/{args.count}] {instance_id}: failed with exit code "
                f"{result.returncode}; continuing.",
                flush=True,
            )
            if not metadata_path.exists():
                print(
                    "The task failed before run metadata was created. Stopping the "
                    "batch because the vLLM endpoint or shared dependencies may be "
                    "unavailable; rerun the same command to resume.",
                    flush=True,
                )
                break
        else:
            print(f"[{index}/{args.count}] {instance_id}: completed.", flush=True)

    counts = state_counts(batch_root, instances, model_slug)
    has_errors = counts["failed"] > 0 or counts["pending"] > 0 or counts["running"] > 0
    final_status = "completed_with_errors" if has_errors else "completed"
    exit_code = 1 if has_errors else 0
    write_batch_metadata(
        model_metadata_path,
        args=args,
        manifest_path=manifest_path,
        batch_root=batch_root,
        model_slug=model_slug,
        status=final_status,
        started_at=started_at,
        attempt=attempt,
        current_instance=None,
        current_index=None,
        exit_code=exit_code,
    )

    report_command = [
        sys.executable,
        str(repo_root / "scripts/report_swebench_batch.py"),
        "--batch",
        args.batch_slug,
        "--root",
        str(output_root),
        "--write",
    ]
    subprocess.run(report_command, cwd=repo_root, check=False)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
