#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any

from swebench_arm64 import adapt_test_spec_for_arm64
from swebench_docker_platform import (
    docker_platform_for_arch,
    ensure_ubuntu_base_image,
)
from swebench_repo_setup import remove_stale_repo_branch_hint


LOGGER = logging.getLogger("run_swebench_agent_vllm_one")


def auto_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "x86_64"


def load_instance(dataset_name: str, split: str, instance_id: str) -> dict[str, Any]:
    from datasets import load_dataset

    rows = load_dataset(dataset_name, split=split)
    for row in rows:
        if row.get("instance_id") == instance_id:
            return dict(row)
    raise ValueError(f"{instance_id!r} was not found in {dataset_name}:{split}")


def ensure_instance_image(
    *,
    instance: dict[str, Any],
    arch: str,
    force_rebuild: bool,
) -> Any:
    import docker
    from swebench.harness.docker_build import build_instance_images
    from swebench.harness.test_spec.test_spec import make_test_spec

    client = docker.from_env()
    test_spec = make_test_spec(instance, namespace=None, arch=arch)
    changes = adapt_test_spec_for_arm64(test_spec)
    if changes:
        LOGGER.info(
            "Adapted %d Conda package pins for ARM64 in %s.",
            changes,
            test_spec.instance_id,
        )
    if remove_stale_repo_branch_hint(test_spec):
        LOGGER.info(
            "Removed stale repository branch hint for %s.", test_spec.instance_id
        )
    existing = client.images.list(name=test_spec.instance_image_key)
    if existing and not force_rebuild:
        LOGGER.info("Using existing instance image: %s", test_spec.instance_image_key)
        return test_spec

    ensure_ubuntu_base_image(arch, client)
    LOGGER.info("Building instance image: %s", test_spec.instance_image_key)
    successful, failed = build_instance_images(
        client=client,
        dataset=[test_spec],
        force_rebuild=force_rebuild,
        max_workers=1,
    )
    if failed:
        raise RuntimeError(f"Failed to build SWE-bench image: {failed}")
    client.images.get(test_spec.instance_image_key)
    return test_spec


def build_environment_config(config: dict[str, Any], image: str, arch: str) -> dict[str, Any]:
    env_config = dict(config.get("environment") or {})
    env_config["image"] = image
    env_config.setdefault("cwd", "/testbed")
    env_vars = dict(env_config.get("env") or {})
    env_vars.setdefault("PAGER", "cat")
    env_vars.setdefault("MANPAGER", "cat")
    env_vars.setdefault("BASH_ENV", "/root/.bashrc")
    env_config["env"] = env_vars

    run_args = list(env_config.get("run_args") or ["--rm"])
    if "--platform" not in run_args:
        run_args = ["--platform", docker_platform_for_arch(arch), *run_args]
    env_config["run_args"] = run_args
    return env_config


def write_prediction(
    *,
    output: Path,
    instance_id: str,
    model_name: str,
    patch: str,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preds.json"
    prediction = {
        instance_id: {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }
    }
    path.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one SWE-bench prediction with mini-swe-agent and vLLM."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--arch", choices=["auto", "x86_64", "arm64"], default="auto")
    parser.add_argument("--environment-class", choices=["docker"], default="docker")
    parser.add_argument("--force-rebuild-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    arch = auto_arch() if args.arch == "auto" else args.arch

    from minisweagent.agents import get_agent
    from minisweagent.config import get_config_from_spec
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge

    instance = load_instance(args.dataset, args.split, args.instance_id)
    test_spec = ensure_instance_image(
        instance=instance,
        arch=arch,
        force_rebuild=args.force_rebuild_images,
    )

    configs = [get_config_from_spec("swebench.yaml")]
    configs.extend(get_config_from_spec(str(path)) for path in args.config)
    config = recursive_merge(*configs)

    traj_dir = args.output / args.instance_id
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{args.instance_id}.traj.json"
    agent_config = dict(config.get("agent") or {})
    agent_config["output_path"] = str(traj_path)

    env_config = build_environment_config(
        config,
        image=test_spec.instance_image_key,
        arch=arch,
    )
    env = DockerEnvironment(**env_config)
    info: dict[str, Any] = {}
    run_error: Exception | None = None
    try:
        agent = get_agent(
            get_model(config=config.get("model") or {}),
            env,
            agent_config,
            default_type="default",
        )
        info = agent.run(instance["problem_statement"])
    except Exception as error:
        run_error = error
        info = {"exit_status": f"error: {type(error).__name__}", "submission": ""}
        LOGGER.exception("mini-swe-agent run failed")
    finally:
        try:
            env.cleanup()
        except Exception:
            LOGGER.exception("Docker environment cleanup failed")

    submission = ""
    if isinstance(info, dict):
        submission = info.get("submission") or ""
    write_prediction(
        output=args.output,
        instance_id=args.instance_id,
        model_name=args.served_model_name,
        patch=submission,
    )
    if run_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
