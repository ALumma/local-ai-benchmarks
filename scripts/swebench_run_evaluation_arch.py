#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

from swebench_arm64 import adapt_test_spec_for_arm64
from swebench_docker_platform import ensure_ubuntu_base_image


def normalize_namespace(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() == "none":
        return None
    return value


def auto_arch() -> str:
    import platform

    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "x86_64"


def patch_swebench_arch(arch: str) -> None:
    import swebench.harness.docker_build as docker_build
    import swebench.harness.reporting as reporting
    import swebench.harness.run_evaluation as run_evaluation
    import swebench.harness.test_spec.test_spec as test_spec_module

    original_make_test_spec = test_spec_module.make_test_spec

    def make_test_spec_with_arch(instance: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["arch"] = arch
        test_spec = original_make_test_spec(instance, *args, **kwargs)
        changes = adapt_test_spec_for_arm64(test_spec)
        if changes:
            print(
                f"Adapted {changes} Conda package pins for ARM64 in "
                f"{test_spec.instance_id}."
            )
        return test_spec

    test_spec_module.make_test_spec = make_test_spec_with_arch
    docker_build.make_test_spec = make_test_spec_with_arch
    reporting.make_test_spec = make_test_spec_with_arch
    if hasattr(run_evaluation, "make_test_spec"):
        run_evaluation.make_test_spec = make_test_spec_with_arch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation with an explicit Docker image arch."
    )
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance_ids", nargs="+")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--cache_level", choices=["none", "base", "env", "instance"], default="env")
    parser.add_argument("--clean", type=str, default="False")
    parser.add_argument("--open_file_limit", type=int, default=4096)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--rewrite_reports", type=str, default="False")
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--instance_image_tag", default="latest")
    parser.add_argument("--report_dir", default=".")
    parser.add_argument("--arch", choices=["auto", "x86_64", "arm64"], default="auto")
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> int:
    args = parse_args()
    arch = auto_arch() if args.arch == "auto" else args.arch
    namespace = normalize_namespace(args.namespace)
    if namespace is None and not args.modal:
        ensure_ubuntu_base_image(arch)
    patch_swebench_arch(arch)

    from swebench.harness.run_evaluation import main as run_main

    run_main(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=args.instance_ids,
        predictions_path=args.predictions_path,
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level=args.cache_level,
        clean=str_to_bool(args.clean),
        open_file_limit=args.open_file_limit,
        run_id=args.run_id,
        timeout=args.timeout,
        namespace=namespace,
        rewrite_reports=str_to_bool(args.rewrite_reports),
        modal=args.modal,
        instance_image_tag=args.instance_image_tag,
        report_dir=args.report_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
