from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_swebench_batch import aggregate_models
from derive_swebench_batch import (
    copy_model_runs,
    derive_manifest,
    select_completed_instances,
    validate_existing_manifest,
)
from run_swebench_verified_batch_vllm import select_instances, validate_manifest


class BatchSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "instance_id": f"repo__task-{index:03d}",
                "repo": "owner/repo",
                "version": "1",
            }
            for index in range(100)
        ]

    def test_selection_is_fixed_unique_and_order_independent(self) -> None:
        required = ["repo__task-050"]
        first = select_instances(
            self.rows,
            count=50,
            seed=20260809,
            required_instance_ids=required,
        )
        second = select_instances(
            list(reversed(self.rows)),
            count=50,
            seed=20260809,
            required_instance_ids=required,
        )

        first_ids = [row["instance_id"] for row in first]
        self.assertEqual(first, second)
        self.assertEqual(first_ids[0], required[0])
        self.assertEqual(len(first_ids), 50)
        self.assertEqual(len(set(first_ids)), 50)

    def test_manifest_validation_rejects_changed_seed(self) -> None:
        manifest = {
            "dataset": "dataset",
            "split": "test",
            "count": 2,
            "seed": 1,
            "required_instances": ["one"],
            "instances": [
                {"instance_id": "one"},
                {"instance_id": "two"},
            ],
        }
        args = Namespace(
            dataset="dataset",
            split="test",
            count=2,
            seed=2,
            required_instance=["one"],
        )

        with self.assertRaisesRegex(ValueError, "seed"):
            validate_manifest(manifest, args)


class BatchAggregationTests(unittest.TestCase):
    def test_accuracy_uses_full_selected_denominator(self) -> None:
        manifest = {"count": 50, "instances": []}
        records = [
            {
                "model": "model-a",
                "model_profile": "profile-a",
                "status": "completed",
                "resolved": True,
                "patch_bytes": 100,
                "agent_seconds": 60,
                "evaluation_seconds": 10,
            },
            {
                "model": "model-a",
                "model_profile": "profile-a",
                "status": "completed",
                "resolved": False,
                "patch_bytes": 50,
                "agent_seconds": 120,
                "evaluation_seconds": 20,
            },
            {
                "model": "model-a",
                "model_profile": "profile-a",
                "status": "failed",
                "resolved": None,
                "patch_bytes": 0,
                "agent_seconds": 5,
                "evaluation_seconds": None,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            summary = aggregate_models(Path(temporary), manifest, records)[0]

        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["resolved"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["pending"], 47)
        self.assertEqual(summary["accuracy"], 1 / 50)
        self.assertEqual(summary["median_agent_seconds"], 90)


class BatchDerivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_manifest = {
            "schema_version": 1,
            "created_at": "2026-08-09T00:00:00Z",
            "dataset": "dataset",
            "split": "test",
            "count": 5,
            "seed": 123,
            "selection_method": "required_then_sha256_rank",
            "required_instances": ["repo__task-000"],
            "instances": [
                {"instance_id": f"repo__task-{index:03d}"}
                for index in range(5)
            ],
        }

    def test_derived_manifest_is_an_ordered_prefix(self) -> None:
        derived = derive_manifest(
            self.source_manifest,
            source_slug="source-batch",
            count=3,
        )

        self.assertEqual(derived["count"], 3)
        self.assertEqual(
            derived["instances"], self.source_manifest["instances"][:3]
        )
        self.assertEqual(
            derived["derived_from"],
            {"batch_slug": "source-batch", "count": 5},
        )
        self.assertEqual(self.source_manifest["count"], 5)

    def test_existing_manifest_validation_ignores_creation_time(self) -> None:
        expected = derive_manifest(
            self.source_manifest,
            source_slug="source-batch",
            count=3,
        )
        existing = dict(expected)
        existing["created_at"] = "later"

        validate_existing_manifest(existing, expected)
        existing["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_existing_manifest(existing, expected)

    def test_completed_filter_preserves_source_order(self) -> None:
        instances = self.source_manifest["instances"]
        model_slug = "model-a"
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            for index in [0, 2, 3, 4]:
                metadata = (
                    source_root
                    / "runs"
                    / instances[index]["instance_id"]
                    / model_slug
                    / "run_metadata.json"
                )
                metadata.parent.mkdir(parents=True)
                metadata.write_text(
                    '{"status":"completed"}\n', encoding="utf-8"
                )

            selected = select_completed_instances(
                source_root=source_root,
                source_instances=instances,
                model_slug=model_slug,
                count=3,
            )

        self.assertEqual(selected, [instances[0], instances[2], instances[3]])

    def test_completed_filter_is_recorded_in_manifest(self) -> None:
        instances = self.source_manifest["instances"]
        derived = derive_manifest(
            self.source_manifest,
            source_slug="source-batch",
            count=3,
            selected_instances=[instances[0], instances[2], instances[3]],
            completed_model="model-a",
        )

        self.assertEqual(
            derived["selection_method"],
            "source_order_filtered_by_completed_runs",
        )
        self.assertEqual(
            derived["derived_from"]["filter"],
            {"model": "model-a", "run_status": "completed"},
        )

    def test_copy_model_runs_does_not_overwrite_target(self) -> None:
        instances = self.source_manifest["instances"][:3]
        model_slug = "model-a"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            target_root = root / "target"
            source_model = (
                source_root
                / "runs"
                / instances[0]["instance_id"]
                / model_slug
            )
            source_model.mkdir(parents=True)
            (source_model / "run_metadata.json").write_text(
                '{"status":"completed"}\n', encoding="utf-8"
            )

            counts = copy_model_runs(
                source_root=source_root,
                target_root=target_root,
                instances=instances,
                model_slug=model_slug,
            )
            self.assertEqual(counts, {"copied": 1, "existing": 0, "missing": 2})

            target_metadata = (
                target_root
                / "runs"
                / instances[0]["instance_id"]
                / model_slug
                / "run_metadata.json"
            )
            target_metadata.write_text('{"status":"newer"}\n', encoding="utf-8")
            counts = copy_model_runs(
                source_root=source_root,
                target_root=target_root,
                instances=instances,
                model_slug=model_slug,
            )
            self.assertEqual(counts, {"copied": 0, "existing": 1, "missing": 2})
            self.assertEqual(
                target_metadata.read_text(encoding="utf-8"),
                '{"status":"newer"}\n',
            )


if __name__ == "__main__":
    unittest.main()
