from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_swebench_batch import aggregate_models
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


if __name__ == "__main__":
    unittest.main()
