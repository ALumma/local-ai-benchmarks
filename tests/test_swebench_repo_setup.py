from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swebench_repo_setup import remove_stale_repo_branch_hint


class RepoSetupTests(unittest.TestCase):
    def test_removes_known_stale_branch_hint(self) -> None:
        test_spec = SimpleNamespace(
            repo="sympy/sympy",
            version="1.7",
            repo_script_list=[
                "git clone -o origin --branch 1.7 --single-branch "
                "https://github.com/sympy/sympy /testbed",
                "git reset --hard abc123",
            ],
        )

        self.assertEqual(remove_stale_repo_branch_hint(test_spec), 1)
        self.assertEqual(
            test_spec.repo_script_list[0],
            "git clone -o origin https://github.com/sympy/sympy /testbed",
        )
        self.assertEqual(test_spec.repo_script_list[1], "git reset --hard abc123")
        self.assertEqual(remove_stale_repo_branch_hint(test_spec), 0)

    def test_leaves_other_repo_versions_unchanged(self) -> None:
        command = (
            "git clone -o origin --branch stable/2.2.x --single-branch "
            "https://github.com/django/django /testbed"
        )
        test_spec = SimpleNamespace(
            repo="django/django",
            version="2.2",
            repo_script_list=[command],
        )

        self.assertEqual(remove_stale_repo_branch_hint(test_spec), 0)
        self.assertEqual(test_spec.repo_script_list, [command])


if __name__ == "__main__":
    unittest.main()
