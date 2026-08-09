from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swebench_arm64 import adapt_test_spec_for_arm64, relax_conda_lock_for_arm64


LOCKED_ENVIRONMENT = """name: testbed
channels:
  - defaults
  - conda-forge
dependencies:
  - ca-certificates=2024.9.24=h06a4308_0
  - ld_impl_linux-64=2.40=h12ee557_0
  - python=3.9.20=he870216_1
  - pip:
      - flake8-comprehensions==3.15.0
prefix: /opt/miniconda3/envs/testbed
"""


class RelaxCondaLockTests(unittest.TestCase):
    def test_relaxes_only_direct_conda_build_pins(self) -> None:
        adapted, changes = relax_conda_lock_for_arm64(LOCKED_ENVIRONMENT)

        self.assertEqual(changes, 3)
        self.assertIn("ca-certificates=2024.9.24\n", adapted)
        self.assertIn("ld_impl_linux-aarch64=2.40\n", adapted)
        self.assertIn("python=3.9.20\n", adapted)
        self.assertIn("flake8-comprehensions==3.15.0", adapted)
        self.assertIn("prefix: /opt/miniconda3/envs/testbed", adapted)

    def test_test_spec_adaptation_is_arm64_only_and_idempotent(self) -> None:
        arm_spec = SimpleNamespace(
            arch="arm64",
            env_script_list=[f"cat <<'EOF' > /root/environment.yml\n{LOCKED_ENVIRONMENT}EOF"],
        )
        x86_spec = SimpleNamespace(
            arch="x86_64",
            env_script_list=[LOCKED_ENVIRONMENT],
        )

        self.assertEqual(adapt_test_spec_for_arm64(arm_spec), 3)
        self.assertEqual(adapt_test_spec_for_arm64(arm_spec), 0)
        self.assertEqual(adapt_test_spec_for_arm64(x86_spec), 0)
        self.assertEqual(x86_spec.env_script_list, [LOCKED_ENVIRONMENT])


if __name__ == "__main__":
    unittest.main()
