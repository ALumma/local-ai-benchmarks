from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_swebench_agent_vllm_one import (  # noqa: E402
    AgentTimeoutError,
    agent_wall_clock_timeout,
)


class AgentWallClockTimeoutTests(unittest.TestCase):
    def test_timeout_interrupts_blocking_work(self) -> None:
        started = time.monotonic()

        with self.assertRaises(AgentTimeoutError):
            with agent_wall_clock_timeout(0.05):
                time.sleep(1)

        self.assertLess(time.monotonic() - started, 0.5)

    def test_timeout_is_cancelled_after_context_exits(self) -> None:
        with agent_wall_clock_timeout(0.1):
            pass

        time.sleep(0.15)

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            with agent_wall_clock_timeout(0):
                pass


if __name__ == "__main__":
    unittest.main()
