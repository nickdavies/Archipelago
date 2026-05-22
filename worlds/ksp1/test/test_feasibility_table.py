"""Drift check for the checked-in proxy-bodies feasibility table.

The table at ``worlds/ksp1/data/feasibility.py`` is generated offline
by ``worlds/ksp1/scripts/generate_feasibility.py``.  If the dv model,
mission graph, or part database changes without the table being
regenerated, the world's banned-location set silently goes out of sync
with what the capability system would actually compute.

This test invokes the generator's verification path (``--check``) and
fails the suite when the on-disk table doesn't match what the script
would produce — same effect as running it in CI.
"""
import subprocess
import sys
import unittest
from pathlib import Path


class TestFeasibilityTableMatchesGenerator(unittest.TestCase):
    def test_checked_in_table_is_current(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]  # Archipelago/
        result = subprocess.run(
            [sys.executable, "-m",
             "worlds.ksp1.scripts.generate_feasibility", "--check"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                "worlds/ksp1/data/feasibility.py is out of date.  "
                "Regenerate with:\n\n"
                "    cd Archipelago && .venv/bin/python -m "
                "worlds.ksp1.scripts.generate_feasibility --write\n\n"
                f"Generator stderr:\n{result.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
