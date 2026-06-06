"""Invariant tests for the sphere ladder pre-fill mechanism.

These tests build a world (running gen_steps up through pre_fill) and
check structural properties of the resulting ``SphereLadder``: the
predictable spine is present, the chain's cumulative rank ceilings only
grow, and goal spheres trail.  Rank-space bumper feasibility (each
anchor converges to a workable kit) is covered by ``test_rank_bumper``.
"""
from __future__ import annotations

import unittest

from worlds.ksp1.test.base import KSP1TestBase


class TestPredictableSpheres(KSP1TestBase):
    """The launch / orbit / goal spine must always be present."""
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_launch_and_orbit_present(self) -> None:
        ladder = self.world._sphere_ladder
        names = {s.location_name for s in ladder.spheres}
        self.assertIn("Kerbin First Launch", names,
                      "S_launch must always be in the predictable spine")
        self.assertIn("Kerbin Orbit 1", names,
                      "S_orbit must always be in the predictable spine")

    def test_at_least_one_goal_sphere(self) -> None:
        ladder = self.world._sphere_ladder
        goal_spheres = [s for s in ladder.spheres if s.name.startswith("S_goal")]
        self.assertGreaterEqual(
            len(goal_spheres), 1,
            "At least one goal sphere should anchor the ladder",
        )


class TestSphereChainOrdered(KSP1TestBase):
    """The ladder is sorted so S_launch leads, S_orbit follows, and
    S_goal* trail.  The semantic invariant: each sphere's cumulative rank
    ceiling dominates the prior sphere's — the chain only relaxes
    constraints, never tightens them.
    """
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_cumulative_ranks_monotone(self) -> None:
        ladder = self.world._sphere_ladder
        prev = None
        for s in ladder.spheres:
            cum = s.ranks
            if prev is not None:
                for axis, prev_rank in prev.upper_bounds:
                    cur_rank = cum.get(axis)
                    self.assertIsNotNone(
                        cur_rank,
                        f"Sphere {s.name} dropped axis {axis.value} that the "
                        f"prior sphere constrained — chain must only grow.",
                    )
                    self.assertGreaterEqual(
                        cur_rank, prev_rank,
                        f"Sphere {s.name} lowered {axis.value} ceiling from "
                        f"{prev_rank} to {cur_rank} — ladder must only grow.",
                    )
            prev = cum

    def test_chain_groups_anchored(self) -> None:
        """S_launch / S_orbit lead; S_goal* always trail; intermediates middle."""
        ladder = self.world._sphere_ladder
        spheres = ladder.spheres
        if not spheres:
            return
        self.assertEqual(spheres[0].name, "S_launch")
        seen_goal = False
        for s in spheres:
            if s.name.startswith("S_goal"):
                seen_goal = True
            elif seen_goal:
                self.fail(
                    f"Non-goal sphere {s.name} appears after a S_goal sphere"
                )


class TestSmallGoalMunFlag(KSP1TestBase):
    """The mun_flag goal has S_goal close to S_orbit; the ladder must
    still construct a valid chain without raising.
    """
    options = {"goal": "mun_flag", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_chain_constructed(self) -> None:
        ladder = self.world._sphere_ladder
        self.assertGreaterEqual(len(ladder.spheres), 2)
        names = {s.location_name for s in ladder.spheres}
        self.assertIn("Kerbin First Launch", names)
        self.assertIn("Kerbin Orbit 1", names)


if __name__ == "__main__":
    unittest.main()
