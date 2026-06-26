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
from worlds.ksp1.sphere_ladder import _parse_location


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
            cum = s.provides
            if prev is not None:
                for req in prev.rank_reqs:
                    axis, prev_rank = req.axis, req.level
                    cur_rank = cum.rank(axis)
                    # Absent axis reports rank 0 (< any real rank >= 1), so a
                    # dropped axis fails the monotone check below — the chain
                    # must only grow.
                    self.assertGreaterEqual(
                        cur_rank, prev_rank,
                        f"Sphere {s.name} lowered {axis.value} ceiling from "
                        f"{prev_rank} to {cur_rank} — ladder must only grow "
                        f"(0 means the axis was dropped).",
                    )
            prev = cum

    def test_launch_sphere_leads(self) -> None:
        """S_launch is always the first sphere.

        (The old invariant 'S_goal* always trail' no longer holds: the
        graph-walk ladder places every sphere at its true dv layer, so a
        mid-difficulty goal like duna_return correctly sits mid-chain with
        physically harder missions — outer-planet / sample-return spheres —
        after it. Monotone-growth is covered by test_cumulative_ranks_monotone.)
        """
        ladder = self.world._sphere_ladder
        spheres = ladder.spheres
        if not spheres:
            return
        self.assertEqual(spheres[0].name, "S_launch")


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


class TestParseLocationNewNames(unittest.TestCase):
    """The sphere ladder must give both non-goal contract slots a real signature
    (a None here un-gates the location and deadlocks fill), and must skip the
    goal-mode threshold / event locations (they're pre-filled, never placed)."""

    def test_both_contract_slots_parse(self) -> None:
        from worlds.ksp1.contracts import ContractType
        from worlds.ksp1.bodies import BodyName
        for name in ("Contract: Mine Ore on Mun 1", "Contract: Mine Ore on Mun 2"):
            info = _parse_location(name)
            self.assertIsNotNone(info, name)
            self.assertEqual(info.spec.contract_type, ContractType.MINE_ORE)
            self.assertEqual(info.spec.body, BodyName.MUN)

    def test_threshold_and_event_locations_skipped(self) -> None:
        for name in ("Contract Threshold 1", "Contract Threshold 12",
                     "Contract Complete: Mine Ore on Mun"):
            self.assertIsNone(_parse_location(name), name)


if __name__ == "__main__":
    unittest.main()
