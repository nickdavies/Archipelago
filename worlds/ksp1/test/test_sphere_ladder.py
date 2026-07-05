"""Invariant tests for the sphere ladder pre-fill mechanism.

These tests build a world (running gen_steps up through pre_fill) and
check structural properties of the resulting ``SphereLadder``: the
chain's cumulative rank ceilings only grow.  Rank-space bumper feasibility
(each anchor converges to a workable kit) is covered by ``test_rank_bumper``.
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
    """Each sphere's cumulative rank ceiling dominates the prior sphere's —
    the chain only relaxes constraints, never tightens them.
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


class TestLocationDescriptor(unittest.TestCase):
    """The physics meaning of a location is a real ``LocationDescriptor``
    (locations.py); the sphere ladder reads it off ``loc.descriptor`` instead of
    decoding the name.  A contract slot MUST carry a spec-bearing descriptor (a
    None un-gates the location and deadlocks fill); a per-body mission carries its
    event; the body-agnostic Splashdown is not capability-gated."""

    def test_contract_descriptor_carries_spec(self) -> None:
        from worlds.ksp1.contracts import ContractSpec, ContractType
        from worlds.ksp1.bodies import BodyName
        from worlds.ksp1.locations import LocationDescriptor
        d = LocationDescriptor.from_spec(
            ContractSpec(ContractType.MINE_ORE, BodyName.MUN))
        self.assertEqual(d.spec.contract_type, ContractType.MINE_ORE)
        self.assertEqual(d.spec.body, BodyName.MUN)
        self.assertIsNone(d.event)  # contracts aren't per-body mission events

    def test_mission_descriptor_carries_event(self) -> None:
        from worlds.ksp1.bodies import BodyName, MissionType
        from worlds.ksp1.locations import (
            EVENT_BY_NAME, EventName, LocationDescriptor, MissionLocation,
        )
        d = LocationDescriptor.from_mission(
            MissionLocation(BodyName.MUN, EventName.ORBIT, 1),
            EVENT_BY_NAME[EventName.ORBIT])
        self.assertEqual(d.body, BodyName.MUN)
        self.assertEqual(d.event, EventName.ORBIT)
        self.assertEqual(d.mission_type, MissionType.ORBIT)

    def test_splashdown_is_not_capability_gated(self) -> None:
        from worlds.ksp1.locations import LocationBuilder, LocationDescriptor
        # Body-agnostic Splashdown (body=None) has no single-body physics gate.
        self.assertIsNone(
            LocationDescriptor.from_home(LocationBuilder._SPLASHDOWN_DEF))


class TestDescriptorCoverage(KSP1TestBase):
    """Every capability-gated location that reaches the ladder must carry a
    descriptor attached at creation (a missing one silently un-gates it and
    deadlocks fill); the Splashdown / threshold locations carry None."""
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = False  # descriptors are attached in create_regions

    def test_missions_and_contracts_have_descriptors(self) -> None:
        from worlds.ksp1.locations import MISSION_LOCATION_NAMES
        world = self.world
        mission_names = set(MISSION_LOCATION_NAMES)
        contract_names = {
            n for spec in (*world.contract_specs, *world.goal_contract_specs)
            for n in spec.location_names(world.locations_per_contract)
        }
        saw_mission = saw_contract = False
        for loc in world.multiworld.get_locations(world.player):
            if loc.name in mission_names:
                self.assertIsNotNone(loc.descriptor, loc.name)
                self.assertIsNotNone(loc.descriptor.event, loc.name)
                saw_mission = True
            elif loc.name in contract_names:
                self.assertIsNotNone(loc.descriptor, loc.name)
                self.assertIsNotNone(loc.descriptor.spec, loc.name)
                saw_contract = True
            elif loc.name == "Splashdown" or loc.name.startswith("Contract Threshold"):
                self.assertIsNone(loc.descriptor, loc.name)
        self.assertTrue(saw_mission, "expected mission locations in the world")
        self.assertTrue(saw_contract, "expected contract locations in the world")


if __name__ == "__main__":
    unittest.main()
