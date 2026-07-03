"""Apollo-split return architecture (capability ceiling).

The whole-stack cascade forces a return mission's descent + surface-ascent
stages to carry the entire return-transfer stack.  With a docking port + RCS
in the kit, capability retries the profile as an Apollo split: the return
stack parks in destination orbit while the lander flies down/up with only
the pod, then rejoins (a charged rendezvous burn) for the trip home.

These tests pin the three load-bearing properties:

* the retry genuinely raises the ceiling (a mission the standard
  architecture cannot close at max kit becomes feasible),
* it is gated on concrete docking gear (no port → verdict identical to the
  standard architecture),
* the detailed evaluation reproduces the gating verdict with the real
  docking parts on the stage manifests (manifest mass == charged mass).
"""
from __future__ import annotations

import unittest

from worlds.ksp1.bodies import (
    BodyName, DIFFICULTY_PROFILES, MissionBuilder, MissionType,
)
from worlds.ksp1.capability import (
    _apollo_split_for, _group_edges, compute_capability_from_items,
    evaluate_mission_detailed,
)
from worlds.ksp1.locations import EventName
from worlds.ksp1.parts import CapabilityFlag, DEFAULT_PART_MANAGER
from worlds.ksp1.scripts.generate_feasibility import (
    DEFAULT_OVERHEAD, _profile_with_overhead,
)

_PART_DB = DEFAULT_PART_MANAGER.parts

# The feasibility table probes at difficulty + rep-selection overhead; that is
# the bar at which Tylo SSR is standard-infeasible and Apollo closes it.  At
# PLAIN generous the whole-stack architecture already closes Tylo SSR, so the
# ceiling tests must probe exactly like the table generator does.
_PROBE_PROFILE = _profile_with_overhead("generous", DEFAULT_OVERHEAD)

_PORT_NAMES = frozenset(
    nm for nm, parts in _PART_DB.items()
    if any(CapabilityFlag.DOCKING_PORT in getattr(p, "provides", ())
           for p in parts)
)


def _max_kit(exclude: frozenset[str] = frozenset()):
    counts = {n: 1 for n in _PART_DB if n not in exclude}
    return lambda name: counts.get(name, 0)


class TestApolloCeiling(unittest.TestCase):
    """Max-kit feasibility at the table's probe bar (generous + overhead),
    Kerbin home.

    Tylo Sample Return is the reference mission: pre-Apollo it was
    max-kit-infeasible at this bar (it sat in the feasibility table), and
    the Apollo split closes it.  If tuning ever makes it standard-feasible
    here, replace it with whichever deep return the regenerated pre-Apollo
    table names.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.mb = MissionBuilder(home=BodyName.KERBIN)
        cls.cap, cls.flags = compute_capability_from_items(
            _max_kit(),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=cls.mb,
        )

    def test_apollo_closes_tylo_sample_return(self) -> None:
        self.assertTrue(
            self.cap.bodies[BodyName.TYLO].access[EventName.SAMPLE_RETURN],
            "Tylo SSR should be feasible at max kit via the Apollo split",
        )

    def test_no_docking_port_no_apollo(self) -> None:
        """Without a docking port the retry never fires — the verdict is the
        standard architecture's (infeasible for Tylo SSR at generous)."""
        cap, _flags = compute_capability_from_items(
            _max_kit(exclude=_PORT_NAMES),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=self.mb,
        )
        self.assertFalse(
            cap.bodies[BodyName.TYLO].access[EventName.SAMPLE_RETURN],
            "stripping docking ports must drop Tylo SSR back to infeasible",
        )

    def test_detailed_build_carries_docking_gear(self) -> None:
        """The detailed (spoiler / cross-check) path reproduces the gating
        verdict, with one port per docked side on the manifests."""
        res = evaluate_mission_detailed(
            self.flags, DIFFICULTY_PROFILES[_PROBE_PROFILE],
            BodyName.TYLO, MissionType.SAMPLE_RETURN,
            crewed=True, mission_builder=self.mb,
        )
        self.assertTrue(res.feasible, res.failure_reasons)
        port_count = sum(
            cnt for sr in res.stage_results
            for (cnt, nm) in sr.equipment if nm in _PORT_NAMES
        )
        self.assertEqual(
            port_count, 2,
            "expected one docking port on the lander and one on the parked "
            f"stack; manifests: {[sr.equipment for sr in res.stage_results]}",
        )


class TestApolloSplitFinder(unittest.TestCase):
    """Boundary detection on real profiles (no optimizer runs — fast)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mb = MissionBuilder(home=BodyName.KERBIN)
        # Flags with docking gear: max kit through the cheap pre-pass only.
        _cap, cls.flags = compute_capability_from_items(
            _max_kit(),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=cls.mb,
        )
        cls.pod = cls.flags.lightest_capsule

    def _groups(self, body: BodyName, mt: MissionType):
        profile = self.mb.profiles_for(body, mt)[0]
        return _group_edges(profile, staging_tier=2)

    def test_split_found_on_destination_return(self) -> None:
        groups = self._groups(BodyName.TYLO, MissionType.RETURN)
        split = _apollo_split_for(groups, BodyName.KERBIN, self.flags, self.pod)
        self.assertIsNotNone(split)
        self.assertEqual(split.ascent_gidx, split.land_gidx + 1)
        self.assertGreater(split.land_gidx, 0,
                           "an outbound side must exist below the landing")
        self.assertLess(split.ascent_gidx + 1, len(groups),
                        "a return stack must exist to park")

    def test_home_return_not_applicable(self) -> None:
        groups = self._groups(BodyName.KERBIN, MissionType.RETURN)
        self.assertIsNone(
            _apollo_split_for(groups, BodyName.KERBIN, self.flags, self.pod),
            "a home round trip has nothing to park",
        )

    def test_one_way_mission_not_applicable(self) -> None:
        groups = self._groups(BodyName.TYLO, MissionType.LAND)
        self.assertIsNone(
            _apollo_split_for(groups, BodyName.KERBIN, self.flags, self.pod),
            "a one-way landing has no post-ascent legs",
        )


if __name__ == "__main__":
    unittest.main()
