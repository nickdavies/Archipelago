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
    _GAMEPLAY_BY_DIFFICULTY,
)
from worlds.ksp1.capability import (
    _apollo_split_for, _group_edges, compute_capability_from_items,
    evaluate_mission_detailed,
)
from worlds.ksp1.locations import EventName
from worlds.ksp1.parts import CapabilityFlag, DEFAULT_PART_MANAGER
from worlds.ksp1.scripts.generate_feasibility import _profile_with_overhead

_PART_DB = DEFAULT_PART_MANAGER.parts

# Frozen at the historical +0.25 bar ON PURPOSE (not DEFAULT_OVERHEAD): these
# tests need a bar at which Tylo SSR is standard-infeasible and Apollo closes
# it.  At plain generous — and at the shipped 0.10 overhead — the whole-stack
# architecture already closes it, so tracking the shipping bar would leave
# the Apollo machinery untested.
_PROBE_PROFILE = _profile_with_overhead("generous", 0.25)

_PORT_NAMES = frozenset(
    nm for nm, parts in _PART_DB.items()
    if any(CapabilityFlag.DOCKING_PORT in getattr(p, "provides", ())
           for p in parts)
)

_PROBE_CORE_NAMES = frozenset(
    nm for nm, parts in _PART_DB.items()
    if any(CapabilityFlag.PROBE_CORE in getattr(p, "provides", ())
           for p in parts)
)

_RCS_NAMES = frozenset(
    nm for nm, parts in _PART_DB.items()
    if any(CapabilityFlag.RCS in getattr(p, "provides", ())
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

    def test_crewed_apollo_without_probe_cores(self) -> None:
        """A crewed mission may leave a pilot aboard a second capsule as the
        parked stack's command source (the Apollo CM pattern) — stripping
        every probe core must NOT drop crewed Tylo SSR."""
        cap, _flags = compute_capability_from_items(
            _max_kit(exclude=_PROBE_CORE_NAMES),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=self.mb,
        )
        self.assertTrue(
            cap.bodies[BodyName.TYLO].access[EventName.SAMPLE_RETURN],
            "crewed Apollo should close with a capsule-parked stack",
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
        self.assertTrue(
            res.via_apollo,
            "an Apollo-retry closure must mark via_apollo so bracket-side "
            "consumers gate on the rendezvous it imposed",
        )

    def test_standard_closure_not_marked_via_apollo(self) -> None:
        """A mission the standard architecture closes never carries the
        Apollo rendezvous supplement."""
        res = evaluate_mission_detailed(
            self.flags, DIFFICULTY_PROFILES[_PROBE_PROFILE],
            BodyName.MUN, MissionType.RETURN,
            crewed=True, mission_builder=self.mb,
        )
        self.assertTrue(res.feasible, res.failure_reasons)
        self.assertFalse(res.via_apollo)


class TestDockingAttitudeGear(unittest.TestCase):
    """Docking gear gates (operator rule): the approach always needs torque
    authority (at least wheels), and below expert gameplay an RCS translation
    kit on top — an expert player can dock on main-engine translation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mb = MissionBuilder(home=BodyName.KERBIN)
        cls.cap_no_rcs, cls.flags_no_rcs = compute_capability_from_items(
            _max_kit(exclude=_RCS_NAMES),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=cls.mb,
        )

    def test_no_rcs_blocks_apollo_below_expert(self) -> None:
        """CONSERVATIVE_GAMEPLAY (casual/normal): stripping every RCS
        thruster must drop the Apollo-only Tylo SSR back to infeasible."""
        self.assertFalse(
            self.cap_no_rcs.bodies[BodyName.TYLO].access[
                EventName.SAMPLE_RETURN],
            "docking below expert requires the RCS approach kit",
        )

    def test_expert_docks_without_rcs(self) -> None:
        """Expert gameplay: wheels alone suffice — the same RCS-less kit
        closes Tylo SSR via Apollo, and no RCS part is charged."""
        expert_mb = MissionBuilder(home=BodyName.KERBIN)
        expert_mb.gameplay = _GAMEPLAY_BY_DIFFICULTY[2]
        res = evaluate_mission_detailed(
            self.flags_no_rcs, DIFFICULTY_PROFILES[_PROBE_PROFILE],
            BodyName.TYLO, MissionType.SAMPLE_RETURN,
            crewed=True, mission_builder=expert_mb,
        )
        self.assertTrue(res.feasible, res.failure_reasons)
        self.assertTrue(res.via_apollo)
        charged_rcs = [
            nm for sr in res.stage_results
            for (_cnt, nm) in sr.equipment if nm in _RCS_NAMES
        ]
        self.assertEqual(
            charged_rcs, [],
            "expert docking must not charge an unrequired RCS kit",
        )


class TestApolloBracketBuildingGate(unittest.TestCase):
    """bugs/104: the kit-dependent rendezvous supplement.

    ``mission_logic_needs`` is kit-independent, so a home-SYSTEM heavy-moon
    return closed only by the Apollo retry (Laythe-home Tylo Return is the
    live case) derives CAN_NAVIGATE_LOCAL's building set — while the real
    evaluator imposed ``requires_rendezvous=True`` (conics + nodes).  The
    bracket scan passes ``via_apollo`` so the gate requires what was proven.
    """

    def test_via_apollo_adds_rendezvous_buildings(self) -> None:
        from worlds.ksp1.sphere_ladder import _mission_building_reqs
        from worlds.ksp1.locations import LocationDescriptor
        from worlds.ksp1.items import PROGRESSIVE_MISSION_CONTROL_NAME
        mb = MissionBuilder(home=BodyName.LAYTHE)
        info = LocationDescriptor(
            body=BodyName.TYLO, mission_type=MissionType.RETURN,
            crewed=None, threshold_km=None)
        # Options leave home-system nav ungated: the kit-independent needs
        # carry no Mission Control requirement...
        base = _mission_building_reqs(
            info, mission_builder=mb, buildings_in_logic=True,
            local_needs_conics=False, local_needs_nodes=False)
        self.assertNotIn(
            PROGRESSIVE_MISSION_CONTROL_NAME, dict(base),
            "fixture invalidated: the base needs already require nodes — "
            "pick a target/options pair where they don't")
        # ...but an Apollo-closed bracket must add conics + nodes.
        supplemented = _mission_building_reqs(
            info, mission_builder=mb, buildings_in_logic=True,
            local_needs_conics=False, local_needs_nodes=False,
            via_apollo=True)
        self.assertGreaterEqual(
            dict(supplemented).get(PROGRESSIVE_MISSION_CONTROL_NAME, 0), 1,
            f"via_apollo must require Mission Control (nodes); got "
            f"{supplemented}")


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
        split = _apollo_split_for(groups, BodyName.KERBIN, self.flags,
                                  self.pod, is_crewed=True)
        self.assertIsNotNone(split)
        self.assertEqual(split.ascent_gidx, split.land_gidx + 1)
        self.assertGreater(split.land_gidx, 0,
                           "an outbound side must exist below the landing")
        self.assertLess(split.ascent_gidx + 1, len(groups),
                        "a return stack must exist to park")

    def test_home_return_not_applicable(self) -> None:
        groups = self._groups(BodyName.KERBIN, MissionType.RETURN)
        self.assertIsNone(
            _apollo_split_for(groups, BodyName.KERBIN, self.flags, self.pod,
                              is_crewed=True),
            "a home round trip has nothing to park",
        )

    def test_one_way_mission_not_applicable(self) -> None:
        groups = self._groups(BodyName.TYLO, MissionType.LAND)
        self.assertIsNone(
            _apollo_split_for(groups, BodyName.KERBIN, self.flags, self.pod,
                              is_crewed=True),
            "a one-way landing has no post-ascent legs",
        )


if __name__ == "__main__":
    unittest.main()
