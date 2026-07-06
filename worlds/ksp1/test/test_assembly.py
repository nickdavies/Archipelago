"""Multi-launch orbital assembly (tail-only): eligibility lock, retry gating,
manifests, and partition invariants.

The assembly retry lifts a mission's orbital stack in ≤3 chunks docked in
home low orbit when the single launch is inexpressible even at unlimited pad.
Probes at the table generator's exact bar (difficulty + rep-selection
overhead) like test_apollo_split / test_escalated_builds.
"""
import unittest

import worlds.ksp1.capability as capability
from worlds.ksp1.bodies import (
    BodyName, DIFFICULTY_PROFILES, MissionBuilder, MissionType)
from worlds.ksp1.capability import (
    ProfileResult, _assembly_partitions, compute_capability_from_items,
    evaluate_mission_detailed)
from worlds.ksp1.data.feasibility import ASSEMBLY_ELIGIBLE_MISSIONS
from worlds.ksp1.parts import DEFAULT_PART_MANAGER
from worlds.ksp1.parts.types import CapabilityFlag
from worlds.ksp1.rocket_math import StageResult
from worlds.ksp1.scripts.generate_feasibility import (
    DEFAULT_OVERHEAD, _profile_with_overhead)

_PROBE_PROFILE = _profile_with_overhead("small", DEFAULT_OVERHEAD)
_PART_DB = DEFAULT_PART_MANAGER.parts


def _names_providing(flag: CapabilityFlag) -> frozenset[str]:
    return frozenset(
        nm for nm, parts in _PART_DB.items()
        if any(flag in getattr(p, "provides", ()) for p in parts))


_PORT_NAMES = _names_providing(CapabilityFlag.DOCKING_PORT)
_PROBE_CORE_NAMES = _names_providing(CapabilityFlag.PROBE_CORE)


def _max_kit(exclude: frozenset[str] = frozenset()):
    counts = {n: 1 for n in _PART_DB if n not in exclude}
    return lambda name: counts.get(name, 0)


class TestEligibilitySet(unittest.TestCase):
    """Lock the probed set: the Eve-destination tail from the three homes
    whose single launch can't lift the stack (Kerbin SSR; Laythe/Tylo
    Return + SSR).  A change here means the dv model moved — re-inspect."""

    def test_probed_set_contents(self) -> None:
        self.assertEqual(ASSEMBLY_ELIGIBLE_MISSIONS, frozenset({
            (BodyName.KERBIN, BodyName.EVE, MissionType.SAMPLE_RETURN),
            (BodyName.LAYTHE, BodyName.EVE, MissionType.RETURN),
            (BodyName.LAYTHE, BodyName.EVE, MissionType.SAMPLE_RETURN),
            (BodyName.TYLO, BodyName.EVE, MissionType.RETURN),
            (BodyName.TYLO, BodyName.EVE, MissionType.SAMPLE_RETURN),
        }))


class TestKerbinEveSSR(unittest.TestCase):
    """Kerbin-home crewed Eve SSR at the probe bar: the flagship assembly
    closure (single launch inexpressible even at unlimited pad)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mb = MissionBuilder(home=BodyName.KERBIN)
        cls.cap, cls.flags = compute_capability_from_items(
            _max_kit(), difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True, mission_builder=cls.mb)

    def _detailed(self, flags=None):
        return evaluate_mission_detailed(
            flags or self.flags, DIFFICULTY_PROFILES[_PROBE_PROFILE],
            BodyName.EVE, MissionType.SAMPLE_RETURN, crewed=True,
            mission_builder=self.mb)

    def test_closes_via_assembly_with_manifested_gear(self) -> None:
        res = self._detailed()
        self.assertTrue(res.feasible)
        self.assertTrue(res.via_assembly)
        # launch_mass is the HEAVIEST single lifter, not the stack total.
        self.assertGreater(res.launch_mass, 0.0)
        # ≥2 launches: group 0 must hold more than one lifter's stages, and
        # at least one docked joint puts 2 ports on a chunk-bottom manifest
        # (operator requirement: the launches' manifests show the cost).
        equip = [e for sr in res.stage_results for e in sr.equipment]
        port_counts = [n for n, nm in equip if nm in _PORT_NAMES]
        self.assertIn(2, port_counts,
                      "a chunk joint must manifest both docking ports")
        # Parked chunks are pilotless: probe core(s) on the mission manifests.
        self.assertTrue(
            any(nm in _PROBE_CORE_NAMES for _n, nm in equip),
            "a parked chunk must carry a probe core")

    def test_assembly_off_stays_infeasible(self) -> None:
        """Common-path guard: with the eligibility set emptied the retry
        never fires and the mission stays out — assembly is unreachable from
        the standard/Apollo paths."""
        capability._ASSEMBLY_OVERRIDE = frozenset()
        try:
            res = self._detailed()
        finally:
            capability._ASSEMBLY_OVERRIDE = None
        self.assertFalse(res.feasible)
        self.assertFalse(res.via_assembly)

    def test_no_docking_port_no_assembly(self) -> None:
        """The retry is ports-gated: stripping every docking port drops the
        mission back out (monotone: adding ports only ever helps)."""
        cap, flags = compute_capability_from_items(
            _max_kit(exclude=_PORT_NAMES), difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True, mission_builder=self.mb)
        res = self._detailed(flags)
        self.assertFalse(res.feasible)

    def test_no_probe_cores_no_parked_chunk(self) -> None:
        """Every non-terminal chunk parks pilotless; without any probe core
        the chunk gear is unbuildable and assembly must refuse (the crewed
        pod can command only its own chunk)."""
        cap, flags = compute_capability_from_items(
            _max_kit(exclude=_PROBE_CORE_NAMES),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True, mission_builder=self.mb)
        res = self._detailed(flags)
        self.assertFalse(res.feasible)


class TestPartitionEnumerator(unittest.TestCase):
    """Pure-function invariants of _assembly_partitions."""

    @staticmethod
    def _failed(cum_by_group: dict[int, float], n_groups: int) -> ProfileResult:
        def stage(m):
            return StageResult(
                delta_v=0.0, twr_at_ignition=0.0, twr_at_burnout=0.0,
                engine_is_throttleable=True, engine_has_gimbal=True,
                stage_mass_wet=m, stage_mass_dry=m, engine_count=1,
                fill_fraction=1.0, engine_name="x")
        stages = [stage(m) for m in cum_by_group.values()]
        return ProfileResult(
            False,
            partial_stages=stages,
            partial_stage_groups=list(cum_by_group.keys()),
            edge_groups=[[] for _ in range(n_groups)])

    def test_partitions_shape_and_order(self) -> None:
        # 4 orbital groups (1..4), cumulative masses descending upward.
        failed = self._failed({1: 1000.0, 2: 600.0, 3: 300.0, 4: 100.0}, 5)
        parts = _assembly_partitions(failed)
        self.assertTrue(parts)
        for bottoms in parts:
            self.assertEqual(bottoms[0], 1)
            self.assertLessEqual(len(bottoms), 3)
            self.assertEqual(list(bottoms), sorted(set(bottoms)))
            # chunk masses telescope back to the full stack mass
            masses = []
            for i, b in enumerate(bottoms):
                upper = ({1: 1000.0, 2: 600.0, 3: 300.0, 4: 100.0}
                         [bottoms[i + 1]] if i + 1 < len(bottoms) else 0.0)
                masses.append({1: 1000.0, 2: 600.0, 3: 300.0, 4: 100.0}[b]
                              - upper)
            self.assertAlmostEqual(sum(masses), 1000.0)
        # best-first: no later candidate has a smaller max chunk
        def max_chunk(bottoms):
            cum = {1: 1000.0, 2: 600.0, 3: 300.0, 4: 100.0}
            return max(cum[b] - (cum[bottoms[i + 1]]
                                 if i + 1 < len(bottoms) else 0.0)
                       for i, b in enumerate(bottoms))
        maxes = [max_chunk(b) for b in parts]
        self.assertEqual(maxes, sorted(maxes))

    def test_incomplete_stack_yields_nothing(self) -> None:
        # Group 2 missing (an upper stage failed): assembly can't help.
        failed = self._failed({1: 1000.0, 3: 300.0, 4: 100.0}, 5)
        self.assertEqual(_assembly_partitions(failed), [])

    def test_too_few_groups_yields_nothing(self) -> None:
        failed = self._failed({1: 1000.0}, 2)
        self.assertEqual(_assembly_partitions(failed), [])


if __name__ == "__main__":
    unittest.main()
