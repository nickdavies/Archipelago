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
from worlds.ksp1.scripts.generate_feasibility import _profile_with_overhead

# Frozen at the historical +0.25 bar ON PURPOSE (not DEFAULT_OVERHEAD): these
# tests exercise the assembly machinery and the bugs-110/111 failing-launch
# pins, which need a bar where the single launch is inexpressible.  At the
# shipped 0.10 bar the flagship missions close single-launch and the retry
# never fires — the machinery would go untested.
_PROBE_PROFILE = _profile_with_overhead("small", 0.25)
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
    whose single launch can't lift the stack at some difficulty (Kerbin SSR;
    Laythe SSR; Tylo Return + SSR — Laythe Eve Return closes single-launch
    at every difficulty since the 0.10 overhead).  A change here means the
    dv model moved — re-inspect."""

    def test_probed_set_contents(self) -> None:
        self.assertEqual(ASSEMBLY_ELIGIBLE_MISSIONS, frozenset({
            (BodyName.KERBIN, BodyName.EVE, MissionType.SAMPLE_RETURN),
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
    """Pure-function invariants of _assembly_partitions (standalone-mass
    input; see _assembly_standalone_masses and bugs 110/111)."""

    _MASSES = {1: 400.0, 2: 300.0, 3: 200.0, 4: 100.0}

    @staticmethod
    def _failed(mass_by_group: dict[int, float], n_groups: int) -> ProfileResult:
        return ProfileResult(
            False,
            partial_group_mass=dict(mass_by_group),
            edge_groups=[[] for _ in range(n_groups)])

    def _chunk_masses(self, bottoms, n_groups=5):
        out = []
        for i, b in enumerate(bottoms):
            hi = bottoms[i + 1] if i + 1 < len(bottoms) else n_groups
            out.append(sum(self._MASSES[g] for g in range(b, hi)))
        return out

    def test_partitions_shape_and_order(self) -> None:
        # 4 orbital groups (1..4) with per-group standalone masses.
        failed = self._failed(self._MASSES, 5)
        parts = _assembly_partitions(failed)
        self.assertTrue(parts)
        for bottoms in parts:
            self.assertEqual(bottoms[0], 1)
            self.assertLessEqual(len(bottoms), 3)
            self.assertEqual(list(bottoms), sorted(set(bottoms)))
            # chunk masses telescope back to the full stack mass
            self.assertAlmostEqual(sum(self._chunk_masses(bottoms)),
                                   sum(self._MASSES.values()))
        # best-first: no later candidate has a smaller max chunk
        maxes = [max(self._chunk_masses(b)) for b in parts]
        self.assertEqual(maxes, sorted(maxes))

    def test_incomplete_stack_yields_nothing(self) -> None:
        # Group 2 missing (an upper stage failed): assembly can't help.
        failed = self._failed({1: 400.0, 3: 200.0, 4: 100.0}, 5)
        self.assertEqual(_assembly_partitions(failed), [])

    def test_too_few_groups_yields_nothing(self) -> None:
        failed = self._failed({1: 400.0}, 2)
        self.assertEqual(_assembly_partitions(failed), [])


class TestStandaloneMasses(unittest.TestCase):
    """Regression pins for bugs 110/111: every orbital group of a failed
    home-launch probe — passive-descent groups included — must appear in
    ``partial_group_mass`` with a POSITIVE standalone mass, and the map must
    telescope exactly to the single-launch payload (so any chunk partition
    conserves mass, Apollo branches included)."""

    def _probe(self, home: BodyName, body: BodyName, mt: MissionType,
               crewed: bool):
        mb = MissionBuilder(home=home)
        cap, flags = compute_capability_from_items(
            _max_kit(), difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True, mission_builder=mb)
        profiles = mb.profiles_for(body, mt)
        self.assertTrue(profiles)
        diff = DIFFICULTY_PROFILES[_PROBE_PROFILE]
        return capability._evaluate_profile(
            profiles[0], flags, diff, mt, is_crewed=crewed, home=home,
            requires_rendezvous=True,
            apollo_split=capability._apollo_candidate(flags, mt))

    def _assert_complete_positive_telescoping(self, res) -> None:
        self.assertFalse(res.feasible)
        n_groups = len(res.edge_groups)
        self.assertGreaterEqual(n_groups, 3)
        masses = res.partial_group_mass
        # bug 110: passive-descent groups used to be missing entirely.
        self.assertEqual(set(masses), set(range(1, n_groups)))
        # bug 111: Apollo branch boundaries used to yield negative masses.
        for g, m in masses.items():
            self.assertGreater(m, 0.0, f"group {g} standalone mass {m}")
        # Telescoping: the map sums to the single-launch payload the failed
        # home launch was asked to lift.
        self.assertAlmostEqual(sum(masses.values()), res.launch_mass,
                               places=6)

    def test_eve_home_kerbin_land_passive_groups(self) -> None:
        """Eve-home Kerbin LAND fails on the mesa launch; the Kerbin
        chute-descent group is passive and must still be in the map."""
        res = self._probe(BodyName.EVE, BodyName.KERBIN, MissionType.LAND,
                          crewed=False)
        self._assert_complete_positive_telescoping(res)

    def test_tylo_home_eve_ssr_apollo_branches(self) -> None:
        """Tylo-home Eve SSR (Apollo-shaped) fails on the home launch; the
        parked-stack / lander branch groups must have positive masses."""
        res = self._probe(BodyName.TYLO, BodyName.EVE,
                          MissionType.SAMPLE_RETURN, crewed=True)
        self._assert_complete_positive_telescoping(res)


if __name__ == "__main__":
    unittest.main()
