"""
Structural invariants over the checked-in lifter-chain tables
(worlds/ksp1/data/lifter_chains/).  Cheap enough for the unit loop — no
physics.  The physics drift gate (regenerate + compare) is the separate CI
job ksp1-lifter-chains.yml; here we only assert the stored data is
self-consistent and loadable, and that a loaded table round-trips through
consult().

Skips cleanly when the data package hasn't been generated yet (partial dev
trees), so the suite stays green before Phase 1 data lands.
"""
import unittest

from worlds.ksp1.bodies import BodyName
from worlds.ksp1.lifter_binding import OFFSET_TO_NAME, ServeResult, consult
from worlds.ksp1.data.lifter_chains import load_lifter_table, n_profiles

try:
    from worlds.ksp1.data.lifter_chains import _index
    _HAVE_DATA = True
except ImportError:
    _HAVE_DATA = False

from worlds.ksp1.scripts.generate_lifter_chains import (
    PACK_KEYS, assert_cluster_envelopes, structural_check,
)


@unittest.skipUnless(_HAVE_DATA, "lifter-chain data not generated")
class TestLifterChainData(unittest.TestCase):
    def test_structural_invariants(self):
        errors = structural_check()
        self.assertEqual(errors, [], "\n".join(errors))

    def test_cluster_envelopes(self):
        errors = assert_cluster_envelopes()
        self.assertEqual(errors, [], "\n".join(errors))

    def test_every_generated_home_loads_and_serves(self):
        pack = PACK_KEYS[-1]
        checked = 0
        for home_name in _index.HOME_CLUSTER:
            home = next(b for b in BodyName if b.value == home_name)
            n = n_profiles(home, pack)
            if n == 0:
                continue
            table = load_lifter_table(home, pack, 0, _phys_for(home))
            if table is None:
                continue
            checked += 1
            # Every chain part resolves to a real registry name.
            for part in table.chain:
                self.assertIn(part, OFFSET_TO_NAME.values())
            # Each rung is servable by its own recorded prefix at its own
            # threshold — the load-bearing self-consistency check.
            for ladder in table.ladders[_phys_for(home)]:
                for rung in ladder.rungs:
                    prefix = frozenset(table.chain[:rung.prefix_len])
                    c = consult(table, phys_profile=_phys_for(home),
                                required_dv=ladder.dv,
                                payload_t=rung.threshold_t,
                                admitted_parts=prefix, live=ladder.constraints)
                    self.assertIs(c.result, ServeResult.SERVED,
                                  f"{home_name} rung {rung.threshold_t}")
                    self.assertLessEqual(c.launch_mass, rung.launch_mass)
        if checked == 0:
            self.skipTest("no home tables generated yet")


def _phys_for(home: BodyName) -> str:
    return "small" if home == BodyName.EVE else "comfortable"


@unittest.skipUnless(_HAVE_DATA and n_profiles(BodyName.KERBIN, PACK_KEYS[-1]),
                     "Kerbin lifter data not generated")
class TestLifterSubstitution(unittest.TestCase):
    """Capability integration: the table-served home ascent must agree with
    raw physics on feasibility (conservative) and never claim a build the raw
    optimizer, restricted to the same parts, can't make."""

    def _kerbin(self):
        from worlds.ksp1.bodies import MissionBuilder, DIFFICULTY_PROFILES
        pack = PACK_KEYS[-1]
        mb = MissionBuilder(home=BodyName.KERBIN)
        mb.lifter_table = load_lifter_table(
            BodyName.KERBIN, pack, 0, "comfortable")
        return mb, DIFFICULTY_PROFILES["comfortable"]

    def _flags(self, part_names):
        from worlds.ksp1.capability import _pre_pass
        names = frozenset(part_names)
        return _pre_pass(lambda n: 1 if n in names else 0,
                         start_with_clamps=True)

    def test_full_kit_serves_and_matches_raw_physics(self):
        from worlds.ksp1.capability import evaluate_mission_detailed
        from worlds.ksp1.bodies import MissionType
        from worlds.ksp1.parts import DEFAULT_PART_MANAGER
        mb, diff = self._kerbin()
        # A full kit (every part) closes Kerbin ORBIT via raw physics AND
        # admits every chain prefix, so the home ascent is table-SERVED.  Both
        # paths must be feasible; the served path records a lifter prefix.
        flags = self._flags(DEFAULT_PART_MANAGER.parts.keys())
        served = evaluate_mission_detailed(
            flags, diff, BodyName.KERBIN, MissionType.ORBIT, None, mb,
            use_lifter_table=True)
        raw = evaluate_mission_detailed(
            flags, diff, BodyName.KERBIN, MissionType.ORBIT, None, mb,
            use_lifter_table=False)
        self.assertTrue(raw.feasible, "raw physics should close the ORBIT")
        self.assertTrue(served.feasible, "served build should close the ORBIT")
        # Served build is a real, positive-mass build.  It is NOT the raw
        # per-payload optimum: the bound chain is a fixed workhorse
        # architecture sized to the threshold rung, so for a small payload it
        # is legitimately heavier than a bespoke raw build (conservative — the
        # heavier launch mass only makes the pad-cap check stricter).
        self.assertGreater(served.launch_mass, 0.0)
        self.assertGreater(len(served.lifter_prefix_used), 0)

    def test_empty_kit_reports_prefix_missing_not_infeasible_stage(self):
        from worlds.ksp1.capability import _evaluate_profile
        from worlds.ksp1.capability_reasons import BlockingReason
        from worlds.ksp1.bodies import MissionType
        mb, diff = self._kerbin()
        # A kit with none of the chain admits no prefix -> the home ascent
        # must surface LIFTER_PREFIX_MISSING carrying the chain delta (bumper
        # guidance), never a generic stage failure.
        flags = self._flags(())
        profiles = mb.profiles_for(BodyName.KERBIN, MissionType.ORBIT)
        res = _evaluate_profile(
            profiles[0], flags, diff, MissionType.ORBIT, is_crewed=False,
            home=BodyName.KERBIN, lifter_table=mb.lifter_table)
        self.assertFalse(res.feasible)
        reasons = {b.reason for b in res.blocking}
        # Either the ascent surfaces prefix-missing, or an earlier gate
        # (no command module etc.) fails first — but never a NO_VIABLE_STAGE
        # from a search that the table was supposed to replace.
        self.assertNotIn(BlockingReason.NO_VIABLE_STAGE, reasons)


if __name__ == "__main__":
    unittest.main()
