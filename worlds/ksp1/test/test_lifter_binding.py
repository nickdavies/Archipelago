"""
Unit tests for lifter_binding.py — consult semantics, hint round-trip,
constraint direction-awareness, dv-variant enumeration.  Pure unit tests; the
miniature tables are hand-built (no generator, no world).
"""
import unittest

from worlds.ksp1.bodies import (
    BodyName, DIFFICULTY_PROFILES, MissionBuilder, effective_dv,
)
from worlds.ksp1.lifter_binding import (
    AscentConstraints, BoundLadder, BoundLifterTable, BoundRung, LifterHint,
    NAME_TO_OFFSET, ServeResult, consult, hint_from_stages,
    home_ascent_dv_variants,
)
from worlds.ksp1.rocket_math import StageResult

# Real part names (registry ksp_names) so offset resolution is exercised.
_ENGINE = "liquidEngine2.v2"
_TANK = "fuelTank"
_STACK_DEC = "Decoupler.0"
_RADIAL_DEC = "radialDecoupler"
_FUEL_LINE = "fuelLine"
_SRB = "solidBooster.v2"

CHAIN = (_ENGINE, _TANK, _STACK_DEC, _SRB, _RADIAL_DEC, _FUEL_LINE)

CONSTRAINTS = AscentConstraints(
    in_atmosphere=True, min_twr_liftoff=1.5, requires_throttleable=True,
    srb_needs_rcs=True, needs_heat_shield=False,
    pad_altitude_m=0.0,
)

HINT_A = LifterHint(split=(1.0,),
                    stages=((0, 0, NAME_TO_OFFSET[_ENGINE]),))
HINT_B = LifterHint(split=(0.3, 0.7),
                    stages=((4, 1, NAME_TO_OFFSET[_ENGINE]),
                            (0, 0, NAME_TO_OFFSET[_SRB])))


def _table() -> BoundLifterTable:
    ladders = {
        "comfortable": (
            BoundLadder(dv=4025.0, ceiling_t=300.0, constraints=CONSTRAINTS,
                        rungs=(BoundRung(10.0, 2, 50.0, HINT_A),
                               BoundRung(100.0, 4, 400.0, HINT_B))),
            BoundLadder(dv=4226.2, ceiling_t=250.0, constraints=CONSTRAINTS,
                        rungs=(BoundRung(10.0, 3, 60.0, HINT_A),
                               BoundRung(100.0, 6, 450.0, HINT_B))),
        ),
    }
    return BoundLifterTable(
        home=BodyName.KERBIN, profile_id=7, chain=CHAIN, ladders=ladders,
    )


class TestConsult(unittest.TestCase):
    def setUp(self):
        self.table = _table()

    def _consult(self, payload, admitted, dv=4025.0, phys="comfortable",
                 live=CONSTRAINTS):
        return consult(self.table, phys_profile=phys, required_dv=dv,
                       payload_t=payload, admitted_parts=frozenset(admitted),
                       live=live)

    def test_served_smallest_rung(self):
        c = self._consult(5.0, CHAIN[:2])
        self.assertIs(c.result, ServeResult.SERVED)
        self.assertEqual(c.launch_mass, 50.0)
        self.assertEqual(c.prefix_used, frozenset(CHAIN[:2]))
        self.assertIs(c.hint, HINT_A)

    def test_served_superset_kit(self):
        c = self._consult(5.0, set(CHAIN) | {"someOtherPart"})
        self.assertIs(c.result, ServeResult.SERVED)

    def test_prefix_missing_is_chain_ordered(self):
        c = self._consult(50.0, CHAIN[:2])   # needs prefix 4
        self.assertIs(c.result, ServeResult.PREFIX_MISSING)
        self.assertEqual(c.missing_parts, CHAIN[2:4])

    def test_over_ceiling(self):
        # A payload above what the FULL chain can lift (its top rung, 100t) is
        # over this seed's capacity -> OVER_CEILING (assembly), reporting the
        # chain's reach, not the pool ceiling.
        c = self._consult(400.0, set(CHAIN))
        self.assertIs(c.result, ServeResult.OVER_CEILING)
        self.assertEqual(c.ceiling_t, 100.0)

    def test_above_chain_max_is_over_ceiling(self):
        # Payload between the top rung (100t) and the pool ceiling (300t): the
        # committed chain can't single-launch it regardless of what's
        # collected -> OVER_CEILING (assembly), NOT a fall-through to live.
        for admitted in (CHAIN[:4], set(CHAIN)):
            c = self._consult(200.0, admitted)
            self.assertIs(c.result, ServeResult.OVER_CEILING, admitted)
            self.assertEqual(c.ceiling_t, 100.0)

    def test_dv_variant_selection(self):
        # Between the two bound dvs -> serves from the higher (conservative).
        c = self._consult(5.0, set(CHAIN), dv=4100.0)
        self.assertIs(c.result, ServeResult.SERVED)
        self.assertEqual(c.launch_mass, 60.0)
        self.assertEqual(c.dv_bound, 4226.2)

    def test_dv_above_max_not_covered(self):
        c = self._consult(5.0, set(CHAIN), dv=4300.0)
        self.assertIs(c.result, ServeResult.NOT_COVERED)

    def test_dv_epsilon_serves_only_within_half_ms(self):
        self.assertIs(self._consult(5.0, set(CHAIN), dv=4226.6).result,
                      ServeResult.SERVED)
        self.assertIs(self._consult(5.0, set(CHAIN), dv=4226.8).result,
                      ServeResult.NOT_COVERED)

    def test_unknown_phys_profile_not_covered(self):
        c = self._consult(5.0, set(CHAIN), phys="zero")
        self.assertIs(c.result, ServeResult.NOT_COVERED)

    def test_constraint_mismatch_not_covered(self):
        vac = AscentConstraints(
            in_atmosphere=False, min_twr_liftoff=1.2,
            requires_throttleable=True,
            srb_needs_rcs=True, needs_heat_shield=False, pad_altitude_m=0.0)
        c = self._consult(5.0, set(CHAIN), live=vac)
        self.assertIs(c.result, ServeResult.NOT_COVERED)


class TestConstraintServes(unittest.TestCase):
    def test_direction_awareness(self):
        base = CONSTRAINTS
        # Stricter bind serves looser live requests.
        looser = AscentConstraints(
            in_atmosphere=True, min_twr_liftoff=1.3,
            requires_throttleable=False,
            srb_needs_rcs=False, needs_heat_shield=False, pad_altitude_m=0.0)
        self.assertTrue(base.serves(looser))
        self.assertTrue(base.serves(base))
        # Looser bind must never serve a stricter live request.
        for field, val in (("min_twr_liftoff", 1.6),
                           ("needs_heat_shield", True)):
            stricter = AscentConstraints(**{**base.__dict__, field: val})
            self.assertFalse(base.serves(stricter), field)
        # Physics identity fields must match exactly, either direction.
        for field, val in (("in_atmosphere", False),
                           ("pad_altitude_m", 6140.0)):
            other = AscentConstraints(**{**base.__dict__, field: val})
            self.assertFalse(base.serves(other), field)
            self.assertFalse(other.serves(base), field)


class TestHint(unittest.TestCase):
    def test_to_guide_resolves_names(self):
        split, stages = HINT_B.to_guide()
        self.assertEqual(split, (0.3, 0.7))
        self.assertEqual(stages, ((4, 1, _ENGINE), (0, 0, _SRB)))

    def test_hint_from_stages_round_trip(self):
        stage_bottom = StageResult(
            delta_v=1234.5, twr_at_ignition=1.7, twr_at_burnout=3.4,
            engine_is_throttleable=True, engine_has_gimbal=False,
            stage_mass_wet=88.25, stage_mass_dry=17.5, engine_count=5,
            fill_fraction=0.75, engine_name=_ENGINE,
            tank_manifest=((4, _TANK),),
            equipment=[(2, _RADIAL_DEC), (2, _FUEL_LINE)],
            n_boosters=4, booster_engines=1,
        )
        stage_top = StageResult(
            delta_v=900.0, twr_at_ignition=2.0, twr_at_burnout=4.0,
            engine_is_throttleable=False, engine_has_gimbal=False,
            stage_mass_wet=20.0, stage_mass_dry=4.0, engine_count=1,
            fill_fraction=1.0, engine_name=_SRB,
            tank_manifest=(), equipment=[],
        )
        hint = hint_from_stages([stage_bottom, stage_top])
        self.assertEqual(hint.stages,
                         ((4, 1, NAME_TO_OFFSET[_ENGINE]),
                          (0, 0, NAME_TO_OFFSET[_SRB])))
        # split is canonicalized by the generator, not recovered here
        self.assertEqual(hint.split, ())
        _, guide_stages = LifterHint(split=(0.5, 0.5),
                                     stages=hint.stages).to_guide()
        self.assertEqual(guide_stages, ((4, 1, _ENGINE), (0, 0, _SRB)))


class TestDvVariants(unittest.TestCase):
    def test_kerbin_comfortable_matches_live_derivation(self):
        mb = MissionBuilder(home=BodyName.KERBIN)
        diff = DIFFICULTY_PROFILES["comfortable"]
        variants = home_ascent_dv_variants(mb, diff)
        self.assertEqual(len(variants), 2)
        base, rdv = variants
        self.assertLess(base, rdv)
        # Cross-check against the value production evaluates the Kerbin home
        # ascent at (measured via instrumented generation): 4025.0.
        self.assertAlmostEqual(base, 4025.0, places=1)
        self.assertAlmostEqual(
            rdv,
            effective_dv(3400.0 + MissionBuilder._RESCUE_RENDEZVOUS_DV, diff),
            places=1)

    def test_eve_home_uses_elevated_pad_dv(self):
        mb = MissionBuilder(home=BodyName.EVE)
        diff = DIFFICULTY_PROFILES["small"]
        base, rdv = home_ascent_dv_variants(mb, diff)
        # The home ascent edge must carry the mesa-pad discount, not sea-level
        # dvGL: measured production value is 9498.8 (expert/small).
        self.assertAlmostEqual(base, 9498.8, delta=1.0)
        self.assertGreater(rdv, base)


if __name__ == "__main__":
    unittest.main()
