"""Per-body expectation table and kit-monotonicity sweep for the unified
staged atmospheric-landing model (capability._solve_atmo_landing).

Physics primitives are unit-tested in test_aero_descent.py; the capability
wiring in test_capability.py::TestStagedLandingMix.  This file locks the
end-to-end behaviour that matters for the seed: which bodies land passively,
which need a burn, and the Golden-Rule monotonicity property (a bigger kit
never loses a landing, and never lands heavier).
"""
import unittest

from worlds.ksp1.bodies import BodyName, BODY_BY_NAME, DIFFICULTY_PROFILES
from worlds.ksp1.capability import _solve_atmo_landing, EquipmentFlags
from worlds.ksp1.parts import DEFAULT_PART_MANAGER

_DB = DEFAULT_PART_MANAGER.parts


def _part(name: str):
    return next(p for p in _DB[name])


# Part fixtures (real PART_DB parts)
_HS1 = _part("HeatShield1")            # 1.25m rigid shield
_INFLATABLE = _part("InflatableHeatShield")   # 10m, huge deployed drag
_MK16 = _part("parachuteSingle")       # inline main
_MK2R = _part("parachuteRadial")       # radial main
_MK25 = _part("parachuteDrogue")       # inline drogue
_MK12R = _part("radialDrogue")         # radial drogue
_SWIVEL = _part("liquidEngine2.v2")    # throttleable gimballed
_TANK = _part("fuelTank.long")


def _flags(shields=(), chutes=(), engine=False):
    f = EquipmentFlags()
    f.available_heat_shields = list(shields)
    if shields:
        f.has_heat_shield = True
        f.best_heat_shield = max(shields, key=lambda h: h.size_class)
        f.best_drag_shield = max(shields, key=lambda h: h.drag_area)
    f.available_parachutes = list(chutes)
    f.has_parachutes = bool(chutes)
    key = lambda p: p.mass / max(p.drag_area, 1e-3)
    def best(pred):
        cs = [p for p in chutes if pred(p)]
        return min(cs, key=key) if cs else None
    f.best_radial_main = best(lambda p: p.is_radial and not p.is_drogue)
    f.best_inline_main = best(lambda p: not p.is_radial and not p.is_drogue)
    f.best_radial_drogue = best(lambda p: p.is_radial and p.is_drogue)
    f.best_inline_drogue = best(lambda p: not p.is_radial and p.is_drogue)
    if engine:
        f.available_engines = [_SWIVEL]
        f.has_throttleable_engine = True
        f.available_tanks = [_TANK]
    return f


def _cover(flags, pod_size=1.25):
    """Lightest owned shield covering the pod, or None — mirrors the pre-check
    covering pair-pick (no undersized fallback) that now feeds the evaluator."""
    covering = [hs for hs in flags.available_heat_shields
                if hs.size_class >= pod_size]
    return min(covering, key=lambda h: h.mass) if covering else None


def _solve(payload, body, flags, diff_name="comfortable", reentry=False,
           pod_size=1.25):
    diff = DIFFICULTY_PROFILES[diff_name]
    twr = max(1.3, diff.min_twr_atmo)
    v = (1.4 * body.lo_escape_velocity) if reentry else body.lo_circular_velocity
    return _solve_atmo_landing(payload, body, flags, diff,
                               twr_floor=twr, v_entry=v,
                               dvGL_cap=body.dv.dvGL or 0.0,
                               coverage_shield=_cover(flags, pod_size),
                               pod_size=pod_size)


class TestPerBodyExpectations(unittest.TestCase):
    KERBIN = BODY_BY_NAME[BodyName.KERBIN]
    DUNA = BODY_BY_NAME[BodyName.DUNA]
    LAYTHE = BODY_BY_NAME[BodyName.LAYTHE]
    EVE = BODY_BY_NAME[BodyName.EVE]

    def test_kerbin_capsule_passive(self) -> None:
        mix = _solve(0.9, self.KERBIN, _flags(shields=[_HS1], chutes=[_MK16]))
        self.assertTrue(mix.feasible)
        self.assertFalse(mix.needs_burn)

    def test_kerbin_interplanetary_reentry_passive(self) -> None:
        # Higher entry speed than from low orbit, still passive for a light pod.
        mix = _solve(0.9, self.KERBIN, _flags(shields=[_HS1], chutes=[_MK16]),
                     reentry=True)
        self.assertTrue(mix.feasible)
        self.assertFalse(mix.needs_burn)

    def test_duna_inline_main_needs_burn(self) -> None:
        # Thin atmosphere + a single inline chute (capped at 1) can't brake a
        # 5t lander; with an engine it finishes propulsively — the partial-
        # propulsive path the old flat aero=100 pretended was free.
        mix = _solve(5.0, self.DUNA,
                     _flags(shields=[_HS1], chutes=[_MK16], engine=True))
        self.assertTrue(mix.feasible)
        self.assertTrue(mix.needs_burn)
        self.assertGreater(mix.burn_dv, 0.0)

    def test_duna_drogue_enables_passive(self) -> None:
        # A drogue bridges the high-speed regime -> passive on Duna.
        mix = _solve(5.0, self.DUNA,
                     _flags(shields=[_HS1], chutes=[_MK2R, _MK12R]))
        self.assertTrue(mix.feasible)
        self.assertFalse(mix.needs_burn)

    def test_duna_inflatable_enables_passive(self) -> None:
        mix = _solve(5.0, self.DUNA,
                     _flags(shields=[_HS1, _INFLATABLE], chutes=[_MK2R]))
        self.assertTrue(mix.feasible)
        self.assertFalse(mix.needs_burn)

    def test_eve_needs_drogue_or_inflatable(self) -> None:
        # Eve has the fastest entry in the system: under just a small rigid
        # shield even a light craft settles right at the mains' safe-deploy
        # speed, so mains-only can't deploy safely.  A drogue (high-q bridge)
        # rescues a light craft passively; the inflatable shield's huge drag
        # rescues a heavy one.  This is the enabler pattern working.
        light_mains = _solve(2.0, self.EVE, _flags(shields=[_HS1], chutes=[_MK2R]))
        self.assertFalse(light_mains.feasible)
        light_drogue = _solve(2.0, self.EVE,
                             _flags(shields=[_HS1], chutes=[_MK2R, _MK12R]))
        self.assertTrue(light_drogue.feasible)
        self.assertFalse(light_drogue.needs_burn)
        heavy_inflatable = _solve(10.0, self.EVE,
                                 _flags(shields=[_HS1, _INFLATABLE], chutes=[_MK2R]))
        self.assertTrue(heavy_inflatable.feasible)
        self.assertFalse(heavy_inflatable.needs_burn)

    def test_eve_engine_finishes(self) -> None:
        # With a throttleable engine, the residual finishes propulsively.
        mix = _solve(2.0, self.EVE,
                     _flags(shields=[_HS1], chutes=[_MK16], engine=True))
        self.assertTrue(mix.feasible)
        self.assertTrue(mix.needs_burn)

    def test_no_engine_no_drag_infeasible(self) -> None:
        # Heavy craft, one inline chute, no engine: can't land -> infeasible,
        # NOT a false-positive passive claim.
        mix = _solve(15.0, self.DUNA, _flags(shields=[_HS1], chutes=[_MK16]))
        self.assertFalse(mix.feasible)
        self.assertGreater(mix.residual_speed, 6.0)


class TestKitMonotonicity(unittest.TestCase):
    """Golden Rule: growing the kit never loses a landing and never lands
    heavier.  Walk a canonical ladder adding one capability at a time."""

    def _ladder(self):
        # Each rung is a superset of the previous.
        return [
            ("shield+inline-main", _flags(shields=[_HS1], chutes=[_MK16])),
            ("+radial-main", _flags(shields=[_HS1], chutes=[_MK16, _MK2R])),
            ("+drogue", _flags(shields=[_HS1], chutes=[_MK16, _MK2R, _MK12R])),
            ("+inflatable", _flags(shields=[_HS1, _INFLATABLE],
                                   chutes=[_MK16, _MK2R, _MK12R])),
            ("+engine", _flags(shields=[_HS1, _INFLATABLE],
                               chutes=[_MK16, _MK2R, _MK12R], engine=True)),
        ]

    def test_no_landing_lost_and_mass_non_increasing(self) -> None:
        for body_name in (BodyName.KERBIN, BodyName.DUNA, BodyName.LAYTHE,
                          BodyName.EVE):
            body = BODY_BY_NAME[body_name]
            for payload in (1.5, 5.0, 10.0):
                prev_feasible = False
                prev_mass = float("inf")
                for label, flags in self._ladder():
                    mix = _solve(payload, body, flags)
                    # Once feasible, never regress to infeasible with MORE kit.
                    if prev_feasible:
                        self.assertTrue(
                            mix.feasible,
                            f"{body_name} {payload}t: '{label}' lost a landing "
                            f"that a smaller kit had")
                    # Landing hardware mass must not grow as the kit grows
                    # (a superset kit picks an equal-or-lighter mix).
                    if mix.feasible and not mix.needs_burn and prev_feasible:
                        self.assertLessEqual(
                            mix.hardware_mass, prev_mass + 1e-6,
                            f"{body_name} {payload}t: '{label}' landed heavier "
                            f"than a smaller kit")
                    if mix.feasible:
                        prev_feasible = True
                        if not mix.needs_burn:
                            prev_mass = min(prev_mass, mix.hardware_mass)


if __name__ == "__main__":
    unittest.main()
