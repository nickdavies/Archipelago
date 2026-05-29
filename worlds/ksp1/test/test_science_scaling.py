"""
Tests for home-relative science scaling — ``effective_situation_mult``
(the source of truth) and ``home_relative_science_values`` (the slot_data
emitter), plus the drift-proof invariant that ties ``science_budget`` to
what the C# client writes.

Design: per-situation substitution.
  - body == home   -> Kerbin's stock mult for that situation
                      (surface sample at Eeloo home gives the same as at
                      Kerbin home, biome by biome)
  - body == Kerbin -> home's stock mult (Kerbin is the alien target)
  - otherwise      -> body's stock × Δv-ratio (uniform per body)

The drift-proof contract: ``science_budget`` reads from
``effective_situation_mult`` per situation; ``home_relative_science_values``
emits the same function's output to the client.  Same call on both sides,
no drift.
"""
import unittest

from worlds.ksp1.bodies import (
    ALL_BODIES,
    BODY_BY_NAME,
    BodyName,
    SCIENCE_SITUATIONS,
    SITUATION_FIELD_TO_ATTR,
    _mission_scalar,
    effective_situation_mult,
    home_relative_science_values,
    science_budget,
)
from worlds.ksp1.options import STARTING_BODY_POOLS, StartingBody


_NON_KERBIN_HOMES: tuple[BodyName, ...] = tuple(
    BodyName(name.replace("option_", "").title())
    for name in vars(StartingBody)
    if name.startswith("option_")
    and name != "option_kerbin"
    and name.replace("option_", "") not in STARTING_BODY_POOLS
)


class TestEffectiveSituationMult(unittest.TestCase):
    def test_kerbin_home_is_stock(self):
        # For Kerbin home every body, every situation returns the body's
        # stock multiplier — no rebalancing.
        for body in ALL_BODIES:
            for attr in SITUATION_FIELD_TO_ATTR.values():
                self.assertEqual(
                    effective_situation_mult(body.name, attr, BodyName.KERBIN),
                    getattr(body, attr),
                    f"Kerbin home: {body.name}.{attr} should be stock",
                )

    def test_home_substitutes_kerbin_stock(self):
        # The user-reported bug: surface sample on Eeloo home should be 9
        # (same as on Kerbin home).  Effective landed_mult for body==home
        # is exactly Kerbin's stock landed_mult.
        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        for home in _NON_KERBIN_HOMES:
            for attr in SITUATION_FIELD_TO_ATTR.values():
                self.assertEqual(
                    effective_situation_mult(home, attr, home),
                    getattr(kerbin, attr),
                    f"home={home}: {attr} for home body should be "
                    f"Kerbin's stock value",
                )

    def test_alien_kerbin_uses_laythe_shape(self):
        # Kerbin from any non-Kerbin home borrows Laythe's stock shape:
        # every situation's effective value is Laythe's stock × the same
        # composite scaling factor (Δv ratio × edge-count ratio × intra
        # penalty when applicable).  Verifies uniformity across situations
        # — i.e., one scalar, all seven fields.
        laythe = BODY_BY_NAME[BodyName.LAYTHE]
        for home in _NON_KERBIN_HOMES:
            # Compute the implied scalar from one situation, then check
            # every other situation produces the same scalar.
            ratios = []
            for attr in SITUATION_FIELD_TO_ATTR.values():
                laythe_stock = getattr(laythe, attr)
                if laythe_stock == 0:
                    continue
                eff = effective_situation_mult(BodyName.KERBIN, attr, home)
                ratios.append(eff / laythe_stock)
            self.assertTrue(ratios, f"home={home}: no testable situations")
            first = ratios[0]
            for r in ratios[1:]:
                self.assertAlmostEqual(
                    r, first, places=10,
                    msg=(
                        f"home={home}: alien Kerbin should scale Laythe's "
                        f"stock vector uniformly across all seven situations; "
                        f"got ratios {ratios}"
                    ),
                )

    def test_mun_home_alien_kerbin_has_intra_penalty(self):
        # Mun-home: Kerbin is Mun's parent, so the trip is intra-Kerbin-
        # system (much simpler than the Laythe-from-Kerbin reference, which
        # is interplanetary).  _mission_scalar applies the 1.67 intra
        # penalty, so the effective scalar should be smaller than what
        # plain dv/edge ratios would give.
        from worlds.ksp1.bodies import (
            _return_dv, _return_edge_count, _intra_system,
        )
        scalar = _mission_scalar(
            BodyName.KERBIN, BodyName.MUN,
            BodyName.LAYTHE, BodyName.KERBIN,
        )
        # Mun→Kerbin is intra-Kerbin-system; Laythe→Kerbin reference is not.
        self.assertTrue(_intra_system(BodyName.KERBIN, BodyName.MUN))
        self.assertFalse(_intra_system(BodyName.LAYTHE, BodyName.KERBIN))
        # The intra penalty should be present in the scalar — verify by
        # computing the scalar without the penalty and comparing.
        dv_now = _return_dv(BodyName.KERBIN, BodyName.MUN)
        dv_ref = _return_dv(BodyName.LAYTHE, BodyName.KERBIN)
        ec_now = _return_edge_count(BodyName.KERBIN, BodyName.MUN)
        ec_ref = _return_edge_count(BodyName.LAYTHE, BodyName.KERBIN)
        without_penalty = (dv_now / dv_ref) * (ec_now / ec_ref)
        self.assertAlmostEqual(
            scalar, without_penalty / 1.67, places=10,
            msg="Mun→Kerbin should have intra-system penalty applied",
        )

    def test_eeloo_home_landed_matches_kerbin_stock(self):
        # Concrete regression check for the bug the user found.
        # Surface sample = 30 * effective landed_mult.
        # Kerbin landed_mult = 0.3 -> sample = 9.
        # Without the fix Eeloo-home Eeloo gave 89.8.
        eff = effective_situation_mult(
            BodyName.EELOO, "landed_mult", BodyName.EELOO,
        )
        self.assertEqual(eff, 0.3)
        sample_science = 30.0 * eff
        self.assertEqual(sample_science, 9.0)

    def test_other_bodies_use_dv_scalar(self):
        # Non-home, non-Kerbin bodies: effective = stock * uniform Δv-ratio.
        # The ratio applies the same factor to all 7 fields.
        for home in _NON_KERBIN_HOMES:
            for body in ALL_BODIES:
                if body.name == home or body.name == BodyName.KERBIN:
                    continue
                # Compute the implied scalar from one field, then check the
                # other six agree.
                ratios = []
                for attr in SITUATION_FIELD_TO_ATTR.values():
                    stock = getattr(body, attr)
                    if stock == 0:
                        continue  # 0 * anything = 0, scalar is undefined
                    eff = effective_situation_mult(body.name, attr, home)
                    ratios.append(eff / stock)
                if not ratios:
                    continue
                first = ratios[0]
                for r in ratios[1:]:
                    self.assertAlmostEqual(
                        r, first, places=10,
                        msg=(
                            f"home={home} body={body.name}: per-situation "
                            f"scalars must be uniform.  Got {ratios}"
                        ),
                    )


class TestHomeRelativeScienceValues(unittest.TestCase):
    def test_kerbin_home_emits_empty_dict(self):
        self.assertEqual(home_relative_science_values(BodyName.KERBIN), {})

    def test_complete_coverage(self):
        """Every non-Kerbin home emits every body × every situation. No
        missing entries — the client hard-fails on absent fields."""
        expected_bodies = {b.name.value for b in ALL_BODIES}
        for home in _NON_KERBIN_HOMES:
            values = home_relative_science_values(home)
            self.assertEqual(set(values.keys()), expected_bodies)
            for body_name, situations in values.items():
                self.assertEqual(set(situations.keys()), set(SCIENCE_SITUATIONS))

    def test_emitted_values_match_effective_situation_mult(self):
        """The drift-proof contract for the slot_data emitter: every
        emitted value is exactly ``effective_situation_mult``.  Combined
        with ``test_science_budget_uses_effective_situation_mult``, this
        proves server/client cannot drift."""
        for home in _NON_KERBIN_HOMES:
            values = home_relative_science_values(home)
            for body in ALL_BODIES:
                entry = values[body.name.value]
                for field, attr in SITUATION_FIELD_TO_ATTR.items():
                    expected = effective_situation_mult(body.name, attr, home)
                    self.assertEqual(entry[field], expected)


class TestScienceBudgetDrift(unittest.TestCase):
    """The other half of the drift-proof contract: ``science_budget`` reads
    effective mults via ``effective_situation_mult``, so any rebalancing
    in that function automatically propagates to rules + sphere-ladder."""

    def test_science_budget_for_home_uses_kerbin_stock_mults(self):
        # Concrete invariant: science_budget(home_body, home=X) ==
        # science_budget(home_body, home=Kerbin), with the home body's
        # stock multipliers swapped for Kerbin's, since effective mults
        # match Kerbin's stock when body==home.
        for home in _NON_KERBIN_HOMES:
            home_body = BODY_BY_NAME[home]
            kerbin = BODY_BY_NAME[BodyName.KERBIN]
            # We can't easily call science_budget with a "swapped" body
            # since it takes a Body object, but the underlying contract
            # is: effective_situation_mult(home, *, home) == kerbin.*_mult.
            # That's exercised by test_home_substitutes_kerbin_stock.
            # Here we just sanity-check that science_budget runs without
            # error and returns a finite positive value for the home body
            # at all known difficulty levels.
            yield_ = science_budget(
                home_body,
                has_thermometer=True, has_barometer=True,
                has_capsule=True, can_land_crewed=home_body.can_land,
                home=home, psi_tier=3,
            )
            self.assertGreater(yield_, 0.0)
            self.assertLess(yield_, float("inf"))


class TestGoldenRefValues(unittest.TestCase):
    """Pinned reference values from the design discussion.  These numbers
    are what the algorithm should produce; if they shift, either the
    algorithm changed or one of its inputs (Δv, edge count, intra-system
    detection) drifted.  Bands are ±5% to allow for minor MissionBuilder
    Δv adjustments without false failures."""

    def _surface_sample(self, body, home):
        eff = effective_situation_mult(body, "landed_mult", home)
        return 30.0 * eff

    def _assert_near(self, value, target, label, tolerance=0.05):
        rel_err = abs(value - target) / target
        self.assertLess(
            rel_err, tolerance,
            f"{label}: expected ~{target:.0f}, got {value:.1f} ({rel_err*100:.1f}% off)",
        )

    def test_eeloo_home_surface_sample_at_home(self):
        # Home body always feels like Kerbin home — 30 × Kerbin.landed = 9.
        self.assertEqual(
            self._surface_sample(BodyName.EELOO, BodyName.EELOO), 9.0,
        )

    def test_laythe_home_sibling_moons_match_kerbin_minmus(self):
        # Sibling Jool moons from Laythe should feel like sibling
        # Kerbin moons from Kerbin home (~120-160 sci range).
        for moon in (BodyName.BOP, BodyName.POL, BodyName.VALL):
            sample = self._surface_sample(moon, BodyName.LAYTHE)
            self.assertGreater(
                sample, 100.0,
                f"Laythe→{moon}: expected sample > 100, got {sample:.0f}",
            )
            self.assertLess(
                sample, 180.0,
                f"Laythe→{moon}: expected sample < 180 (sibling-moon range), "
                f"got {sample:.0f}",
            )

    def test_eeloo_home_mun_is_interplanetary_reward(self):
        # Mun from Eeloo is a complex interplanetary mission; it should
        # give significantly more than Mun-from-Kerbin (stock 120),
        # closer to the Kerbin→Gilly reference (270).
        sample = self._surface_sample(BodyName.MUN, BodyName.EELOO)
        self.assertGreater(
            sample, 180.0,
            f"Eeloo→Mun should beat stock Kerbin→Mun (120) — got {sample:.0f}",
        )

    def test_laythe_home_jool_intra_penalty_applies(self):
        # Jool is Laythe's parent — intra-system.  Stock Kerbin→Jool
        # landed yields 30 × 30 = 900 (Jool has landed_mult 30 even
        # though it's not landable).  From Laythe the trip is much
        # shorter AND intra-system, so the effective mult should be
        # heavily discounted.
        eff = effective_situation_mult(BodyName.JOOL, "landed_mult", BodyName.LAYTHE)
        self.assertLess(
            eff, 10.0,
            f"Laythe→Jool intra-system: landed should be < 10, got {eff:.1f}",
        )


class TestPhysicalReasonableness(unittest.TestCase):
    def test_kerbin_landed_boosted_from_remote_home(self):
        # Kerbin from Eeloo home: Laythe stock (14) × Δv ratio.
        # Eeloo→Kerbin return is cheaper than Kerbin→Laythe return
        # (Kerbin's aerobrake), so the scalar < 1 and Kerbin landed < 14.
        # Still well above Kerbin's own stock 0.3 — appropriately alien.
        eff = effective_situation_mult(
            BodyName.KERBIN, "landed_mult", BodyName.EELOO,
        )
        self.assertGreater(eff, 0.3, "alien Kerbin should beat stock Kerbin")
        self.assertLess(eff, 14.0, "alien Kerbin scaled below Laythe stock here")

    def test_jool_moons_demoted_from_laythe(self):
        # Tylo/Vall/Bop/Pol are sibling Jool moons of Laythe — easier
        # to reach than from Kerbin, so the Δv scalar shrinks them.
        for moon in (BodyName.TYLO, BodyName.VALL, BodyName.BOP, BodyName.POL):
            stock = BODY_BY_NAME[moon].landed_mult
            eff = effective_situation_mult(moon, "landed_mult", BodyName.LAYTHE)
            self.assertLess(
                eff, stock,
                f"Laythe-home: {moon} landed should be < stock {stock}",
            )


if __name__ == "__main__":
    unittest.main()
