"""Eve-as-home: the mountain-pad ascent discount.

The AP home pad on Eve sits on the 25°S mesa at 6,140 m ASL
(AP_KSC_Sites), above the densest slab of Eve's 5-atm soup — the HOME
ascent is therefore far cheaper than the sea-level ``dvGL`` (8,000).
``Body.home_pad_ascent_dv`` derives the discount (vacuum floor + the
atmospheric excess scaled by √pressure-fraction, conservative vs
community highland-ascent data), and ONLY the home body's trunk ascent
uses it: a lander at a destination touched down anywhere, so
destination ascents keep sea level.

The curated Eve ban is also home-exempt: starting on Eve IS the opt-in,
and every Eve-home mission traverses the Eve ascent as its pad launch.
"""
from __future__ import annotations

import unittest

from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, EdgeType, EVE, MissionBuilder, MissionType,
)
from worlds.ksp1.test.base import KSP1TestBase


class TestPadAscentDerivation(unittest.TestCase):
    def test_eve_mesa_value(self) -> None:
        """√p scaling from the 6,140 m mesa: ~6,417 m/s (community highland
        figures run ~6,000-6,500 — we sit at the conservative top)."""
        self.assertAlmostEqual(EVE.home_pad_ascent_dv(), 6416.6, delta=1.0)

    def test_all_other_bodies_pass_through(self) -> None:
        for body in ALL_BODIES:
            if body.name == BodyName.EVE:
                continue
            self.assertEqual(body.home_pad_ascent_dv(), body.dv.dvGL,
                             body.name)


class TestHomeVsDestinationAsymmetry(unittest.TestCase):
    def _eve_ascent_dv(self, home: BodyName) -> float:
        mb = MissionBuilder(home=home)
        mission = (MissionType.ORBIT if home == BodyName.EVE
                   else MissionType.RETURN)
        for profile in mb.profiles_for(BodyName.EVE, mission):
            for e in profile:
                if (e.edge_type == EdgeType.ATMOSPHERIC_ASCENT
                        and e.body == BodyName.EVE):
                    return e.base_dv
        self.fail(f"no Eve ascent edge found from home={home}")

    def test_home_ascent_uses_pad_destination_uses_sea_level(self) -> None:
        self.assertAlmostEqual(self._eve_ascent_dv(BodyName.EVE),
                               6416.6, delta=1.0)
        self.assertEqual(self._eve_ascent_dv(BodyName.KERBIN), 8000.0)


class TestEveHomeGeneration(KSP1TestBase):
    """End-to-end: an Eve-home seed at expert difficulty (small physics —
    where the mesa discount opens everything) must fill and be beatable.
    The inherited default tests are the assertion; the curated ban's
    home-exemption is load-bearing here (without it every mission would
    be EXCLUDED and the seed would be empty)."""
    run_default_tests = True
    needs_real_pre_fill = True
    options = {
        "goal": "duna_return",
        "starting_body": "eve",
        "difficulty": "expert",
    }

    def test_home_missions_not_banned(self) -> None:
        world = self.multiworld.worlds[self.player]
        self.assertNotIn(
            (BodyName.EVE, MissionType.ORBIT), world.unachievable_missions,
            "Eve-home orbit must not be voided by the curated ban",
        )


if __name__ == "__main__":
    unittest.main()
