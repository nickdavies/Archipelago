"""Eve ascent calibration: mountain pad + highlands-in-logic landers.

Eve's SEA-LEVEL ascent is ~11,500-12,000 m/s (community-verified); the
tabulated ``dvGL`` is 12,000.  No in-logic ascent pays that raw figure:

* The AP home pad sits on the 25°S mesa at 6,140 m ASL (AP_KSC_Sites),
  above the densest slab of the 5-atm soup.  ``Body.home_pad_ascent_dv``
  derives the pad ascent (vacuum floor + atmospheric excess scaled by
  √pressure-fraction) → ~8,996 m/s.
* Destination landers assume a HIGHLANDS touchdown in the same elevation
  band (``assume_highlands_landing``) — the landing site is
  player-controlled and high ground is the universal Eve strategy — so
  the trunk ascent edge uses the same pad-anchored figure both ways.
  Splashdown corollary: an ocean landing is ONE-WAY (see the field's
  caveat in bodies.py).

Eve-as-HOME is CLOSED for this release (operator decision, 2026-07-03):
even with the escalated home-ascent builds the recalibrated table keeps
expert without returns/flags beyond Gilly, so ``starting_body=eve`` is
rejected at generation until orbital assembly / ISRU raise the ceiling.
The mesa-pad model and the ESCALATED_HOME_ASCENT_EDGES machinery stay
live so reopening is a one-gate flip.
"""
from __future__ import annotations

import random
import unittest
from argparse import Namespace

from Options import OptionError

from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, EdgeType, EVE, MissionBuilder, MissionType,
)


class TestPadAscentDerivation(unittest.TestCase):
    def test_eve_mesa_value(self) -> None:
        """√p scaling from the 6,140 m mesa on the 12,000 sea-level base:
        ~8,996 m/s — between the community highlands (~8,000) and sea-level
        (~12,000) figures, conservative for the mesa's altitude band."""
        self.assertAlmostEqual(EVE.home_pad_ascent_dv(), 8996.4, delta=1.0)

    def test_all_other_bodies_pass_through(self) -> None:
        for body in ALL_BODIES:
            if body.name == BodyName.EVE:
                continue
            self.assertEqual(body.home_pad_ascent_dv(), body.dv.dvGL,
                             body.name)

    def test_only_eve_assumes_highlands_landing(self) -> None:
        for body in ALL_BODIES:
            self.assertEqual(body.assume_highlands_landing,
                             body.name == BodyName.EVE, body.name)


class TestAscentEdgeCalibration(unittest.TestCase):
    """Both directions use the pad-anchored figure — the home ascent
    launches from the mesa pad; a destination lander is assumed to touch
    down in the highlands band (never pays raw sea level)."""

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

    def test_home_and_destination_both_pad_anchored(self) -> None:
        self.assertAlmostEqual(self._eve_ascent_dv(BodyName.EVE),
                               8996.4, delta=1.0)
        self.assertAlmostEqual(self._eve_ascent_dv(BodyName.KERBIN),
                               8996.4, delta=1.0)


class TestEveHomeClosed(unittest.TestCase):
    """The product gate: ``starting_body=eve`` must be rejected in
    generate_early with a clear OptionError (not strand into an empty or
    unwinnable seed)."""

    def test_eve_home_rejected(self) -> None:
        import worlds  # noqa: F401
        import worlds.AutoWorld as AutoWorld
        from BaseClasses import CollectionState, MultiWorld
        from Generate import get_seed_name

        mw = MultiWorld(1)
        mw.game[1] = "Kerbal Space Program 1"
        mw.player_name = {1: "Tester"}
        mw.set_seed(1)
        random.seed(mw.seed)
        mw.seed_name = get_seed_name(random)
        args = Namespace()
        world_type = AutoWorld.AutoWorldRegister.world_types[
            "Kerbal Space Program 1"]
        opts = {"goal": "duna_return", "starting_body": "eve",
                "difficulty": "expert"}
        for name, option in world_type.options_dataclass.type_hints.items():
            setattr(args, name,
                    {1: option.from_any(opts.get(name, option.default))})
        mw.set_options(args)
        mw.state = CollectionState(mw)
        with self.assertRaisesRegex(OptionError, "starting_body=eve"):
            AutoWorld.call_all(mw, "generate_early")


if __name__ == "__main__":
    unittest.main()
