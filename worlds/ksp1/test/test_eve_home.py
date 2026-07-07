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

Eve-as-HOME is supported: the mesa-pad ascent closes under escalated
home-ascent builds with asparagus sub-stages, and heavier departure
stacks are composed by multi-launch orbital assembly.  The feasibility
table gates per physics difficulty (fully in logic at comfortable/small/
zero; the generous deep-tail Return/SSR stay proxy-routed).
"""
from __future__ import annotations

import random
import unittest
from argparse import Namespace

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


class TestEveHomeGenerates(unittest.TestCase):
    """Eve-as-home generates cleanly through generate_early — the home body
    is exempt from the curated Eve ban (picking Eve IS the opt-in), so a
    non-Eve goal from an Eve home resolves without raising."""

    def _generate_early(self, difficulty: str) -> "object":
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
                "difficulty": difficulty}
        for name, option in world_type.options_dataclass.type_hints.items():
            setattr(args, name,
                    {1: option.from_any(opts.get(name, option.default))})
        mw.set_options(args)
        mw.state = CollectionState(mw)
        AutoWorld.call_all(mw, "generate_early")
        return mw.worlds[1]

    def test_eve_home_generates_expert(self) -> None:
        world = self._generate_early("expert")
        self.assertEqual(world.mission_builder.home, BodyName.EVE)
        # Comfortable/small tables are fully open, so the Eve-home seed
        # carries no model-infeasible mission classes at expert.
        self.assertEqual(world.unachievable_missions, frozenset())

    def test_eve_home_generates_casual_with_proxy_tail(self) -> None:
        # Casual → generous physics: the deep-tail Return/SSR stay
        # model-infeasible and route through the proxy (graceful, not a
        # strand).  Generation must still succeed.
        world = self._generate_early("casual")
        self.assertEqual(world.mission_builder.home, BodyName.EVE)
        self.assertTrue(world.unachievable_missions)


if __name__ == "__main__":
    unittest.main()
