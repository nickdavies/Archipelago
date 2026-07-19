"""Regression: this seed's exact home-orbit kit is locked to the player's world.

In a multiworld under Minimal accessibility (which only guarantees the *goal* is
reachable) AP's fill can scatter a player's bootstrap — especially the scarce
control source — into another player's world behind a deep check, leaving the
KSP player with ~0 in-logic checks for hours. This happened in a shipped seed
(``AP_32744882047302664921``): the only pod landed ~8h deep in another world.

The tail of ``apply_sphere_ladder`` marks THIS SEED's exact home-orbit kit (the
sphere-0 kit ∪ S_orbit's cumulative reps — a handful of parts that vary per
seed, NOT a whole category) as ``options.local_items``. AP's ``locality_rules``
then constrains those items to the player's own locations. The ladder is invoked
from ``set_rules`` (not ``pre_fill``) so this runs before ``locality_rules``.

Marking a whole *category* local (all pods / engines / tanks) is deliberately
avoided — it would gut the point of a multiworld — so these tests also assert
the marked set is the small per-seed kit, not a category.
"""
import unittest

from worlds.generic.Rules import locality_rules
from worlds.ksp1.parts import CapabilityFlag
from worlds.ksp1.test.base import KSP1TestBase
from test.general import setup_multiworld


def _has_control_source(world, name: str) -> bool:
    flags: set = set()
    for part in world.part_manager.parts.get(name, ()):
        flags |= set(getattr(part, "provides", ()) or ())
    return bool({CapabilityFlag.CAPSULE, CapabilityFlag.PROBE_CORE} & flags)


def _ladder_orbit_kit(world) -> set:
    ladder = world._sphere_ladder
    kit = set(ladder.spheres[0].reps_collected) if ladder.spheres else set()
    s_orbit = next((s for s in ladder.spheres if s.name == "S_orbit"), None)
    if s_orbit is not None:
        kit |= set(s_orbit.reps_collected)
    return kit


class _OrbitKitLocalMixin:
    """Assert the ladder pinned this seed's home-orbit kit to local_items."""

    needs_real_pre_fill = True  # options.local_items is set by apply_sphere_ladder

    def test_orbit_kit_marked_local(self):
        local = self.world.options.local_items.value
        self.assertTrue(
            local,
            "apply_sphere_ladder must mark this seed's home-orbit kit as "
            "local_items so a multiworld fill can't scatter the player's "
            "bootstrap into another world",
        )
        # It must be the per-seed kit, not a whole category — a category would
        # gut the multiworld (all pods/engines/tanks stuck local). The
        # home-orbit kit is a handful of parts; the smallest category
        # (control sources) is 27.
        self.assertLess(
            len(local), 15,
            f"local_items looks like a category, not the per-seed kit: "
            f"{sorted(local)}",
        )
        # Nothing reaches orbit without a control source, so the kit has one.
        self.assertTrue(
            any(_has_control_source(self.world, n) for n in local),
            f"orbit kit has no control source: {sorted(local)}",
        )

    def test_local_items_is_exactly_the_orbit_kit(self):
        expected = _ladder_orbit_kit(self.world) - {
            it.name for it in self.multiworld.precollected_items[self.player]
        }
        self.assertEqual(self.world.options.local_items.value, expected)


class TestOrbitKitLocalKerbinJool(_OrbitKitLocalMixin, KSP1TestBase):
    # Rover pre-fill off: it precollects a control source, which would drop out
    # of local_items (precollected != local). These tests verify the local_items
    # mechanism in isolation; the pre-fill's own bootstrap protection is separate.
    options = {"goal": "jool_moons_return", "starting_body": "kerbin",
               "guarantee_science_rover": 0}


class TestOrbitKitLocalKerbinSSR(_OrbitKitLocalMixin, KSP1TestBase):
    options = {"goal": "standard_sample_returns", "starting_body": "kerbin",
               "guarantee_science_rover": 0}


class TestOrbitKitLocalMunHome(_OrbitKitLocalMixin, KSP1TestBase):
    """Alien home: the kit is read off the ladder, not hardcoded to Kerbin."""
    options = {"goal": "jool_moons_return", "starting_body": "mun",
               "guarantee_science_rover": 0}


class TestOrbitKitForbiddenInOtherWorld(unittest.TestCase):
    """Two-player multiworld: a player's orbit-kit items must be rejected by
    every other player's location, so fill can only place them locally."""

    def test_orbit_kit_forbidden_across_worlds(self):
        from worlds.ksp1 import KSP1World
        opts = {"goal": "mun_flag", "starting_body": "kerbin"}
        # setup_multiworld runs through pre_fill; the ladder (in set_rules) has
        # already populated options.local_items for both players.
        mw = setup_multiworld([KSP1World, KSP1World],
                              options=[opts, opts], seed=1)
        # locality_rules compiles local_items into item rules (Main runs it
        # before pre_fill; the test harness does not, so invoke it here).
        locality_rules(mw)

        for owner, other in ((1, 2), (2, 1)):
            kit = mw.worlds[owner].options.local_items.value
            self.assertTrue(kit, f"player {owner}'s orbit kit should be marked local")
            owned = [it for it in mw.itempool
                     if it.player == owner and it.name in kit]
            self.assertTrue(owned, f"player {owner}'s kit items should be pooled")
            sample = owned[0]
            leaks = [l.name for l in mw.get_locations(other) if l.item_rule(sample)]
            self.assertEqual(
                leaks, [],
                f"player {owner}'s orbit-kit item {sample.name!r} must be "
                f"forbidden on all of player {other}'s locations; "
                f"{len(leaks)} accept it (e.g. {leaks[:3]})",
            )


if __name__ == "__main__":
    unittest.main()
