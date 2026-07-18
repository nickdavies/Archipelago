"""Tests for the hidden / unlockable celestial bodies feature.

Covers the generation side: which bodies hide under each BodyVisibilityMode,
the per-body region topology (planets off Menu, moons off their planet, home
always off Menu), the ``Discover`` gate on hidden-body locations and its
parent->child structure, the science-insufficient tech-tier planet-subset gate,
and the invariant that every Discover gate item stays PROGRESSION through the
sphere-ladder demote (the regression that silently made gated locations
unreachable).
"""
from __future__ import annotations

from BaseClasses import CollectionState, ItemClassification

from ..items import discover_item_name
from .base import KSP1TestBase


def _discover_names(world) -> set[str]:
    return {discover_item_name(b) for b in world.gated_hidden_bodies}


class TestFeatureOff(KSP1TestBase):
    """all_visible is a clean no-op: no hidden bodies, no Discover items, no
    tier gates — the world is the plain (feature-off) world."""
    options = {"body_visibility_mode": "all_visible", "goal": "duna_return"}

    def test_nothing_hidden(self):
        w = self.world
        self.assertEqual(len(w.hidden_bodies), 0)
        self.assertEqual(len(w.gated_hidden_bodies), 0)

    def test_no_discover_items_in_pool(self):
        disc = [i for i in self.multiworld.itempool if i.name.startswith("Discover ")]
        self.assertEqual(disc, [])

    def test_body_regions_ungated(self):
        # Every body region reachable from an empty state (no gate).
        state = CollectionState(self.multiworld)
        state.update_reachable_regions(self.player)
        for name in ("Duna", "Jool", "Eeloo"):
            self.assertTrue(state.can_reach_region(name, self.player),
                            f"{name} region should be ungated when all_visible")


class TestHomeSystemKerbin(KSP1TestBase):
    """Kerbin home, home_system: home system visible, everything else hidden."""
    options = {"body_visibility_mode": "home_system", "goal": "duna_return"}

    def test_home_system_visible_rest_hidden(self):
        w = self.world
        hidden = {str(b) for b in w.hidden_bodies}
        # Home system stays visible.
        for v in ("Kerbin", "Mun", "Minmus"):
            self.assertNotIn(v, hidden)
        # Interplanetary bodies hidden.
        for h in ("Moho", "Eve", "Duna", "Jool", "Eeloo", "Laythe"):
            self.assertIn(h, hidden)

    def test_home_and_sun_never_hidden(self):
        hidden = {str(b) for b in self.world.hidden_bodies}
        self.assertNotIn(str(self.world.mission_builder.home), hidden)
        self.assertNotIn("Sun", hidden)

    def test_discover_items_pooled_one_per_gated_body(self):
        pool = {i.name for i in self.multiworld.itempool if i.name.startswith("Discover ")}
        self.assertEqual(pool, _discover_names(self.world))
        self.assertTrue(pool, "expected Discover items when bodies are hidden")

    def test_hidden_mission_needs_discover(self):
        # With all items EXCEPT Discover Duna, Duna's mission is unreachable;
        # adding Discover Duna opens it.
        w = self.world
        all_disc = _discover_names(w)

        def state(exclude):
            s = CollectionState(self.multiworld)
            for it in self.multiworld.precollected_items[self.player]:
                s.collect(it, prevent_sweep=True)
            for it in self.multiworld.itempool:
                if it.name in exclude:
                    continue
                s.collect(it, prevent_sweep=True)
            s.update_reachable_regions(self.player)
            return s

        without = state(all_disc)
        self.assertFalse(self.multiworld.get_location("Duna Orbit 1", self.player)
                         .can_reach(without))
        with_duna = state(all_disc - {"Discover Duna"})
        self.assertTrue(self.multiworld.get_location("Duna Orbit 1", self.player)
                        .can_reach(with_duna))

    def test_moon_requires_parent_discover(self):
        # Ike (moon of Duna) needs Discover Duna (parent) AND Discover Ike.
        w = self.world

        def state(extra):
            s = CollectionState(self.multiworld)
            for it in self.multiworld.precollected_items[self.player]:
                s.collect(it, prevent_sweep=True)
            for it in self.multiworld.itempool:
                if it.name.startswith("Discover ") and it.name not in extra:
                    continue
                s.collect(it, prevent_sweep=True)
            s.update_reachable_regions(self.player)
            return s

        ike = self.multiworld.get_location("Ike Orbit 1", self.player)
        self.assertFalse(ike.can_reach(state({"Discover Ike"})),
                         "Ike needs its parent Duna discovered too")
        self.assertTrue(ike.can_reach(state({"Discover Ike", "Discover Duna"})))


class TestHomeSystemTopology(KSP1TestBase):
    """Region topology: planets off Menu, moons off their planet, home off Menu."""
    options = {"body_visibility_mode": "home_system", "goal": "duna_return"}

    def test_moon_region_hangs_off_planet(self):
        for moon, planet in (("Mun", "Kerbin"), ("Ike", "Duna"),
                             ("Laythe", "Jool")):
            region = self.multiworld.get_region(moon, self.player)
            parents = {e.parent_region.name for e in region.entrances}
            self.assertEqual(parents, {planet},
                             f"{moon} should connect from {planet}")

    def test_planet_region_off_menu(self):
        for planet in ("Duna", "Jool", "Eve"):
            region = self.multiworld.get_region(planet, self.player)
            parents = {e.parent_region.name for e in region.entrances}
            self.assertEqual(parents, {"Menu"})

    def test_home_region_off_menu(self):
        home = str(self.world.mission_builder.home)
        region = self.multiworld.get_region(home, self.player)
        parents = {e.parent_region.name for e in region.entrances}
        self.assertEqual(parents, {"Menu"})


class TestHomeOnlyMoonHome(KSP1TestBase):
    """Laythe home, home_only: only Laythe visible; its parent Jool and sibling
    moons are hidden, yet the home region stays reachable from Menu."""
    options = {"body_visibility_mode": "home_only", "starting_body": "laythe",
               "goal": "mun_flag"}

    def test_only_home_visible(self):
        w = self.world
        hidden = {str(b) for b in w.hidden_bodies}
        self.assertNotIn("Laythe", hidden)
        # Parent planet and siblings hidden under home_only.
        for h in ("Jool", "Vall", "Tylo", "Kerbin", "Mun"):
            self.assertIn(h, hidden)

    def test_home_reachable_without_discovering_parent(self):
        # Laythe home must be reachable from an empty state even though its
        # parent Jool is hidden/gated.
        state = CollectionState(self.multiworld)
        state.update_reachable_regions(self.player)
        self.assertTrue(state.can_reach_region("Laythe", self.player))
        self.assertFalse(state.can_reach_region("Jool", self.player))


class TestDiscoverGateItemsStayProgression(KSP1TestBase):
    """Regression: every Discover gate item must remain PROGRESSION after the
    sphere-ladder demote, else AP's advancement sweep skips it and the gated
    locations are unreachable (the 2821-subtest-failure bug)."""
    options = {"body_visibility_mode": "home_system", "goal": "duna_return"}
    needs_real_pre_fill = True

    def test_discover_items_progression_after_demote(self):
        want = _discover_names(self.world)
        by_name = {i.name: i for i in self.multiworld.itempool
                   if i.name.startswith("Discover ")}
        self.assertEqual(set(by_name), want)
        for name, item in by_name.items():
            self.assertTrue(item.classification & ItemClassification.progression,
                            f"{name} demoted below PROGRESSION")

    def test_all_discover_collected_by_all_state(self):
        state = self.multiworld.get_all_state(False)
        for name in _discover_names(self.world):
            self.assertTrue(state.has(name, self.player),
                            f"{name} not collected by get_all_state")


class TestHiddenScienceRequiresDiscovery(KSP1TestBase):
    """The tech tree is gated on science, and a hidden body's science is gated on
    its Discover item (the accurate model that replaced the explicit tech-tier
    gate).  In home_only Kerbin the whole system is hidden, so science banked
    without any Discover item is far lower than with them — the tree can't be
    funded until the player discovers bodies."""
    options = {"body_visibility_mode": "home_only", "goal": "complete_tech_tree"}
    needs_real_pre_fill = True

    def _science(self, drop_discover: bool) -> float:
        from ..rules import _accessible_science, effective_science_safety
        state = CollectionState(self.multiworld)
        for it in self.multiworld.precollected_items[self.player]:
            state.collect(it, prevent_sweep=True)
        for it in self.multiworld.itempool:
            if drop_discover and it.name.startswith("Discover "):
                continue
            state.collect(it, prevent_sweep=True)
        state.update_reachable_regions(self.player)
        safety = effective_science_safety(
            self.world.options, self.world.options.difficulty.value)
        return _accessible_science(
            state, self.player, safety, self.world.mission_builder.home)

    def test_discovery_multiplies_bankable_science(self):
        with_disc = self._science(drop_discover=False)
        without_disc = self._science(drop_discover=True)
        self.assertGreater(
            with_disc, without_disc * 5,
            f"discovery should unlock the hidden system's science "
            f"({round(without_disc)} without -> {round(with_disc)} with)")


class TestExplainDiscoveryGate(KSP1TestBase):
    """`/explain` surfaces the hidden-body Discover gate as the headline reason
    when a location's body isn't reachable yet (bug 4)."""
    options = {"body_visibility_mode": "home_system", "goal": "duna_return"}

    def _state(self, discovered=()):
        s = CollectionState(self.multiworld)
        for it in self.multiworld.precollected_items[self.player]:
            s.collect(it, prevent_sweep=True)
        for it in self.multiworld.itempool:
            if it.name.startswith("Discover ") and it.name not in discovered:
                continue
            s.collect(it, prevent_sweep=True)
        s.update_reachable_regions(self.player)
        return s

    def _explain(self, loc_name, state) -> str:
        out = self.world.explain_rule(loc_name, state)
        return "\n".join(seg["text"] for seg in out)

    def test_undiscovered_body_is_headline_reason(self):
        txt = self._explain("Duna Orbit 1", self._state())
        self.assertIn("Body discovered: NO", txt)
        self.assertIn("cannot reach Duna region", txt)
        self.assertIn("Discover Duna", txt)

    def test_discovered_body_reported_yes(self):
        txt = self._explain("Duna Orbit 1", self._state(discovered={"Discover Duna"}))
        self.assertIn("Body discovered: YES", txt)

    def test_moon_reports_topmost_undiscovered_ancestor(self):
        # Ike discovered but its parent Duna not: the first region you can't
        # reach is Duna (the shallowest gate), not Ike itself.
        txt = self._explain("Ike Orbit 1", self._state(discovered={"Discover Ike"}))
        self.assertIn("cannot reach Duna region", txt)
        self.assertNotIn("cannot reach Ike region", txt)


class TestExplainDiscoveryGateFeatureOff(KSP1TestBase):
    """Feature off: /explain output carries no discovery section (unchanged)."""
    options = {"body_visibility_mode": "all_visible", "goal": "duna_return"}

    def test_no_discovery_line_when_nothing_gated(self):
        s = CollectionState(self.multiworld)
        s.update_reachable_regions(self.player)
        out = self.world.explain_rule("Duna Orbit 1", s)
        txt = "\n".join(seg["text"] for seg in out)
        self.assertNotIn("Body discovered", txt)


class TestFeatureOnFullReachability(KSP1TestBase):
    """End-to-end: with the feature ON, WorldTestBase's default suite (fill,
    accessibility=full reachability, beatability) must pass — the strongest
    guard that Discover gating stays consistent through a real fill."""
    run_default_tests = True
    needs_real_pre_fill = True
    options = {"body_visibility_mode": "home_system", "goal": "duna_return"}


class TestFeatureOnTechTreeReachability(KSP1TestBase):
    """Same, for complete_tech_tree — exercises the science-gated-on-discovery
    tech tree through a real fill."""
    run_default_tests = True
    needs_real_pre_fill = True
    options = {"body_visibility_mode": "home_system", "goal": "complete_tech_tree"}
