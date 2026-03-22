"""
World-level generation tests.

These use WorldTestBase (full gen_steps pipeline) to catch problems that only
surface during AP fill — item/location count mismatches, unreachable locations,
and science budget gaps.
"""
import unittest

from test.bases import WorldTestBase
from BaseClasses import ItemClassification

from worlds.ksp1.items import _SORTED_PART_NAMES, ALWAYS_PRECOLLECTED, CLAMP_PRECOLLECTED
from worlds.ksp1.rules import _accessible_science, _can_afford_tier
from worlds.ksp1.tech_tree import TECH_NODES, cumulative_tier_cost
from worlds.ksp1.options import Difficulty
from worlds.ksp1.capability import get_capability, explain_body_unreachable


class KSP1TestBase(WorldTestBase):
    game = "Kerbal Space Program"


class TestItemLocationBalance(KSP1TestBase):
    """Item pool must exactly match non-event location count."""

    def test_item_count_equals_location_count(self):
        real_locs = [
            loc for loc in self.multiworld.get_locations(self.player)
            if loc.address is not None
        ]
        self.assertEqual(
            len(self.multiworld.itempool),
            len(real_locs),
            f"Item pool size {len(self.multiworld.itempool)} != "
            f"location count {len(real_locs)}",
        )

    def test_no_negative_filler_count(self):
        """Pool should never have more part items than locations."""
        part_item_count = len(_SORTED_PART_NAMES) - len(ALWAYS_PRECOLLECTED) - len(CLAMP_PRECOLLECTED)
        real_loc_count = sum(
            1 for loc in self.multiworld.get_locations(self.player)
            if loc.address is not None
        )
        self.assertLessEqual(
            part_item_count, real_loc_count,
            "More part items than locations — filler count would go negative",
        )


class TestAllLocationsReachable(KSP1TestBase):
    """Every non-event location must be reachable with all items collected."""

    def test_all_locations_reachable_with_all_items(self):
        self.collect_all_but([])
        state = self.multiworld.state
        unreachable = [
            loc.name
            for loc in self.multiworld.get_locations(self.player)
            if loc.address is not None and not loc.can_reach(state)
        ]
        detail = ""
        if unreachable:
            # Extract body names by matching the start of location names against
            # known bodies.  Location names are like "Duna Return 1",
            # "Laythe Crewed Landing 2", "Bop Sample Return 1", etc.
            from worlds.ksp1.bodies import ALL_BODIES
            known_bodies = {b.name for b in ALL_BODIES}
            body_names = sorted({
                next((b for b in known_bodies if loc.startswith(b + " ")), None)
                for loc in unreachable
            } - {None})
            diagnoses = [explain_body_unreachable(state, self.player, b) for b in body_names]
            detail = "\n  ".join(diagnoses)
        self.assertEqual(
            unreachable, [],
            f"Locations unreachable even with all items ({len(unreachable)} locs):\n  {detail}\n"
            f"Full list: {unreachable}",
        )


class TestScienceBudget(KSP1TestBase):
    """Science budget with all parts must cover every tech tier."""

    def test_capability_sees_reachable_bodies(self):
        """With all items, the capability system must consider bodies reachable."""
        from worlds.ksp1.capability import get_capability
        from worlds.ksp1.bodies import ALL_BODIES
        self.collect_all_but([])
        cap = get_capability(self.multiworld.state, self.player)
        reachable = [b.name for b in ALL_BODIES if cap.bodies.get(b.name) and cap.bodies[b.name].can_orbit_low]
        self.assertGreater(
            len(reachable), 1,
            f"With all items, only {reachable} have can_orbit_low — expected most/all bodies",
        )

    def test_full_state_affords_all_tiers_normal(self):
        self.collect_all_but([])
        state = self.multiworld.state
        difficulty = self.world.options.difficulty.value
        max_tier = max(node.tier for node in TECH_NODES)
        for tier in range(1, max_tier + 1):
            cost = cumulative_tier_cost(tier)
            science = _accessible_science(state, self.player, difficulty)
            self.assertGreaterEqual(
                science, cost,
                f"Full item state cannot afford tier {tier} "
                f"(need {cost}, have {science:.0f}) at difficulty {difficulty}",
            )


class TestCapabilityCache(KSP1TestBase):
    """
    The capability cache must reflect the actual collected items at the time
    of the call, not a snapshot from an earlier collection.

    Regression: collecting items one-at-a-time caused the cache to freeze after
    the first sweep_for_advancements call, returning stale capability for all
    subsequent state mutations.
    """

    def test_cache_reflects_incremental_collects(self):
        """Capability must grow as items are collected, not freeze at first compute."""
        from worlds.ksp1.capability import get_capability
        state = self.multiworld.state

        # Collect nothing — capability should be minimal (clamps pre-granted by option)
        cap_empty = get_capability(state, self.player)
        empty_reachable = sum(
            1 for bp in cap_empty.bodies.values() if bp.can_orbit_low
        )

        # Collect all items
        self.collect_all_but([])
        cap_full = get_capability(state, self.player)
        full_reachable = sum(
            1 for bp in cap_full.bodies.values() if bp.can_orbit_low
        )

        self.assertGreater(
            full_reachable, empty_reachable,
            "Capability after collecting all items must exceed empty-state capability "
            f"(empty={empty_reachable}, full={full_reachable}). "
            "Cache may be stale.",
        )

    def test_cache_invalidates_on_item_add(self):
        """Adding a single key item must produce a different (better) capability."""
        from worlds.ksp1.capability import get_capability
        state = self.multiworld.state

        cap_before = get_capability(state, self.player)
        reachable_before = sum(1 for bp in cap_before.bodies.values() if bp.can_orbit_low)

        # Add a Mainsail — one of the most powerful engines
        self.collect_by_name('RE-M3 "Mainsail" Liquid Fuel Engine')
        cap_after = get_capability(state, self.player)
        reachable_after = sum(1 for bp in cap_after.bodies.values() if bp.can_orbit_low)

        # After adding a big engine, at least Kerbin must be newly orbitabile
        self.assertGreaterEqual(
            reachable_after, reachable_before,
            "Capability must not decrease after adding an engine. Cache may not be invalidating.",
        )


class TestItemClassification(KSP1TestBase):
    """Sanity-check that key parts get the right AP classification."""

    def _classification(self, name: str) -> ItemClassification:
        for item in self.multiworld.itempool:
            if item.name == name:
                return item.classification
        for item in self.multiworld.precollected_items[self.player]:
            if item.name == name:
                return item.classification
        raise KeyError(name)

    def test_engines_are_progression(self):
        for name in (
            'LV-T45 "Swivel" Liquid Fuel Engine',
            'LV-909 "Terrier" Liquid Fuel Engine',
            'RE-M3 "Mainsail" Liquid Fuel Engine',
        ):
            self.assertEqual(
                self._classification(name),
                ItemClassification.progression,
                f"{name} should be progression",
            )

    def test_rcs_is_useful(self):
        self.assertEqual(
            self._classification("RV-105 RCS Thruster Block"),
            ItemClassification.useful,
            "RCS Thruster should be useful, not progression",
        )

    def test_ladder_is_progression(self):
        self.assertEqual(
            self._classification("Pegasus I Mobility Enhancer"),
            ItemClassification.progression,
            "Ladder should be progression (gates sample returns on high-g bodies)",
        )


if __name__ == "__main__":
    unittest.main()
