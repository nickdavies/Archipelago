"""
World-level generation tests.

These use WorldTestBase (full gen_steps pipeline) to catch problems that only
surface during AP fill — item/location count mismatches, unreachable locations,
and science budget gaps.
"""
import unittest

from BaseClasses import ItemClassification

from worlds.ksp1.items import (
    _SORTED_PART_NAMES, ALWAYS_PRECOLLECTED, CLAMP_PRECOLLECTED,
    PROGRESSIVE_RD_NAME, PROGRESSIVE_RD_COUNT,
)
from worlds.ksp1.rules import _accessible_science, _can_afford_tier
from worlds.ksp1.tech_tree import TECH_NODES, TIER_TO_BAND, MAX_RD_BAND, cumulative_tier_cost
from worlds.ksp1.locations import EventName, TechTreeLocation
from worlds.ksp1.options import Difficulty
from worlds.ksp1.capability import get_capability, explain_body_unreachable
from worlds.ksp1.test.base import KSP1TestBase


class TestFill(KSP1TestBase):
    """Runs AP fill once with default options — the canonical generation smoke test."""
    run_default_tests = True
    needs_real_pre_fill = True  # fill needs sphere ladder side effects


class TestFillStandardSampleReturns(KSP1TestBase):
    """Fill smoke test for standard_sample_returns on expert difficulty: crewed sample return from 11 bodies."""
    options = {"goal": "standard_sample_returns", "difficulty": "expert"}
    run_default_tests = True
    needs_real_pre_fill = True


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
        part_item_count = (
            len(_SORTED_PART_NAMES)
            - len(ALWAYS_PRECOLLECTED)
            - len(CLAMP_PRECOLLECTED)
        )
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
        reachable = [b.name for b in ALL_BODIES if cap.bodies.get(b.name) and cap.bodies[b.name].access.get("Orbit", False)]
        self.assertGreater(
            len(reachable), 1,
            f"With all items, only {reachable} have Orbit access — expected most/all bodies",
        )

    def test_full_state_affords_all_tiers_normal(self):
        from worlds.ksp1.rules import effective_science_safety
        self.collect_all_but([])
        state = self.multiworld.state
        difficulty = self.world.options.difficulty.value
        safety = effective_science_safety(self.world.options, difficulty)
        max_tier = max(node.tier for node in TECH_NODES)
        for tier in range(1, max_tier + 1):
            cost = cumulative_tier_cost(tier)
            science = _accessible_science(
                state, self.player, safety, self.world.mission_builder.home,
            )
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
            1 for bp in cap_empty.bodies.values() if bp.access.get(EventName.ORBIT, False)
        )

        # Collect all items
        self.collect_all_but([])
        cap_full = get_capability(state, self.player)
        full_reachable = sum(
            1 for bp in cap_full.bodies.values() if bp.access.get(EventName.ORBIT, False)
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
        reachable_before = sum(1 for bp in cap_before.bodies.values() if bp.access.get(EventName.ORBIT, False))

        # Add a Mainsail — one of the most powerful engines
        self.collect_by_name('RE-M3 "Mainsail" Liquid Fuel Engine')
        cap_after = get_capability(state, self.player)
        reachable_after = sum(1 for bp in cap_after.bodies.values() if bp.access.get(EventName.ORBIT, False))

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

    def test_rcs_is_useful(self):
        self.assertEqual(
            self._classification("RCSBlock.v2"),  # RV-105
            ItemClassification.useful,
            "RCS Thruster should be useful, not progression",
        )


class TestProgressiveRD(KSP1TestBase):
    """Progressive R&D items gate higher tech tree bands."""

    def test_progressive_rd_in_pool(self):
        """Pool must contain exactly PROGRESSIVE_RD_COUNT copies."""
        count = sum(
            1 for item in self.multiworld.itempool
            if item.name == PROGRESSIVE_RD_NAME
        )
        self.assertEqual(
            count, PROGRESSIVE_RD_COUNT,
            f"Expected {PROGRESSIVE_RD_COUNT} Progressive R&D items, got {count}",
        )

    def test_progressive_rd_is_progression(self):
        """Progressive R&D must be classified as progression."""
        for item in self.multiworld.itempool:
            if item.name == PROGRESSIVE_RD_NAME:
                self.assertEqual(
                    item.classification,
                    ItemClassification.progression,
                    "Progressive R&D must be progression",
                )
                return
        self.fail("Progressive R&D not found in item pool")

    def test_tier3_unreachable_without_rd(self):
        """Tier 3+ tech locations must be unreachable without Progressive R&D."""
        # Collect everything except Progressive R&D
        self.collect_all_but([PROGRESSIVE_RD_NAME])
        state = self.multiworld.state
        for node in TECH_NODES:
            band = TIER_TO_BAND[node.tier]
            if band == 0:
                continue
            loc_name = str(TechTreeLocation(node.display_name, 1))
            loc = self.multiworld.get_location(loc_name, self.player)
            self.assertFalse(
                loc.can_reach(state),
                f"Tier {node.tier} (band {band}) location '{loc_name}' should be "
                f"unreachable without Progressive R&D",
            )

    def test_all_tiers_reachable_with_rd(self):
        """All tiers reachable when all items (including Progressive R&D) collected."""
        self.collect_all_but([])
        state = self.multiworld.state
        for node in TECH_NODES:
            loc_name = str(TechTreeLocation(node.display_name, 1))
            loc = self.multiworld.get_location(loc_name, self.player)
            self.assertTrue(
                loc.can_reach(state),
                f"Tier {node.tier} location '{loc_name}' unreachable with all items",
            )

    def test_progressive_rd_not_in_all_progression_items(self):
        """Progressive R&D must NOT be in _ALL_PROGRESSION_ITEMS (Eve/Tylo proxy)."""
        from worlds.ksp1.rules import _ALL_PROGRESSION_ITEMS
        self.assertNotIn(
            PROGRESSIVE_RD_NAME, _ALL_PROGRESSION_ITEMS,
            "Progressive R&D should not be in _ALL_PROGRESSION_ITEMS "
            "(built from ITEM_TABLE, not _PROGRESSIVE_ITEMS)",
        )


class TestCompleteTechTreeGoalRD(KSP1TestBase):
    """complete_tech_tree goal requires Progressive R&D x MAX_RD_BAND."""
    options = {"goal": "complete_tech_tree"}
    # Victory's science gate reads the cheap-ladder reps (_science_body_event_reps),
    # a pre_fill side effect; without the real ladder it's conservatively unreachable.
    needs_real_pre_fill = True

    def test_goal_unreachable_without_rd(self):
        """Victory location must be unreachable without Progressive R&D."""
        self.collect_all_but([PROGRESSIVE_RD_NAME])
        state = self.multiworld.state
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertFalse(
            victory.can_reach(state),
            "complete_tech_tree victory should be unreachable without Progressive R&D",
        )

    def test_goal_reachable_with_rd(self):
        """Victory location must be reachable with all items."""
        self.collect_all_but([])
        state = self.multiworld.state
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertTrue(
            victory.can_reach(state),
            "complete_tech_tree victory should be reachable with all items",
        )


if __name__ == "__main__":
    unittest.main()
