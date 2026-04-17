"""Tests for progressive part representative selection and pacing (bug 056)."""
import unittest

from test.bases import WorldTestBase
from BaseClasses import ItemClassification

from worlds.ksp1.items import PROGRESSIVE_PART_ITEM_NAMES
from worlds.ksp1.item_power import ITEM_TIERS
from worlds.ksp1.parts import (
    PROGRESSIVE_PART_TIERS, PROGRESSIVE_PART_NAMES, PROGRESSIVE_PART_COUNTS,
)


class KSP1TestBase(WorldTestBase):
    game = "Kerbal Space Program"


class TestRepresentativeSelection(KSP1TestBase):
    """Representatives are selected and removed from the pool."""

    def test_representatives_stored_on_world(self):
        """world.progressive_representatives must exist with correct structure."""
        reps = self.world.progressive_representatives
        self.assertIsInstance(reps, dict)
        for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
            self.assertIn(prog_name, reps, f"Missing representatives for {prog_name}")
            for tier_num in tiers:
                self.assertIn(
                    tier_num, reps[prog_name],
                    f"Missing tier {tier_num} representative for {prog_name}",
                )
                rep = reps[prog_name][tier_num]
                self.assertIn(
                    rep, tiers[tier_num],
                    f"Representative {rep!r} not in tier {tier_num} parts for {prog_name}",
                )

    def test_representatives_not_in_pool(self):
        """Representative parts must NOT appear as individual items in the pool."""
        reps = self.world.progressive_representatives
        all_reps = {
            rep
            for tier_reps in reps.values()
            for rep in tier_reps.values()
        }
        pool_names = {item.name for item in self.multiworld.itempool}
        in_pool = all_reps & pool_names
        self.assertEqual(
            in_pool, set(),
            f"Representatives should not be in pool: {in_pool}",
        )

    def test_non_representative_absorbed_parts_in_pool(self):
        """Non-representative absorbed parts must be in the pool as useful."""
        reps = self.world.progressive_representatives
        all_reps = {
            rep
            for tier_reps in reps.values()
            for rep in tier_reps.values()
        }
        pool_names = {item.name for item in self.multiworld.itempool}
        for ksp_name in PROGRESSIVE_PART_NAMES:
            if ksp_name in all_reps:
                continue
            self.assertIn(
                ksp_name, pool_names,
                f"Non-representative absorbed part {ksp_name!r} missing from pool",
            )

    def test_non_representative_absorbed_parts_are_useful(self):
        """Non-representative absorbed parts must be classified as useful."""
        reps = self.world.progressive_representatives
        all_reps = {
            rep
            for tier_reps in reps.values()
            for rep in tier_reps.values()
        }
        for item in self.multiworld.itempool:
            if item.name in PROGRESSIVE_PART_NAMES and item.name not in all_reps:
                self.assertEqual(
                    item.classification,
                    ItemClassification.useful,
                    f"Absorbed part {item.name!r} should be useful, got {item.classification}",
                )

    def test_progressive_items_still_in_pool(self):
        """Progressive items must still appear in pool with correct counts."""
        for prog_name, count in PROGRESSIVE_PART_COUNTS.items():
            pool_count = sum(
                1 for item in self.multiworld.itempool
                if item.name == prog_name
            )
            self.assertEqual(
                pool_count, count,
                f"Expected {count} copies of {prog_name}, got {pool_count}",
            )

    def test_representative_count_equals_progressive_copies(self):
        """Total representatives equals sum of progressive item copies."""
        reps = self.world.progressive_representatives
        total_reps = sum(len(tier_reps) for tier_reps in reps.values())
        total_copies = sum(PROGRESSIVE_PART_COUNTS.values())
        self.assertEqual(
            total_reps, total_copies,
            f"Representative count {total_reps} != progressive copies {total_copies}",
        )


class TestProgressiveTierFloors(unittest.TestCase):
    """Progressive T2+ parts get power tier floors to prevent early placement."""

    def test_t2_parts_have_tier_at_least_1(self):
        """All Progressive T2 parts must have power tier >= 1."""
        for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
            for ksp_name in tiers.get(2, []):
                tier = ITEM_TIERS.get(ksp_name, 0)
                self.assertGreaterEqual(
                    tier, 1,
                    f"{ksp_name} (Progressive T2 in {prog_name}) has tier {tier}, expected >= 1",
                )

    def test_t3_parts_have_tier_at_least_2(self):
        """All Progressive T3+ parts must have power tier >= 2."""
        for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
            for tier_num, parts in tiers.items():
                if tier_num < 3:
                    continue
                for ksp_name in parts:
                    tier = ITEM_TIERS.get(ksp_name, 0)
                    self.assertGreaterEqual(
                        tier, 2,
                        f"{ksp_name} (Progressive T{tier_num} in {prog_name}) "
                        f"has tier {tier}, expected >= 2",
                    )

    def test_t1_parts_keep_physics_tier(self):
        """Progressive T1 parts should not have their tier elevated by floors."""
        from worlds.ksp1.item_power import _PROGRESSIVE_TIER_FLOORS
        for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
            for ksp_name in tiers.get(1, []):
                self.assertNotIn(
                    ksp_name, _PROGRESSIVE_TIER_FLOORS,
                    f"T1 part {ksp_name} should not have a progressive tier floor",
                )

    def test_gigantor_is_tier2(self):
        """Gigantor (Progressive Solar T3) must be tier 2."""
        self.assertEqual(ITEM_TIERS.get("largeSolarPanel", 0), 2)

    def test_d25_is_at_least_tier1(self):
        """TD-25 decoupler (Progressive Stack Decoupler T2) must be tier >= 1."""
        self.assertGreaterEqual(ITEM_TIERS.get("Decoupler.2", 0), 1)


if __name__ == "__main__":
    unittest.main()
