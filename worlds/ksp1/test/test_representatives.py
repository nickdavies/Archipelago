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
    game = "Kerbal Space Program 1"


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

    def test_progressive_items_in_pool_account_for_precollects(self):
        """Pool count for each progressive = total count minus precollected copies."""
        precollected = [
            i.name for i in self.multiworld.precollected_items[self.player]
        ]
        for prog_name, count in PROGRESSIVE_PART_COUNTS.items():
            n_precollected = sum(1 for n in precollected if n == prog_name)
            expected = count - n_precollected
            pool_count = sum(
                1 for item in self.multiworld.itempool
                if item.name == prog_name
            )
            self.assertEqual(
                pool_count, expected,
                f"Expected {expected} copies of {prog_name} in pool "
                f"(total {count} minus {n_precollected} precollected), got {pool_count}",
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


if __name__ == "__main__":
    unittest.main()
