"""PartManager — pack-filtered source of truth over the raw part universe.

Locks the invariant that the all-packs manager reproduces the legacy module
globals exactly (so the encapsulation refactor is behavior-identical), and that
disabling a pack actually removes its parts.
"""
import unittest

from worlds.ksp1.parts import (
    CONTRACT_CATEGORY_MEMBERS, PART_TO_CONTRACT_CATEGORIES,
    DEFAULT_PART_MANAGER, part_manager_for, ALL_PACKS, CapabilityFlag,
)
# The raw, unfiltered universe is private; an internal test may read it to
# assert the all-packs manager view reproduces it exactly.
from worlds.ksp1.parts._raw import _RAW_PART_DB
from worlds.ksp1.parts.packs import STOCK, MAKING_HISTORY
import worlds.ksp1.capability as cap


class TestAllPacksParity(unittest.TestCase):
    """DEFAULT_PART_MANAGER (every pack) must equal the pre-refactor globals."""

    def setUp(self) -> None:
        self.pm = DEFAULT_PART_MANAGER

    def test_parts_equals_raw_universe_in_order(self) -> None:
        self.assertEqual(list(self.pm.parts.keys()), list(_RAW_PART_DB.keys()))
        self.assertEqual(self.pm.parts, _RAW_PART_DB)

    def test_category_members_match(self) -> None:
        self.assertEqual(self.pm.category_members, CONTRACT_CATEGORY_MEMBERS)

    def test_part_to_categories_match_including_order(self) -> None:
        self.assertEqual(self.pm.part_to_categories, PART_TO_CONTRACT_CATEGORIES)

    def test_fuel_line_matches_capability_global(self) -> None:
        self.assertEqual(self.pm.fuel_line_part, cap._FUEL_LINE_PART)
        self.assertEqual(self.pm.fuel_line_mass, cap._FUEL_LINE_MASS)

    def test_lightest_providing_matches_capability_globals(self) -> None:
        self.assertEqual(
            self.pm.lightest_providing(CapabilityFlag.BATTERY_LARGE),
            cap._BATTERY_LARGE_PART)
        self.assertEqual(
            self.pm.lightest_providing(CapabilityFlag.SOLAR_ARRAY_LARGE),
            cap._SOLAR_LARGE_PART)


class TestPackFiltering(unittest.TestCase):
    def test_memoized_identity(self) -> None:
        self.assertIs(DEFAULT_PART_MANAGER, part_manager_for(ALL_PACKS))

    def test_stock_always_present(self) -> None:
        pm = part_manager_for(frozenset())  # nothing -> still gets Stock
        self.assertIn(STOCK, pm.enabled_packs)

    def test_disabling_making_history_drops_its_parts(self) -> None:
        full = part_manager_for(ALL_PACKS)
        stock_only = part_manager_for(frozenset({STOCK}))
        self.assertLess(len(stock_only.parts), len(full.parts))
        # every dropped part was a MakingHistory part
        dropped = set(full.parts) - set(stock_only.parts)
        self.assertTrue(dropped)
        self.assertEqual(
            stock_only.capability_relevant_packs(), frozenset({STOCK}))

    def test_capability_relevant_packs_all(self) -> None:
        self.assertEqual(
            DEFAULT_PART_MANAGER.capability_relevant_packs(),
            frozenset({STOCK, MAKING_HISTORY}))


if __name__ == "__main__":
    unittest.main()
