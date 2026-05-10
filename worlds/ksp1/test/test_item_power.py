"""
Tests for physics-based item power scoring and tier classification.

Post bug-074: parts in progressive groups have NO tier (their gating comes
from the progressive item itself). Only standalone parts and non-part items
carry a tier classification.
"""
import unittest

from worlds.ksp1.item_power import ITEM_TIERS
from worlds.ksp1.parts import PROGRESSIVE_PART_NAMES


class TestProgressiveGroupedPartsHaveNoTier(unittest.TestCase):
    """Parts inside progressive groups should not appear in ITEM_TIERS.

    Their placement is gated by the progressive item alone; the per-part
    tier system is redundant for them and was found to cause tail-cornering
    in remaining_fill (bug 074).
    """

    def test_progressive_grouped_parts_absent_from_item_tiers(self):
        for name in PROGRESSIVE_PART_NAMES:
            self.assertNotIn(
                name, ITEM_TIERS,
                f"{name} is in a progressive group but still has an explicit tier",
            )


class TestNonProgressivePartTiers(unittest.TestCase):
    """Tier assignments for items NOT in any progressive group."""

    def _tier(self, name: str) -> int:
        return ITEM_TIERS.get(name, 0)

    def test_strut_tier0(self):
        self.assertEqual(self._tier("strutConnector"), 0)

    def test_structural_wing_tier0(self):
        self.assertEqual(self._tier("structuralWing"), 0)

    def test_launch_clamp_has_tier(self):
        # Not in any progressive group; physics tier preserved.
        self.assertGreaterEqual(self._tier("launchClamp1"), 1)

    def test_docking_port_lateral_has_tier(self):
        # Reclassified to useful and excluded from progressive group.
        # Physics tier preserved as a placement gate.
        self.assertGreaterEqual(self._tier("dockingPortLateral"), 1)


class TestNonPartTiers(unittest.TestCase):
    """Tier assignments for non-part items (science packs, etc.)."""

    def _tier(self, name: str) -> int:
        return ITEM_TIERS.get(name, 0)

    def test_science_pack_50_tier1(self):
        self.assertEqual(self._tier("Science Pack 50"), 1)

    def test_science_pack_100_tier2(self):
        self.assertEqual(self._tier("Science Pack 100"), 2)

    def test_science_pack_250_tier2(self):
        self.assertEqual(self._tier("Science Pack 250"), 2)

    def test_progressive_rd_tier2(self):
        self.assertEqual(self._tier("Progressive R&D"), 2)


class TestTierCounts(unittest.TestCase):
    """Sanity check that tier distribution leaves enough unrestricted slots."""

    def test_tier2_not_majority(self):
        """Tier 2 must be < 50% of all items to avoid fill problems."""
        from worlds.ksp1.parts import PART_DB
        tier2_count = sum(1 for v in ITEM_TIERS.values() if v == 2)
        self.assertLess(
            tier2_count, len(PART_DB) * 0.5,
            f"Tier 2 has {tier2_count}/{len(PART_DB)} items — too many, will cause fill issues",
        )


class TestJsonSync(unittest.TestCase):
    """item_tiers.json must stay in sync with the computation in compute_item_tiers.py."""

    def test_json_matches_computation(self):
        """Regenerate tiers from PART_DB and verify they match the shipped JSON."""
        from worlds.ksp1.scripts.compute_item_tiers import compute_item_tiers
        from worlds.ksp1.item_power import _load_item_tiers
        computed = compute_item_tiers()
        self.assertEqual(
            _load_item_tiers(), computed,
            "data/item_tiers.json is stale — run: "
            "python -m worlds.ksp1.scripts.compute_item_tiers",
        )


if __name__ == "__main__":
    unittest.main()
