"""Tests for physics-based item power scoring and tier classification."""
import unittest

from worlds.ksp1.item_power import ITEM_TIERS


class TestItemTierAssignments(unittest.TestCase):
    """Known engines/items land in expected tiers."""

    def setUp(self):
        self.tiers = ITEM_TIERS

    def _tier(self, name: str) -> int:
        return self.tiers.get(name, 0)

    # -- Tier 2: big engines, staging enablers, large tanks ----------------

    def test_mammoth_tier2(self):
        self.assertEqual(self._tier("Size3EngineCluster"), 2)

    def test_mainsail_tier2(self):
        self.assertEqual(self._tier("liquidEngineMainsail.v2"), 2)

    def test_vector_tier2(self):
        self.assertEqual(self._tier("SSME"), 2)

    def test_nerv_tier2(self):
        self.assertEqual(self._tier("nuclearEngine"), 2)

    def test_ion_tier2(self):
        self.assertEqual(self._tier("ionEngine"), 2)

    def test_wolfhound_tier2(self):
        self.assertEqual(self._tier("LiquidEngineRE-J10"), 2)

    def test_stack_decoupler_tier2(self):
        self.assertEqual(self._tier("Decoupler.1"), 2)

    def test_radial_decoupler_tier2(self):
        self.assertEqual(self._tier("radialDecoupler"), 2)

    def test_docking_port_tier2(self):
        self.assertEqual(self._tier("dockingPort2"), 2)

    def test_fuel_line_tier2(self):
        self.assertEqual(self._tier("fuelLine"), 2)

    def test_jumbo64_tier2(self):
        self.assertEqual(self._tier("Rockomax64.BW"), 2)

    # -- Tier 1: orbital-capable engines, medium items ---------------------

    def test_reliant_tier1(self):
        self.assertEqual(self._tier("liquidEngine.v2"), 1)

    def test_swivel_tier1(self):
        self.assertEqual(self._tier("liquidEngine2.v2"), 1)

    def test_thumper_tier1(self):
        self.assertEqual(self._tier("solidBooster1-1"), 1)

    def test_heat_shield_tier1(self):
        self.assertEqual(self._tier("HeatShield1"), 1)

    def test_launch_clamp_tier1(self):
        self.assertEqual(self._tier("launchClamp1"), 1)

    # -- Tier 0: starter items, vacuum-only engines ------------------------

    def test_flea_tier0(self):
        self.assertEqual(self._tier("solidBooster.sm.v2"), 0)

    def test_hammer_tier0(self):
        self.assertEqual(self._tier("solidBooster.v2"), 0)

    def test_terrier_tier0(self):
        self.assertEqual(self._tier("liquidEngine3.v2"), 0)

    def test_spark_tier0(self):
        self.assertEqual(self._tier("liquidEngineMini.v2"), 0)

    def test_capsule_tier0(self):
        self.assertEqual(self._tier("mk1pod.v2"), 0)

    def test_parachute_tier0(self):
        self.assertEqual(self._tier("parachuteSingle"), 0)

    def test_landing_leg_tier0(self):
        self.assertEqual(self._tier("landingLeg1"), 0)

    def test_probe_core_tier0(self):
        self.assertEqual(self._tier("probeCoreOcto.v2"), 0)

    # -- Structural: non-progression items should be tier 0 ----------------

    def test_structural_wing_tier0(self):
        self.assertEqual(self._tier("structuralWing"), 0)

    def test_strut_tier0(self):
        self.assertEqual(self._tier("strutConnector"), 0)


class TestTierCounts(unittest.TestCase):
    """Sanity check that tier distribution leaves enough unrestricted slots."""

    def test_tier2_not_majority(self):
        """Tier 2 must be < 50% of all items to avoid fill problems."""
        from worlds.ksp1.parts import PART_DB
        tiers = ITEM_TIERS
        tier2_count = sum(1 for v in tiers.values() if v == 2)
        self.assertLess(
            tier2_count, len(PART_DB) * 0.5,
            f"Tier 2 has {tier2_count}/{len(PART_DB)} items — too many, will cause fill issues",
        )

    def test_all_tiers_populated(self):
        """All three tiers should have items."""
        from worlds.ksp1.parts import PART_DB
        tiers = ITEM_TIERS
        tier_counts = {0: 0, 1: 0, 2: 0}
        for v in tiers.values():
            tier_counts[v] += 1
        tier_counts[0] = len(PART_DB) - len(tiers)
        for tier, count in tier_counts.items():
            self.assertGreater(count, 0, f"Tier {tier} is empty")


class TestJsonSync(unittest.TestCase):
    """item_tiers.json must stay in sync with the computation in compute_item_tiers.py."""

    def test_json_matches_computation(self):
        """Regenerate tiers from PART_DB and verify they match the shipped JSON."""
        from worlds.ksp1.scripts.compute_item_tiers import compute_item_tiers
        computed = compute_item_tiers()
        self.assertEqual(
            ITEM_TIERS, computed,
            "data/item_tiers.json is stale — run: "
            "python -m worlds.ksp1.scripts.compute_item_tiers",
        )


if __name__ == "__main__":
    unittest.main()
