"""
Universal Tracker regen round-trip tests.

Verifies that generating a world, extracting slot_data, and regenerating
via re_gen_passthrough produces the same progressive representatives,
location set, and access rule results.
"""
import unittest

from test.general import setup_multiworld

from worlds.ksp1.bodies import BodyName
from worlds.ksp1.world import KSP1World


class TestUTRegen(unittest.TestCase):
    """Generate → slot_data → regen → verify same world."""

    def _regen_from_slot_data(self, seed: int, options: dict | None = None):
        """Run gen, extract slot_data, regen with re_gen_passthrough, return both worlds."""
        opts = options or {}
        mw1 = setup_multiworld(
            KSP1World,
            steps=("generate_early", "create_regions", "create_items", "set_rules"),
            seed=seed,
            options=opts,
        )
        world1: KSP1World = mw1.worlds[1]
        slot_data = world1.fill_slot_data()

        # Create a second multiworld with re_gen_passthrough set (simulates UT).
        mw2 = setup_multiworld(
            KSP1World,
            steps=(),  # we'll run steps manually after setting passthrough
            seed=seed + 1,  # different seed to prove regen uses slot_data, not RNG
            options=opts,
        )
        mw2.re_gen_passthrough = {KSP1World.game: slot_data}
        from test.general import call_all
        for step in ("generate_early", "create_regions", "create_items", "set_rules"):
            call_all(mw2, step)

        world2: KSP1World = mw2.worlds[1]
        return world1, world2, slot_data

    def test_progressive_representatives_match(self):
        """Regen must produce identical progressive representatives."""
        world1, world2, _ = self._regen_from_slot_data(seed=42)
        self.assertEqual(world1.progressive_representatives, world2.progressive_representatives)

    def test_location_set_matches(self):
        """Regen must produce the same set of location names."""
        world1, world2, _ = self._regen_from_slot_data(seed=42)
        locs1 = {loc.name for loc in world1.multiworld.get_locations(world1.player)}
        locs2 = {loc.name for loc in world2.multiworld.get_locations(world2.player)}
        self.assertEqual(locs1, locs2)

    def test_slot_data_round_trip(self):
        """Regen slot_data must match original (excluding seed-dependent filler)."""
        world1, world2, original_sd = self._regen_from_slot_data(seed=42)
        regen_sd = world2.fill_slot_data()
        # These keys must match exactly — they control access rules.
        for key in ("goal", "difficulty", "start_with_launch_clamps",
                     "tech_slots_per_node", "goal_locations", "goal_display_name",
                     "progressive_representatives"):
            self.assertEqual(
                original_sd[key], regen_sd[key],
                f"slot_data[{key!r}] mismatch after regen",
            )

    def test_custom_goal_round_trip(self):
        """Custom goal bodies survive the slot_data → regen cycle."""
        opts = {
            "goal": "custom",
            "flag_bodies": {BodyName.MUN, BodyName.DUNA},
            "return_bodies": {BodyName.MINMUS},
            "sample_return_bodies": {BodyName.IKE},
        }
        world1, world2, _ = self._regen_from_slot_data(seed=99, options=opts)
        self.assertEqual(
            world1.goal_spec.flag_bodies, world2.goal_spec.flag_bodies,
        )
        self.assertEqual(
            world1.goal_spec.return_bodies, world2.goal_spec.return_bodies,
        )
        self.assertEqual(
            world1.goal_spec.sample_return_bodies, world2.goal_spec.sample_return_bodies,
        )


class TestExplainRule(unittest.TestCase):
    """Test the explain_rule UT hook."""

    def setUp(self):
        self.mw = setup_multiworld(
            KSP1World,
            steps=("generate_early", "create_regions", "create_items", "set_rules"),
            seed=42,
        )
        self.world: KSP1World = self.mw.worlds[1]
        self.state = self.mw.state

    def test_known_location_returns_text(self):
        result = self.world.explain_rule("Mun Orbit 1", self.state)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(any(BodyName.MUN in part.get("text", "") for part in result))

    def test_unknown_target_returns_none(self):
        result = self.world.explain_rule("Nonexistent Location XYZ", self.state)
        self.assertIsNone(result)

    def test_parts_subcommand(self):
        result = self.world.explain_rule("parts", self.state)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertIn("Received Parts", result[0]["text"])


class TestCustomUTSort(unittest.TestCase):
    """Test the custom_ut_sort UT hook."""

    def setUp(self):
        self.mw = setup_multiworld(
            KSP1World,
            steps=("generate_early", "create_regions", "create_items", "set_rules"),
            seed=42,
        )
        self.world: KSP1World = self.mw.worlds[1]

    def test_body_locations_sort_before_tech_tree(self):
        body_key = self.world.custom_ut_sort(BodyName.MUN, "Mun Orbit 1")
        tech_key = self.world.custom_ut_sort("Tech Tier 1", "Basic Rocketry 1")
        self.assertLess(body_key, tech_key)

    def test_tech_tree_sorts_before_ksc(self):
        tech_key = self.world.custom_ut_sort("Tech Tier 1", "Basic Rocketry 1")
        ksc_key = self.world.custom_ut_sort("KSC", "Science from KSC Administration")
        self.assertLess(tech_key, ksc_key)

    def test_body_order_matches_all_bodies(self):
        from worlds.ksp1.bodies import ALL_BODIES
        keys = [
            self.world.custom_ut_sort(b.name, f"{b.name} Orbit 1")
            for b in ALL_BODIES
        ]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
