"""
Universal Tracker regen round-trip tests.

Verifies that generating a world, extracting slot_data, and regenerating
via re_gen_passthrough produces the same location set and access rule
results.
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
                     "tech_slots_per_node", "physics_difficulty",
                     "goal_locations", "goal_display_name",
                     "enabled_part_packs", "buildings_in_logic"):
            self.assertEqual(
                original_sd[key], regen_sd[key],
                f"slot_data[{key!r}] mismatch after regen",
            )

    def test_explicit_physics_difficulty_round_trip(self):
        """An explicit (non-auto) physics profile must survive regen — else UT
        would rebuild the seed's logic with the default (auto) dv margins.

        The regen world is built with DEFAULT options (no physics override) so
        the only path to 'zero' is the restored slot_data, isolating the hook.
        """
        from test.general import call_all
        from worlds.ksp1.bodies import effective_physics_profile_name

        mw1 = setup_multiworld(
            KSP1World,
            steps=("generate_early", "create_regions", "create_items", "set_rules"),
            seed=42,
            options={"physics_difficulty": "zero"},
        )
        world1: KSP1World = mw1.worlds[1]
        slot_data = world1.fill_slot_data()
        self.assertEqual(slot_data["physics_difficulty"], "zero")

        mw2 = setup_multiworld(KSP1World, steps=(), seed=43, options={})
        mw2.re_gen_passthrough = {KSP1World.game: slot_data}
        for step in ("generate_early", "create_regions", "create_items", "set_rules"):
            call_all(mw2, step)
        world2: KSP1World = mw2.worlds[1]

        # Default opts would resolve to 'comfortable'; slot_data must force 'zero'.
        self.assertEqual(effective_physics_profile_name(world2.options), "zero")
        self.assertEqual(world2.fill_slot_data()["physics_difficulty"], "zero")

    def test_custom_goal_round_trip(self):
        """Custom goal bodies survive the slot_data → regen cycle."""
        opts = {
            "goal": "custom",
            "flag_bodies": {BodyName.MUN, BodyName.DUNA},
            "return_bodies": {BodyName.MINMUS},
            "sample_return_bodies": {BodyName.IKE},
            "orbit_bodies": {BodyName.EVE},
            "flyby_bodies": {BodyName.JOOL},
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
        self.assertEqual(
            world1.goal_spec.orbit_bodies, world2.goal_spec.orbit_bodies,
        )
        self.assertEqual(
            world1.goal_spec.flyby_bodies, world2.goal_spec.flyby_bodies,
        )

    def test_count_mode_thresholds_round_trip(self):
        """count-mode X and threshold defs reconstruct identically after regen."""
        opts = {
            "goal": "flag_every_body",
            "goal_contract_mode": "count",
            "contracts_available": 10,
        }
        world1, world2, original_sd = self._regen_from_slot_data(seed=7, options=opts)
        self.assertEqual(world1.contracts_required, world2.contracts_required)
        self.assertEqual(world1.contract_threshold_defs, world2.contract_threshold_defs)
        regen_sd = world2.fill_slot_data()
        for key in ("goal_contract_mode", "contracts_required", "contract_thresholds"):
            self.assertEqual(original_sd[key], regen_sd[key],
                             f"slot_data[{key!r}] mismatch after regen")

    def test_progressive_unlock_thresholds_round_trip(self):
        """progressive_unlock threshold ordering survives regen (launch-mass sort
        is deterministic, so the same goal item lands on the same threshold)."""
        opts = {
            "goal": "flag_every_body",
            "goal_contract_mode": "progressive_unlock",
            "contracts_available": 10,
        }
        world1, world2, _ = self._regen_from_slot_data(seed=11, options=opts)
        self.assertEqual(world1.contract_threshold_defs, world2.contract_threshold_defs)

    def test_random_orbit_params_round_trip(self):
        """RANDOM_ORBIT target orbits reconstruct identically after regen (a
        re-roll would diverge and the sphere-ladder cost would drift)."""
        opts = {"contract_type_weights": {"random_orbit": 5, "orbit": 1, "mine_ore": 1}}
        world1, world2, original_sd = self._regen_from_slot_data(seed=21, options=opts)
        self.assertEqual(
            world1.mission_builder.random_orbit_params,
            world2.mission_builder.random_orbit_params,
        )
        self.assertEqual(original_sd["random_orbit_params"],
                         world2.fill_slot_data()["random_orbit_params"])

    def test_contracts_and_science_in_logic_under_ut(self):
        """Under UT (no pre_fill) contract locations and tech nodes must NOT be
        permanently out of logic.

        The cheap ladder proxies (_cheap_contract_reps / _science_body_event_reps)
        are built only in pre_fill, which UT skips.  Without the live-capability
        fallback every contract returned unreachable and every tech node saw a
        0.0 science budget, so nothing past bootstrap was ever in logic.  With a
        full inventory both must be reachable, and the _ut_active gate must be
        what enables it (off the UT path the conservative floor still stands).
        """
        from worlds.ksp1.rules import _accessible_science

        opts = {
            "goal": "flag_every_body",
            "goal_contract_mode": "count",
            "contracts_available": 10,
        }
        _, world2, _ = self._regen_from_slot_data(seed=7, options=opts)
        self.assertTrue(getattr(world2, "_ut_active", False),
                        "UT regen must set _ut_active")
        player = world2.player
        home = world2.mission_builder.home
        state = world2.multiworld.get_all_state(False)

        contract_locs = [
            loc for loc in world2.multiworld.get_locations(player)
            if loc.name.startswith("Contract:")
        ]
        self.assertTrue(contract_locs, "seed produced no contract locations")

        # Fix on: full inventory reaches contracts + banks science.
        self.assertTrue(
            any(loc.can_reach(state) for loc in contract_locs),
            "no contract location reachable under UT even with full inventory",
        )
        self.assertGreater(
            _accessible_science(state, player, 1.0, home), 0.0,
            "accessible science is 0 under UT — tech nodes all out of logic",
        )

        # Negative control: the _ut_active gate is load-bearing.  With it off,
        # the absent proxies force the conservative floor (unreachable / 0.0),
        # which is exactly the pre-fix behaviour.
        world2._ut_active = False
        try:
            self.assertFalse(
                any(loc.can_reach(state) for loc in contract_locs),
                "contract reachable with proxy absent AND _ut_active off",
            )
            self.assertEqual(
                _accessible_science(state, player, 1.0, home), 0.0,
                "science non-zero with brackets absent AND _ut_active off",
            )
        finally:
            world2._ut_active = True

    def test_random_contracts_round_trip(self):
        """random_contracts goal (free flag-on-home) reconstructs after regen."""
        opts = {
            "goal": "random_contracts",
            "goal_contract_mode": "count",
            "contracts_available": 10,
        }
        world1, world2, _ = self._regen_from_slot_data(seed=13, options=opts)
        self.assertTrue(world2.goal_spec.free_goal)
        self.assertEqual(world1.goal_spec.flag_bodies, world2.goal_spec.flag_bodies)
        self.assertEqual(world1.contract_threshold_defs, world2.contract_threshold_defs)

    def test_hidden_bodies_round_trip(self):
        """Feature-on regen (a DIFFERENT RNG seed) must reproduce the exact
        hidden set + gated set from slot_data — not re-roll them — or UT would
        show the wrong gates."""
        opts = {"body_visibility_mode": "home_system", "goal": "complete_tech_tree"}
        world1, world2, sd = self._regen_from_slot_data(seed=42, options=opts)
        self.assertEqual(sorted(map(str, world1.hidden_bodies)),
                         sorted(map(str, world2.hidden_bodies)))
        self.assertEqual(sorted(map(str, world1.gated_hidden_bodies)),
                         sorted(map(str, world2.gated_hidden_bodies)))
        # slot_data carries the resolved mode + a body_item_map per gated body.
        self.assertEqual(sd["body_visibility_mode"], world1.body_visibility_mode)
        self.assertEqual(len(sd["body_item_map"]), len(world1.gated_hidden_bodies))

    def test_all_visible_round_trip(self):
        """Feature-off must stay off after regen (resolved all_visible is carried,
        so regen doesn't re-resolve auto and start hiding bodies)."""
        opts = {"body_visibility_mode": "all_visible", "goal": "duna_return"}
        world1, world2, sd = self._regen_from_slot_data(seed=7, options=opts)
        self.assertEqual(len(world2.hidden_bodies), 0)
        self.assertEqual(len(world2.gated_hidden_bodies), 0)
        self.assertNotIn("body_item_map", sd)


class TestBuildingsInLogicUTRegen(unittest.TestCase):
    """UT round-trip for buildings_in_logic (default ON) and related slot_data."""

    def _regen_from_slot_data(self, seed: int, options: dict | None = None,
                              regen_options: dict | None = None):
        from test.general import call_all
        opts = options or {}
        regen_opts = regen_options if regen_options is not None else opts
        mw1 = setup_multiworld(
            KSP1World,
            steps=("generate_early", "create_regions", "create_items", "set_rules"),
            seed=seed,
            options=opts,
        )
        world1: KSP1World = mw1.worlds[1]
        slot_data = world1.fill_slot_data()

        mw2 = setup_multiworld(
            KSP1World,
            steps=(),
            seed=seed + 1,
            options=regen_opts,
        )
        mw2.re_gen_passthrough = {KSP1World.game: slot_data}
        for step in ("generate_early", "create_regions", "create_items", "set_rules"):
            call_all(mw2, step)
        return world1, mw2.worlds[1], slot_data

    def test_buildings_in_logic_default_on_round_trip(self):
        """Default seeds emit buildings_in_logic=1 and regen restores it."""
        from worlds.ksp1.items import (
            PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
            PROGRESSIVE_MISSION_CONTROL_NAME,
            PROGRESSIVE_TRACKING_STATION_NAME,
        )
        from worlds.ksp1.world import _GATED_FACILITY_IDS

        world1, world2, slot_data = self._regen_from_slot_data(seed=42, options={})
        self.assertEqual(slot_data["buildings_in_logic"], 1)
        self.assertEqual(world1.options.buildings_in_logic.value, 1)
        self.assertEqual(world2.options.buildings_in_logic.value, 1)

        locs1 = {loc.name for loc in world1.multiworld.get_locations(world1.player)}
        locs2 = {loc.name for loc in world2.multiworld.get_locations(world2.player)}
        self.assertEqual(locs1, locs2)

        pool_names = {
            item.name for item in world1.multiworld.itempool
            if item.player == world1.player
        }
        for name in (PROGRESSIVE_TRACKING_STATION_NAME,
                     PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
                     PROGRESSIVE_MISSION_CONTROL_NAME):
            self.assertIn(name, pool_names)

        for fac in _GATED_FACILITY_IDS:
            self.assertEqual(slot_data["career"]["building_levels"][fac], 0)

    def test_buildings_in_logic_explicit_off_round_trip(self):
        """Explicit OFF in gen survives regen even when regen yaml defaults ON."""
        world1, world2, slot_data = self._regen_from_slot_data(
            seed=42,
            options={"buildings_in_logic": 0},
            regen_options={},
        )
        self.assertEqual(slot_data["buildings_in_logic"], 0)
        self.assertEqual(world1.options.buildings_in_logic.value, 0)
        self.assertEqual(world2.options.buildings_in_logic.value, 0)

    def test_enabled_part_packs_round_trip(self):
        """Stock-only enabled_part_packs survives regen."""
        from worlds.ksp1.parts.packs import STOCK

        world1, world2, slot_data = self._regen_from_slot_data(
            seed=42, options={"enabled_part_packs": []})
        self.assertEqual(slot_data["enabled_part_packs"], ["Stock"])
        self.assertEqual(world1.part_manager.enabled_packs, frozenset({STOCK}))
        self.assertEqual(world2.part_manager.enabled_packs, frozenset({STOCK}))


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

    def test_parts_filter_still_works(self):
        """The 'progressive' sub-keyword must not break ordinary filters."""
        result = self.world.explain_rule("parts engine", self.state)
        self.assertIsNotNone(result)
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
