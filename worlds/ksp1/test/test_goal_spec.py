"""Tests for the GoalSpec system (resolve_goal_spec, goal_spec_location_names, victory rules)."""
import unittest

from test.bases import WorldTestBase

from worlds.ksp1.options import Goal
from worlds.ksp1.rules import (
    GoalSpec,
    resolve_goal_spec,
    goal_spec_location_names,
    _PRESET_GOALS,
    _ALL_LANDABLE_BODIES,
    _STANDARD_RETURN_BODIES,
)
from worlds.ksp1.tech_tree import LEAF_TECH_NODES


# ---------------------------------------------------------------------------
# Unit tests for resolve_goal_spec (no full world needed)
# ---------------------------------------------------------------------------

class _FakeOption:
    """Minimal stand-in for an AP Option with a .value attribute."""
    def __init__(self, value):
        self.value = value
        self.current_option_name = str(value)


class _FakeSetOption:
    """Minimal stand-in for an AP OptionSet with a .value set."""
    def __init__(self, values=None):
        self.value = set(values) if values else set()


class _FakeOptions:
    """Minimal stand-in for KSP1Options."""
    def __init__(self, goal=0, flag=None, ret=None, sample=None):
        self.goal = _FakeOption(goal)
        self.flag_bodies = _FakeSetOption(flag)
        self.return_bodies = _FakeSetOption(ret)
        self.sample_return_bodies = _FakeSetOption(sample)


class TestResolveGoalSpec(unittest.TestCase):

    def test_preset_duna_return(self):
        opts = _FakeOptions(goal=Goal.option_duna_return)
        spec = resolve_goal_spec(opts)
        self.assertEqual(spec.display_name, "Duna Return")
        self.assertEqual(spec.return_bodies, ("Duna",))
        self.assertFalse(spec.flag_bodies)
        self.assertFalse(spec.sample_return_bodies)
        self.assertFalse(spec.complete_tech_tree)

    def test_preset_mun_flag(self):
        opts = _FakeOptions(goal=Goal.option_mun_flag)
        spec = resolve_goal_spec(opts)
        self.assertEqual(spec.display_name, "Mun Flag Plant")
        self.assertEqual(spec.flag_bodies, ("Mun",))

    def test_preset_mun_sample_return(self):
        opts = _FakeOptions(goal=Goal.option_mun_sample_return)
        spec = resolve_goal_spec(opts)
        self.assertEqual(spec.display_name, "Mun Sample Return")
        self.assertEqual(spec.sample_return_bodies, ("Mun",))

    def test_preset_complete_tech_tree(self):
        opts = _FakeOptions(goal=Goal.option_complete_tech_tree)
        spec = resolve_goal_spec(opts)
        self.assertTrue(spec.complete_tech_tree)

    def test_preset_flag_every_body(self):
        opts = _FakeOptions(goal=Goal.option_flag_every_body)
        spec = resolve_goal_spec(opts)
        self.assertEqual(set(spec.flag_bodies), set(_ALL_LANDABLE_BODIES))

    def test_custom_flag_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, flag=["Mun", "Minmus"])
        spec = resolve_goal_spec(opts)
        self.assertEqual(spec.flag_bodies, ("Minmus", "Mun"))
        self.assertIn("Custom:", spec.display_name)

    def test_custom_return_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, ret=["Duna", "Eve"])
        spec = resolve_goal_spec(opts)
        self.assertEqual(set(spec.return_bodies), {"Duna", "Eve"})

    def test_custom_mixed(self):
        opts = _FakeOptions(goal=Goal.option_custom, flag=["Mun"], ret=["Duna"], sample=["Eeloo"])
        spec = resolve_goal_spec(opts)
        self.assertEqual(spec.flag_bodies, ("Mun",))
        self.assertEqual(spec.return_bodies, ("Duna",))
        self.assertEqual(spec.sample_return_bodies, ("Eeloo",))

    def test_body_lists_with_non_custom_raises(self):
        opts = _FakeOptions(goal=Goal.option_duna_return, flag=["Mun"])
        with self.assertRaises(RuntimeError):
            resolve_goal_spec(opts)

    def test_custom_empty_lists_raises(self):
        opts = _FakeOptions(goal=Goal.option_custom)
        with self.assertRaises(RuntimeError):
            resolve_goal_spec(opts)

    def test_all_presets_resolve(self):
        for goal_value in _PRESET_GOALS:
            opts = _FakeOptions(goal=goal_value)
            spec = resolve_goal_spec(opts)
            self.assertIsInstance(spec, GoalSpec)
            self.assertTrue(spec.display_name)


# ---------------------------------------------------------------------------
# Unit tests for goal_spec_location_names
# ---------------------------------------------------------------------------

class TestGoalSpecLocationNames(unittest.TestCase):

    def test_single_return(self):
        spec = GoalSpec(display_name="test", return_bodies=("Duna",))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Duna Return 1"])

    def test_single_flag(self):
        spec = GoalSpec(display_name="test", flag_bodies=("Mun",))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Mun Flag Plant 1"])

    def test_single_sample_return(self):
        spec = GoalSpec(display_name="test", sample_return_bodies=("Mun",))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Mun Sample Return 1"])

    def test_complete_tech_tree_uses_leaves(self):
        spec = GoalSpec(display_name="test", complete_tech_tree=True)
        names = goal_spec_location_names(spec)
        self.assertEqual(len(names), len(LEAF_TECH_NODES))
        # All should end with " 1"
        for n in names:
            self.assertTrue(n.endswith(" 1"), f"'{n}' should end with ' 1'")

    def test_flag_every_body_count(self):
        spec = _PRESET_GOALS[Goal.option_flag_every_body]
        names = goal_spec_location_names(spec)
        self.assertEqual(len(names), len(_ALL_LANDABLE_BODIES))

    def test_standard_returns_count(self):
        spec = _PRESET_GOALS[Goal.option_standard_returns]
        names = goal_spec_location_names(spec)
        self.assertEqual(len(names), len(_STANDARD_RETURN_BODIES))

    def test_mixed_custom(self):
        spec = GoalSpec(
            display_name="test",
            flag_bodies=("Mun",),
            return_bodies=("Duna",),
            sample_return_bodies=("Eeloo",),
        )
        names = goal_spec_location_names(spec)
        self.assertEqual(len(names), 3)
        self.assertIn("Mun Flag Plant 1", names)
        self.assertIn("Duna Return 1", names)
        self.assertIn("Eeloo Sample Return 1", names)


# ---------------------------------------------------------------------------
# Full world tests for victory access rules
# ---------------------------------------------------------------------------

class KSP1TestBase(WorldTestBase):
    game = "Kerbal Space Program 1"


class TestMunReturnGoalReachability(KSP1TestBase):
    """Custom return_bodies=["Mun"] goal — reachable with all items, not with none."""
    options = {"goal": "mun_sample_return"}

    def test_victory_reachable_with_all_items(self):
        self.collect_all_but([])
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertTrue(
            victory.can_reach(self.multiworld.state),
            "Mun Sample Return victory should be reachable with all items",
        )

    def test_victory_unreachable_with_no_items(self):
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertFalse(
            victory.can_reach(self.multiworld.state),
            "Mun Sample Return victory should be unreachable with no items",
        )


class TestMunFlagGoal(KSP1TestBase):
    """Mun flag preset goal."""
    options = {"goal": "mun_flag"}

    def test_victory_reachable_with_all_items(self):
        self.collect_all_but([])
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertTrue(
            victory.can_reach(self.multiworld.state),
            "Mun Flag victory should be reachable with all items",
        )


class TestEveReturnGoal(KSP1TestBase):
    """Eve return uses all-parts proxy."""
    options = {"goal": "eve_return"}

    def test_victory_reachable_with_all_items(self):
        self.collect_all_but([])
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertTrue(
            victory.can_reach(self.multiworld.state),
            "Eve Return victory should be reachable with all items",
        )

    def test_victory_unreachable_with_no_items(self):
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertFalse(
            victory.can_reach(self.multiworld.state),
            "Eve Return victory should be unreachable with no items",
        )


class TestDunaReturnGoalSlotData(KSP1TestBase):
    """Verify slot_data for a preset goal."""
    options = {"goal": "duna_return"}

    def test_slot_data_goal_locations(self):
        slot_data = self.world.fill_slot_data()
        self.assertEqual(slot_data["goal_locations"], ["Duna Return 1"])
        self.assertEqual(slot_data["goal_display_name"], "Duna Return")


class TestCompleteTechTreeGoalLeafNodes(KSP1TestBase):
    """Complete tech tree goal should use leaf nodes in slot_data."""
    options = {"goal": "complete_tech_tree"}

    def test_slot_data_uses_leaf_nodes(self):
        slot_data = self.world.fill_slot_data()
        goal_locs = slot_data["goal_locations"]
        self.assertEqual(len(goal_locs), len(LEAF_TECH_NODES))


# ---------------------------------------------------------------------------
# is_kerbin_system_only property
# ---------------------------------------------------------------------------

class TestIsKerbinSystemOnly(unittest.TestCase):

    def test_mun_flag_is_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_mun_flag]
        self.assertTrue(spec.is_kerbin_system_only)

    def test_mun_sample_return_is_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_mun_sample_return]
        self.assertTrue(spec.is_kerbin_system_only)

    def test_custom_kerbin_system_bodies(self):
        spec = GoalSpec(display_name="test", flag_bodies=("Mun", "Minmus"))
        self.assertTrue(spec.is_kerbin_system_only)

    def test_duna_return_is_not_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_duna_return]
        self.assertFalse(spec.is_kerbin_system_only)

    def test_standard_returns_is_not_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_standard_returns]
        self.assertFalse(spec.is_kerbin_system_only)

    def test_complete_tech_tree_is_not_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_complete_tech_tree]
        self.assertFalse(spec.is_kerbin_system_only)

    def test_mixed_custom_with_interplanetary_is_not(self):
        spec = GoalSpec(display_name="test", flag_bodies=("Mun",), return_bodies=("Duna",))
        self.assertFalse(spec.is_kerbin_system_only)

    def test_empty_spec_is_not_kerbin_system(self):
        spec = GoalSpec(display_name="test")
        self.assertFalse(spec.is_kerbin_system_only)


# ---------------------------------------------------------------------------
# Interplanetary progression restriction for Kerbin-system goals
# ---------------------------------------------------------------------------

class TestMunFlagRejectsInterplanetaryProgression(KSP1TestBase):
    """Mun flag goal should reject progression items from interplanetary locations via item_rule."""
    options = {"goal": "mun_flag"}

    def test_interplanetary_rejects_advancement(self):
        loc = self.multiworld.get_location("Duna Orbit 1", self.player)
        adv_item = next(
            i for i in self.multiworld.itempool
            if i.player == self.player and i.advancement
        )
        self.assertFalse(
            loc.item_rule(adv_item),
            f"Progression item '{adv_item.name}' should be rejected at Duna Orbit 1",
        )

    def test_interplanetary_allows_filler(self):
        from worlds.ksp1.items import create_item
        loc = self.multiworld.get_location("Duna Orbit 1", self.player)
        filler = create_item(self.world, "Science Pack 10")
        self.assertTrue(
            loc.item_rule(filler),
            "Filler item should be allowed at Duna Orbit 1",
        )

    def test_kerbin_system_allows_advancement(self):
        loc = self.multiworld.get_location("Mun Orbit 1", self.player)
        adv_item = next(
            i for i in self.multiworld.itempool
            if i.player == self.player and i.advancement
        )
        self.assertTrue(
            loc.item_rule(adv_item),
            f"Progression item '{adv_item.name}' should be allowed at Mun Orbit 1",
        )


class TestDunaReturnAllowsInterplanetaryProgression(KSP1TestBase):
    """Non-Kerbin-system goal should allow progression at interplanetary locations."""
    options = {"goal": "duna_return"}

    def test_duna_orbit_allows_advancement(self):
        loc = self.multiworld.get_location("Duna Orbit 1", self.player)
        adv_item = next(
            i for i in self.multiworld.itempool
            if i.player == self.player and i.advancement
        )
        self.assertTrue(
            loc.item_rule(adv_item),
            f"Progression item '{adv_item.name}' should be allowed at Duna Orbit 1 for duna_return goal",
        )


if __name__ == "__main__":
    unittest.main()
