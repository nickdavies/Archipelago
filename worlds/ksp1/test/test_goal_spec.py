"""Tests for the GoalSpec system (resolve_goal_spec, goal_spec_location_names, victory rules)."""
import unittest

from worlds.ksp1.bodies import BodyName
from worlds.ksp1.options import Goal
from worlds.ksp1.data.feasibility import MODEL_INFEASIBLE_LOCATIONS_BY_DIFFICULTY

# Tests resolve against the default physics profile (comfortable) table.
MODEL_INFEASIBLE_LOCATIONS = MODEL_INFEASIBLE_LOCATIONS_BY_DIFFICULTY["comfortable"]
from worlds.ksp1.locations import EVENT_BY_NAME, EventName, MissionLocation
from worlds.ksp1.rules import (
    GoalSpec,
    resolve_goal_spec,
    goal_spec_location_names,
    _PRESET_GOALS,
    _ALL_LANDABLE_BODIES,
)

# Tests pull the infeasible-locations set straight from the checked-in
# feasibility table — same source the world uses at generate-early.
# Regenerated via ``worlds/ksp1/scripts/generate_feasibility.py``.
_KERBIN_INFEASIBLE_LOCATIONS = MODEL_INFEASIBLE_LOCATIONS[BodyName.KERBIN]


def _kerbin_proxy_bodies_for(event: EventName) -> frozenset[BodyName]:
    """Body subset whose every (body, event) AP-location slot is in the
    Kerbin-home infeasible set — i.e. the bodies the Standard Returns /
    Sample Returns presets will exclude.
    """
    out: set[BodyName] = set()
    for b in _ALL_LANDABLE_BODIES:
        scale = EVENT_BY_NAME[event].scale
        names = {str(MissionLocation(b, event, slot)) for slot in range(1, scale + 1)}
        if names and names <= _KERBIN_INFEASIBLE_LOCATIONS:
            out.add(b)
    return frozenset(out)

# All resolve_goal_spec calls in this file run for Kerbin home — that's the
# only home Phase 3a supports anyway.  Centralised so future home-aware
# preset tests are easy to add.
_HOME = BodyName.KERBIN
from worlds.ksp1.tech_tree import LEAF_TECH_NODES
from worlds.ksp1.test.base import KSP1TestBase as _SharedKSP1TestBase


# ---------------------------------------------------------------------------
# Unit tests for resolve_goal_spec (no full world needed)
# ---------------------------------------------------------------------------

class _FakeOption:
    """Minimal stand-in for an AP Option with a .value attribute."""
    def __init__(self, value):
        self.value = value
        self.current_option_name = str(value)


class _FakeSetOption:
    """Minimal stand-in for an AP OptionSet.

    AP's OptionSet stores the raw YAML body names as plain ``str`` (not
    BodyName members), so coerce here — otherwise tests that pass BodyName
    enums silently bypass the string->enum conversion that production hits.
    """
    def __init__(self, values=None):
        self.value = {str(v) for v in values} if values else set()


class _FakeOptions:
    """Minimal stand-in for KSP1Options."""
    def __init__(self, goal=0, flag=None, ret=None, sample=None, orbit=None, flyby=None):
        self.goal = _FakeOption(goal)
        self.flag_bodies = _FakeSetOption(flag)
        self.return_bodies = _FakeSetOption(ret)
        self.sample_return_bodies = _FakeSetOption(sample)
        self.orbit_bodies = _FakeSetOption(orbit)
        self.flyby_bodies = _FakeSetOption(flyby)


class TestResolveGoalSpec(unittest.TestCase):

    def test_preset_duna_return(self):
        opts = _FakeOptions(goal=Goal.option_duna_return)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.display_name, "Duna Return")
        self.assertEqual(spec.return_bodies, (BodyName.DUNA,))
        self.assertFalse(spec.flag_bodies)
        self.assertFalse(spec.sample_return_bodies)
        self.assertFalse(spec.complete_tech_tree)

    def test_preset_mun_flag(self):
        opts = _FakeOptions(goal=Goal.option_mun_flag)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.display_name, "Mun Flag Plant")
        self.assertEqual(spec.flag_bodies, (BodyName.MUN,))

    def test_preset_mun_sample_return(self):
        opts = _FakeOptions(goal=Goal.option_mun_sample_return)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.display_name, "Mun Sample Return")
        self.assertEqual(spec.sample_return_bodies, (BodyName.MUN,))

    def test_preset_complete_tech_tree(self):
        opts = _FakeOptions(goal=Goal.option_complete_tech_tree)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertTrue(spec.complete_tech_tree)

    def test_preset_flag_every_body(self):
        opts = _FakeOptions(goal=Goal.option_flag_every_body)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        # resolve_goal_spec strips the home body — flag-every-body for a
        # Kerbin home becomes flag-every-body-except-Kerbin.
        self.assertEqual(
            set(spec.flag_bodies),
            set(_ALL_LANDABLE_BODIES) - {_HOME},
        )

    def test_custom_flag_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, flag=[BodyName.MUN, BodyName.MINMUS])
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.flag_bodies, (BodyName.MINMUS, BodyName.MUN))
        self.assertIn("Custom:", spec.display_name)

    def test_custom_return_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, ret=[BodyName.DUNA, BodyName.EVE])
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(set(spec.return_bodies), {BodyName.DUNA, BodyName.EVE})

    def test_custom_mixed(self):
        opts = _FakeOptions(goal=Goal.option_custom, flag=[BodyName.MUN], ret=[BodyName.DUNA], sample=[BodyName.EELOO])
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.flag_bodies, (BodyName.MUN,))
        self.assertEqual(spec.return_bodies, (BodyName.DUNA,))
        self.assertEqual(spec.sample_return_bodies, (BodyName.EELOO,))

    def test_custom_orbit_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, orbit=[BodyName.DUNA, BodyName.MUN])
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(set(spec.orbit_bodies), {BodyName.DUNA, BodyName.MUN})
        self.assertFalse(spec.flyby_bodies)
        self.assertIn("Custom:", spec.display_name)
        self.assertIn("Orbit", spec.display_name)

    def test_custom_flyby_bodies(self):
        opts = _FakeOptions(goal=Goal.option_custom, flyby=[BodyName.JOOL])
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.flyby_bodies, (BodyName.JOOL,))
        self.assertFalse(spec.orbit_bodies)
        self.assertIn("Flyby", spec.display_name)

    def test_custom_raw_string_bodies(self):
        # AP feeds raw YAML strings, not BodyName members.  The resolved spec
        # must hold BodyName enums so downstream rules (e.g. ``b.value``) work.
        # Regression for: custom goal -> 'str' object has no attribute 'value'.
        opts = _FakeOptions(goal=Goal.option_custom, sample={"Kerbin"})
        spec = resolve_goal_spec(opts, BodyName.LAYTHE, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertEqual(spec.sample_return_bodies, (BodyName.KERBIN,))
        self.assertTrue(all(isinstance(b, BodyName) for b in spec.sample_return_bodies))

    def test_custom_orbit_only_raises_without_goal_custom(self):
        opts = _FakeOptions(goal=Goal.option_duna_return, orbit=[BodyName.MUN])
        with self.assertRaises(RuntimeError):
            resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)

    def test_body_lists_with_non_custom_raises(self):
        opts = _FakeOptions(goal=Goal.option_duna_return, flag=[BodyName.MUN])
        with self.assertRaises(RuntimeError):
            resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)

    def test_custom_empty_lists_raises(self):
        opts = _FakeOptions(goal=Goal.option_custom)
        with self.assertRaises(RuntimeError):
            resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)

    def test_all_presets_resolve(self):
        # Some presets are home-system-local and only resolve against a
        # compatible home (e.g. ``jool_moons_return`` needs a Jool moon).
        # Pick the home per preset; default to Kerbin otherwise.
        preset_home = {
            Goal.option_jool_moons_return: BodyName.LAYTHE,
        }
        for goal_value, preset in _PRESET_GOALS.items():
            home = preset_home.get(goal_value, _HOME)
            infeasible = MODEL_INFEASIBLE_LOCATIONS.get(home, frozenset())
            opts = _FakeOptions(goal=goal_value)
            spec = resolve_goal_spec(opts, home, infeasible)
            self.assertIsInstance(spec, GoalSpec)
            self.assertTrue(spec.display_name)
            self.assertEqual(spec.home, home)


# ---------------------------------------------------------------------------
# Unit tests for goal_spec_location_names
# ---------------------------------------------------------------------------

class TestGoalSpecLocationNames(unittest.TestCase):

    def test_single_return(self):
        # Goal sentinels are the goal CONTRACTS, not the mission events.
        spec = GoalSpec(display_name="test", return_bodies=(BodyName.DUNA,))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Contract: Return from Duna"])

    def test_single_flag(self):
        spec = GoalSpec(display_name="test", flag_bodies=(BodyName.MUN,))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Contract: Flag Plant on Mun"])

    def test_single_sample_return(self):
        spec = GoalSpec(display_name="test", sample_return_bodies=(BodyName.MUN,))
        names = goal_spec_location_names(spec)
        self.assertEqual(names, ["Contract: Sample Return from Mun"])

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

    def test_standard_returns_excludes_home_and_proxy_bodies(self):
        # Standard Returns is built per-world from "all landable bodies
        # minus the home minus bodies whose every RETURN slot is in the
        # model-infeasible set".  For Kerbin home that proxy set comes
        # from the capability system — whichever bodies the dv model
        # can't reach even with max kit are excluded automatically.
        opts = _FakeOptions(goal=Goal.option_standard_returns)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        proxy = _kerbin_proxy_bodies_for(EventName.RETURN)
        self.assertNotIn(_HOME, spec.return_bodies)
        for proxy_body in proxy:
            self.assertNotIn(proxy_body, spec.return_bodies,
                             f"{proxy_body} is in proxy set but appeared in standard returns")
        expected_count = len(_ALL_LANDABLE_BODIES) - 1 - len(proxy & set(_ALL_LANDABLE_BODIES))
        self.assertEqual(len(goal_spec_location_names(spec)), expected_count)

    def test_mixed_custom(self):
        spec = GoalSpec(
            display_name="test",
            flag_bodies=(BodyName.MUN,),
            return_bodies=(BodyName.DUNA,),
            sample_return_bodies=(BodyName.EELOO,),
        )
        names = goal_spec_location_names(spec)
        self.assertEqual(len(names), 3)
        self.assertIn("Contract: Flag Plant on Mun", names)
        self.assertIn("Contract: Return from Duna", names)
        self.assertIn("Contract: Sample Return from Eeloo", names)


# ---------------------------------------------------------------------------
# Full world tests for victory access rules
# ---------------------------------------------------------------------------

KSP1TestBase = _SharedKSP1TestBase


class TestMunReturnGoalReachability(KSP1TestBase):
    """Custom return_bodies=["Mun"] goal — reachable with all items, not with none."""
    options = {"goal": "mun_sample_return"}
    # Victory routes through the cheap-ladder reps (_cheap_mission_reps), a pre_fill
    # side effect; without the real ladder it's conservatively unreachable.
    needs_real_pre_fill = True

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
    # Victory routes through the cheap-ladder reps (_cheap_mission_reps), a pre_fill
    # side effect; without the real ladder it's conservatively unreachable.
    needs_real_pre_fill = True

    def test_victory_reachable_with_all_items(self):
        self.collect_all_but([])
        victory = self.multiworld.get_location("Victory", self.player)
        self.assertTrue(
            victory.can_reach(self.multiworld.state),
            "Mun Flag victory should be reachable with all items",
        )


class TestDunaReturnGoalSlotData(KSP1TestBase):
    """Verify slot_data for a preset goal."""
    options = {"goal": "duna_return"}

    def test_slot_data_goal_locations(self):
        slot_data = self.world.fill_slot_data()
        # The goal sentinel is the goal contract, not the mission event.
        self.assertEqual(slot_data["goal_locations"], ["Contract: Return from Duna"])
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
        self.assertTrue(spec.is_home_system_only(_HOME))

    def test_mun_sample_return_is_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_mun_sample_return]
        self.assertTrue(spec.is_home_system_only(_HOME))

    def test_custom_kerbin_system_bodies(self):
        spec = GoalSpec(display_name="test", flag_bodies=(BodyName.MUN, BodyName.MINMUS))
        self.assertTrue(spec.is_home_system_only(_HOME))

    def test_duna_return_is_not_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_duna_return]
        self.assertFalse(spec.is_home_system_only(_HOME))

    def test_standard_returns_is_not_kerbin_system(self):
        opts = _FakeOptions(goal=Goal.option_standard_returns)
        spec = resolve_goal_spec(opts, _HOME, _KERBIN_INFEASIBLE_LOCATIONS)
        self.assertFalse(spec.is_home_system_only(_HOME))

    def test_complete_tech_tree_is_not_kerbin_system(self):
        spec = _PRESET_GOALS[Goal.option_complete_tech_tree]
        self.assertFalse(spec.is_home_system_only(_HOME))

    def test_mixed_custom_with_interplanetary_is_not(self):
        spec = GoalSpec(display_name="test", flag_bodies=(BodyName.MUN,), return_bodies=(BodyName.DUNA,))
        self.assertFalse(spec.is_home_system_only(_HOME))

    def test_empty_spec_is_not_kerbin_system(self):
        spec = GoalSpec(display_name="test")
        self.assertFalse(spec.is_home_system_only(_HOME))


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
