"""Universal Tracker map tab: the generated tracker/*.json stay in step with the
location model, and the UT-only entrances mirror their mission locations.

The marker coordinates come from an external repo, so CI can't regenerate the
files; instead this checks their *content* against the catalog, which is what
drifts when locations change (rerun ``scripts/generate_ut_map.py --write``).
"""
import json
import unittest
from pathlib import Path

from test.general import call_all, setup_multiworld
from worlds.ksp1.bodies import ALL_BODIES
from worlds.ksp1.locations.catalog import LocationCatalog
from worlds.ksp1.locations.defs import MissionLocation
from worlds.ksp1.ut_map import UT_ENTRANCES, UT_MAP_HOMES, location_map, location_node
from worlds.ksp1.world import KSP1World

_TRACKER = Path(__file__).parent.parent / "tracker"
_REGEN = "rerun `python -m worlds.ksp1.scripts.generate_ut_map --source <pack> --write`"


def _node_sections() -> dict[str, set[str]]:
    """Node label → every section listed under it (map-specific nodes have one
    entry per map)."""
    out: dict[str, set[str]] = {}
    for loc in json.loads((_TRACKER / "locations.json").read_text()):
        out.setdefault(loc["name"], set()).update(s["name"] for s in loc["sections"])
    return out


class TestTrackerFilesMatchCatalog(unittest.TestCase):
    def test_every_mapped_location_is_on_its_node(self):
        nodes = _node_sections()
        for defn in LocationCatalog.all():
            node = location_node(defn)
            if node is not None:
                self.assertIn(str(defn), nodes.get(node, set()),
                              f"{defn} missing from node {node!r}; {_REGEN}")

    def test_every_section_is_a_real_location_or_entrance(self):
        missions = set(LocationCatalog.missions())
        entrances = {e.name(b.name) for b in ALL_BODIES for e in UT_ENTRANCES
                     if MissionLocation(b.name, e.event, 1) in missions}
        for node, names in _node_sections().items():
            for name in names:
                defn = LocationCatalog.by_name(name)
                self.assertTrue(
                    name in entrances or (defn is not None and location_node(defn) == node),
                    f"{name!r} on node {node!r} is not a location of that node; {_REGEN}")

    def test_map_specific_sections_only_on_their_map(self):
        """UT filters sections by the connected slot, not by map, so a
        location drawn on a shared node shows on every map in that entry.
        Home specials must therefore only sit in entries for their home's map."""
        locations = json.loads((_TRACKER / "locations.json").read_text())
        for entry in locations:
            maps = {ml["map"] for ml in entry["map_locations"]}
            for section in entry["sections"]:
                defn = LocationCatalog.by_name(section["name"])
                only_on = location_map(defn) if defn is not None else None
                if only_on is not None:
                    self.assertEqual(maps, {str(only_on)},
                                     f"{section['name']} drawn on maps {sorted(maps)}; {_REGEN}")

    def test_maps_in_map_index_order(self):
        maps = json.loads((_TRACKER / "maps.json").read_text())
        self.assertEqual([m["name"] for m in maps], [str(h) for h in UT_MAP_HOMES])


class TestUtMapEntrances(unittest.TestCase):
    """Under UT regen, one entrance per emitted (body, UT_ENTRANCES) mission,
    each reachable exactly when its mission location is."""

    @classmethod
    def setUpClass(cls):
        steps = ("generate_early", "create_regions", "create_items", "set_rules")
        original = setup_multiworld(KSP1World, steps=steps, seed=5)
        slot_data = original.worlds[1].fill_slot_data()
        cls.multiworld = setup_multiworld(KSP1World, steps=(), seed=5)
        cls.multiworld.re_gen_passthrough = {KSP1World.game: slot_data}
        for step in steps:
            call_all(cls.multiworld, step)
        cls.world = cls.multiworld.worlds[1]

    def _ut_entrances(self):
        names = {e.name(b.name) for b in ALL_BODIES for e in UT_ENTRANCES}
        return {ent.name: ent for ent in self.multiworld.get_entrances(1) if ent.name in names}

    def test_entrances_exactly_for_emitted_missions(self):
        configured = self.world.seed_locations.configured
        regions = {r.name for r in self.multiworld.get_regions(1)}
        expected = {
            e.name(b.name) for b in ALL_BODIES if str(b.name) in regions
            for e in UT_ENTRANCES if configured.emitted_mission(b.name, e.event)
        }
        self.assertEqual(set(self._ut_entrances()), expected)

    def test_entrance_mirrors_its_location(self):
        configured = self.world.seed_locations.configured
        lookup = {e.name(b.name): (b.name, e) for b in ALL_BODIES for e in UT_ENTRANCES}
        for state in (self.multiworld.state, self.multiworld.get_all_state(False)):
            for name, ent in self._ut_entrances().items():
                body, entrance = lookup[name]
                loc = self.world.get_location(str(configured.emitted_mission(body, entrance.event)))
                self.assertEqual(ent.can_reach(state), loc.can_reach(state), name)

    def test_map_index_is_home_map(self):
        maps = json.loads((_TRACKER / "maps.json").read_text())
        self.assertEqual(maps[self.world.map_index(None)]["name"],
                         str(self.world.mission_builder.home))
