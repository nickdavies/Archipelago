"""KSP1 item/location name groups (``item_groups.py``, ``locations/groups.py``).

Core AP tests already check that groups are non-empty, name only real
items/locations, and don't collide with an item/location name.  These pin the
KSP-specific properties: coverage of the item/location universe, body scoping,
and that no group name shadows a substring players filter by.
"""
import unittest

from worlds.ksp1.bodies import BodyName
from worlds.ksp1.contracts import ContractSpec, ContractType
from worlds.ksp1.item_groups import ITEM_NAME_GROUPS
from worlds.ksp1.items import ITEM_NAME_TO_ID, ITEM_TABLE
from worlds.ksp1.locations.catalog import LocationCatalog
from worlds.ksp1.locations.defs import ContractLocation, MissionLocation, SplashdownLocation
from worlds.ksp1.locations.events import EventName
from worlds.ksp1.locations.groups import LOCATION_NAME_GROUPS
from worlds.ksp1.tech_tree import TECH_NODES
from worlds.ksp1.world import KSP1World


class TestItemNameGroups(unittest.TestCase):
    def test_world_exposes_groups(self):
        for name in ITEM_NAME_GROUPS:
            self.assertIn(name, KSP1World.item_name_groups)

    def test_every_non_part_item_is_grouped(self):
        """A new item table (or a new progressive) must be given a group."""
        grouped = set().union(*ITEM_NAME_GROUPS.values())
        ungrouped = sorted(
            n for n in ITEM_NAME_TO_ID
            if n not in ITEM_TABLE and n not in grouped and n != "Victory")
        self.assertEqual(ungrouped, [])

    def test_parts_group_is_every_part(self):
        self.assertEqual(ITEM_NAME_GROUPS["Parts"], frozenset(ITEM_TABLE))


class TestLocationNameGroups(unittest.TestCase):
    def test_world_exposes_groups(self):
        for name in LOCATION_NAME_GROUPS:
            self.assertIn(name, KSP1World.location_name_groups)

    def test_every_location_but_splashdown_is_grouped(self):
        grouped = set().union(*LOCATION_NAME_GROUPS.values())
        ungrouped = sorted(
            str(d) for d in LocationCatalog.all()
            if str(d) not in grouped and not isinstance(d, SplashdownLocation))
        self.assertEqual(ungrouped, [])

    def test_body_group_holds_missions_and_contracts(self):
        mun = LOCATION_NAME_GROUPS[str(BodyName.MUN)]
        orbit = str(MissionLocation(BodyName.MUN, EventName.ORBIT, 1))
        contract = str(ContractLocation(ContractSpec(ContractType.ORBIT, BodyName.MUN), 1))
        self.assertIn(orbit, mun)
        self.assertIn(orbit, LOCATION_NAME_GROUPS["Orbits"])
        self.assertIn(contract, mun)
        self.assertIn(contract, LOCATION_NAME_GROUPS["Contracts"])
        self.assertNotIn(orbit, LOCATION_NAME_GROUPS[str(BodyName.MINMUS)])

    def test_no_group_shadows_a_filter_word(self):
        """``!missing X`` filters by group membership instead of substring when
        X is a group name, so a group must never be named after an event or a
        tech node (e.g. "Landing" would hide the "Landing N" tech slots)."""
        filter_words = {str(e) for e in EventName} | {n.display_name for n in TECH_NODES}
        self.assertEqual(sorted(filter_words & set(LOCATION_NAME_GROUPS)), [])
