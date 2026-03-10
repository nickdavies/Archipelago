from typing import Any

from worlds.AutoWorld import WebWorld, World

from . import items, locations, regions, rules
from .options import KSP1Options


class KSP1WebWorld(WebWorld):
    theme = "ocean"
    # TODO: Add tutorial entries once setup docs are written.


class KSP1World(World):
    """
    Kerbal Space Program is a space flight simulation game where you design and
    fly rockets to explore the Kerbol system.  Parts are randomized across the
    multiworld, and mission completions are the location checks.
    """

    game = "Kerbal Space Program"
    web = KSP1WebWorld()

    options_dataclass = KSP1Options
    options: KSP1Options

    item_name_to_id = items.ITEM_NAME_TO_ID
    location_name_to_id = locations.LOCATION_NAME_TO_ID

    def create_regions(self) -> None:
        regions.create_all_regions(self)
        locations.create_all_locations(self)

    def create_items(self) -> None:
        items.create_all_items(self)

    def set_rules(self) -> None:
        rules.set_all_rules(self)
        rules.set_completion_condition(self)

    def create_item(self, name: str) -> items.KSP1Item:
        return items.create_item(self, name)

    def get_filler_item_name(self) -> str:
        return items.get_filler_item_name(self)

    def fill_slot_data(self) -> dict[str, Any]:
        return self.options.as_dict("goal", "difficulty")
