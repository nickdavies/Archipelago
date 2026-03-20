from typing import Any

from worlds.AutoWorld import WebWorld, World

from . import items, locations, regions, rules
from .items import ITEM_NAME_TO_ID
from .locations import LOCATION_NAME_TO_ID
from .options import KSP1Options
from .tech_tree import NODES_BY_TIER


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

    item_name_to_id = ITEM_NAME_TO_ID
    location_name_to_id = LOCATION_NAME_TO_ID

    def generate_early(self) -> None:
        """Apply ExcludeLateTechTree to the exclude_locations option set."""
        if self.options.exclude_late_tech_tree:
            tier9_locs: set[str] = {
                f"{node.display_name} {slot}"
                for node in NODES_BY_TIER.get(9, [])
                for slot in range(1, 6)
            }
            self.options.exclude_locations.value |= tier9_locs

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
