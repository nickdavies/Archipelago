from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Location

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000


class KSP1Location(Location):
    game = "Kerbal Space Program"


# TODO: Define all locations derived from the mission/body matrix.
# Key: location name, Value: id offset from KSP1_BASE_ID.
LOCATION_TABLE: dict[str, int] = {
    # Placeholder — replace with the full location list.
    "Reached 10km": 0,
    "Reached Kerbin Orbit": 1,
}

LOCATION_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, offset in LOCATION_TABLE.items()
}


def create_all_locations(world: KSP1World) -> None:
    # TODO: Build the full location set from goal + body matrix.
    menu = world.get_region("Menu")
    menu.add_locations(LOCATION_NAME_TO_ID, KSP1Location)
