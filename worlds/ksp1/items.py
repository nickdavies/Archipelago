from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Item, ItemClassification

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000


class KSP1Item(Item):
    game = "Kerbal Space Program"


# TODO: Define part groups and their classifications.
# Key: item name, Value: (id offset, classification)
ITEM_TABLE: dict[str, tuple[int, ItemClassification]] = {
    # Placeholder — replace with real part groups.
    "Placeholder Part": (0, ItemClassification.filler),
}

ITEM_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, (offset, _) in ITEM_TABLE.items()
}


def create_item(world: KSP1World, name: str) -> KSP1Item:
    offset, classification = ITEM_TABLE[name]
    return KSP1Item(name, classification, KSP1_BASE_ID + offset, world.player)


def get_filler_item_name(world: KSP1World) -> str:
    # TODO: Return a real filler item name.
    return "Placeholder Part"


def create_all_items(world: KSP1World) -> None:
    # TODO: Build the full item pool from part group definitions.
    unfilled = len(world.multiworld.get_unfilled_locations(world.player))
    pool = [world.create_item("Placeholder Part") for _ in range(unfilled)]
    world.multiworld.itempool += pool
