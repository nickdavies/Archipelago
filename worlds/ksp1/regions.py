from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Region

if TYPE_CHECKING:
    from .world import KSP1World

# TODO: Model the full body hierarchy as regions.
# The SOI/orbit/surface structure from the design doc maps naturally here.
# Regions that gate child bodies via entrances with access rules will implement
# the reachability chain (e.g. Jool orbit gates all Jool moon regions).

REGION_NAMES: list[str] = [
    "Menu",
    # TODO: Add one region per significant mission waypoint.
]


def create_all_regions(world: KSP1World) -> None:
    for name in REGION_NAMES:
        world.multiworld.regions.append(Region(name, world.player, world.multiworld))

    # TODO: Connect regions via entrances with capability-based access rules.
