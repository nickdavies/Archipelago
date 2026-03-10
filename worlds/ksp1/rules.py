from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import KSP1World


def set_all_rules(world: KSP1World) -> None:
    # TODO: Set access rules on locations and region entrances.
    # Rules should delegate to the cached RocketCapability on the CollectionState
    # rather than computing anything inline.
    #
    # Example pattern (once capability.py is wired up):
    #   location = world.get_location("Mun Landing (Crewed)")
    #   location.access_rule = lambda state: get_capability(state, world.player).bodies["mun"].can_land_crewed
    pass


def set_completion_condition(world: KSP1World) -> None:
    # TODO: Set completion condition based on world.options.goal.
    world.multiworld.completion_condition[world.player] = lambda state: True
