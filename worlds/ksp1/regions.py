"""
Region definitions for KSP1 Archipelago.

The Menu region contains all non-tech-tree locations.  Each of the 62 tech
tree nodes is its own region, connected from Menu with entrance rules that
enforce:
  1. Progressive R&D band for the node's tier
  2. Cumulative science budget through the node's tier
  3. Parent node dependencies (AND or OR based on any_to_unlock)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState, Region

from .items import PROGRESSIVE_RD_NAME
from .tech_tree import NODE_BY_ID, TECH_NODES, TechNode, TIER_TO_BAND

if TYPE_CHECKING:
    from .world import KSP1World


def create_all_regions(world: KSP1World) -> None:
    from .rules import _can_afford_tier, effective_science_safety

    player = world.player
    difficulty = world.options.difficulty.value
    safety = effective_science_safety(world.options, difficulty)

    menu = Region("Menu", player, world.multiworld)
    world.multiworld.regions.append(menu)

    for node in TECH_NODES:
        region = Region(node.display_name, player, world.multiworld)
        world.multiworld.regions.append(region)

        rule = _make_node_entrance_rule(node, player, safety, _can_afford_tier)
        menu.connect(region, rule=rule)


def _make_node_entrance_rule(
    node: TechNode,
    player: int,
    safety: float,
    can_afford_tier: Callable[[CollectionState, int, int, float], bool],
) -> Callable[[CollectionState], bool]:
    """Build an entrance rule for a tech node region.

    Checks R&D band, science budget, and parent node reachability (AND/OR).
    """
    band = TIER_TO_BAND[node.tier]
    real_parents = [p for p in node.parents if p != "start"]
    # Pre-resolve parent display names to avoid repeated dict lookups.
    parent_region_names = [NODE_BY_ID[p].display_name for p in real_parents]
    tier = node.tier
    any_to_unlock = node.any_to_unlock

    def rule(state: CollectionState) -> bool:
        if band > 0 and not state.has(PROGRESSIVE_RD_NAME, player, band):
            return False
        if not can_afford_tier(state, player, tier, safety):
            return False
        if not parent_region_names:
            return True
        if any_to_unlock:
            return any(state.can_reach_region(name, player) for name in parent_region_names)
        else:
            return all(state.can_reach_region(name, player) for name in parent_region_names)

    return rule
