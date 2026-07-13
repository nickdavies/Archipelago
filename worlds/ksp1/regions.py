"""
Region definitions for KSP1 Archipelago.

Region layout:
  * ``Menu`` — the root; holds non-body locations (KSC biomes, starting
    inventory, body-agnostic Splashdown, victory/threshold events).
  * One region per **tech tree node** (62), connected from Menu with entrance
    rules that enforce: progressive R&D band, cumulative science budget, and
    parent-node dependencies (AND/OR by ``any_to_unlock``).
  * One region per **celestial body** that owns (or transitively parents)
    locations.  Planets connect from Menu; each moon connects from its planet
    region, so a moon is reachable only if its planet is.  Every per-body
    location (mission events, home specials, contract completions) lives in its
    body's region.  Entrance rules are open today; the hidden-body ``Discover``
    gate is layered onto these same entrances in a later phase.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState, Region

from .bodies import ALL_BODIES, BODY_BY_NAME, BodyName
from .items import PROGRESSIVE_RD_NAME
from .tech_tree import NODE_BY_ID, TECH_NODES, TechNode, TIER_TO_BAND

if TYPE_CHECKING:
    from .world import KSP1World


def body_region_name(body: BodyName) -> str:
    """Canonical AP region name for a celestial body.

    Single source of truth for the body→region-name mapping; both region
    creation (here) and location assignment (``locations.create_all_locations``)
    call this rather than forming the string inline.
    """
    return str(body)


def create_all_regions(world: KSP1World) -> None:
    from .rules import _can_afford_tier, effective_science_safety

    player = world.player
    difficulty = world.options.difficulty.value
    safety = effective_science_safety(world.options, difficulty)
    home = world.mission_builder.home

    menu = Region("Menu", player, world.multiworld)
    world.multiworld.regions.append(menu)

    for node in TECH_NODES:
        region = Region(node.display_name, player, world.multiworld)
        world.multiworld.regions.append(region)

        rule = _make_node_entrance_rule(node, player, safety, home, _can_afford_tier)
        menu.connect(region, rule=rule)

    _create_body_regions(world, menu)


def bodies_needing_regions(world: KSP1World) -> list[BodyName]:
    """Bodies that own — or transitively parent — at least one location.

    A body owns locations if it has emitted mission events, home-body specials,
    or per-body contract completions.  A moon's planet is included even when the
    planet owns nothing itself, so the moon region has a parent to hang off.
    Returned in ``ALL_BODIES`` order (parents precede their moons) for stable,
    hash-seed-independent region ordering.
    """
    lb = world.location_builder
    owning: set[BodyName] = set()
    for ml in lb.mission_locations:
        owning.add(ml.body)
    for hloc in lb.locations:
        if hloc.body is not None:
            owning.add(hloc.body)
    for spec in (*world.contract_specs, *world.goal_contract_specs):
        owning.add(spec.body)

    needed: set[BodyName] = set()
    for body in owning:
        cur: BodyName | None = body
        while cur is not None and cur not in needed:
            needed.add(cur)
            cur = BODY_BY_NAME[cur].parent

    return [b.name for b in ALL_BODIES if b.name in needed]


def _create_body_regions(world: KSP1World, menu: Region) -> None:
    """Create and connect the per-body region hierarchy.

    Every region is created before any connection is made, so a moon's planet
    region always exists when we connect the moon to it.  Planets connect from
    Menu; moons connect from their planet via a ``can_reach_region`` rule (the
    idiom the tech-node parents already use).
    """
    player = world.player
    mw = world.multiworld

    bodies = bodies_needing_regions(world)
    regions_by_body: dict[BodyName, Region] = {}
    for body in bodies:
        region = Region(body_region_name(body), player, mw)
        mw.regions.append(region)
        regions_by_body[body] = region

    for body in bodies:
        region = regions_by_body[body]
        parent = BODY_BY_NAME[body].parent
        if parent is not None and parent in regions_by_body:
            planet = regions_by_body[parent]
            planet.connect(region, rule=_make_body_entrance_rule(planet.name, player))
        else:
            menu.connect(region)


def _make_body_entrance_rule(
    parent_region_name: str, player: int
) -> Callable[[CollectionState], bool]:
    """Entrance rule for a moon region: reachable once its planet region is."""
    return lambda state: state.can_reach_region(parent_region_name, player)


def _make_node_entrance_rule(
    node: TechNode,
    player: int,
    safety: float,
    home: BodyName,
    can_afford_tier: Callable[[CollectionState, int, int, float, BodyName], bool],
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
        if not can_afford_tier(state, player, tier, safety, home):
            return False
        if not parent_region_names:
            return True
        if any_to_unlock:
            return any(state.can_reach_region(name, player) for name in parent_region_names)
        else:
            return all(state.can_reach_region(name, player) for name in parent_region_names)

    return rule
