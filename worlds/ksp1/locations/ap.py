"""The Archipelago boundary of the location model.

The only module in ``locations/`` that imports Archipelago: the AP
``Location`` subclass, the readers that turn AP options into
``ConfiguredLocations`` inputs, and ``create_all_locations``, which registers a
world's emitted defs with AP by name.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, assert_never

from BaseClasses import Location

from .catalog import LocationCatalog, MAX_STARTING_INV
from .defs import (
    ContractLocation, HomeLocation, KSP1LocationDef, KscBiomeLocation,
    MissionLocation, SplashdownLocation, StartingInventoryLocation,
    TechTreeLocation, ThresholdLocation,
)
from .descriptor import LocationDescriptor, descriptor_for

if TYPE_CHECKING:
    from ..world import KSP1World


class KSP1Location(Location):
    game = "Kerbal Space Program 1"
    # The typed def this AP location was created from (``None`` only for
    # address-None event locations, which are plain ``Location``s anyway).
    definition: KSP1LocationDef | None = None
    # The physics meaning, derived from ``definition`` at creation
    # (``descriptor_for``); ``None`` for locations whose access isn't
    # capability-gated by one body's mission.  The sphere ladder reads
    # structured meaning off this object instead of decoding the display name.
    descriptor: LocationDescriptor | None = None


#: Tech tree slots per node, scaled by difficulty.
#: Keys are Difficulty option values (casual=0, normal=1, expert=2).
TECH_SLOTS_BY_DIFFICULTY: dict[int, int] = {0: 4, 1: 4, 2: 3}

#: Starting inventory slot counts by difficulty.
STARTING_INV_COUNTS: dict[int, int] = {0: 20, 1: 15, 2: 10}

#: Extra starting-inventory slots when progressive_launch_pad is enabled —
#: gives the pool more zero-rule capacity to absorb items pushed out of
#: deeper locations by the launch-pad chain (avoids pool-tension fill
#: failures on narrow goals).
PROGRESSIVE_LAUNCH_PAD_STARTER_BONUS: int = 3


def effective_tech_slots_per_node(options, difficulty: int) -> int:
    """Tech slots/node for this world. Respects ``tech_slots_per_node`` if set."""
    override = getattr(options, "tech_slots_per_node", None)
    if override is not None and override.value >= 0:
        return override.value
    return TECH_SLOTS_BY_DIFFICULTY[difficulty]


def effective_starting_inv_count(options, difficulty: int) -> int:
    """Starter-inventory slot count for this world (incl. launch-pad bonus)."""
    override = getattr(options, "starting_inventory_count", None)
    if override is not None and override.value >= 0:
        base = override.value
    else:
        base = STARTING_INV_COUNTS[difficulty]
    if (getattr(options, "progressive_launch_pad", None)
            and options.progressive_launch_pad.value):
        base = min(base + PROGRESSIVE_LAUNCH_PAD_STARTER_BONUS, MAX_STARTING_INV)
    return base


def _region_for(defn: KSP1LocationDef) -> str:
    """The AP region a location lives in.  Per-body locations go in their
    body's region so the hidden-body Discover gate can sit on one entrance per
    body; tech slots in their node's region; the rest in Menu."""
    from ..regions import body_region_name
    match defn:
        case MissionLocation() | HomeLocation() | ContractLocation():
            return body_region_name(defn.body)
        case TechTreeLocation():
            return defn.node.display_name
        case (StartingInventoryLocation() | KscBiomeLocation() | SplashdownLocation()
              | ThresholdLocation()):
            return "Menu"
        case _:
            assert_never(defn)


def create_all_locations(world: KSP1World) -> None:
    """Register the world's emitted locations with AP, each in its region, and
    attach its def + physics descriptor.

    Threshold locations are the exception: ``rules.create_threshold_locations``
    registers them alongside the contract-completion events they gate on.

    Registration order is part of seed reproducibility (it fixes
    ``multiworld.get_locations`` order), so it follows the families in a fixed
    sequence rather than iterating a set.
    """
    seed = world.seed_locations
    configured = seed.configured
    player = world.player

    # Starting inventory: constrained to local items only.  These auto-check on
    # connect, so a remote player's item landing here would ship out without
    # bootstrapping the local player and risk a multiworld deadlock if
    # reciprocity stalls.
    ordered: list[KSP1LocationDef] = [
        *configured.starting_inventory, *configured.ksc_biomes,
        *configured.home_specials, configured.splashdown, *configured.missions,
        *seed.contracts, *configured.tech,
    ]
    by_region: dict[str, dict[str, int]] = defaultdict(dict)
    for defn in ordered:
        by_region[_region_for(defn)][str(defn)] = LocationCatalog.id_of(defn)
    for region_name, region_locs in by_region.items():
        world.get_region(region_name).add_locations(region_locs, KSP1Location)

    local_only = lambda item, p=player: item.player == p
    starting_names = {str(d) for d in configured.starting_inventory}
    defs_by_name = {str(d): d for d in ordered}
    for loc in world.multiworld.get_locations(player):
        defn = defs_by_name.get(loc.name)
        if defn is None:
            continue
        loc.definition = defn
        loc.descriptor = descriptor_for(defn)
        if loc.name in starting_names:
            loc.item_rule = local_only
