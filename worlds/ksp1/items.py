"""
Item definitions for KSP1 Archipelago.

Every entry in PART_DB is one AP item.  Item classification follows the
part type:

  Progression  -- Engines, fuel tanks, heat shields, parachutes, landing legs,
                  decouplers, capsules, probe cores, RTGs, relay antennas,
                  solar panels, launch clamps, fuel lines, ladders, docking
                  ports, science instruments.
  Useful       -- RCS, reaction wheels, batteries, ISRU.
  Filler       -- MiscEquipment with no provides flags (wings, structural,
                  lights, etc.) and generated filler (science packs, cosmetic
                  unlocks) from get_filler_item_name().

The world creates (P - precollected) items.  AP auto-fills remaining location
slots via get_filler_item_name().
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from BaseClasses import Item, ItemClassification

from .parts import (
    PART_DB, PART_REGISTRY,
    Engine, FuelTank, SolidBooster, HeatShield, Parachute,
    LandingLeg, Decoupler, MiscEquipment,
)

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000


class KSP1Item(Item):
    game = "Kerbal Space Program"


# ---------------------------------------------------------------------------
# Progression-flag sets for MiscEquipment
# ---------------------------------------------------------------------------

_PROGRESSION_PROVIDES: frozenset[str] = frozenset({
    "capsule",
    "probe_core",
    "solar_fixed",
    "solar_retractable",
    "solar_array_large",
    "rtg",
    "relay_t1",
    "relay_t2",
    "relay_t3",
    "launch_clamp",
    "docking_port",        # gates staging tier 3 (enables 4-stage missions)
    "fuel_line",           # gates asparagus staging
    "ladder",              # gates sample returns on high-gravity bodies
    "science_instrument",  # gates science collection for tech tree
})

_USEFUL_PROVIDES: frozenset[str] = frozenset({
    "reaction_wheel",
    "rcs",
    "battery_large",
    "isru",
})


def _classify_part_item(item_name: str) -> ItemClassification:
    """
    Return the AP classification for a part item based on its part list.

    A item is Progression if any of its parts gate new mission capability.
    It is Useful if it helps but rarely gates.  Otherwise Filler.
    """
    parts = PART_DB.get(item_name, [])
    if not parts:
        return ItemClassification.filler

    for part in parts:
        if isinstance(part, (Engine, FuelTank, SolidBooster, HeatShield,
                              Parachute, LandingLeg, Decoupler)):
            return ItemClassification.progression
        if isinstance(part, MiscEquipment):
            if part.provides & _PROGRESSION_PROVIDES:
                return ItemClassification.progression
            if part.provides & _USEFUL_PROVIDES:
                return ItemClassification.useful

    return ItemClassification.filler


# ---------------------------------------------------------------------------
# Build ITEM_TABLE from PART_REGISTRY (stable offsets in 1000–1999)
# ---------------------------------------------------------------------------

# Sorted names for deterministic pool iteration
_SORTED_PART_NAMES: list[str] = sorted(PART_DB.keys())

ITEM_TABLE: dict[str, tuple[int, ItemClassification]] = {
    m.ap_item: (m.offset, _classify_part_item(m.ap_item))
    for m in PART_REGISTRY
}

# Filler items: offsets 100–199 (not in the main part pool)
_FILLER_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    "Science Pack 10":   (100, ItemClassification.filler),
    "Science Pack 25":   (101, ItemClassification.filler),
    "Science Pack 50":   (102, ItemClassification.filler),
    "Science Pack 100":  (103, ItemClassification.filler),
    "Science Pack 250":  (104, ItemClassification.filler),
    "Engineering Report": (105, ItemClassification.filler),
    "Cosmetic Unlock":   (106, ItemClassification.filler),
}

# Victory item: offset 0 (Special range 0–99)
_VICTORY_ITEM: dict[str, tuple[int, ItemClassification]] = {
    "Victory": (0, ItemClassification.progression),
}

ITEM_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, (offset, _) in {
        **ITEM_TABLE,
        **_FILLER_ITEMS,
        **_VICTORY_ITEM,
    }.items()
}

# ---------------------------------------------------------------------------
# Precollected items (never placed in the pool)
# ---------------------------------------------------------------------------

#: Always precollected — structural necessity in every seed.
ALWAYS_PRECOLLECTED: tuple[str, ...] = ("EAS-4 Strut Connector",)

#: Precollected when start_with_launch_clamps option is enabled.
CLAMP_PRECOLLECTED: tuple[str, ...] = ("TT18-A Launch Stability Enhancer",)

# ---------------------------------------------------------------------------
# Item creation helpers
# ---------------------------------------------------------------------------

def create_item(world: KSP1World, name: str) -> KSP1Item:
    if name in ITEM_TABLE:
        offset, classification = ITEM_TABLE[name]
    elif name in _FILLER_ITEMS:
        offset, classification = _FILLER_ITEMS[name]
    elif name in _VICTORY_ITEM:
        offset, classification = _VICTORY_ITEM[name]
    else:
        raise KeyError(f"Unknown KSP1 item: {name!r}")
    return KSP1Item(name, classification, KSP1_BASE_ID + offset, world.player)


_FILLER_NAMES_WEIGHTED: list[str] = (
    # ~40% useful science packs (weighted by list frequency)
    ["Science Pack 50"] * 8
    + ["Science Pack 100"] * 6
    + ["Science Pack 250"] * 4
    + ["Science Pack 10"] * 4
    + ["Science Pack 25"] * 4
    # ~60% junk / cosmetic
    + ["Engineering Report"] * 12
    + ["Cosmetic Unlock"] * 12
)


def get_filler_item_name(world: KSP1World) -> str:
    return world.random.choice(_FILLER_NAMES_WEIGHTED)


def create_all_items(world: KSP1World) -> None:
    """
    Add (P - precollected) part items to the multiworld item pool.

    Precollected items are pushed to the player's starting inventory and
    removed from the pool so AP fills their location slots with other items.
    """
    precollected: set[str] = set(ALWAYS_PRECOLLECTED)
    if world.options.start_with_launch_clamps:
        precollected.update(CLAMP_PRECOLLECTED)

    for name in precollected:
        world.multiworld.push_precollected(create_item(world, name))

    pool: list[KSP1Item] = [
        create_item(world, name)
        for name in _SORTED_PART_NAMES
        if name not in precollected
    ]

    # Pad the pool with filler items so item count == location count.
    # create_regions() runs before create_items(), so all locations exist.
    real_location_count = sum(
        1 for loc in world.multiworld.get_locations(world.player)
        if loc.address is not None
    )
    filler_count = real_location_count - len(pool)
    for _ in range(filler_count):
        pool.append(create_item(world, get_filler_item_name(world)))

    world.multiworld.itempool += pool
