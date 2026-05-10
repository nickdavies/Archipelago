"""
Item definitions for KSP1 Archipelago.

Every entry in PART_DB is one AP item.  Item classification follows the
part type:

  Progression  -- Engines, fuel tanks, heat shields, parachutes, landing legs,
                  decouplers, capsules, probe cores, RTGs, relay antennas,
                  solar panels, launch clamps, fuel lines, ladders, docking
                  ports, science instruments.
  Useful       -- RCS, reaction wheels, batteries, ISRU, monoprop/external/
                  adapter tanks, Mk2/Mk3 aircraft tanks, Sepratron, LES.
  Filler       -- MiscEquipment with no provides flags (wings, structural,
                  lights, etc.) and generated filler (science packs, cosmetic
                  unlocks) from get_filler_item_name().

Progressive items (e.g. "Progressive Launch Engine" ×3) gate access to part
tiers. Per tier, one part is randomly selected as the representative and
removed from the pool (it IS the progressive item — a rename). Remaining
tier parts stay in pool as useful items with placement rules: they cannot
be placed in locations reachable before the player has the corresponding
progressive item count. See plans/progressive_parts.md for full spec.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Item, ItemClassification

from .parts import (
    PART_DB, PART_REGISTRY, CapabilityFlag,
    Engine, FuelTank, SolidBooster, HeatShield, Parachute,
    LandingLeg, Decoupler, MiscEquipment,
    PROGRESSIVE_PART_NAMES, PROGRESSIVE_PART_COUNTS, PROGRESSIVE_PART_TIERS,
)

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000


class KSP1Item(Item):
    game = "Kerbal Space Program 1"


# ---------------------------------------------------------------------------
# Progression-flag sets for MiscEquipment
# ---------------------------------------------------------------------------

_PROGRESSION_PROVIDES: frozenset[CapabilityFlag] = frozenset({
    CapabilityFlag.CAPSULE,
    CapabilityFlag.PROBE_CORE,
    CapabilityFlag.SOLAR_FIXED,
    CapabilityFlag.SOLAR_RETRACTABLE,
    CapabilityFlag.SOLAR_ARRAY_LARGE,
    CapabilityFlag.RTG,
    CapabilityFlag.RELAY_T1,
    CapabilityFlag.RELAY_T2,
    CapabilityFlag.RELAY_T3,
    CapabilityFlag.RELAY_T4,
    CapabilityFlag.LAUNCH_CLAMP,
    CapabilityFlag.DOCKING_PORT,
    CapabilityFlag.FUEL_LINE,
    CapabilityFlag.LADDER,
    CapabilityFlag.SCIENCE_INSTRUMENT,
    CapabilityFlag.MULTI_MOUNT,
})

_USEFUL_PROVIDES: frozenset[CapabilityFlag] = frozenset({
    CapabilityFlag.REACTION_WHEEL,
    CapabilityFlag.RCS,
    CapabilityFlag.BATTERY_LARGE,
    CapabilityFlag.ISRU,
})

# Parts reclassified from progression → useful.
# These are niche items that don't gate meaningful missions individually:
# monoprop engine, monoprop tanks, Mk2/Mk3 aircraft tanks, external tanks,
# adapter fuel tanks, Sepratron, Launch Escape System.
_RECLASSIFY_USEFUL: frozenset[str] = frozenset({
    # Monoprop engine (niche)
    "omsEngine",
    # Monoprop tanks (all)
    "RCSFuelTank", "RCSTank1-2", "Size1p5.Monoprop",
    "mk2FuselageShortMono", "mk3FuselageMONO",
    "monopropMiniSphere", "radialRCSTank", "rcsTankMini", "rcsTankRadialLong",
    # Mk2/Mk3 aircraft LFO tanks
    "mk2FuselageShortLFO", "mk2FuselageLongLFO",
    "mk2SpacePlaneAdapter", "mk2.1m.AdapterLong",
    "mk3FuselageLFO.25", "mk3FuselageLFO.50", "mk3FuselageLFO.100",
    # Mk2/Mk3 aircraft LF-only tanks
    "mk2Fuselage", "mk2FuselageShortLiquid",
    "mk3FuselageLF.25", "mk3FuselageLF.50", "mk3FuselageLF.100",
    # External tanks (Baguette, Dumpling, Doughnut)
    "externalTankCapsule", "externalTankRound", "externalTankToroid",
    # Adapter fuel tanks (structural role)
    "adapterSize2-Size1", "adapterSize2-Size1Slant",
    "adapterSize2-Mk2", "adapterMk3-Mk2",
    "adapterMk3-Size2", "adapterMk3-Size2Slant",
    "adapterSize3-Mk3", "Size3To2Adapter.v2",
    "noseConeAdapter",
    # MH size adapter tanks
    "Size1p5.Size0.Adapter.01", "Size1p5.Size1.Adapter.01",
    "Size1p5.Size1.Adapter.02",
    # Sepratron and Launch Escape System (not real boosters)
    "sepMotor1", "LaunchEscapeSystem",
    # Inline radial docking port (not a stack separator; convenience-only)
    "dockingPortLateral",
})


def _classify_part_item(item_name: str) -> ItemClassification:
    """
    Return the AP classification for a part item based on its part list.

    Parts in progressive chains are classified as useful — the progressive
    item is the progression gate, and the representative (removed from pool
    during generation) IS the progressive item.  Parts in _RECLASSIFY_USEFUL
    are also downgraded from progression to useful.
    """
    if item_name in PROGRESSIVE_PART_NAMES or item_name in _RECLASSIFY_USEFUL:
        return ItemClassification.useful

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

# Sorted names for deterministic pool iteration.
_SORTED_PART_NAMES: list[str] = sorted(PART_DB.keys())

ITEM_TABLE: dict[str, tuple[int, ItemClassification]] = {
    m.ksp_name: (m.offset, _classify_part_item(m.ksp_name))
    for m in PART_REGISTRY
}

# Filler items: offsets 100–199 (not in the main part pool)
_FILLER_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    "Science Pack 1":    (107, ItemClassification.filler),
    "Science Pack 5":    (108, ItemClassification.filler),
    "Science Pack 10":   (100, ItemClassification.filler),
    "Science Pack 25":   (101, ItemClassification.filler),
    "Science Pack 50":   (102, ItemClassification.filler),
    "Science Pack 100":  (103, ItemClassification.filler),
    "Science Pack 250":  (104, ItemClassification.filler),
}

# Victory item: offset 0 (Special range 0–99)
_VICTORY_ITEM: dict[str, tuple[int, ItemClassification]] = {
    "Victory": (0, ItemClassification.progression),
}

# Progressive items: offsets 50–99 (special range, not physical parts)
_PROGRESSIVE_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    "Progressive R&D":              (50, ItemClassification.progression),
    "Progressive Launch Engine":    (51, ItemClassification.progression),
    "Progressive Vacuum Engine":    (52, ItemClassification.progression),
    "Progressive SRB":              (53, ItemClassification.progression),
    "Progressive LFO Tank":         (54, ItemClassification.progression),
    "Progressive Heat Shield":      (55, ItemClassification.progression),
    "Progressive Stack Decoupler":  (56, ItemClassification.progression),
    "Progressive Radial Decoupler": (57, ItemClassification.progression),
    "Progressive Capsule":          (58, ItemClassification.progression),
    "Progressive Probe Core":       (59, ItemClassification.progression),
    "Progressive Solar Panel":      (60, ItemClassification.progression),
    "Progressive Relay":            (61, ItemClassification.progression),
    "Progressive Engine Plate":     (62, ItemClassification.progression),
}

PROGRESSIVE_RD_NAME: str = "Progressive R&D"
PROGRESSIVE_RD_COUNT: int = 3

# All progressive part item names (excluding Progressive R&D)
PROGRESSIVE_PART_ITEM_NAMES: frozenset[str] = frozenset(
    name for name in _PROGRESSIVE_ITEMS if name != PROGRESSIVE_RD_NAME
)

ITEM_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, (offset, _) in {
        **ITEM_TABLE,
        **_FILLER_ITEMS,
        **_VICTORY_ITEM,
        **_PROGRESSIVE_ITEMS,
    }.items()
}

# ---------------------------------------------------------------------------
# Precollected items (never placed in the pool)
# ---------------------------------------------------------------------------

#: Always precollected — structural necessity in every seed.
ALWAYS_PRECOLLECTED: tuple[str, ...] = ("strutConnector",)

#: Precollected when start_with_launch_clamps option is enabled.
CLAMP_PRECOLLECTED: tuple[str, ...] = ("launchClamp1",)

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
    elif name in _PROGRESSIVE_ITEMS:
        offset, classification = _PROGRESSIVE_ITEMS[name]
    else:
        raise KeyError(f"Unknown KSP1 item: {name!r}")
    return KSP1Item(name, classification, KSP1_BASE_ID + offset, world.player)


SCIENCE_PACK_NAMES: frozenset[str] = frozenset(_FILLER_ITEMS)

_FILLER_NAMES_WEIGHTED: list[str] = (
    ["Science Pack 1"] * 10
    + ["Science Pack 5"] * 8
    + ["Science Pack 10"] * 6
    + ["Science Pack 25"] * 4
    + ["Science Pack 50"] * 3
    + ["Science Pack 100"] * 2
    + ["Science Pack 250"] * 1
)


def get_filler_item_name(world: KSP1World) -> str:
    return world.random.choice(_FILLER_NAMES_WEIGHTED)


def create_all_items(world: KSP1World) -> None:
    """
    Add part items and progressive items to the multiworld item pool.

    Per progressive tier, one part is randomly selected as the "representative"
    and removed from the pool — it IS the progressive item (a rename).
    Remaining tier parts stay in pool as useful items with power-tier pacing.
    """
    precollected: set[str] = set(ALWAYS_PRECOLLECTED)
    if world.options.start_with_launch_clamps:
        precollected.update(CLAMP_PRECOLLECTED)

    for name in precollected:
        world.multiworld.push_precollected(create_item(world, name))

    # Select one representative per progressive tier (deterministic via world.random).
    # During UT regen, use the pre-assigned reps from slot_data instead.
    ut_reps = getattr(world, "_ut_progressive_representatives", None)
    representatives: dict[str, dict[int, str]] = {}
    all_representatives: set[str] = set()
    for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
        representatives[prog_name] = {}
        for tier_num, parts in sorted(tiers.items()):
            if ut_reps and prog_name in ut_reps and tier_num in ut_reps[prog_name]:
                rep = ut_reps[prog_name][tier_num]
            else:
                rep = world.random.choice(parts)
            representatives[prog_name][tier_num] = rep
            all_representatives.add(rep)

    world.progressive_representatives = representatives

    # Build pool: skip precollected and representatives (they ARE the progressive items).
    pool: list[KSP1Item] = [
        create_item(world, name)
        for name in _SORTED_PART_NAMES
        if name not in precollected and name not in all_representatives
    ]

    # Progressive part items (progression gates for part tiers).
    for prog_name, count in PROGRESSIVE_PART_COUNTS.items():
        for _ in range(count):
            pool.append(create_item(world, prog_name))

    # Progressive R&D items (gates higher tech tree bands).
    for _ in range(PROGRESSIVE_RD_COUNT):
        pool.append(create_item(world, PROGRESSIVE_RD_NAME))

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
