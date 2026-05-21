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
    # Drogue chutes (slowing-only, not landing-capable)
    "parachuteDrogue", "radialDrogue",
    # Basic ladder (Progressive Ladder uses telescopic variants)
    "ladder1",
    # External Command Seat: open 1-crew seat, not a sealed capsule.
    # Moved out of Progressive Capsule tier 1 (its capability is too
    # different from real pods to share the rep slot).
    "seatExternalCmd",
})

# Note: ionEngine + xenon tanks (xenonTank/Large/Radial) and the LF-only
# fuselages (miniFuselage, MK1Fuselage) were displaced from Progressive
# Vacuum Engine tier 3 to keep that tier engine-only. They are intentionally
# left as PROGRESSION items (default class for FuelTank/Engine) so that fill
# places them at reachable locations — ion engine + xenon must be findable
# together, otherwise ion provides zero thrust.

# Parts forced to FILLER classification regardless of progressive group
# membership or capability flags.  These are non-bootstrap-critical items
# whose presence in the useful pool inflates remaining_fill pressure
# without adding meaningful capability.
_RECLASSIFY_FILLER: frozenset[str] = frozenset({
    # Launch Escape System: emergency-only solid booster, no real capability.
    "LaunchEscapeSystem",
    # MEMLander: 2-crew lander cabin in Progressive Capsule t2; alternates exist.
    "MEMLander",
    # MiniISRU: small ISRU; capability granted by full ISRU at higher tiers.
    "MiniISRU",
})

# Progressive chains whose non-rep parts are filler-classified instead of
# useful.  Rep-impact analysis (worlds/ksp1/test/_rep_analysis.py) showed
# these chains have <3% reachability spread across rep choices, i.e. the
# alternative parts at each tier don't materially improve solvability —
# the rep alone is sufficient.  Marking the non-reps as filler reduces
# useful-pool pressure without hurting capability variance.
_FILLER_CLASS_CHAINS: frozenset[str] = frozenset({
    "Progressive Solar Panel",
    "Progressive Stack Decoupler",
    "Progressive Radial Decoupler",
    "Progressive Capsule",
    "Progressive Probe Core",
})

_FILLER_CLASS_CHAIN_PARTS: frozenset[str] = frozenset(
    part_name
    for chain in _FILLER_CLASS_CHAINS
    for tier_parts in PROGRESSIVE_PART_TIERS[chain].values()
    for part_name in tier_parts
)


def _classify_part_item(item_name: str) -> ItemClassification:
    """
    Return the AP classification for a part item based on its part list.

    Parts in progressive chains are classified as useful — the progressive
    item is the progression gate, and the representative (removed from pool
    during generation) IS the progressive item.  Parts in _RECLASSIFY_USEFUL
    are also downgraded from progression to useful. Parts in
    _RECLASSIFY_FILLER are forced to filler regardless of any other rule.
    """
    if item_name in _RECLASSIFY_FILLER:
        return ItemClassification.filler
    if item_name in _FILLER_CLASS_CHAIN_PARTS:
        return ItemClassification.filler
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

# Progressive item names (single source of truth — used in pool registration,
# capability lookups, precollect logic, and tests).
PROGRESSIVE_RD_NAME: str = "Progressive R&D"
PROGRESSIVE_LAUNCH_ENGINE_NAME: str = "Progressive Launch Engine"
PROGRESSIVE_VACUUM_ENGINE_NAME: str = "Progressive Vacuum Engine"
PROGRESSIVE_SRB_NAME: str = "Progressive SRB"
PROGRESSIVE_LFO_TANK_NAME: str = "Progressive LFO Tank"
PROGRESSIVE_HEAT_SHIELD_NAME: str = "Progressive Heat Shield"
PROGRESSIVE_STACK_DECOUPLER_NAME: str = "Progressive Stack Decoupler"
PROGRESSIVE_RADIAL_DECOUPLER_NAME: str = "Progressive Radial Decoupler"
PROGRESSIVE_CAPSULE_NAME: str = "Progressive Capsule"
PROGRESSIVE_PROBE_CORE_NAME: str = "Progressive Probe Core"
PROGRESSIVE_SOLAR_PANEL_NAME: str = "Progressive Solar Panel"
PROGRESSIVE_RELAY_NAME: str = "Progressive Relay"
PROGRESSIVE_ENGINE_PLATE_NAME: str = "Progressive Engine Plate"
PROGRESSIVE_PARACHUTE_NAME: str = "Progressive Parachute"
PROGRESSIVE_LADDER_NAME: str = "Progressive Ladder"
PROGRESSIVE_LANDING_LEG_NAME: str = "Progressive Landing Leg"
PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: str = "Progressive Science Instrument"
PROGRESSIVE_RADIAL_ENGINE_NAME: str = "Progressive Radial Engine"
PROGRESSIVE_LAUNCH_PAD_NAME: str = "Progressive Launch Pad"
PROGRESSIVE_SAS_NAME: str = "Progressive SAS"
PROGRESSIVE_XENON_TANK_NAME: str = "Progressive Xenon Tank"

# Tonnage caps by collected count (index = number of copies received).
# Index 0 = no copies = starting cap. Starting at 100t lets sphere-0 do
# basic Kerbin / Mun / Minmus orbit + landing without any Launch Pad item,
# which breaks the bootstrap deadlock when the item is banned from the
# early bucket.
PROGRESSIVE_LAUNCH_PAD_CAPS: tuple[float, ...] = (100.0, 200.0, 500.0, float("inf"))
PROGRESSIVE_LAUNCH_PAD_COUNT: int = len(PROGRESSIVE_LAUNCH_PAD_CAPS) - 1  # 3 copies

# Progressive items: offsets 50–99 (special range, not physical parts)
_PROGRESSIVE_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    PROGRESSIVE_RD_NAME:                (50, ItemClassification.progression),
    PROGRESSIVE_LAUNCH_ENGINE_NAME:     (51, ItemClassification.progression),
    PROGRESSIVE_VACUUM_ENGINE_NAME:     (52, ItemClassification.progression),
    PROGRESSIVE_SRB_NAME:               (53, ItemClassification.progression),
    PROGRESSIVE_LFO_TANK_NAME:          (54, ItemClassification.progression),
    PROGRESSIVE_HEAT_SHIELD_NAME:       (55, ItemClassification.progression),
    PROGRESSIVE_STACK_DECOUPLER_NAME:   (56, ItemClassification.progression),
    PROGRESSIVE_RADIAL_DECOUPLER_NAME:  (57, ItemClassification.progression),
    PROGRESSIVE_CAPSULE_NAME:           (58, ItemClassification.progression),
    PROGRESSIVE_PROBE_CORE_NAME:        (59, ItemClassification.progression),
    PROGRESSIVE_SOLAR_PANEL_NAME:       (60, ItemClassification.progression),
    PROGRESSIVE_RELAY_NAME:             (61, ItemClassification.progression),
    PROGRESSIVE_ENGINE_PLATE_NAME:      (62, ItemClassification.progression),
    PROGRESSIVE_PARACHUTE_NAME:         (63, ItemClassification.progression),
    PROGRESSIVE_LADDER_NAME:            (64, ItemClassification.progression),
    PROGRESSIVE_LANDING_LEG_NAME:       (65, ItemClassification.progression),
    # Science instruments are useful, not progression: they affect science
    # earnings but not capability gating (post bug-074 redesign).
    PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: (66, ItemClassification.useful),
    PROGRESSIVE_RADIAL_ENGINE_NAME:     (67, ItemClassification.progression),
    PROGRESSIVE_LAUNCH_PAD_NAME:        (68, ItemClassification.progression),
    PROGRESSIVE_SAS_NAME:               (69, ItemClassification.progression),
    PROGRESSIVE_XENON_TANK_NAME:        (70, ItemClassification.progression),
}

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


def select_progressive_representatives(world: KSP1World) -> None:
    """
    Pick one part per progressive tier as that tier's "representative" — the
    real item the AP progressive item resolves to.  Deterministic via
    ``world.random``; UT regen restores the prior pick from slot_data.

    Must run in ``generate_early`` because other worlds' ``create_regions``
    can evaluate KSP1 entrance rules via cross-player reachability sweeps
    (e.g. pokemon_rb door_shuffle) before any world's ``create_items`` runs.
    """
    ut_reps = getattr(world, "_ut_progressive_representatives", None)
    representatives: dict[str, dict[int, str]] = {}
    for prog_name, tiers in PROGRESSIVE_PART_TIERS.items():
        representatives[prog_name] = {}
        for tier_num, parts in sorted(tiers.items()):
            if ut_reps and prog_name in ut_reps and tier_num in ut_reps[prog_name]:
                rep = ut_reps[prog_name][tier_num]
            else:
                rep = world.random.choice(parts)
            representatives[prog_name][tier_num] = rep
    world.progressive_representatives = representatives


def create_all_items(world: KSP1World) -> None:
    """
    Add part items and progressive items to the multiworld item pool.

    Representatives are picked earlier in ``generate_early`` (see
    ``select_progressive_representatives``); this consumes them.
    """
    precollected: set[str] = set(ALWAYS_PRECOLLECTED)
    if world.options.start_with_launch_clamps:
        precollected.update(CLAMP_PRECOLLECTED)

    for name in precollected:
        world.multiworld.push_precollected(create_item(world, name))

    all_representatives: set[str] = {
        rep
        for tiers in world.progressive_representatives.values()
        for rep in tiers.values()
    }

    # Build pool: skip precollected and representatives (they ARE the progressive items).
    pool: list[KSP1Item] = [
        create_item(world, name)
        for name in _SORTED_PART_NAMES
        if name not in precollected and name not in all_representatives
    ]

    # Progressive part items (progression gates for part tiers).
    # Tag each copy with `_sphere_tier` (1-based copy index) so the
    # sphere-ladder Rule B can ban individual copies from harder
    # locations while leaving later copies free.  See
    # ``worlds/ksp1/sphere_ladder.py``.
    for prog_name, count in PROGRESSIVE_PART_COUNTS.items():
        for tier in range(1, count + 1):
            item = create_item(world, prog_name)
            item._sphere_tier = tier
            pool.append(item)

    # Progressive R&D items (gates higher tech tree bands).
    for tier in range(1, PROGRESSIVE_RD_COUNT + 1):
        item = create_item(world, PROGRESSIVE_RD_NAME)
        item._sphere_tier = tier
        pool.append(item)

    # Progressive Launch Pad items (mass-cap progression — only when enabled).
    if world.options.progressive_launch_pad:
        for tier in range(1, PROGRESSIVE_LAUNCH_PAD_COUNT + 1):
            item = create_item(world, PROGRESSIVE_LAUNCH_PAD_NAME)
            item._sphere_tier = tier
            pool.append(item)

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
