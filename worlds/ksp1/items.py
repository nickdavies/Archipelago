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
    DEFAULT_PART_MANAGER, PART_REGISTRY, CapabilityFlag,
    Engine, FuelTank, SolidBooster, HeatShield, Parachute,
    LandingLeg, Decoupler, MiscEquipment,
)
from .contracts import all_possible_contract_specs
from .ranks import RankContext, rank_sig_for
from .options import GoalContractMode

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
    CapabilityFlag.LAUNCH_CLAMP,
})

# Parts reclassified from progression → useful.
# These are niche items that don't gate meaningful missions individually:
# monoprop engine, monoprop tanks, Mk2/Mk3 aircraft tanks, external tanks,
# adapter fuel tanks, Sepratron, Launch Escape System.
# Phase 2: deleted `_RECLASSIFY_USEFUL` and `_RECLASSIFY_FILLER`.  Those
# hand-curated lists existed to relieve fill-pressure under the legacy
# progressive system, where the progressive item was the gate and
# individual parts were redundant alternates.  In rank-space, the
# bumper decides which specific parts are reps — its picks become
# PROGRESSION via the promote-reps step in apply_sphere_ladder, and
# every non-rep gets demoted to USEFUL by the same pass.  No hand
# tuning required; the data-driven default (PROGRESSION for any part
# that's an engine/tank/etc., USEFUL for ``_USEFUL_PROVIDES`` misc,
# FILLER for everything else) is sufficient.



def _classify_part_item(item_name: str) -> ItemClassification:
    """Initial classification for a part item, derived purely from the
    part's dataclass type and ``MiscEquipment.provides`` flags.

    Phase 2: every part with a structural or capability role starts as
    PROGRESSION; the sphere-ladder pass then promotes the bumper's
    selected reps (which are already PROGRESSION by default) and
    demotes everything else to USEFUL.  Parts with no provides flags
    (decorative wings, lights, fairings) start as FILLER.
    """
    # Classification is pack-independent (an item's class never changes with the
    # enabled set), so it reads the full installed universe.
    parts = DEFAULT_PART_MANAGER.parts.get(item_name, [])
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

# Progressive item names — the few that survived Phase 2.  Part-category
# progressives have been retired in favor of rank-axis gating.  These
# three remain because they're not rocket-part gates:
#   - R&D: tech-tree band advancement
#   - Launch Pad: mass-cap progression (optional)
#   - Science Instrument: feeds psi_tier into the science budget formula
PROGRESSIVE_RD_NAME: str = "Progressive R&D"
PROGRESSIVE_LAUNCH_PAD_NAME: str = "Progressive Launch Pad"
PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: str = "Progressive Science Instrument"

# Curated-building progressives (buildings_in_logic). Only added to the pool
# when the option is on; otherwise these names are never created and the world
# is byte-for-byte unchanged.  Each maps to a curated capability effect via
# ``effects.building_effects``:
#   * Astronaut Complex -> free EVA         (Effect.CAN_EVA)
#   * Tracking Station  -> DSN comms range  (Effect.DSN_POWER; see comms.py)
# VAB/SPH (Effect.VESSEL_MASS_LIMIT) are kept in the client/server wire but ship
# MAXED this release — the item below is NOT pooled and NOT capability-gated —
# so a future VAB->part-count gate is a server-only change (no breaking client
# change).  Stock KSP has 2 upgrade levels per facility (base 0 -> 1 -> 2), so
# each gets 2 copies (levels 1 and 2 on top of base level 0).
PROGRESSIVE_VAB_NAME: str = "Progressive VAB"
PROGRESSIVE_TRACKING_STATION_NAME: str = "Progressive Tracking Station"
PROGRESSIVE_ASTRONAUT_COMPLEX_NAME: str = "Progressive Astronaut Complex"
PROGRESSIVE_MISSION_CONTROL_NAME: str = "Progressive Mission Control"

# Number of pooled copies per building (stock = 2 upgrade levels).  Mission
# Control only models the maneuver-nodes level (stock L2 = our L1), so 1 copy.
PROGRESSIVE_VAB_COUNT: int = 2
PROGRESSIVE_TRACKING_STATION_COUNT: int = 2
PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT: int = 2
PROGRESSIVE_MISSION_CONTROL_COUNT: int = 1

# Bridge: which AP progressive item supplies each curated ``Building``.  The
# effects layer (``effects.min_building_level_for``) returns a ``Building`` +
# level; this map turns that into the ``Counted`` requirement's kind (the item
# name) so the sphere ladder threads the right item into the chain.  Defined
# here (items.py owns item names) rather than in effects.py to keep effects.py
# free of item-pool concerns / import cycles.  Imported lazily by capability /
# sphere_ladder.
def _building_to_item_name() -> dict:
    from .effects import Building
    return {
        Building.VAB: PROGRESSIVE_VAB_NAME,
        Building.SPH: PROGRESSIVE_VAB_NAME,  # SPH folds into VAB (vessel limits)
        Building.TRACKING_STATION: PROGRESSIVE_TRACKING_STATION_NAME,
        Building.ASTRONAUT_COMPLEX: PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
        Building.MISSION_CONTROL: PROGRESSIVE_MISSION_CONTROL_NAME,
        # R&D facility rides the existing Progressive R&D item (no separate item);
        # the requirement level is an R&D *count* threshold, not a facility level.
        Building.RESEARCH_AND_DEVELOPMENT: PROGRESSIVE_RD_NAME,
        Building.LAUNCH_PAD: PROGRESSIVE_LAUNCH_PAD_NAME,
    }

# Kerbin baseline tonnage caps by collected count (index = number of copies
# received).  Index 0 = no copies = starting cap.  Starting at 100t lets
# sphere-0 do basic Kerbin / Mun / Minmus orbit + landing without any Launch
# Pad item, which breaks the bootstrap deadlock when the item is banned from
# the early bucket.  (A lower, more binding base is desirable but makes the
# pad load-bearing in the bootstrap band -- the chain-ordered copies then
# strand in the restrictive fill; that needs milestone-based pad placement,
# tracked separately.)  Non-Kerbin homes scale these by their surface→low-
# orbit dv ratio (see ``progressive_launch_pad_caps_for``).
PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN: tuple[float, ...] = (
    20.0, 100.0, 400.0, float("inf"))

# Count of copies in the item pool — fixed regardless of home.  Only the
# tonnage caps scale; the player always collects the same number of items.
PROGRESSIVE_LAUNCH_PAD_COUNT: int = len(PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN) - 1  # 3 copies

# Backward-compat alias used by callers that haven't been updated yet to
# ``progressive_launch_pad_caps_for``.  Removed when no consumers remain.
PROGRESSIVE_LAUNCH_PAD_CAPS: tuple[float, ...] = PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN


# See ``bodies.progressive_launch_pad_caps_for`` for the per-home cap
# function.  Kept defined there to keep items.py free of body-dynamics
# math; this module just owns the Kerbin baseline tuple.

# Progressive items: offsets 50–99 (special range, not physical parts).
# Only the three non-part progressives remain after Phase 2.
_PROGRESSIVE_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    PROGRESSIVE_RD_NAME:                (50, ItemClassification.progression),
    PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: (66, ItemClassification.progression),
    PROGRESSIVE_LAUNCH_PAD_NAME:        (68, ItemClassification.progression),
    # Curated-building progressives (buildings_in_logic) — ids are always
    # registered (so create_item resolves the name), but the items are only
    # POOLED when the option is on.  When off, none are ever created.
    PROGRESSIVE_VAB_NAME:               (70, ItemClassification.progression),
    PROGRESSIVE_TRACKING_STATION_NAME:  (71, ItemClassification.progression),
    PROGRESSIVE_ASTRONAUT_COMPLEX_NAME: (72, ItemClassification.progression),
    PROGRESSIVE_MISSION_CONTROL_NAME:   (73, ItemClassification.progression),
}

PROGRESSIVE_RD_COUNT: int = 5   # must equal tech_tree.MAX_RD_BAND
PROGRESSIVE_PSI_COUNT: int = 3

# Contract items: a large dedicated block at offset 10_000+ (well clear of the
# cramped legacy ranges). One stable id per possible (type, body) — the universe
# is fixed even though any given seed places only a subset. Every contract item
# is progression (it self-gates its location and paces the run). Item and
# location ids share a base, so the two blocks must not overlap: items live at
# 10_000+, contract locations at 20_000+ (see locations.py / test_parts_data).
_CONTRACT_ITEM_BASE_OFFSET = 10_000
_CONTRACT_ITEMS: dict[str, tuple[int, ItemClassification]] = {
    spec.item_name: (_CONTRACT_ITEM_BASE_OFFSET + i, ItemClassification.progression)
    for i, spec in enumerate(
        sorted(all_possible_contract_specs(), key=lambda s: s.contract_id))
}

ITEM_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, (offset, _) in {
        **ITEM_TABLE,
        **_FILLER_ITEMS,
        **_VICTORY_ITEM,
        **_PROGRESSIVE_ITEMS,
        **_CONTRACT_ITEMS,
    }.items()
}

# ---------------------------------------------------------------------------
# Precollected items (never placed in the pool)
# ---------------------------------------------------------------------------

#: Always precollected — structural necessity in every seed.
ALWAYS_PRECOLLECTED: tuple[str, ...] = ("strutConnector",)

#: Precollected only for the complete_tech_tree goal, whose science funding
#: assumes basic temperature science on every body.  Scoped to that goal so
#: it doesn't perturb the item pool / fill of goals that don't need it.
TECH_TREE_PRECOLLECTED: tuple[str, ...] = ("sensorThermometer",)

#: Precollected when start_with_launch_clamps option is enabled.
CLAMP_PRECOLLECTED: tuple[str, ...] = ("launchClamp1",)

# ---------------------------------------------------------------------------
# Item creation helpers
# ---------------------------------------------------------------------------

def create_item(world: KSP1World, name: str) -> KSP1Item:
    if name in ITEM_TABLE:
        offset, classification = ITEM_TABLE[name]
        # Per-seed promotion: a part required by some generated contract MUST be
        # progression so AP guarantees it reachable before the contract location.
        # Only this seed's contracts trigger it (mining parts stay filler/useful
        # when no mine contract was placed).
        if name in getattr(world, "contract_required_part_names", frozenset()):
            classification = ItemClassification.progression
    elif name in _FILLER_ITEMS:
        offset, classification = _FILLER_ITEMS[name]
    elif name in _VICTORY_ITEM:
        offset, classification = _VICTORY_ITEM[name]
    elif name in _PROGRESSIVE_ITEMS:
        offset, classification = _PROGRESSIVE_ITEMS[name]
    elif name in _CONTRACT_ITEMS:
        offset, classification = _CONTRACT_ITEMS[name]
    else:
        raise KeyError(f"Unknown KSP1 item: {name!r}")
    item = KSP1Item(name, classification, KSP1_BASE_ID + offset, world.player)
    # Phase 2: every item carries its per-axis rank signature so the
    # sphere-ladder item_rule can gate placement uniformly.  Items not
    # on any rank axis (filler, R&D, Pad, PSI) have an empty sig and
    # are unaffected by rank ceilings.
    ctx = getattr(world, "_rank_context", None)
    if ctx is None:
        # Generated outside a world build (test fixture, tool).  Use
        # the module default so the attribute is always present.
        from .ranks import DEFAULT_CONTEXT
        ctx = DEFAULT_CONTEXT
    item.rank_sig = rank_sig_for(name, ctx)
    return item


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

    Phase 2: every part in ``PART_DB`` is added as an *individual* AP
    item.  The three surviving progressives (R&D, PSI, Launch Pad) are
    counted into the pool with ``_sphere_tier`` set per copy.
    """
    precollected: set[str] = set(ALWAYS_PRECOLLECTED)
    if world.options.start_with_launch_clamps:
        precollected.update(CLAMP_PRECOLLECTED)
    if world.goal_spec.complete_tech_tree:
        precollected.update(TECH_TREE_PRECOLLECTED)

    for name in precollected:
        world.multiworld.push_precollected(create_item(world, name))

    # Pool: every enabled part as an individual item, skipping precollected.
    # Sorted for deterministic iteration; the enabled set comes from the world's
    # PartManager (Phase 2: a disabled pack's parts are simply absent here).
    pool: list[KSP1Item] = [
        create_item(world, name)
        for name in sorted(world.part_manager.parts)
        if name not in precollected
    ]

    # Progressive Science Instrument copies (one per psi_tier level).
    for tier in range(1, PROGRESSIVE_PSI_COUNT + 1):
        item = create_item(world, PROGRESSIVE_SCIENCE_INSTRUMENT_NAME)
        item._sphere_tier = tier
        pool.append(item)

    # Progressive R&D items (gates higher tech tree bands). For the tech-tree
    # goal these double as the goal items in non-findable modes: starting
    # precollects all copies; count/progressive_unlock lock some/all on threshold
    # locations (_resolve_goal_contract_mode), so only the still-pooled copies are
    # added here. Non-tech goals always pool every copy regardless of mode.
    mode = world.options.goal_contract_mode.value
    is_tech = world.goal_spec.complete_tech_tree
    rd_locked = sum(
        1 for _loc, _cnt, item_name in world.contract_threshold_defs
        if item_name == PROGRESSIVE_RD_NAME
    )
    if is_tech and mode == GoalContractMode.option_starting:
        for _ in range(PROGRESSIVE_RD_COUNT):
            world.multiworld.push_precollected(create_item(world, PROGRESSIVE_RD_NAME))
        rd_in_pool = 0
    else:
        rd_in_pool = PROGRESSIVE_RD_COUNT - rd_locked
    for tier in range(1, rd_in_pool + 1):
        item = create_item(world, PROGRESSIVE_RD_NAME)
        item._sphere_tier = tier
        pool.append(item)

    # Progressive Launch Pad items (mass-cap progression — only when enabled).
    if world.options.progressive_launch_pad:
        for tier in range(1, PROGRESSIVE_LAUNCH_PAD_COUNT + 1):
            item = create_item(world, PROGRESSIVE_LAUNCH_PAD_NAME)
            item._sphere_tier = tier
            pool.append(item)

    # Curated-building progressives (buildings_in_logic — only when enabled).
    # Each copy carries ``_sphere_tier`` exactly like the Pad/R&D counted
    # progressives so the placement window (``_item_min_sphere``) gates the Nth
    # copy at the sphere that first needs building level N.
    if world.options.buildings_in_logic:
        for name, count in (
            (PROGRESSIVE_TRACKING_STATION_NAME,
             PROGRESSIVE_TRACKING_STATION_COUNT),
            (PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
             PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT),
            (PROGRESSIVE_MISSION_CONTROL_NAME,
             PROGRESSIVE_MISSION_CONTROL_COUNT),
        ):
            for tier in range(1, count + 1):
                item = create_item(world, name)
                item._sphere_tier = tier
                pool.append(item)

    # Contract gate items. Non-goal contracts are always pooled. Goal contract
    # items depend on the goal contract mode:
    #   findable           -> pool (found like any other item; today's behavior)
    #   starting           -> precollected (extra starting items)
    #   count / progressive -> locked on a threshold location (not pooled; the
    #                          player receives them by completing X contracts)
    for spec in world.contract_specs:
        pool.append(create_item(world, spec.item_name))
    for spec in world.goal_contract_specs:
        if mode == GoalContractMode.option_starting:
            world.multiworld.push_precollected(create_item(world, spec.item_name))
        elif mode == GoalContractMode.option_findable:
            pool.append(create_item(world, spec.item_name))
        # count / progressive_unlock: locked on a threshold location instead.

    # Pad the pool with filler items so item count == count of locations that
    # still need filling. create_regions() runs before create_items(), so all
    # locations exist. Pre-filled locations (goal-mode threshold locations carry
    # a locked goal/R&D item) are excluded — their item isn't in the pool, so
    # counting them would over-pad and break the items==locations balance.
    unfilled_location_count = sum(
        1 for loc in world.multiworld.get_locations(world.player)
        if loc.address is not None and loc.item is None
    )
    filler_count = unfilled_location_count - len(pool)
    if filler_count < 0:
        raise AssertionError(
            f"KSP1 item pool overflow: pool={len(pool)} > locations="
            f"{unfilled_location_count}.  Phase 2 should keep this in balance"
            f" but a check failed — investigate before generating."
        )
    for _ in range(filler_count):
        pool.append(create_item(world, get_filler_item_name(world)))

    world.multiworld.itempool += pool
