"""
Location definitions for KSP1 Archipelago.

Four location sources (total ~524 max, filtered by difficulty):

  1. Starting Inventory Locations  (5/10/15/20 by difficulty)
     Zero access requirements; AP fill places the items needed to bootstrap.

  2. KSC Biome Locations  (12 total, always all 12)
     Earned by performing science experiments at KSC buildings/grounds.
     Requires EVA (capsule) or rover (probe + wheels + power + instrument).

  3. Mission Event Locations  (256 total)
     12 Kerbin-specific + 244 per-body event-scaled checks.
     Kerbol excluded (root body — can't escape/flyby, orbit infeasible).
     Eve Return/Sample Return exist but require all progression parts.
     Scale is by event difficulty, not body distance:
       Flyby/SOI Leave/Orbit/EVA in Orbit = 1 slot each,
       Landing/Crewed Landing/Flag Plant = 2 slots each,
       Return/Sample Return = 3 slots each.
     Per landable body: 4×1 + 3×2 + 2×3 = 16 locations.
     Per non-landable body (Jool): 4×1 = 4 locations.
     Kerbol excluded entirely (root body).

  4. Tech Tree Locations  (124–248 by difficulty)
     2–4 locations per node × 62 nodes, scaled by difficulty.
     Access rule: player can earn enough science to afford the node's tier.

Location IDs use KSP1_BASE_ID + offset.  The registry includes all 20
possible starting inventory slots and max (4) tech slots so the world can
create the correct subset at runtime based on difficulty.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from BaseClasses import Location

from .bodies import ALL_BODIES, BodyName, MissionType
from .tech_tree import TECH_NODES

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000

# Offset ranges (items use 0–1999, locations use 2000–3999)
_STARTING_INV_OFFSET_START = 2000  # Starting inventory: 2000-2019
_KSC_BIOME_OFFSET_START = 2080     # KSC biomes: 2080-2099
_KERBIN_OFFSET_START = 2100        # Kerbin special: 2100-2199
_MISSION_OFFSET_START = 2200       # Per-body mission events: 2200-2999
_TECH_OFFSET_START = 3000          # Tech tree: 3000-3999


class KSP1Location(Location):
    game = "Kerbal Space Program 1"


# ---------------------------------------------------------------------------
# Structured location types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionLocation:
    """Structured representation of a per-body mission event location.

    Canonical format: "{body} {event} {slot}" — e.g. "Mun Orbit 1".
    """
    body: BodyName       # BodyName.MUN, BodyName.DUNA, etc.
    event: EventName     # EventName.ORBIT, EventName.FLAG_PLANT, etc.
    slot: int            # 1-based

    def __str__(self) -> str:
        return f"{self.body} {self.event} {self.slot}"

    @classmethod
    def parse(cls, s: str) -> MissionLocation | None:
        """Parse 'Body Event N' → MissionLocation, or None if not parseable."""
        for body in ALL_BODIES:
            prefix = body.name + " "
            if s.startswith(prefix):
                rest = s[len(prefix):]
                # Last token is the slot number, everything before is the event
                parts = rest.rsplit(" ", 1)
                if len(parts) == 2:
                    event_name, slot_str = parts
                    try:
                        slot = int(slot_str)
                    except ValueError:
                        return None
                    if event_name in EVENT_BY_NAME:
                        return cls(body.name, EventName(event_name), slot)
                return None
        return None


@dataclass(frozen=True)
class TechTreeLocation:
    """Structured representation of a tech tree slot location.

    Canonical format: "{node_name} {slot}" — e.g. "Basic Rocketry 3".
    """
    node_name: str  # "Basic Rocketry", "General Rocketry", etc.
    slot: int

    def __str__(self) -> str:
        return f"{self.node_name} {self.slot}"


# ---------------------------------------------------------------------------
# Event types — single source of truth for all event metadata
# ---------------------------------------------------------------------------

class EventName(StrEnum):
    """Canonical event names — use these instead of string literals."""
    FLYBY = "Flyby"
    SOI_LEAVE = "SOI Leave"
    ORBIT = "Orbit"
    EVA_IN_ORBIT = "EVA in Orbit"
    LANDING = "Landing"
    CREWED_LANDING = "Crewed Landing"
    FLAG_PLANT = "Flag Plant"
    RETURN = "Return"
    SAMPLE_RETURN = "Sample Return"


@dataclass(frozen=True)
class EventDef:
    """Metadata for a body mission event type.

    One row per event — single source of truth. All consumers (locations,
    rules, capability, CLI) derive their needs from this table.
    """
    name: EventName
    scale: int                            # location slots per body for this event
    mission_type: MissionType             # key into MissionBuilder.profiles_for
    crewed: bool | None                   # None=try both, True=crewed only, False=unmanned only
    requires_landing: bool                # only applies to landable bodies

ALL_EVENTS: tuple[EventDef, ...] = (
    EventDef(EventName.ORBIT,          1, MissionType.ORBIT,         None,  False),
    EventDef(EventName.EVA_IN_ORBIT,   1, MissionType.ORBIT,         True,  False),
    EventDef(EventName.FLYBY,          1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.SOI_LEAVE,      1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.LANDING,        2, MissionType.LAND,          None,  True),
    EventDef(EventName.CREWED_LANDING, 2, MissionType.LAND,          True,  True),
    EventDef(EventName.FLAG_PLANT,     2, MissionType.FLAG_PLANT,    True,  True),
    EventDef(EventName.RETURN,         3, MissionType.RETURN,        None,  True),
    EventDef(EventName.SAMPLE_RETURN,  3, MissionType.SAMPLE_RETURN, True,  True),
)

EVENT_BY_NAME: dict[str, EventDef] = {e.name: e for e in ALL_EVENTS}


# ---------------------------------------------------------------------------
# Starting inventory locations (registry includes all 20; world creates N)
# ---------------------------------------------------------------------------

MAX_STARTING_INV = 20
MAX_TECH_SLOTS = 4

#: Tech tree slots per node, scaled by difficulty.
#: Keys are Difficulty option values (casual=0, normal=1, expert=2, insane=3).
TECH_SLOTS_BY_DIFFICULTY: dict[int, int] = {0: 4, 1: 4, 2: 3, 3: 2}

#: Starting inventory slot counts by difficulty.
STARTING_INV_COUNTS: dict[int, int] = {0: 20, 1: 15, 2: 10, 3: 5}

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

STARTING_INV_NAMES: list[str] = [
    f"Starting Inventory {i + 1}" for i in range(MAX_STARTING_INV)
]

# Structured tech tree locations and their string names.
TECH_TREE_LOCATIONS: list[TechTreeLocation] = [
    TechTreeLocation(node.display_name, slot)
    for node in TECH_NODES
    for slot in range(1, MAX_TECH_SLOTS + 1)
]
TECH_TREE_LOCATION_NAMES: list[str] = [str(t) for t in TECH_TREE_LOCATIONS]

assert len(TECH_TREE_LOCATION_NAMES) == 248  # 62 nodes × 4 slots

# ---------------------------------------------------------------------------
# KSC biome locations (always all 12, no difficulty scaling)
# ---------------------------------------------------------------------------

#: Prefix on every KSC-biome AP location name, so the check name makes it
#: obvious the player has to collect science there.  Authoritative on the
#: server; the client receives the full ``biome_key -> location_name`` map
#: via slot_data and stays prefix-agnostic.
KSC_LOCATION_PREFIX: str = "Science from "

#: Single source of truth for KSC biomes:
#: (KSP game-internal biome key, building display name).
#:
#: Order matters for the client's sub-biome fallback: it iterates this map
#: and falls back to ``startswith`` on the key when no exact match is found
#: (e.g. ``VABMainBuilding`` → ``VAB``).  The bare ``KSC`` entry is a
#: catch-all prefix and MUST stay last so more-specific keys win.
KSC_BIOMES: list[tuple[str, str]] = [
    ("LaunchPad",        "KSC LaunchPad"),
    ("Runway",           "KSC Runway"),
    ("Administration",   "KSC Administration"),
    ("AstronautComplex", "KSC Astronaut Complex"),
    ("FlagPole",         "KSC Flag Pole (Astronaut Complex)"),
    ("SPH",              "KSC SPH"),
    ("VAB",              "KSC VAB"),
    ("TrackingStation",  "KSC Tracking Station"),
    ("MissionControl",   "KSC Mission Control"),
    ("Crawlerway",       "KSC Crawlerway"),
    ("R&D",              "KSC R&D"),
    ("KSC",              "KSC Grounds"),
]

KSC_BIOME_NAMES: list[str] = [KSC_LOCATION_PREFIX + name for _, name in KSC_BIOMES]

# ---------------------------------------------------------------------------
# Kerbin-specific mission locations (12 total, fixed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KerbinLocationDef:
    """Metadata for a Kerbin-specific mission location.

    This is the single source of truth for Kerbin location names, mission types,
    and altitude thresholds — used by rules.py (access rules) and
    capability_format.py (CLI/tracker display).
    """
    name: str
    mission_type: MissionType
    threshold_km: float | None = None


KERBIN_LOCATIONS: tuple[KerbinLocationDef, ...] = (
    KerbinLocationDef("Kerbin First Launch", MissionType.FIRST_LAUNCH),
    KerbinLocationDef("Kerbin First Landing", MissionType.FIRST_LANDING),
    KerbinLocationDef("Kerbin First Crash", MissionType.SOUNDING, 0.1),
    KerbinLocationDef("Kerbin 5km Altitude", MissionType.SOUNDING, 5.0),
    KerbinLocationDef("Kerbin 15km Altitude", MissionType.SOUNDING, 15.0),
    KerbinLocationDef("Kerbin 25km Altitude", MissionType.SOUNDING, 25.0),
    KerbinLocationDef("Kerbin 35km Altitude", MissionType.SOUNDING, 35.0),
    KerbinLocationDef("Kerbin 45km Altitude", MissionType.SOUNDING, 45.0),
    KerbinLocationDef("Kerbin 55km Altitude", MissionType.SOUNDING, 55.0),
    KerbinLocationDef("Kerbin 70km Altitude", MissionType.SOUNDING, 70.0),
    KerbinLocationDef("Kerbin Splashdown", MissionType.SPLASHDOWN, 1.0),
    KerbinLocationDef("Kerbin First Staging", MissionType.FIRST_STAGING),
)

KERBIN_LOCATION_NAMES: list[str] = [loc.name for loc in KERBIN_LOCATIONS]

assert len(KERBIN_LOCATION_NAMES) == 12

# ---------------------------------------------------------------------------
# Per-body mission location names (217 total, generated from body data)
# ---------------------------------------------------------------------------

def get_body_events(body) -> tuple[EventName, ...]:
    """Return the AP event list for a body."""
    if body.name == BodyName.KERBOL:
        return ()  # root body — can't flyby/escape, orbit is infeasible
    if body.can_land:
        return tuple(e.name for e in ALL_EVENTS)
    return tuple(e.name for e in ALL_EVENTS if not e.requires_landing)


def _build_mission_locations() -> list[MissionLocation]:
    """
    Generate all per-body event-scaled mission locations.
    Order: body order in ALL_BODIES, then events, then slots.
    """
    locs: list[MissionLocation] = []
    for body in ALL_BODIES:
        for event in get_body_events(body):
            for slot in range(1, EVENT_BY_NAME[event].scale + 1):
                locs.append(MissionLocation(body.name, event, slot))
    return locs


MISSION_LOCATIONS: list[MissionLocation] = _build_mission_locations()
MISSION_LOCATION_NAMES: list[str] = [str(m) for m in MISSION_LOCATIONS]

# 15 landable × 16 + 1 non-landable (Jool) × 4 = 244  (Kerbol excluded)
assert len(MISSION_LOCATION_NAMES) == 244, (
    f"Expected 244 per-body mission locations, got {len(MISSION_LOCATION_NAMES)}"
)

# Bodies in the Kerbin system — derived from ALL_BODIES, not hardcoded.
KERBIN_SYSTEM_BODY_NAMES: frozenset[BodyName] = frozenset(
    b.name for b in ALL_BODIES if b.name == BodyName.KERBIN or b.parent == BodyName.KERBIN
)


# ---------------------------------------------------------------------------
# Build the full LOCATION_TABLE (name → id offset)
# ---------------------------------------------------------------------------

def _build_location_table() -> dict[str, int]:
    table: dict[str, int] = {}
    offset = _STARTING_INV_OFFSET_START
    for name in STARTING_INV_NAMES:
        table[name] = offset
        offset += 1

    offset = _KSC_BIOME_OFFSET_START
    for name in KSC_BIOME_NAMES:
        table[name] = offset
        offset += 1

    offset = _KERBIN_OFFSET_START
    for name in KERBIN_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    offset = _MISSION_OFFSET_START
    for name in MISSION_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    offset = _TECH_OFFSET_START
    for name in TECH_TREE_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    return table


LOCATION_TABLE: dict[str, int] = _build_location_table()

LOCATION_NAME_TO_ID: dict[str, int] = {
    name: KSP1_BASE_ID + offset
    for name, offset in LOCATION_TABLE.items()
}

# ---------------------------------------------------------------------------
# Helper: which locations belong to a given body + event?
# ---------------------------------------------------------------------------

def event_locations(body_name: BodyName, event: EventName) -> list[MissionLocation]:
    """Return the list of MissionLocation objects for one body/event combination."""
    return [MissionLocation(body_name, event, i) for i in range(1, EVENT_BY_NAME[event].scale + 1)]


# ---------------------------------------------------------------------------
# Location creation (called by world.create_regions)
# ---------------------------------------------------------------------------

def create_all_locations(world: KSP1World) -> None:
    """
    Create and attach all locations to the Menu region.

    Starting inventory locations: only the first N (by difficulty) are created.
    KSC biome locations: always all 12.
    Tech tree slots per node: 2–4 by difficulty.
    All mission locations are always created.
    """
    difficulty = world.options.difficulty.value
    num_starting = effective_starting_inv_count(world.options, difficulty)
    num_tech_slots = effective_tech_slots_per_node(world.options, difficulty)

    menu = world.get_region("Menu")

    # Starting inventory (zero-requirement bootstrapping locations).
    # Constrain to local items only: these auto-check on connect, so a remote
    # player's item landing here would ship out without bootstrapping the
    # local player and risk a multiworld deadlock if reciprocity stalls.
    starting_names = STARTING_INV_NAMES[:num_starting]
    starting_locs = {name: LOCATION_NAME_TO_ID[name] for name in starting_names}
    menu.add_locations(starting_locs, KSP1Location)
    player = world.player
    local_only = lambda item, p=player: item.player == p
    for loc in menu.locations:
        if loc.name in starting_locs:
            loc.item_rule = local_only

    # KSC biome locations (earned by doing science at KSC buildings)
    biome_locs = {name: LOCATION_NAME_TO_ID[name] for name in KSC_BIOME_NAMES}
    menu.add_locations(biome_locs, KSP1Location)

    # Kerbin-specific events
    kerbin_locs = {name: LOCATION_NAME_TO_ID[name] for name in KERBIN_LOCATION_NAMES}
    menu.add_locations(kerbin_locs, KSP1Location)

    # Per-body mission events
    mission_locs = {name: LOCATION_NAME_TO_ID[name] for name in MISSION_LOCATION_NAMES}
    menu.add_locations(mission_locs, KSP1Location)

    # Tech tree node slots — each node's slots go into its own region.
    for node in TECH_NODES:
        region = world.get_region(node.display_name)
        node_locs: dict[str, int] = {}
        for slot in range(1, num_tech_slots + 1):
            name = str(TechTreeLocation(node.display_name, slot))
            node_locs[name] = LOCATION_NAME_TO_ID[name]
        region.add_locations(node_locs, KSP1Location)
