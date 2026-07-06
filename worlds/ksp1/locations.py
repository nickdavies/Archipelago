"""
Location definitions for KSP1 Archipelago.

Four location sources (total ~524 max, filtered by difficulty):

  1. Starting Inventory Locations  (5/10/15/20 by difficulty)
     Zero access requirements; AP fill places the items needed to bootstrap.

  2. KSC Biome Locations  (12 total, always all 12)
     Earned by performing science experiments at KSC buildings/grounds.
     Requires EVA (capsule) or rover (probe + wheels + power + instrument).

  3. Mission Event Locations  (256 total)
     11 home-body-specific + 1 body-agnostic (Splashdown) + 244 per-body
     event-scaled checks.
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

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, MissionType,
    home_altitude_milestones,
)
from .contracts import (
    ContractSpec, all_possible_contract_specs, GOAL_CONTRACT_TYPES,
    MAX_LOCATIONS_PER_CONTRACT,
)
from .tech_tree import TECH_NODES

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000

# Offset ranges (items use 0–1999, locations use 2000–3999, alt-home extends to 4xxx)
_STARTING_INV_OFFSET_START = 2000  # Starting inventory: 2000-2019
_SPLASHDOWN_OFFSET = 2079          # Body-agnostic Splashdown (one ID, any ocean body)
_KSC_BIOME_OFFSET_START = 2080     # KSC biomes: 2080-2099
_HOME_OFFSET_START = 2100          # Kerbin home specials: 2100-2199 (11 used, rest reserved)
_MISSION_OFFSET_START = 2200       # Per-body mission events: 2200-2999
_TECH_OFFSET_START = 3000          # Tech tree: 3000-3999
_ALT_HOME_OFFSET_START = 4000      # Non-Kerbin home specials: 4000-4153 (14 bodies × 11 = 154)
_THRESHOLD_OFFSET_START = 19_000   # Goal-mode threshold locations: 19_000-19_099 (max 77 used)
_CONTRACT_OFFSET_START = 20_000     # Contract completion locations: large dedicated block, 20_000+


class KSP1Location(Location):
    game = "Kerbal Space Program 1"
    # The physics meaning of this location, attached at creation
    # (create_all_locations).  ``None`` for locations whose access isn't
    # capability-gated (tech tree / KSC biomes / starting inventory /
    # body-agnostic Splashdown).  The sphere ladder reads structured meaning
    # off this object instead of decoding the display name.
    descriptor: "LocationDescriptor | None" = None


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
    # Whether completing this event requires a Kerbal EVA (walk out of the
    # craft).  Source of truth for the curated Astronaut-Complex ``can_eva``
    # gate (buildings_in_logic).  FLAG_PLANT / SAMPLE_RETURN imply it from
    # their mission_type, but EVA-in-orbit shares the plain ORBIT type and so
    # needs this explicit flag.  Default False.
    requires_eva: bool = False

ALL_EVENTS: tuple[EventDef, ...] = (
    EventDef(EventName.ORBIT,          1, MissionType.ORBIT,         None,  False),
    EventDef(EventName.EVA_IN_ORBIT,   1, MissionType.ORBIT,         True,  False, requires_eva=True),
    EventDef(EventName.FLYBY,          1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.SOI_LEAVE,      1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.LANDING,        2, MissionType.LAND,          None,  True),
    EventDef(EventName.CREWED_LANDING, 2, MissionType.LAND,          True,  True),
    EventDef(EventName.FLAG_PLANT,     2, MissionType.FLAG_PLANT,    True,  True,  requires_eva=True),
    EventDef(EventName.RETURN,         3, MissionType.RETURN,        None,  True),
    EventDef(EventName.SAMPLE_RETURN,  3, MissionType.SAMPLE_RETURN, True,  True,  requires_eva=True),
)

EVENT_BY_NAME: dict[str, EventDef] = {e.name: e for e in ALL_EVENTS}


# ---------------------------------------------------------------------------
# Starting inventory locations (registry includes all 20; world creates N)
# ---------------------------------------------------------------------------

MAX_STARTING_INV = 20
MAX_TECH_SLOTS = 4

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

STARTING_INV_NAMES: list[str] = [
    f"Starting Inventory {i + 1}" for i in range(MAX_STARTING_INV)
]
# Exact-membership set for consumers that need to recognise a starting-inventory
# location — treat the name as an opaque identifier, never a prefix to match.
STARTING_INV_NAME_SET: frozenset[str] = frozenset(STARTING_INV_NAMES)

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
# Home-body-specific mission locations (11 per home, + 1 body-agnostic Splashdown)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomeLocationDef:
    """Metadata for a home-body-specific mission location.

    Single source of truth for the home-body location names, mission types,
    altitude thresholds, and the body itself — used by rules.py (access
    rules), sphere_ladder.py (via the location's LocationDescriptor), and
    capability_format.py (CLI/tracker display).

    ``body=None`` means the location is body-agnostic (e.g., "Splashdown"
    can be achieved on any ocean body, not just the home).
    """
    name: str
    mission_type: MissionType
    threshold_km: float | None = None
    body: BodyName | None = None


@dataclass(frozen=True)
class LocationDescriptor:
    """The physics meaning of a capability-gated location, carried as a real
    object on its ``KSP1Location`` (``loc.descriptor``).

    Built once at location creation from the structured producer objects
    (``MissionLocation`` + ``EventDef`` / ``HomeLocationDef`` / ``ContractSpec``)
    so the sphere ladder never recovers structure by decoding the display name.
    Slot-agnostic: the event slots that share a mission (e.g. Mun Landing 1/2/3)
    map to one descriptor.
    """
    body: BodyName
    mission_type: MissionType
    crewed: bool | None
    threshold_km: float | None = None
    # Contract-completion locations carry their ContractSpec; the delivery
    # payload is sized per-rung from it (see contract_payload_parts).  ``None``
    # for ordinary (non-contract) missions.
    spec: ContractSpec | None = None
    # Whether the mission requires a Kerbal EVA — drives the curated
    # Astronaut-Complex ``can_eva`` gate (buildings_in_logic).  ``None`` means
    # "let the evaluator derive it from mission_type".
    requires_eva: bool | None = None
    # The specific event (ORBIT and EVA_IN_ORBIT stay distinct even though they
    # share ``mission_type=ORBIT``).  ``None`` for home specials and contracts —
    # used by the mission-only graph-walk to skip non-mission descriptors.
    event: EventName | None = None

    @classmethod
    def from_mission(cls, ml: "MissionLocation",
                     event_def: "EventDef") -> "LocationDescriptor":
        return cls(
            body=ml.body,
            mission_type=event_def.mission_type,
            crewed=event_def.crewed,
            threshold_km=None,
            requires_eva=event_def.requires_eva,
            event=ml.event,
        )

    @classmethod
    def from_home(cls, hloc: "HomeLocationDef") -> "LocationDescriptor | None":
        # Body-agnostic entries (Splashdown) have no single body to drive the
        # bumper's mission-centric work; their requirements are dominated by the
        # per-body LAND missions the bumper already handles.
        if hloc.body is None:
            return None
        return cls(
            body=hloc.body,
            mission_type=hloc.mission_type,
            crewed=None,
            threshold_km=hloc.threshold_km,
        )

    @classmethod
    def from_spec(cls, spec: "ContractSpec") -> "LocationDescriptor":
        # Physics-gated like a mission of the contract's base type, but with the
        # required equipment as delivered payload (sized per-rung from the spec).
        td = spec.type_def
        return cls(
            body=spec.body,
            mission_type=td.base_mission_type,
            crewed=td.crewed,
            threshold_km=None,
            spec=spec,
        )


# Single, body-agnostic Splashdown location: one AP check that fires when
# the player splashes on any ocean body (Kerbin / Eve / Laythe).  Replaces
# the legacy per-body "{body} Splashdown" entries (Mun/Duna/etc. had no
# ocean and the location was unreachable).
SPLASHDOWN_LOCATION_NAME: str = "Splashdown"


# Number of altitude milestones generated per home body.  Held constant
# across all 15 landable bodies so each body contributes exactly 11
# home-body locations (4 fixed + 7 milestones) and IDs stay regular.
# The body-agnostic "Splashdown" location is separate (a single ID).
_HOME_ALTITUDE_MILESTONE_COUNT = 7


class LocationBuilder:
    """Owns the per-home location set for a single world.

    Mirrors the ``MissionBuilder`` pattern: instantiated once per world
    with the chosen home body, eagerly computes the home-body specials
    (first launch / landing / crash, altitude milestones, first staging)
    for that home, plus the body-agnostic "Splashdown" location, and
    exposes them as instance attributes.  Callers that need the active
    home's location set hold a reference to the builder rather than
    reading module-level constants.

    The class also owns the static name → def lookup across **all** 15
    landable bodies, used by sphere-ladder parsing and the CLI's check
    map without needing to know the current home.
    """

    # Landable bodies in the canonical BodyName enum order.  Class-level
    # so it can be reused as the iteration order for the static lookup.
    _LANDABLE_BODIES: tuple[BodyName, ...] = tuple(
        BodyName(b.name) for b in ALL_BODIES if b.can_land
    )

    # Static {name → def} table covering every landable body's home set.
    # Body prefixes are unique so there are no collisions.  Populated
    # lazily on first access (see ``all_home_locations``).
    _all_locations: dict[str, "HomeLocationDef"] | None = None

    # The one body-agnostic location in the home-style set.  Carried by
    # every world regardless of whether the home itself has an ocean —
    # on non-ocean homes the player has to reach Kerbin/Eve/Laythe.
    _SPLASHDOWN_DEF: "HomeLocationDef" = HomeLocationDef(
        SPLASHDOWN_LOCATION_NAME, MissionType.SPLASHDOWN, 1.0, body=None,
    )

    def __init__(self, home: BodyName) -> None:
        self.home: BodyName = home
        self.locations: tuple[HomeLocationDef, ...] = (
            self._build_for(home) + (self._SPLASHDOWN_DEF,)
        )
        self.names: list[str] = [loc.name for loc in self.locations]
        assert len(self.locations) == 4 + _HOME_ALTITUDE_MILESTONE_COUNT + 1
        # Per-home KSC biome set.  The "KSC Grounds" entry (KSP's catchall
        # ``KSC`` biome key — the grass-and-water terrain around the
        # buildings) is Kerbin-only; Kerbal Konstructs places the named
        # buildings on alien homes but the surrounding terrain doesn't
        # report as ``KSC`` there.  Drop it from the active set for
        # non-Kerbin homes so AP doesn't ship a location the mod can't
        # check.  Building-named biomes (LaunchPad, VAB, …) stay because
        # those are placed assets the client can detect anywhere.
        self.ksc_biomes: list[tuple[str, str]] = [
            (key, name) for (key, name) in KSC_BIOMES
            if home == BodyName.KERBIN or key != "KSC"
        ]
        self.ksc_biome_names: list[str] = [
            KSC_LOCATION_PREFIX + name for _, name in self.ksc_biomes
        ]

    @staticmethod
    def _build_for(home: BodyName) -> tuple[HomeLocationDef, ...]:
        prefix = str(home)
        milestones = home_altitude_milestones(
            BODY_BY_NAME[home], n=_HOME_ALTITUDE_MILESTONE_COUNT
        )
        entries: list[HomeLocationDef] = [
            HomeLocationDef(f"{prefix} First Launch", MissionType.FIRST_LAUNCH, body=home),
            HomeLocationDef(f"{prefix} First Landing", MissionType.FIRST_LANDING, body=home),
            HomeLocationDef(f"{prefix} First Crash", MissionType.SOUNDING, 0.1, body=home),
        ]
        entries.extend(
            HomeLocationDef(f"{prefix} {km}km Altitude", MissionType.SOUNDING, float(km), body=home)
            for km in milestones
        )
        entries.append(HomeLocationDef(f"{prefix} First Staging", MissionType.FIRST_STAGING, body=home))
        return tuple(entries)

    @classmethod
    def all_home_locations(cls) -> dict[str, HomeLocationDef]:
        """Flat ``name → HomeLocationDef`` map across all 15 landable bodies.

        Used by the data-package builder (every possible home location
        gets an AP location id) and by sphere-ladder / CLI lookups that
        need to resolve a location name without knowing the home.

        Includes the body-agnostic Splashdown entry exactly once.
        """
        if cls._all_locations is None:
            d: dict[str, HomeLocationDef] = {
                loc.name: loc
                for body in cls._LANDABLE_BODIES
                for loc in cls._build_for(body)
            }
            d[cls._SPLASHDOWN_DEF.name] = cls._SPLASHDOWN_DEF
            cls._all_locations = d
        return cls._all_locations


# Kerbin's home set is built eagerly here purely so the AP data package's
# location id table can keep Kerbin's legacy id range (2100-2110, 11 entries).
# The id table is module-level static (see ``_build_location_table``); the
# per-world ``LocationBuilder`` instance is the runtime API.
_KERBIN_HOME_LOCATIONS: tuple[HomeLocationDef, ...] = LocationBuilder._build_for(BodyName.KERBIN)

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

# ---------------------------------------------------------------------------
# Build the full LOCATION_TABLE (name → id offset)
# ---------------------------------------------------------------------------

def _build_location_table() -> dict[str, int]:
    table: dict[str, int] = {}
    offset = _STARTING_INV_OFFSET_START
    for name in STARTING_INV_NAMES:
        table[name] = offset
        offset += 1

    # Body-agnostic Splashdown: single ID, shared across all worlds.
    table[SPLASHDOWN_LOCATION_NAME] = _SPLASHDOWN_OFFSET

    offset = _KSC_BIOME_OFFSET_START
    for name in KSC_BIOME_NAMES:
        table[name] = offset
        offset += 1

    # Kerbin home specials use the 2100-block (11 entries, no Splashdown).
    offset = _HOME_OFFSET_START
    for loc in _KERBIN_HOME_LOCATIONS:
        table[loc.name] = offset
        offset += 1

    offset = _MISSION_OFFSET_START
    for name in MISSION_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    offset = _TECH_OFFSET_START
    for name in TECH_TREE_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    # Other landable bodies' home specials get fresh IDs from 4000 up so the
    # data package can advertise every possible (body, home-location) pair
    # without disturbing Kerbin's existing IDs.
    offset = _ALT_HOME_OFFSET_START
    for body in LocationBuilder._LANDABLE_BODIES:
        if body == BodyName.KERBIN:
            continue
        for loc in LocationBuilder._build_for(body):
            table[loc.name] = offset
            offset += 1

    # Goal-mode threshold locations: real, pre-filled locations the client
    # reports once the completed-contract count reaches each threshold. The
    # universe of names is fixed (a seed uses at most one per goal contract);
    # 77 is the max possible goal-achievement count, comfortably under 100.
    offset = _THRESHOLD_OFFSET_START
    for name in THRESHOLD_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    # Contract completion locations: the bare goal form plus every possible
    # non-goal slot suffix per (type, body). Same (type, body) sort order as the
    # contract items in items.py so the blocks stay aligned.
    offset = _CONTRACT_OFFSET_START
    for name in CONTRACT_LOCATION_NAMES:
        table[name] = offset
        offset += 1

    return table


# Every reward-location name a contract could ever register: for each possible
# (type, body), the bare goal-contract form AND every non-goal slot suffix up to
# MAX_LOCATIONS_PER_CONTRACT (base 2 + the Contract Repeats ceiling). Registering the
# universe maximum keeps location ids stable regardless of the seed's
# contract_repeats value; a given seed creates only its resolved subset (see
# create_regions). Sorted by contract_id so the id block is stable.
CONTRACT_LOCATION_NAMES: tuple[str, ...] = tuple(
    name
    for spec in sorted(all_possible_contract_specs(), key=lambda s: s.contract_id)
    for name in (
        spec.display_name,
        *(f"{spec.display_name} {i}" for i in range(1, MAX_LOCATIONS_PER_CONTRACT + 1)),
    )
)

#: Set form for O(1) "is this a contract location?" checks (UT / sphere ladder).
CONTRACT_LOCATION_NAME_SET: frozenset[str] = frozenset(CONTRACT_LOCATION_NAMES)

# Goal-mode threshold location names (count / progressive_unlock). One per goal
# contract is used per seed; the registry sizes to the universe maximum — the
# count of every possible goal achievement (each goal contract type on every
# body it can target). Derived, so adding bodies/types can't silently overflow.
MAX_CONTRACT_THRESHOLDS: int = sum(
    1 for spec in all_possible_contract_specs()
    if spec.contract_type in GOAL_CONTRACT_TYPES
)
THRESHOLD_LOCATION_NAMES: tuple[str, ...] = tuple(
    f"Contract Threshold {i}" for i in range(1, MAX_CONTRACT_THRESHOLDS + 1)
)


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

    # KSC biome locations (earned by doing science at KSC buildings) — the
    # active set comes from ``world.location_builder`` so the catchall
    # ``KSC Grounds`` (Kerbin-only terrain biome) is skipped on alien homes.
    biome_locs = {
        name: LOCATION_NAME_TO_ID[name]
        for name in world.location_builder.ksc_biome_names
    }
    menu.add_locations(biome_locs, KSP1Location)

    # Home-body specials (first launch / first staging / altitude milestones /
    # splashdown / first crash / first landing) — only for the home body the
    # world is actually pinned to; the other 14 sets exist only in the data
    # package so AP can render them on the universal tracker.
    home_locs = {name: LOCATION_NAME_TO_ID[name] for name in world.location_builder.names}
    menu.add_locations(home_locs, KSP1Location)

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

    # Contract completion locations (only the contracts this seed generated).
    # Non-goal contracts register ``world.locations_per_contract`` slot locations
    # (base 2 + Contract Repeats); goal contracts one.
    contract_locs = {
        name: LOCATION_NAME_TO_ID[name]
        for spec in (*world.contract_specs, *world.goal_contract_specs)
        for name in spec.location_names(world.locations_per_contract)
    }
    menu.add_locations(contract_locs, KSP1Location)

    # Attach each capability-gated location's physics descriptor to the created
    # object so the sphere ladder reads structured meaning off ``loc.descriptor``
    # instead of decoding the display name (bug 086).  Built from the structured
    # producers in hand; the wiring dict is transient (discarded here).  Tech /
    # KSC / starting / body-agnostic Splashdown locations stay ``descriptor=None``
    # (not physics-gated).  Tech-tree locations live in per-node regions, never
    # in ``menu``, so they are untouched here.
    descriptors: dict[str, LocationDescriptor] = {
        str(ml): LocationDescriptor.from_mission(ml, EVENT_BY_NAME[ml.event])
        for ml in MISSION_LOCATIONS
    }
    for hloc in world.location_builder.locations:
        home_desc = LocationDescriptor.from_home(hloc)
        if home_desc is not None:
            descriptors[hloc.name] = home_desc
    for spec in (*world.contract_specs, *world.goal_contract_specs):
        spec_desc = LocationDescriptor.from_spec(spec)
        for name in spec.location_names(world.locations_per_contract):
            descriptors[name] = spec_desc
    for loc in menu.locations:
        loc.descriptor = descriptors.get(loc.name)
