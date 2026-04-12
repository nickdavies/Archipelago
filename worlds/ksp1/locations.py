"""
Location definitions for KSP1 Archipelago.

Four location sources (total 475 max, filtered by difficulty):

  1. Starting Inventory Locations  (5/10/15/20 by difficulty)
     Zero access requirements; AP fill places the items needed to bootstrap.

  2. KSC Biome Locations  (11 total, always all 11)
     Earned by performing science experiments at KSC buildings.
     Requires EVA (capsule) or rover (probe + wheels + power + instrument).

  3. Mission Event Locations  (229 total)
     13 Kerbin-specific + 216 per-body event-scaled checks.
     Eve Return/Sample Return exist but require all progression parts.
     Scale is by event difficulty, not body distance:
       Flyby/SOI Leave/Orbit = 1 slot each,
       Landing/Crewed Landing/Flag Plant = 2 slots each,
       Return/Sample Return = 3 slots each.
     Per landable body: 3×1 + 3×2 + 2×3 = 15 locations.
     Per non-landable body (Jool, Kerbol): 3×1 = 3 locations.

  4. Tech Tree Locations  (129–215 by difficulty)
     3–5 locations per node × 43 nodes, scaled by difficulty.
     Access rule: player can earn enough science to afford the node's tier.

Location IDs use KSP1_BASE_ID + offset.  The registry includes all 20
possible starting inventory slots and max (5) tech slots so the world can
create the correct subset at runtime based on difficulty.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Location

from .bodies import ALL_BODIES
from .tech_tree import TECH_TREE_LOCATION_NAMES

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
    game = "Kerbal Space Program"


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

#: Events shared by all bodies that can be landed on.
LANDABLE_EVENTS: tuple[str, ...] = (
    "Flyby",
    "SOI Leave",
    "Orbit",
    "Landing",
    "Crewed Landing",
    "Flag Plant",
    "Return",
    "Sample Return",
)

#: Events for non-landable bodies (Jool, Kerbol).
ORBITAL_ONLY_EVENTS: tuple[str, ...] = (
    "Flyby",
    "SOI Leave",
    "Orbit",
)

#: Location slots per event, scaled by achievement difficulty.
#: Easy events (fly past) get 1 slot; hard events (sample return) get 3.
EVENT_SCALE: dict[str, int] = {
    "Flyby": 1, "SOI Leave": 1, "Orbit": 1,
    "Landing": 2, "Crewed Landing": 2, "Flag Plant": 2,
    "Return": 3, "Sample Return": 3,
}

#: Bodies excluded from the per-body event table (have their own Kerbin events).
_KERBIN_EXCLUDED: frozenset[str] = frozenset({"Kerbin"})


# ---------------------------------------------------------------------------
# Starting inventory locations (registry includes all 20; world creates N)
# ---------------------------------------------------------------------------

MAX_STARTING_INV = 20
MAX_TECH_SLOTS = 5

#: Tech tree slots per node, scaled by difficulty.
#: Keys are Difficulty option values (casual=0, normal=1, expert=2, insane=3).
TECH_SLOTS_BY_DIFFICULTY: dict[int, int] = {0: 5, 1: 5, 2: 4, 3: 3}

STARTING_INV_NAMES: list[str] = [
    f"Starting Inventory {i + 1}" for i in range(MAX_STARTING_INV)
]

# ---------------------------------------------------------------------------
# KSC biome locations (always all 11, no difficulty scaling)
# ---------------------------------------------------------------------------

KSC_BIOME_NAMES: list[str] = [
    "KSC LaunchPad",
    "KSC Runway",
    "KSC VAB",
    "KSC SPH",
    "KSC Tracking Station",
    "KSC Astronaut Complex",
    "KSC Administration",
    "KSC Mission Control",
    "KSC R&D",
    "KSC Crawlerway",
    "KSC Flag Pole",
]

# ---------------------------------------------------------------------------
# Kerbin-specific mission locations (13 total, fixed)
# ---------------------------------------------------------------------------

KERBIN_LOCATION_NAMES: list[str] = [
    "Kerbin First Launch",
    "Kerbin First Landing",
    "Kerbin 5km Altitude",
    "Kerbin 15km Altitude",
    "Kerbin 25km Altitude",
    "Kerbin 35km Altitude",
    "Kerbin 45km Altitude",
    "Kerbin 55km Altitude",
    "Kerbin 70km Altitude",
    "Kerbin Orbit",
    "Kerbin Splashdown",
    "Kerbin First Staging",
    "Kerbin EVA in Orbit",
]

assert len(KERBIN_LOCATION_NAMES) == 13

# ---------------------------------------------------------------------------
# Per-body mission location names (233 total, generated from body data)
# ---------------------------------------------------------------------------

def get_body_events(body) -> tuple[str, ...]:
    """Return the AP event list for a body."""
    return LANDABLE_EVENTS if body.can_land else ORBITAL_ONLY_EVENTS


def _build_mission_locations() -> list[str]:
    """
    Generate all per-body event-scaled mission location names.
    Order: body order in ALL_BODIES (skipping Kerbin), then events, then slots.
    """
    names: list[str] = []
    for body in ALL_BODIES:
        if body.name in _KERBIN_EXCLUDED:
            continue
        for event in get_body_events(body):
            for slot in range(1, EVENT_SCALE[event] + 1):
                names.append(f"{body.name} {event} {slot}")
    return names


MISSION_LOCATION_NAMES: list[str] = _build_mission_locations()

# 14 landable × 15 + 2 non-landable × 3 = 216
assert len(MISSION_LOCATION_NAMES) == 216, (
    f"Expected 216 per-body mission locations, got {len(MISSION_LOCATION_NAMES)}"
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

def event_location_names(body_name: str, event: str) -> list[str]:
    """Return the list of location names for one body/event combination."""
    return [f"{body_name} {event} {i}" for i in range(1, EVENT_SCALE[event] + 1)]


# ---------------------------------------------------------------------------
# Location creation (called by world.create_regions)
# ---------------------------------------------------------------------------

def create_all_locations(world: KSP1World) -> None:
    """
    Create and attach all locations to the Menu region.

    Starting inventory locations: only the first N (by difficulty) are created.
    KSC biome locations: always all 11.
    Tech tree slots per node: 3–5 by difficulty.
    All mission locations are always created.
    """
    from .options import Difficulty
    from .tech_tree import TECH_NODES

    difficulty = world.options.difficulty.value
    starting_inv_counts = {
        Difficulty.option_casual: 20,
        Difficulty.option_normal: 15,
        Difficulty.option_expert: 10,
        Difficulty.option_insane: 5,
    }
    num_starting = starting_inv_counts[difficulty]
    num_tech_slots = TECH_SLOTS_BY_DIFFICULTY[difficulty]

    menu = world.get_region("Menu")

    # Starting inventory (zero-requirement bootstrapping locations)
    starting_locs = {
        name: LOCATION_NAME_TO_ID[name]
        for name in STARTING_INV_NAMES[:num_starting]
    }
    menu.add_locations(starting_locs, KSP1Location)

    # KSC biome locations (earned by doing science at KSC buildings)
    biome_locs = {name: LOCATION_NAME_TO_ID[name] for name in KSC_BIOME_NAMES}
    menu.add_locations(biome_locs, KSP1Location)

    # Kerbin-specific events
    kerbin_locs = {name: LOCATION_NAME_TO_ID[name] for name in KERBIN_LOCATION_NAMES}
    menu.add_locations(kerbin_locs, KSP1Location)

    # Per-body mission events
    mission_locs = {name: LOCATION_NAME_TO_ID[name] for name in MISSION_LOCATION_NAMES}
    menu.add_locations(mission_locs, KSP1Location)

    # Tech tree node slots (filtered by difficulty)
    tech_locs: dict[str, int] = {}
    for node in TECH_NODES:
        for slot in range(1, num_tech_slots + 1):
            name = f"{node.display_name} {slot}"
            tech_locs[name] = LOCATION_NAME_TO_ID[name]
    menu.add_locations(tech_locs, KSP1Location)
