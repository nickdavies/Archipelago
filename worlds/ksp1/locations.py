"""
Location definitions for KSP1 Archipelago.

Three location sources (total 459 + N, where N = KSC starts by difficulty):

  1. KSC Starting Locations  (N = 5/10/15/20 by difficulty)
     Zero access requirements; AP fill places the items needed to bootstrap.

  2. Mission Event Locations  (244 total)
     11 Kerbin-specific + 233 per-body distance-scaled checks.
     Each event generates `check_scale` location checks (1/2/3).
     Landable bodies: 8 event types × scale.
     Non-landable (Jool, Kerbol): 3 event types × scale.

  3. Tech Tree Locations  (215 total)
     5 locations per node × 43 nodes.
     Access rule: player can earn enough science to afford the node's tier.

Location IDs use KSP1_BASE_ID + offset.  The registry includes all 20
possible KSC start slots so the world can create the correct subset at
runtime based on difficulty.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from BaseClasses import Location

from .bodies import ALL_BODIES, BODY_BY_NAME
from .tech_tree import TECH_TREE_LOCATION_NAMES

if TYPE_CHECKING:
    from .world import KSP1World

KSP1_BASE_ID = 7_700_000

# Offset ranges (all within location namespace; no collision with item offsets)
_KSC_OFFSET_START = 0          # KSC starts: 0-19
_KERBIN_OFFSET_START = 20      # Kerbin special: 20-30
_MISSION_OFFSET_START = 31     # Per-body mission events: 31-470
_TECH_OFFSET_START = 500       # Tech tree: 500-714


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

#: Bodies excluded from the per-body event table (have their own Kerbin events).
_KERBIN_EXCLUDED: frozenset[str] = frozenset({"Kerbin"})

# ---------------------------------------------------------------------------
# KSC starting locations (registry includes all 20; world creates N of them)
# ---------------------------------------------------------------------------

MAX_KSC_STARTS = 20

KSC_LOCATION_NAMES: list[str] = [
    f"KSC Start {i + 1}" for i in range(MAX_KSC_STARTS)
]

# ---------------------------------------------------------------------------
# Kerbin-specific mission locations (11 total, fixed)
# ---------------------------------------------------------------------------

KERBIN_LOCATION_NAMES: list[str] = [
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

assert len(KERBIN_LOCATION_NAMES) == 11

# ---------------------------------------------------------------------------
# Per-body mission location names (233 total, generated from body data)
# ---------------------------------------------------------------------------

def _build_mission_locations() -> list[str]:
    """
    Generate all per-body scaled mission location names.
    Order: body order in ALL_BODIES (skipping Kerbin), then events, then slots.
    """
    names: list[str] = []
    for body in ALL_BODIES:
        if body.name in _KERBIN_EXCLUDED:
            continue
        events = LANDABLE_EVENTS if body.can_land else ORBITAL_ONLY_EVENTS
        for event in events:
            for slot in range(1, body.check_scale + 1):
                names.append(f"{body.name} {event} {slot}")
    return names


MISSION_LOCATION_NAMES: list[str] = _build_mission_locations()

# Verify the per-body count matches the plan (233 checks)
assert len(MISSION_LOCATION_NAMES) == 233, (
    f"Expected 233 per-body mission locations, got {len(MISSION_LOCATION_NAMES)}"
)

# ---------------------------------------------------------------------------
# Build the full LOCATION_TABLE (name → id offset)
# ---------------------------------------------------------------------------

def _build_location_table() -> dict[str, int]:
    table: dict[str, int] = {}
    offset = _KSC_OFFSET_START
    for name in KSC_LOCATION_NAMES:
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
    body = BODY_BY_NAME[body_name]
    return [f"{body_name} {event} {i}" for i in range(1, body.check_scale + 1)]


# ---------------------------------------------------------------------------
# Location creation (called by world.create_regions)
# ---------------------------------------------------------------------------

def create_all_locations(world: KSP1World) -> None:
    """
    Create and attach all locations to the Menu region.

    KSC start locations: only the first N (by difficulty) are created.
    All mission and tech tree locations are always created.
    """
    from .options import Difficulty

    difficulty = world.options.difficulty.value
    ksc_counts = {
        Difficulty.option_casual: 20,
        Difficulty.option_normal: 15,
        Difficulty.option_expert: 10,
        Difficulty.option_insane: 5,
    }
    num_ksc = ksc_counts[difficulty]

    menu = world.get_region("Menu")

    # KSC starts (zero-requirement bootstrapping locations)
    ksc_locs = {
        name: LOCATION_NAME_TO_ID[name]
        for name in KSC_LOCATION_NAMES[:num_ksc]
    }
    menu.add_locations(ksc_locs, KSP1Location)

    # Kerbin-specific events
    kerbin_locs = {name: LOCATION_NAME_TO_ID[name] for name in KERBIN_LOCATION_NAMES}
    menu.add_locations(kerbin_locs, KSP1Location)

    # Per-body mission events
    mission_locs = {name: LOCATION_NAME_TO_ID[name] for name in MISSION_LOCATION_NAMES}
    menu.add_locations(mission_locs, KSP1Location)

    # Tech tree node slots
    tech_locs = {name: LOCATION_NAME_TO_ID[name] for name in TECH_TREE_LOCATION_NAMES}
    menu.add_locations(tech_locs, KSP1Location)
