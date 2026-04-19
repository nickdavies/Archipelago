"""
Access rules for KSP1 Archipelago locations and victory conditions.

Rule families:

  1. Mission event rules  — delegate to get_capability() body access profiles.
     All check-slots for one event share a single rule (they trigger together).

  2. Victory conditions  — goal-specific rules set on the completion event.

  3. Item pacing rules  — item_rules on early locations to prevent
     high-tier items from appearing too early.

Tech tree rules are region entrance rules (see regions.py).

Golden rule: when in doubt, say something is NOT achievable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState, ItemClassification

from .bodies import ALL_BODIES, BODY_BY_NAME, science_budget
from .capability import get_capability
from .items import ITEM_TABLE, PROGRESSIVE_RD_NAME, PROGRESSIVE_PART_ITEM_NAMES, SCIENCE_PACK_NAMES
from .locations import (
    EVENT_SCALE,
    KSC_BIOME_NAMES,
    KERBIN_LOCATION_NAMES,
    MISSION_LOCATION_NAMES,
    STARTING_INV_NAMES,
    TECH_SLOTS_BY_DIFFICULTY,
    event_location_names,
)
from .options import Difficulty, Goal, ItemPacing
from .tech_tree import MAX_TIER, MAX_RD_BAND, cumulative_tier_cost, TECH_NODES

if TYPE_CHECKING:
    from .world import KSP1World

# ---------------------------------------------------------------------------
# Safety factors for science heuristic (fraction of estimated budget counted)
# ---------------------------------------------------------------------------

_SCIENCE_SAFETY: dict[int, float] = {
    Difficulty.option_casual: 0.50,
    Difficulty.option_normal: 0.70,
    Difficulty.option_expert: 0.85,
    Difficulty.option_insane: 1.00,
}

# Science needed to declare the tech tree complete (buy all 62 nodes)
_TECH_TREE_COMPLETE_SCIENCE = cumulative_tier_cost(MAX_TIER)

# All progression-classified items (individual parts + progressive part items).
# Eve Return/Sample Return require these — the capability system can't compute
# Eve ascent so we use this as a proxy for "you have everything needed."
# Progressive R&D is excluded (separate tech tree gate, not rocket capability).
_ALL_PROGRESSION_ITEMS: frozenset[str] = frozenset(
    name for name, (_, cls) in ITEM_TABLE.items()
    if cls == ItemClassification.progression
) | PROGRESSIVE_PART_ITEM_NAMES


def _make_all_parts_rule(player: int) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return state.has_all(_ALL_PROGRESSION_ITEMS, player)
    return rule


# ---------------------------------------------------------------------------
# Science heuristic helpers
# ---------------------------------------------------------------------------

def _accessible_science(state: CollectionState, player: int, difficulty: int) -> float:
    """
    Estimate the total science the player can earn from all bodies they can
    currently reach, given their current instrument and crew equipment.

    Multiplied by the difficulty safety factor before returning.
    """
    cap = get_capability(state, player)

    total = 0.0
    for body in ALL_BODIES:
        body_cap = cap.bodies.get(body.name)
        if body_cap is None or not body_cap.can_orbit_low:
            continue
        total += science_budget(
            body, cap.has_thermometer, cap.has_barometer,
            cap.has_capsule, body_cap.can_land_crewed,
        )

    return total * _SCIENCE_SAFETY[difficulty]


def _can_afford_tier(state: CollectionState, player: int, tier: int, difficulty: int) -> bool:
    """Return True if the player's accessible science budget can cover all nodes through *tier*."""
    return _accessible_science(state, player, difficulty) >= cumulative_tier_cost(tier)


# ---------------------------------------------------------------------------
# Rule factories
# ---------------------------------------------------------------------------

def _make_science_threshold_rule(
    player: int, threshold: float, difficulty: int
) -> Callable[[CollectionState], bool]:
    """Return a rule that passes when accessible science * safety >= threshold."""
    safety = _SCIENCE_SAFETY[difficulty]
    def rule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        total = 0.0
        for body in ALL_BODIES:
            body_cap = cap.bodies.get(body.name)
            if body_cap is None or not body_cap.can_orbit_low:
                continue
            total += science_budget(
                body, cap.has_thermometer, cap.has_barometer,
                cap.has_capsule, body_cap.can_land_crewed,
            )
        return total * safety >= threshold
    return rule


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def set_all_rules(world: KSP1World) -> None:
    player = world.player
    difficulty = world.options.difficulty.value

    _set_ksc_biome_rules(world, player)
    _set_kerbin_rules(world, player)
    _set_mission_rules(world, player)
    # Tech tree rules are now region entrance rules (see regions.py).
    _set_item_pacing_rules(world, player, difficulty)


def set_completion_condition(world: KSP1World) -> None:
    player = world.player
    difficulty = world.options.difficulty.value
    goal = world.options.goal.value

    _set_victory_rules(world, player, goal, difficulty)


# ---------------------------------------------------------------------------
# KSC biome location rules
# ---------------------------------------------------------------------------

def _can_do_ksc_science(state: CollectionState, player: int) -> bool:
    """
    Can the player do science at KSC biomes?

    Path 1: Crewed EVA (EVA report / surface sample) — just needs a capsule.
    Path 2: Rover with science instruments — probe + wheels + power + instrument.
    """
    cap = get_capability(state, player)
    if cap.has_capsule:
        return True
    if (cap.has_probe_core and cap.has_wheel
            and cap.power_profile != "none"
            and (cap.has_thermometer or cap.has_barometer)):
        return True
    return False


def _set_ksc_biome_rules(world: KSP1World, player: int) -> None:
    """Apply the KSC science rule to all KSC biome locations."""
    def rule(state: CollectionState) -> bool:
        return _can_do_ksc_science(state, player)

    for name in KSC_BIOME_NAMES:
        world.get_location(name).access_rule = rule


# ---------------------------------------------------------------------------
# Kerbin-specific location rules
# ---------------------------------------------------------------------------

# Altitude (km) each sounding-rocket milestone requires.
# Formula: h = Δv² · (twr−1) / (2·g·twr) — see capability._compute_sounding_altitude.
_ALTITUDE_THRESHOLDS_KM: dict[str, float] = {
    "Kerbin 5km Altitude":  5.0,
    "Kerbin 15km Altitude": 15.0,
    "Kerbin 25km Altitude": 25.0,
    "Kerbin 35km Altitude": 35.0,
    "Kerbin 45km Altitude": 45.0,
    "Kerbin 55km Altitude": 55.0,
    "Kerbin 70km Altitude": 70.0,
}


def _make_altitude_rule(player: int, threshold_km: float) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return get_capability(state, player).sounding_altitude_km >= threshold_km
    return rule


def _set_kerbin_rules(world: KSP1World, player: int) -> None:
    """
    Rules for the 12 Kerbin-specific locations.

    First Launch: any propulsion OR capsule (kerbal EVA counts as launch).
    First Landing: propulsion + safe descent OR capsule (EVA landing).
    First Crash: sounding rocket to 0.1 km (must get airborne to crash).
    Altitude milestones: sounding rocket must reach the stated altitude.
    First Staging: decoupler.
    Splashdown: 1 km sounding altitude + safe landing.
    """

    def has_any_propulsion_or_capsule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        # Engine path OR capsule-only path (kerbal EVA = "launch")
        return cap.sounding_altitude_km > 0 or cap.has_capsule

    def can_land_safely(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        # Engine path: sounding rocket + safe descent
        if cap.sounding_altitude_km > 0 and (cap.has_parachutes or cap.has_throttleable_engine):
            return True
        # EVA path: capsule alone (kerbal hops off pad, lands on feet)
        return cap.has_capsule

    def has_staging(state: CollectionState) -> bool:
        return get_capability(state, player).staging_tier >= 1

    world.get_location("Kerbin First Launch").access_rule = has_any_propulsion_or_capsule
    world.get_location("Kerbin First Crash").access_rule = _make_altitude_rule(player, 0.1)
    world.get_location("Kerbin First Landing").access_rule = can_land_safely

    for name, threshold_km in _ALTITUDE_THRESHOLDS_KM.items():
        world.get_location(name).access_rule = _make_altitude_rule(player, threshold_km)

    world.get_location("Kerbin First Staging").access_rule = has_staging

    def can_splashdown(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        return (cap.sounding_altitude_km >= 1.0
                and (cap.has_parachutes or cap.has_throttleable_engine))

    world.get_location("Kerbin Splashdown").access_rule = can_splashdown


# ---------------------------------------------------------------------------
# Per-body mission location rules
# ---------------------------------------------------------------------------

def _mission_rule_for_event(
    player: int, body_name: str, event: str
) -> Callable[[CollectionState], bool]:
    """
    Return the access rule for a given body + event combination.

    All check-slots for one event share this single rule (they fire together).
    """
    # Eve surface ascent is beyond the capability model.
    # Tylo/Laythe returns cascade too much payload mass to Kerbin ascent
    # for the greedy backward pass to handle (~200t Tylo, ~1100t Laythe).
    if body_name == "Eve" and event in ("Return", "Sample Return"):
        return _make_all_parts_rule(player)
    if body_name in ("Tylo", "Laythe") and event in ("Return", "Sample Return"):
        return _make_all_parts_rule(player)

    if event == "Flyby":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_escape
        return rule

    if event == "Orbit":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_orbit_low
        return rule

    if event == "EVA in Orbit":
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and bp.can_orbit_low and cap.has_capsule
        return rule

    if event == "SOI Leave":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_escape
        return rule

    if event == "Landing":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_land_unmanned
        return rule

    if event == "Crewed Landing":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_land_crewed
        return rule

    if event == "Flag Plant":
        def rule(state: CollectionState) -> bool:
            bp = get_capability(state, player).bodies.get(body_name)
            return bp is not None and bp.can_flag_plant
        return rule

    if event == "Return":
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and (bp.can_return_to_kerbin or bp.can_return_crewed)
        return rule

    if event == "Sample Return":
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and bp.can_sample_return
        return rule

    # Fallback (should not be reached)
    def rule(state: CollectionState) -> bool:
        return False
    return rule


def _set_mission_rules(world: KSP1World, player: int) -> None:
    """
    Apply access rules to all per-body mission event locations.

    All slots of the same event (e.g. "Eve Orbit 1" and "Eve Orbit 2") share
    the same rule — they check simultaneously when the player performs the event.
    """
    from .bodies import ALL_BODIES
    from .locations import get_body_events

    for body in ALL_BODIES:
        for event in get_body_events(body):
            rule = _mission_rule_for_event(player, body.name, event)
            for loc_name in event_location_names(body.name, event):
                loc = world.get_location(loc_name)
                loc.access_rule = rule


# ---------------------------------------------------------------------------
# Item pacing rules (item_rules on early locations)
# ---------------------------------------------------------------------------

# Starting inventory slot counts by difficulty (mirrors locations.py)
_STARTING_INV_COUNTS: dict[int, int] = {
    Difficulty.option_casual: 20,
    Difficulty.option_normal: 15,
    Difficulty.option_expert: 10,
    Difficulty.option_insane: 5,
}

# Early tech tree: tiers 1-3
_EARLY_TECH_MAX_TIER = 3


def _make_power_rule(player: int, item_tiers: dict[str, int], max_tier: int):
    """
    Item rule: reject KSP items above max_tier.

    Non-KSP items (other worlds in multiworld) always pass.
    Items not in item_tiers are tier 0 and always pass.
    """
    def rule(item) -> bool:
        return item.player != player or item_tiers.get(item.name, 0) <= max_tier
    return rule


def _make_science_pack_rule():
    """Item rule: reject science pack filler items."""
    names = SCIENCE_PACK_NAMES
    def rule(item) -> bool:
        return item.name not in names
    return rule


def _set_item_pacing_rules(world: KSP1World, player: int, difficulty: int) -> None:
    """
    Apply item_rules that prevent high-power items from appearing in
    early locations, creating a gradual power curve.

    Also restricts science packs from flooding early tech tree slots.
    """
    from worlds.generic.Rules import add_item_rule
    from .item_power import ITEM_TIERS

    pacing = world.options.item_pacing.value
    if pacing == ItemPacing.option_off:
        return

    item_tiers = ITEM_TIERS
    num_slots = TECH_SLOTS_BY_DIFFICULTY[difficulty]

    # Band A: Starting Inventory — reject tier 2
    num_starting = _STARTING_INV_COUNTS[difficulty]
    power_rule = _make_power_rule(player, item_tiers, max_tier=1)
    for name in STARTING_INV_NAMES[:num_starting]:
        add_item_rule(world.get_location(name), power_rule)

    # Band B: KSC biomes + early Kerbin — reject tier 2
    for name in KSC_BIOME_NAMES:
        add_item_rule(world.get_location(name), power_rule)
    for name in KERBIN_LOCATION_NAMES:
        add_item_rule(world.get_location(name), power_rule)
    # Early Kerbin mission events (everything except Flyby/SOI Leave which need escape)
    for event in ("Orbit", "EVA in Orbit", "Landing", "Crewed Landing", "Flag Plant", "Return", "Sample Return"):
        for slot in range(1, EVENT_SCALE[event] + 1):
            add_item_rule(world.get_location(f"Kerbin {event} {slot}"), power_rule)

    # Band C: Early tech tree (tiers 1-3) — reject tier 2, strict mode only
    if pacing >= ItemPacing.option_strict:
        for node in TECH_NODES:
            if node.tier > _EARLY_TECH_MAX_TIER:
                continue
            for slot in range(1, num_slots + 1):
                loc = world.get_location(f"{node.display_name} {slot}")
                add_item_rule(loc, _make_power_rule(player, item_tiers, max_tier=1))

    # Science pack restriction on early tech tree (both gentle and strict)
    science_rule = _make_science_pack_rule()
    for node in TECH_NODES:
        if node.tier > _EARLY_TECH_MAX_TIER:
            continue
        for slot in range(1, num_slots + 1):
            add_item_rule(world.get_location(f"{node.display_name} {slot}"), science_rule)


# ---------------------------------------------------------------------------
# Victory conditions
# ---------------------------------------------------------------------------

_STANDARD_RETURN_BODIES: tuple[str, ...] = (
    "Mun", "Minmus", "Moho", "Gilly", "Duna", "Ike",
    "Dres", "Vall", "Bop", "Pol", "Eeloo",
)

_ALL_LANDABLE_BODIES: tuple[str, ...] = (
    "Kerbin", "Mun", "Minmus", "Moho", "Eve", "Gilly",
    "Duna", "Ike", "Dres", "Laythe", "Vall", "Tylo",
    "Bop", "Pol", "Eeloo",
)

_GOAL_DISPLAY_NAMES: dict[int, str] = {
    Goal.option_duna_return: "Duna Return",
    Goal.option_eeloo_return: "Eeloo Return",
    Goal.option_flag_every_body: "Flag Every Body",
    Goal.option_standard_returns: "Standard Returns",
    Goal.option_standard_sample_returns: "Standard Sample Returns",
    Goal.option_complete_tech_tree: "Complete Tech Tree",
    Goal.option_eve_return: "Eve Return",
}


def goal_location_names(goal: int) -> list[str]:
    """Return the sentinel location names whose checks indicate goal completion.

    Each entry is the slot-1 location for a required event. The client caches
    these IDs at connect time and polls checkedLocationIds to detect victory.
    """
    if goal == Goal.option_duna_return:
        return ["Duna Return 1"]
    if goal == Goal.option_eeloo_return:
        return ["Eeloo Return 1"]
    if goal == Goal.option_eve_return:
        return ["Eve Return 1"]
    if goal == Goal.option_flag_every_body:
        return [f"{b} Flag Plant 1" for b in _ALL_LANDABLE_BODIES]
    if goal == Goal.option_standard_returns:
        return [f"{b} Return 1" for b in _STANDARD_RETURN_BODIES]
    if goal == Goal.option_standard_sample_returns:
        return [f"{b} Sample Return 1" for b in _STANDARD_RETURN_BODIES]
    if goal == Goal.option_complete_tech_tree:
        return [f"{n.display_name} 1" for n in TECH_NODES]
    return []


def goal_display_name(goal: int) -> str:
    """Return a human-readable label for the goal."""
    return _GOAL_DISPLAY_NAMES.get(goal, "Unknown")


def create_victory_location(world: KSP1World) -> None:
    """Create the Victory event location with a locked Victory item.

    Called during create_regions so location count is stable before set_rules.
    """
    from BaseClasses import Location
    from .items import create_item

    menu = world.get_region("Menu")
    victory_location = Location(world.player, "Victory", None, menu)
    menu.locations.append(victory_location)
    victory_location.place_locked_item(create_item(world, "Victory"))


def _set_victory_rules(
    world: KSP1World, player: int, goal: int, difficulty: int
) -> None:
    """Set the access rule and completion condition on the Victory event."""
    victory_location = world.get_location("Victory")
    victory_location.access_rule = _make_goal_rule(player, goal, difficulty)

    world.multiworld.completion_condition[player] = (
        lambda state: state.can_reach("Victory", "Location", player)
    )


def _make_goal_rule(
    player: int, goal: int, difficulty: int
) -> Callable[[CollectionState], bool]:
    if goal == Goal.option_duna_return:
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get("Duna")
            return bp is not None and (bp.can_return_to_kerbin or bp.can_return_crewed)
        return rule

    if goal == Goal.option_eeloo_return:
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get("Eeloo")
            return bp is not None and (bp.can_return_to_kerbin or bp.can_return_crewed)
        return rule

    if goal == Goal.option_eve_return:
        return _make_all_parts_rule(player)

    if goal == Goal.option_flag_every_body:
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            return all(
                cap.bodies.get(b, type("_", (), {"can_land_crewed": False})()).can_land_crewed  # type: ignore[return-value]
                for b in _ALL_LANDABLE_BODIES
            )
        return rule

    if goal == Goal.option_standard_returns:
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in _STANDARD_RETURN_BODIES:
                bp = cap.bodies.get(b)
                if bp is None or not (bp.can_return_to_kerbin or bp.can_return_crewed):
                    return False
            return True
        return rule

    if goal == Goal.option_standard_sample_returns:
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in _STANDARD_RETURN_BODIES:
                bp = cap.bodies.get(b)
                if bp is None or not bp.can_sample_return:
                    return False
            return True
        return rule

    if goal == Goal.option_complete_tech_tree:
        # Victory when the player has all R&D upgrades and can afford all 62 nodes
        science_rule = _make_science_threshold_rule(player, _TECH_TREE_COMPLETE_SCIENCE, difficulty)
        def rule(state: CollectionState) -> bool:
            if not state.has(PROGRESSIVE_RD_NAME, player, MAX_RD_BAND):
                return False
            return science_rule(state)
        return rule

    # Fallback (should never be reached)
    def rule(state: CollectionState) -> bool:
        return True
    return rule
