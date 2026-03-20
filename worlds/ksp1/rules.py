"""
Access rules for KSP1 Archipelago locations and victory conditions.

Three rule families:

  1. Mission event rules  — delegate to get_capability() body access profiles.
     All check-slots for one event share a single rule (they trigger together).

  2. Tech tree rules  — science heuristic.  Gate on cumulative science the
     player can earn from accessible bodies given their current instruments.

  3. Victory conditions  — goal-specific rules set on the completion event.

Golden rule: when in doubt, say something is NOT achievable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState

from .bodies import ALL_BODIES, BODY_BY_NAME, science_budget
from .capability import get_capability
from .locations import (
    KERBIN_LOCATION_NAMES,
    MISSION_LOCATION_NAMES,
    TECH_TREE_LOCATION_NAMES,
    event_location_names,
)
from .options import Difficulty, Goal
from .tech_tree import NODES_BY_TIER, NODE_BY_ID, cumulative_tier_cost, TECH_NODES

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

# Science needed to declare the tech tree complete (buy all 43 nodes)
_TECH_TREE_COMPLETE_SCIENCE = cumulative_tier_cost(9)


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
    has_thermo = state.has("Thermometer", player)
    has_baro = state.has("Barometer", player)
    has_capsule = cap.has_capsule

    total = 0.0
    for body in ALL_BODIES:
        body_cap = cap.bodies.get(body.name)
        if body_cap is None or not body_cap.can_orbit_low:
            continue
        can_land_crewed = body_cap.can_land_crewed
        total += science_budget(body, has_thermo, has_baro, has_capsule, can_land_crewed)

    return total * _SCIENCE_SAFETY[difficulty]


def _can_afford_tier(state: CollectionState, player: int, tier: int, difficulty: int) -> bool:
    """Return True if the player's accessible science budget can cover all nodes through *tier*."""
    return _accessible_science(state, player, difficulty) >= cumulative_tier_cost(tier)


# ---------------------------------------------------------------------------
# Rule factories
# ---------------------------------------------------------------------------

def _make_tier_rule(player: int, tier: int, difficulty: int) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return _can_afford_tier(state, player, tier, difficulty)
    return rule


def _make_science_threshold_rule(
    player: int, threshold: float, difficulty: int
) -> Callable[[CollectionState], bool]:
    """Return a rule that passes when accessible science * safety >= threshold."""
    safety = _SCIENCE_SAFETY[difficulty]
    def rule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        has_thermo = state.has("Thermometer", player)
        has_baro = state.has("Barometer", player)
        has_capsule = cap.has_capsule
        total = 0.0
        for body in ALL_BODIES:
            body_cap = cap.bodies.get(body.name)
            if body_cap is None or not body_cap.can_orbit_low:
                continue
            total += science_budget(
                body, has_thermo, has_baro, has_capsule,
                body_cap.can_land_crewed
            )
        return total * safety >= threshold
    return rule


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def set_all_rules(world: KSP1World) -> None:
    player = world.player
    difficulty = world.options.difficulty.value

    _set_kerbin_rules(world, player)
    _set_mission_rules(world, player)
    _set_tech_tree_rules(world, player, difficulty)


def set_completion_condition(world: KSP1World) -> None:
    player = world.player
    difficulty = world.options.difficulty.value
    goal = world.options.goal.value

    _place_victory_event(world, player, goal, difficulty)


# ---------------------------------------------------------------------------
# Kerbin-specific location rules
# ---------------------------------------------------------------------------

def _set_kerbin_rules(world: KSP1World, player: int) -> None:
    """
    Rules for the 11 Kerbin-specific locations.

    Altitude milestones require progressively more delta-v (orbit proxy).
    EVA in Orbit requires a capsule + orbit capability.
    First Staging requires a decoupler.
    """

    def has_orbit(state: CollectionState) -> bool:
        return get_capability(state, player).bodies["Kerbin"].can_orbit_low

    def has_staging(state: CollectionState) -> bool:
        return get_capability(state, player).staging_tier >= 1

    def has_crewed_orbit(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        return cap.has_capsule and cap.bodies["Kerbin"].can_orbit_low

    # Altitude milestones — proxy: increasingly tight orbit check
    # In practice, 5-55 km altitudes require successively more dv.
    # We gate them all on "can orbit Kerbin" (conservative — orbit needs more dv).
    for name in KERBIN_LOCATION_NAMES:
        if "km Altitude" in name or name == "Kerbin Orbit":
            loc = world.get_location(name)
            loc.access_rule = has_orbit
        elif name == "Kerbin EVA in Orbit":
            loc = world.get_location(name)
            loc.access_rule = has_crewed_orbit
        elif name == "Kerbin First Staging":
            loc = world.get_location(name)
            loc.access_rule = has_staging
        # "Kerbin Splashdown" has no rule (land in ocean = no special gear)


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
    def _can_orbit(state: CollectionState) -> bool:
        return get_capability(state, player).bodies.get(body_name,
               type("_", (), {"can_orbit_low": False})()).can_orbit_low  # type: ignore[return-value]

    if event in ("Flyby", "SOI Leave", "Orbit"):
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and bp.can_orbit_low
        return rule

    if event == "Landing":
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and bp.can_land_unmanned
        return rule

    if event in ("Crewed Landing", "Flag Plant"):
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get(body_name)
            return bp is not None and bp.can_land_crewed
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
    from .locations import LANDABLE_EVENTS, ORBITAL_ONLY_EVENTS

    for body in ALL_BODIES:
        if body.name == "Kerbin":
            continue
        events = LANDABLE_EVENTS if body.can_land else ORBITAL_ONLY_EVENTS
        for event in events:
            rule = _mission_rule_for_event(player, body.name, event)
            for loc_name in event_location_names(body.name, event):
                loc = world.get_location(loc_name)
                loc.access_rule = rule


# ---------------------------------------------------------------------------
# Tech tree location rules
# ---------------------------------------------------------------------------

def _set_tech_tree_rules(world: KSP1World, player: int, difficulty: int) -> None:
    """
    Each tech tree slot is gated on the player being able to earn enough
    science to afford all nodes through that slot's tier.
    """
    for node in TECH_NODES:
        rule = _make_tier_rule(player, node.tier, difficulty)
        for slot in range(1, 6):
            loc_name = f"{node.display_name} {slot}"
            loc = world.get_location(loc_name)
            loc.access_rule = rule


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


def _place_victory_event(
    world: KSP1World, player: int, goal: int, difficulty: int
) -> None:
    """
    Create an event location named "Victory" and set the completion condition.
    The event location has a locked "Victory" item and no location ID (event).
    """
    from BaseClasses import Region
    from .items import create_item

    menu = world.get_region("Menu")
    victory_location = world.create_location("Victory", None, menu)
    victory_location.place_locked_item(create_item(world, "Victory"))

    # Assign the access rule based on the selected goal
    rule = _make_goal_rule(player, goal, difficulty)
    victory_location.access_rule = rule

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
        def rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            bp = cap.bodies.get("Eve")
            return bp is not None and (bp.can_return_to_kerbin or bp.can_return_crewed)
        return rule

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
        # Victory when the player can afford all 43 nodes
        return _make_science_threshold_rule(player, _TECH_TREE_COMPLETE_SCIENCE, difficulty)

    # Fallback (should never be reached)
    def rule(state: CollectionState) -> bool:
        return True
    return rule
