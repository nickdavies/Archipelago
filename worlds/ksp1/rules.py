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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState, ItemClassification

from .bodies import ALL_BODIES, BODY_BY_NAME, BodyName, MissionType, science_budget
from .capability import get_capability
from .items import ITEM_TABLE, PROGRESSIVE_RD_NAME, PROGRESSIVE_PART_ITEM_NAMES, SCIENCE_PACK_NAMES
from .locations import (
    EVENT_BY_NAME,
    EventName,
    KERBIN_LOCATIONS,
    KERBIN_SYSTEM_BODY_NAMES,
    KSC_BIOME_NAMES,
    KERBIN_LOCATION_NAMES,
    MISSION_LOCATION_NAMES,
    MissionLocation,
    STARTING_INV_COUNTS,
    STARTING_INV_NAMES,
    TECH_SLOTS_BY_DIFFICULTY,
    TechTreeLocation,
    event_locations,
)
from .options import Difficulty, Goal, ItemPacing
from .tech_tree import MAX_TIER, MAX_RD_BAND, cumulative_tier_cost, TECH_NODES, LEAF_TECH_NODES

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
        if body_cap is None or not body_cap.access.get(EventName.ORBIT, False):
            continue
        total += science_budget(
            body, cap.has_thermometer, cap.has_barometer,
            cap.has_capsule, body_cap.access.get(EventName.CREWED_LANDING, False),
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
            if body_cap is None or not body_cap.access.get(EventName.ORBIT, False):
                continue
            total += science_budget(
                body, cap.has_thermometer, cap.has_barometer,
                cap.has_capsule, body_cap.access.get(EventName.CREWED_LANDING, False),
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
    if world.goal_spec.is_kerbin_system_only:
        _set_interplanetary_item_rules(world, player)


def set_completion_condition(world: KSP1World, goal_spec: GoalSpec) -> None:
    player = world.player
    difficulty = world.options.difficulty.value

    _set_victory_rules(world, player, goal_spec, difficulty)


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
#
# Mission types and thresholds are defined in locations.KERBIN_LOCATIONS
# (single source of truth).  This module maps mission_type → access rule.
# ---------------------------------------------------------------------------

def _make_altitude_rule(player: int, threshold_km: float) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return get_capability(state, player).sounding_altitude_km >= threshold_km
    return rule


def _make_first_launch_rule(player: int) -> Callable[[CollectionState], bool]:
    """Any propulsion OR capsule (kerbal EVA counts as launch)."""
    def rule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        return cap.sounding_altitude_km > 0 or cap.has_capsule
    return rule


def _make_first_landing_rule(player: int) -> Callable[[CollectionState], bool]:
    """Propulsion + safe descent OR capsule (EVA landing)."""
    def rule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        if cap.sounding_altitude_km > 0 and (cap.has_parachutes or cap.has_throttleable_engine):
            return True
        return cap.has_capsule
    return rule


def _make_staging_rule(player: int) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return get_capability(state, player).staging_tier >= 1
    return rule


def _make_splashdown_rule(player: int, threshold_km: float) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        cap = get_capability(state, player)
        return (cap.sounding_altitude_km >= threshold_km
                and (cap.has_parachutes or cap.has_throttleable_engine))
    return rule


_KERBIN_RULE_FACTORIES = {
    MissionType.SOUNDING: lambda player, loc: _make_altitude_rule(player, loc.threshold_km or 0.0),
    MissionType.FIRST_LAUNCH: lambda player, loc: _make_first_launch_rule(player),
    MissionType.FIRST_LANDING: lambda player, loc: _make_first_landing_rule(player),
    MissionType.FIRST_STAGING: lambda player, loc: _make_staging_rule(player),
    MissionType.SPLASHDOWN: lambda player, loc: _make_splashdown_rule(player, loc.threshold_km or 1.0),
}


def _set_kerbin_rules(world: KSP1World, player: int) -> None:
    """Apply access rules to all Kerbin-specific locations.

    Iterates KERBIN_LOCATIONS and dispatches to the appropriate rule factory
    based on mission_type.
    """
    for loc in KERBIN_LOCATIONS:
        factory = _KERBIN_RULE_FACTORIES.get(loc.mission_type)
        if factory is None:
            raise ValueError(f"Unknown Kerbin mission type: {loc.mission_type!r}")
        world.get_location(loc.name).access_rule = factory(player, loc)


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
    # Tylo/Laythe returns cascade too much payload mass for the optimizer.
    if body_name == BodyName.EVE and event in (EventName.RETURN, EventName.SAMPLE_RETURN):
        return _make_all_parts_rule(player)
    if body_name in (BodyName.TYLO, BodyName.LAYTHE) and event in (EventName.RETURN, EventName.SAMPLE_RETURN):
        return _make_all_parts_rule(player)

    event_def = EVENT_BY_NAME.get(event)
    if event_def is None:
        def rule(state: CollectionState) -> bool:
            return False
        return rule

    def rule(state: CollectionState) -> bool:
        bp = get_capability(state, player).bodies.get(body_name)
        return bp is not None and bp.access.get(event, False)
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
            for loc in event_locations(body.name, event):
                world.get_location(str(loc)).access_rule = rule


# ---------------------------------------------------------------------------
# Item pacing rules (item_rules on early locations)
# ---------------------------------------------------------------------------

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
    num_starting = STARTING_INV_COUNTS[difficulty]
    power_rule = _make_power_rule(player, item_tiers, max_tier=1)
    for name in STARTING_INV_NAMES[:num_starting]:
        add_item_rule(world.get_location(name), power_rule)

    # Band B: KSC biomes + early Kerbin — reject tier 2
    for name in KSC_BIOME_NAMES:
        add_item_rule(world.get_location(name), power_rule)
    for name in KERBIN_LOCATION_NAMES:
        add_item_rule(world.get_location(name), power_rule)
    # Early Kerbin mission events (everything except Flyby/SOI Leave which need escape)
    for event in (EventName.ORBIT, EventName.EVA_IN_ORBIT, EventName.LANDING, EventName.CREWED_LANDING, EventName.FLAG_PLANT, EventName.RETURN, EventName.SAMPLE_RETURN):
        for loc in event_locations(BodyName.KERBIN, event):
            add_item_rule(world.get_location(str(loc)), power_rule)

    # Band C: Early tech tree (tiers 1-3) — reject tier 2, strict mode only
    if pacing >= ItemPacing.option_strict:
        for node in TECH_NODES:
            if node.tier > _EARLY_TECH_MAX_TIER:
                continue
            for slot in range(1, num_slots + 1):
                name = str(TechTreeLocation(node.display_name, slot))
                add_item_rule(world.get_location(name), _make_power_rule(player, item_tiers, max_tier=1))

    # Science pack restriction on early tech tree (both gentle and strict)
    science_rule = _make_science_pack_rule()
    for node in TECH_NODES:
        if node.tier > _EARLY_TECH_MAX_TIER:
            continue
        for slot in range(1, num_slots + 1):
            name = str(TechTreeLocation(node.display_name, slot))
            add_item_rule(world.get_location(name), science_rule)


# ---------------------------------------------------------------------------
# Interplanetary progression restriction (Kerbin-system-only goals)
# ---------------------------------------------------------------------------

def _set_interplanetary_item_rules(world: KSP1World, player: int) -> None:
    """
    For Kerbin-system-only goals, prevent progression items from appearing
    in interplanetary mission locations.

    Uses item_rules (not exclude_locations) so that useful and filler items
    can still fill these slots — avoids FillError at higher difficulties.
    """
    from worlds.generic.Rules import add_item_rule
    from .locations import MISSION_LOCATIONS

    def no_advancement(item) -> bool:
        return item.player != player or not item.advancement

    for loc in MISSION_LOCATIONS:
        if loc.body not in KERBIN_SYSTEM_BODY_NAMES:
            add_item_rule(world.get_location(str(loc)), no_advancement)


# ---------------------------------------------------------------------------
# Victory conditions — GoalSpec system
# ---------------------------------------------------------------------------

# All derived from bodies.py — single source of truth.
_ALL_LANDABLE_BODIES: tuple[BodyName, ...] = tuple(
    b.name for b in ALL_BODIES if b.can_land
)

# Bodies whose return/sample-return rules use the all-parts proxy
# (capability system can't model their ascent profiles).
_ALL_PARTS_PROXY_BODIES: frozenset[BodyName] = frozenset(
    b.name for b in ALL_BODIES if b.all_parts_proxy
)

# Landable bodies with normal (non-proxy) return profiles, excluding Kerbin.
_STANDARD_RETURN_BODIES: tuple[BodyName, ...] = tuple(
    b.name for b in ALL_BODIES
    if b.can_land and not b.all_parts_proxy and b.name != BodyName.KERBIN
)

@dataclass(frozen=True)
class GoalSpec:
    """Decomposed goal: every goal (preset or custom) becomes one of these."""
    display_name: str
    flag_bodies: tuple[BodyName, ...] = ()
    return_bodies: tuple[BodyName, ...] = ()
    sample_return_bodies: tuple[BodyName, ...] = ()
    complete_tech_tree: bool = False

    @property
    def is_kerbin_system_only(self) -> bool:
        """True when every goal body is in the Kerbin system (Kerbin/Mun/Minmus)."""
        if self.complete_tech_tree:
            return False
        all_bodies = set(self.flag_bodies) | set(self.return_bodies) | set(self.sample_return_bodies)
        return bool(all_bodies) and all_bodies <= KERBIN_SYSTEM_BODY_NAMES


_PRESET_GOALS: dict[int, GoalSpec] = {
    Goal.option_duna_return: GoalSpec(
        display_name="Duna Return",
        return_bodies=(BodyName.DUNA,),
    ),
    Goal.option_eeloo_return: GoalSpec(
        display_name="Eeloo Return",
        return_bodies=(BodyName.EELOO,),
    ),
    Goal.option_eve_return: GoalSpec(
        display_name="Eve Return",
        return_bodies=(BodyName.EVE,),
    ),
    Goal.option_flag_every_body: GoalSpec(
        display_name="Flag Every Body",
        flag_bodies=_ALL_LANDABLE_BODIES,
    ),
    Goal.option_standard_returns: GoalSpec(
        display_name="Standard Returns",
        return_bodies=_STANDARD_RETURN_BODIES,
    ),
    Goal.option_standard_sample_returns: GoalSpec(
        display_name="Standard Sample Returns",
        sample_return_bodies=_STANDARD_RETURN_BODIES,
    ),
    Goal.option_complete_tech_tree: GoalSpec(
        display_name="Complete Tech Tree",
        complete_tech_tree=True,
    ),
    Goal.option_mun_flag: GoalSpec(
        display_name="Mun Flag Plant",
        flag_bodies=(BodyName.MUN,),
    ),
    Goal.option_mun_sample_return: GoalSpec(
        display_name="Mun Sample Return",
        sample_return_bodies=(BodyName.MUN,),
    ),
}


def resolve_goal_spec(options) -> GoalSpec:
    """Build a GoalSpec from the player's option values.

    Raises if the configuration is ambiguous or incomplete.
    """
    goal_value = options.goal.value
    has_body_lists = bool(
        options.flag_bodies.value
        or options.return_bodies.value
        or options.sample_return_bodies.value
    )

    if goal_value != Goal.option_custom and has_body_lists:
        raise RuntimeError(
            f"Body-list options (flag_bodies, return_bodies, sample_return_bodies) "
            f"are set but goal is '{options.goal.current_option_name}', not 'custom'. "
            f"Set goal to 'custom' to use body-list options."
        )

    if goal_value == Goal.option_custom:
        if not has_body_lists:
            raise RuntimeError(
                "Goal is 'custom' but all body-list options are empty. "
                "Set at least one of flag_bodies, return_bodies, or sample_return_bodies."
            )
        # Build display name from the body lists.
        parts = []
        if options.flag_bodies.value:
            parts.append("Flag " + ", ".join(sorted(options.flag_bodies.value)))
        if options.return_bodies.value:
            parts.append("Return " + ", ".join(sorted(options.return_bodies.value)))
        if options.sample_return_bodies.value:
            parts.append("Sample Return " + ", ".join(sorted(options.sample_return_bodies.value)))
        display = "Custom: " + " + ".join(parts)

        return GoalSpec(
            display_name=display,
            flag_bodies=tuple(sorted(options.flag_bodies.value)),
            return_bodies=tuple(sorted(options.return_bodies.value)),
            sample_return_bodies=tuple(sorted(options.sample_return_bodies.value)),
        )

    spec = _PRESET_GOALS.get(goal_value)
    if spec is None:
        raise RuntimeError(f"Unknown goal value: {goal_value}")
    return spec


def goal_spec_location_names(spec: GoalSpec) -> list[str]:
    """Return sentinel location names whose checks indicate goal completion."""
    names: list[str] = []
    for b in spec.flag_bodies:
        names.append(str(MissionLocation(b, EventName.FLAG_PLANT, 1)))
    for b in spec.return_bodies:
        names.append(str(MissionLocation(b, EventName.RETURN, 1)))
    for b in spec.sample_return_bodies:
        names.append(str(MissionLocation(b, EventName.SAMPLE_RETURN, 1)))
    if spec.complete_tech_tree:
        for n in LEAF_TECH_NODES:
            names.append(str(TechTreeLocation(n.display_name, 1)))
    return names


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
    world: KSP1World, player: int, spec: GoalSpec, difficulty: int
) -> None:
    """Set the access rule and completion condition on the Victory event."""
    victory_location = world.get_location("Victory")
    victory_location.access_rule = _make_goal_spec_rule(player, spec, difficulty)

    world.multiworld.completion_condition[player] = (
        lambda state: state.can_reach("Victory", "Location", player)
    )


def _make_goal_spec_rule(
    player: int, spec: GoalSpec, difficulty: int
) -> Callable[[CollectionState], bool]:
    """Build a composite access rule from a GoalSpec."""
    sub_rules: list[Callable[[CollectionState], bool]] = []

    # Flag bodies
    if spec.flag_bodies:
        flag_bodies = spec.flag_bodies
        def flag_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in flag_bodies:
                bp = cap.bodies.get(b)
                if bp is None or not bp.access.get(EventName.FLAG_PLANT, False):
                    return False
            return True
        sub_rules.append(flag_rule)

    # Return bodies
    proxy_return = [b for b in spec.return_bodies if b in _ALL_PARTS_PROXY_BODIES]
    normal_return = [b for b in spec.return_bodies if b not in _ALL_PARTS_PROXY_BODIES]
    if normal_return:
        nr = tuple(normal_return)
        def return_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in nr:
                bp = cap.bodies.get(b)
                if bp is None or not bp.access.get(EventName.RETURN, False):
                    return False
            return True
        sub_rules.append(return_rule)
    if proxy_return:
        sub_rules.append(_make_all_parts_rule(player))

    # Sample return bodies
    proxy_sample = [b for b in spec.sample_return_bodies if b in _ALL_PARTS_PROXY_BODIES]
    normal_sample = [b for b in spec.sample_return_bodies if b not in _ALL_PARTS_PROXY_BODIES]
    if normal_sample:
        ns = tuple(normal_sample)
        def sample_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in ns:
                bp = cap.bodies.get(b)
                if bp is None or not bp.access.get(EventName.SAMPLE_RETURN, False):
                    return False
            return True
        sub_rules.append(sample_rule)
    if proxy_sample:
        sub_rules.append(_make_all_parts_rule(player))

    # Complete tech tree
    if spec.complete_tech_tree:
        science_rule = _make_science_threshold_rule(player, _TECH_TREE_COMPLETE_SCIENCE, difficulty)
        def tech_rule(state: CollectionState) -> bool:
            if not state.has(PROGRESSIVE_RD_NAME, player, MAX_RD_BAND):
                return False
            return science_rule(state)
        sub_rules.append(tech_rule)

    if not sub_rules:
        # Should not happen — resolve_goal_spec prevents empty specs.
        def always_true(state: CollectionState) -> bool:
            return True
        return always_true

    if len(sub_rules) == 1:
        return sub_rules[0]

    def composite(state: CollectionState) -> bool:
        return all(r(state) for r in sub_rules)
    return composite
