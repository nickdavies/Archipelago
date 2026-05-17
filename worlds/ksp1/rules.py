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

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, MissionType,
    home_system_bodies, science_budget,
)
from .capability import get_capability
from .items import ITEM_TABLE, PROGRESSIVE_RD_NAME, PROGRESSIVE_PART_ITEM_NAMES, SCIENCE_PACK_NAMES
from .locations import (
    EVENT_BY_NAME,
    EventName,
    KSC_BIOME_NAMES,
    MISSION_LOCATION_NAMES,
    MissionLocation,
    STARTING_INV_COUNTS,
    STARTING_INV_NAMES,
    TECH_SLOTS_BY_DIFFICULTY,
    TechTreeLocation,
    effective_starting_inv_count,
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
        body_cap = cap.bodies[body.name]
        if not body_cap.access[EventName.ORBIT]:
            continue
        total += science_budget(
            body, cap.has_thermometer, cap.has_barometer,
            cap.has_capsule, body_cap.access[EventName.CREWED_LANDING],
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
            body_cap = cap.bodies[body.name]
            if not body_cap.access[EventName.ORBIT]:
                continue
            total += science_budget(
                body, cap.has_thermometer, cap.has_barometer,
                cap.has_capsule, body_cap.access[EventName.CREWED_LANDING],
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
    _apply_home_system_local_exclusions(world)
    # Tech tree rules are now region entrance rules (see regions.py).
    _set_item_pacing_rules(world, player, difficulty)
    _set_early_bucket_item_bans(world, player, difficulty)
    if world.goal_spec.is_home_system_only(world.mission_builder.home):
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

    for name in world.location_builder.ksc_biome_names:
        world.get_location(name).access_rule = rule


# ---------------------------------------------------------------------------
# Kerbin-specific location rules
#
# Mission types and thresholds are owned by ``LocationBuilder`` in
# locations.py (single source of truth).  This module maps mission_type →
# access rule.
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
    """Apply access rules to the home-body specials.

    Iterates the active home's location set and dispatches to the appropriate
    rule factory based on mission_type.  Rule semantics are body-agnostic;
    the rules already operate against ``world.mission_builder.home`` via the
    capability system.
    """
    for loc in world.location_builder.locations:
        factory = _KERBIN_RULE_FACTORIES.get(loc.mission_type)
        if factory is None:
            raise ValueError(f"Unknown home mission type: {loc.mission_type!r}")
        world.get_location(loc.name).access_rule = factory(player, loc)


# ---------------------------------------------------------------------------
# Per-body mission location rules
# ---------------------------------------------------------------------------

def _mission_rule_for_event(
    player: int, body_name: str, event: str
) -> Callable[[CollectionState], bool]:
    """
    Return the capability-based access rule for a given body + event combination.

    All check-slots for one event share this rule (they fire together).
    Used when the model can verify the mission; the all-parts proxy is
    applied per-location by ``_set_mission_rules`` for locations the model
    can't verify from this home.
    """
    EVENT_BY_NAME[event]  # validate event exists; crash on typo

    def rule(state: CollectionState) -> bool:
        return get_capability(state, player).bodies[body_name].access[event]
    return rule


def _set_mission_rules(world: KSP1World, player: int) -> None:
    """
    Apply access rules to all per-body mission event locations.

    Per location: if it's listed in ``world.model_infeasible_locations``
    (the home-specific set the dv model can't verify), fall back to the
    "all progression items collected" proxy.  Otherwise the location
    shares the capability-based rule for its (body, event) pair.  This
    keeps location reachability aligned with the victory rule
    (``_make_goal_spec_rule``), which routes the same way per body/event.
    """
    from .bodies import ALL_BODIES
    from .locations import get_body_events

    infeasible = world.model_infeasible_locations
    proxy_rule = _make_all_parts_rule(player)

    for body in ALL_BODIES:
        for event in get_body_events(body):
            cap_rule = _mission_rule_for_event(player, body.name, event)
            for loc in event_locations(body.name, event):
                name = str(loc)
                world.get_location(name).access_rule = (
                    proxy_rule if name in infeasible else cap_rule
                )


def _apply_home_system_local_exclusions(world: KSP1World) -> None:
    """For ``home_system_local`` goals, mark every per-body mission location
    outside the home's SOI neighbourhood as ``EXCLUDED`` for progression.

    AP fill won't drop advancement or useful items at EXCLUDED locations,
    so progression items can't land at side-quest missions the player
    would have to detour into Kerbol heliocentric space to reach.  Goal
    bodies are guaranteed in-system by ``_validate_home_system_local``,
    so the goal path is never affected.  Filler can still place there.
    """
    from BaseClasses import LocationProgressType
    from .bodies import ALL_BODIES, home_system_bodies
    from .locations import get_body_events

    spec = world.goal_spec
    if not spec.home_system_local:
        return
    assert spec.home is not None
    in_system = home_system_bodies(spec.home)
    for body in ALL_BODIES:
        if body.name in in_system:
            continue
        for event in get_body_events(body):
            for loc in event_locations(body.name, event):
                world.get_location(str(loc)).progress_type = LocationProgressType.EXCLUDED


# ---------------------------------------------------------------------------
# Item pacing rules (item_rules on early locations)
# ---------------------------------------------------------------------------

# Early tech tree band: tiers 1-3
_EARLY_TECH_MAX_TIER = 3

# Items banned from early bands.  Kept explicit (no tier scorer) because the
# ratio-based tier system was overengineered for this small allowlist and
# kept creating phantom tier-2 items (bug: miniFuselage / MK1Fuselage).
#
# Why each is banned:
#   Progressive R&D    — gates tech tree bands; "least fun" check, keep late
#   Science Pack 100   — large science windfall doesn't belong in starter
#   Science Pack 250   — same
_EARLY_BANNED_ITEMS: frozenset[str] = frozenset({
    "Progressive R&D",
    "Science Pack 100",
    "Science Pack 250",
})


def _make_early_ban_rule(player: int):
    """Item rule: reject pacing-sensitive items for our player.

    Other-world items always pass.
    """
    banned = _EARLY_BANNED_ITEMS
    def rule(item) -> bool:
        return item.player != player or item.name not in banned
    return rule


def _make_science_pack_rule():
    """Item rule: reject science pack filler items."""
    names = SCIENCE_PACK_NAMES
    def rule(item) -> bool:
        return item.name not in names
    return rule


def _set_item_pacing_rules(world: KSP1World, player: int, difficulty: int) -> None:
    """
    Apply item_rules to early locations.

    Bans `_EARLY_BANNED_ITEMS` from Band A (starting inventory) and Band B
    (KSC biomes + Kerbin specials + early Kerbin events). Under strict
    pacing, also bans them from Band C (tier 1-3 tech tree).  Additionally
    blocks all science pack filler from early tech tree under any pacing.
    """
    from worlds.generic.Rules import add_item_rule

    pacing = world.options.item_pacing.value
    if pacing == ItemPacing.option_off:
        return

    num_slots = TECH_SLOTS_BY_DIFFICULTY[difficulty]
    num_starting = effective_starting_inv_count(world.options, difficulty)
    early_ban_rule = _make_early_ban_rule(player)

    # Band A: Starting Inventory
    for name in STARTING_INV_NAMES[:num_starting]:
        add_item_rule(world.get_location(name), early_ban_rule)

    # Band B: KSC biomes + home-body specials + early home events
    for name in world.location_builder.ksc_biome_names:
        add_item_rule(world.get_location(name), early_ban_rule)
    home = world.mission_builder.home
    for name in world.location_builder.names:
        add_item_rule(world.get_location(name), early_ban_rule)
    # Early home-body mission events (everything except Flyby/SOI Leave which need escape)
    for event in (EventName.ORBIT, EventName.EVA_IN_ORBIT, EventName.LANDING,
                  EventName.CREWED_LANDING, EventName.FLAG_PLANT,
                  EventName.RETURN, EventName.SAMPLE_RETURN):
        for loc in event_locations(home, event):
            add_item_rule(world.get_location(str(loc)), early_ban_rule)

    # Band C: Early tech tree (tiers 1-3) — strict mode only
    if pacing >= ItemPacing.option_strict:
        for node in TECH_NODES:
            if node.tier > _EARLY_TECH_MAX_TIER:
                continue
            for slot in range(1, num_slots + 1):
                name = str(TechTreeLocation(node.display_name, slot))
                add_item_rule(world.get_location(name), early_ban_rule)

    # Science pack restriction on early tech tree (both gentle and strict)
    science_rule = _make_science_pack_rule()
    for node in TECH_NODES:
        if node.tier > _EARLY_TECH_MAX_TIER:
            continue
        for slot in range(1, num_slots + 1):
            name = str(TechTreeLocation(node.display_name, slot))
            add_item_rule(world.get_location(name), science_rule)


# ---------------------------------------------------------------------------
# Early-bucket targeted item bans
# ---------------------------------------------------------------------------

def _set_early_bucket_item_bans(world: KSP1World, player: int, difficulty: int) -> None:
    """Ban Progressive Launch Pad from starter inventory only.

    The Pad item is a blow-open item: collecting it raises the launch-mass
    cap by a big jump and opens many bodies at once.  Keeping it out of
    the starter bucket spreads its discovery across the game.
    Sphere-ladder Rule B handles other early-bucket restrictions
    (e.g. the differentiators previously banned by the now-removed
    ``ban_differentiators_early`` option).
    """
    if not world.options.progressive_launch_pad:
        return

    from worlds.generic.Rules import add_item_rule
    from .items import PROGRESSIVE_LAUNCH_PAD_NAME

    num_starter = effective_starting_inv_count(world.options, difficulty)
    starter_bucket = list(STARTING_INV_NAMES[:num_starter])

    def starter_rule(item) -> bool:
        return item.player != player or item.name != PROGRESSIVE_LAUNCH_PAD_NAME
    for name in starter_bucket:
        add_item_rule(world.get_location(name), starter_rule)


# ---------------------------------------------------------------------------
# Interplanetary progression restriction (Kerbin-system-only goals)
# ---------------------------------------------------------------------------

def _set_interplanetary_item_rules(world: KSP1World, player: int) -> None:
    """
    For home-system-only goals, prevent progression items from appearing
    in interplanetary mission locations (locations outside the home
    body's local neighbourhood).

    Uses item_rules (not exclude_locations) so that useful and filler items
    can still fill these slots — avoids FillError at higher difficulties.
    """
    from worlds.generic.Rules import add_item_rule
    from .locations import MISSION_LOCATIONS

    home_system = home_system_bodies(world.mission_builder.home)

    def no_advancement(item) -> bool:
        return item.player != player or not item.advancement

    for loc in MISSION_LOCATIONS:
        if loc.body not in home_system:
            add_item_rule(world.get_location(str(loc)), no_advancement)


# ---------------------------------------------------------------------------
# Victory conditions — GoalSpec system
# ---------------------------------------------------------------------------

# All derived from bodies.py — single source of truth.
_ALL_LANDABLE_BODIES: tuple[BodyName, ...] = tuple(
    b.name for b in ALL_BODIES if b.can_land
)

@dataclass(frozen=True)
class GoalSpec:
    """Decomposed goal: every goal (preset or custom) becomes one of these.

    Carries the fully-materialized settings the world resolved at
    ``generate_early`` time: which bodies satisfy each victory event,
    the home body the player launches from (after any future
    randomization), and the home-system-local flag.

    ``home_system_local=True`` means the goal stays within the home
    body's local SOI neighbourhood (planet+moons or moon+parent+siblings).
    The world setup then forces non-home-system mission locations out
    of logic so AP fill places no progression items where the player
    would otherwise have to detour into Kerbol heliocentric space to
    collect them.  See ``home_system_bodies`` in bodies.py for the set.

    ``home`` is None on the raw ``_PRESET_GOALS`` templates; populated
    on every instance returned by ``resolve_goal_spec``.  Treat
    ``spec.home`` as authoritative inside the world (callers that still
    accept ``home`` separately are legacy; new code should read from the
    spec).
    """
    display_name: str
    flag_bodies: tuple[BodyName, ...] = ()
    return_bodies: tuple[BodyName, ...] = ()
    sample_return_bodies: tuple[BodyName, ...] = ()
    orbit_bodies: tuple[BodyName, ...] = ()
    flyby_bodies: tuple[BodyName, ...] = ()
    complete_tech_tree: bool = False
    home_system_local: bool = False
    home: BodyName | None = None

    def is_home_system_only(self, home: BodyName) -> bool:
        """True when every goal body is in ``home``'s local neighbourhood.

        For Kerbin-home this is the historical "Kerbin/Mun/Minmus only"
        check; for moon-homes it includes the parent planet and sibling
        moons; for vacuum/atmo planet-homes it's the home plus its
        moons.  Returns False if the goal requires the tech tree (which
        is a separate progression axis from body reach).
        """
        if self.complete_tech_tree:
            return False
        all_bodies = (
            set(self.flag_bodies)
            | set(self.return_bodies)
            | set(self.sample_return_bodies)
            | set(self.orbit_bodies)
            | set(self.flyby_bodies)
        )
        return bool(all_bodies) and all_bodies <= home_system_bodies(home)


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
    # Standard return / sample-return body lists are built per-world in
    # ``resolve_goal_spec`` once the proxy set is known (it depends on the
    # home body's reachable-with-full-kit mission graph).
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
    Goal.option_jool_moons_return: GoalSpec(
        display_name="Jool Moons Return",
        return_bodies=(
            BodyName.LAYTHE, BodyName.VALL, BodyName.TYLO,
            BodyName.BOP, BodyName.POL,
        ),
        home_system_local=True,
    ),
}


def resolve_goal_spec(options, home: BodyName,
                      model_infeasible_locations: frozenset[str]) -> GoalSpec:
    """Build a GoalSpec from the player's option values.

    Raises if the configuration is ambiguous or incomplete.

    ``home`` is filtered out of every body list — returning from / planting
    a flag on your starting body would be free, so it doesn't make sense
    as a goal regardless of preset.

    ``model_infeasible_locations`` is the per-world set of AP location
    names whose mission the dv model can't verify even with full kit
    (see ``scripts/generate_feasibility.py``).  Bodies whose RETURN /
    SAMPLE_RETURN are entirely in this set are excluded from the
    dynamically-built Standard Returns / Sample Returns body lists; if
    the player insists via custom presets, the completion rule falls
    back to the "all-parts collected" proxy.
    """
    goal_value = options.goal.value
    has_body_lists = bool(
        options.flag_bodies.value
        or options.return_bodies.value
        or options.sample_return_bodies.value
        or options.orbit_bodies.value
        or options.flyby_bodies.value
    )

    if goal_value != Goal.option_custom and has_body_lists:
        raise RuntimeError(
            f"Body-list options (flag_bodies, return_bodies, sample_return_bodies, "
            f"orbit_bodies, flyby_bodies) are set but goal is "
            f"'{options.goal.current_option_name}', not 'custom'. "
            f"Set goal to 'custom' to use body-list options."
        )

    if goal_value == Goal.option_custom:
        if not has_body_lists:
            raise RuntimeError(
                "Goal is 'custom' but all body-list options are empty. "
                "Set at least one of flag_bodies, return_bodies, sample_return_bodies, "
                "orbit_bodies, or flyby_bodies."
            )
        # Build display name from the body lists.
        parts = []
        if options.flag_bodies.value:
            parts.append("Flag " + ", ".join(sorted(options.flag_bodies.value)))
        if options.return_bodies.value:
            parts.append("Return " + ", ".join(sorted(options.return_bodies.value)))
        if options.sample_return_bodies.value:
            parts.append("Sample Return " + ", ".join(sorted(options.sample_return_bodies.value)))
        if options.orbit_bodies.value:
            parts.append("Orbit " + ", ".join(sorted(options.orbit_bodies.value)))
        if options.flyby_bodies.value:
            parts.append("Flyby " + ", ".join(sorted(options.flyby_bodies.value)))
        display = "Custom: " + " + ".join(parts)

        spec = GoalSpec(
            display_name=display,
            flag_bodies=tuple(sorted(options.flag_bodies.value)),
            return_bodies=tuple(sorted(options.return_bodies.value)),
            sample_return_bodies=tuple(sorted(options.sample_return_bodies.value)),
            orbit_bodies=tuple(sorted(options.orbit_bodies.value)),
            flyby_bodies=tuple(sorted(options.flyby_bodies.value)),
        )
    elif goal_value == Goal.option_standard_returns:
        standard_bodies = tuple(
            b for b in _ALL_LANDABLE_BODIES
            if not _all_locations_infeasible(b, EventName.RETURN,
                                              model_infeasible_locations)
        )
        spec = GoalSpec(
            display_name="Standard Returns",
            return_bodies=standard_bodies,
        )
    elif goal_value == Goal.option_standard_sample_returns:
        standard_bodies = tuple(
            b for b in _ALL_LANDABLE_BODIES
            if not _all_locations_infeasible(b, EventName.SAMPLE_RETURN,
                                              model_infeasible_locations)
        )
        spec = GoalSpec(
            display_name="Standard Sample Returns",
            sample_return_bodies=standard_bodies,
        )
    else:
        spec = _PRESET_GOALS.get(goal_value)
        if spec is None:
            raise RuntimeError(f"Unknown goal value: {goal_value}")

    materialized = _filter_home_from_spec(spec, home)
    _validate_home_system_local(materialized)
    return materialized


def _filter_home_from_spec(spec: GoalSpec, home: BodyName) -> GoalSpec:
    """Drop ``home`` from every body list (a goal can't ask the player to do
    a mission on their starting body — trivially achievable), and stamp
    ``home`` onto the returned spec so the spec is fully materialized.
    """
    def _strip(bodies: tuple[BodyName, ...]) -> tuple[BodyName, ...]:
        return tuple(b for b in bodies if b != home)
    return GoalSpec(
        display_name=spec.display_name,
        flag_bodies=_strip(spec.flag_bodies),
        return_bodies=_strip(spec.return_bodies),
        sample_return_bodies=_strip(spec.sample_return_bodies),
        orbit_bodies=_strip(spec.orbit_bodies),
        flyby_bodies=_strip(spec.flyby_bodies),
        complete_tech_tree=spec.complete_tech_tree,
        home_system_local=spec.home_system_local,
        home=home,
    )


def _validate_home_system_local(spec: GoalSpec) -> None:
    """If ``home_system_local`` is on, every goal body must sit in the home
    system.  Raised here so misconfigured presets / customs fail at
    ``generate_early`` time instead of producing an unsolvable seed.
    """
    if not spec.home_system_local:
        return
    assert spec.home is not None, "home must be set before validating home_system_local"
    valid = home_system_bodies(spec.home)
    all_bodies = (
        set(spec.flag_bodies)
        | set(spec.return_bodies)
        | set(spec.sample_return_bodies)
        | set(spec.orbit_bodies)
        | set(spec.flyby_bodies)
    )
    extras = all_bodies - valid
    if extras:
        from Options import OptionError
        raise OptionError(
            f"Goal '{spec.display_name}' is home_system_local but includes "
            f"bodies outside {sorted(valid)}: {sorted(extras)}.  Either pick "
            f"a compatible home or drop those bodies from the goal."
        )


def goal_spec_location_names(spec: GoalSpec) -> list[str]:
    """Return sentinel location names whose checks indicate goal completion."""
    names: list[str] = []
    for b in spec.flag_bodies:
        names.append(str(MissionLocation(b, EventName.FLAG_PLANT, 1)))
    for b in spec.return_bodies:
        names.append(str(MissionLocation(b, EventName.RETURN, 1)))
    for b in spec.sample_return_bodies:
        names.append(str(MissionLocation(b, EventName.SAMPLE_RETURN, 1)))
    for b in spec.orbit_bodies:
        names.append(str(MissionLocation(b, EventName.ORBIT, 1)))
    for b in spec.flyby_bodies:
        names.append(str(MissionLocation(b, EventName.FLYBY, 1)))
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


def _all_locations_infeasible(
    body: BodyName, event: EventName,
    model_infeasible_locations: frozenset[str],
) -> bool:
    """True iff every AP-location slot for ``(body, event)`` is in the
    model-infeasible set.  Used by goal building to filter bodies whose
    entire mission class is unverifiable from the current home."""
    locs = event_locations(body, event)
    return bool(locs) and all(str(loc) in model_infeasible_locations
                              for loc in locs)


def _set_victory_rules(
    world: KSP1World, player: int, spec: GoalSpec, difficulty: int
) -> None:
    """Set the access rule and completion condition on the Victory event."""
    victory_location = world.get_location("Victory")
    victory_location.access_rule = _make_goal_spec_rule(
        player, spec, difficulty, world.model_infeasible_locations,
    )

    world.multiworld.completion_condition[player] = (
        lambda state: state.can_reach("Victory", "Location", player)
    )


def _make_goal_spec_rule(
    player: int, spec: GoalSpec, difficulty: int,
    model_infeasible_locations: frozenset[str],
) -> Callable[[CollectionState], bool]:
    """Build a composite access rule from a GoalSpec.

    ``model_infeasible_locations`` is the per-world set of AP location
    names whose mission the dv model can't verify.  Mission classes
    where *every* slot for a body falls in this set route through the
    "all-parts collected" proxy rule instead of the dv-based access
    check.
    """
    sub_rules: list[Callable[[CollectionState], bool]] = []

    # Flag bodies
    if spec.flag_bodies:
        flag_bodies = spec.flag_bodies
        def flag_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in flag_bodies:
                if not cap.bodies[b].access[EventName.FLAG_PLANT]:
                    return False
            return True
        sub_rules.append(flag_rule)

    # Return bodies
    proxy_return = [b for b in spec.return_bodies
                    if _all_locations_infeasible(b, EventName.RETURN,
                                                   model_infeasible_locations)]
    normal_return = [b for b in spec.return_bodies if b not in proxy_return]
    if normal_return:
        nr = tuple(normal_return)
        def return_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in nr:
                if not cap.bodies[b].access[EventName.RETURN]:
                    return False
            return True
        sub_rules.append(return_rule)
    if proxy_return:
        sub_rules.append(_make_all_parts_rule(player))

    # Sample return bodies
    proxy_sample = [b for b in spec.sample_return_bodies
                    if _all_locations_infeasible(b, EventName.SAMPLE_RETURN,
                                                   model_infeasible_locations)]
    normal_sample = [b for b in spec.sample_return_bodies if b not in proxy_sample]
    if normal_sample:
        ns = tuple(normal_sample)
        def sample_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in ns:
                if not cap.bodies[b].access[EventName.SAMPLE_RETURN]:
                    return False
            return True
        sub_rules.append(sample_rule)
    if proxy_sample:
        sub_rules.append(_make_all_parts_rule(player))

    # Orbit bodies
    if spec.orbit_bodies:
        orbit_bodies = spec.orbit_bodies
        def orbit_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in orbit_bodies:
                if not cap.bodies[b].access[EventName.ORBIT]:
                    return False
            return True
        sub_rules.append(orbit_rule)

    # Flyby bodies
    if spec.flyby_bodies:
        flyby_bodies = spec.flyby_bodies
        def flyby_rule(state: CollectionState) -> bool:
            cap = get_capability(state, player)
            for b in flyby_bodies:
                if not cap.bodies[b].access[EventName.FLYBY]:
                    return False
            return True
        sub_rules.append(flyby_rule)

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
