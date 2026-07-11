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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from BaseClasses import CollectionState, ItemClassification

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, MissionType,
    home_system_bodies, relay_tier_table_for, science_budget,
)
from .comms import DSN_POWER_MAX, dsn_required_relay_table
from .capability import get_capability, cheap_flags, _compute_sounding_altitude
from .items import ITEM_TABLE, PROGRESSIVE_RD_NAME, SCIENCE_PACK_NAMES
from .locations import (
    EVENT_BY_NAME,
    EventName,
    HomeLocationDef,
    KSC_BIOME_NAMES,
    MissionLocation,
    STARTING_INV_NAMES,
    TechTreeLocation,
    effective_starting_inv_count,
    effective_tech_slots_per_node,
    event_locations,
)
from .options import Difficulty, Goal, GoalContractMode
from .tech_tree import MAX_TIER, MAX_RD_BAND, cumulative_tier_cost, TECH_NODES, LEAF_TECH_NODES
from .gates import AccumulationGate, Resource

if TYPE_CHECKING:
    from .world import KSP1World

# ---------------------------------------------------------------------------
# Safety factors for science heuristic (fraction of estimated budget counted)
# ---------------------------------------------------------------------------

_SCIENCE_SAFETY: dict[int, float] = {
    Difficulty.option_casual: 0.50,
    Difficulty.option_normal: 0.70,
    Difficulty.option_expert: 0.85,
}


def effective_science_safety(options, difficulty: int) -> float:
    """Science-budget safety fraction (0..1). Respects ``science_safety_factor``."""
    override = getattr(options, "science_safety_factor", None)
    if override is not None and override.value >= 0:
        return override.value / 100.0
    return _SCIENCE_SAFETY[difficulty]

# Fraction of a body's science budget counted when the player can only
# transmit (relay link) and can't physically return.  Tunable — stock KSP
# applies a per-experiment transmission penalty; this discount approximates
# the aggregate effect for budget-affordability purposes.
_TRANSMIT_ONLY_DISCOUNT: float = 0.75

# Science needed to declare the tech tree complete (buy all 62 nodes)
_TECH_TREE_COMPLETE_SCIENCE = cumulative_tier_cost(MAX_TIER)

# Phase 2: every part item (no longer wrapped behind progressives).
# Eve / Tylo / Laythe Return + Sample Return use this as a proxy for "you
# have everything the capability solver can't model from physics."  Per
# the design, these missions are hard-banned outside their target homes,
# so the strictness of "every part" is academic in practice.  Progressive
# R&D is excluded (separate tech-tree gate, not rocket capability); the
# remaining kept progressives (Pad, PSI) are also excluded because they
# don't represent rocket parts.
_ALL_PROGRESSION_ITEMS: frozenset[str] = frozenset(ITEM_TABLE.keys())


def _make_all_parts_rule(player: int) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return state.has_all(_ALL_PROGRESSION_ITEMS, player)
    return rule


# ---------------------------------------------------------------------------
# Gate chokepoint — the ONLY sanctioned way to gate a location on held items
# ---------------------------------------------------------------------------
#
# Building the has-closure and recording the item dependency in ONE call makes
# it structurally impossible to gate a location on an item without marking that
# item logic-required.  The sphere-ladder classification pass keeps every
# logic-required pooled item PROGRESSION; ``_assert_gate_items_progression``
# fails generation if one slips through.  Together they turn the recurring "a
# needed item got demoted to USEFUL, stranding whatever was placed behind it"
# bug into a construction-time error instead of a rare unsolvable seed.

def require_items(world: "KSP1World",
                  item_names) -> Callable[[CollectionState], bool]:
    """Return a ``has_all(item_names)`` rule and record those names as
    logic-required on ``world``.  Use whenever a location's reachability
    depends on holding a set of specific items."""
    names = tuple(item_names)
    player = world.player
    world.logic_required_items.update(names)
    return lambda state: state.has_all(names, player)


def require_item(world: "KSP1World", name: str,
                 count: int = 1) -> Callable[[CollectionState], bool]:
    """Return a ``has(name[, count])`` rule and record ``name`` as
    logic-required on ``world``.  The single-item form of ``require_items``."""
    player = world.player
    world.logic_required_items.add(name)
    if count == 1:
        return lambda state: state.has(name, player)
    return lambda state: state.has(name, player, count)


def _make_goal_event_rule(
    player: int, bodies, event: EventName,
) -> Callable[[CollectionState], bool]:
    """Goal sub-rule: every ``body`` is reachable for ``event`` under the cheap
    ladder oracle — the SAME ``has_all(bracket reps)`` the event's location rule
    uses (``world._cheap_mission_reps``).  Keeps the victory condition on the
    cheap system instead of a live ``get_capability`` per goal body/event.

    A body/event with no bracket entry splits two ways, mirroring the location
    layer exactly:

    * model-infeasible → the all-parts proxy (its original purpose: "the
      player owns everything, we can't verify" — such goals are normally
      filtered anyway);
    * feasible-but-unbracketed (no sphere's reps-only kit proved it — e.g.
      an assembly-closed mission the bumper could only RESCUE) → the real
      capability check, the same correct-but-slow fallback unbracketed
      LOCATIONS keep.  ``get_capability`` is state-cached, so the marginal
      cost is a dict lookup.  The old behaviour fell to the all-parts proxy
      here, which is unsatisfiable under advancement sweeps (useful-class
      parts are never auto-collected) — Victory deadlocked.

    Beyond the physics parts, each body/event carries its capability counted gate
    (nav / EVA / samples / DSN / pad, from ``world._cheap_mission_counted``), so
    victory enforces the SAME non-physics gate the mission location does — it
    can't be declared for an interplanetary goal with no Mission Control.
    """
    bt = tuple(bodies)
    ev = event.value
    mt = EVENT_BY_NAME[ev].mission_type

    def rule(state: CollectionState) -> bool:
        world = state.multiworld.worlds[player]
        reps_map = getattr(world, "_cheap_mission_reps", None)
        if reps_map is None:
            return False  # pre-ladder: conservatively unreachable (Golden Rule)
        counted_map = getattr(world, "_cheap_mission_counted", {})
        for b in bt:
            reps = reps_map.get((b.value, ev))
            if reps is not None:
                if not state.has_all(reps, player):
                    return False
            elif (b, mt) in world.unachievable_missions:
                if not state.has_all(_ALL_PROGRESSION_ITEMS, player):
                    return False
            else:
                from .capability import get_capability
                cap = get_capability(state, player)
                if not cap.bodies[b].access.get(ev, False):
                    return False
            counted = counted_map.get((b.value, ev), ())
            if not all(state.has(kind, player, lvl) for kind, lvl in counted):
                return False
        return True

    return rule


# ---------------------------------------------------------------------------
# Science heuristic helpers
# ---------------------------------------------------------------------------

def bankable_science(cap, psi_tier: int, home: BodyName,
                     access=None) -> float:
    """Per-body science contributions, gated on the player's ability to
    actually extract science from each body.

    A reachable body contributes 0 unless the player can either (a)
    physically recover (RETURN path exists) or (b) transmit (relay tier
    meets the body's heliocentric requirement).  Transmit-only paths
    apply ``_TRANSMIT_ONLY_DISCOUNT``.

    This is the canonical "what science can the player bank in this state"
    function.  Both the tech-tree victory rule and the sphere-ladder
    per-sphere tier-funding pass MUST use it — duplicating the loop with
    a different gate produces a silent mismatch where the ladder thinks
    the seed is solvable but the rule disagrees at fill time.

    ``access`` optionally supplies per-body ORBIT/RETURN/CREWED_LANDING
    reachability as ``{body_name: {EventName: bool}}``.  When given, the
    per-body access is read from it instead of ``cap.bodies[*].access`` —
    the funding pass uses this to feed cached, monotonically-accumulated
    access so it need not re-run the (expensive) per-body optimizer for
    bodies already proven reachable at an earlier sphere.  Instrument and
    relay flags still come from ``cap`` (cheap, flag-level).
    """
    # Transmitting science needs a comms link, which the Tracking Station (DSN)
    # gates when buildings_in_logic is on.  Below max DSN the antenna needs a
    # higher tier to reach; at max DSN (option off) this is the plain
    # antenna-only table — byte-identical to before.
    relay_table = (dsn_required_relay_table(home, cap.dsn_power)
                   if cap.dsn_power < DSN_POWER_MAX else relay_tier_table_for(home))
    total = 0.0
    for body in ALL_BODIES:
        if access is not None:
            acc = access[body.name]
            a_orbit = acc[EventName.ORBIT]
            a_return = acc[EventName.RETURN]
            a_land = acc[EventName.LANDING]
            a_crewed = acc[EventName.CREWED_LANDING]
        else:
            body_cap = cap.bodies[body.name]
            a_orbit = body_cap.access[EventName.ORBIT]
            a_return = body_cap.access[EventName.RETURN]
            a_land = body_cap.access[EventName.LANDING]
            a_crewed = body_cap.access[EventName.CREWED_LANDING]
        if not a_orbit:
            continue
        can_recover = a_return
        can_transmit = cap.relay_tier >= relay_table[body.name]
        if not (can_recover or can_transmit):
            continue
        contribution = science_budget(
            body, cap.has_thermometer, cap.has_barometer,
            cap.has_capsule, a_crewed,
            home=home, psi_tier=psi_tier,
            can_land_uncrewed=a_land,
        )
        if not can_recover:
            contribution *= _TRANSMIT_ONLY_DISCOUNT
        total += contribution
    return total


def _cheap_bankable_science(
    state: CollectionState, player: int, world, home: BodyName,
) -> float:
    """``bankable_science`` with per-body access from the ladder's precomputed
    science brackets (``world._science_body_event_reps``) instead of a live
    ``get_capability``.  Per-body access ``has_all(bracket reps)`` is conservative
    (⟹ the kit really flies it) and consistent with the funding placement that
    derived the brackets, so the science gate stays on the cheap ladder oracle.
    Instrument/relay inputs come from the cheap pre-pass + ``state.count``.
    """
    reps_map = world._science_body_event_reps
    flags = cheap_flags(state, player)
    psi_tier = state.count("Progressive Science Instrument", player)
    # DSN-aware transmit gate (see bankable_science); max DSN -> plain table.
    relay_table = (dsn_required_relay_table(home, flags.dsn_power)
                   if flags.dsn_power < DSN_POWER_MAX else relay_tier_table_for(home))
    total = 0.0
    for body in ALL_BODIES:
        orbit = reps_map.get((body.name, EventName.ORBIT))
        if orbit is None or not state.has_all(orbit, player):
            continue
        ret = reps_map.get((body.name, EventName.RETURN))
        can_recover = ret is not None and state.has_all(ret, player)
        can_transmit = flags.relay_tier >= relay_table[body.name]
        if not (can_recover or can_transmit):
            continue
        cl = reps_map.get((body.name, EventName.CREWED_LANDING))
        crewed = cl is not None and state.has_all(cl, player)
        lnd = reps_map.get((body.name, EventName.LANDING))
        can_land = lnd is not None and state.has_all(lnd, player)
        contribution = science_budget(
            body, flags.has_thermometer, flags.has_barometer,
            flags.has_capsule, crewed, home=home, psi_tier=psi_tier,
            can_land_uncrewed=can_land)
        if not can_recover:
            contribution *= _TRANSMIT_ONLY_DISCOUNT
        total += contribution
    return total


def _accessible_science(
    state: CollectionState, player: int, safety: float, home: BodyName,
) -> float:
    """
    Estimate the total science the player can earn from all bodies they can
    currently reach, given their current instrument and crew equipment.

    Multiplied by the safety factor before returning.  Uses the cheap ladder
    science brackets (built in pre_fill).  During normal generation the rules
    only evaluate after pre_fill, so an absent-brackets state means the ladder
    isn't built yet — conservatively report no accessible science (Golden Rule).
    The one exception is Universal Tracker, which rebuilds logic through
    set_rules only (no pre_fill): there we fall back to the live capability
    science sum the brackets approximate, so tech nodes aren't all reported out
    of logic.
    """
    world = state.multiworld.worlds[player]
    if getattr(world, "_science_body_event_reps", None) is None:
        if getattr(world, "_ut_active", False):
            cap = get_capability(state, player)
            psi_tier = state.count("Progressive Science Instrument", player)
            return bankable_science(cap, psi_tier, home) * safety
        return 0.0
    return _cheap_bankable_science(state, player, world, home) * safety


def _can_afford_tier(
    state: CollectionState, player: int, tier: int, safety: float, home: BodyName,
) -> bool:
    """Return True if the player's accessible science budget can cover all nodes through *tier*."""
    return _accessible_science(state, player, safety, home) >= cumulative_tier_cost(tier)


# ---------------------------------------------------------------------------
# Rule factories
# ---------------------------------------------------------------------------

def _make_science_threshold_rule(
    player: int, threshold: float, safety: float, home: BodyName,
) -> Callable[[CollectionState], bool]:
    """Return a rule that passes when accessible science * safety >= threshold.
    Expressed as a SCIENCE :class:`~.gates.AccumulationGate`."""
    def measure(state: CollectionState) -> float:
        return _accessible_science(state, player, safety, home)
    return AccumulationGate(Resource.SCIENCE, threshold).runtime_rule(measure)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def set_all_rules(world: KSP1World) -> None:
    player = world.player
    difficulty = world.options.difficulty.value

    _set_ksc_biome_rules(world, player)
    _set_home_rules(world, player)
    _set_mission_rules(world, player)
    _set_contract_rules(world, player)
    _set_threshold_rules(world, player)
    _apply_home_system_local_exclusions(world)
    # Tech tree rules are now region entrance rules (see regions.py).
    _ban_early_science_windfalls(world, player, difficulty)
    _set_early_bucket_item_bans(world, player, difficulty)
    if world.goal_spec.is_home_system_only(world.mission_builder.home):
        _set_interplanetary_item_rules(world, player)


def set_completion_condition(world: KSP1World, goal_spec: GoalSpec) -> None:
    player = world.player
    difficulty = world.options.difficulty.value
    safety = effective_science_safety(world.options, difficulty)

    _set_victory_rules(world, player, goal_spec, safety)
    _set_goal_locations_local_only(world, player, goal_spec)


def _set_goal_locations_local_only(
    world: KSP1World, player: int, goal_spec: GoalSpec
) -> None:
    """Force every goal-sentinel location to hold only this player's own items.

    The client declares victory when these locations are checked. Without the
    constraint a goal location could hold a *remote* player's item — and that
    player running ``!collect`` (which checks their items out of every world,
    yours included) would mark your goal location complete and end your game for
    you. Local-only means the only way to check a goal location is to actually
    fly the mission.
    """
    from worlds.generic.Rules import add_item_rule
    local_only = lambda item, p=player: item.player == p
    for name in goal_spec_location_names(goal_spec):
        add_item_rule(world.get_location(name), local_only)


# ---------------------------------------------------------------------------
# KSC biome location rules
# ---------------------------------------------------------------------------

def _can_do_ksc_science(state: CollectionState, player: int) -> bool:
    """
    Can the player do science at KSC biomes?

    Path 1: Crewed EVA (EVA report / surface sample) — just needs a capsule.
    Path 2: Rover with science instruments — probe + wheels + power + instrument.

    Reads only equipment flags (no per-body mission access), so it uses the
    cheap pre-pass instead of the full ``get_capability`` — KSC science never
    needs the rocket optimizer.
    """
    flags = cheap_flags(state, player)
    if flags.has_capsule:
        return True
    if (flags.has_probe_core and flags.has_wheel
            and (flags.has_rtg or flags.has_solar)
            and (flags.has_thermometer or flags.has_barometer)):
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

# These home-body rules read only equipment flags + the sounding-rocket sizing,
# so they use the cheap pre-pass instead of the full ``get_capability`` (no
# per-body mission optimizer).  ``_compute_sounding_altitude(flags, home_body)``
# is exactly what ``cap.sounding_altitude_km`` is computed from.
def _make_altitude_rule(player: int, threshold_km: float) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        flags = cheap_flags(state, player)
        home_body = state.multiworld.worlds[player].mission_builder.home_body
        return _compute_sounding_altitude(flags, home_body) >= threshold_km
    return rule


def _make_first_launch_rule(player: int) -> Callable[[CollectionState], bool]:
    """Any propulsion OR capsule (kerbal EVA counts as launch)."""
    def rule(state: CollectionState) -> bool:
        flags = cheap_flags(state, player)
        home_body = state.multiworld.worlds[player].mission_builder.home_body
        return _compute_sounding_altitude(flags, home_body) > 0 or flags.has_capsule
    return rule


def _make_first_landing_rule(player: int) -> Callable[[CollectionState], bool]:
    """Propulsion + safe descent OR capsule (EVA landing)."""
    def rule(state: CollectionState) -> bool:
        flags = cheap_flags(state, player)
        home_body = state.multiworld.worlds[player].mission_builder.home_body
        if (_compute_sounding_altitude(flags, home_body) > 0
                and (flags.has_parachutes or flags.has_throttleable_engine)):
            return True
        return flags.has_capsule
    return rule


def _make_staging_rule(player: int) -> Callable[[CollectionState], bool]:
    def rule(state: CollectionState) -> bool:
        return cheap_flags(state, player).staging_tier >= 1
    return rule


def _make_splashdown_rule(
    player: int, threshold_km: float, home: BodyName,
) -> Callable[[CollectionState], bool]:
    """Splashdown is achievable if either:

    * the home body has an ocean and the player has a sounding rocket
      above ``threshold_km`` with safe descent (parachute or throttleable
      engine), or
    * any other ocean body (Kerbin / Eve / Laythe) is fully reachable for
      a Landing mission — in which case the player can ditch into water.

    Reuses the precomputed per-body LANDING access dict; no extra profile
    evaluation on the rule hot path.
    """
    home_body = BODY_BY_NAME[home]
    home_has_ocean = home_body.has_ocean
    other_ocean_bodies: tuple[BodyName, ...] = tuple(
        BodyName(b.name) for b in ALL_BODIES if b.has_ocean and b.name != home
    )

    def rule(state: CollectionState) -> bool:
        # Home-ocean path uses only cheap flags + the sounding-rocket sizing.
        flags = cheap_flags(state, player)
        if home_has_ocean:
            if (_compute_sounding_altitude(flags, home_body) >= threshold_km
                    and (flags.has_parachutes or flags.has_throttleable_engine)):
                return True
        # Other-ocean-body LANDING uses the cheap ladder brackets (the same
        # has_all(reps) the Landing locations gate on), not a live capability.
        world = state.multiworld.worlds[player]
        reps_map = getattr(world, "_cheap_mission_reps", None)
        if reps_map is not None:
            for body_name in other_ocean_bodies:
                reps = reps_map.get((body_name.value, EventName.LANDING.value))
                if reps is not None and state.has_all(reps, player):
                    return True
        return False  # pre-ladder or no ocean-body landing reachable (conservative)
    return rule


_HOME_RULE_FACTORIES: dict[MissionType, Callable[[int, "HomeLocationDef", BodyName],
                                                 Callable[[CollectionState], bool]]] = {
    MissionType.SOUNDING:       lambda player, loc, home: _make_altitude_rule(player, loc.threshold_km or 0.0),
    MissionType.FIRST_LAUNCH:   lambda player, loc, home: _make_first_launch_rule(player),
    MissionType.FIRST_LANDING:  lambda player, loc, home: _make_first_landing_rule(player),
    MissionType.FIRST_STAGING:  lambda player, loc, home: _make_staging_rule(player),
    MissionType.SPLASHDOWN:     lambda player, loc, home: _make_splashdown_rule(player, loc.threshold_km or 1.0, home),
}


def _set_home_rules(world: KSP1World, player: int) -> None:
    """Apply access rules to the home-body specials and to the single
    body-agnostic Splashdown location.  Dispatches by ``mission_type``.
    """
    home = world.location_builder.home
    for loc in world.location_builder.locations:
        factory = _HOME_RULE_FACTORIES.get(loc.mission_type)
        if factory is None:
            raise ValueError(f"Unknown home mission type: {loc.mission_type!r}")
        world.get_location(loc.name).access_rule = factory(player, loc, home)


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


def _migrated_event_map() -> dict:
    """ContractType -> the matching milestone EventName it 'migrates' (gates
    equal-or-after, and shares model-infeasible proxy routing with). Shared by
    _set_mission_rules and _set_contract_rules so the milestone gating and the
    contract-location routing can't drift. (Local import dodges a cycle.)"""
    from .contracts import ContractType
    return {
        ContractType.FLAG_PLANT: EventName.FLAG_PLANT,
        ContractType.SAMPLE_RETURN: EventName.SAMPLE_RETURN,
        ContractType.ORBIT: EventName.ORBIT,
        ContractType.RETURN: EventName.RETURN,
        ContractType.FLYBY: EventName.FLYBY,
    }


def _set_contract_rules(world: KSP1World, player: int) -> None:
    """
    Apply access rules to contract completion locations.

    A contract location is reachable only when all three gates hold: the player
    holds the contract item, has the required parts, and can deliver the
    contract's equipment payload to the body. The latter two are folded into the
    cached ``contract_access[contract_id]`` boolean (computed once per state by
    get_capability), so the rule is ``state.has(item) AND O(1) lookup``.

    Contract locations are among the HARDEST in the seed (they need the full
    delivery capability + input parts). The sphere ladder gives them a real
    signature (see loc.descriptor / sphere_ladder._evaluate) so its Rule B bans
    only items BELOW the contract's sphere — keeping bootstrap items off them
    (which would otherwise deadlock: a launch engine placed at "Contract: Mine
    Ore on Eve" is unreachable without the very engine it gates) while still
    letting LATE progression land there, so contracts remain real pacing gates.
    """
    infeasible = world.model_infeasible_locations
    proxy_rule = _make_all_parts_rule(player)
    event_of = _migrated_event_map()
    # The single record of which contracts route through the all-parts proxy.
    # /explain reads this set (world._contract_uses_proxy) rather than re-deriving
    # the predicate, so the reported gate can't drift from the rule actually set.
    world._proxy_contract_ids = set()
    # Single record of which locations carry a contract rule (completion slots,
    # completion events, and goal-contract mission events).  The sphere-ladder
    # rule installer reads this to LEAVE these rules in place during fill instead
    # of overriding them with the generic bracket rule: the contract rule is
    # already cheap (no get_capability), so overriding it gains no speed but
    # diverges the fill-time rule from the post_fill rule (different rep set, and
    # the bracket rule omits the award gate on goal-contract events) — which
    # stranded items and forced the expensive strict_ladder fallback re-fill.
    world._contract_ruled_locations = set()
    # The REAL capability rule for each contract-ruled location: award gate AND
    # the live ``contract_access[cid]`` oracle.  Stashed here (keyed by location
    # name) so the sphere-ladder can merge it into ``_strict_ladder_saved_rules``;
    # the post_fill cross-check then swaps it in — verifying contract capability
    # exactly as it verifies missions, instead of re-checking contracts on the
    # same cheap rule the fill used (the old contract blind spot).  Proxy
    # contracts are omitted: their cheap all-parts rule already IS their real rule
    # (the dv model can't verify the achievement, so contract_access is False and
    # a swap would wrongly close the goal).
    world._contract_real_rules = {}
    # In count / progressive_unlock each non-goal contract has a completion-event
    # location; it shares the contract's rule so has("Contract Count Progress", X)
    # counts contracts completable in logic.
    counts_contracts = world.options.goal_contract_mode.value in (
        GoalContractMode.option_count,
        GoalContractMode.option_progressive_unlock,
    )
    for spec in (*world.contract_specs, *world.goal_contract_specs):
        ev = event_of.get(spec.contract_type)
        # A goal contract on a model-infeasible achievement (e.g. Eve/Laythe
        # return from a far home, which the dv model can't verify) routes to the
        # all-parts proxy, exactly as the matching milestone event and the
        # victory rule do — keeping the contract location reachable-in-logic
        # whenever its achievement is, rather than a dead filler-only slot.
        # Non-goal contracts are feasibility-filtered at generation, so they
        # never land on a model-infeasible body and keep the capability gate.
        uses_proxy = (spec.is_goal and ev is not None
                      and _all_locations_infeasible(spec.body, ev, infeasible))
        # The contract's gate item, routed through the chokepoint so it's
        # recorded logic-required (kept PROGRESSION) — you can't complete a
        # contract without first finding its item, and a demoted gate item
        # strands whatever progression fill placed on the contract location.
        gate = require_item(world, spec.item_name)
        # The real-rule swap the post_fill cross-check installs for a non-proxy
        # contract: award gate AND the live capability oracle.  ``None`` for proxy
        # contracts (kept on their cheap all-parts rule during the cross-check).
        real_rule = None
        if uses_proxy:
            world._proxy_contract_ids.add(spec.contract_id)
            def rule(state: CollectionState, cid=spec.contract_id, _gate=gate,
                     _proxy=proxy_rule) -> bool:
                if not (_gate(state) and _proxy(state)):
                    return False
                # A model-infeasible goal contract still has real capability gates
                # (nav / EVA / samples).  ``has_all(ALL parts)`` can't express a
                # count>=2 building level, so check the counted reqs explicitly.
                world = state.multiworld.worlds[player]
                counted = getattr(
                    world, "_cheap_contract_counted_reqs", {}).get(cid, ())
                return all(state.has(kind, player, lvl) for kind, lvl in counted)
        else:
            def real_rule(state: CollectionState, cid=spec.contract_id,
                          _gate=gate) -> bool:
                return _gate(state) and get_capability(
                    state, player).contract_access.get(cid, False)

            def rule(state: CollectionState, cid=spec.contract_id,
                     _gate=gate) -> bool:
                if not _gate(state):
                    return False
                world = state.multiworld.worlds[player]
                creps = getattr(world, "_cheap_contract_reps", None)
                if creps is None:
                    # The cheap proxy is built in pre_fill.  Universal Tracker
                    # rebuilds logic through set_rules only (no pre_fill), so the
                    # proxy is absent — fall back to the live capability oracle it
                    # approximates (get_capability works under UT; ordinary
                    # mission rules already use it).  Off the UT path this stays
                    # the conservative pre-ladder floor, so normal fill is
                    # unchanged.
                    if getattr(world, "_ut_active", False):
                        return get_capability(state, player) \
                            .contract_access.get(cid, False)
                    return False  # pre-ladder: conservatively not completable
                # Cheap delivery gate: the contract's bracket reps (has_all ⟹ the
                # kit delivers, conservative).  An unbracketed non-proxy contract
                # falls back to the all-parts proxy.
                reps = creps.get(cid)
                if not state.has_all(
                        reps if reps is not None else _ALL_PROGRESSION_ITEMS,
                        player):
                    return False
                # ...plus the CAPABILITY counted gate (nav/EVA/samples/DSN + the
                # pad) the contract really needs — derived spec-direct in
                # sphere_ladder._install_cheap_mission_reps, NOT from the sphere
                # position.  has_all above only covers the physics RANK reps;
                # without this a contract is reachable with no Mission Control
                # (interplanetary), no pad tier, etc.  R&D/PSI placement artifacts
                # are deliberately NOT here (they'd bind the contract to its
                # physics position; see _install_cheap_mission_reps).
                counted = getattr(
                    world, "_cheap_contract_counted_reqs", {}).get(cid, ())
                return all(state.has(kind, player, lvl) for kind, lvl in counted)
        def _apply(name: str, _rule=rule, _real=real_rule) -> None:
            """Install the cheap contract rule on ``name``, record it
            contract-ruled, and stash its real-rule swap (non-proxy only)."""
            world.get_location(name).access_rule = _rule
            world._contract_ruled_locations.add(name)
            if _real is not None:
                world._contract_real_rules[name] = _real

        # Every non-goal reward slot (base 2 + Contract Repeats) shares the one
        # gate+capability rule, so the extra slots land at the contract's own
        # sphere as buffer-fill.
        for loc_name in spec.location_names(world.locations_per_contract):
            _apply(loc_name)

        # Non-goal completion event shares the rule (count / progressive_unlock):
        # reachable iff the contract is completable, so it contributes one to the
        # "Contract Count Progress" count exactly when the contract is done in logic.
        if counts_contracts and not spec.is_goal:
            _apply(spec.completion_event_name)

        # A goal contract's matching mission event(s) share its EXACT rule, so
        # the (now ordinary) event is reachable iff the goal contract is
        # completable — logically equal, not merely gated after. Runs after
        # _set_mission_rules, overwriting that event's separately-computed
        # capability rule (which could drift from the contract's feasibility).
        if spec.is_goal and ev is not None:
            for ev_loc in event_locations(spec.body, ev):
                _apply(str(ev_loc))


def _set_threshold_rules(world: KSP1World, player: int) -> None:
    """Gate each goal-mode threshold location on the completed-contract count:
    ``state.has("Contract Count Progress", required_count)``. The event items are
    swept in as their (contract-gated) event locations become reachable, so a
    threshold unlocks exactly when ``required_count`` contracts are completable
    in logic, releasing its locked goal item. No-op in findable / starting."""
    from .contracts import CONTRACT_COUNT_PROGRESS_EVENT
    def measure(state: CollectionState) -> float:
        return state.count(CONTRACT_COUNT_PROGRESS_EVENT, player)
    for loc_name, count, _item in world.contract_threshold_defs:
        gate = AccumulationGate(Resource.CONTRACT_COMPLETION, count)
        world.get_location(loc_name).access_rule = gate.runtime_rule(measure)


def _set_mission_rules(world: KSP1World, player: int) -> None:
    """
    Apply access rules to the per-body mission event locations this world
    emits.

    The LocationBuilder emits only reachable missions — infeasible
    ``(body, mission_type)`` pairs (dv-infeasible ∪ curated ban) are not
    created at all — so every location here carries its real capability-based
    rule for its (body, event) pair, aligned with the victory rule
    (``_make_goal_spec_rule``), which routes the same way per body/event.
    Goal-contract events additionally gate on the contract item.
    """
    # Migrated-type contracts gate their MATCHING event equal-or-after the
    # contract item, so a player who has the contract does one mission for both
    # (never forced to double-run) and can't clear the event before the contract.
    # Keyed by the specific event (Orbit, not EVA-in-orbit; not LAND for a mine
    # contract whose base mission merely happens to be LAND).
    _migrated_event = _migrated_event_map()
    gate_item: dict[tuple[str, str], str] = {}
    # Only GOAL contracts gate their matching event: you can't clear the goal
    # mission event before holding the goal-contract item (keeps client events and
    # the server victory condition aligned). Non-goal (pacing) contracts must NOT
    # gate their event — doing so makes ordinary mission locations (e.g. Kerbin
    # Orbit 1) reachable only via a pacing-contract item, a gate the cheap
    # sphere-bracket fill rules don't model, which strands the goal path and makes
    # otherwise-trivial seeds unsolvable.
    for spec in world.goal_contract_specs:
        ev = _migrated_event.get(spec.contract_type)
        if ev is not None:
            gate_item[(spec.body, ev)] = spec.item_name

    # Group the world's emitted mission locations by (body, event) so each
    # event's shared capability rule is built once.  The emitted set is owned
    # by the LocationBuilder — never reconstructed from module-level data.
    by_event: dict[tuple[BodyName, EventName], list[MissionLocation]] = {}
    for ml in world.location_builder.mission_locations:
        by_event.setdefault((ml.body, ml.event), []).append(ml)
    for (body, event), locs in by_event.items():
        cap_rule = _mission_rule_for_event(player, body, event)
        item = gate_item.get((body, event))
        for loc in locs:
            name = str(loc)
            ap_loc = world.get_location(name)
            if item is None:
                ap_loc.access_rule = cap_rule
            else:
                # Goal-contract gate item via the chokepoint (kept
                # PROGRESSION) — the event is reachable only with the
                # goal-contract item held.
                gate = require_item(world, item)
                def rule(state: CollectionState, _base=cap_rule, _gate=gate) -> bool:
                    return _gate(state) and _base(state)
                ap_loc.access_rule = rule


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
    from .bodies import home_system_bodies

    spec = world.goal_spec
    if not spec.home_system_local:
        return
    assert spec.home is not None
    in_system = home_system_bodies(spec.home)
    for ml in world.location_builder.mission_locations:
        if ml.body in in_system:
            continue
        world.get_location(str(ml)).progress_type = LocationProgressType.EXCLUDED


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


def _ban_early_science_windfalls(world: KSP1World, player: int, difficulty: int) -> None:
    """Keep early checks from handing out science / tech windfalls.

    Bans ``_EARLY_BANNED_ITEMS`` (Progressive R&D + the two largest science
    packs) from the starting inventory, KSC biomes, home-body specials, and early
    home mission events, and blocks all science-pack filler from early-tier tech
    nodes (tiers 1-3).  Always on: the early game should hand out parts to fly
    with, not a science jackpot or the tech-band gate.
    """
    from worlds.generic.Rules import add_item_rule

    num_slots = effective_tech_slots_per_node(world.options, difficulty)
    num_starting = effective_starting_inv_count(world.options, difficulty)
    early_ban_rule = _make_early_ban_rule(player)

    # Starting Inventory
    for name in STARTING_INV_NAMES[:num_starting]:
        add_item_rule(world.get_location(name), early_ban_rule)

    # KSC biomes + home-body specials + early home mission events (everything
    # except Flyby/SOI Leave, which need escape).
    for name in world.location_builder.ksc_biome_names:
        add_item_rule(world.get_location(name), early_ban_rule)
    for name in world.location_builder.names:
        add_item_rule(world.get_location(name), early_ban_rule)
    home = world.mission_builder.home
    for event in (EventName.ORBIT, EventName.EVA_IN_ORBIT, EventName.LANDING,
                  EventName.CREWED_LANDING, EventName.FLAG_PLANT,
                  EventName.RETURN, EventName.SAMPLE_RETURN):
        for loc in event_locations(home, event):
            add_item_rule(world.get_location(str(loc)), early_ban_rule)

    # Science-pack filler off early-tier tech nodes (tiers 1-3).
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
    cap by a big jump and opens many bodies at once.  It must never be
    *handed to the player at game load* — even at a binding low base where
    the pad is needed early, the player should have to earn it on a real
    location, not start with it.  So it's banned from the starter bucket
    (the auto-checked starting-inventory locations); the placement system
    still has to find it an early *non-starter* home.
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

    home_system = home_system_bodies(world.mission_builder.home)

    def no_advancement(item) -> bool:
        return item.player != player or not item.advancement

    for loc in world.location_builder.mission_locations:
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
    # The random_contracts "free" goal: a single home-body flag plant whose only
    # real gate is completing X contracts. Its content is the contracts, not the
    # destination, so it must NOT restrict contract generation to the home system
    # (see is_home_system_only) and its tiny launch mass must not cap contract
    # difficulty (see generate_contracts).
    free_goal: bool = False

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
        # The free goal's flag-at-home target would otherwise read as
        # "home system only" and wrongly confine every contract to home.
        if self.free_goal:
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
        # home_system_local is DERIVED in _filter_home_from_spec (True from a
        # Jool home where these moons are local, False from Kerbin/Duna where
        # this is a valid cross-system run) — never hardcoded on the preset.
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

    # The random_contracts "free" goal: a single home-body flag plant. Built
    # directly (already materialized) so it bypasses _filter_home_from_spec,
    # which would strip the home body and leave an empty, target-less spec.
    if goal_value == Goal.option_random_contracts:
        return GoalSpec(
            display_name="Random Contracts",
            flag_bodies=(home,),
            free_goal=True,
            home=home,
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

        # OptionSet stores the raw YAML body names as plain strings; the
        # GoalSpec and every downstream consumer expect BodyName members
        # (e.g. ``b.value`` in the goal/event rules).  The preset specs use
        # BodyName directly, so only this custom path needs the conversion.
        def _to_bodies(values) -> tuple[BodyName, ...]:
            return tuple(sorted(BodyName(v) for v in values))

        spec = GoalSpec(
            display_name=display,
            flag_bodies=_to_bodies(options.flag_bodies.value),
            return_bodies=_to_bodies(options.return_bodies.value),
            sample_return_bodies=_to_bodies(options.sample_return_bodies.value),
            orbit_bodies=_to_bodies(options.orbit_bodies.value),
            flyby_bodies=_to_bodies(options.flyby_bodies.value),
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
    a mission on their starting body — trivially achievable), stamp ``home``,
    and DERIVE ``home_system_local``.

    ``home_system_local`` is purely a byproduct of whether the resolved goal
    happens to sit entirely within the home system — never a preset/user choice.
    When it does (e.g. mun_flag from Kerbin, or jool_moons_return from a Jool
    home), it's a focused local run and external bodies are banned from logic.
    A goal that reaches outside the home system — e.g. jool_moons_return from
    Kerbin, or mun_flag from Duna — is a perfectly valid, if ambitious,
    cross-system mission; it's simply not "local".
    """
    def _strip(bodies: tuple[BodyName, ...]) -> tuple[BodyName, ...]:
        return tuple(b for b in bodies if b != home)
    materialized = GoalSpec(
        display_name=spec.display_name,
        flag_bodies=_strip(spec.flag_bodies),
        return_bodies=_strip(spec.return_bodies),
        sample_return_bodies=_strip(spec.sample_return_bodies),
        orbit_bodies=_strip(spec.orbit_bodies),
        flyby_bodies=_strip(spec.flyby_bodies),
        complete_tech_tree=spec.complete_tech_tree,
        free_goal=spec.free_goal,
        home=home,
    )
    return replace(materialized,
                   home_system_local=materialized.is_home_system_only(home))


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
    """Return the location names whose checks indicate goal completion: the goal
    **contract** locations (one per goal achievement).

    The goal mission events themselves (``Mun Flag Plant 1`` etc.) are ordinary
    checks — ``_set_mission_rules`` gates each equal-or-after its contract item,
    so they can never be required before the contract, but they no longer signal
    victory. The leaf tech-tree nodes remain the sentinels for the tech-tree goal
    (it has no body-achievement contracts)."""
    from .contracts import _goal_contract_specs
    names = [s.location_name for s in _goal_contract_specs(spec)]
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


def create_threshold_locations(world: KSP1World) -> None:
    """Create goal-mode threshold + contract-completion-event locations
    (count / progressive_unlock only; no-op otherwise).

    Threshold locations are REAL, pre-filled locations: each holds a locked goal
    contract item (or Progressive R&D copy for the tech-tree goal). The client
    reports them once the completed-contract count reaches the threshold,
    releasing the locked item through the normal AP channel — so the existing
    item-gated contract-offer machinery needs no change.

    For each non-goal contract we also mint an address-None EVENT location
    ("Contract Complete: ...") locked with a "Contract Count Progress" event item; its
    access rule (set in _set_contract_rules, identical to the contract's) makes
    ``state.has("Contract Count Progress", X)`` mean "X contracts completable in
    logic", which is what the threshold access rules gate on.
    """
    from BaseClasses import Location
    from .items import create_item, KSP1Item
    from .contracts import CONTRACT_COUNT_PROGRESS_EVENT
    from .locations import KSP1Location, LOCATION_NAME_TO_ID

    defs = world.contract_threshold_defs
    if not defs:
        return

    menu = world.get_region("Menu")

    # Threshold locations: real (addressed), pre-filled with the locked item.
    threshold_locs = {
        loc_name: LOCATION_NAME_TO_ID[loc_name] for loc_name, _c, _i in defs
    }
    menu.add_locations(threshold_locs, KSP1Location)
    for loc_name, _count, item_name in defs:
        world.get_location(loc_name).place_locked_item(create_item(world, item_name))

    # Contract-completion events: one per non-goal contract, address None.
    for spec in world.contract_specs:
        ev_name = spec.completion_event_name
        ev_loc = Location(world.player, ev_name, None, menu)
        menu.locations.append(ev_loc)
        ev_loc.place_locked_item(
            KSP1Item(CONTRACT_COUNT_PROGRESS_EVENT, ItemClassification.progression,
                     None, world.player))


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
    world: KSP1World, player: int, spec: GoalSpec, safety: float
) -> None:
    """Set the access rule and completion condition on the Victory event.

    Goals are contracts: on top of the capability/tech check (which keeps the
    proxy fallback for model-infeasible bodies and the tech-tree science budget),
    victory also requires every goal-contract ITEM to be collected. Combined with
    the event gating (a goal event is reachable only with its goal-contract item),
    you cannot complete a goal mission without first finding its contract.
    """
    base_rule = _make_goal_spec_rule(
        player, spec, safety, world.model_infeasible_locations,
        world.mission_builder.home,
    )
    # Goal-contract items gate Victory; route through the chokepoint so they're
    # kept PROGRESSION (a demoted goal item the beatability sweep never collects
    # makes the goal unreachable).
    goal_gate = require_items(
        world, [s.item_name for s in world.goal_contract_specs])

    def victory_rule(state: CollectionState) -> bool:
        return goal_gate(state) and base_rule(state)

    world.get_location("Victory").access_rule = victory_rule
    world.multiworld.completion_condition[player] = (
        lambda state: state.can_reach("Victory", "Location", player)
    )


def _make_goal_spec_rule(
    player: int, spec: GoalSpec, safety: float,
    model_infeasible_locations: frozenset[str],
    home: BodyName,
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
        sub_rules.append(_make_goal_event_rule(
            player, spec.flag_bodies, EventName.FLAG_PLANT))

    # Return bodies
    proxy_return = [b for b in spec.return_bodies
                    if _all_locations_infeasible(b, EventName.RETURN,
                                                   model_infeasible_locations)]
    normal_return = [b for b in spec.return_bodies if b not in proxy_return]
    if normal_return:
        sub_rules.append(_make_goal_event_rule(
            player, normal_return, EventName.RETURN))
    if proxy_return:
        sub_rules.append(_make_all_parts_rule(player))

    # Sample return bodies
    proxy_sample = [b for b in spec.sample_return_bodies
                    if _all_locations_infeasible(b, EventName.SAMPLE_RETURN,
                                                   model_infeasible_locations)]
    normal_sample = [b for b in spec.sample_return_bodies if b not in proxy_sample]
    if normal_sample:
        sub_rules.append(_make_goal_event_rule(
            player, normal_sample, EventName.SAMPLE_RETURN))
    if proxy_sample:
        sub_rules.append(_make_all_parts_rule(player))

    # Orbit bodies
    if spec.orbit_bodies:
        sub_rules.append(_make_goal_event_rule(
            player, spec.orbit_bodies, EventName.ORBIT))

    # Flyby bodies
    if spec.flyby_bodies:
        sub_rules.append(_make_goal_event_rule(
            player, spec.flyby_bodies, EventName.FLYBY))

    # Complete tech tree
    if spec.complete_tech_tree:
        science_rule = _make_science_threshold_rule(
            player, _TECH_TREE_COMPLETE_SCIENCE, safety, home,
        )
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
