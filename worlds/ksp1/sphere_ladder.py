"""
Sphere-ladder pre-fill for KSP1.

Builds a chain of capability "spheres" (locations the player can reach
given a cumulative kit) and installs item-placement rules so each
sphere's required progressive items can only land at locations strictly
easier than that sphere.  This closes the chicken-and-egg fill failures
where ``distribute_items_restrictive`` places, say, a Progressive Heat
Shield at "Tylo EVA in Orbit" — a location that itself requires the heat
shield to reach.  See ``/home/nick/.claude/plans/modular-hatching-hickey.md``.

Phase 1 (this commit): predictable spheres only (S_launch / S_orbit /
S_goal), Rule A extension (bootstrap-local), sphere-1 boost via
``local_early_items``.  No Rule B, no intermediates, no tech-tier
post-pass yet — those come in Phases 2/3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable, TYPE_CHECKING, Optional

from Options import OptionError

from .bodies import (
    BODY_BY_NAME, BodyName, DIFFICULTY_PROFILES, DifficultyProfile,
    MISSION_PROFILES, MissionType,
)
from .capability import (
    EquipmentFlags, ProfileResult,
    _evaluate_sounding, _pre_pass, evaluate_mission_detailed,
)
from .capability_reasons import BlockingInfo, BlockingReason
from .bodies import BodyName as _BN
from .locations import (
    EVENT_BY_NAME, EventName, KERBIN_LOCATIONS, MissionLocation,
    KSC_BIOME_NAMES, KSC_LOCATION_PREFIX,
)


# Body/event combinations whose access rule is a proxy (state.has_all
# progression items) rather than physics-based capability. The capability
# system can't compute Eve ascent / Tylo aerobrake, so the rule short-
# circuits to "have everything". Sphere ladder cannot honor these via
# `minimal_rocket_for` (it bottoms out on NO_VIABLE_STAGE no matter
# what); we skip them and let the proxy rule + main fill handle them.
# Keep in sync with rules.py:271-274.
_PROXY_GOALS: frozenset[tuple[str, str]] = frozenset({
    (_BN.EVE, EventName.RETURN),
    (_BN.EVE, EventName.SAMPLE_RETURN),
    (_BN.TYLO, EventName.RETURN),
    (_BN.TYLO, EventName.SAMPLE_RETURN),
    (_BN.LAYTHE, EventName.RETURN),
    (_BN.LAYTHE, EventName.SAMPLE_RETURN),
})


def _is_proxy_goal(location_name: str) -> bool:
    parsed = MissionLocation.parse(location_name)
    if parsed is None:
        return False
    return (parsed.body, parsed.event) in _PROXY_GOALS
from .items import (
    PROGRESSIVE_LAUNCH_PAD_COUNT, PROGRESSIVE_LAUNCH_PAD_NAME,
    PROGRESSIVE_RD_COUNT, PROGRESSIVE_RD_NAME,
)
from .parts import (
    PROGRESSIVE_PART_COUNTS as _BASE_PROGRESSIVE_PART_COUNTS,
    PROGRESSIVE_PART_NAMES,
)


# Unified per-item caps for sphere-ladder bumping.  Combines progressive
# *part* counts (engine/tank/etc.) with the non-part progressives
# (Launch Pad, R&D) and one-off items like rtg.
PROGRESSIVE_CAPS: dict[str, int] = {
    **_BASE_PROGRESSIVE_PART_COUNTS,
    PROGRESSIVE_LAUNCH_PAD_NAME: PROGRESSIVE_LAUNCH_PAD_COUNT,
    PROGRESSIVE_RD_NAME: PROGRESSIVE_RD_COUNT,
    "rtg": 1,
}

if TYPE_CHECKING:
    from .world import KSP1World


# ---------------------------------------------------------------------------
# Ranked priority groups for "which progressive should we bump next?"
# Groups are walked in order; the first group that overlaps the candidate
# set is the random pick pool.  Variance comes from the in-group choice;
# pacing comes from the cross-group ordering.
# ---------------------------------------------------------------------------

BUMP_PRIORITY_GROUPS: tuple[frozenset[str], ...] = (
    # Group 1: thrust/fuel — common bootstrap, low impact on seed openness.
    frozenset({
        "Progressive Launch Engine",
        "Progressive Vacuum Engine",
        "Progressive LFO Tank",
        "Progressive SRB",
        "Progressive Stack Decoupler",
        "Progressive Radial Decoupler",
        "Progressive Engine Plate",
    }),
    # Group 2: aero/heat + attitude — moderate impact.
    frozenset({
        "Progressive Heat Shield",
        "Progressive Parachute",
        "Progressive Probe Core",
        "Progressive SAS",
    }),
    # Group 3: power & comms (excluding the big openers).
    frozenset({
        "Progressive Solar Panel",
        "rtg",
    }),
    # Group 4: control/landing surface items.
    frozenset({
        "Progressive Landing Leg",
        "Progressive Capsule",
        "Progressive Ladder",
    }),
    # Group 5: blow-open items — give last because they unlock a lot.
    frozenset({
        "Progressive Launch Pad",
        "Progressive Relay",
    }),
)


# ``BlockingReason`` → set of progressive items that could plausibly fix it.
# Producer (minimal_rocket_for) intersects these with what's still under
# its per-item cap to form the "wants" set, then resolves the bump via
# BUMP_PRIORITY_GROUPS.
_BUMP_TABLE: dict[BlockingReason, frozenset[str]] = {
    BlockingReason.NO_VIABLE_STAGE: frozenset({
        "Progressive LFO Tank",
        "Progressive Launch Engine",
        "Progressive Vacuum Engine",
        "Progressive Stack Decoupler",
        "Progressive SRB",
        # Engine clustering (multi-mount) lets weak single engines combine
        # for enough thrust; radial decouplers enable asparagus staging
        # for high-dv ascents.  Both are common load-bearing items when
        # the tier-N rep can't fly the mission alone.
        "Progressive Engine Plate",
        "Progressive Radial Decoupler",
        # Mass-cap failures sometimes surface as "no viable stage" when
        # the optimizer rejects every candidate over the cap.
        "Progressive Launch Pad",
    }),
    BlockingReason.NO_ENGINE: frozenset({
        "Progressive Vacuum Engine",
        "Progressive Launch Engine",
    }),
    BlockingReason.NO_LAUNCH_ENGINE: frozenset({"Progressive Launch Engine"}),
    BlockingReason.NO_FUEL: frozenset({"Progressive LFO Tank"}),
    BlockingReason.NO_PROPULSION: frozenset({
        "Progressive Launch Engine",
        "Progressive LFO Tank",
        "Progressive SRB",
    }),
    BlockingReason.STAGING_TIER_INSUFFICIENT: frozenset({
        "Progressive Stack Decoupler",
    }),
    BlockingReason.NO_HEAT_SHIELD: frozenset({"Progressive Heat Shield"}),
    BlockingReason.RELAY_TIER_TOO_LOW: frozenset({"Progressive Relay"}),
    BlockingReason.INSUFFICIENT_POWER_SOLAR_OK: frozenset({
        "Progressive Solar Panel",
        "rtg",
    }),
    BlockingReason.INSUFFICIENT_POWER_NEEDS_RTG: frozenset({"rtg"}),
    BlockingReason.NO_PROBE_CORE: frozenset({"Progressive Probe Core"}),
    BlockingReason.NO_CAPSULE: frozenset({"Progressive Capsule"}),
    BlockingReason.NO_COMMAND_MODULE: frozenset({
        "Progressive Capsule",
        "Progressive Probe Core",
    }),
    BlockingReason.PARACHUTE_TERMINAL_VELOCITY: frozenset({"Progressive Parachute"}),
    BlockingReason.NO_PARACHUTE: frozenset({"Progressive Parachute"}),
    BlockingReason.NO_SAFE_DESCENT: frozenset({"Progressive Parachute"}),
    BlockingReason.CAPSULE_SOUNDING_INCOMPLETE: frozenset({
        "Progressive Parachute",
        "Progressive Stack Decoupler",
    }),
    BlockingReason.LANDING_LEGS_MISSING: frozenset({"Progressive Landing Leg"}),
    BlockingReason.NO_LADDER: frozenset({"Progressive Ladder"}),
    BlockingReason.LAUNCH_MASS_EXCEEDED: frozenset({"Progressive Launch Pad"}),
    BlockingReason.SOUNDING_ALTITUDE_TOO_LOW: frozenset({
        "Progressive Launch Engine",
        "Progressive LFO Tank",
        "Progressive SRB",
    }),
    BlockingReason.NO_SOUNDING_ALTITUDE: frozenset({
        "Progressive Launch Engine",
        "Progressive LFO Tank",
        "Progressive SRB",
    }),
    BlockingReason.NO_ATTITUDE_CONTROL: frozenset({
        # Progressive SAS is the dedicated cheap fix (sasModule line);
        # Probe Core tier 2+ also provides reaction wheels (some tier-1
        # reps don't, e.g. rover bodies); Capsule provides built-in
        # reaction wheels but is the heaviest commit.
        "Progressive SAS",
        "Progressive Probe Core",
        "Progressive Capsule",
    }),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinimalRocket:
    """Result of ``minimal_rocket_for``: the smallest progressive-item
    delta that, given ``prior_kit``, makes a target location reachable.
    """
    delta: dict[str, int]            # NEW items beyond prior_kit
    cumulative: dict[str, int]       # delta merged with prior_kit
    flags: EquipmentFlags
    profile_dv: float                # cheapest profile's dv (best-effort)
    # Equipment requirements: maps flag name → required level.
    # Binary flags use level=1.  Tiered flags (relay_tier, landing_leg_tier,
    # staging_tier) use the integer tier.  Partial-order comparison treats
    # requirements as "needs at least this level".
    requirements: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LocationSignature:
    """Position of a location in the capability partial order.

    ``requirements`` is a tuple of (flag_name, required_level) pairs.
    Comparison: ``A`` requires-subset ``B`` iff every (k, level_A) in A
    has a matching (k, level_B) in B with level_B >= level_A.
    Strict-less iff requires-subset AND (dv strictly less OR strict
    requirement superset).

    ``body_chain_depth`` is informational; it does not participate in
    the partial-order test (per design feedback).
    """
    dv: float
    requirements: tuple[tuple[str, int], ...]
    body_chain_depth: int


def _reqs_subset(a: tuple[tuple[str, int], ...],
                 b: tuple[tuple[str, int], ...]) -> bool:
    """Return True iff requirement-set A is a subset of B (a needs no
    more, possibly less, than b)."""
    b_dict = dict(b)
    for k, v in a:
        bv = b_dict.get(k)
        if bv is None or bv < v:
            return False
    return True


def _reqs_strict_less(a: tuple[tuple[str, int], ...],
                      b: tuple[tuple[str, int], ...]) -> bool:
    """A strict-requires-less B iff A ⊆ B AND A != B."""
    if a == b:
        return False
    return _reqs_subset(a, b)


@dataclass
class SphereBoundary:
    """A sphere in the ladder: a location whose minimum kit is
    enforced ahead of fill.  Items in ``delta`` are constrained
    (Rule B) to land only at strictly-easier locations than this
    sphere's signature.
    """
    name: str
    location_name: str
    is_predictable: bool             # True for S_launch / S_orbit / S_goal*
    rocket: MinimalRocket
    signature: Optional[LocationSignature] = None


@dataclass
class SphereLadder:
    """The complete ladder for one seed; stored on the world."""
    spheres: list[SphereBoundary] = field(default_factory=list)
    # Total cumulative kit after walking the whole chain (for diagnostics).
    cumulative_kit: dict[str, int] = field(default_factory=dict)
    # Per-location signature, used by tests/diagnostics.
    location_signatures: dict[str, LocationSignature] = field(default_factory=dict)


def _strict_less(a: LocationSignature, b: LocationSignature) -> bool:
    """Partial-order strict-less. Returns True iff a < b — i.e. A
    requires no more than B AND (A.dv strictly less OR A's requirements
    are a strict subset of B's)."""
    if a is b:
        return False
    if a.dv > b.dv:
        return False
    if not _reqs_subset(a.requirements, b.requirements):
        return False
    if a.dv == b.dv and a.requirements == b.requirements:
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LocationMissionInfo:
    body: str
    mission_type: MissionType
    crewed: Optional[bool]
    threshold_km: Optional[float]


def _parse_location(name: str) -> Optional[_LocationMissionInfo]:
    """Resolve a location name to a (body, mission_type, crewed, threshold)
    tuple.  Returns ``None`` for locations whose access isn't physics-gated
    (tech tree, KSC biomes, starting inventory) — those need their own
    handling and are skipped by the greedy walk.
    """
    # Kerbin event/altitude locations.
    for kloc in KERBIN_LOCATIONS:
        if kloc.name == name:
            return _LocationMissionInfo(
                body=BodyName.KERBIN,
                mission_type=kloc.mission_type,
                crewed=None,
                threshold_km=kloc.threshold_km,
            )
    # Per-body mission locations.
    parsed = MissionLocation.parse(name)
    if parsed is not None:
        event_def = EVENT_BY_NAME[parsed.event]
        return _LocationMissionInfo(
            body=parsed.body,
            mission_type=event_def.mission_type,
            crewed=event_def.crewed,
            threshold_km=None,
        )
    # Tech tree / KSC / starting inventory: not capability-gated.
    return None


def _evaluate(
    flags: EquipmentFlags,
    info: _LocationMissionInfo,
    diff: DifficultyProfile,
) -> ProfileResult:
    """Dispatch to the right evaluator for a location's mission type."""
    if info.mission_type == MissionType.SOUNDING:
        return _evaluate_sounding(flags, info.threshold_km or 0.0)
    return evaluate_mission_detailed(
        flags, diff,
        info.body, info.mission_type, info.crewed,
        info.threshold_km,
    )


def _extract_requirements(flags: EquipmentFlags) -> tuple[tuple[str, int], ...]:
    """Return the ``EquipmentFlags`` requirements as a sorted tuple of
    (flag_name, required_level) pairs.  Binary flags use level 1; tiered
    flags use their integer tier.  Sorted so equal requirement sets
    produce equal tuples (lets ``==`` work for the strict-less check).
    """
    reqs: dict[str, int] = {}
    bools = (
        "has_heat_shield", "has_parachutes", "has_probe_core", "has_capsule",
        "has_rtg", "has_solar", "has_solar_retractable", "has_isru",
        "has_docking_port", "has_ladder", "has_launch_clamp",
        "has_throttleable_engine", "has_aero_control_surface",
        "has_reaction_wheels", "has_rcs",
    )
    for name in bools:
        if getattr(flags, name, False):
            reqs[name] = 1
    if flags.landing_leg_tier > 0:
        reqs["landing_leg_tier"] = flags.landing_leg_tier
    if flags.relay_tier > 0:
        reqs["relay_tier"] = flags.relay_tier
    if flags.staging_tier > 0:
        reqs["staging_tier"] = flags.staging_tier
    return tuple(sorted(reqs.items()))


def _group_index(cand: str) -> int:
    """Index of the first ``BUMP_PRIORITY_GROUPS`` group containing ``cand``,
    used as a deterministic tiebreaker.  Returns ``len(groups)`` for items
    that don't appear in any group (sorts last).
    """
    for i, group in enumerate(BUMP_PRIORITY_GROUPS):
        if cand in group:
            return i
    return len(BUMP_PRIORITY_GROUPS)


def _pick_bump(
    blocking: list[BlockingInfo],
    kit: dict[str, int],
    rng: Random,
    evaluate_with_bump: Callable[[str], "tuple[bool, float, int]"],
) -> Optional[str]:
    """Choose the next progressive to bump.

    Primary objective: minimize the rocket's launch mass after the bump
    (cheapest capability increase wins).  Secondary: prefer bumps that
    reduce the number of remaining blockers when no candidate becomes
    feasible.  Tertiary: ``BUMP_PRIORITY_GROUPS`` order (now just a
    tiebreaker).  Quaternary: deterministic random.

    ``evaluate_with_bump(cand)`` runs ``_pre_pass`` + ``_evaluate`` with
    a kit that has ``cand`` bumped by one, returning
    ``(feasible, launch_mass, blocking_count)``.

    1. Collect candidates from each blocking entry via ``_BUMP_TABLE``.
    2. Filter to items not yet at their per-item cap.
    3. Score each candidate by hypothetical evaluation.
    4. Pick the lowest-mass feasible candidate; if none feasible, the
       one that reduces blocker count the most.
    """
    # Iterate _BUMP_TABLE candidates in sorted order so the chain is
    # deterministic across Python processes (frozenset iteration depends
    # on PYTHONHASHSEED).
    wants: list[str] = []
    seen: set[str] = set()
    for b in blocking:
        for cand in sorted(_BUMP_TABLE.get(b.reason, ())):
            if cand in seen:
                continue
            cap = PROGRESSIVE_CAPS.get(cand, 1)
            if kit.get(cand, 0) < cap:
                seen.add(cand)
                wants.append(cand)
    if not wants:
        return None

    # Score each candidate.  Sorted tuple ordering: feasible first
    # (False < True after flipping), then by mass, then by blocker count,
    # then by priority-group index, then by deterministic random tiebreak.
    scored: list[tuple[int, float, int, int, float, str]] = []
    for cand in wants:
        feasible, mass, n_blocking = evaluate_with_bump(cand)
        # Primary: infeasible candidates rank lower (sort by 0 vs 1).
        feasibility_rank = 0 if feasible else 1
        # Secondary (feasible): mass.  Secondary (infeasible): blocker count.
        mass_score = mass if feasible else float("inf")
        scored.append((
            feasibility_rank, mass_score, n_blocking,
            _group_index(cand), rng.random(), cand,
        ))
    scored.sort()
    return scored[0][-1]


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------

def minimal_rocket_for(
    location_name: str,
    prior_kit: dict[str, int],
    rep_names: frozenset[str],
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    rng: Random,
    precollected_names: frozenset[str] = frozenset(),
) -> Optional[MinimalRocket]:
    """Compute the minimum delta of progressive items beyond ``prior_kit``
    that makes ``location_name`` reachable.

    The ladder rocket is built EXCLUSIVELY from parts AP can guarantee
    the player has at this sphere:
      - Progressive items in ``prior_kit`` / ``kit`` (which auto-grant
        their reps via ``_pre_pass``)
      - Items in ``precollected_names`` (multiworld.precollected_items)
    Everything else is treated as absent.  Non-progressive structural
    parts that happen to carry fuel (Mk3 fuselages, Size3To2Adapter, …)
    or any other capability-significant part NOT covered by the
    progressive system is unavailable to the ladder, full stop.

    Returns ``None`` if the location is not capability-gated (tech tree,
    KSC biome, starting inventory) OR if the rep set + per-item caps
    cannot reach it under any kit.
    """
    info = _parse_location(location_name)
    if info is None:
        return None

    diff = DIFFICULTY_PROFILES[difficulty]
    kit: dict[str, int] = dict(prior_kit)

    def _count_fn(name: str, _k: dict[str, int] = kit) -> int:
        if name in PROGRESSIVE_CAPS:
            return _k.get(name, 0)
        if name in precollected_names:
            return 1
        return 0

    def _evaluate_with_bump(cand: str) -> tuple[bool, float, int]:
        """Score a hypothetical bump of ``cand`` by running pre_pass +
        evaluate on a kit with that candidate incremented by one.
        Returns (feasible, launch_mass, n_blocking).  Used by
        ``_pick_bump`` to score candidates by capability increase.
        """
        trial_kit = dict(kit)
        trial_kit[cand] = trial_kit.get(cand, 0) + 1

        def _trial_count_fn(name: str, _tk: dict[str, int] = trial_kit) -> int:
            if name in PROGRESSIVE_CAPS:
                return _tk.get(name, 0)
            if name in precollected_names:
                return 1
            return 0

        trial_flags = _pre_pass(
            _trial_count_fn,
            start_with_clamps=start_with_clamps,
            rep_names=rep_names,
            progressive_launch_pad=progressive_launch_pad,
        )
        trial_result = _evaluate(trial_flags, info, diff)
        return (
            trial_result.feasible,
            trial_result.launch_mass if trial_result.feasible else float("inf"),
            len(trial_result.blocking),
        )

    # Safety bound: with 17 progressive groups and per-group caps ≤ 5,
    # the total possible bumps is ~50.  We allow 200 to absorb wasted
    # bumps when randomness picks an item that doesn't close any current
    # blocking.
    for _ in range(200):
        flags = _pre_pass(
            _count_fn,
            start_with_clamps=start_with_clamps,
            rep_names=rep_names,
            progressive_launch_pad=progressive_launch_pad,
        )
        result = _evaluate(flags, info, diff)
        if result.feasible:
            delta = {
                k: v - prior_kit.get(k, 0)
                for k, v in kit.items()
                if v - prior_kit.get(k, 0) > 0
            }
            return MinimalRocket(
                delta=delta,
                cumulative=dict(kit),
                flags=flags,
                profile_dv=result.launch_mass,
                requirements=_extract_requirements(flags),
            )
        item = _pick_bump(result.blocking, kit, rng, _evaluate_with_bump)
        if item is None:
            return None
        kit[item] = kit.get(item, 0) + 1

    return None


# ---------------------------------------------------------------------------
# Sphere ladder construction
# ---------------------------------------------------------------------------

def _kit_merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Element-wise max of two kits (cumulative; not additive)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0), v)
    return out


def _build_rocket_or_raise(
    world: "KSP1World",
    location_name: str,
    cumulative: dict[str, int],
    sphere_label: str,
) -> MinimalRocket:
    """Build a MinimalRocket for a *predictable* sphere; raise
    ``OptionError`` if the rep set can't reach the location.
    """
    rep_names = frozenset(
        rep
        for tiers in world.progressive_representatives.values()
        for rep in tiers.values()
    )
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    difficulty = ["casual", "normal", "expert", "insane"][
        world.options.difficulty.value
    ]
    rocket = minimal_rocket_for(
        location_name=location_name,
        prior_kit=cumulative,
        rep_names=rep_names,
        difficulty=difficulty,
        progressive_launch_pad=bool(world.options.progressive_launch_pad),
        start_with_clamps=bool(world.options.start_with_launch_clamps),
        rng=world.random,
        precollected_names=precollected_names,
    )
    if rocket is None:
        raise OptionError(
            f"KSP1 sphere ladder: predictable sphere '{sphere_label}' "
            f"({location_name}) is unreachable even with maxed-out progressive "
            f"items given this seed's representative selection. This usually "
            f"means the random rep choices left no engine/tank capable of "
            f"the mission. Try a different seed or report this as a bug."
        )
    return rocket


def _goal_dv(name: str) -> float:
    """Cheapest profile delta-v for a goal location, used for sphere
    ordering and 'hardest-goal' selection."""
    info = _parse_location(name)
    if info is None:
        return 0.0
    profiles = MISSION_PROFILES.get((info.body, info.mission_type), [])
    if not profiles:
        return 0.0
    return min(sum(e.base_dv for e in profile) for profile in profiles)


def _select_intermediates(
    world: "KSP1World",
    launch_dv: float,
    orbit_dv: float,
    goal_dv: float,
    signatures: dict[str, LocationSignature],
) -> tuple[list[str], list[str]]:
    """Pick random intermediate spheres between predictable anchors.

    Two bands (per plan):
      - launch → orbit: randint(1, 3) intermediates
      - orbit → goal:   randint(2, 7) intermediates

    Each intermediate's signature must be strictly between the band's
    endpoints in dv (filtering by dv only; full partial-order check
    happens during chain walking).
    """
    if goal_dv <= 0.0:
        # No goal sphere (e.g. complete_tech_tree) — no orbit→goal band.
        goal_dv = float("inf")

    pool_low: list[str] = []
    pool_mid: list[str] = []
    for name, sig in signatures.items():
        if name in ("Kerbin First Launch", "Kerbin Orbit 1"):
            continue
        if launch_dv < sig.dv < orbit_dv:
            pool_low.append(name)
        elif orbit_dv < sig.dv < goal_dv:
            pool_mid.append(name)

    rng = world.random
    n_low = min(rng.randint(1, 3), len(pool_low))
    n_mid = min(rng.randint(2, 7), len(pool_mid))
    chosen_low = rng.sample(pool_low, n_low) if n_low else []
    chosen_mid = rng.sample(pool_mid, n_mid) if n_mid else []
    return chosen_low, chosen_mid


def _predictable_spheres(world: "KSP1World") -> list[tuple[str, str]]:
    """Return (label, location_name) tuples for the always-enforced spheres.

    Phase 1: enforce S_launch, S_orbit, and ONE S_goal (the hardest by
    base dv).  Other goal locations get covered by the Pareto-frontier
    logic added in Phase 2.  Single-sphere goal here keeps the chain
    short and avoids inheriting the strictest rep-feasibility constraint
    across every goal location.
    """
    from .rules import goal_spec_location_names
    out: list[tuple[str, str]] = [
        ("S_launch", "Kerbin First Launch"),
        ("S_orbit", "Kerbin Orbit 1"),
    ]
    goal_names = list(goal_spec_location_names(world.goal_spec))
    # Skip goals not physics-gated by capability:
    # - Proxy goals (Eve/Tylo/Laythe Return/SR): rule is state.has_all
    #   progression items; capability can't model them.
    # - Tech tree goals: gate on accumulated science, not delta-v.
    #   Phase 3 adds tech-tier post-pass handling.
    feasible_goals = [
        n for n in goal_names
        if not _is_proxy_goal(n) and _parse_location(n) is not None
    ]
    if feasible_goals:
        hardest = max(feasible_goals, key=_goal_dv)
        out.append((f"S_goal[{hardest}]", hardest))
    return out


def _body_chain_depth(body_name: str) -> int:
    """SOI parent-chain depth from Kerbin. Kerbol=0; Kerbin=1; Mun/Jool=2;
    Pol/Bop/Vall/Laythe/Tylo=3."""
    body = BODY_BY_NAME.get(body_name)
    depth = 0
    while body is not None and body.parent is not None:
        depth += 1
        body = BODY_BY_NAME.get(body.parent)
    return depth


def _compute_location_signatures(
    world: "KSP1World",
) -> tuple[dict[str, LocationSignature], dict[str, dict[str, int]]]:
    """Pre-compute LocationSignature AND ``min_kit`` for every
    capability-gated player location.  Tech-tree, KSC, starting-
    inventory, and proxy goals are skipped (handled by Phase 1 / Phase
    3 paths).

    Each signature reflects ``minimal_rocket_for(loc, empty_kit, …)``
    — the absolute easiest reach from scratch — so the partial order
    is over the *intrinsic* difficulty of each location.

    Returns ``(signatures, min_kits)`` where ``min_kits[loc_name]`` is
    the dict of progressive items needed to reach ``loc_name`` from
    an empty kit.  Rule B's per-location self-ban consumes this.
    """
    rep_names = frozenset(
        rep
        for tiers in world.progressive_representatives.values()
        for rep in tiers.values()
    )
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    difficulty = ["casual", "normal", "expert", "insane"][
        world.options.difficulty.value
    ]
    pad_on = bool(world.options.progressive_launch_pad)
    clamps = bool(world.options.start_with_launch_clamps)

    sigs: dict[str, LocationSignature] = {}
    min_kits: dict[str, dict[str, int]] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if _is_proxy_goal(loc.name):
            continue
        info = _parse_location(loc.name)
        if info is None:
            continue  # tech tree / KSC / starting inventory
        rocket = minimal_rocket_for(
            loc.name, prior_kit={}, rep_names=rep_names,
            difficulty=difficulty, progressive_launch_pad=pad_on,
            start_with_clamps=clamps, rng=world.random,
            precollected_names=precollected_names,
        )
        if rocket is None:
            continue
        # Use the mission's intrinsic cheapest-profile dv as the
        # ordering scalar.  The mass-based ``rocket.profile_dv``
        # (launch_mass) does not order correctly across bodies because
        # heavier missions ≠ harder dv requirements.
        intrinsic_dv = _goal_dv(loc.name)
        sigs[loc.name] = LocationSignature(
            dv=intrinsic_dv,
            requirements=rocket.requirements,
            body_chain_depth=_body_chain_depth(info.body),
        )
        min_kits[loc.name] = dict(rocket.delta)
    return sigs, min_kits


def _install_tier_ban_rule(
    world: "KSP1World",
    ladder: SphereLadder,
    bootstrap_locations: set[str],
    location_min_kits: dict[str, dict[str, int]],
    band_funding: dict[int, SphereBoundary],
) -> None:
    """Rule B: for each non-bootstrap, capability-gated location L,
    ban progressive items (name, tier) that L would need to be reached.

    The ban set is the union of:
    (1) Items in ``min_kit_for(L)`` — direct chicken-and-egg avoidance.
        Catches incomparable-but-overlapping cases the sphere chain
        misses.
    (2) Items in any sphere ``S.delta`` where S is strictly less than
        L in the partial order — ensures the sphere chain's earlier
        items don't backfill onto harder locations.
    (3) Progressive R&D copies that the player wouldn't yet have enough
        science to safely spend.  R&D=B is a soft-lock guard: it must
        only be collectible once ``science ≥ cumulative_tier_cost(2*B)``,
        which by construction is when the player has the funding sphere
        ``S_B``'s cumulative kit.  So R&D=B is banned at any location L
        whose ``min_kit(L)`` isn't fully contained in ``S_B.cumulative``
        — i.e., L isn't reachable yet when band B becomes affordable.

    The ban is per-copy: if min_kit says ``Progressive LFO Tank: 2``,
    only tiers 1 and 2 are banned; tier 3+ copies remain free.
    """
    player = world.player

    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        loc_sig = ladder.location_signatures.get(loc.name)
        if loc_sig is None:
            continue

        banned_keys: set[tuple[str, int]] = set()

        # (1) Per-location self-ban: items in L's own min_kit cannot
        # land at L without a direct chicken-and-egg.
        loc_min_kit = location_min_kits.get(loc.name, {})
        for name, count in loc_min_kit.items():
            for tier in range(1, count + 1):
                banned_keys.add((name, tier))

        # (2) Sphere chain back-fill prevention.
        for sphere in ladder.spheres:
            sphere_sig = sphere.signature
            if sphere_sig is None:
                continue
            if not _strict_less(sphere_sig, loc_sig):
                continue
            for name, count in sphere.rocket.delta.items():
                for tier in range(1, count + 1):
                    banned_keys.add((name, tier))

        # (3) Progressive R&D soft-lock guard.
        # R&D=B should only be collectible at a location reachable with
        # S_B's cumulative kit; otherwise the player might not have
        # enough science yet to safely afford every band-B tech node.
        # Test: is min_kit(L) ⊆ S_B.cumulative?  If not, ban R&D=1..B at L.
        for band, funding_sphere in band_funding.items():
            S_B_cum = funding_sphere.rocket.cumulative
            reachable_at_S_B = all(
                S_B_cum.get(name, 0) >= count
                for name, count in loc_min_kit.items()
            )
            if not reachable_at_S_B:
                for tier in range(1, band + 1):
                    banned_keys.add((PROGRESSIVE_RD_NAME, tier))

        if not banned_keys:
            continue

        frozen_bans = frozenset(banned_keys)
        existing = loc.item_rule

        def _rule(item, _bans=frozen_bans, _p=player, _orig=existing) -> bool:
            # Honor any pre-existing rule first (e.g. mun_flag's
            # interplanetary-progression ban, starting-inventory's
            # local-only rule).  Then apply per-copy tier ban.
            if _orig is not None and not _orig(item):
                return False
            if item.player != _p:
                return True
            tier = getattr(item, "_sphere_tier", None)
            if tier is None:
                return True
            return (item.name, tier) not in _bans

        loc.item_rule = _rule


def _compute_tech_tier_signatures(
    world: "KSP1World",
    ladder: SphereLadder,
) -> tuple[dict[str, LocationSignature], dict[str, dict[str, int]], dict[int, SphereBoundary]]:
    """Post-pass: assign signatures + min-kits to tech-tree slot locations.

    Tech-tree access gates on accumulated science (and Progressive R&D
    per band).  Science is produced by completing mission locations,
    which the player can do at each sphere boundary in the chain.  So:

      1. Walk the accepted sphere chain in dv order.
      2. At each sphere, compute capability with that sphere's cumulative
         kit, then sum ``science_budget(...)`` across every body whose
         orbit is reachable — this is the science the player would have
         banked by reaching that sphere.
      3. For each tech tier T, find the first sphere whose science
         crosses ``cumulative_tier_cost(T)``.  That sphere's signature
         (augmented with ``progressive_rd >= TIER_TO_BAND[T]``) becomes
         tier T's hardness.

    Returns ``(signatures, min_kits)`` keyed by tech-tree location name.
    Tech-tree locations naturally inherit Rule B from these signatures
    via the existing ``_install_tier_ban_rule`` logic.
    """
    from .bodies import ALL_BODIES, science_budget
    from .capability import compute_capability_from_items
    from .locations import EventName, TECH_SLOTS_BY_DIFFICULTY, TechTreeLocation
    from .rules import _SCIENCE_SAFETY
    from .tech_tree import TECH_NODES, TIER_TO_BAND, cumulative_tier_cost

    difficulty_idx = world.options.difficulty.value
    difficulty_name = ["casual", "normal", "expert", "insane"][difficulty_idx]
    safety = _SCIENCE_SAFETY[difficulty_idx]
    pad_on = bool(world.options.progressive_launch_pad)
    clamps = bool(world.options.start_with_launch_clamps)
    rep_names = frozenset(
        rep
        for tiers in world.progressive_representatives.values()
        for rep in tiers.values()
    )
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )

    # Step 1: compute science accumulated at each sphere in the chain.
    # Same conservative _count_fn as minimal_rocket_for: only progressives
    # (which auto-grant reps via _pre_pass) and precollected items are
    # considered available.  Non-progressive parts that haven't been
    # placed by AP yet do not count toward the science budget.
    sphere_science: list[tuple[SphereBoundary, float]] = []
    for sphere in ladder.spheres:
        kit = sphere.rocket.cumulative

        def _count_fn(name: str, _k: dict[str, int] = kit,
                      _pre: frozenset[str] = precollected_names) -> int:
            if name in PROGRESSIVE_CAPS:
                return _k.get(name, 0)
            if name in _pre:
                return 1
            return 0

        cap, _flags = compute_capability_from_items(
            _count_fn, difficulty_name,
            start_with_clamps=clamps,
            rep_names=rep_names,
            progressive_launch_pad=pad_on,
        )
        total = 0.0
        for body in ALL_BODIES:
            body_cap = cap.bodies[body.name]
            if not body_cap.access[EventName.ORBIT]:
                continue
            total += science_budget(
                body, cap.has_thermometer, cap.has_barometer,
                cap.has_capsule, body_cap.access[EventName.CREWED_LANDING],
            )
        sphere_science.append((sphere, total * safety))

    # Step 2: per-tier, find funding sphere and assemble signature/kit.
    sigs: dict[str, LocationSignature] = {}
    min_kits: dict[str, dict[str, int]] = {}
    # band -> earliest funding sphere (lowest-dv sphere that funds any
    # tier in this band).  Used by the caller to inject Progressive R&D
    # into the chain at the right sphere depth.
    band_funding: dict[int, SphereBoundary] = {}
    num_slots = TECH_SLOTS_BY_DIFFICULTY[difficulty_idx]
    tier_set = sorted({n.tier for n in TECH_NODES})

    # Max kit needed at any point in the chain (per-name max over
    # sphere cumulatives — NOT max over deltas, which would miss
    # multi-sphere builds like Pad×2 across two spheres).  Used as the
    # min_kit for unfundable tech tiers: those locations cannot be
    # reached under this seed's goal, so they must not host any item
    # the chain needs.
    chain_full_kit: dict[str, int] = {}
    for sphere in ladder.spheres:
        for name, count in sphere.rocket.cumulative.items():
            chain_full_kit[name] = max(chain_full_kit.get(name, 0), count)

    # Sentinel "beyond goal" signature for unfundable tiers — must compare
    # strictly greater than every sphere so Rule B's sphere-chain back-fill
    # also bans chain deltas at these locations (belt-and-suspenders with
    # the self-ban driven by chain_full_kit).
    sphere_sigs = [s.signature for s in ladder.spheres if s.signature is not None]
    if sphere_sigs:
        max_dv = max(s.dv for s in sphere_sigs)
        union_reqs: dict[str, int] = {}
        for s in sphere_sigs:
            for k, v in s.requirements:
                union_reqs[k] = max(union_reqs.get(k, 0), v)
        sentinel_base = LocationSignature(
            dv=max_dv + 1.0e6,
            requirements=tuple(sorted(union_reqs.items())),
            body_chain_depth=max(s.body_chain_depth for s in sphere_sigs) + 100,
        )
    else:
        sentinel_base = None

    for tier in tier_set:
        target = cumulative_tier_cost(tier)
        rd_required = TIER_TO_BAND.get(tier, 0)
        funding: SphereBoundary | None = None
        for sphere, science in sphere_science:
            if science >= target and sphere.signature is not None:
                funding = sphere
                break

        if funding is None:
            # Tier unfundable: no sphere accumulates enough science to
            # reach it under this seed's goal.  Treat the locations as
            # post-goal and forbid every progressive item used anywhere
            # in the chain so they don't strand items.
            if sentinel_base is None:
                continue
            reqs_dict = dict(sentinel_base.requirements)
            if rd_required > 0:
                reqs_dict["progressive_rd"] = rd_required
            sig = LocationSignature(
                dv=sentinel_base.dv,
                requirements=tuple(sorted(reqs_dict.items())),
                body_chain_depth=sentinel_base.body_chain_depth,
            )
            kit = dict(chain_full_kit)
            if rd_required > 0:
                kit[PROGRESSIVE_RD_NAME] = rd_required
        else:
            # Augment requirements with Progressive R&D level if needed.
            reqs_dict = dict(funding.signature.requirements)
            if rd_required > 0:
                reqs_dict["progressive_rd"] = rd_required
            sig = LocationSignature(
                dv=funding.signature.dv,
                requirements=tuple(sorted(reqs_dict.items())),
                body_chain_depth=funding.signature.body_chain_depth,
            )
            # min_kit is the funding sphere's cumulative kit plus R&D.
            kit = dict(funding.rocket.cumulative)
            if rd_required > 0:
                kit[PROGRESSIVE_RD_NAME] = rd_required
            # Record this band's funding sphere if it's earlier than any
            # existing entry for this band.  Caller will use this to inject
            # Progressive R&D into the chain at the right depth.
            if rd_required > 0:
                prev = band_funding.get(rd_required)
                if prev is None or (
                    prev.signature is not None
                    and funding.signature.dv < prev.signature.dv
                ):
                    band_funding[rd_required] = funding

        for node in TECH_NODES:
            if node.tier != tier:
                continue
            for slot in range(1, num_slots + 1):
                loc_name = str(TechTreeLocation(node.display_name, slot))
                sigs[loc_name] = sig
                min_kits[loc_name] = kit

    return sigs, min_kits, band_funding


def _install_bootstrap_local_rule(world: "KSP1World") -> None:
    """Extend the local-only item rule (Rule A) to KSC biomes and the
    ``Kerbin First Launch`` location.  Starting-inventory locations
    already have this rule (locations.py:387).

    The default ``Location.item_rule`` is a no-op accepting everything;
    composing with it always returns True for the prior, so we replace
    it directly.  Other-player items already on this location at this
    point should not exist (pre_fill runs before main fill).
    """
    player = world.player

    def local_only(item, _p=player) -> bool:
        return item.player == _p

    extra_names = set(KSC_BIOME_NAMES) | {"Kerbin First Launch"}
    for loc in world.multiworld.get_locations(player):
        if loc.name in extra_names:
            loc.item_rule = local_only


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_sphere_ladder(world: "KSP1World") -> None:
    """Build the sphere ladder for ``world`` and install fill-time
    placement guidance.  Called from ``world.pre_fill``.

    Scope:
      - Compute LocationSignature for every capability-gated location.
      - Build the predictable ladder (S_launch / S_orbit / S_goal).
      - Add random intermediate spheres between the predictable anchors.
      - Walk the combined chain in dv order, accumulating kits.
      - Install Rule A (bootstrap-local) on KSC + First Launch.
      - Install Rule B (per-copy tier ban) on every non-bootstrap
        capability-gated location.
      - Register ``S_launch.delta`` as ``local_early_items``.
    """
    ladder = SphereLadder()
    # Compute signatures + min-kits up front (intrinsic; don't depend
    # on the chain).
    ladder.location_signatures, location_min_kits = _compute_location_signatures(world)

    # Predictable spheres provide the spine.
    predictable_labels = [(label, name) for label, name in _predictable_spheres(world)]
    predictable_names = {name for _, name in predictable_labels}

    # Look up dv of the three anchors (defaults if missing).
    launch_sig = ladder.location_signatures.get("Kerbin First Launch")
    orbit_sig = ladder.location_signatures.get("Kerbin Orbit 1")
    launch_dv = launch_sig.dv if launch_sig else 0.0
    orbit_dv = orbit_sig.dv if orbit_sig else 3400.0
    goal_dv = 0.0
    for label, name in predictable_labels:
        if label.startswith("S_goal"):
            sig = ladder.location_signatures.get(name)
            if sig:
                goal_dv = max(goal_dv, sig.dv)

    # Pick intermediates between launch→orbit and orbit→goal.
    chosen_low, chosen_mid = _select_intermediates(
        world, launch_dv, orbit_dv, goal_dv, ladder.location_signatures,
    )

    # Build full sphere list sorted by signature dv (tie-break on body
    # chain depth then name for determinism).
    all_sphere_names: list[tuple[str, str, bool]] = []
    for label, name in predictable_labels:
        all_sphere_names.append((label, name, True))
    for name in chosen_low:
        all_sphere_names.append((f"S_intermediate_lo[{name}]", name, False))
    for name in chosen_mid:
        all_sphere_names.append((f"S_intermediate_mid[{name}]", name, False))

    def _sort_key(entry):
        _label, name, _pred = entry
        sig = ladder.location_signatures.get(name)
        if sig is None:
            return (0.0, 0, name)
        return (sig.dv, sig.body_chain_depth, name)

    all_sphere_names.sort(key=_sort_key)

    cumulative: dict[str, int] = {}
    for label, location_name, is_pred in all_sphere_names:
        if is_pred:
            rocket = _build_rocket_or_raise(world, location_name, cumulative, label)
        else:
            rep_names = frozenset(
                rep
                for tiers in world.progressive_representatives.values()
                for rep in tiers.values()
            )
            precollected_names = frozenset(
                it.name for it in world.multiworld.precollected_items[world.player]
            )
            difficulty = ["casual", "normal", "expert", "insane"][
                world.options.difficulty.value
            ]
            rocket = minimal_rocket_for(
                location_name, cumulative, rep_names,
                difficulty,
                bool(world.options.progressive_launch_pad),
                bool(world.options.start_with_launch_clamps),
                world.random,
                precollected_names=precollected_names,
            )
            if rocket is None:
                continue  # intermediate: skip silently if infeasible
        ladder.spheres.append(SphereBoundary(
            name=label,
            location_name=location_name,
            is_predictable=is_pred,
            rocket=rocket,
            signature=ladder.location_signatures.get(location_name),
        ))
        cumulative = _kit_merge(cumulative, rocket.cumulative)

    ladder.cumulative_kit = cumulative

    # Post-pass: assign signatures + min-kits to tech-tree slot locations
    # by inverting accumulated science against the sphere chain.  Tech-tree
    # locations are gated on science (not capability dv), so they need their
    # own derivation path — but Rule B picks them up automatically once
    # they have an entry in location_signatures / location_min_kits.
    tech_sigs, tech_kits, band_funding = _compute_tech_tier_signatures(world, ladder)
    ladder.location_signatures.update(tech_sigs)
    location_min_kits.update(tech_kits)

    # R&D ban is handled directly in _install_tier_ban_rule via the
    # band_funding map (see soft-lock guard in Rule B part 3).  No
    # in-chain R&D injection: keeping the chain's delta/cumulative
    # honest about what's physically needed avoids partial-order
    # incomparability between funding spheres (which may have unique
    # reqs like relay_tier=3) and other location signatures.

    # Sentinel pass for ladder-UNREACHABLE mission locations.
    # ``_compute_location_signatures`` skips locations where
    # ``minimal_rocket_for`` returns None — i.e., the rep set cannot fly
    # the mission even with maxed progressives (e.g. Vall Sample Return
    # for a duna_return goal).  Without a signature, Rule B doesn't fire
    # and any progressive item can land there — including chain-critical
    # ones, which then strand.  Treat them as "post-goal": ban every
    # progressive item appearing anywhere in the chain.  Mirrors the
    # tech-tier unfundable handling.
    _chain_full_kit: dict[str, int] = {}
    for sphere in ladder.spheres:
        for name, count in sphere.rocket.cumulative.items():
            _chain_full_kit[name] = max(_chain_full_kit.get(name, 0), count)
    # Also include items required by any location's min_kit — critically,
    # Progressive R&D copies needed to unlock tech-tree locations that
    # hold chain items.  Without this, R&D copies can land at sentinel
    # locations and the player is locked out of tech-hosted chain items.
    for kit in location_min_kits.values():
        for name, count in kit.items():
            _chain_full_kit[name] = max(_chain_full_kit.get(name, 0), count)
    _sphere_sigs = [s.signature for s in ladder.spheres if s.signature is not None]
    if _sphere_sigs:
        _max_dv = max(s.dv for s in _sphere_sigs)
        _union_reqs: dict[str, int] = {}
        for s in _sphere_sigs:
            for k, v in s.requirements:
                _union_reqs[k] = max(_union_reqs.get(k, 0), v)
        _unreachable_sig = LocationSignature(
            dv=_max_dv + 1.0e6,
            requirements=tuple(sorted(_union_reqs.items())),
            body_chain_depth=max(s.body_chain_depth for s in _sphere_sigs) + 100,
        )
        for loc in world.multiworld.get_locations(world.player):
            if loc.address is None:
                continue
            if loc.name in ladder.location_signatures:
                continue
            if _parse_location(loc.name) is None:
                continue  # tech-tree / KSC / starting-inv handled elsewhere
            # Proxy goal locations (Eve/Tylo/Laythe Return + Sample Return)
            # are NOT excluded here.  They have a special access rule
            # (state.has_all_progression) but until that rule is satisfied
            # the player can't reach them, so chain items still strand.
            ladder.location_signatures[loc.name] = _unreachable_sig
            location_min_kits[loc.name] = dict(_chain_full_kit)

    world._sphere_ladder = ladder

    # Rule A: bootstrap-local restriction on starting-inv / KSC /
    # First Launch.  Starting-inventory locations already carry this
    # rule via locations.py; we extend to KSC + First Launch here.
    _install_bootstrap_local_rule(world)
    bootstrap_locations = set(KSC_BIOME_NAMES) | {"Kerbin First Launch"}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name.startswith("Starting Inventory"):
            bootstrap_locations.add(loc.name)

    # Rule B: per-copy progressive bans on capability-gated locations.
    _install_tier_ban_rule(world, ladder, bootstrap_locations, location_min_kits, band_funding)

    # Sphere-1 boost: items needed to clear S_launch get placed at sphere-0
    # locations by AP's distribute_early_items.
    launch_sphere = next(
        (s for s in ladder.spheres if s.location_name == "Kerbin First Launch"),
        None,
    )
    if launch_sphere is not None:
        launch_delta = launch_sphere.rocket.delta
        local_early = world.multiworld.local_early_items[world.player]
        for name, count in launch_delta.items():
            local_early[name] = max(local_early.get(name, 0), count)
