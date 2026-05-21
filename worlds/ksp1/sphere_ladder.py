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

import hashlib
from dataclasses import dataclass, field
from random import Random
from typing import Callable, TYPE_CHECKING, Optional

from Options import OptionError

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, DIFFICULTY_PROFILES, DifficultyProfile,
    MissionBuilder, MissionType,
)
from .capability import (
    EquipmentFlags, ProfileResult,
    _evaluate_sounding, _pre_pass, evaluate_mission_detailed,
)
from .capability_reasons import (
    BlockingInfo, BlockingReason, StageDiagnostic, StageFailure,
)
from .locations import (
    EVENT_BY_NAME, EventName, LocationBuilder, MissionLocation,
    KSC_BIOME_NAMES, KSC_LOCATION_PREFIX,
)
from .items import (
    PROGRESSIVE_LAUNCH_PAD_COUNT, PROGRESSIVE_LAUNCH_PAD_NAME,
    PROGRESSIVE_RD_COUNT, PROGRESSIVE_RD_NAME,
)
from .parts import (
    PROGRESSIVE_PART_COUNTS as _BASE_PROGRESSIVE_PART_COUNTS,
    PROGRESSIVE_PART_NAMES,
    PROGRESSIVE_PART_TIERS,
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


# Inverse map for warm-start construction: part_name -> [(chain, min_tier)].
# Used by ``_construct_warm_start_kit`` to translate the max-kit oracle's
# chosen parts into a starting kit for the bumper.  Built once at module
# load from PROGRESSIVE_PART_TIERS.
_PART_TO_PROGRESSIVE: dict[str, list[tuple[str, int]]] = {}
for _chain, _tiers in PROGRESSIVE_PART_TIERS.items():
    for _tier, _parts in _tiers.items():
        for _part in _parts:
            _PART_TO_PROGRESSIVE.setdefault(_part, []).append((_chain, _tier))
# Single-part chains not in PROGRESSIVE_PART_TIERS (just one item, no tier ladder).
_PART_TO_PROGRESSIVE.setdefault("rtg", []).append(("rtg", 1))


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
        "Progressive Xenon Tank",
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
        "Progressive Xenon Tank",
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
        # Radial engines provide extra thrust mounted off the main stack;
        # critical for some seeds' Kerbin ascent rocket dv when stack
        # engines alone aren't enough.
        "Progressive Radial Engine",
        # Mass-cap failures sometimes surface as "no viable stage" when
        # the optimizer rejects every candidate over the cap.
        "Progressive Launch Pad",
        # Payload-reducing chains: heavier terminal/support equipment
        # propagates to a heavier launch stage, which can flip an
        # otherwise-feasible ascent infeasible. F4 multi-stage's tighter
        # margins exposed cases where bumping these chains unlocks a
        # lighter terminal that the launch stage can lift.  Listed last
        # in the priority sense (ranked-bump puts engines/tanks first).
        "Progressive Capsule",
        "Progressive Probe Core",
        "Progressive Solar Panel",
        "Progressive Relay",
        "rtg",
        "Progressive Parachute",
        "Progressive Landing Leg",
        "Progressive Heat Shield",
    }),
    BlockingReason.NO_ENGINE: frozenset({
        "Progressive Vacuum Engine",
        "Progressive Launch Engine",
    }),
    BlockingReason.NO_LAUNCH_ENGINE: frozenset({"Progressive Launch Engine"}),
    BlockingReason.NO_FUEL: frozenset({
        "Progressive LFO Tank",
        "Progressive Xenon Tank",
    }),
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


# Progressive items intentionally NOT covered by `_BUMP_TABLE`.  These
# don't affect rocket capability so they don't help the greedy walk:
#   - Progressive R&D gates tech-tree access (handled by tech-tier
#     signatures + min-kit ban in `_install_tier_ban_rule`).
#   - Progressive Science Instrument affects science earnings, not
#     capability dv/mass.
_BUMP_TABLE_EXEMPT: frozenset[str] = frozenset({
    PROGRESSIVE_RD_NAME,
    "Progressive Science Instrument",
})


# Coverage assertion: every progressive item must either be a bump
# candidate for some blocking reason, or be listed as exempt.  This
# catches the class of bug where a new Progressive chain is added
# (e.g., Progressive SAS, Progressive Radial Engine) but the greedy
# walk never tries it because no `_BUMP_TABLE` entry references it.
_BUMP_TABLE_COVERED: frozenset[str] = frozenset().union(*_BUMP_TABLE.values())
_BUMP_TABLE_MISSING = (
    set(PROGRESSIVE_CAPS) - _BUMP_TABLE_COVERED - _BUMP_TABLE_EXEMPT
)
assert not _BUMP_TABLE_MISSING, (
    f"Sphere ladder: these progressive items appear in PROGRESSIVE_CAPS "
    f"but are not in any _BUMP_TABLE entry and not in _BUMP_TABLE_EXEMPT: "
    f"{sorted(_BUMP_TABLE_MISSING)}. Either add them to a relevant "
    f"BlockingReason's candidate set or declare them exempt."
)


# Engine fuel_type → progressive fuel chain. Used by the diagnostic-driven
# candidate function to bump the *right* fuel type when an engine has no
# compatible tank.
_FUEL_CHAIN_BY_ENGINE_TYPE: dict[str, str] = {
    "lfo": "Progressive LFO Tank",
    "lf": "Progressive LFO Tank",   # LFO covers LF via fuel-drop
    "xenon": "Progressive Xenon Tank",
    # Monoprop tanks are not progressive (specialized) — no chain to bump.
}


# Chains whose higher-tier reps tend to shrink an *upstream* stage's
# binding-constraint mass — either by reducing terminal equipment mass
# (lighter capsule/probe, fewer-but-stronger chutes/legs) or by improving
# upper-stage propulsion efficiency (Rhino at VE-t3 → less fuel for transit
# → less mass for the ascent stage to lift). Used by the payload audit
# when an upstream stage is TWR/dv-short but in-stage thrust+fuel are
# already maxed.
_PAYLOAD_MASS_REDUCING_CHAINS: tuple[str, ...] = (
    # Upper-stage propulsion efficiency (largest single lever).
    "Progressive Vacuum Engine",    # higher tier = more efficient/heavy-lift
    "Progressive Launch Engine",    # higher tier = better atm Isp
    # Terminal equipment mass.
    "Progressive Solar Panel",      # higher tier = lighter panels at distance
    "Progressive SAS",              # higher tier = lighter reaction wheel
    "Progressive Capsule",          # higher tier = built-in wheels + monoprop
    "Progressive Probe Core",       # higher tier = built-in wheels, lighter
    "Progressive Heat Shield",      # higher tier = larger shield, fewer needed
    "Progressive Parachute",        # higher tier = better drag, fewer needed
    "Progressive Landing Leg",      # higher tier = stronger, fewer needed
)


# Chains whose usefulness is *narrow* — they only matter when the mission
# actually exercises the corresponding equipment.  Bumping Heat Shield for
# a Mun-Orbit mission, or Landing Leg for a flyby, never resolves anything;
# the bumper just wastes an iteration.  Filtered out unless the mission
# profile says they're used.
_NARROW_CHAINS_REQUIRING_PROFILE_USE: frozenset[str] = frozenset({
    "Progressive Heat Shield",
    "Progressive Parachute",
    "Progressive Landing Leg",
    "Progressive Ladder",
})


def _relevant_narrow_chains(
    body_name: str,
    mission_type,
    crewed: Optional[bool],
    mission_builder: MissionBuilder,
) -> frozenset[str]:
    """Return the subset of ``_NARROW_CHAINS_REQUIRING_PROFILE_USE`` that
    the mission's profile alternatives actually exercise.

    Heat Shield is relevant iff some profile edge needs it (aerobrake or
    atmo-landing-aero).  Parachute is relevant iff some profile edge is
    an atmospheric descent that can use one (ATMO_LANDING_AERO or
    AEROBRAKE_CAPTURE).  Landing Leg is relevant iff some edge needs
    legs.  Ladder is relevant iff some edge needs a ladder, OR the
    mission is a SAMPLE_RETURN to a body with insufficient EVA jetpack
    TWR (capability's ``_inject_ladder`` injects the requirement at eval
    time, so it doesn't show up on the static profile edges).

    Narrow chains NOT in the returned set should be filtered out of the
    bumper's candidate pool for this mission.
    """
    from .bodies import EdgeType, BODY_BY_NAME
    from .capability import _MIN_EVA_JETPACK_TWR
    profiles = mission_builder.profiles_for(body_name, mission_type)
    if not profiles:
        # No physics profile (e.g., first-launch / sounding pseudo-events).
        # Be permissive — return the whole set so we don't accidentally
        # block a valid bump.
        return _NARROW_CHAINS_REQUIRING_PROFILE_USE
    relevant: set[str] = set()
    for profile in profiles:
        for edge in profile:
            if edge.needs_heat_shield:
                relevant.add("Progressive Heat Shield")
            if edge.needs_landing_legs:
                relevant.add("Progressive Landing Leg")
            if edge.needs_ladder:
                relevant.add("Progressive Ladder")
            if edge.edge_type in (EdgeType.ATMO_LANDING_AERO,
                                  EdgeType.AEROBRAKE_CAPTURE):
                relevant.add("Progressive Parachute")
    # Mirror capability._inject_ladder: a high-gravity sample-return
    # target needs a Kerbal to climb back into the craft, so a ladder
    # is required even when no static edge carries the flag.
    if mission_type == MissionType.SAMPLE_RETURN:
        body = BODY_BY_NAME.get(body_name)
        if body is not None and body.eva_jetpack_twr < _MIN_EVA_JETPACK_TWR:
            relevant.add("Progressive Ladder")
    return frozenset(relevant)


def _bump_candidates_for_stage_diag(
    diag: StageDiagnostic,
) -> frozenset[str]:
    """Map a stage-failure diagnostic to the set of progressive items
    that could plausibly resolve it.

    This *replaces* the generic ``_BUMP_TABLE[NO_VIABLE_STAGE]`` guess —
    each diagnostic narrows the candidate set to the items that physically
    address that failure mode.
    """
    f = diag.failure
    # Filter failures: only the relevant blocker.
    if f == StageFailure.HEAT_SHIELD_TOO_SMALL:
        return frozenset({"Progressive Heat Shield"})
    if f in (StageFailure.REQUIRE_GIMBAL_NONE, StageFailure.REQUIRE_THROTTLE_NONE):
        # The optimizer already exhausted available engines; need MORE engines.
        # In atmosphere, that's launch-engine tier; in vacuum it's vacuum.
        if diag.in_atmosphere:
            return frozenset({
                "Progressive Launch Engine",
                "Progressive SRB",
                "Progressive Radial Engine",
            })
        return frozenset({
            "Progressive Vacuum Engine",
            "Progressive Radial Engine",
        })
    if f == StageFailure.NO_ENGINES_AFTER_FILTER:
        if diag.in_atmosphere:
            return frozenset({
                "Progressive Launch Engine",
                "Progressive SRB",
                "Progressive Radial Engine",
            })
        return frozenset({
            "Progressive Vacuum Engine",
            "Progressive Radial Engine",
        })
    if f == StageFailure.NO_TANK_FOR_FUEL_TYPE:
        cands: set[str] = set()
        for ft in diag.engine_fuel_types_attempted:
            chain = _FUEL_CHAIN_BY_ENGINE_TYPE.get(ft)
            if chain:
                cands.add(chain)
        return frozenset(cands)
    if f == StageFailure.ENGINE_TOO_BIG_FOR_TANK:
        # Bigger tanks (higher LFO Tank tier) ship larger sizes.
        return frozenset({"Progressive LFO Tank"})
    if f == StageFailure.MASS_CAP_EXCEEDED:
        return frozenset({"Progressive Launch Pad"})
    # Performance failures — DV_SHORT / TWR_SHORT / DRY_MASS_KILLS_RATIO.
    # These all benefit from more thrust + more fuel. Pick by atmo vs vac.
    base: set[str] = {
        "Progressive LFO Tank",
        "Progressive Xenon Tank",
        "Progressive Stack Decoupler",
        "Progressive Radial Decoupler",
        "Progressive Engine Plate",
        "Progressive Radial Engine",
    }
    if diag.in_atmosphere:
        base |= {"Progressive Launch Engine", "Progressive SRB"}
    else:
        base |= {"Progressive Vacuum Engine"}
    if f == StageFailure.MASS_CAP_EXCEEDED:
        base.add("Progressive Launch Pad")
    return frozenset(base)


def _payload_mass_audit_candidates(
    flags: EquipmentFlags,
    kit: dict[str, int],
    narrow_relevant: Optional[frozenset[str]] = None,
) -> frozenset[str]:
    """When the bumper is stuck on a stage that can't lift its payload, the
    binding constraint may be downstream equipment mass rather than ascent
    thrust. Return chains whose next tier *might* reduce payload mass
    (lighter reps, built-in wheels, more efficient power).

    We can't easily compute the mass delta cheaply, so this is a heuristic:
    return chains not yet maxed where bumping has a known mass-reduction
    pathway. ``_pick_bump``'s evaluator filters out useless bumps anyway.

    Narrow chains (Heat Shield, Parachute, Landing Leg, Ladder) are only
    included when ``narrow_relevant`` says the mission actually uses them.
    """
    cands: set[str] = set()
    for chain in _PAYLOAD_MASS_REDUCING_CHAINS:
        cap = PROGRESSIVE_CAPS.get(chain, 1)
        if kit.get(chain, 0) >= cap:
            continue
        if chain in _NARROW_CHAINS_REQUIRING_PROFILE_USE:
            if narrow_relevant is not None and chain not in narrow_relevant:
                continue
        cands.add(chain)
    return frozenset(cands)


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
    # Home-body event/altitude locations — flat lookup spans all 15 home
    # bodies; the prefix in the location name uniquely identifies the body.
    hloc = LocationBuilder.all_home_locations().get(name)
    if hloc is not None:
        return _LocationMissionInfo(
            body=hloc.body,
            mission_type=hloc.mission_type,
            crewed=None,
            threshold_km=hloc.threshold_km,
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
    mission_builder: MissionBuilder,
) -> ProfileResult:
    """Dispatch to the right evaluator for a location's mission type."""
    if info.mission_type == MissionType.SOUNDING:
        return _evaluate_sounding(flags, info.threshold_km or 0.0,
                                  mission_builder.home_body)
    return evaluate_mission_detailed(
        flags, diff,
        info.body, info.mission_type, info.crewed,
        mission_builder,
        threshold_km=info.threshold_km,
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


# Chains whose usefulness is conditional on the seed's rep set: there's
# no point suggesting them when their consumer part isn't unlockable.
# Key: chain name → predicate(rep_names) returning True iff candidate is
# potentially useful for this seed.
_CONDITIONAL_CHAINS: dict[str, Callable[[frozenset[str]], bool]] = {
    # Xenon Tank only fuels ion engines.  ``ionEngine`` is the only stock
    # part with fuel_type=xenon and lives at Vacuum Engine tier 4 alongside
    # ``nuclearEngine`` (one of the two is the rep).  If the seed picked
    # nuclear, xenon tanks never serve a purpose — including them in the
    # bump candidate set wastes iterations to no effect.
    "Progressive Xenon Tank": lambda reps: "ionEngine" in reps,
}


def _wants_for_blocker(
    b: BlockingInfo,
    kit: dict[str, int],
    flags: Optional[EquipmentFlags] = None,
    rep_names: frozenset[str] = frozenset(),
    narrow_relevant: Optional[frozenset[str]] = None,
) -> frozenset[str]:
    """Return uncapped candidate items for a single blocker.

    For ``NO_VIABLE_STAGE`` blockers, this consults ``b.stage_diag`` for
    a structured near-miss reason and asks the optimizer-aware mapper.
    For all other blocker types, falls back to the hand-curated
    ``_BUMP_TABLE``.

    Chains in ``_CONDITIONAL_CHAINS`` are dropped when their predicate
    against ``rep_names`` says they're not useful for this seed.

    Chains in ``_NARROW_CHAINS_REQUIRING_PROFILE_USE`` are dropped when
    ``narrow_relevant`` (the set of narrow chains the current mission's
    profile actually exercises) does not include them.
    """
    if b.reason == BlockingReason.NO_VIABLE_STAGE and b.stage_diag is not None:
        cands = _bump_candidates_for_stage_diag(b.stage_diag)
    else:
        cands = _BUMP_TABLE.get(b.reason, frozenset())
    def _allowed(c: str) -> bool:
        if kit.get(c, 0) >= PROGRESSIVE_CAPS.get(c, 1):
            return False
        if not _CONDITIONAL_CHAINS.get(c, lambda _r: True)(rep_names):
            return False
        if c in _NARROW_CHAINS_REQUIRING_PROFILE_USE:
            if narrow_relevant is not None and c not in narrow_relevant:
                return False
        return True
    return frozenset(c for c in cands if _allowed(c))


def _pick_bump(
    blocking: list[BlockingInfo],
    kit: dict[str, int],
    rng: Random,
    evaluate_with_bump: Callable[[str], "tuple[bool, float, int]"],
    *,
    flags: Optional[EquipmentFlags] = None,
    enable_payload_audit: bool = False,
    enable_pair_lookahead: bool = False,
    evaluate_kit: Optional[Callable[[dict], "tuple[bool, float, int]"]] = None,
    rep_names: frozenset[str] = frozenset(),
    narrow_relevant: Optional[frozenset[str]] = None,
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

    When ``enable_payload_audit`` is True and no single bump improves the
    blocker count, expand the candidate set with payload-reducing chains
    (``_PAYLOAD_MASS_REDUCING_CHAINS``).  Useful when an upstream stage's
    binding constraint is downstream equipment mass.

    When ``enable_pair_lookahead`` is True and the augmented single-bump
    pool still has no feasible candidate, take the top-K
    blocker-reducing candidates and try all K² pair-bumps.  Returns the
    first item of a feasible pair (the second is found on the *next*
    bumper iteration).
    """
    wants: list[str] = []
    seen: set[str] = set()
    for b in blocking:
        for cand in sorted(_wants_for_blocker(
            b, kit, flags=flags, rep_names=rep_names,
            narrow_relevant=narrow_relevant,
        )):
            if cand in seen:
                continue
            seen.add(cand)
            wants.append(cand)

    has_mass_related_blocker = any(
        b.reason == BlockingReason.NO_VIABLE_STAGE
        and b.stage_diag is not None
        and b.stage_diag.failure in (
            StageFailure.DV_SHORT,
            StageFailure.TWR_SHORT,
            StageFailure.DRY_MASS_KILLS_RATIO,
            StageFailure.MASS_CAP_EXCEEDED,
        )
        for b in blocking
    )

    # When `wants` is empty (every primary candidate for the current
    # blocker set is at PROGRESSIVE_CAPS already), don't give up — if
    # we're stuck on a mass-related stage failure, the payload audit
    # may still find an indirect lever (e.g. higher Vacuum Engine tier
    # shrinks upper-stage mass, restoring atmospheric-ascent TWR).
    if not wants:
        if has_mass_related_blocker and flags is not None:
            audit = _payload_mass_audit_candidates(flags, kit, narrow_relevant=narrow_relevant)
            wants = sorted(audit)
        if not wants:
            return None

    def _score(cands: list[str]) -> list[tuple[int, float, int, int, float, str]]:
        out: list[tuple[int, float, int, int, float, str]] = []
        for cand in cands:
            feasible, mass, n_blocking = evaluate_with_bump(cand)
            feasibility_rank = 0 if feasible else 1
            # `mass` comes from evaluate_with_bump pre-processed: it's the
            # feasible launch_mass, or the partial-mass-attempt for stage-
            # failure cases (lower = closer to feasible), or inf for early
            # validation failures with no partial mass.  This lets the
            # scorer rank "payload-mass-reducing bump shrank the rocket"
            # higher than "fuel-tank bump made it heavier and still
            # infeasible" without changing the candidate set.
            out.append((
                feasibility_rank, mass, n_blocking,
                _group_index(cand), rng.random(), cand,
            ))
        return out

    scored = _score(wants)
    scored.sort()
    best_feasible = scored[0][0] == 0
    best_blocker_count = scored[0][2]

    # If a single bump unlocks feasibility, return it.
    if best_feasible:
        return scored[0][-1]

    # Payload audit (additive expansion): when no single bump unlocks
    # AND no single bump reduces blocker count AND the dominant blocker
    # is a stage-level mass/thrust issue, the binding constraint is
    # likely downstream equipment mass that we can shrink via
    # higher-tier reps. The mass-related-blocker gate prevents the
    # audit from poisoning tiebreaks for unrelated blockers (e.g. Relay
    # tier).
    current_blocker_count = len(blocking)
    if (enable_payload_audit
            and not best_feasible
            and has_mass_related_blocker
            and best_blocker_count >= current_blocker_count):
        audit = (
            _payload_mass_audit_candidates(flags, kit, narrow_relevant=narrow_relevant)
            if flags else frozenset()
        )
        extra = sorted(audit - set(wants))
        if extra:
            extra_scored = _score(extra)
            scored = scored + extra_scored
            scored.sort()
            if scored[0][0] == 0:
                return scored[0][-1]
            wants = wants + extra
            best_blocker_count = scored[0][2]

    # 2-step pair lookahead: try the top K candidates in all unordered
    # pairs. Catches the "needs 2 simultaneous bumps" tight cases that
    # single-bump greedy can't escape (e.g. Solar=3 + SAS=3 unlocks Vall
    # but neither alone does).
    if (enable_pair_lookahead
            and not best_feasible
            and evaluate_kit is not None):
        K = 6
        # Rank by blocker-count reduction then group index so we explore
        # the most promising items in pair combinations.
        top = [t[-1] for t in scored[:K]]
        best_pair: Optional[tuple[float, str, str]] = None
        for i, a in enumerate(top):
            for b_cand in top[i + 1:]:
                test_kit = dict(kit)
                test_kit[a] = test_kit.get(a, 0) + 1
                test_kit[b_cand] = test_kit.get(b_cand, 0) + 1
                f_pair, m_pair, _ = evaluate_kit(test_kit)
                if f_pair and (best_pair is None or m_pair < best_pair[0]):
                    best_pair = (m_pair, a, b_cand)
        if best_pair is not None:
            _, a, b_cand = best_pair
            # Return whichever of (a, b) is in an earlier priority group
            # (gives the bumper a deterministic, principled order).
            if _group_index(a) <= _group_index(b_cand):
                return a
            return b_cand

    return scored[0][-1]


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------

# Canonical-key cache for ``minimal_rocket_for``.  Many location names map
# to the same ``_LocationMissionInfo`` (e.g. ``Mun Landing 1``..``Mun Landing N``
# are all the same mission), and within one ``apply_sphere_ladder`` call the
# same (canonical_info, prior_kit) tuple is queried repeatedly.  Caching
# at this layer dedupes those calls before any oracle work runs.
#
# Cache is module-level; cleared at the top of ``apply_sphere_ladder`` so it
# never crosses worlds.  Key includes everything that affects the result
# (rep_names, difficulty, pad/clamps, precollected, mission_builder identity,
# prior_kit) and excludes ``rng`` — the cached MinimalRocket is the same
# regardless of which RNG would have been used for greedy tiebreakers,
# since the canonical mission only has one minimal-kit answer.
_CACHE_SENTINEL = object()

# DIAGNOSTIC ONLY — disabled in normal runs.  When enabled (set to a list
# instance), every _minimal_rocket_for_uncached call appends one
# (final_iter, returned_feasible) tuple.  Used by scratchpad/profile/
# bumper_iter_stats.py to evaluate whether raising the 200-iter cap
# would help.  Leave None in production.
_BUMPER_ITER_TRACE: Optional[list] = None


def _construct_warm_start_kit(
    info,
    rep_names: frozenset[str],
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    precollected_names: frozenset[str],
    mission_builder: MissionBuilder,
) -> dict[str, int]:
    """Run the capability oracle with a maxed-out progressive kit, then
    translate the parts it chose into the smallest kit that grants those
    same parts.  Returns a kit dict suitable for merging with ``prior_kit``
    as a warm start for the greedy bumper.

    Coverage gap: this only captures chains directly tied to a part
    (engine, tank, equipment, terminal command/support).  Gating chains
    that affect *configuration* without producing a named part — Engine
    Plate (multi-mount), Radial Decoupler (parallel staging), Launch Pad
    (mass cap) — are NOT captured here; the bumper fills those in on
    top of the warm start.
    """
    diff = DIFFICULTY_PROFILES[difficulty]

    def max_count_fn(name: str, _caps=PROGRESSIVE_CAPS,
                     _pre=precollected_names) -> int:
        if name in _caps:
            return _caps[name]
        if name in _pre:
            return 1
        return 0

    max_flags = _pre_pass_cached(
        # Build a kit dict at caps; _pre_pass_cached caches by kit_tuple.
        {chain: cap for chain, cap in PROGRESSIVE_CAPS.items()},
        start_with_clamps=start_with_clamps,
        rep_names=rep_names,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=mission_builder.launch_pad_caps,
        precollected_names=precollected_names,
    )
    result = evaluate_mission_detailed(
        max_flags, diff, info.body, info.mission_type, info.crewed,
        mission_builder, threshold_km=info.threshold_km or 0.0,
    )
    if not result.feasible:
        # Max kit can't reach this location at all — no warm start to give.
        return {}

    # Collect every part name referenced by the oracle's chosen build.
    used: set[str] = set()
    for sr in result.stage_results:
        if sr.engine_name and sr.engine_name != "(SRB integral)":
            used.add(sr.engine_name)
        if sr.tank_name and sr.tank_name != "(SRB integral)":
            used.add(sr.tank_name)
        for _count, pname in sr.equipment:
            used.add(pname)
    for _count, pname in result.terminal_parts:
        used.add(pname)

    # Reverse-map each part to the cheapest (lowest-tier) chain that grants
    # it.  Take max across all uses — if Engine X is in chain "Launch Engine"
    # tier 2, the warm start sets Launch Engine = 2.
    kit: dict[str, int] = {}
    for part in used:
        entries = _PART_TO_PROGRESSIVE.get(part)
        if not entries:
            continue  # non-progressive part (always granted, no kit cost)
        chain, tier = min(entries, key=lambda x: x[1])
        kit[chain] = max(kit.get(chain, 0), tier)
    return kit


def _derive_local_bumper_rng(
    info,
    prior_kit: dict[str, int],
    rep_names: frozenset[str],
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    precollected_names: frozenset[str],
) -> Random:
    """Build a ``Random`` whose seed is a stable hash of every input that
    distinguishes one ``minimal_rocket_for`` invocation from another.

    Stable means: same seed across process reboots (uses ``hashlib.sha256``
    rather than Python's randomized ``hash()``).  All event-slot
    duplicates ("Mun Landing 1" / "Mun Landing 2" / ...) share a
    canonical key, so they get an identical local RNG and therefore an
    identical bumper trajectory — making the canonical-key cache a pure
    perf optimization with no behavior shift.
    """
    h = hashlib.sha256()
    h.update(repr((
        info.body, str(info.mission_type), info.crewed, info.threshold_km,
        difficulty,
        progressive_launch_pad,
        start_with_clamps,
        tuple(sorted(prior_kit.items())),
        tuple(sorted(rep_names)),
        tuple(sorted(precollected_names)),
    )).encode("utf-8"))
    seed_int = int.from_bytes(h.digest()[:8], "big")
    return Random(seed_int)
_MINIMAL_ROCKET_CACHE: dict[tuple, Optional["MinimalRocket"]] = {}
_MINIMAL_ROCKET_CACHE_STATS: dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "bypassed_no_canonical": 0,  # _parse_location returned None
}

# Cross-call ``_pre_pass`` cache.  The bumper repeatedly evaluates kits that
# differ by a single bump, and many of those kits recur across different
# ``minimal_rocket_for`` calls (especially during ``_compute_location_signatures``
# where every call starts from ``prior_kit={}`` and bumps the same small set
# of candidates).  Caching by canonical (kit, options) avoids re-running
# ``_pre_pass`` for kits we've already seen.
#
# Lives next to the minimal_rocket_for cache; cleared together at the top
# of ``apply_sphere_ladder``.  Cached ``EquipmentFlags`` objects are shared
# between callers; downstream evaluators (``evaluate_mission_detailed`` and
# friends) treat ``flags`` as read-only.
_PRE_PASS_CACHE: dict[tuple, EquipmentFlags] = {}
_PRE_PASS_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


def clear_minimal_rocket_cache() -> None:
    """Reset the per-call caches and their hit/miss counters."""
    _MINIMAL_ROCKET_CACHE.clear()
    for k in _MINIMAL_ROCKET_CACHE_STATS:
        _MINIMAL_ROCKET_CACHE_STATS[k] = 0
    _PRE_PASS_CACHE.clear()
    for k in _PRE_PASS_CACHE_STATS:
        _PRE_PASS_CACHE_STATS[k] = 0


def get_minimal_rocket_cache_stats() -> dict[str, int]:
    """Snapshot the cache hit/miss/bypass counters."""
    return dict(_MINIMAL_ROCKET_CACHE_STATS)


def get_pre_pass_cache_stats() -> dict[str, int]:
    """Snapshot the _pre_pass cache hit/miss counters."""
    return dict(_PRE_PASS_CACHE_STATS)


def _pre_pass_cached(
    kit: dict[str, int],
    *,
    start_with_clamps: bool,
    rep_names: frozenset[str],
    progressive_launch_pad: bool,
    launch_pad_caps: tuple,
    precollected_names: frozenset[str],
) -> EquipmentFlags:
    """Cache-wrapped ``_pre_pass``.  Takes the kit dict directly rather than
    a closure so the cache key is hashable; builds the closure internally.
    Callers must treat the returned ``EquipmentFlags`` as read-only.
    """
    key = (
        tuple(sorted(kit.items())),
        start_with_clamps,
        rep_names,
        progressive_launch_pad,
        launch_pad_caps,
        precollected_names,
    )
    cached = _PRE_PASS_CACHE.get(key)
    if cached is not None:
        _PRE_PASS_CACHE_STATS["hits"] += 1
        return cached
    _PRE_PASS_CACHE_STATS["misses"] += 1

    def cf(name, _k=kit, _pre=precollected_names):
        if name in PROGRESSIVE_CAPS:
            return _k.get(name, 0)
        if name in _pre:
            return 1
        return 0

    flags = _pre_pass(
        cf,
        start_with_clamps=start_with_clamps,
        rep_names=rep_names,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=launch_pad_caps,
    )
    _PRE_PASS_CACHE[key] = flags
    return flags


def minimal_rocket_for(
    location_name: str,
    prior_kit: dict[str, int],
    rep_names: frozenset[str],
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    rng: Random,
    mission_builder: MissionBuilder,
    precollected_names: frozenset[str] = frozenset(),
) -> Optional[MinimalRocket]:
    """Cache wrapper around ``_minimal_rocket_for_uncached``.  See that
    function's docstring for the underlying contract."""
    info = _parse_location(location_name)
    if info is None:
        _MINIMAL_ROCKET_CACHE_STATS["bypassed_no_canonical"] += 1
        return None

    key = (
        info.body, info.mission_type, info.crewed, info.threshold_km,
        rep_names,
        difficulty,
        progressive_launch_pad,
        start_with_clamps,
        precollected_names,
        id(mission_builder),
        tuple(sorted(prior_kit.items())),
    )
    cached = _MINIMAL_ROCKET_CACHE.get(key, _CACHE_SENTINEL)
    if cached is not _CACHE_SENTINEL:
        _MINIMAL_ROCKET_CACHE_STATS["hits"] += 1
        return cached
    _MINIMAL_ROCKET_CACHE_STATS["misses"] += 1
    result = _minimal_rocket_for_uncached(
        location_name=location_name,
        prior_kit=prior_kit,
        rep_names=rep_names,
        difficulty=difficulty,
        progressive_launch_pad=progressive_launch_pad,
        start_with_clamps=start_with_clamps,
        rng=rng,
        mission_builder=mission_builder,
        precollected_names=precollected_names,
    )
    _MINIMAL_ROCKET_CACHE[key] = result
    return result


def _minimal_rocket_for_uncached(
    location_name: str,
    prior_kit: dict[str, int],
    rep_names: frozenset[str],
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    rng: Random,
    mission_builder: MissionBuilder,
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

    # Warm-start: ask the max-kit oracle which parts it would use, translate
    # those into the smallest progressive kit that grants them, and merge
    # with prior_kit (element-wise max, capped).  The bumper starts from
    # this merged kit and fills in any gating chains the oracle output
    # doesn't expose (Engine Plate, Radial Decoupler, Launch Pad).
    #
    # Doesn't displace the greedy loop — if the warm start is already
    # feasible, the loop exits at iter 0; if not, it bumps a handful of
    # remaining chains.  The win is fewer bumps total and (typically) a
    # smaller, structurally different kit that gives the AP fill solver
    # different Rule B bans to work with.
    _warm = _construct_warm_start_kit(
        info, rep_names, difficulty,
        progressive_launch_pad, start_with_clamps,
        precollected_names, mission_builder,
    )

    # Derive a deterministic per-canonical-key RNG.  The bumper's only
    # RNG use is the ``rng.random()`` tiebreaker in ``_pick_bump``'s score
    # tuple — same-priority candidates pick a random one to break ties.
    # When the canonical-key cache is enabled, every event-slot duplicate
    # (e.g. "Mun Landing 1" / "Mun Landing 2") shares one cache entry, so
    # without this derivation only the first call's RNG state would be
    # captured and the cached ``min_kit`` would depend on call order.
    # Deriving locally from (canonical_key, prior_kit, rep_names) makes
    # the result independent of whatever order the caller iterates
    # locations in, while still varying per-seed (rep_names is part of
    # the seed identity) and per-canonical-key.
    #
    # ``hashlib.sha256`` is used instead of Python's ``hash()`` because
    # the latter is randomized per process (PYTHONHASHSEED) and would
    # produce different RNGs across reruns of the same seed.
    rng = _derive_local_bumper_rng(
        info, prior_kit, rep_names, difficulty,
        progressive_launch_pad, start_with_clamps, precollected_names,
    )

    diff = DIFFICULTY_PROFILES[difficulty]
    # Start from prior_kit, then layer in the warm-start kit element-wise.
    # Cap to PROGRESSIVE_CAPS so we never start above legal kit size.
    kit: dict[str, int] = dict(prior_kit)
    for chain, tier in _warm.items():
        new_tier = max(kit.get(chain, 0), tier)
        cap = PROGRESSIVE_CAPS.get(chain, new_tier)
        kit[chain] = min(new_tier, cap)
    # Pre-compute which narrow chains (HS / Parachute / Legs / Ladder) the
    # mission actually exercises.  Bumping these for missions that don't
    # use them is a wasted iteration; the bumper filters them out.
    narrow_relevant = _relevant_narrow_chains(info.body, info.mission_type, info.crewed, mission_builder)

    def _evaluate_with_bump(cand: str) -> tuple[bool, float, int]:
        """Score a hypothetical bump of ``cand`` by running pre_pass +
        evaluate on a kit with that candidate incremented by one.
        Returns (feasible, launch_mass, n_blocking).  Used by
        ``_pick_bump`` to score candidates by capability increase.
        """
        trial_kit = dict(kit)
        trial_kit[cand] = trial_kit.get(cand, 0) + 1
        trial_flags = _pre_pass_cached(
            trial_kit,
            start_with_clamps=start_with_clamps,
            rep_names=rep_names,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            precollected_names=precollected_names,
        )
        trial_result = _evaluate(trial_flags, info, diff, mission_builder)
        # When infeasible, `launch_mass` carries the optimizer's partial-
        # mass-attempt (running payload at the failing stage).  Use it
        # so the scorer can rank "this bump got us closer" without needing
        # actual feasibility.  Zero means no partial info available
        # (early validation failure); treat as inf so it's deprioritized.
        if trial_result.feasible:
            mass_score = trial_result.launch_mass
        elif trial_result.launch_mass > 0.0:
            mass_score = trial_result.launch_mass
        else:
            mass_score = float("inf")
        return (
            trial_result.feasible,
            mass_score,
            len(trial_result.blocking),
        )

    def _evaluate_kit(trial_kit: dict[str, int]) -> tuple[bool, float, int]:
        """Score an arbitrary kit (not just a single bump from current).
        Used by pair-lookahead in ``_pick_bump``.
        """
        trial_flags = _pre_pass_cached(
            trial_kit,
            start_with_clamps=start_with_clamps,
            rep_names=rep_names,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            precollected_names=precollected_names,
        )
        trial_result = _evaluate(trial_flags, info, diff, mission_builder)
        # When infeasible, `launch_mass` carries the optimizer's partial-
        # mass-attempt (running payload at the failing stage).  Use it
        # so the scorer can rank "this bump got us closer" without needing
        # actual feasibility.  Zero means no partial info available
        # (early validation failure); treat as inf so it's deprioritized.
        if trial_result.feasible:
            mass_score = trial_result.launch_mass
        elif trial_result.launch_mass > 0.0:
            mass_score = trial_result.launch_mass
        else:
            mass_score = float("inf")
        return (
            trial_result.feasible,
            mass_score,
            len(trial_result.blocking),
        )

    # Safety bound: with 17 progressive groups and per-group caps ≤ 5,
    # the total possible bumps is ~50.  We allow 200 to absorb wasted
    # bumps when randomness picks an item that doesn't close any current
    # blocking.
    stuck_iters = 0   # consecutive iters where blocker count didn't drop
    prev_blocker_count = -1
    _final_iter = -1
    for _iter in range(200):
        _final_iter = _iter
        flags = _pre_pass_cached(
            kit,
            start_with_clamps=start_with_clamps,
            rep_names=rep_names,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            precollected_names=precollected_names,
        )
        result = _evaluate(flags, info, diff, mission_builder)
        if result.feasible:
            delta = {
                k: v - prior_kit.get(k, 0)
                for k, v in kit.items()
                if v - prior_kit.get(k, 0) > 0
            }
            if _BUMPER_ITER_TRACE is not None:
                _BUMPER_ITER_TRACE.append((_iter, True, location_name))
            return MinimalRocket(
                delta=delta,
                cumulative=dict(kit),
                flags=flags,
                profile_dv=result.launch_mass,
                requirements=_extract_requirements(flags),
            )
        # Track stuck-ness — enables payload audit + pair lookahead once
        # the simple greedy single-bump phase stops making progress.
        cur_blockers = len(result.blocking)
        if cur_blockers >= prev_blocker_count >= 0:
            stuck_iters += 1
        else:
            stuck_iters = 0
        prev_blocker_count = cur_blockers
        item = _pick_bump(
            result.blocking, kit, rng, _evaluate_with_bump,
            flags=flags,
            enable_payload_audit=stuck_iters >= 2,
            enable_pair_lookahead=stuck_iters >= 4,
            evaluate_kit=_evaluate_kit,
            rep_names=rep_names,
            narrow_relevant=narrow_relevant,
        )
        if item is None:
            if _BUMPER_ITER_TRACE is not None:
                _BUMPER_ITER_TRACE.append((_iter, False, location_name))
            return None
        kit[item] = kit.get(item, 0) + 1

    if _BUMPER_ITER_TRACE is not None:
        _BUMPER_ITER_TRACE.append((_final_iter, False, location_name))
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
        mission_builder=world.mission_builder,
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


def _goal_dv(name: str, mission_builder: MissionBuilder) -> float:
    """Cheapest profile delta-v for a goal location, used for sphere
    ordering and 'hardest-goal' selection."""
    info = _parse_location(name)
    if info is None:
        return 0.0
    profiles = mission_builder.profiles_for(info.body, info.mission_type)
    if not profiles:
        return 0.0
    return min(sum(e.base_dv for e in profile) for profile in profiles)


def _goal_relevant_bodies(world: "KSP1World") -> frozenset[str]:
    """Return body names that should be eligible as intermediate spheres
    for the player's goal.  A body is "relevant" iff:

    - it's in Kerbin's SOI (always: Kerbin/Mun/Minmus are warmup territory)
    - it's a goal body itself
    - it's a parent SOI of a goal body (containing-SOI chain)
    - it's a sibling of a goal body (shares the same parent SOI)
    - it's a child of a goal body (orbits the goal body)

    For a Minmus-only goal: {Kerbin, Mun, Minmus} — no interplanetary
    intermediates.  For a Duna goal: adds Sun-SOI planets + Duna's moons.
    Prevents the bumper from being forced to over-spend on Relay / Heat
    Shield to clear off-path intermediates.
    """
    from .bodies import ALL_BODIES, BODY_BY_NAME, BodyName, home_system_bodies
    from .rules import goal_spec_location_names

    # Home-system bodies are always relevant — the player has to fly
    # through them to leave home, so they're warmup territory.
    relevant: set[str] = {str(b) for b in home_system_bodies(world.mission_builder.home)}

    goal_bodies: set[str] = set()
    for loc_name in goal_spec_location_names(world.goal_spec):
        parsed = MissionLocation.parse(loc_name)
        if parsed is not None:
            goal_bodies.add(parsed.body)

    for g in goal_bodies:
        relevant.add(g)
        g_body = BODY_BY_NAME.get(g)
        if g_body is None:
            continue
        # Parent SOI chain
        cur = g_body.parent
        while cur is not None:
            relevant.add(cur)
            cur_body = BODY_BY_NAME.get(cur)
            cur = cur_body.parent if cur_body else None
        # Siblings + children
        for b in ALL_BODIES:
            if b.parent == g_body.parent and b.name != g_body.name:
                relevant.add(b.name)
            if b.parent == g_body.name:
                relevant.add(b.name)

    return frozenset(relevant)


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

    Filters:
      1. dv strictly between the band's endpoints
      2. body is "goal-relevant" (see ``_goal_relevant_bodies``) — keeps
         off-path bodies out (e.g. Gilly for a Minmus goal).
    """
    if goal_dv <= 0.0:
        # No goal sphere (e.g. complete_tech_tree) — no orbit→goal band.
        goal_dv = float("inf")

    relevant_bodies = _goal_relevant_bodies(world)
    home = str(world.mission_builder.home)
    bootstrap_names = (f"{home} First Launch", f"{home} Orbit 1")

    pool_low: list[str] = []
    pool_mid: list[str] = []
    for name, sig in signatures.items():
        if name in bootstrap_names:
            continue
        parsed = MissionLocation.parse(name)
        if parsed is None or parsed.body not in relevant_bodies:
            # Tech-tree / KSC / off-path body — not a valid intermediate.
            continue
        if launch_dv < sig.dv < orbit_dv:
            pool_low.append(name)
        elif orbit_dv < sig.dv < goal_dv:
            pool_mid.append(name)

    rng = world.random
    n_low = min(rng.randint(1, 3), len(pool_low))
    # Floor at 3 mid intermediates so the orbit→goal kit growth is spread
    # across several spheres rather than packed into S_goal (fewer fill errs
    # when the goal requires a lot of progressives — e.g. Duna Return).
    n_mid = min(rng.randint(3, 7), len(pool_mid))
    chosen_low = rng.sample(pool_low, n_low) if n_low else []
    chosen_mid = rng.sample(pool_mid, n_mid) if n_mid else []
    return chosen_low, chosen_mid


def _predictable_spheres(world: "KSP1World") -> list[tuple[str, str]]:
    """Return (label, location_name) tuples for the always-enforced spheres.

    S_launch + S_orbit anchor the bootstrap.  Then **every** physics-gated
    goal location becomes its own S_goal sphere.  For multi-mission goals
    like ``standard_sample_returns`` (11 sample-return locations), this
    ensures the chain's cumulative kit is verified to reach *all* of them
    — not just the hardest by dv.

    Without this, a seed whose rep set can reach the hardest-dv goal but
    not some easier-but-physics-different goal (e.g. Vall SR reachable
    but Duna SR not) produces a silent HARDFAIL: fill succeeds, but the
    game is unwinnable because one goal location isn't reachable.  With
    every goal as a predictable sphere, the chain walker either includes
    the kit for it (cumulative grows) or aborts via OptionError at
    pre_fill time — same loud failure mode as today's single-goal logic,
    just catching more cases.

    The min_kit-size sort handles ordering: smaller-kit goals are walked
    first, so the cumulative grows gradually through the goal band.

    Skip goals not physics-gated by capability:
      - Proxy goals (listed in ``world.model_infeasible_locations`` for
        the active home): rule is state.has_all progression items;
        capability can't model them.  They become reachable when the
        chain's cumulative covers every chain item.
      - Tech tree goals: gate on accumulated science.  Handled by the
        tech-tier post-pass.
    """
    from .rules import goal_spec_location_names
    home = str(world.mission_builder.home)
    out: list[tuple[str, str]] = [
        ("S_launch", f"{home} First Launch"),
        ("S_orbit", f"{home} Orbit 1"),
    ]
    goal_names = list(goal_spec_location_names(world.goal_spec))
    infeasible = world.model_infeasible_locations
    feasible_goals = [
        n for n in goal_names
        if n not in infeasible and _parse_location(n) is not None
    ]
    for goal_name in feasible_goals:
        out.append((f"S_goal[{goal_name}]", goal_name))
    return out


def _path_to_root(body_name: str) -> list[str]:
    """SOI parent chain from ``body_name`` up to (and including) Kerbol.

    The raw ``Body.parent`` field is ``None`` for both Kerbol itself and for
    every heliocentric planet (Kerbin, Eve, …).  We treat Kerbol as the
    implicit root for any planet whose parent is None, so all bodies share
    a single rooted tree useful for SOI-graph distance.
    """
    path: list[str] = []
    cur = BODY_BY_NAME.get(body_name)
    while cur is not None:
        path.append(cur.name)
        cur = BODY_BY_NAME.get(cur.parent) if cur.parent else None
    # Force Kerbol as the rooted ancestor for non-Kerbol bodies; the raw
    # data leaves the star and heliocentric planets at parent=None.
    if path and path[-1] != BodyName.KERBOL:
        path.append(BodyName.KERBOL)
    return path


def _body_chain_depth(body_name: str, home_name: str) -> int:
    """SOI graph distance from ``home_name`` to ``body_name``.

    Counts hops along the SOI tree (each moon is one hop from its parent;
    heliocentric planets are one hop from Kerbol).  Used as a tiebreaker
    when ordering location signatures.  Examples for Laythe-home:

        Laythe          -> 0
        Vall/Tylo/Bop   -> 2  (sibling: Laythe → Jool → Tylo)
        Jool            -> 1
        Kerbol          -> 2  (Laythe → Jool → Kerbol)
        Kerbin          -> 3
        Mun             -> 4
        Eve             -> 3
        Gilly           -> 4
    """
    if body_name == home_name:
        return 0
    body_path = _path_to_root(body_name)
    home_path = _path_to_root(home_name)
    body_idx = {name: i for i, name in enumerate(body_path)}
    for i, ancestor in enumerate(home_path):
        if ancestor in body_idx:
            return i + body_idx[ancestor]
    # Disjoint trees shouldn't happen with Kerbol as the forced root.
    return len(body_path) + len(home_path)


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
    infeasible = world.model_infeasible_locations

    sigs: dict[str, LocationSignature] = {}
    min_kits: dict[str, dict[str, int]] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name in infeasible:
            continue
        info = _parse_location(loc.name)
        if info is None:
            continue  # tech tree / KSC / starting inventory
        rocket = minimal_rocket_for(
            loc.name, prior_kit={}, rep_names=rep_names,
            difficulty=difficulty, progressive_launch_pad=pad_on,
            start_with_clamps=clamps, rng=world.random,
            mission_builder=world.mission_builder,
            precollected_names=precollected_names,
        )
        if rocket is None:
            continue
        # Use the mission's intrinsic cheapest-profile dv as the
        # ordering scalar.  The mass-based ``rocket.profile_dv``
        # (launch_mass) does not order correctly across bodies because
        # heavier missions ≠ harder dv requirements.
        intrinsic_dv = _goal_dv(loc.name, world.mission_builder)
        sigs[loc.name] = LocationSignature(
            dv=intrinsic_dv,
            requirements=rocket.requirements,
            body_chain_depth=_body_chain_depth(info.body, world.mission_builder.home),
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
    (2) Items in any sphere ``S.delta`` where ``min_kit(L)`` is NOT a
        subset of the PRIOR sphere's cumulative kit — i.e., reaching L
        would require items the player doesn't yet have when S's delta
        becomes available.  This is the chain-ordering invariant: an
        item bumped at sphere S must be at a location reachable with
        sphere S-1's cumulative kit, otherwise the player can't
        collect it in time.  Replaces the older partial-order strict-
        less check, which missed cases where L's signature is
        incomparable with S's (e.g. tech-tree locs with R&D reqs).
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

    # Pre-compute prior-sphere cumulative for each sphere index, used
    # by Rule B (2).  prior_cum[i] = kit the player has BEFORE
    # sphere i's delta bumps it.  prior_cum[0] = {} (empty).
    prior_cum: list[dict[str, int]] = [{}]
    for sphere in ladder.spheres:
        prior_cum.append(dict(sphere.rocket.cumulative))

    # chain_required[name] = max count of ``name`` the ladder actually
    # consumes anywhere in the chain.  Used to clamp Rule B (1) so it
    # only bans tiers the player must collect *for goal*; spare copies
    # (e.g. Capsule tiers 2/3 when the chain needs Capsule=1) are free
    # to land at unreachable-at-goal locations.
    chain_required: dict[str, int] = dict(ladder.cumulative_kit)

    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        loc_sig = ladder.location_signatures.get(loc.name)
        if loc_sig is None:
            continue

        banned_keys: set[tuple[str, int]] = set()

        # (1) Per-location self-ban: items in L's own min_kit cannot
        # land at L without a direct chicken-and-egg.  Clamp to
        # chain_required: a location's min_kit may demand more of a
        # chain than the ladder ever uses (e.g. Eve Crewed Landing
        # wants Capsule=3 but a Minmus-only goal's chain needs
        # Capsule=1).  Banning the extra tiers strands spare copies
        # at out-of-goal locations the player can't reach anyway.
        loc_min_kit = location_min_kits.get(loc.name, {})
        for name, count in loc_min_kit.items():
            effective = min(count, chain_required.get(name, 0))
            for tier in range(1, effective + 1):
                banned_keys.add((name, tier))

        # (2) Chain-ordering ban: an item bumped at sphere S must be
        # at a location reachable with S-1's cum kit.  If
        # min_kit(L) ⊈ sphere(S-1).cum, ban S's delta items at L.
        for i, sphere in enumerate(ladder.spheres):
            prior = prior_cum[i]  # sphere(i-1)'s cumulative, or {} for i=0
            reachable_with_prior = all(
                prior.get(name, 0) >= count
                for name, count in loc_min_kit.items()
            )
            if reachable_with_prior:
                continue
            # L isn't reachable with sphere(i-1)'s kit, so sphere(i)'s
            # delta items can't be collected here (player wouldn't have
            # them yet OR placing them here is a chicken-and-egg).
            for name, count in sphere.rocket.delta.items():
                # Only ban the SPECIFIC tiers this sphere introduces,
                # which are (prior_count+1)..(prior_count+count).
                prior_count = prior.get(name, 0)
                for tier in range(prior_count + 1, prior_count + count + 1):
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
    from .locations import EventName, TechTreeLocation, effective_tech_slots_per_node
    from .rules import effective_science_safety
    from .tech_tree import TECH_NODES, TIER_TO_BAND, cumulative_tier_cost

    difficulty_idx = world.options.difficulty.value
    difficulty_name = ["casual", "normal", "expert", "insane"][difficulty_idx]
    safety = effective_science_safety(world.options, difficulty_idx)
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
            mission_builder=world.mission_builder,
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
    num_slots = effective_tech_slots_per_node(world.options, difficulty_idx)
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


def _reclassify_spare_progressives(
    world: "KSP1World",
    ladder: SphereLadder,
    band_funding: dict[int, "SphereBoundary"],
) -> int:
    """Demote progressive item copies the ladder doesn't actually need
    for the goal to ``ItemClassification.useful``.

    AP's main fill constrains advancement items to *reachable* locations;
    useful items can land anywhere.  When `chain_required[X] < total[X]`,
    the spare copies of X don't gate progression for this goal — they
    only unlock cosmetic-or-quality-of-life upper tiers (e.g. Mk1-3 pod
    vs Mk1 pod).  Demoting them lets the fill scatter them past goal,
    relieving bootstrap dump pressure on the chains that *do* gate.

    chain_required is sourced from ``ladder.cumulative_kit`` for capability
    chains; R&D is added separately from ``band_funding`` because it's
    band-gated, not capability-gated, and isn't in ``cumulative_kit``.

    Rule B's per-tier bans still apply — only the AP reachability
    constraint changes.  Returns the number of copies demoted.
    """
    from BaseClasses import ItemClassification

    chain_required: dict[str, int] = dict(ladder.cumulative_kit)
    if band_funding:
        chain_required[PROGRESSIVE_RD_NAME] = max(band_funding.keys())

    player = world.player
    demoted = 0
    for item in world.multiworld.itempool:
        if item.player != player:
            continue
        tier = getattr(item, "_sphere_tier", None)
        if tier is None:
            continue
        if tier > chain_required.get(item.name, 0):
            item.classification = ItemClassification.useful
            demoted += 1
    return demoted


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

    home = str(world.mission_builder.home)
    extra_names = set(KSC_BIOME_NAMES) | {f"{home} First Launch"}
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
    clear_minimal_rocket_cache()
    from .rocket_math import clear_find_optimal_stage_cache
    clear_find_optimal_stage_cache()
    ladder = SphereLadder()
    # Compute signatures + min-kits up front (intrinsic; don't depend
    # on the chain).
    ladder.location_signatures, location_min_kits = _compute_location_signatures(world)

    # Predictable spheres provide the spine.
    predictable_labels = [(label, name) for label, name in _predictable_spheres(world)]
    predictable_names = {name for _, name in predictable_labels}

    # Look up dv of the three anchors (defaults if missing).
    home = str(world.mission_builder.home)
    launch_sig = ladder.location_signatures.get(f"{home} First Launch")
    orbit_sig = ladder.location_signatures.get(f"{home} Orbit 1")
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
        label, name, _pred = entry
        # Predictable spheres are anchored to canonical positions:
        # S_launch first (group 0), S_orbit next (group 1), intermediates
        # in the middle (group 2), and S_goal last (group 3).  Within
        # the intermediate band, sort by min_kit complexity first so
        # the chain walk grows gradually — sphere with small min_kit
        # adds a tiny delta on top of prior; sphere with large min_kit
        # absorbs the bigger jump only after smaller ones have built
        # up the cumulative kit.  This prevents the "Duna Landing 1
        # picked as first mid-band and gets 16 chain bumps in one
        # sphere" failure mode.
        if label == "S_launch":
            group = 0
        elif label == "S_orbit":
            group = 1
        elif label.startswith("S_goal"):
            group = 3
        else:
            group = 2
        sig = ladder.location_signatures.get(name)
        if sig is None:
            return (group, 0, 0.0, 0, name)
        min_kit_size = len(location_min_kits.get(name, {}))
        return (group, min_kit_size, sig.dv, sig.body_chain_depth, name)

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
                world.mission_builder,
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
    bootstrap_locations = set(KSC_BIOME_NAMES) | {f"{home} First Launch"}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name.startswith("Starting Inventory"):
            bootstrap_locations.add(loc.name)

    # Rule B: per-copy progressive bans on capability-gated locations.
    _install_tier_ban_rule(world, ladder, bootstrap_locations, location_min_kits, band_funding)

    # Demote progressive copies the ladder doesn't need to ``useful`` so
    # they can scatter past goal.  Must run after the ladder is built
    # (we need ``cumulative_kit`` and ``band_funding``) and before the
    # main fill examines item classifications.
    _reclassify_spare_progressives(world, ladder, band_funding)

    # Sphere-1 boost: items needed to clear S_launch get placed at sphere-0
    # locations by AP's distribute_early_items.
    launch_sphere = next(
        (s for s in ladder.spheres if s.location_name == f"{home} First Launch"),
        None,
    )
    if launch_sphere is not None:
        launch_delta = launch_sphere.rocket.delta
        local_early = world.multiworld.local_early_items[world.player]
        for name, count in launch_delta.items():
            local_early[name] = max(local_early.get(name, 0), count)
