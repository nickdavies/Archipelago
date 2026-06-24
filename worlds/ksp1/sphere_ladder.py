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

import logging
import os
# Non-progression (useful/filler) parts get a lower-bound placement floor a
# fixed fraction of the ladder below their tier: ``floor = max(0, ms - FRAC*n)``.
# The big margin keeps the band so wide that no same-tier category can
# over-subscribe its slice of the ladder (which strands the tail with no valid
# item->location matching → FillError), while still keeping a high-tier part out
# of the early game (powerful != early).  A per-capacity floor (the old
# ``cascade_lo``) is the "right" model but a static floor can't both pace a
# contended category AND leave it room; the wide flat margin sidesteps that and
# is overpower-bounded + solve-clean (0% fill failures at expert).  Tunable.
_USEFUL_FLOOR_MARGIN_FRAC = 0.30
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
    build_kit_for_result,
)
from .capability_reasons import (
    BlockingInfo, BlockingReason, StageDiagnostic, StageFailure,
)
from .locations import (
    EVENT_BY_NAME, EventName, LocationBuilder, MissionLocation,
    KSC_BIOME_NAMES, KSC_LOCATION_PREFIX,
)
from .contracts import (
    parse_contract_location_name, contract_payload_parts, required_part_breakdown,
    _CHAIN_GUARANTEED_CATEGORIES,
)
from .items import (
    PROGRESSIVE_LAUNCH_PAD_COUNT, PROGRESSIVE_LAUNCH_PAD_NAME,
    PROGRESSIVE_RD_COUNT, PROGRESSIVE_RD_NAME,
    PROGRESSIVE_VAB_NAME, PROGRESSIVE_VAB_COUNT,
    PROGRESSIVE_ASTRONAUT_COMPLEX_NAME, PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT,
)
from .parts import (
    CapabilityFlag,
    Decoupler,
    Engine,
    FuelTank,
    MiscEquipment,
    PART_DB,
)
from .part_geometry import PartRole
from .ranks import (
    DEFAULT_CONTEXT, RANK_AXES, RANK_AXES_BY_KEY, RankAxisKey, RankContext,
    max_rank_for, rank_sig_for, ranks_for_context,
)
from .requirements import Counted, Item, Rank, Signature, Threshold

# Parts providing the basic temperature/pressure instruments that
# ``bankable_science`` credits on every body.  Computed from PART_DB by
# capability flag so modded instruments are picked up automatically.
_BASIC_SCIENCE_INSTRUMENTS: frozenset[str] = frozenset(
    nm for nm, parts in PART_DB.items()
    if any(isinstance(p, MiscEquipment)
           and (CapabilityFlag.THERMOMETER in p.provides
                or CapabilityFlag.BAROMETER in p.provides)
           for p in parts)
)


# Deep-interplanetary enabler parts — injected into the cumulative kit once the
# chain crosses ``_DEEP_INJECT_DV_FRAC`` of its dv range, but ONLY for seeds whose
# hardest mission is genuinely interplanetary (``_DEEP_INJECT_MIN_DV``).  The
# high-dv transfer stages of deep missions (~8.5 km/s vacuum burns) close ONLY
# via a high-Isp nuclear engine (serial) or chemical asparagus (fuel-line
# crossfeed) — both are top-rank outliers the reps-only bumper reaches only ~45%
# of the time, so without this it dead-ends on DRY_MASS_KILLS_RATIO and the goal
# anchor raises OptionError.  See ``project_060_deep_interplanetary_enablers``.
# nuclear + fuel line are named (critical, present in every pack — user-approved);
# the radial decoupler (sheds the asparagus booster ring) is derived by property.
_NUCLEAR_ENGINE_NAME = "nuclearEngine"
_FUEL_LINE_NAME = "fuelLine"
_LIGHTEST_RADIAL_DECOUPLER: Optional[str] = min(
    (nm for nm, parts in PART_DB.items()
     if any(isinstance(p, Decoupler) and p.kind == "radial" for p in parts)),
    key=lambda nm: PART_DB[nm][0].mass, default=None,
)
_DEEP_SPACE_ENABLERS: frozenset[str] = frozenset(
    n for n in (_NUCLEAR_ENGINE_NAME, _FUEL_LINE_NAME, _LIGHTEST_RADIAL_DECOUPLER)
    if n is not None and n in PART_DB
)
# Inject once the chain's dv crosses this fraction of its max (mid-run band), and
# only when that max is interplanetary-deep — keeps Mun/Minmus/simple seeds free
# of the enablers (preserves early-game variance; the bumper finds its own kit).
_DEEP_INJECT_DV_FRAC: float = 0.5
_DEEP_INJECT_MIN_DV: float = 12000.0


if TYPE_CHECKING:
    from .world import KSP1World
    from .contracts import ContractSpec


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


@dataclass
class SphereBoundary:
    """A sphere in the rank-space ladder.

    ``provides`` is the cumulative capability signature (rank ceilings +
    counted-progressive levels) the sphere proves; ``delta`` is the
    increment over the prior sphere.  Both fold the old split
    ``(MinimumRanks ranks, dict extras)`` into one :class:`Signature`.
    ``reps_collected`` is the union of all bumper-selected reps through
    this sphere — used by chain-walker reps-only feasibility proofs and
    by downstream tech-tier band funding.
    """
    name: str
    location_name: str
    is_predictable: bool
    provides: Signature
    delta: Signature
    reps_collected: frozenset[str] = frozenset()
    flags: EquipmentFlags = field(default_factory=lambda: None)  # type: ignore[arg-type]
    profile_dv: float = 0.0
    signature: Optional[LocationSignature] = None


@dataclass
class SphereLadder:
    """The complete ladder for one seed; stored on the world."""
    spheres: list[SphereBoundary] = field(default_factory=list)
    # Total cumulative kit after walking the whole chain (for diagnostics).
    cumulative_kit: dict[str, int] = field(default_factory=dict)
    # Per-location signature, used by tests/diagnostics.
    location_signatures: dict[str, LocationSignature] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LocationMissionInfo:
    body: str
    mission_type: MissionType
    crewed: Optional[bool]
    threshold_km: Optional[float]
    # Contract-completion locations carry their ContractSpec; the delivery
    # payload is then sized per-rung from that rung's flags (via
    # ``contract_payload_parts``) so the ladder signature matches the runtime
    # access rule. ``None`` for ordinary (non-contract) missions.
    spec: Optional["ContractSpec"] = None
    # Whether this location's mission requires a Kerbal EVA — drives the
    # curated Astronaut-Complex ``can_eva`` gate (buildings_in_logic).  For
    # FLAG_PLANT/SAMPLE_RETURN this is implied by mission_type; for EVA-in-orbit
    # it comes from the EventDef (it shares the ORBIT type).  ``None`` means
    # "let the evaluator derive it from mission_type".
    requires_eva: Optional[bool] = None


def _parse_location(name: str) -> Optional[_LocationMissionInfo]:
    """Resolve a location name to a (body, mission_type, crewed, threshold)
    tuple.  Returns ``None`` for locations whose access isn't physics-gated
    (tech tree, KSC biomes, starting inventory, body-agnostic Splashdown)
    — those need their own handling and are skipped by the greedy walk.
    """
    # Home-body event/altitude locations — flat lookup spans all 15 home
    # bodies; the prefix in the location name uniquely identifies the body.
    hloc = LocationBuilder.all_home_locations().get(name)
    if hloc is not None:
        # Body-agnostic entries (Splashdown) have no single body to drive
        # the bumper's mission-centric work; their requirements are
        # dominated by per-body LAND missions the bumper already handles.
        if hloc.body is None:
            return None
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
            requires_eva=event_def.requires_eva,
        )
    # Contract completion locations: physics-gated like a mission of the
    # contract's base type, but with the required equipment as delivered payload.
    spec = parse_contract_location_name(name)
    if spec is not None:
        td = spec.type_def
        return _LocationMissionInfo(
            body=spec.body,
            mission_type=td.base_mission_type,
            crewed=td.crewed,
            threshold_km=None,
            spec=spec,
        )
    # Tech tree / KSC / starting inventory: not capability-gated.
    return None


def _mission_key(info: "_LocationMissionInfo") -> tuple:
    """Canonical dedup key for a location's mission — locations sharing it get
    one capability evaluation (signature + feasibility bracket).

    Ordinary missions key on trajectory only ``(body, type, crewed, threshold)``
    so the event slots that share one mission collapse (e.g. Mun Landing 1/2/3 —
    deliberate, see project memory).

    Contract locations additionally key on ``contract_id`` (type:body): two
    contracts with the same base trajectory can still differ in BOTH the
    required-part payload the bracket charges (Transmit Science needs a relay
    antenna; a bare Orbit contract needs none) AND the mission transform that
    rewrites the edge dv (POLAR/STATIONARY inject extra burns). Collapsing them
    onto the trajectory key drops those distinctions, bracketing a contract
    EARLIER than the sphere that actually grants its required rep — the runtime
    rule then strands that rep on the contract's own reward location."""
    base = (info.body, info.mission_type, info.crewed, info.threshold_km)
    if info.spec is not None:
        return base + (info.spec.contract_id,)
    return base


# A chain-guaranteed contract category that has no part at the current kit maps
# to the chain whose bump unlocks it, so the bumper knows what to add. Promoted
# standalone categories never land here — contract_payload_parts supplies their
# guaranteed representative directly.
_PAYLOAD_CATEGORY_BLOCKING: dict[str, BlockingReason] = {
    "crew_cabin": BlockingReason.NO_CAPSULE,
    "relay": BlockingReason.RELAY_TIER_TOO_LOW,
    "power": BlockingReason.INSUFFICIENT_POWER_SOLAR_OK,
}
assert set(_PAYLOAD_CATEGORY_BLOCKING) == _CHAIN_GUARANTEED_CATEGORIES, (
    "every chain-guaranteed contract category needs a bump-chain mapping"
)


def _missing_payload_blocking(
    spec: "ContractSpec", flags: EquipmentFlags,
) -> list[BlockingInfo]:
    """Blocking info for each chain-guaranteed required category with no part at
    this kit — drives the bumper to add Progressive Capsule / Relay / Solar."""
    return [
        BlockingInfo(reason=_PAYLOAD_CATEGORY_BLOCKING[cat])
        for cat, got in required_part_breakdown(spec, flags)
        if got is None and cat in _CHAIN_GUARANTEED_CATEGORIES
    ]


def _contract_payload_rep_names(info: "_LocationMissionInfo",
                                flags: EquipmentFlags) -> set[str]:
    """Names of a contract location's delivery payload parts (drill / ore_tank /
    battery / science_lab / crew cabins).  These are PAYLOAD, not rank reps, so
    the bumper never designates them — but the contract location REQUIRES them.
    Fold them into the location's ``reps_collected`` at each feasible return so
    the demote keep-set (cumulative_reps) keeps them PROGRESSION and the fill
    can't strand them (same mechanism as the science-instrument inject).
    Standalone categories resolve via their lightest-member fallback; chain-axis
    members are already reps, so adding their names is a harmless no-op.  Empty
    for non-contract locations or when no payload is needed."""
    if info.spec is None:
        return set()
    cp = contract_payload_parts(info.spec, flags)
    return {p.name for p in cp} if cp else set()


def _evaluate(
    flags: EquipmentFlags,
    info: _LocationMissionInfo,
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
    run_parallel: bool = True,
) -> ProfileResult:
    """Dispatch to the right evaluator for a location's mission type.

    ``run_parallel=False`` (used by the bumper's guidance trials) skips the
    exact asparagus search — serial mass is a cheap, order-preserving proxy for
    ranking candidate bumps; the main-loop feasibility check and rescue keep the
    exact parallel build.
    """
    if info.mission_type == MissionType.SOUNDING:
        return _evaluate_sounding(flags, info.threshold_km or 0.0,
                                  mission_builder.home_body)
    extra_payload: tuple = ()
    mission_transform = None
    if info.spec is not None:
        payload = contract_payload_parts(info.spec, flags)
        if payload is None:
            # A chain-guaranteed delivery part (crew/relay/power) isn't unlocked
            # at this kit — infeasible here; name the chain so the bumper bumps it.
            return ProfileResult(
                False, blocking=_missing_payload_blocking(info.spec, flags))
        extra_payload = payload
        # Same edge modifier the runtime rule applies — polar ascent penalty /
        # stationary raise at home — so the signature isn't optimistic there.
        mission_transform = info.spec.mission_transform(mission_builder)
    return evaluate_mission_detailed(
        flags, diff,
        info.body, info.mission_type, info.crewed,
        mission_builder,
        threshold_km=info.threshold_km,
        extra_payload_parts=extra_payload,
        mission_transform=mission_transform,
        requires_eva=info.requires_eva,
        run_parallel=run_parallel,
    )


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
    # LF Tank only fuels nuclear engines (with full fuel mass — vs LFO
    # tanks which Nerv can drain of LF only, at half the original fuel
    # mass).  If the seed didn't roll nuclear, LF tanks never help.
    "Progressive LF Tank": lambda reps: "nuclearEngine" in reps,
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
# Rank-space sphere walker — Phase 1 scaffold.
#
# This subsystem mirrors the progressive-tier walker above using
# ``Signature`` ceilings keyed by ``RankAxisKey``.  It runs *alongside*
# the existing walker (does not replace it yet); the existing walker
# remains the placement authority.  The rank walker's purpose is to:
#   1. Compute per-sphere minimum rank ceilings for the predictable
#      anchors (S_launch / S_orbit / S_goal).
#   2. Record designated reps — one part per ``(axis, rank)`` bump — so
#      later phases can drive per-seed variance from real bumper output.
#
# Phase 2 will swap this in as the source of truth and retire the
# progressive-tier walker.  Until then, the rank walker is permitted to
# fail silently; failures emit a single warning and the existing walker's
# output stands.
# ---------------------------------------------------------------------------


# BlockingReason → rank axes that could plausibly resolve it.  Hash order is
# stable for cache hashing; per-priority-group selection happens in
# ``_RANK_PRIORITY_GROUPS``.
_RANK_BUMP_TABLE: dict[BlockingReason, tuple[RankAxisKey, ...]] = {
    # A stage that can't close (dv/twr/dry-mass) is fixed only by propulsion,
    # staging, or a lighter command module (payload reduction) — so the lever set
    # is restricted to those.  Equipment is deliberately excluded:
    #   * LANDING_LEG / RELAY / SOLAR / SAS are pure mass with no dv or profile
    #     effect; they reach the kit via their own pre-check blockers
    #     (LANDING_LEGS_MISSING, RELAY_TIER_TOO_LOW, ...).
    #   * HEAT_SHIELD / PARACHUTE *do* cut dv (they unlock the cheap
    #     ATMO_LANDING_AERO profile, dv=100 m/s, over a propulsive landing —
    #     bodies.py:1266) — but the capability surfaces that precisely: it returns
    #     the UNION of blockers across all profile alternatives (capability.py:
    #     2586), emitting NO_HEAT_SHIELD / NO_PARACHUTE when an aero profile is the
    #     cheaper unlock, which map to those axes.
    # Trialing all six here instead was ~44% of all bumper trials, ~94% no-ops
    # (KSP_BUMP_STATS): each candidate costs a full serial FOS, and equipment can
    # never close a stage failure.
    BlockingReason.NO_VIABLE_STAGE: (
        RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK, RankAxisKey.XENON_TANK,
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE,
        RankAxisKey.STACK_DECOUPLER, RankAxisKey.SRB,
        RankAxisKey.RADIAL_DECOUPLER,
        RankAxisKey.CAPSULE, RankAxisKey.PROBE_SAS,
    ),
    BlockingReason.NO_ENGINE: (
        RankAxisKey.VAC_ENGINE, RankAxisKey.LAUNCH_ENGINE,
    ),
    BlockingReason.NO_LAUNCH_ENGINE: (RankAxisKey.LAUNCH_ENGINE,),
    BlockingReason.NO_FUEL: (
        RankAxisKey.LFO_TANK, RankAxisKey.XENON_TANK, RankAxisKey.LF_TANK,
    ),
    BlockingReason.NO_PROPULSION: (
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.LFO_TANK, RankAxisKey.SRB,
    ),
    BlockingReason.STAGING_TIER_INSUFFICIENT: (RankAxisKey.STACK_DECOUPLER,),
    BlockingReason.NO_HEAT_SHIELD: (RankAxisKey.HEAT_SHIELD,),
    BlockingReason.RELAY_TIER_TOO_LOW: (RankAxisKey.RELAY,),
    BlockingReason.INSUFFICIENT_POWER_SOLAR_OK: (RankAxisKey.SOLAR,),
    # RTG is a discrete unlock today; not modeled on a rank axis (Phase 2).
    BlockingReason.INSUFFICIENT_POWER_NEEDS_RTG: (),
    BlockingReason.NO_PROBE_CORE: (RankAxisKey.PROBE_SAS,),
    BlockingReason.NO_CAPSULE: (RankAxisKey.CAPSULE,),
    BlockingReason.NO_COMMAND_MODULE: (
        RankAxisKey.CAPSULE, RankAxisKey.PROBE_SAS,
    ),
    BlockingReason.PARACHUTE_TERMINAL_VELOCITY: (RankAxisKey.PARACHUTE,),
    BlockingReason.NO_PARACHUTE: (RankAxisKey.PARACHUTE,),
    BlockingReason.NO_SAFE_DESCENT: (RankAxisKey.PARACHUTE,),
    BlockingReason.CAPSULE_SOUNDING_INCOMPLETE: (
        RankAxisKey.PARACHUTE, RankAxisKey.STACK_DECOUPLER,
    ),
    BlockingReason.LANDING_LEGS_MISSING: (RankAxisKey.LANDING_LEG,),
    # Ladder doesn't have a rank axis yet — handled discretely in Phase 2.
    BlockingReason.NO_LADDER: (),
    # Launch Pad is a counted progressive in Phase 1; rank gating in Phase 2.
    BlockingReason.LAUNCH_MASS_EXCEEDED: (),
    BlockingReason.SOUNDING_ALTITUDE_TOO_LOW: (
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.LFO_TANK, RankAxisKey.SRB,
    ),
    BlockingReason.NO_SOUNDING_ALTITUDE: (
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.LFO_TANK, RankAxisKey.SRB,
    ),
    BlockingReason.NO_ATTITUDE_CONTROL: (
        RankAxisKey.SAS, RankAxisKey.PROBE_SAS, RankAxisKey.CAPSULE,
    ),
}


# Ranked priority groups for choosing which axis to bump next.  The bumper
# walks groups in order and picks within the first group that overlaps
# the candidate set.
RANK_PRIORITY_GROUPS: tuple[frozenset[RankAxisKey], ...] = (
    # Group 1: thrust/fuel.
    frozenset({
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE,
        RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK, RankAxisKey.XENON_TANK,
        RankAxisKey.MONOPROP_TANK, RankAxisKey.SRB,
        RankAxisKey.STACK_DECOUPLER, RankAxisKey.RADIAL_DECOUPLER,
    }),
    # Group 2: aero/heat + attitude.
    frozenset({
        RankAxisKey.HEAT_SHIELD, RankAxisKey.PARACHUTE,
        RankAxisKey.PROBE_SAS, RankAxisKey.SAS,
    }),
    # Group 3: power.
    frozenset({RankAxisKey.SOLAR}),
    # Group 4: control / landing / payload.
    frozenset({RankAxisKey.LANDING_LEG, RankAxisKey.CAPSULE}),
    # Group 5: blow-open.
    frozenset({RankAxisKey.RELAY}),
)


# Per-axis cache of `[items_at_rank_K]` for K in 1..buckets.  Used by the
# rep recorder to pick a random part among the new admissions when a
# rank ceiling rises from K-1 to K.  Keyed by ``RankContext`` since the
# rank table varies (only ``SRB`` differs today, but the cache is general).
_ITEMS_AT_RANK_CACHE: dict[
    RankContext, dict[RankAxisKey, dict[int, tuple[str, ...]]]
] = {}


def _items_at_rank(ctx: RankContext) -> dict[RankAxisKey, dict[int, tuple[str, ...]]]:
    cached = _ITEMS_AT_RANK_CACHE.get(ctx)
    if cached is not None:
        return cached
    table = ranks_for_context(ctx)
    out: dict[RankAxisKey, dict[int, tuple[str, ...]]] = {}
    for axis_key, items in table.items():
        by_rank: dict[int, list[str]] = {}
        for name, rank in items.items():
            by_rank.setdefault(rank, []).append(name)
        out[axis_key] = {r: tuple(sorted(names)) for r, names in by_rank.items()}
    _ITEMS_AT_RANK_CACHE[ctx] = out
    return out


def _rank_admits_item(item_name: str, ranks: Signature, ctx: RankContext) -> bool:
    """Item is admitted iff every axis it participates in has the item's
    rank ≤ the ceiling on that axis.  Items with empty rank_sig (filler,
    non-ranked progressives) are admitted unconditionally.

    An axis absent from ``ranks`` reports level 0; since real part ranks
    are ≥ 1, ``item_rank > ranks.rank(axis)`` then fails — preserving the
    old "absent axis = unavailable" meaning.
    """
    sig = rank_sig_for(item_name, ctx)
    if not sig.axes:
        return True
    for axis_key, item_rank in sig.axes:
        if item_rank > ranks.rank(axis_key):
            return False
    return True


# Cache for ``_pre_pass_for_ranks``, keyed by the hashable
# ``Signature.reqs`` tuple plus options.
_RANK_PRE_PASS_CACHE: dict[tuple, EquipmentFlags] = {}

# Bump-selection telemetry (gated by KSP_BUMP_STATS=1; off by default, zero
# cost otherwise).  Each appended record is one trialed candidate:
# (failure_types, axis, new_rank, feasible, reduced_blockers, chosen).  Used
# offline to learn which axes never/rarely help the bumper so the candidate set
# can be pruned by DATA, not guesswork.
_BUMP_STATS_ON: bool = os.environ.get("KSP_BUMP_STATS") == "1"
_BUMP_STATS: list = []


def _enrich_kit_alternates(kit, ctx: RankContext) -> None:
    """Populate ``kit.alternates`` and ``kit.stage_*_alternates`` in place.

    Two sources of alternates:

    1. **Rank-equivalence** (cheap):  if the chosen part has a rank on
       the relevant axis, alternates = every other part at that same
       (axis, rank).  Rank buckets group parts by capability-score, so
       same-rank parts deliver comparable performance — most random
       substitutes will still be viable.

    2. **Capability-flag membership** (cheap):  for presence-only roles
       (RTG, fuel line, RCS, reaction wheel, ladder, aero control) the
       part isn't on a rank axis but has a ``provides`` flag.
       Alternates = all parts with the same flag.

    For per-stage propulsion (engine + tank), alternates are derived
    from each stage's chosen part's rank.

    Caveat: rank-equivalence is necessary but not sufficient for the
    cascading mission-profile case (substituting one part can change a
    downstream stage's optimal pick).  Callers verify the substituted
    kit reproduces feasibility before adopting it; otherwise fall back
    to ``kit`` as-extracted.
    """
    from .parts import PART_DB, MiscEquipment, CapabilityFlag
    from .ranks import RankAxisKey

    by_rank_per_axis = _items_at_rank(ctx)

    def _same_axis_rank_alternates(part_name: str, axis: RankAxisKey) -> frozenset[str]:
        sig = rank_sig_for(part_name, ctx)
        for ax, rk in sig.axes:
            if ax == axis:
                pool = by_rank_per_axis.get(axis, {}).get(rk, ())
                return frozenset(p for p in pool if p != part_name)
        return frozenset()

    def _flag_alternates(flag: CapabilityFlag,
                         exclude: Optional[str] = None) -> frozenset[str]:
        out: set[str] = set()
        for nm, parts in PART_DB.items():
            if exclude is not None and nm == exclude:
                continue
            for p in parts:
                if isinstance(p, MiscEquipment) and flag in p.provides:
                    out.add(nm)
                    break
        return frozenset(out)

    # Map KitUsed field name → (axis, lookup-strategy).  axis is None
    # for parts that live on no rank axis — those use the flag fallback.
    rank_axis_for_field = {
        'capsule': RankAxisKey.CAPSULE,
        'probe_core': RankAxisKey.PROBE_SAS,
        'parachute': RankAxisKey.PARACHUTE,
        'solar': RankAxisKey.SOLAR,
        'solar_retractable': RankAxisKey.SOLAR,
        'monoprop_tank': RankAxisKey.MONOPROP_TANK,
        'stack_decoupler': RankAxisKey.STACK_DECOUPLER,
        'radial_decoupler': RankAxisKey.RADIAL_DECOUPLER,
        'srb': RankAxisKey.SRB,
    }
    flag_for_field = {
        'rtg': CapabilityFlag.RTG,
        'fuel_line': CapabilityFlag.FUEL_LINE,
        'rcs_thruster': CapabilityFlag.RCS,
        'reaction_wheel': CapabilityFlag.REACTION_WHEEL,
        'aero_control': CapabilityFlag.AERO_CONTROL,
        'ladder': CapabilityFlag.LADDER,
    }

    for field_name, axis in rank_axis_for_field.items():
        chosen = getattr(kit, field_name)
        if chosen:
            kit.alternates[field_name] = _same_axis_rank_alternates(chosen, axis)
    for field_name, flag in flag_for_field.items():
        chosen = getattr(kit, field_name)
        if chosen:
            kit.alternates[field_name] = _flag_alternates(flag, exclude=chosen)

    # Per-stage propulsion: alternate set for each stage's engine + tank
    # picks, derived from the part's rank on the relevant axis (engine on
    # LAUNCH_ENGINE or VAC_ENGINE — try both — tank on its fuel-type axis).
    engine_axes = (RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE)
    tank_axes = (RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK,
                 RankAxisKey.XENON_TANK, RankAxisKey.MONOPROP_TANK)
    for eng in kit.stage_engines:
        alts: set[str] = set()
        for ax in engine_axes:
            alts |= _same_axis_rank_alternates(eng, ax)
        kit.stage_engine_alternates.append(frozenset(alts))
    for tank in kit.stage_tanks:
        alts = set()
        for ax in tank_axes:
            alts |= _same_axis_rank_alternates(tank, ax)
        kit.stage_tank_alternates.append(frozenset(alts))


def _random_kit_variant(kit, rng: Random) -> frozenset[str]:
    """Build a per-seed variant of ``kit`` by random-picking among the
    chosen + alternates for every role.  Returns the full set of part
    names.  Caller verifies feasibility before adopting — rank-equivalence
    doesn't guarantee cross-stage cascades survive.
    """
    # ``sorted`` before every rng.choice: a frozenset's iteration order is
    # hash-randomized per process (PYTHONHASHSEED), so ``tuple(alts)`` would
    # make the variant pick — and thus the resulting kit / cheap rules /
    # fill — non-reproducible across solve-check workers.  Sorting pins the
    # candidate order so the seeded rng gives the same pick everywhere.
    def _pick(chosen, alts):
        if not alts:
            return chosen
        return rng.choice([chosen] + sorted(alts))

    out: set[str] = set()
    out.update(kit.stage_equipment)
    out.update(kit.landing_legs)
    out.update(kit.heat_shields)
    out.update(kit.relays)
    for i, eng in enumerate(kit.stage_engines):
        alts = kit.stage_engine_alternates[i] if i < len(kit.stage_engine_alternates) else frozenset()
        out.add(_pick(eng, alts))
    for i, tank in enumerate(kit.stage_tanks):
        alts = kit.stage_tank_alternates[i] if i < len(kit.stage_tank_alternates) else frozenset()
        out.add(_pick(tank, alts))
    for field_name in (
        'capsule', 'probe_core', 'parachute',
        'rtg', 'solar', 'solar_retractable', 'monoprop_tank',
        'rcs_thruster', 'reaction_wheel', 'aero_control', 'ladder',
        'stack_decoupler', 'radial_decoupler', 'fuel_line', 'srb',
        'ion_power',
    ):
        chosen = getattr(kit, field_name)
        if not chosen:
            continue
        out.add(_pick(chosen, kit.alternates.get(field_name, frozenset())))
    return frozenset(out)


def _pre_pass_for_ranks(
    ranks: Signature,
    ctx: RankContext,
    *,
    start_with_clamps: bool,
    progressive_launch_pad: bool,
    launch_pad_caps: tuple[float, ...] | None,
    pad_tier: int = 0,
    precollected_names: frozenset[str] = frozenset(),
    reps_only: Optional[frozenset[str]] = None,
    buildings_in_logic: bool = False,
    home: "BodyName | None" = None,
) -> EquipmentFlags:
    """Build EquipmentFlags from a rank ceiling OR a specific rep set.

    Two modes:

    * **Full rank-admit** (``reps_only=None``): admits every PART_DB item
      whose rank on every applicable axis is ≤ the ceiling on that axis.
      Used for *intrinsic* per-location queries (what's the minimum kit
      this location needs) and for diagnostic exploration.

    * **Reps-only** (``reps_only`` is a frozenset of part names): admits
      ONLY those parts + precollected items.  Used by the chain-walker
      bumper because solvability requires the proof to use only the
      parts the player actually has — i.e., the bumper's recorded reps
      — not the broader set the rank ceiling abstractly admits.  Without
      this, the chain claims feasibility under a kit the AP fill never
      reproduces (only reps land as PROGRESSION; alternates are USEFUL
      and may not be collected in time).

    Precollected items are admitted in both modes.
    """
    cache_key = (
        ranks.reqs, ctx,
        start_with_clamps, progressive_launch_pad, launch_pad_caps,
        pad_tier, precollected_names, reps_only,
        buildings_in_logic, home,
    )
    cached = _RANK_PRE_PASS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if reps_only is not None:
        admitted = set(reps_only) | (precollected_names & PART_DB.keys())
    else:
        admitted = set()
        for item_name in PART_DB:
            if _rank_admits_item(item_name, ranks, ctx):
                admitted.add(item_name)
        admitted |= precollected_names & PART_DB.keys()

    # Curated-building levels ride the same Signature the bumper threads (like
    # pad_tier).  When buildings_in_logic is OFF these are never consulted by
    # ``_pre_pass`` (the flag is off), so the lookup is harmless.
    _building_levels: dict[str, int] = {}
    if buildings_in_logic:
        _building_levels = {
            PROGRESSIVE_VAB_NAME: ranks.counted(PROGRESSIVE_VAB_NAME),
            PROGRESSIVE_ASTRONAUT_COMPLEX_NAME:
                ranks.counted(PROGRESSIVE_ASTRONAUT_COMPLEX_NAME),
        }

    def cf(name: str, _adm=admitted, _pad=pad_tier, _bl=_building_levels) -> int:
        if name == PROGRESSIVE_LAUNCH_PAD_NAME:
            return _pad
        bl = _bl.get(name)
        if bl is not None:
            return bl
        return 1 if name in _adm else 0

    flags = _pre_pass(
        cf,
        start_with_clamps=start_with_clamps,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=launch_pad_caps,
        buildings_in_logic=buildings_in_logic,
        home=home,
    )
    # In rank-space mode the progressive binary gate flags
    # (``has_launch_engine``, ``has_vacuum_engine``, ``has_lfo_fuel``,
    # ``has_srb_fuel``) are derived from progressive item counts in
    # ``_pre_pass`` — but the rank walker never feeds those counts in.
    # Recover the same semantics from the actual parts admitted: a part
    # is "a launch engine" if it produces atmospheric thrust, "a vacuum
    # engine" if it produces vacuum thrust (every engine does, but we
    # keep the symmetry with the progressive path), etc.  Matches what
    # the Phase 2 plan calls for in ``capability._pre_pass``'s simplified
    # form.
    flags.has_launch_engine = any(e.atm_thrust > 0 for e in flags.available_engines)
    flags.has_vacuum_engine = any(e.vac_thrust > 0 for e in flags.available_engines)
    flags.has_lfo_fuel = any(t.fuel_type == "lfo" for t in flags.available_tanks)
    flags.has_srb_fuel = bool(flags.available_srbs)
    _RANK_PRE_PASS_CACHE[cache_key] = flags
    return flags


@dataclass(frozen=True)
class RankBumperResult:
    """Output of ``minimal_ranks_for``.

    ``signature`` is the full capability signature the bumper proved (rank
    ceilings + counted-progressive levels); ``delta`` is the increment over
    the prior signature.  Both fold the old split ``(MinimumRanks ranks,
    dict extras)`` into one :class:`Signature`.  The counted progressives
    (Pad / R&D / PSI) live as :class:`Counted` reqs — Pad is the only one
    the bumper itself touches today (LAUNCH_MASS_EXCEEDED → Pad bump);
    R&D / PSI are injected by the chain orchestrator via tech-tree
    band-funding logic.
    """
    signature: Signature
    delta: Signature
    reps: dict[tuple[RankAxisKey, int], str]
    flags: EquipmentFlags
    profile_dv: float
    reps_collected: frozenset[str] = frozenset()


def _signature_delta(prior: Signature, current: Signature) -> Signature:
    """The increment of ``current`` over ``prior`` as a :class:`Signature`.

    Mirrors the old split delta exactly:
      * **Rank** reqs: the *new ceiling* of every axis whose level grew
        (the old ``ranks_delta`` stored the absolute ceiling, not the
        increment — and it is never read, only recorded).
      * **Counted** reqs: the *increment* (``new - prior``) of every kind
        whose level grew (the old ``extras_delta`` stored the increment;
        the band-funding pass reads it).
    """
    reqs: list[Threshold] = []
    for r in current.rank_reqs:
        if r.level > prior.rank(r.axis):
            reqs.append(Rank(r.axis, r.level))
    for c in current.counted_reqs:
        inc = c.level - prior.counted(c.kind)
        if inc > 0:
            reqs.append(Counted(c.kind, inc))
    return Signature.of(reqs)


def _pick_rank_rep(axis: RankAxisKey, new_rank: int,
                   ctx: RankContext, rng: Random) -> Optional[str]:
    """Pick a random part among the items newly admitted by bumping
    ``axis`` to ``new_rank``.  Returns ``None`` if there are none.

    Random pick — only used when there's no blocker context to score
    against (the cheap path inside trial-eval, where the trial bump's
    rep is itself a temporary).  The main bumper loop uses
    ``_pick_rank_rep_scored`` which trial-evaluates each candidate
    against the mission's current blockers.
    """
    by_rank = _items_at_rank(ctx).get(axis, {})
    candidates = by_rank.get(new_rank, ())
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_rank_rep_scored(
    axis: RankAxisKey, new_rank: int, ctx: RankContext, rng: Random,
    *, ranks: Signature, reps_collected: set, info, diff,
    start_with_clamps: bool, progressive_launch_pad: bool,
    launch_pad_caps, pad_tier: int, precollected_names: frozenset,
    mission_builder,
    buildings_in_logic: bool = False, home: "BodyName | None" = None,
) -> Optional[str]:
    """Trial each candidate rep at ``(axis, new_rank)``: add it to a
    copy of ``reps_collected``, lift co-axis ranks per its rank_sig,
    re-evaluate the mission, pick the candidate with the lowest blocker
    count (then lowest launch_mass).  Replaces the random rep pick that
    was choosing mid-tier engines at rank 5 instead of the high-Isp
    ones the mission actually needs.
    """
    by_rank = _items_at_rank(ctx).get(axis, {})
    candidates = by_rank.get(new_rank, ())
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Cap trial-evaluated candidates for speed; random sample (not
    # top-by-score) preserves the bumper's per-seed rep variance.
    cands = list(candidates)
    if len(cands) > 6:
        cands = rng.sample(cands, 6)
    scored: list[tuple[int, float, int, float, str]] = []
    base_ranks = ranks.with_rank(axis, new_rank)
    for cand in sorted(cands):
        trial_reps = set(reps_collected); trial_reps.add(cand)
        trial_ranks = base_ranks
        sig = rank_sig_for(cand, ctx)
        for co_axis, co_rank in sig.axes:
            if co_axis == axis:
                continue
            if trial_ranks.rank(co_axis) < co_rank:
                trial_ranks = trial_ranks.with_rank(co_axis, co_rank)
        trial_flags = _pre_pass_for_ranks(
            trial_ranks, ctx,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=launch_pad_caps,
            pad_tier=pad_tier,
            precollected_names=precollected_names,
            reps_only=frozenset(trial_reps),
            buildings_in_logic=buildings_in_logic, home=home,
        )
        trial_result = _evaluate(trial_flags, info, diff, mission_builder,
                                 run_parallel=False)
        feasibility = 0 if trial_result.feasible else 1
        mass = trial_result.launch_mass or float("inf")
        # Blocker reduction is the real "closer to feasible" signal;
        # mass-only sorts pick weak lightweight engines for infeasible
        # trials.  Mirror the same fix as in _pick_rank_bump_scored.
        scored.append((feasibility, len(trial_result.blocking), mass,
                       rng.random(), cand))
    scored.sort()
    return scored[0][-1]


def _rank_axis_at_cap(axis: RankAxisKey, ranks: Signature) -> bool:
    """Return True if ``axis`` ceiling already equals the axis's effective
    max rank (min(distinct scores, cap)) — no more bumps possible."""
    max_buckets = max_rank_for(axis)
    current = ranks.rank(axis)
    return current >= max_buckets


def _axes_for_stage_diag(stage_diag) -> tuple[RankAxisKey, ...]:
    """Map ``StageDiagnostic.failure`` to the rank axes that resolve it.

    Ports the legacy ``_bump_candidates_for_stage_diag`` logic into the
    rank-axis world — same atmosphere-aware engine selection, same fuel-
    type tank dispatch, same broad fallback for performance failures.
    """
    from .capability_reasons import StageFailure
    if stage_diag is None:
        return ()
    f = stage_diag.failure
    in_atm = stage_diag.in_atmosphere
    # Filter failures.
    if f == StageFailure.HEAT_SHIELD_TOO_SMALL:
        return (RankAxisKey.HEAT_SHIELD,)
    if f in (StageFailure.NO_ENGINES_AFTER_FILTER,
              StageFailure.REQUIRE_GIMBAL_NONE,
              StageFailure.REQUIRE_THROTTLE_NONE):
        # Need more engines pass the filters.  In atmosphere → launch
        # tier (+ SRB).  In vacuum → vacuum tier.
        if in_atm:
            return (RankAxisKey.LAUNCH_ENGINE, RankAxisKey.SRB)
        return (RankAxisKey.VAC_ENGINE,)
    # Tank-side failures: target the fuel-type tank axis.
    if f == StageFailure.NO_TANK_FOR_FUEL_TYPE:
        out: list[RankAxisKey] = []
        for ft in stage_diag.engine_fuel_types_attempted:
            if ft in ("lfo", "lf"):
                out.append(RankAxisKey.LFO_TANK)  # LFO covers LF via fuel-drop
            elif ft == "xenon":
                out.append(RankAxisKey.XENON_TANK)
            elif ft == "monoprop":
                out.append(RankAxisKey.MONOPROP_TANK)
        return tuple(out)
    if f == StageFailure.ENGINE_TOO_BIG_FOR_TANK:
        return (RankAxisKey.LFO_TANK,)
    if f == StageFailure.MASS_CAP_EXCEEDED:
        # Handled outside rank axes via the Pad extras bump.
        return ()
    # Performance failures (DV_SHORT / TWR_SHORT / DRY_MASS_KILLS_RATIO):
    # a too-weak rocket is fixed by EITHER more propulsion/staging OR less
    # payload mass (a lighter command module).  A rank bump admits *better*
    # (e.g. lighter-dry) parts, so the propulsion/staging/command lever set can
    # plausibly help and the scored picker ranks them.  Equipment is NOT in this
    # set — it can never close a stage and was ~44% of all bumper trials at ~94%
    # no-op (see the NO_VIABLE_STAGE table above for the full rationale).
    return _RANK_BUMP_TABLE[BlockingReason.NO_VIABLE_STAGE]


def _pick_rank_bump(blocking, ranks: Signature, rng: Random) -> Optional[RankAxisKey]:
    """Choose an axis to bump based on the current blocking reasons.

    Walks ``RANK_PRIORITY_GROUPS`` in order; the first group that
    overlaps the candidate axes is the random pick pool.  Axes already
    at their max bucket are excluded.

    For ``NO_VIABLE_STAGE`` blockers carrying a ``stage_diag``, uses
    ``_axes_for_stage_diag`` to target the specific axis (e.g. tank
    fuel-type, engine class) rather than the 15-axis catchall.
    """
    candidates: set[RankAxisKey] = set()
    for b in blocking:
        if b.reason == BlockingReason.NO_VIABLE_STAGE and b.stage_diag is not None:
            for axis in _axes_for_stage_diag(b.stage_diag):
                if not _rank_axis_at_cap(axis, ranks):
                    candidates.add(axis)
            continue
        for axis in _RANK_BUMP_TABLE.get(b.reason, ()):
            if _rank_axis_at_cap(axis, ranks):
                continue
            candidates.add(axis)
    if not candidates:
        return None
    for group in RANK_PRIORITY_GROUPS:
        overlap = candidates & group
        if overlap:
            return rng.choice(sorted(overlap, key=lambda a: a.value))
    return rng.choice(sorted(candidates, key=lambda a: a.value))


def minimal_ranks_for(
    location_name: str,
    prior: Signature,
    ctx: RankContext,
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    rng: Random,
    mission_builder: MissionBuilder,
    prior_reps: frozenset[str] = frozenset(),
    precollected_names: frozenset[str] = frozenset(),
    max_iterations: int = 500,
    reps_only_mode: bool = True,
    buildings_in_logic: bool = False,
) -> Optional[RankBumperResult]:
    """Rank-space sphere walker (Phase 1 scaffold).

    Greedy bumper: starting from ``prior`` (a :class:`Signature` carrying
    both rank ceilings and counted-progressive levels), repeatedly bump an
    axis ceiling by 1 until the mission becomes feasible.  Each bump records
    a designated rep (a random newly-admitted part) in
    ``RankBumperResult.reps``.

    Returns ``None`` if the location isn't capability-gated (tech-tree
    biome / starting inventory) OR if no kit reaches it within
    ``max_iterations`` bumps.
    """
    info = _parse_location(location_name)
    if info is None:
        return None
    diff = DIFFICULTY_PROFILES[difficulty]
    home = mission_builder.home
    sig = prior
    reps: dict[tuple[RankAxisKey, int], str] = {}
    reps_collected: set[str] = set(prior_reps)
    prev_blocker_count = -1
    stuck_iters = 0
    pad_cap_count = (
        len(mission_builder.launch_pad_caps) - 1
        if mission_builder.launch_pad_caps else 0
    )
    for _iter in range(max_iterations):
        flags = _pre_pass_for_ranks(
            sig, ctx,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
            precollected_names=precollected_names,
            reps_only=frozenset(reps_collected) if reps_only_mode else None,
            buildings_in_logic=buildings_in_logic, home=home,
        )
        result = _evaluate(flags, info, diff, mission_builder)
        if result.feasible:
            if (reps_only_mode and os.environ.get("KSP_MINIMIZE_KIT", "1") == "1"
                    and (reps_collected - set(prior_reps))):
                # Minimization pass.  The greedy from-empty bumper raises an
                # axis (e.g. vac_engine rank) when cheaper enablers (tanks /
                # staging) weren't built up yet, and never backtracks — leaving
                # an inflated kit (orbit "needs" vac:5 when vac:1 + fuel works).
                # Drop delta reps the mission no longer needs, then re-derive
                # ranks from the survivors so the sphere kit is truly minimal.
                _pp = dict(
                    start_with_clamps=start_with_clamps,
                    progressive_launch_pad=progressive_launch_pad,
                    launch_pad_caps=mission_builder.launch_pad_caps,
                    pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                    precollected_names=precollected_names,
                    buildings_in_logic=buildings_in_logic, home=home,
                )
                kept = set(reps_collected)
                _changed = True
                while _changed:
                    _changed = False
                    for _rep in sorted(kept - set(prior_reps)):
                        _trial = kept - {_rep}
                        _tf = _pre_pass_for_ranks(
                            sig, ctx, reps_only=frozenset(_trial), **_pp)
                        if _evaluate(_tf, info, diff, mission_builder).feasible:
                            kept = _trial
                            _changed = True
                if kept != reps_collected:
                    reps_collected = kept
                    # Re-derive the rank ceiling from prior ranks + surviving
                    # reps; carry the current counted (extras incl. pad) levels
                    # — minimization only trims rank reps, never the pad.
                    _new = Signature.of((*prior.rank_reqs, *sig.counted_reqs))
                    for _rep in reps_collected:
                        for _ax, _rk in rank_sig_for(_rep, ctx).axes:
                            if _rk > _new.rank(_ax):
                                _new = _new.with_rank(_ax, _rk)
                    sig = _new
                    reps = {k: v for k, v in reps.items() if v in reps_collected}
                    flags = _pre_pass_for_ranks(
                        sig, ctx, reps_only=frozenset(reps_collected), **_pp)
            return RankBumperResult(
                signature=sig,
                delta=_signature_delta(prior, sig),
                reps=reps,
                reps_collected=frozenset(
                    reps_collected | _contract_payload_rep_names(info, flags)),
                flags=flags,
                profile_dv=result.launch_mass,
            )
        # Mass-cap blocker: bump Pad outside the rank model.  This is the
        # one extra counted-progressive the bumper itself touches —
        # R&D / PSI are injected by the chain orchestrator via
        # tech-tree band funding.
        mass_block = any(
            b.reason == BlockingReason.LAUNCH_MASS_EXCEEDED
            for b in result.blocking
        )
        if mass_block and progressive_launch_pad:
            cur_pad = sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME)
            if cur_pad < pad_cap_count:
                sig = sig.with_counted(PROGRESSIVE_LAUNCH_PAD_NAME, cur_pad + 1)
                continue
        # Curated-building blockers (buildings_in_logic): bump the building
        # Counted level outside the rank model, mirroring the Pad mass-cap
        # bump above.  VESSEL_MASS_EXCEEDED -> VAB level, CANNOT_EVA ->
        # Astronaut Complex level.  Each building's max level comes from its
        # pooled copy count.
        if buildings_in_logic:
            building_block = False
            for kind, cap, reason in (
                (PROGRESSIVE_VAB_NAME, PROGRESSIVE_VAB_COUNT,
                 BlockingReason.VESSEL_MASS_EXCEEDED),
                (PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
                 PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT,
                 BlockingReason.CANNOT_EVA),
            ):
                if any(b.reason == reason for b in result.blocking):
                    cur = sig.counted(kind)
                    if cur < cap:
                        sig = sig.with_counted(kind, cur + 1)
                        building_block = True
            if building_block:
                continue
        # Discrete-unlock blockers: blockers whose resolution is a
        # specific named part not on any rank axis.  Map each blocker
        # to the ``CapabilityFlag`` it needs; resolve to part names by
        # scanning PART_DB for any MiscEquipment that provides the flag.
        # Replaces the legacy ``("rtg",)`` / ``("telescopicLadder", …)``
        # hand-curated part lists — adding a modded RTG or ladder part
        # now works automatically as long as it provides the right flag.
        discrete_unlocks_for: dict[BlockingReason, CapabilityFlag] = {
            BlockingReason.INSUFFICIENT_POWER_NEEDS_RTG: CapabilityFlag.RTG,
            BlockingReason.NO_LADDER: CapabilityFlag.LADDER,
        }
        added_discrete = False
        for b in result.blocking:
            needed_flag = discrete_unlocks_for.get(b.reason)
            if needed_flag is None:
                continue
            for item_name, parts in PART_DB.items():
                for p in parts:
                    if isinstance(p, MiscEquipment) and needed_flag in p.provides:
                        if item_name not in reps_collected:
                            reps_collected.add(item_name)
                            added_discrete = True
                        break
        if added_discrete:
            continue

        # Reactive constraint-driven fix: ENGINE_TOO_BIG_FOR_TANK names the
        # minimum tank size that mounts the stuck engine.  Pick a random tank
        # of the right fuel type at or above that size — every such tank is a
        # logically-valid fix — instead of bumping the tank rank and hoping the
        # rep happens to be big enough (the old path that drove rescues).
        if os.environ.get("KSP_REACTIVE_TANK", "1") == "1":
            reactive_added = False
            for b in result.blocking:
                sd = getattr(b, "stage_diag", None)
                if (sd is None
                        or sd.failure != StageFailure.ENGINE_TOO_BIG_FOR_TANK
                        or sd.min_tank_size_needed <= 0):
                    continue
                want_ft = {("lfo" if ft in ("lfo", "lf") else ft)
                           for ft in sd.engine_fuel_types_attempted} or {"lfo"}
                valid = [
                    name for name, parts in PART_DB.items()
                    if name not in reps_collected
                    for p in parts
                    if isinstance(p, FuelTank) and p.fuel_type in want_ft
                    and PartRole.SPINE in p.roles
                    and p.size_class >= sd.min_tank_size_needed
                ]
                if not valid:
                    continue
                pick = rng.choice(sorted(valid))
                reps_collected.add(pick)
                for _ax, _rk in rank_sig_for(pick, ctx).axes:
                    if (_ax, _rk) not in reps:
                        reps[(_ax, _rk)] = pick
                    if _rk > sig.rank(_ax):
                        sig = sig.with_rank(_ax, _rk)
                reactive_added = True
                break
            if reactive_added:
                continue

        # Reactive constraint-driven fix #2: REQUIRE_THROTTLE_NONE /
        # REQUIRE_GIMBAL_NONE name an engine *property* the stage needs that
        # no available engine has.  Throttle/gimbal aren't rank axes (they're
        # engine flags), so bumping the engine rank and hoping the picked rep
        # happens to be throttleable/gimballed is a rescue-driving gamble.
        # Instead pick a random engine that actually has the property, burns a
        # fuel type we can already fund, and produces thrust in the stage's
        # environment — every such engine is a logically-valid fix.
        if os.environ.get("KSP_REACTIVE_ENGINE", "1") == "1":
            reactive_added = False
            fundable = {ft for ft, tks in (flags.tanks_by_fuel_type or {}).items()
                        if tks}
            for b in result.blocking:
                sd = getattr(b, "stage_diag", None)
                if sd is None:
                    continue
                if sd.failure == StageFailure.REQUIRE_THROTTLE_NONE:
                    prop = "throttleable"
                elif sd.failure == StageFailure.REQUIRE_GIMBAL_NONE:
                    prop = "has_gimbal"
                else:
                    continue
                in_atm = sd.in_atmosphere

                def _engine_ok(p, *, require_fundable: bool) -> bool:
                    if not isinstance(p, Engine) or not getattr(p, prop):
                        return False
                    if p.fuel_type == "xenon":  # ion is out of logic
                        return False
                    if (p.atm_thrust if in_atm else p.vac_thrust) <= 0:
                        return False
                    return (p.fuel_type in fundable) if require_fundable else True

                # Prefer an engine we can fuel right now; fall back to any
                # engine with the property (the bump loop funds its tank via
                # the NO_TANK_FOR_FUEL_TYPE -> tank-axis path).
                valid = [
                    name for name, parts in PART_DB.items()
                    if name not in reps_collected
                    for p in parts if _engine_ok(p, require_fundable=True)
                ]
                if not valid:
                    valid = [
                        name for name, parts in PART_DB.items()
                        if name not in reps_collected
                        for p in parts if _engine_ok(p, require_fundable=False)
                    ]
                if not valid:
                    continue
                pick = rng.choice(sorted(valid))
                reps_collected.add(pick)
                for _ax, _rk in rank_sig_for(pick, ctx).axes:
                    if (_ax, _rk) not in reps:
                        reps[(_ax, _rk)] = pick
                    if _rk > sig.rank(_ax):
                        sig = sig.with_rank(_ax, _rk)
                reactive_added = True
                break
            if reactive_added:
                continue

        cur_count = len(result.blocking)
        if prev_blocker_count >= 0 and cur_count >= prev_blocker_count:
            stuck_iters += 1
        else:
            stuck_iters = 0
        prev_blocker_count = cur_count

        # Pick a bump.  Scored selection (mass-min objective +
        # stage_diag candidate narrowing) is the primary mechanism —
        # ports the legacy _pick_bump intelligence to the rank system.
        axis = _pick_rank_bump_scored(
            result.blocking, sig, ctx, rng,
            difficulty=difficulty,
            progressive_launch_pad=progressive_launch_pad,
            start_with_clamps=start_with_clamps,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
            precollected_names=precollected_names,
            mission_builder=mission_builder,
            info=info,
            diff=diff,
            reps_collected=reps_collected,
            reps_only_mode=reps_only_mode,
            buildings_in_logic=buildings_in_logic, home=home,
        )
        if axis is None:
            # Fall back to the catchall candidate set if stage_diag-targeted
            # axes are exhausted (e.g. non-NO_VIABLE_STAGE blockers only).
            axis = _pick_rank_bump(result.blocking, sig, rng)
        if axis is None:
            # The greedy bumper could not map any remaining blocker to an axis
            # it can still raise — a structural smell (the rep at some capped
            # axis can't satisfy the mission).  Make it loud: every rescue is a
            # mission the rank ladder couldn't build cleanly.
            diag_lines = []
            for b in result.blocking:
                sd = getattr(b, "stage_diag", None)
                if sd is not None:
                    diag_lines.append(
                        f"{b.reason.name}/{sd.failure.value} "
                        f"dv={sd.best_dv_achieved:.0f}/{sd.required_dv:.0f} "
                        f"twr={sd.best_twr_achieved:.2f}/{sd.twr_floor:.2f} "
                        f"payload={sd.payload_mass:.1f}t cap={sd.mass_cap:.0f}t"
                    )
                else:
                    diag_lines.append(b.reason.name)
            partial = getattr(result, "partial_stages", [])
            stage_summary = " | ".join(
                f"{s.engine_count}x{s.engine_name}+{sum(n for n, _ in s.tank_manifest)}tk "
                f"dv={s.delta_v:.0f} wet={s.stage_mass_wet:.1f}t"
                for s in partial
            )
            logging.warning(
                "KSP1 sphere-bumper RESCUE (bailed to full-admit kit): "
                "%s/%s crewed=%s\n  blockers: %s\n  partial rocket (launch->top): %s",
                info.body, info.mission_type, info.crewed,
                "; ".join(diag_lines), stage_summary or "(none built)",
            )
            # Capability-guided rescue: when greedy ran out of axis
            # bumps, run a single full-admit eval at MAX ranks.  If
            # feasible, ``ProfileResult.kit_used`` is the complete
            # structured set of parts the optimizer relied on —
            # including presence-only representatives (radial
            # decoupler for staging_tier=2, fuelLine for asparagus,
            # etc.) that aren't directly used in any stage but enable
            # the dry-mass factors the optimization assumed.
            #
            # The chain's ranks must be lifted to cover the rank_sigs
            # of every kit part — otherwise item_rule rejects them
            # from sphere placement.  We expand ``ranks`` to the union
            # of (current ceiling, max kit-part rank) per axis, then
            # verify the lifted-ranks + union-reps combination
            # actually reproduces feasibility.
            max_ranks_for_rescue = Signature.of(
                Rank(a, max_rank_for(a)) for a in RankAxisKey
            )
            if buildings_in_logic:
                # Full-admit rescue: max the building levels too, else a
                # vessel-mass / EVA gate would falsely fail the rescue probe.
                max_ranks_for_rescue = (max_ranks_for_rescue
                    .with_counted(PROGRESSIVE_VAB_NAME, PROGRESSIVE_VAB_COUNT)
                    .with_counted(PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
                                  PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT))
            rescue_flags = _pre_pass_for_ranks(
                max_ranks_for_rescue, ctx,
                start_with_clamps=start_with_clamps,
                progressive_launch_pad=progressive_launch_pad,
                launch_pad_caps=mission_builder.launch_pad_caps,
                pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                precollected_names=precollected_names,
                reps_only=None,
                buildings_in_logic=buildings_in_logic, home=home,
            )
            rescue_result = _evaluate(rescue_flags, info, diff, mission_builder)
            rescue_kit = build_kit_for_result(rescue_flags, rescue_result)
            if rescue_kit is not None:
                kit = rescue_kit
                _enrich_kit_alternates(kit, ctx)
                # Try a per-seed random variant of the kit first
                # (alternates derived from rank-equivalence + provides-flag
                # membership).  If the variant verifies, use it — that
                # preserves per-seed variance even on the rescue path.
                # If verification fails (cascade broke), fall back to the
                # deterministic ``kit.all_parts()`` which is guaranteed
                # to reproduce capability's feasibility claim.
                variant_parts = _random_kit_variant(kit, rng)
                lifted_ranks = sig
                for u in variant_parts:
                    rsig = rank_sig_for(u, ctx)
                    for ax, rk in rsig.axes:
                        if rk > lifted_ranks.rank(ax):
                            lifted_ranks = lifted_ranks.with_rank(ax, rk)
                union_reps = reps_collected | variant_parts
                verify_flags = _pre_pass_for_ranks(
                    lifted_ranks, ctx,
                    start_with_clamps=start_with_clamps,
                    progressive_launch_pad=progressive_launch_pad,
                    launch_pad_caps=mission_builder.launch_pad_caps,
                    pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                    precollected_names=precollected_names,
                    reps_only=frozenset(union_reps),
                    buildings_in_logic=buildings_in_logic, home=home,
                )
                verify_result = _evaluate(verify_flags, info, diff, mission_builder)
                kit_parts = variant_parts
                if not verify_result.feasible:
                    # Variant broke a cascade.  Fall back to the
                    # deterministic optimal kit and re-verify.
                    kit_parts = kit.all_parts()
                    lifted_ranks = sig
                    for u in kit_parts:
                        rsig = rank_sig_for(u, ctx)
                        for ax, rk in rsig.axes:
                            if rk > lifted_ranks.rank(ax):
                                lifted_ranks = lifted_ranks.with_rank(ax, rk)
                    union_reps = reps_collected | kit_parts
                    verify_flags = _pre_pass_for_ranks(
                        lifted_ranks, ctx,
                        start_with_clamps=start_with_clamps,
                        progressive_launch_pad=progressive_launch_pad,
                        launch_pad_caps=mission_builder.launch_pad_caps,
                        pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                        precollected_names=precollected_names,
                        reps_only=frozenset(union_reps),
                        buildings_in_logic=buildings_in_logic, home=home,
                    )
                    verify_result = _evaluate(verify_flags, info, diff, mission_builder)
                if verify_result.feasible:
                    for u in kit_parts:
                        if u not in reps_collected:
                            reps_collected.add(u)
                            rsig = rank_sig_for(u, ctx)
                            for ax, rk in rsig.axes:
                                if (ax, rk) not in reps:
                                    reps[(ax, rk)] = u
                    # Minimize the rescue kit.  The max-flags optimizer grabs
                    # the BEST (highest-rank) parts it can — relay:8, srb:8,
                    # solar:7 — even when the mission needs far less.  Drop every
                    # rep the mission stays feasible without and re-derive the
                    # lifted ranks, so a rescue contributes a minimal kit instead
                    # of dumping the whole maxed set (+60 reps) onto this sphere.
                    if reps_only_mode and os.environ.get("KSP_MINIMIZE_KIT", "1") == "1":
                        _rpp = dict(
                            start_with_clamps=start_with_clamps,
                            progressive_launch_pad=progressive_launch_pad,
                            launch_pad_caps=mission_builder.launch_pad_caps,
                            pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                            precollected_names=precollected_names,
                            buildings_in_logic=buildings_in_logic, home=home,
                        )
                        kept = set(reps_collected)
                        _changed = True
                        while _changed:
                            _changed = False
                            for _rep in sorted(kept - set(prior_reps)):
                                _trial = kept - {_rep}
                                _tf = _pre_pass_for_ranks(
                                    lifted_ranks, ctx,
                                    reps_only=frozenset(_trial), **_rpp)
                                if _evaluate(_tf, info, diff,
                                             mission_builder).feasible:
                                    kept = _trial
                                    _changed = True
                        if kept != reps_collected:
                            reps_collected = kept
                            # Re-derive the rank ceiling from prior ranks +
                            # survivors; carry the current counted (extras incl.
                            # pad) levels so the signature keeps them.
                            lifted_ranks = Signature.of(
                                (*prior.rank_reqs, *sig.counted_reqs))
                            for _rep in reps_collected:
                                for _ax, _rk in rank_sig_for(_rep, ctx).axes:
                                    if _rk > lifted_ranks.rank(_ax):
                                        lifted_ranks = lifted_ranks.with_rank(_ax, _rk)
                            reps = {k: v for k, v in reps.items()
                                    if v in reps_collected}
                            verify_flags = _pre_pass_for_ranks(
                                lifted_ranks, ctx,
                                reps_only=frozenset(reps_collected), **_rpp)
                            verify_result = _evaluate(
                                verify_flags, info, diff, mission_builder)
                    return RankBumperResult(
                        signature=lifted_ranks,
                        delta=_signature_delta(prior, lifted_ranks),
                        reps=reps,
                        reps_collected=frozenset(
                            reps_collected
                            | _contract_payload_rep_names(info, verify_flags)),
                        flags=verify_flags,
                        profile_dv=verify_result.launch_mass,
                    )
            # Swap-fallback: greedy hill-climb on rep alternatives.
            cur_blockers = len(result.blocking)
            cur_mass = result.launch_mass or float("inf")
            for _swap_iter in range(8):
                best_swap: Optional[tuple[int, float, RankAxisKey, int, str, str]] = None
                for (swap_axis, swap_rank), current_rep in list(reps.items()):
                    alternatives = _items_at_rank(ctx).get(swap_axis, {}).get(swap_rank, ())
                    for alt in alternatives:
                        if alt == current_rep:
                            continue
                        new_reps = (reps_collected - {current_rep}) | {alt}
                        swap_flags = _pre_pass_for_ranks(
                            sig, ctx,
                            start_with_clamps=start_with_clamps,
                            progressive_launch_pad=progressive_launch_pad,
                            launch_pad_caps=mission_builder.launch_pad_caps,
                            pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                            precollected_names=precollected_names,
                            reps_only=frozenset(new_reps),
                            buildings_in_logic=buildings_in_logic, home=home,
                        )
                        swap_result = _evaluate(swap_flags, info, diff, mission_builder)
                        nb = len(swap_result.blocking)
                        m = swap_result.launch_mass or float("inf")
                        score = (nb, m)
                        if (nb < cur_blockers or
                            (nb == cur_blockers and m < cur_mass)):
                            if best_swap is None or (nb, m) < (best_swap[0], best_swap[1]):
                                best_swap = (nb, m, swap_axis, swap_rank, current_rep, alt)
                if best_swap is None:
                    break
                nb, m, sa, sr, cur, alt = best_swap
                reps_collected.discard(cur)
                reps_collected.add(alt)
                reps[(sa, sr)] = alt
                cur_blockers = nb
                cur_mass = m
                if nb == 0:
                    # Feasible — return
                    final_flags = _pre_pass_for_ranks(
                        sig, ctx,
                        start_with_clamps=start_with_clamps,
                        progressive_launch_pad=progressive_launch_pad,
                        launch_pad_caps=mission_builder.launch_pad_caps,
                        pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                        precollected_names=precollected_names,
                        reps_only=frozenset(reps_collected),
                        buildings_in_logic=buildings_in_logic, home=home,
                    )
                    return RankBumperResult(
                        signature=sig,
                        delta=_signature_delta(prior, sig),
                        reps=reps,
                        reps_collected=frozenset(
                            reps_collected
                            | _contract_payload_rep_names(info, final_flags)),
                        flags=final_flags, profile_dv=m,
                    )
            return None
        new_rank = sig.rank(axis) + 1
        # Pick the part at this (axis, rank) that actually clears the most
        # blockers, not a random one — a random low-rank pick is often too
        # weak, forcing the bumper to over-raise the rank (orbit "needs" vac:5,
        # Pol "needs" launch:5 when a good rank-2 part flies it).
        rep_name = _pick_rank_rep_scored(
            axis, new_rank, ctx, rng,
            ranks=sig, reps_collected=reps_collected, info=info, diff=diff,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=sig.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
            precollected_names=precollected_names,
            mission_builder=mission_builder,
            buildings_in_logic=buildings_in_logic, home=home,
        )
        if rep_name is not None:
            reps[(axis, new_rank)] = rep_name
            reps_collected.add(rep_name)
        sig = sig.with_rank(axis, new_rank)
        # Co-axis lift.
        if rep_name is not None:
            rsig = rank_sig_for(rep_name, ctx)
            for co_axis, co_rank in rsig.axes:
                if co_axis == axis:
                    continue
                if co_rank > sig.rank(co_axis):
                    sig = sig.with_rank(co_axis, co_rank)
    return None


# Payload-mass-reducing rank axes: when the bumper is stuck on a
# stage-level mass blocker, expanding to these axes can shrink the
# *upstream* stage's payload via lighter terminal equipment / more
# efficient upper-stage propulsion.  Mirrors legacy
# ``_PAYLOAD_MASS_REDUCING_CHAINS`` translated to rank axes.
_PAYLOAD_MASS_REDUCING_AXES: tuple[RankAxisKey, ...] = (
    RankAxisKey.VAC_ENGINE,    # higher rank = more efficient upper-stage
    RankAxisKey.LAUNCH_ENGINE, # higher rank = better atm Isp
    RankAxisKey.SOLAR,         # higher rank = lighter panels at distance
    RankAxisKey.SAS,           # higher rank = lighter reaction wheel
    RankAxisKey.CAPSULE,       # higher rank = built-in wheels + monoprop
    RankAxisKey.PROBE_SAS,     # higher rank = lighter probe with wheels
    RankAxisKey.HEAT_SHIELD,   # higher rank = larger shield, fewer needed
    RankAxisKey.PARACHUTE,     # higher rank = better drag, fewer needed
    RankAxisKey.LANDING_LEG,   # higher rank = stronger, fewer needed
)

# Stage-failure modes that benefit from a payload-mass audit when the
# bumper's primary candidate set is exhausted.
_MASS_RELATED_FAILURES = frozenset({
    "dv_short", "twr_short", "dry_mass_kills_ratio", "mass_cap_exceeded",
})

# Empirically-dominant bump axes (KSP_BUMP_STATS analysis over varied
# goals/homes incl. hard aliens): these — engines, fuel tanks, staging
# decouplers, SRB, heat shield — account for ~99% of chosen bumps on
# performance failures.  The remaining axes (payload-reducer support gear:
# capsule / probe / solar / SAS / parachute / landing leg, and relay) are a
# real but <1% tail.  ``_pick_rank_bump_scored`` trials this tier FIRST and
# only expands to the tail when tier-1 makes no progress — keeping the tail
# reachable (robust to physics changes) instead of pruning it.
_TIER1_BUMP_AXES: frozenset[RankAxisKey] = frozenset({
    RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE, RankAxisKey.SRB,
    RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK, RankAxisKey.XENON_TANK,
    RankAxisKey.MONOPROP_TANK, RankAxisKey.STACK_DECOUPLER,
    RankAxisKey.RADIAL_DECOUPLER, RankAxisKey.HEAT_SHIELD,
})


def _pick_rank_bump_scored(blocking, ranks: Signature, ctx: RankContext,
                           rng: Random, *,
                           difficulty: str,
                           progressive_launch_pad: bool,
                           start_with_clamps: bool,
                           launch_pad_caps,
                           pad_tier: int,
                           precollected_names: frozenset[str],
                           mission_builder, info, diff,
                           reps_collected: Optional[set[str]] = None,
                           reps_only_mode: bool = True,
                           buildings_in_logic: bool = False,
                           home: "BodyName | None" = None,
                           ) -> Optional[RankAxisKey]:
    """Trial-bump every candidate axis; pick the one with the best
    ``(feasibility, launch_mass, n_blocking)`` score.

    Candidate source: ``stage_diag.failure`` for ``NO_VIABLE_STAGE`` (via
    ``_axes_for_stage_diag``), ``_RANK_BUMP_TABLE`` for other blockers.
    """
    candidates: set[RankAxisKey] = set()
    for b in blocking:
        if b.reason == BlockingReason.NO_VIABLE_STAGE and b.stage_diag is not None:
            axes = _axes_for_stage_diag(b.stage_diag)
        else:
            axes = _RANK_BUMP_TABLE.get(b.reason, ())
        for axis in axes:
            if not _rank_axis_at_cap(axis, ranks):
                candidates.add(axis)
    if not candidates:
        return None
    # Deterministic, priority-group-aware candidate order: ``set`` iteration is
    # hash-randomized per process under the default PYTHONHASHSEED, which made
    # different solve-check workers consume ``rng`` in different orders → diverge.
    def _cand_sort_key(a: RankAxisKey) -> tuple[int, str]:
        for i, g in enumerate(RANK_PRIORITY_GROUPS):
            if a in g:
                return (i, a.value)
        return (len(RANK_PRIORITY_GROUPS), a.value)

    def _score(cands: list[RankAxisKey]) -> list:
        out: list[tuple[int, float, int, int, float, RankAxisKey]] = []
        for cand in sorted(cands, key=_cand_sort_key):
            new_rank = ranks.rank(cand) + 1
            trial_ranks = ranks.with_rank(cand, new_rank)
            trial_reps_set = None
            if reps_only_mode and reps_collected is not None:
                trial_reps = set(reps_collected)
                sample_rep = _pick_rank_rep(cand, new_rank, ctx, rng)
                if sample_rep is not None:
                    trial_reps.add(sample_rep)
                    _sig = rank_sig_for(sample_rep, ctx)
                    for _co_axis, _co_rank in _sig.axes:
                        if _co_axis == cand:
                            continue
                        if trial_ranks.rank(_co_axis) < _co_rank:
                            trial_ranks = trial_ranks.with_rank(_co_axis, _co_rank)
                trial_reps_set = frozenset(trial_reps)
            trial_flags = _pre_pass_for_ranks(
                trial_ranks, ctx,
                start_with_clamps=start_with_clamps,
                progressive_launch_pad=progressive_launch_pad,
                launch_pad_caps=launch_pad_caps,
                pad_tier=pad_tier,
                precollected_names=precollected_names,
                reps_only=trial_reps_set,
                buildings_in_logic=buildings_in_logic, home=home,
            )
            trial_result = _evaluate(trial_flags, info, diff, mission_builder,
                                     run_parallel=False)
            feasibility_rank = 0 if trial_result.feasible else 1
            mass = trial_result.launch_mass or float("inf")
            group_idx = len(RANK_PRIORITY_GROUPS)
            for i, g in enumerate(RANK_PRIORITY_GROUPS):
                if cand in g:
                    group_idx = i
                    break
            # Sort order: feasibility > blocker reduction > mass (n_blockers
            # before mass: a tiny weak engine "looks lighter" on infeasible
            # trials and would get mis-picked; blocker-reduction is the real
            # closer-to-feasible signal).
            out.append((
                feasibility_rank, len(trial_result.blocking), mass,
                group_idx, rng.random(), cand,
            ))
        return out

    # Tiered: trial the empirically-dominant axes (``_TIER1_BUMP_AXES``) first;
    # expand to the rare payload-reducer/relay tail ONLY when tier-1 makes no
    # progress (no feasible candidate AND none cuts the blocker count).  Skips
    # the long tail in the common case while keeping it reachable.
    _cur_nblock = len(blocking)
    tier1 = [c for c in candidates if c in _TIER1_BUMP_AXES]
    tier2 = [c for c in candidates if c not in _TIER1_BUMP_AXES]
    if not tier1:
        tier1, tier2 = list(candidates), []
    scored = _score(tier1)
    scored.sort()
    _progress = bool(scored) and (scored[0][0] == 0 or scored[0][1] < _cur_nblock)
    if not _progress and tier2:
        scored.extend(_score(tier2))
        scored.sort()

    if _BUMP_STATS_ON and scored:
        _chosen = scored[0][-1]
        _cur_nblock = len(blocking)
        _failures = tuple(sorted({
            b.stage_diag.failure.value for b in blocking
            if getattr(b, "stage_diag", None) is not None
        }))
        for _feas, _nblk, _mass, _gidx, _rnd, _cand in scored:
            _BUMP_STATS.append((
                _failures, _cand.value, ranks.rank(_cand) + 1,
                _feas == 0, _nblk < _cur_nblock, _cand is _chosen,
            ))

    return scored[0][-1]


# ---------------------------------------------------------------------------
# Sphere ladder construction
# ---------------------------------------------------------------------------


def _goal_dv(name: str, mission_builder: MissionBuilder) -> float:
    """Cheapest profile delta-v for a goal location, used for sphere
    ordering and 'hardest-goal' selection."""
    info = _parse_location(name)
    if info is None:
        return 0.0
    profiles = mission_builder.profiles_for(info.body, info.mission_type)
    if not profiles:
        return 0.0
    if info.spec is not None:
        # Apply the contract's edge modifier (polar/stationary at home) so the
        # ordering dv matches the runtime rule, not the base orbit.
        transform = info.spec.mission_transform(mission_builder)
        profiles = [transform(profile) for profile in profiles]
    return min(sum(e.base_dv for e in profile) for profile in profiles)


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
      - Tech tree goals: gate on accumulated science.  The greedy body
        selection in ``_pick_tech_tree_anchors`` adds body-orbit
        S_tier_anchor spheres until their combined science (with
        PSI=3 injected) covers ``cumulative_tier_cost(MAX_TIER)``.
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
    # Tech-tree anchors (only for complete_tech_tree).  These are body-orbit
    # locations the chain extends through so cumulative science covers
    # cumulative_tier_cost(MAX_TIER).  Builder validates feasibility.
    out.extend(_pick_tech_tree_anchors(world))
    # Goal-mode anchors (count / progressive_unlock): the player must complete X
    # non-goal contracts to unlock the goal, so the chain must thread the kit to
    # reach those contracts.  Without this, a trivial goal (random contracts'
    # flag-at-home) builds a ladder too shallow to bootstrap the deeper contracts,
    # and fill strands their kit unreachably so the threshold can never be
    # satisfied.  Each contract reward is physics-gated (it has a signature), so it
    # threads exactly like a goal sphere.
    #
    # This applies to the tech-tree goal in count/progressive too: even though its
    # science anchors above build a deep ladder, those anchors thread the science
    # kit, not the contract kit (drill / ore_tank / battery / lab), so the
    # contracts still strand without their own S_contract anchors.  (For FINDABLE
    # mode the contracts aren't threshold-gated and this block is never reached, so
    # the tech-tree findable goal stays unconstrained by contract anchors.)
    from .options import GoalContractMode
    if world.options.goal_contract_mode.value in (
            GoalContractMode.option_count,
            GoalContractMode.option_progressive_unlock):
        for spec in world.contract_specs:
            name = spec.location_name  # slot 1; both slots share one signature
            if name not in infeasible and _parse_location(name) is not None:
                out.append((f"S_contract[{name}]", name))
    return out


# Items injected into the cumulative kit of tech-tree anchor spheres.
# These are bookkeeping items (gate tech-tree access, not rocket physics);
# the bumper merges them into the rocket's delta+cumulative after
# ``minimal_ranks_for`` builds the physics part.  Result: chain_required
# carries them through, Rule B distributes copies by sphere ordering.
_TECH_ANCHOR_INJECT = {
    PROGRESSIVE_RD_NAME: 3,                   # = MAX_RD_BAND
    "Progressive Science Instrument": 3,
}


# Tie-band width for tech-tree anchor selection.  At each greedy step
# the next pick is made by ``world.random.choice`` over bodies whose
# return-Δv is within ``cheapest * (1 + _TIER_ANCHOR_DV_BAND_FRAC)``.
# The band keeps every candidate in the same physical "tier" of mission
# (interplanetary-vs-Sun, intra-Jool, ...) so the campaign stays
# physically close to home, while giving seed-to-seed variety in *which*
# bodies the player tours.
_TIER_ANCHOR_DV_BAND_FRAC: float = 0.20


def _pick_tech_tree_anchors(
    world: "KSP1World",
) -> list[tuple[str, str]]:
    """Return (label, location_name) anchor spheres for complete_tech_tree.

    Greedy ``"X Return 1"`` selection: starting from home-system science
    (with PSI=3 + full crew/instrument kit), add interplanetary body
    returns one at a time, picking randomly from a Δv tie band around
    the cheapest unpicked body (see ``_TIER_ANCHOR_DV_BAND_FRAC``).
    Stops when the running total of ``body_max_yield`` covers
    ``cumulative_tier_cost(MAX_TIER) / safety``.

    The tie-band random pick uses ``world.random`` (seed-derived), so
    the same seed always produces the same anchor list — different
    seeds for the same home pick different bodies within physically
    similar Δv neighbourhoods.

    Anchors at ``"X Return 1"`` because ``bankable_science`` only counts
    a body the player can recover from (RETURN access) or transmit from
    (high relay tier).  Orbit anchors leave bankable at zero for that
    body in the post-fill sphere walk.

    Each anchor body also acts as fill scaffolding: explicitly
    sphere-anchoring forces AP fill to thread the kit items (capsule,
    parachute, heat-shield) into reachable spheres.  Transitive RETURN
    capability across bodies does not survive without that scaffolding.

    The validation invariant: when this function returns, the chain's
    accumulated science across home-system + selected anchors must satisfy
    the whole-tree threshold.  If no body set covers it, raises
    ``OptionError`` at ``pre_fill`` time — loud failure beats silent
    unsolvable seed.

    Returns ``[]`` for non-tech-tree goals.
    """
    if not world.goal_spec.complete_tech_tree:
        return []

    from .bodies import (
        ALL_BODIES, BODY_BY_NAME, BodyName,
        home_system_bodies, science_budget,
    )
    from .rules import effective_science_safety
    from .tech_tree import cumulative_tier_cost, MAX_TIER

    safety = effective_science_safety(world.options, world.options.difficulty.value)
    target_raw = cumulative_tier_cost(MAX_TIER) / safety

    home = world.mission_builder.home
    home_set = home_system_bodies(home)

    # Per-body upper-bound yield: full kit (thermometer + barometer +
    # capsule + crew-land + PSI=3).  Matches the per-sphere tier-funding
    # pass's accounting once the bumper has injected the kit.
    def body_max_yield(body) -> float:
        return science_budget(
            body,
            has_thermometer=True,
            has_barometer=True,
            has_capsule=True,
            can_land_crewed=body.can_land,
            can_land_uncrewed=body.can_land,
            home=home,
            psi_tier=3,
        )

    accumulated = sum(body_max_yield(BODY_BY_NAME[bn]) for bn in home_set)

    # Return-capable interplanetary candidates, cheapest-dv first.
    # (Exclude home-system, Kerbol, and Jool — no RETURN profile.)
    interp_bodies = [
        b for b in ALL_BODIES
        if b.name not in home_set
        and b.name != BodyName.KERBOL
        and b.can_land
    ]
    dv_for = {
        b.name: _goal_dv(f"{b.name} Return 1", world.mission_builder)
        for b in interp_bodies
    }
    interp_bodies.sort(key=lambda b: (dv_for[b.name], b.name.value))

    # Greedy walk with a Δv tie-band: among bodies within
    # ``cheapest * (1 + _TIER_ANCHOR_DV_BAND_FRAC)`` of the current
    # cheapest unpicked, pick randomly via ``world.random`` (seed-derived,
    # so the pick is deterministic per seed).
    anchors: list[tuple[str, str]] = []
    remaining = list(interp_bodies)
    while accumulated < target_raw and remaining:
        cheapest_dv = dv_for[remaining[0].name]
        band_max = cheapest_dv * (1.0 + _TIER_ANCHOR_DV_BAND_FRAC)
        band = [b for b in remaining if dv_for[b.name] <= band_max]
        picked = world.random.choice(band)
        remaining.remove(picked)
        accumulated += body_max_yield(picked)
        anchors.append((f"S_tier_anchor[{picked.name}]", f"{picked.name} Return 1"))

    if accumulated < target_raw:
        from Options import OptionError
        raise OptionError(
            f"KSP1 complete_tech_tree: cumulative science with every "
            f"reachable body return at PSI=3 ({accumulated:.0f}) is below the "
            f"tier-{MAX_TIER} threshold ({target_raw:.0f} raw / "
            f"{cumulative_tier_cost(MAX_TIER)} after safety={safety:.2f}). "
            f"The seed is unsolvable. Try a lower difficulty (looser safety) "
            f"or report this if it appears with default options."
        )
    return anchors


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
    extra_names = set(world.location_builder.ksc_biome_names) | {f"{home} First Launch"}
    for loc in world.multiworld.get_locations(player):
        if loc.name in extra_names:
            loc.item_rule = local_only


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _demote_non_rep_parts(
    world: "KSP1World",
    rep_names: set[str],
    chain_cumulative: Signature,
    chain_extras: Optional[dict[str, int]] = None,
) -> int:
    """Demote every PROGRESSION part the bumper didn't designate as a rep,
    AND strip rank-axis entries that exceed the chain's cumulative ceiling.

    Two effects per item:
      1. Classification: bumper reps stay PROGRESSION; the three
         counted progressives (R&D, Pad, PSI) stay PROGRESSION; every
         other PROGRESSION item demotes to USEFUL.
      2. Rank signature: any (axis, rank) entry whose rank exceeds the
         chain's cumulative ceiling on that axis is dropped from the
         item's ``rank_sig``.  Items past goal on every axis end up with
         an empty signature and place freely (the rank-ceiling rule
         exempts empty-sig items).
    """
    from BaseClasses import ItemClassification
    from .items import (
        PROGRESSIVE_RD_NAME, PROGRESSIVE_LAUNCH_PAD_NAME,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME,
        PROGRESSIVE_VAB_NAME, PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
        PROGRESSIVE_TRACKING_STATION_NAME,
    )
    from .ranks import ItemRankSig
    # The counted progressives whose copies must stay PROGRESSION up to the
    # chain's highest needed level (chain_extras): R&D / Pad / PSI plus the
    # curated buildings (only pooled when buildings_in_logic is on; absent from
    # the pool otherwise, so naming them here is a harmless no-op when off).
    _KEEP_PROGRESSIVE: frozenset[str] = frozenset({
        PROGRESSIVE_RD_NAME,
        PROGRESSIVE_LAUNCH_PAD_NAME,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME,
        PROGRESSIVE_VAB_NAME,
        PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
        PROGRESSIVE_TRACKING_STATION_NAME,
    })
    ceiling = {r.axis: r.level for r in chain_cumulative.rank_reqs}
    player = world.player
    demoted = 0
    promoted = 0
    for item in world.multiworld.itempool:
        if item.player != player:
            continue
        # (2) Trim past-goal axes from the rank_sig so the item isn't
        # gated by axes the chain never raised that far.
        sig = getattr(item, "rank_sig", None)
        if sig is not None and sig.axes:
            kept = tuple(
                (axis_key, rank) for axis_key, rank in sig.axes
                if rank <= ceiling.get(axis_key, 0)
            )
            if len(kept) != len(sig.axes):
                item.rank_sig = ItemRankSig(kept)
        # (1) PROMOTE bumper-picked reps to PROGRESSION regardless of
        # their initial classification.  The chain's feasibility proof
        # assumed the player collects each rep; if some reps are USEFUL
        # / FILLER (e.g., the legacy ``_RECLASSIFY_USEFUL`` list marks
        # ``mk3FuselageLF.50`` as USEFUL because Mk3 fuselages were
        # considered redundant alternates), fill scatters them anywhere
        # and the player may never reach them.  Promoting forces AP to
        # place every rep at a reachable location, matching the bumper's
        # contract.
        # (1a) Spare counted-progressive copies (R&D / Pad / PSI).  A copy
        # gates progression only up to the max tier the chain actually bumped
        # (``chain_extras[name]`` — for R&D the highest funded band, for Pad
        # the highest mass tier, for PSI the highest science tier; the chain
        # never bumps PSI so chain_extras[PSI]=0).  Copies beyond that gate
        # nothing, yet keeping them PROGRESSION clogs the restrictive fill's
        # early region with dead weight — e.g. all 3 PSI and the unused R&D
        # bands land at ms=0 every simple-goal seed and crowd out the reps
        # that genuinely need an early home.  Demote those spares to USEFUL so
        # they scatter freely; the chain-needed tiers fall through and stay
        # PROGRESSION below.
        if item.name in _KEEP_PROGRESSIVE and chain_extras is not None:
            tier = getattr(item, "_sphere_tier", None)
            keep_through = chain_extras.get(item.name, 0)
            # complete_tech_tree's Victory requires EVERY R&D band collected
            # (state.has(PROGRESSIVE_RD_NAME, MAX_RD_BAND)).  So every R&D copy
            # is goal-required and must stay PROGRESSION regardless of how many
            # bands the chain funded — AP only guarantees reachability for
            # progression items, and a demoted (useful) R&D copy can strand at
            # an unreachable location, making the goal unsolvable.
            if (item.name == PROGRESSIVE_RD_NAME
                    and world.goal_spec.complete_tech_tree):
                from .tech_tree import MAX_RD_BAND
                keep_through = MAX_RD_BAND
            if tier is not None and tier > keep_through:
                if item.classification == ItemClassification.progression:
                    item.classification = ItemClassification.useful
                    demoted += 1
                continue
        if item.name in rep_names or item.name in _KEEP_PROGRESSIVE:
            if item.classification != ItemClassification.progression:
                item.classification = ItemClassification.progression
                promoted += 1
            continue
        # (2) Demote non-reps.
        if item.classification != ItemClassification.progression:
            continue
        item.classification = ItemClassification.useful
        demoted += 1
    return demoted


# Access-rule mode (prototype).  Controls how location reachability is
# verified during AP fill:
#   "strict_validation" — full capability physics on the actual collected
#                          state.  Original behavior; correct but slow
#                          (~6.9s of fill per SSR seed re-running the
#                          optimizer thousands of times in the sweep).
#   "strict_ladder"     — cheap sphere-bracket has-item rule PLUS a
#                          one-time validation pass asserting each
#                          bracket's cumulative kit actually reaches the
#                          location via capability.  Proves the ladder's
#                          bracketing is correct.
#   "ladder"            — cheap bracket rule only.  Fastest; trusts the
#                          ladder's proof entirely.
# The chain (pre_fill) is the expensive proof run once; fill then uses
# the cheap rules it produced.  USEFUL parts are NOT exempt from logic:
# their placement is still gated by the rank-ceiling item_rule, so a
# powerful part can't land below its sphere regardless of access mode.
#
# Env-overridable so solve-check can A/B the modes without a code edit.
# Default is strict_ladder: the cheap sphere-bracket rules drive fill, and
# post_fill swaps the capability rules back in to assert the placement is
# winnable under real physics.  Validated at N=100/goal (0 failures).
_ACCESS_RULE_MODE = os.environ.get("KSP_ACCESS_RULE_MODE", "strict_ladder")


def _make_bracket_rule(player: int, reps: tuple, signature: "Signature"):
    """Cheap reachability rule DERIVED from the location's full requirement
    signature — total over every ``Threshold`` kind:

      * ``Rank``    — satisfied by ``reps`` (the bracket's chosen parts for the
        ranks; the rank→parts translation).
      * ``Counted`` — ``count(kind) >= level`` (pad / building tiers).
      * ``Item``    — ``has(name)`` (a non-physics gate, e.g. a contract award,
        whose real rule is ``has(award) AND can_deliver``).

    An unhandled ``Threshold`` kind **raises** — so a requirement declared in the
    signature can never be silently dropped from the access rule.  (This is the
    structural guarantee: the omission that stranded contracts becomes a
    construction-time ``TypeError``, not a writable bug.)  Microsecond
    has/count checks — no capability physics.
    """
    counted: list[tuple[str, int]] = []
    items: list[str] = []
    for r in signature.reqs:
        if isinstance(r, Rank):
            pass  # the rank→parts translation IS ``reps``
        elif isinstance(r, Counted):
            counted.append((r.kind, r.level))
        elif isinstance(r, Item):
            items.append(r.name)
        else:
            raise TypeError(
                f"_make_bracket_rule: unhandled requirement {type(r).__name__} — "
                "every Threshold kind must be translated, else the access rule "
                "silently omits it")
    counted_t = tuple(counted)
    items_t = tuple(items)

    def rule(state) -> bool:
        for rp in reps:
            if not state.has(rp, player):
                return False
        for kind, level in counted_t:
            if state.count(kind, player) < level:
                return False
        for it in items_t:
            if not state.has(it, player):
                return False
        return True
    return rule


def _mission_needs_travel(info: "_LocationMissionInfo",
                          mission_builder: MissionBuilder) -> bool:
    """True iff the mission has a non-empty edge profile (a rocket must fly).

    Home-surface FLAG_PLANT / SAMPLE_RETURN register an EMPTY profile (the
    Kerbal walks off the pad), so they need no travel — and EVA there is
    allowed at Astronaut Complex level 0 in stock KSP.  Mirrors the
    empty-profile exemption in ``capability._evaluate_profile``.
    """
    profiles = mission_builder.profiles_for(info.body, info.mission_type)
    return any(bool(p) for p in profiles)


def _mission_building_reqs(
    info: "_LocationMissionInfo", launch_mass: float, *, home: "BodyName",
    needs_travel: bool,
) -> tuple[tuple[str, int], ...]:
    """Per-mission curated-building requirements as ``(item_name, level)``.

    Derived from the mission's own physics (its launch mass + whether it needs
    EVA), mirroring the per-mission pad gate.  Each curated effect is inverted
    to the minimum building level via ``effects.min_building_level_for`` and
    mapped to its AP progressive item name.  Only positive levels are recorded.

    * VESSEL_MASS_LIMIT (VAB): the lightest VAB level whose buildable-mass cap
      fits this mission's launch mass.
    * CAN_EVA (Astronaut Complex): level 1 for EVA missions that require travel
      (``needs_travel`` — a non-empty profile); home-surface walk-off-pad EVA is
      allowed at AC level 0, matching the empty-profile exemption.
    """
    from .effects import Effect, min_building_level_for
    from .items import _building_to_item_name
    from .capability import MISSION_TYPES_REQUIRING_EVA

    name_for = _building_to_item_name()
    reqs: list[tuple[str, int]] = []

    # VAB vessel-mass cap (only meaningful for missions that fly).
    if needs_travel:
        _vab_building, vab_level = min_building_level_for(
            Effect.VESSEL_MASS_LIMIT, launch_mass, home=home)
        if vab_level > 0:
            reqs.append((name_for[_vab_building], vab_level))

    # Astronaut Complex EVA gate.
    eva_required = (info.requires_eva if info.requires_eva is not None
                    else info.mission_type in MISSION_TYPES_REQUIRING_EVA)
    if eva_required and needs_travel:
        _ac_building, ac_level = min_building_level_for(
            Effect.CAN_EVA, True, home=home)
        if ac_level > 0:
            reqs.append((name_for[_ac_building], ac_level))

    return tuple(reqs)


def _install_ladder_rules(
    world: "KSP1World",
    ladder: SphereLadder,
    location_signatures: dict[str, Signature],
    bootstrap_locations: set,
    save_original: bool = False,
    install_access: bool = True,
) -> None:
    """Compute each capability-gated location's feasibility bracket and,
    when ``install_access`` is set, swap in the cheap ``has_all(reps)``
    access rule driven by that single bracket.

    For each location L:
      * **Bracket** ``j`` = first chain sphere whose reps-only flags
        actually reach L's mission (capability, not rank coverage —
        deduped per mission so the scan runs ~40 times, not ~250).
      * **Access rule** (only if ``install_access``):
        ``state.has_all(sphere[j].reps_collected)`` plus the sphere's
        counted-progressive thresholds.  L is reachable once the player
        holds the kit that the ladder proved reaches it.

    ``install_access=False`` still records the bracket (so the unified
    placement rule's loc_sphere uses the chain's feasibility oracle) but
    leaves the raw capability access rule untouched — the capability /
    strict_validation verification modes gate the whole seed on physics.
    Placement is never installed here; that is the windowed sphere rule's
    job (:func:`_install_unified_sphere_rules`).
    """
    player = world.player
    spheres = ladder.spheres
    diff = DIFFICULTY_PROFILES[
        ["casual", "normal", "expert", "insane"][world.options.difficulty.value]
    ]
    mb = world.mission_builder
    buildings_in_logic = bool(world.options.buildings_in_logic)
    bn_home = mb.home

    # strict_ladder: keep the original capability access rule per
    # location so post_fill can swap it back in and independently
    # re-verify the cheap-rule fill is winnable under real capability.
    saved: dict[str, object] = {}
    # Contract-ruled locations (completion slots, completion events, goal-contract
    # mission events) already carry a CHEAP real rule (has(award) AND
    # has_all(cheap_contract_reps); rules._set_contract_rules).  Overriding it with
    # the generic bracket rule buys no speed (both are cheap) but DIVERGES the
    # fill-time rule from the post_fill rule — the bracket rule keys on the sphere's
    # cumulative reps_collected (not the contract's own cheap_contract_reps) and
    # omits the award gate on goal-contract events, so fill stranded the award /
    # required parts and the strict_ladder cross-check then forced an expensive
    # whole-seed capability re-fill.  Leave these rules in place so fill and
    # post_fill use the SAME rule (single source of truth).
    contract_ruled: set = getattr(world, "_contract_ruled_locations", set())
    bracket_by_mission: dict[tuple, Optional[int]] = {}
    # Per-location feasibility bracket (first sphere whose cumulative kit can
    # FLY the mission).  This is the single source of truth for a mission's
    # sphere — the placement rule reuses it instead of re-deriving via
    # rank-vector domination, which diverges from the chain (see
    # _install_unified_sphere_rules).
    bracket_by_loc: dict[str, int] = {}
    rebracketed = 0
    # Contract locations carry an extra REAL-rule gate beyond physics: the award
    # item (``has(award) AND can_deliver``).  Map every contract slot -> award so
    # the cheap bracket rule includes it and matches the real rule (otherwise the
    # cheap fill strands the contract once real rules return).  Generic over all
    # contract kinds; no per-type special-casing.
    contract_gate: dict[str, str] = {}
    for spec in (*getattr(world, "contract_specs", ()),
                 *getattr(world, "goal_contract_specs", ())):
        for slot in spec.location_names(world.non_goal_slot_count):
            contract_gate[slot] = spec.item_name
    for loc in world.multiworld.get_locations(player):
        if loc.address is None or loc.name in bootstrap_locations:
            continue
        if loc.name not in location_signatures:
            continue  # not capability-gated (proxy) — leave rule
        info = _parse_location(loc.name)
        if info is None:
            continue  # tech anchor — gates on science, not capability
        mkey = _mission_key(info)
        if mkey in bracket_by_mission:
            j, pad_req, building_reqs = bracket_by_mission[mkey]
        else:
            j = None
            pad_req = 0
            building_reqs: tuple[tuple[str, int], ...] = ()
            for i, s in enumerate(spheres):
                if s.flags is None:
                    continue
                r = _evaluate(s.flags, info, diff, mb)
                if r.feasible:
                    j = i
                    # Precise per-mission pad = the lightest pad tier (number
                    # of copies) whose tonnage cap fits THIS mission's launch
                    # mass — its own physics requirement, not the chain's
                    # cumulative pad.  caps[T] is the cap with T copies, so a
                    # ≤caps[0] mission needs 0 copies (no gate) and only heavier
                    # missions record a requirement.  (Approximation: mass is
                    # measured with the bracket sphere's pad; the mass↔pad
                    # staging coupling is left to the dynamic-tier work.)
                    caps = mb.launch_pad_caps
                    pad_req = next((t for t, c in enumerate(caps)
                                    if r.launch_mass <= c), len(caps) - 1)
                    # Precise per-mission building reqs (buildings_in_logic) —
                    # the same self-gate the pad gets, derived from THIS
                    # mission's physics at the bracket sphere.
                    if buildings_in_logic:
                        building_reqs = _mission_building_reqs(
                            info, r.launch_mass, home=bn_home,
                            needs_travel=_mission_needs_travel(info, mb))
                    break
            if j is None and buildings_in_logic:
                # Unbracketed mission (beyond the chain's reps-only reach, e.g.
                # a far body's EVA for a near goal).  It still needs its
                # building gate recorded so a unique-provider building copy
                # can't strand at a location that requires a higher building
                # level than the copy supplies.  Evaluate at the maximal chain
                # kit (last sphere with flags) to read its true gate.
                last_flags = next(
                    (s.flags for s in reversed(spheres) if s.flags is not None),
                    None)
                if last_flags is not None:
                    r2 = _evaluate(last_flags, info, diff, mb)
                    mass = r2.launch_mass if r2.feasible else float("inf")
                    building_reqs = _mission_building_reqs(
                        info, mass, home=bn_home,
                        needs_travel=_mission_needs_travel(info, mb))
            bracket_by_mission[mkey] = (j, pad_req, building_reqs)
        # Record per-mission building reqs even for unbracketed missions so the
        # unique-provider building copies never strand behind them.
        for _kind, _lvl in building_reqs:
            if _lvl > 0:
                location_signatures[loc.name] = location_signatures.get(
                    loc.name, Signature.empty()
                ).with_counted(_kind, _lvl)
        if j is None:
            # No sphere reaches this mission with its reps-only kit —
            # leave the capability rule as the (slow) fallback.
            continue
        bracket_by_loc[loc.name] = j
        # Missions are gated by MASS (the pad tonnage cap), so the pad is their
        # only counted-progressive requirement (R&D/PSI gate science, not
        # missions).  Recording the precise per-mission pad makes the placement
        # self-ban a real destination gate: a pad copy can't land at a mission
        # that already needs that many copies.
        if pad_req > 0:
            location_signatures[loc.name] = location_signatures.get(
                loc.name, Signature.empty()
            ).with_counted(PROGRESSIVE_LAUNCH_PAD_NAME, pad_req)
        # (per-mission building reqs are recorded above, before the j-is-None
        # bail, so they also cover unbracketed-but-eventually-reachable
        # locations — unique-provider building copies must never strand there.)
        # Fold the contract award (a non-physics Item gate) INTO this location's
        # signature.  It is now one Threshold among the physics ranks/counted in
        # the single signature, not a side channel the cheap rule could forget —
        # the rule-deriver picks it up structurally (single source of truth).
        _gate = contract_gate.get(loc.name)
        if _gate is not None:
            location_signatures[loc.name] = location_signatures.get(
                loc.name, Signature.empty()).with_item(_gate)
        if install_access and loc.name not in contract_ruled:
            if save_original:
                saved[loc.name] = loc.access_rule
            sphere = spheres[j]
            # Access rule DERIVED from the location's full signature (reps satisfy
            # the Rank reqs; Counted/Item come from the signature).  Total over
            # Threshold kinds, so no declared requirement can be omitted.
            loc.access_rule = _make_bracket_rule(
                player,
                tuple(sphere.reps_collected),
                location_signatures.get(loc.name, Signature.empty()),
            )
        # No item_rule ban here.  The chicken-and-egg (a rep needed to reach
        # L sitting at L) is prevented by AP's restrictive fill, which never
        # places a progression item at a location unreachable without it —
        # the same protection strict_validation relies on.  Placement balance
        # is the unified sphere rule's job (_install_unified_sphere_rules).
        rebracketed += 1
    world._cheap_access_rebracketed = rebracketed
    world._cheap_access_bracket = bracket_by_loc
    if save_original and install_access:
        world._strict_ladder_saved_rules = saved
    _install_cheap_mission_reps(world, ladder)


def _install_cheap_mission_reps(world: "KSP1World", ladder: SphereLadder) -> None:
    """Precompute, per ``(body, event)``, the cheap bracket reps that gate that
    mission — the SAME ``has_all(reps)`` the location's access rule uses.

    Lets the goal rule (and other body-access consumers) decide "can the player
    do <event> at <body>?" with a microsecond ``state.has_all`` check instead of
    a live ``get_capability`` call, keeping the goal completion condition on the
    same cheap ladder oracle as the location rules.  Keyed by the lowest bracket
    sphere across an event's duplicate slots (they share one mission).
    """
    bracket = getattr(world, "_cheap_access_bracket", {})
    spheres = ladder.spheres
    reps_by_event: dict[tuple[str, str], tuple[int, frozenset[str]]] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        ml = MissionLocation.parse(loc.name)
        if ml is None:
            continue
        j = bracket.get(loc.name)
        if j is None:
            continue
        key = (ml.body, ml.event.value)
        prev = reps_by_event.get(key)
        if prev is None or j < prev[0]:
            reps_by_event[key] = (j, frozenset(spheres[j].reps_collected))
    world._cheap_mission_reps = {k: v[1] for k, v in reps_by_event.items()}

    # Per-contract delivery reps: the bracket reps of the contract's mission
    # location(s) — the cheap stand-in for ``contract_access[cid]`` (can the
    # player deliver the payload).  ``has_all(reps)`` ⟹ the bracket kit flies the
    # contract mission with its payload, so it's conservative-sound like the
    # ordinary mission gates, and lets the contract access rule stay off
    # ``get_capability`` during fill.
    contract_reps: dict[str, frozenset[str]] = {}
    for spec in (*getattr(world, "contract_specs", ()),
                 *getattr(world, "goal_contract_specs", ())):
        best: Optional[tuple[int, frozenset[str]]] = None
        for ln in spec.location_names(world.non_goal_slot_count):
            j = bracket.get(ln)
            if j is not None and (best is None or j < best[0]):
                best = (j, frozenset(spheres[j].reps_collected))
        if best is not None:
            contract_reps[spec.contract_id] = best[1]
    world._cheap_contract_reps = contract_reps


def _first_covering_sphere(spheres, need: Signature) -> int:
    """Ladder index of the first sphere whose ``provides`` covers ``need``.

    Sphere provisions (rank ceilings + counted-progressive levels) grow
    monotonically along the chain, so the first covering sphere is the
    ladder position.  Returns ``len(spheres)`` when no real sphere covers
    it (beyond the chain's reach).

    Covering uses :meth:`Signature.covers`: an axis/kind absent from the
    sphere's provisions is level 0 — *unavailable*, not unconstrained.
    """
    for i, s in enumerate(spheres):
        if s.provides.covers(need):
            return i
    return len(spheres)


_SIZE_ONLY_AXES: Optional[frozenset] = None


def _size_only_axes() -> frozenset:
    """Rank axes whose rank reflects SIZE, not capability — derived, never
    hardcoded.  An axis is size-only iff its parts all share a fuel/dry ratio
    (relative spread < 30%): then a higher rank is a *bigger* part, not a
    *better* one (same dv-per-mass).  Today this derives the LFO and Xenon tank
    axes (constant ratio); LF/monoprop tanks vary, so they stay capability ranks.

    Used by placement only: a size-only axis contributes just its presence
    (rank 1) to an item's placement floor, so a big tank places as early as a
    small one (the first-tank-of-a-fuel-type is the real unlock; size is free).
    The bumper still sees the full rank — size matters for part-count feasibility.
    """
    global _SIZE_ONLY_AXES
    if _SIZE_ONLY_AXES is None:
        from .parts import PART_DB, FuelTank
        from .ranks import RankAxisKey
        ft_axis = {"lfo": RankAxisKey.LFO_TANK, "lf": RankAxisKey.LF_TANK,
                   "xenon": RankAxisKey.XENON_TANK,
                   "monoprop": RankAxisKey.MONOPROP_TANK}
        ratios: dict = {}
        for parts in PART_DB.values():
            for p in parts:
                if isinstance(p, FuelTank) and p.dry_mass > 0:
                    ax = ft_axis.get(p.fuel_type)
                    if ax is not None:
                        ratios.setdefault(ax, []).append(p.fuel_mass / p.dry_mass)
        _SIZE_ONLY_AXES = frozenset(
            ax for ax, rs in ratios.items()
            if rs and (max(rs) - min(rs)) / (sum(rs) / len(rs)) < 0.30)
    return _SIZE_ONLY_AXES


def _item_min_sphere(item, spheres) -> int:
    """Ladder position of an item — the first sphere at which it becomes
    available, and therefore the earliest location sphere it may sit at.

    * Parts build a need signature from their ``rank_sig`` axes (Rank reqs).
    * Counted progressives (R&D / Pad / PSI) build a need signature from a
      single Counted req (the item's name as kind at its ``_sphere_tier``).
    * Items the chain never requires — spare high-rank parts whose ranks
      exceed the goal, or counted progressives the goal never bumps — are
      unconstrained (sphere 0).  This reproduces the legacy "past-chain ⇒
      admitted anywhere" escape, so they remain free filler.
    """
    sig = getattr(item, "rank_sig", None)
    if sig is not None and sig.axes:
        # Size-only axes (ratio-constant tanks) count only their presence toward
        # the placement floor: a bigger tank is the same dv-per-mass, so it places
        # as early as the small one rather than pinning to a late "size" sphere.
        _so = _size_only_axes()
        need = Signature.of(
            Rank(ax, 1 if ax in _so else rk) for ax, rk in sig.axes)
        idx = _first_covering_sphere(spheres, need)
        return 0 if idx == len(spheres) else idx
    tier = getattr(item, "_sphere_tier", None)
    if tier is not None:
        idx = _first_covering_sphere(
            spheres, Signature.of((Counted(item.name, tier),)))
        return 0 if idx == len(spheres) else idx
    return 0


def _install_unified_sphere_rules(
    world: "KSP1World",
    ladder: "SphereLadder",
    location_signatures: dict[str, Signature],
    bootstrap_locations: set[str],
) -> None:
    """The unified placement rule — a per-item sphere *window* keyed on
    each item's ladder position ``ms = _item_min_sphere(item)``:

      * PROGRESSION reps: kit-exact UPPER bound only — admitted at any
        location whose min_kit doesn't already include the rep (i.e. anywhere
        below the sphere that first needs it).  No lower bound: AP's restrictive
        fill needs that freedom to place broadly-gating reps reachably (a lower
        bound collides with the min_kit upper bound and strands them).
      * USEFUL / filler: lower bound only, ``L.sphere >= max(0, ms - margin)``
        where ``margin = _USEFUL_FLOOR_MARGIN_FRAC * n`` — a surprise alternate
        can't arrive *far* before its tier, but the wide margin keeps a
        same-tier category from over-subscribing its band (which strands the
        fill tail with no valid matching).  May appear any time after.

    Parts, R&D, Pad and PSI share this one sphere-index window.  Full
    soundness (every location reachable with the kit placed before it) is
    additionally proven by the capability cross-check in ``post_fill``.
    Locations with no rank requirement (KSC, starting inventory, Victory)
    keep their existing rule.
    """
    player = world.player
    spheres = ladder.spheres
    # Missions: reuse the cheap-access FEASIBILITY bracket (first sphere whose
    # cumulative kit can fly the mission) — the chain's own oracle.  Re-deriving
    # via rank-vector domination diverges, because a mission's independent
    # minimal kit can sit on a different point of the Δv trade-off frontier
    # than the chain ever visits (incomparable vectors → falls to the top).
    # Tech nodes / KSC: their kit IS a chain sphere's (funding kit / capsule),
    # so _first_covering_sphere is exact for them.
    cheap_bracket: dict[str, int] = getattr(world, "_cheap_access_bracket", {})
    loc_sphere: dict[str, int] = {}
    for name, need in location_signatures.items():
        if name in cheap_bracket:
            loc_sphere[name] = cheap_bracket[name]
        else:
            loc_sphere[name] = _first_covering_sphere(spheres, need)

    # Non-progression placement floor: a fixed fraction of the ladder below each
    # part's tier (see _USEFUL_FLOOR_MARGIN_FRAC).  Precomputed once as a sphere
    # offset and applied in the rule below.
    margin_off = round(_USEFUL_FLOOR_MARGIN_FRAC * len(spheres))

    # Bootstrap kit: reps needed from sphere 0 are in EVERY location's cumulative
    # min_kit, so the kit-exact ban would forbid them everywhere.  They belong in
    # starting inventory (zero-requirement locations) — exempt them so AP fill
    # routes them there.
    _bootstrap_kit = spheres[0].reps_collected if spheres else frozenset()

    # Contract AWARD items float FREELY (sequence-break: contracts land at varied
    # points each seed instead of riding the physics ladder, giving off-physics
    # gating + variance, and off-loading low-sphere fill pressure).  An award is
    # placeable at ANY location, banned ONLY on its OWN contract's reward locations
    # (where the access rule is ``has(award) AND ...`` — a self-cycle).  The one
    # exception is a FINDABLE goal item, which caps at the deepest goal sphere so a
    # goal item never floats past the goal itself (a correctness bound, not pacing).
    # NO lower bound: a deep (high-completion) contract must be able to use the
    # plentiful low spheres — flooring its award at a high completion crowds it into
    # the scarce ladder top and wedges fill (measured: SSR count/progressive
    # regressed 5/5 -> 4/5 when floored at completion, recover to 5/5 with no floor,
    # variance preserved).  The ``can_deliver`` half of the access rule keeps the
    # contract physics-sound wherever the award lands.  (Replaces the old
    # sphere-granular ban that forced the award strictly BELOW completion — which
    # pinned contracts to physics order and made a completion=0 award unplaceable.)
    award_own_locs: dict[str, set[str]] = {}
    _goal_award_names: set[str] = set()
    for spec in (*getattr(world, "contract_specs", ()),
                 *getattr(world, "goal_contract_specs", ())):
        own = [ln for ln in spec.location_names(world.non_goal_slot_count)
               if ln in loc_sphere]
        if not own:
            continue
        award_own_locs.setdefault(spec.item_name, set()).update(own)
        if spec.is_goal:
            _goal_award_names.add(spec.item_name)
    # Deepest goal-contract reward location = the findable goal-item float cap.
    _goal_diff_sphere = max(
        (loc_sphere[ln]
         for spec in getattr(world, "goal_contract_specs", ())
         for ln in spec.location_names(world.non_goal_slot_count)
         if ln in loc_sphere),
        default=len(spheres))
    award_names = frozenset(award_own_locs)
    # Ceiling: goal items stay within the goal's difficulty; everything else has
    # no upper bound (len(spheres) > every real location sphere = free float).
    award_ceiling: dict[str, int] = {
        nm: (_goal_diff_sphere if nm in _goal_award_names else len(spheres))
        for nm in award_names
    }

    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        L = loc_sphere.get(loc.name)
        if L is None:
            continue  # ungated (KSC / starting inventory / event) — keep existing rule
        existing = loc.item_rule
        # This location's own signature, e.g. Counted(Pad, 2) + Counted(R&D, 1)
        # = "reaching L needs pad tier 2 and R&D band 1".
        my_sig = location_signatures.get(loc.name, Signature.empty())
        # L's min_kit = the cumulative reps that fly it (the parts its access rule
        # requires).  A rep in this set CANNOT land here (you'd need it to reach L
        # to collect it — a cycle); any other rep may.  This is the kit-exact
        # UPPER bound only (no lower bound): a rep is admitted at every location
        # whose kit doesn't include it, i.e. anywhere below the sphere that first
        # needs it.  Immune to sphere-index lumpiness (keys on the actual kit).
        _min_kit = (spheres[L].reps_collected
                    if L < len(spheres) else frozenset())

        def _rule(item, _p=player, _L=L, _spheres=spheres, _orig=existing,
                  _sig=my_sig, _moff=margin_off, _mk=_min_kit,
                  _boot=_bootstrap_kit, _loc=loc.name, _an=award_names,
                  _ac=award_ceiling, _ao=award_own_locs) -> bool:
            if _orig is not None and not _orig(item):
                return False
            if item.player != _p:
                return True
            if _L >= len(_spheres):          # sentinel: chain-unreachable location
                # _first_covering_sphere returned len(spheres) — no sphere covers
                # this location's signature, so it's unreachable in the chain.
                # An advancement item placed here strands (type-agnostic: parts,
                # counted progressives, contract gate items, goal items).
                return not item.advancement   # filler-only
            if item.advancement:
                if item.name in _an:
                    # Contract AWARD: placeable anywhere up to its ceiling
                    # (completion + N), banned ONLY on its own contract's reward
                    # locations (the has(award) self-cycle).  No lower bound.
                    if _loc in _ao.get(item.name, ()):
                        return False
                    return _L <= _ac[item.name]
                tier = getattr(item, "_sphere_tier", None)
                if tier is not None:
                    # Counted progressive (R&D / Pad / PSI) — the UNIQUE
                    # provider of its tier (no alternate gives you pad tier 2
                    # except the 2nd pad copy).  Admit this copy at L only
                    # where L does NOT already require this tier or higher;
                    # then L is reachable with a lower tier, so collecting the
                    # copy here can never be circular.  Cheap: one lookup in
                    # the chain's precomputed per-location requirement.
                    return _sig.counted(item.name) < tier
                # Progression PART (chain rep): kit-exact UPPER bound only.
                # Admit iff this rep is NOT in L's min_kit — i.e. L is reachable
                # without it, so collecting it here can't be circular.  No lower
                # bound: a rep may land anywhere below the sphere that first needs
                # it (max fill freedom; the restrictive fill needs this room to
                # place broadly-gating reps reachably — a cascade lower bound here
                # collides with the min_kit upper bound and strands reps).
                # Reps absent from every kit (spare high-rank parts) are in no
                # min_kit → admitted everywhere.  Bootstrap reps (needed from
                # sphere 0) are in every kit → exempt to starting inventory.
                if item.name in _boot:
                    return True
                return item.name not in _mk
            # Non-progression PART (filler): lower bound on sphere — a high-rank
            # part may not appear far before its tier (that would hand the player a
            # powerful part early, dropping pacing/fun).  No upper bound.  The
            # floor sits a fixed fraction of the ladder below the part's tier
            # (_USEFUL_FLOOR_MARGIN_FRAC): wide enough that a same-tier category
            # can't over-subscribe its band (which would strand the fill tail),
            # but high-tier parts still floor late.
            ms = _item_min_sphere(item, _spheres)
            if ms == 0:
                return True
            return max(0, ms - _moff) <= _L

        loc.item_rule = _rule


def _compute_tech_tier_signatures_rank(
    world: "KSP1World", ladder: SphereLadder, ctx: RankContext,
    location_signatures: dict[str, Signature],
) -> tuple[dict[str, LocationSignature], dict[str, Signature],
           dict[int, SphereBoundary]]:
    """Rank-space port of the legacy ``_compute_tech_tier_signatures``.

    Walks the chain, computes science accumulation per sphere from the
    capability the sphere's ``provides`` signature proves, and identifies
    the earliest sphere that funds each tech tier's cumulative cost.
    Returns:
      * ``signatures``  — tech-tree location → LocationSignature
      * ``min_sigs``    — tech-tree location → capability :class:`Signature`
        (the funding sphere's rank ceiling + the R&D / PSI / Pad copies the
        player should have by this sphere)
      * ``band_funding`` — R&D band → funding sphere (caller injects R&D
        copies at this sphere's depth)
    """
    from .capability import compute_capability_from_items
    from .locations import TechTreeLocation, effective_tech_slots_per_node
    from .rules import bankable_science, effective_science_safety
    from .tech_tree import TECH_NODES, TIER_TO_BAND, cumulative_tier_cost

    difficulty_idx = world.options.difficulty.value
    difficulty_name = ["casual", "normal", "expert", "insane"][difficulty_idx]
    safety = effective_science_safety(world.options, difficulty_idx)
    pad_on = bool(world.options.progressive_launch_pad)
    clamps = bool(world.options.start_with_launch_clamps)
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    home = world.mission_builder.home

    # Per-sphere science accumulation.  Use sphere.reps_collected
    # (reps-only model) so the capability matches what fill actually
    # places — not what the rank ceiling would *abstractly* admit.
    sphere_science: list[tuple[SphereBoundary, float]] = []
    from .items import PROGRESSIVE_SCIENCE_INSTRUMENT_NAME
    from .locations import EventName as _EvN
    # Per-(body, event) cheap bracket: the reps of the FIRST sphere whose
    # cumulative kit proves that body/event reachable.  bankable_science reads
    # ORBIT/RETURN/CREWED_LANDING, and the per-sphere ``cap`` is already computed
    # here for the science sum — so recording the first-true sphere's reps costs
    # nothing and lets the runtime science rule use a cheap ``has_all(reps)``
    # instead of a live ``get_capability``.  Derived FROM this funding pass, so
    # the runtime measure stays consistent with tier placement by construction
    # (bracket-true at sphere s ⟺ this pass's cap-access at s — access is
    # monotonic along the chain).
    _sci_events = (_EvN.ORBIT, _EvN.RETURN, _EvN.LANDING, _EvN.CREWED_LANDING)
    science_brackets: dict[tuple, frozenset[str]] = {}
    # Per-(body,event) science bracket from the ladder ORDERING, not a per-sphere
    # physics re-solve.  The bumper already placed every mission; the first sphere
    # whose cumulative ``provides`` covers a (body,event) mission's signature is a
    # cheap rank-cover bracket (``_first_covering_sphere``).  Measured to be
    # STRICTLY LATER than the old reps-based ``_assess_one_body`` bracket — i.e.
    # MORE conservative science (Golden-Rule safe: real collectable science is
    # already heavily underestimated) — and it removes the dominant generation
    # cost (no ``cap.bodies[*].access`` touch ⟹ the per-body optimizer never runs;
    # the per-sphere ``cap`` here supplies only cheap flag-level relay/instrument
    # state).
    spheres = ladder.spheres
    _cover_idx: dict[tuple, int] = {}
    for _b in ALL_BODIES:
        for _ev in _sci_events:
            _sig = location_signatures.get(f"{_b.name.value} {_ev.value} 1")
            if _sig is None:
                continue
            _ci = _first_covering_sphere(spheres, _sig)
            if _ci < len(spheres):
                _cover_idx[(_b.name, _ev)] = _ci
                science_brackets[(_b.name, _ev)] = frozenset(
                    spheres[_ci].reps_collected)
    world._science_body_event_reps = science_brackets

    for _si, sphere in enumerate(spheres):
        admitted = (sphere.reps_collected
                    | (precollected_names & frozenset(PART_DB.keys())))
        provides = sphere.provides

        def _count(name: str, _adm=admitted, _prov=provides) -> int:
            counted = _prov.counted(name)
            if counted > 0:
                return counted
            return 1 if name in _adm else 0

        # Flags only (relay tier + instruments).  Bodies stay lazy/untouched, so
        # no per-body optimizer runs; per-body access comes from ``_cover_idx``.
        cap, _flags = compute_capability_from_items(
            _count, difficulty_name,
            start_with_clamps=clamps,
            mission_builder=world.mission_builder,
            progressive_launch_pad=pad_on,
            buildings_in_logic=bool(world.options.buildings_in_logic),
        )
        psi_tier = provides.counted(PROGRESSIVE_SCIENCE_INSTRUMENT_NAME)
        _acc = {
            _b.name: {
                _ev: (_cover_idx.get((_b.name, _ev), len(spheres)) <= _si)
                for _ev in _sci_events
            }
            for _b in ALL_BODIES
        }
        sphere_science.append(
            (sphere,
             bankable_science(cap, psi_tier, home, access=_acc) * safety))

    sigs: dict[str, LocationSignature] = {}
    min_sigs_out: dict[str, Signature] = {}
    band_funding: dict[int, SphereBoundary] = {}
    num_slots = effective_tech_slots_per_node(world.options, difficulty_idx)
    tier_set = sorted({n.tier for n in TECH_NODES})

    for tier in tier_set:
        target = cumulative_tier_cost(tier)
        rd_required = TIER_TO_BAND.get(tier, 0)
        funding: Optional[SphereBoundary] = None
        for sphere, science in sphere_science:
            if science >= target and sphere.signature is not None:
                funding = sphere
                break
        if funding is None:
            # Unfundable tier — no sphere proves enough science under this
            # seed's goal.  Locations remain ungated by rank ceiling; the
            # demote-trim will scatter chain items past them.
            continue
        reqs_dict = dict(funding.signature.requirements)
        if rd_required > 0:
            reqs_dict["progressive_rd"] = rd_required
        sig = LocationSignature(
            dv=funding.signature.dv,
            requirements=tuple(sorted(reqs_dict.items())),
            body_chain_depth=funding.signature.body_chain_depth,
        )
        # The capability need for tech-tree locations is the funding sphere's
        # cumulative rank ceiling plus its counted progressives (PSI / Pad),
        # with the tier's R&D band folded in.
        min_sig = funding.provides
        if rd_required > 0:
            min_sig = min_sig.with_counted(PROGRESSIVE_RD_NAME, rd_required)
        for node in TECH_NODES:
            if node.tier != tier:
                continue
            for slot in range(1, num_slots + 1):
                loc_name = str(TechTreeLocation(node.display_name, slot))
                sigs[loc_name] = sig
                min_sigs_out[loc_name] = min_sig
        if rd_required > 0:
            prev = band_funding.get(rd_required)
            if prev is None or (
                prev.signature is not None
                and funding.signature.dv < prev.signature.dv
            ):
                band_funding[rd_required] = funding
    return sigs, min_sigs_out, band_funding


# Canonical within-body nesting order for the graph walk.  Each event inherits
# the prior event's cumulative kit (flyby -> orbit -> landing -> return ->
# sample-return — the validated PASS-2 hierarchy in
# scratchpad/analyze_prior_path.py).  Events that share a mission_type with a
# canonical step (EVA in Orbit / Crewed Landing / Flag Plant / SOI Leave) are
# slotted alongside their nearest canonical event so every location's mission
# key still receives a tree-walk signature.  Lower value = walked earlier.
_GRAPH_WALK_EVENT_ORDER: dict[str, int] = {
    EventName.FLYBY.value:          0,
    EventName.SOI_LEAVE.value:      0,
    EventName.ORBIT.value:          1,
    EventName.EVA_IN_ORBIT.value:   1,
    EventName.LANDING.value:        2,
    EventName.CREWED_LANDING.value: 2,
    EventName.FLAG_PLANT.value:     2,
    EventName.RETURN.value:         3,
    EventName.SAMPLE_RETURN.value:  4,
}


def _build_ladder_graph_walk(
    world: "KSP1World",
    ladder: SphereLadder,
    ctx: RankContext,
    *,
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    buildings_in_logic: bool,
    precollected_names: frozenset[str],
    home: str,
    bn_home: BodyName,
) -> tuple[SphereLadder, dict[str, Signature], Signature,
           frozenset[str], dict[tuple[RankAxisKey, int], str]]:
    """Build the sphere ladder by a dependency-ordered walk of the mission
    graph (the validated PASS-2 hierarchy in ``scratchpad/analyze_prior_path.py``):

      * ``base`` = home-orbit kit (``minimal_ranks_for("<home> Orbit 1", empty)``).
      * Planets (``parent is None``, excl. Kerbol) and home-moons inherit the
        home-orbit kit; planet-moons inherit their PARENT PLANET's FLYBY kit
        (planets processed before moons so the parent flyby kit exists).
      * Within each body, strict nesting flyby->orbit->landing->return->SR, each
        ``minimal_ranks_for(loc, prior=prev.signature, prior_reps=prev.reps)``.

    Produces the same three outputs ``apply_sphere_ladder`` downstream consumes:
    a per-mission MARGINAL ``location_signatures`` dict, a monotonic-cumulative
    linear ``ladder.spheres`` (sorted by cumulative rank-sum, tie-break
    ``_goal_dv``), and the union ``cumulative_reps`` keep-set.
    """
    _bump_kw = dict(
        difficulty=difficulty,
        progressive_launch_pad=progressive_launch_pad,
        start_with_clamps=start_with_clamps,
        rng=world.random,
        mission_builder=world.mission_builder,
        precollected_names=precollected_names,
        buildings_in_logic=buildings_in_logic,
    )
    infeasible = world.model_infeasible_locations

    def _ranksum(sig: Signature) -> int:
        return sum(sig.rank(a) for a in RankAxisKey)

    # ---- base: home-orbit kit -------------------------------------------
    _ko = minimal_ranks_for(f"{home} Orbit 1", Signature.empty(), ctx,
                            prior_reps=frozenset(), **_bump_kw)
    ko_sig = _ko.signature if _ko is not None else Signature.empty()
    ko_reps = _ko.reps_collected if _ko is not None else frozenset()

    # ---- collect the events actually present per body -------------------
    # All mission locations grouped by (BodyName) -> set of EventName values.
    body_events: dict[str, set[str]] = {}
    # Canonical representative location-name for each (body, event) so the walk
    # reuses the real slot-1 name the bumper expects.
    locname_for: dict[tuple[str, str], str] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None or loc.name in infeasible:
            continue
        parsed = MissionLocation.parse(loc.name)
        if parsed is None:
            continue
        ev = parsed.event.value
        if ev not in _GRAPH_WALK_EVENT_ORDER:
            continue
        body_events.setdefault(parsed.body, set()).add(ev)
        locname_for.setdefault((parsed.body, ev), f"{parsed.body} {ev} 1")

    # ---- walk: planets/home-moons first, then planet-moons --------------
    bodies = [b for b in ALL_BODIES
              if b.name != bn_home and b.name != BodyName.KERBOL]
    ordered = sorted(bodies, key=lambda b: (b.parent is not None, b.name.value))
    body_flyby_kit: dict[BodyName, tuple[Signature, frozenset[str]]] = {}
    # Per-mission MARGINAL signature + the cumulative (sig, reps) AT that mission
    # (used to order the linear ladder and union the keep-set).
    mission_marginal: dict[tuple, Signature] = {}
    mission_cumulative: dict[tuple, tuple[Signature, frozenset[str]]] = {}
    mission_result: dict[tuple, RankBumperResult] = {}
    sphere_rank_reps: dict[tuple[RankAxisKey, int], str] = {}
    cumulative_reps_acc: set[str] = set(ko_reps)

    for b in ordered:
        if b.parent is None or b.parent == bn_home:   # planet or home-moon
            reach = (ko_sig, ko_reps)
        else:                                          # planet-moon: parent flyby
            reach = body_flyby_kit.get(b.parent, (ko_sig, ko_reps))
        cum_sig, cum_reps = reach
        present = body_events.get(b.name.value)
        if not present:
            continue
        # Walk this body's present events in dependency order.
        # Tie-break the event-order on the event value: FLYBY/SOI_LEAVE (both 0),
        # ORBIT/EVA_IN_ORBIT (both 1), LANDING/CREWED_LANDING/FLAG_PLANT (all 2)
        # share an order, and ``present`` is a set whose iteration is hash-
        # randomized per process.  Without the tie-break the shared rng threads
        # through these missions in a different order across processes -> the
        # bumper's rep picks (and thus the whole ladder) become non-reproducible.
        for ev in sorted(present, key=lambda e: (_GRAPH_WALK_EVENT_ORDER[e], e)):
            loc_name = locname_for[(b.name.value, ev)]
            info = _parse_location(loc_name)
            if info is None:
                continue
            mkey = _mission_key(info)
            rocket = minimal_ranks_for(
                loc_name, cum_sig, ctx,
                prior_reps=cum_reps, **_bump_kw,
            )
            if rocket is None:
                continue
            for key, rep in rocket.reps.items():
                sphere_rank_reps[key] = rep
            mission_marginal[mkey] = Signature.of(rocket.signature.rank_reqs)
            mission_cumulative[mkey] = (rocket.signature, rocket.reps_collected)
            mission_result[mkey] = rocket
            cumulative_reps_acc |= set(rocket.reps_collected)
            # Strict nesting: advance the within-body cumulative.
            cum_sig, cum_reps = rocket.signature, rocket.reps_collected
            if ev == EventName.FLYBY.value:
                body_flyby_kit[b.name] = (rocket.signature, rocket.reps_collected)

    cumulative_reps: frozenset[str] = frozenset(cumulative_reps_acc)

    # ---- assign every mission location its tree-walk marginal signature --
    location_signatures: dict[str, Signature] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None or loc.name in infeasible:
            continue
        info = _parse_location(loc.name)
        if info is None:
            continue
        mkey = _mission_key(info)
        derived = mission_marginal.get(mkey)
        if derived is None:
            # A mission location whose canonical (body, event) walk produced no
            # result (infeasible under any kit it was offered) — fall back to a
            # from-empty intrinsic so the location still gets a signature.
            rocket = minimal_ranks_for(
                loc.name, Signature.empty(), ctx,
                prior_reps=frozenset(), **_bump_kw,
            )
            derived = (Signature.of(rocket.signature.rank_reqs)
                       if rocket is not None else None)
            mission_marginal[mkey] = derived
        if derived is None:
            continue
        location_signatures[loc.name] = derived
        ladder.location_signatures[loc.name] = LocationSignature(
            dv=_goal_dv(loc.name, world.mission_builder),
            requirements=tuple(),
            body_chain_depth=_body_chain_depth(
                info.body, world.mission_builder.home),
        )

    # ---- predictable anchors (goal / tech / contract) -------------------
    # Each anchor is a mission whose cumulative kit must be threaded into the
    # ladder.  Reuse the walk's mission_cumulative when the anchor's mission was
    # already walked; otherwise bump it from the home-orbit kit (its own
    # dependency prior is unknown to the body-graph walk — goal/contract anchors
    # may be deep interplanetary returns).  Goal-feasibility semantics mirror the
    # from-empty path's fallback block.
    predictable_labels = [(label, name) for label, name in _predictable_spheres(world)]

    # ---- deep-space enabler inject (mirrors the from-empty path) --------
    # Folded into cumulative_reps/cumulative_sig the first time the walk reaches a
    # mission past _DEEP_INJECT_DV_FRAC of the dv range, so deeper missions can
    # build their high-dv stage.  dv per anchor/mission name via _goal_dv.
    _all_walk_names = (
        list(location_signatures.keys())
        + [name for _, name in predictable_labels]
    )
    _sphere_dv_by_name = {
        n: _goal_dv(n, world.mission_builder) for n in _all_walk_names
    }
    _deep_enablers = _DEEP_SPACE_ENABLERS - precollected_names
    _deep_max_dv = max(_sphere_dv_by_name.values(), default=0.0)
    _deep_inject_dv = (
        _DEEP_INJECT_DV_FRAC * _deep_max_dv
        if _deep_enablers and _deep_max_dv >= _DEEP_INJECT_MIN_DV
        else float("inf")
    )

    def _apply_deep_inject(sig: Signature, reps: frozenset[str],
                           dv: float) -> tuple[Signature, frozenset[str]]:
        if dv < _deep_inject_dv:
            return sig, reps
        reps = reps | _deep_enablers
        for _ep in _deep_enablers:
            for _ax, _rk in rank_sig_for(_ep, ctx).axes:
                if _rk > sig.rank(_ax):
                    sig = sig.with_rank(_ax, _rk)
        return sig, reps

    # If any walked mission crosses the deep threshold, fold the enablers into
    # the global keep-set (the per-mission cumulative inject is applied below
    # when each sphere is assembled, where the location's dv is in hand).
    if _deep_inject_dv != float("inf"):
        cumulative_reps = cumulative_reps | _deep_enablers

    # ---- assemble the predictable anchors into the walk -----------------
    # Process anchors in dv order, accumulating each one's cumulative kit into
    # the prior for the next.  This mirrors the from-empty path's linear walk
    # across goal anchors: a deep goal (e.g. Moho SR) bumps from the kit built
    # up by the easier goals before it, not from a thin home-orbit prior — its
    # high-dv transfer stage closes reliably only with that accumulated kit
    # (plus the deep-space enablers, folded into the prior when interplanetary).
    # (label, location_name, cumulative_sig, cumulative_reps, result, marginal_sig)
    anchor_entries: list[tuple[str, str, Signature, frozenset[str],
                               RankBumperResult, Optional[Signature]]] = []
    _anchor_sig = ko_sig
    _anchor_reps = ko_reps
    for label, name in sorted(
            predictable_labels,
            key=lambda ln: _goal_dv(ln[1], world.mission_builder)):
        dv = _goal_dv(name, world.mission_builder)
        # Fold the deep-space enablers into the accumulated prior when the anchor
        # is interplanetary-deep (the from-empty path does the same before
        # bumping a deep goal anchor).
        _prior_sig, _prior_reps = _apply_deep_inject(
            _anchor_sig, _anchor_reps, dv)
        rocket = minimal_ranks_for(
            name, _prior_sig, ctx, prior_reps=_prior_reps, **_bump_kw,
        )
        if rocket is not None:
            sig, reps = _apply_deep_inject(
                rocket.signature, rocket.reps_collected, dv)
            if reps is not rocket.reps_collected:
                # Re-derive a result carrying the injected reps so reps_collected
                # propagates into the keep-set + ladder.
                rocket = RankBumperResult(
                    signature=sig, delta=rocket.delta, reps=rocket.reps,
                    flags=rocket.flags, profile_dv=rocket.profile_dv,
                    reps_collected=reps,
                )
            for key, rep in rocket.reps.items():
                sphere_rank_reps[key] = rep
            marginal = Signature.of(rocket.signature.rank_reqs)
            location_signatures.setdefault(name, marginal)
            cumulative_reps = cumulative_reps | reps
            anchor_entries.append(
                (label, name, sig, reps, rocket, marginal))
            # Accumulate into the prior for the next (harder) anchor.
            _anchor_sig = _anchor_sig.merged_max(sig)
            _anchor_reps = _anchor_reps | reps
            continue
        # Goal-feasibility fallback (mirror the from-empty path ~3197-3230):
        # the chain-walk failed on this anchor, so fall back to its intrinsic
        # (from-empty) signature merged into the accumulated cumulative.  If even
        # the from-empty intrinsic is None and it isn't model-infeasible, raise.
        intrinsic = location_signatures.get(name)
        if intrinsic is None:
            r2 = minimal_ranks_for(
                name, Signature.empty(), ctx, prior_reps=frozenset(),
                **_bump_kw,
            )
            intrinsic = (Signature.of(r2.signature.rank_reqs)
                         if r2 is not None else None)
        if intrinsic is None:
            if name in infeasible:
                continue
            raise OptionError(
                f"Sphere ladder: predictable anchor {label} "
                f"({name!r}) is unreachable under any rank kit."
            )
        merged = _anchor_sig.merged_max(intrinsic)
        merged, m_reps = _apply_deep_inject(merged, _anchor_reps, dv)
        fb = RankBumperResult(
            signature=merged,
            delta=intrinsic,
            reps={},
            reps_collected=m_reps,
            flags=_pre_pass_for_ranks(
                merged, ctx,
                start_with_clamps=start_with_clamps,
                progressive_launch_pad=progressive_launch_pad,
                launch_pad_caps=world.mission_builder.launch_pad_caps,
                pad_tier=merged.counted(PROGRESSIVE_LAUNCH_PAD_NAME),
                precollected_names=precollected_names,
                buildings_in_logic=buildings_in_logic, home=bn_home,
            ),
            profile_dv=0.0,
        )
        location_signatures.setdefault(name, intrinsic)
        cumulative_reps = cumulative_reps | m_reps
        _anchor_sig = merged
        _anchor_reps = _anchor_reps | m_reps
        anchor_entries.append((label, name, merged, m_reps, fb, intrinsic))

    # ---- build the linear, monotonic-cumulative ladder ------------------
    # One SphereBoundary per walked mission + per predictable anchor, sorted by
    # cumulative rank-sum (tie-break _goal_dv).  provides = running union of all
    # cumulatives up to that sphere, so later.provides >= earlier.provides holds.
    ladder_entries: list[tuple[Signature, float, str, str, bool,
                               RankBumperResult]] = []
    # Walked missions.  Reconstruct a per-mission (label, name) for the boundary.
    seen_names: set[str] = set()
    for b in ordered:
        present = body_events.get(b.name.value)
        if not present:
            continue
        # Tie-break the event-order on the event value: FLYBY/SOI_LEAVE (both 0),
        # ORBIT/EVA_IN_ORBIT (both 1), LANDING/CREWED_LANDING/FLAG_PLANT (all 2)
        # share an order, and ``present`` is a set whose iteration is hash-
        # randomized per process.  Without the tie-break the shared rng threads
        # through these missions in a different order across processes -> the
        # bumper's rep picks (and thus the whole ladder) become non-reproducible.
        for ev in sorted(present, key=lambda e: (_GRAPH_WALK_EVENT_ORDER[e], e)):
            loc_name = locname_for[(b.name.value, ev)]
            info = _parse_location(loc_name)
            if info is None:
                continue
            mkey = _mission_key(info)
            res = mission_result.get(mkey)
            cum = mission_cumulative.get(mkey)
            if res is None or cum is None or loc_name in seen_names:
                continue
            seen_names.add(loc_name)
            cum_sig, _cum_reps = cum
            _dv = _goal_dv(loc_name, world.mission_builder)
            cum_sig, _cum_reps = _apply_deep_inject(cum_sig, _cum_reps, _dv)
            ladder_entries.append((
                cum_sig, _dv,
                f"S_mission[{loc_name}]", loc_name, False,
                RankBumperResult(
                    signature=cum_sig, delta=res.delta, reps=res.reps,
                    flags=res.flags, profile_dv=res.profile_dv,
                    reps_collected=_cum_reps),
            ))
    for label, name, cum_sig, reps, res, _marg in anchor_entries:
        if name in seen_names and not label.startswith("S_goal") \
                and not label.startswith("S_contract"):
            # A plain walked mission already covers this anchor's location.
            continue
        ladder_entries.append((
            cum_sig, _goal_dv(name, world.mission_builder),
            label, name, True,
            RankBumperResult(
                signature=cum_sig, delta=res.delta, reps=res.reps,
                flags=res.flags, profile_dv=res.profile_dv,
                reps_collected=reps),
        ))

    ladder_entries.sort(key=lambda e: (_ranksum(e[0]), e[1], e[3]))

    running_sig = Signature.empty()
    running_reps: set[str] = set()
    cumulative_sig = Signature.empty()
    for cum_sig, _dv, label, name, is_pred, res in ladder_entries:
        running_sig = running_sig.merged_max(cum_sig)
        running_reps |= set(res.reps_collected)
        cumulative_sig = running_sig
        ladder.spheres.append(SphereBoundary(
            name=label,
            location_name=name,
            is_predictable=is_pred,
            provides=running_sig,
            delta=res.delta,
            reps_collected=frozenset(running_reps),
            flags=res.flags,
            profile_dv=res.profile_dv,
            signature=ladder.location_signatures.get(name),
        ))

    cumulative_reps = cumulative_reps | frozenset(running_reps)
    return (ladder, location_signatures, cumulative_sig,
            cumulative_reps, sphere_rank_reps)


def apply_sphere_ladder(world: "KSP1World") -> None:
    """Build the rank-space sphere ladder for ``world`` and install
    fill-time placement guidance.  Phase 2 — the progressive walker has
    been retired.

    Scope:
      - Compute per-location capability Signature + LocationSignature.
      - Build the predictable ladder (S_launch / S_orbit / S_goal).
      - Pick random intermediate spheres.
      - Walk the combined chain in rank space, accumulating ranks +
        designated reps per ``(axis, rank)`` bump.
      - Install bootstrap-local on KSC / First Launch / starting-inv.
      - Install per-location rank-ceiling ``item_rule``.
      - Demote parts that exceed the chain's cumulative ceiling.
      - Register S_launch-admitted parts as ``local_early_items`` so AP
        prefers them on sphere-0 locations.
    """
    from .rocket_math import clear_find_optimal_stage_cache
    clear_find_optimal_stage_cache()
    # Identity-based pre-pass cache: clear so cross-seed flag objects
    # can't collide on id() after garbage collection.
    _RANK_PRE_PASS_CACHE.clear()
    ladder = SphereLadder()
    ctx = getattr(world, "_rank_context", DEFAULT_CONTEXT)
    difficulty = ["casual", "normal", "expert", "insane"][
        world.options.difficulty.value
    ]
    progressive_launch_pad = bool(world.options.progressive_launch_pad)
    buildings_in_logic = bool(world.options.buildings_in_logic)
    start_with_clamps = bool(world.options.start_with_launch_clamps)
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    home = str(world.mission_builder.home)
    bn_home = world.mission_builder.home  # BodyName (effects translation)
    infeasible = world.model_infeasible_locations

    # Per-location intrinsic rank ceilings.
    #
    # PERF: instead of running the full greedy bumper for every location
    # (~250 locations × 60+ iterations each), evaluate capability ONCE
    # at max ranks with full admit (the flags object is shared + cached),
    # then ask the optimizer per location what kit it used.  The kit's
    # parts' rank_sigs give the location's intrinsic ceiling — the
    # minimum ranks at which that mission becomes feasible.  This is the
    # same number the bumper converged to, reached in one optimizer call
    # instead of a long greedy walk.  Results are deduped by canonical
    # mission key (Mun Landing 1/2/3 share one mission → one eval).
    _max_ranks = Signature.of(Rank(a, max_rank_for(a)) for a in RankAxisKey)
    _pad_max = (len(world.mission_builder.launch_pad_caps) - 1
                if world.mission_builder.launch_pad_caps else 0)
    if buildings_in_logic:
        # Full-admit max: max the building levels too so the intrinsic
        # per-location query isn't false-failed by a building gate.
        _max_ranks = (_max_ranks
            .with_counted(PROGRESSIVE_VAB_NAME, PROGRESSIVE_VAB_COUNT)
            .with_counted(PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
                          PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT))
    _max_flags = _pre_pass_for_ranks(
        _max_ranks, ctx,
        start_with_clamps=start_with_clamps,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=world.mission_builder.launch_pad_caps,
        pad_tier=_pad_max,
        precollected_names=precollected_names,
        reps_only=None,
        buildings_in_logic=buildings_in_logic, home=bn_home,
    )
    _diff = DIFFICULTY_PROFILES[difficulty]

    # Dependency-ordered walk of the mission graph builds the sphere ladder and
    # its downstream-consumed outputs: per-location signatures, the cumulative
    # signature/reps, and the per-(axis, rank) reps.
    (ladder, location_signatures, cumulative_sig,
     cumulative_reps, sphere_rank_reps) = _build_ladder_graph_walk(
        world, ladder, ctx,
        difficulty=difficulty,
        progressive_launch_pad=progressive_launch_pad,
        start_with_clamps=start_with_clamps,
        buildings_in_logic=buildings_in_logic,
        precollected_names=precollected_names,
        home=home, bn_home=bn_home,
    )
    world._sphere_ladder = ladder
    world._sphere_rank_reps = sphere_rank_reps
    world._sphere_rank_cumulative = cumulative_sig

    # Science-instrument early cap (S_sci) — complete_tech_tree only.
    # The funding pass credits temperature/pressure instrument science on
    # every reachable body, but no flight mission requires an instrument,
    # so the bumper never makes them reps.  Inject the (non-precollected)
    # basic instruments into the reps of every sphere at/after ~45% of the
    # chain's dv range.  One move, via existing machinery:
    #   * counts them in the funding pass's bankable_science (sound funding
    #     — the deep MAX_TIER funding sphere now actually has the barometer
    #     the body_max_yield estimate assumed);
    #   * makes them proper reps, not orphan progression (the cheap access
    #     rules for >=45%-dv missions require them, so fill can't strand
    #     them — fixes the SSR/mun_flag regressions);
    #   * caps placement to the first ~45% of the run: the chain-ordering
    #     ban keeps them out of >=45%-dv locations, so they land somewhere
    #     in the first half (with variance — not jammed at sphere 0 like
    #     AP's early_items would do).
    if world.goal_spec.complete_tech_tree:
        _sci_inject = _BASIC_SCIENCE_INSTRUMENTS - precollected_names
        _sci_dvs = [s.signature.dv for s in ladder.spheres
                    if s.signature is not None]
        if _sci_inject and _sci_dvs:
            _sci_cap_dv = 0.45 * max(_sci_dvs)
            for _s in ladder.spheres:
                if _s.signature is not None and _s.signature.dv >= _sci_cap_dv:
                    _s.reps_collected = frozenset(_s.reps_collected) | _sci_inject
            # Keep the chain's cumulative rep set (used by the demote below
            # to decide PROGRESSION vs USEFUL) in sync with the injection —
            # otherwise the funding pass credits the instruments but the
            # demote marks them USEFUL, so they're never collected.
            cumulative_reps = cumulative_reps | _sci_inject

    # Tech-tree band funding: walk the chain accumulating science and
    # assign each tech tier a funding sphere.  Without this, R&D copies
    # have no chain-placement target and float to early bands.
    tech_sigs, tech_min_sigs, band_funding = (
        _compute_tech_tier_signatures_rank(world, ladder, ctx, location_signatures)
    )
    ladder.location_signatures.update(tech_sigs)
    # Mirror tech-tree capability signatures so the chain-ordering rule sees
    # rank + R&D / PSI requirements per tech location.
    location_signatures.update(tech_min_sigs)

    # Inject R&D into the chain at each funding sphere so subsequent spheres'
    # cumulative ``provides`` reflect "by sphere S, player has R&D=B".  Without
    # this the chain-ordering rule treats every R&D copy as post-goal and bans
    # them everywhere reachable.
    for band, funding_sphere in band_funding.items():
        # Find funding sphere's index in ladder.spheres.
        idx = None
        for i, s in enumerate(ladder.spheres):
            if s is funding_sphere:
                idx = i
                break
        if idx is None:
            continue
        cur = funding_sphere.delta.counted(PROGRESSIVE_RD_NAME)
        prior_at_funding = funding_sphere.provides.counted(PROGRESSIVE_RD_NAME) - cur
        new_count = max(band, funding_sphere.provides.counted(PROGRESSIVE_RD_NAME))
        funding_sphere.provides = funding_sphere.provides.with_counted(
            PROGRESSIVE_RD_NAME, new_count)
        funding_sphere.delta = funding_sphere.delta.with_counted(
            PROGRESSIVE_RD_NAME, new_count - prior_at_funding)
        # Propagate forward to all later sphere cumulatives.
        for later in ladder.spheres[idx + 1:]:
            later.provides = later.provides.with_counted(
                PROGRESSIVE_RD_NAME, new_count)

    # Bootstrap-local rule on starting-inv / KSC / First Launch.
    _install_bootstrap_local_rule(world)
    bootstrap_locations = set(world.location_builder.ksc_biome_names) | {f"{home} First Launch"}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name.startswith("Starting Inventory"):
            bootstrap_locations.add(loc.name)

    # Step A: pull the early "ungated" locations into the sphere system, so
    # every location is sphere-locked by the kit needed to reach it (the one
    # exception is Starting Inventory, pinned to sphere 0).  KSC science and
    # the first splashdown need a capsule; First Launch and Starting Inventory
    # sit at sphere 0.  Discarding them from ``bootstrap_locations`` lets the
    # unified sphere rule gate them (it skips the bootstrap set).
    _capsule_kit = Signature.empty().with_rank(RankAxisKey.CAPSULE, 1)
    _gate_early: dict[str, Signature] = {
        name: _capsule_kit for name in world.location_builder.ksc_biome_names
    }
    _gate_early[f"{home} First Launch"] = Signature.empty()
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name == "Splashdown":
            _gate_early[loc.name] = _capsule_kit
        elif loc.name.startswith("Starting Inventory"):
            _gate_early[loc.name] = Signature.empty()
    for _name, _need in _gate_early.items():
        location_signatures.setdefault(_name, _need)
        bootstrap_locations.discard(_name)

    chain_full_extras: dict[str, int] = {}
    for sphere in ladder.spheres:
        for c in sphere.provides.counted_reqs:
            chain_full_extras[c.kind] = max(
                chain_full_extras.get(c.kind, 0), c.level)
    # Inject the tech tree's PSI need at the chain level (mirrors the R&D band
    # injection above).  Only the complete_tech_tree goal funds its science
    # assuming PSI=PROGRESSIVE_PSI_COUNT (see _pick_tech_tree_anchors, gated on
    # the same goal_spec flag), but the bumper never bumps PSI into sphere
    # extras — so without this chain_full_extras[PSI] reads 0 and the spare-
    # demote drops the very PSI copies the tree was funded with, starving its
    # science.  (band_funding is the wrong gate: tech *nodes* are optional
    # checks present for every goal, so it's non-empty even for mun_flag.)
    if world.goal_spec.complete_tech_tree:
        from .items import (
            PROGRESSIVE_PSI_COUNT,
            PROGRESSIVE_SCIENCE_INSTRUMENT_NAME as _PSI_NAME,
        )
        chain_full_extras[_PSI_NAME] = max(
            chain_full_extras.get(_PSI_NAME, 0), PROGRESSIVE_PSI_COUNT,
        )
    # Expose the finalized per-location capability signatures for analysis.
    world._location_signatures = location_signatures

    # Access rule.  The feasibility bracket is computed in every mode (so the
    # unified placement rule's loc_sphere always uses the chain's oracle);
    # ``install_access`` only decides whether the cheap has_all(reps) rule is
    # swapped in:
    #   * ladder / strict_ladder — install the cheap bracket rule (strict_ladder
    #     additionally saves the raw capability rule so post_fill can re-prove
    #     the placement under physics and re-fill if the bracket ever diverged).
    #   * capability / strict_validation — leave the raw capability access rule
    #     in place, gating the whole seed on real physics for verification.
    ladder_mode = _ACCESS_RULE_MODE in ("ladder", "strict_ladder")
    _install_ladder_rules(
        world, ladder, location_signatures,
        bootstrap_locations,
        save_original=(_ACCESS_RULE_MODE == "strict_ladder"),
        install_access=ladder_mode,
    )
    # Single placement authority for every mode: the windowed sphere rule.
    _install_unified_sphere_rules(
        world, ladder, location_signatures,
        bootstrap_locations,
    )

    # Demote everything that isn't in the chain's collected reps set,
    # and clear past-goal rank entries so above-ceiling items can
    # scatter freely.  ``cumulative_reps`` is the union of all parts
    # the bumper admitted across the chain — under the bulk-admit
    # extension it includes every part at any (axis, rank) the chain
    # touched.
    rep_part_names = set(cumulative_reps)
    rep_part_names |= set(sphere_rank_reps.values())  # belt-and-suspenders
    # Goal-contract items gate Victory; they must stay PROGRESSION or the
    # beatability sweep (advancement-only) never collects them and the goal is
    # unreachable.  Non-goal pacing contracts may demote freely.
    rep_part_names |= {spec.item_name for spec in world.goal_contract_specs}
    # In count / progressive_unlock modes the player must COMPLETE X non-goal
    # contracts to unlock the goal (each emits a CONTRACT_COMPLETED event the
    # threshold counts), so those contracts are GOAL-PATH-REQUIRED.  Two classes
    # of item must therefore stay PROGRESSION or the threshold / Victory becomes
    # unreachable (AP only guarantees PROGRESSION items reachable; a USEFUL item
    # scatters anywhere, including past-goal bodies the chain never reaches):
    #
    #   (a) Each non-goal contract's GATE ITEM (its ``item_name``): the contract
    #       rule is ``state.has(gate_item) AND can_deliver``, so without the gate
    #       item collected the contract is never completable.  Demoting it to
    #       USEFUL strands it (observed: gate items landing on Ike / Moho / Pol
    #       returns, unreachable in a Mun-flag chain) → 0 contracts completable.
    #   (b) The required PARTS of those contracts: ``required_part_names_for``
    #       returns the lightest standalone rep per required category (drill /
    #       ore_tank / battery / science_lab) so the kept set is minimal; the
    #       chain-guaranteed payload reps (crew / relay / power) are already
    #       folded into cumulative_reps by _contract_payload_rep_names.
    from .contracts import required_part_names_for
    from .options import GoalContractMode
    if world.options.goal_contract_mode.value in (
            GoalContractMode.option_count,
            GoalContractMode.option_progressive_unlock):
        rep_part_names |= {spec.item_name for spec in world.contract_specs}
        rep_part_names |= required_part_names_for(world.contract_specs)
    _demote_non_rep_parts(world, rep_part_names, cumulative_sig,
                          chain_extras=chain_full_extras)
    # === DIAGNOSTIC (temporary, gated) ===
    import os as _os
    if not _os.environ.get('KSP_PHASE2_DIAG'):
        return
    try:
        with open('/home/nick/workspaces/ksp_ap/scratchpad/diag_compare.txt', 'w') as _dout:
            _dout.write(f'=== CHAIN ({len(ladder.spheres)} spheres) ===\n')
            for _i, _s in enumerate(ladder.spheres):
                _dout.write(f'  [{_i}] {_s.name} @ {_s.location_name}\n')
                _dout.write(f'      provides={list(_s.provides.reqs)}\n')
                _dout.write(f'      reps_collected ({len(_s.reps_collected)}):'
                            f' {sorted(_s.reps_collected)}\n')
                _dout.write(f'      bumper flags: has_launch={_s.flags.has_launch_engine}, '
                            f'has_vac={_s.flags.has_vacuum_engine}, '
                            f'engines={len(_s.flags.available_engines)}, '
                            f'tanks={len(_s.flags.available_tanks)}, '
                            f'srbs={len(_s.flags.available_srbs)}, '
                            f'capsule={_s.flags.has_capsule}, '
                            f'probe={_s.flags.has_probe_core}\n')
            # Show what FINAL sphere claims vs what player actually has.
            if ladder.spheres:
                _final = ladder.spheres[-1]
                _dout.write(f'\n=== FINAL SPHERE bumper claim ===\n')
                _dout.write(f'reps_collected: {sorted(_final.reps_collected)}\n')
                _dout.write(f'flags: has_launch={_final.flags.has_launch_engine}, '
                            f'has_vac={_final.flags.has_vacuum_engine}, '
                            f'engines={len(_final.flags.available_engines)}, '
                            f'tanks={len(_final.flags.available_tanks)}\n')
                _dout.write(f'engines: {[e.name for e in _final.flags.available_engines]}\n')
                _dout.write(f'tanks: {[(t.name, t.fuel_type) for t in _final.flags.available_tanks]}\n')
            # Pool composition.
            from BaseClasses import ItemClassification
            _prog = [_it for _it in world.multiworld.itempool
                     if _it.player == world.player
                     and _it.classification == ItemClassification.progression]
            _dout.write(f'\n=== POOL (after demote) ===\n')
            _dout.write(f'PROGRESSION items ({len(_prog)}):\n')
            for _it in _prog:
                _dout.write(f'  {_it.name}\n')
        return
    except Exception as _e:
        with open('/home/nick/workspaces/ksp_ap/scratchpad/diag_compare.txt', 'a') as _eout:
            import traceback
            _eout.write(f'DIAG_FAIL: {_e}\n{traceback.format_exc()}\n')
        return
    try:
        from BaseClasses import CollectionState, ItemClassification
        os.makedirs('/home/nick/workspaces/ksp_ap/scratchpad', exist_ok=True)
        with open('/home/nick/workspaces/ksp_ap/scratchpad/diag.txt', 'w') as _out:
            _out.write(f'=== CHAIN ({len(ladder.spheres)} spheres) ===\n')
            for _i, _s in enumerate(ladder.spheres):
                _out.write(f'  [{_i}] {_s.name} @ {_s.location_name}\n')
                _out.write(f'      provides={list(_s.provides.reqs)}\n')
            _pool = world.multiworld.itempool
            _by_class = {}
            for _it in _pool:
                if _it.player != world.player:
                    continue
                _by_class.setdefault(_it.classification, []).append(_it.name)
            for _c, _names in _by_class.items():
                _out.write(f'POOL[{_c.name}] {len(_names)} items\n')
            _state = CollectionState(world.multiworld)
            _locs = [_l for _l in world.multiworld.get_locations(world.player)
                     if _l.address is not None]
            _reach = sum(1 for _l in _locs if _l.can_reach(_state))
            _out.write(f'LOCS[reachable from precollected]={_reach}/{len(_locs)}\n')
            _prog = [_it for _it in _pool if _it.player == world.player
                     and _it.classification == ItemClassification.progression]
            _out.write(f'PROG_COUNT={len(_prog)}\n')
            for _it in _prog:
                _acc = sum(1 for _l in _locs if _l.item_rule(_it))
                _acc_r = sum(1 for _l in _locs
                             if _l.item_rule(_it) and _l.can_reach(_state))
                _out.write(f'  {_it.name!r:40} rule_ok={_acc:4d} '
                           f'reach_ok={_acc_r:4d}\n')
            # Sample: which locations does the BUMPER's first rep accept?
            _out.write('=== SAMPLE LOCATIONS (capability-gated, reachable) ===\n')
            _reach_locs = [_l for _l in _locs[:50] if _l.can_reach(_state)]
            for _l in _reach_locs[:20]:
                _out.write(f'  {_l.name}\n')
            _out.write(f'=== SPHERE REP PICKS (count={len(sphere_rank_reps)}) ===\n')
            for _k in sorted(sphere_rank_reps.keys(), key=lambda x: (x[0].value, x[1])):
                _out.write(f'  {_k[0].value}={_k[1]} -> {sphere_rank_reps[_k]!r}\n')
            _out.write(f'rep_part_names ({len(rep_part_names)}): {sorted(rep_part_names)}\n')
            # Simulate fill progressively to see what makes progress.
            _out.write('=== SIM: progressively add bumper reps to state ===\n')
            # Build a list of all PROGRESSION items in pool.
            _prog_list = list(_prog)
            # Sort: rank-bearing parts by rank-sum (low first), then R&D/Pad/PSI.
            def _key(it):
                _sig = getattr(it, "rank_sig", None)
                if _sig is None or not _sig.axes:
                    return (1, 999)
                return (0, sum(r for _, r in _sig.axes))
            _prog_list.sort(key=_key)
            _sim_state = CollectionState(world.multiworld)
            _step_reach = sum(1 for _l in _locs if _l.can_reach(_sim_state))
            _out.write(f'  step 0: reachable={_step_reach}\n')
            from .capability import compute_capability_from_items as _ccfi
            _diff_name = ["casual", "normal", "expert", "insane"][world.options.difficulty.value]
            for _i, _it in enumerate(_prog_list[:25], start=1):
                _sim_state.collect(_it, prevent_sweep=True)
                _step_reach = sum(1 for _l in _locs if _l.can_reach(_sim_state))
                if _i <= 5 or _i % 5 == 0:
                    _c, _f = _ccfi(
                        lambda n, _s=_sim_state, _p=world.player: _s.count(n, _p),
                        _diff_name,
                        bool(world.options.start_with_launch_clamps.value),
                        world.mission_builder,
                        progressive_launch_pad=bool(world.options.progressive_launch_pad.value),
                    )
                    _out.write(f'  step {_i:2d} +{_it.name!r}: reach={_step_reach}/'
                               f'{len(_locs)} engines={len(_f.available_engines)} '
                               f'tanks={len(_f.available_tanks)} '
                               f'sounding_km={_c.sounding_altitude_km:.0f}\n')
    except Exception as _e:
        with open('/home/nick/workspaces/ksp_ap/scratchpad/diag.txt', 'a') as _out:
            import traceback
            _out.write(f'DIAG_FAILED: {_e}\n{traceback.format_exc()}\n')
    # === END DIAGNOSTIC ===

    # Sphere-1 boost: items whose rank is admitted by S_launch's ceiling
    # AND that the bumper picked as designated reps at S_launch level
    # go on local-early so AP places them at sphere-0 locations.
    launch_sphere = next(
        (s for s in ladder.spheres if s.name == "S_launch"),
        None,
    )
    local_early = world.multiworld.local_early_items[world.player]
    if launch_sphere is not None:
        for (_axis, _rank), rep_name in sphere_rank_reps.items():
            if launch_sphere.provides.rank(_axis) >= _rank:
                local_early[rep_name] = max(local_early.get(rep_name, 0), 1)

