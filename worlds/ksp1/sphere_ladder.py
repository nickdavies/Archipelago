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
import logging
import os
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
)
from .parts import (
    CapabilityFlag,
    Engine,
    FuelTank,
    MiscEquipment,
    PART_DB,
    PROGRESSIVE_PART_COUNTS as _BASE_PROGRESSIVE_PART_COUNTS,
    PROGRESSIVE_PART_NAMES,
    PROGRESSIVE_PART_TIERS,
)
from .ranks import (
    DEFAULT_CONTEXT, RANK_AXES, RANK_AXES_BY_KEY, RankAxisKey, RankContext,
    max_rank_for, rank_sig_for, ranks_for_context,
)

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
    from .contracts import ContractSpec


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
        "Progressive LF Tank",
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
        "Progressive LF Tank",
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
class MinimumRanks:
    """Per-axis upper bound on admitted ranks for the rank-space sphere
    walker (Phase 1 scaffold).

    Stored as a sorted tuple of ``(RankAxisKey, int)`` for hashability.
    Absent axes are *unconstrained* — every item on that axis is admitted.
    The default ``empty()`` value therefore admits everything; the bumper
    *adds* axes as missions force constraints.

    Internal invariants:
      * ``upper_bounds`` is sorted by ``RankAxisKey.value`` (StrEnum value)
        so equality / hashing is canonical.
      * Each axis appears at most once.
    """
    upper_bounds: tuple[tuple[RankAxisKey, int], ...] = ()

    def get(self, axis: RankAxisKey) -> Optional[int]:
        for k, v in self.upper_bounds:
            if k == axis:
                return v
        return None

    def with_axis(self, axis: RankAxisKey, value: int) -> "MinimumRanks":
        """Return a new MinimumRanks with ``axis`` set to ``value`` (replace
        or insert).  No-op (returns ``self``) if the value is already set."""
        existing = self.get(axis)
        if existing == value:
            return self
        keep = [(k, v) for k, v in self.upper_bounds if k != axis]
        keep.append((axis, value))
        keep.sort(key=lambda x: x[0].value)
        return MinimumRanks(tuple(keep))

    def merged_max(self, other: "MinimumRanks") -> "MinimumRanks":
        """Element-wise max across the union of keys.  Missing-on-one-side
        is treated as unconstrained on that side, so the *other* side's
        value wins."""
        if not self.upper_bounds:
            return other
        if not other.upper_bounds:
            return self
        merged: dict[RankAxisKey, int] = {k: v for k, v in self.upper_bounds}
        for k, v in other.upper_bounds:
            existing = merged.get(k)
            merged[k] = v if existing is None else max(existing, v)
        return MinimumRanks(tuple(sorted(merged.items(), key=lambda x: x[0].value)))

    @classmethod
    def empty(cls) -> "MinimumRanks":
        return cls(())


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
    """A sphere in the rank-space ladder.

    ``ranks`` / ``ranks_delta`` track the cumulative rank ceiling and the
    increment over the prior sphere.  ``extras`` / ``extras_delta`` track
    counted progressives the bumper increments outside the rank model.
    ``reps_collected`` is the union of all bumper-selected reps through
    this sphere — used by chain-walker reps-only feasibility proofs and
    by downstream tech-tier band funding.
    """
    name: str
    location_name: str
    is_predictable: bool
    ranks: MinimumRanks
    ranks_delta: MinimumRanks
    reps_collected: frozenset[str] = frozenset()
    extras: dict[str, int] = field(default_factory=dict)
    extras_delta: dict[str, int] = field(default_factory=dict)
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
    # Contract-completion locations carry their ContractSpec; the delivery
    # payload is then sized per-rung from that rung's flags (via
    # ``contract_payload_parts``) so the ladder signature matches the runtime
    # access rule. ``None`` for ordinary (non-contract) missions.
    spec: Optional["ContractSpec"] = None


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
        "has_docking_port", "has_ladder",
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
    extra_payload: tuple = ()
    mission_transform = None
    if info.spec is not None:
        # Size the payload from the maxed kit's reps; the chosen crew/relay/power
        # parts surface in terminal_parts below, so the warm start seeds the
        # Progressive Capsule / Relay / Solar chains the contract needs.
        payload = contract_payload_parts(info.spec, max_flags)
        if payload is None:
            return {}  # max kit lacks a required chain part — no warm start
        extra_payload = payload
        # Match the runtime rule's edge modifier so the warm start is sized for
        # the real (transformed) mission, not the cheaper base orbit.
        mission_transform = info.spec.mission_transform(mission_builder)
    result = evaluate_mission_detailed(
        max_flags, diff, info.body, info.mission_type, info.crewed,
        mission_builder, threshold_km=info.threshold_km or 0.0,
        extra_payload_parts=extra_payload,
        mission_transform=mission_transform,
    )
    if not result.feasible:
        # Max kit can't reach this location at all — no warm start to give.
        return {}

    # Collect every part name referenced by the oracle's chosen build.
    used: set[str] = set()
    for sr in result.stage_results:
        if sr.engine_name and sr.engine_name != "(SRB integral)":
            used.add(sr.engine_name)
        for _count, tname in sr.tank_manifest:
            if tname and tname not in ("none", "(SRB integral)"):
                used.add(tname)
        if sr.heat_shield_name:
            used.add(sr.heat_shield_name)
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


# ---------------------------------------------------------------------------
# Rank-space sphere walker — Phase 1 scaffold.
#
# This subsystem mirrors the progressive-tier walker above using
# ``MinimumRanks`` ceilings keyed by ``RankAxisKey``.  It runs *alongside*
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


# BlockingReason → rank axes that could plausibly resolve it.  Mirrors
# ``_BUMP_TABLE`` above but keyed by ``RankAxisKey``.  Hash order is
# stable for cache hashing; per-priority-group selection happens in
# ``_RANK_PRIORITY_GROUPS``.
_RANK_BUMP_TABLE: dict[BlockingReason, tuple[RankAxisKey, ...]] = {
    BlockingReason.NO_VIABLE_STAGE: (
        RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK, RankAxisKey.XENON_TANK,
        RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE,
        RankAxisKey.STACK_DECOUPLER, RankAxisKey.SRB,
        RankAxisKey.RADIAL_DECOUPLER,
        RankAxisKey.CAPSULE, RankAxisKey.PROBE_SAS,
        RankAxisKey.SOLAR, RankAxisKey.RELAY,
        RankAxisKey.PARACHUTE, RankAxisKey.LANDING_LEG,
        RankAxisKey.HEAT_SHIELD, RankAxisKey.SAS,
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


# Ranked priority groups (parallels BUMP_PRIORITY_GROUPS).  The bumper
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


def _rank_group_index(axis: RankAxisKey) -> int:
    for i, group in enumerate(RANK_PRIORITY_GROUPS):
        if axis in group:
            return i
    return len(RANK_PRIORITY_GROUPS)


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


def _rank_admits_item(item_name: str, ranks: MinimumRanks, ctx: RankContext) -> bool:
    """Item is admitted iff every axis it participates in has the item's
    rank ≤ the ceiling on that axis.  Items with empty rank_sig (filler,
    non-ranked progressives) are admitted unconditionally."""
    sig = rank_sig_for(item_name, ctx)
    if not sig.axes:
        return True
    for axis_key, item_rank in sig.axes:
        cap = ranks.get(axis_key)
        if cap is None or item_rank > cap:
            return False
    return True


# Cache for ``_pre_pass_for_ranks`` — same shape as ``_pre_pass_cached``.
# Keyed by the hashable ``MinimumRanks.upper_bounds`` tuple plus options.
_RANK_PRE_PASS_CACHE: dict[tuple, EquipmentFlags] = {}


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
    ranks: MinimumRanks,
    ctx: RankContext,
    *,
    start_with_clamps: bool,
    progressive_launch_pad: bool,
    launch_pad_caps: tuple[float, ...] | None,
    pad_tier: int = 0,
    precollected_names: frozenset[str] = frozenset(),
    reps_only: Optional[frozenset[str]] = None,
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
        ranks.upper_bounds, ctx,
        start_with_clamps, progressive_launch_pad, launch_pad_caps,
        pad_tier, precollected_names, reps_only,
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

    def cf(name: str, _adm=admitted, _pad=pad_tier) -> int:
        if name == PROGRESSIVE_LAUNCH_PAD_NAME:
            return _pad
        return 1 if name in _adm else 0

    flags = _pre_pass(
        cf,
        start_with_clamps=start_with_clamps,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=launch_pad_caps,
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

    ``ranks`` / ``delta`` track the rank ceiling and increment.  ``extras``
    / ``extras_delta`` track the counted progressives (Pad / R&D / PSI)
    the bumper incremented outside the rank model — Pad is the only one
    the bumper itself touches today (LAUNCH_MASS_EXCEEDED → Pad bump);
    R&D / PSI are injected by the chain orchestrator via tech-tree
    band-funding logic.
    """
    ranks: MinimumRanks
    delta: MinimumRanks
    reps: dict[tuple[RankAxisKey, int], str]
    flags: EquipmentFlags
    profile_dv: float
    reps_collected: frozenset[str] = frozenset()
    extras: dict[str, int] = field(default_factory=dict)
    extras_delta: dict[str, int] = field(default_factory=dict)


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
    *, ranks: MinimumRanks, reps_collected: set, info, diff,
    start_with_clamps: bool, progressive_launch_pad: bool,
    launch_pad_caps, pad_tier: int, precollected_names: frozenset,
    mission_builder,
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
    base_ranks = ranks.with_axis(axis, new_rank)
    for cand in sorted(cands):
        trial_reps = set(reps_collected); trial_reps.add(cand)
        trial_ranks = base_ranks
        sig = rank_sig_for(cand, ctx)
        for co_axis, co_rank in sig.axes:
            if co_axis == axis:
                continue
            if (trial_ranks.get(co_axis) or 0) < co_rank:
                trial_ranks = trial_ranks.with_axis(co_axis, co_rank)
        trial_flags = _pre_pass_for_ranks(
            trial_ranks, ctx,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=launch_pad_caps,
            pad_tier=pad_tier,
            precollected_names=precollected_names,
            reps_only=frozenset(trial_reps),
        )
        trial_result = _evaluate(trial_flags, info, diff, mission_builder)
        feasibility = 0 if trial_result.feasible else 1
        mass = trial_result.launch_mass or float("inf")
        # Blocker reduction is the real "closer to feasible" signal;
        # mass-only sorts pick weak lightweight engines for infeasible
        # trials.  Mirror the same fix as in _pick_rank_bump_scored.
        scored.append((feasibility, len(trial_result.blocking), mass,
                       rng.random(), cand))
    scored.sort()
    return scored[0][-1]


def _rank_axis_at_cap(axis: RankAxisKey, ranks: MinimumRanks) -> bool:
    """Return True if ``axis`` ceiling already equals the axis's effective
    max rank (min(distinct scores, cap)) — no more bumps possible."""
    max_buckets = max_rank_for(axis)
    current = ranks.get(axis) or 0
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
    # payload mass.  The mass lever matters most on heavy-cascade ascents
    # (Moho/Pol/Eeloo sample return drag a 1000-3000 t terminal payload up
    # the gravity well): a lighter capsule / lighter support equipment
    # shrinks the cascade far more than another engine can lift it.  An
    # earlier narrow set here (thrust/fuel axes only) omitted the
    # payload-reducers and capped out fast, dumping those missions into the
    # rescue path.  Hand back the full NO_VIABLE_STAGE lever set and let the
    # scored picker trial-evaluate which one actually closes the gap.
    return _RANK_BUMP_TABLE[BlockingReason.NO_VIABLE_STAGE]


def _pick_rank_bump(blocking, ranks: MinimumRanks, rng: Random) -> Optional[RankAxisKey]:
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
    prior_ranks: MinimumRanks,
    ctx: RankContext,
    difficulty: str,
    progressive_launch_pad: bool,
    start_with_clamps: bool,
    rng: Random,
    mission_builder: MissionBuilder,
    prior_extras: Optional[dict[str, int]] = None,
    prior_reps: frozenset[str] = frozenset(),
    precollected_names: frozenset[str] = frozenset(),
    max_iterations: int = 500,
    reps_only_mode: bool = True,
) -> Optional[RankBumperResult]:
    """Rank-space sphere walker (Phase 1 scaffold).

    Greedy bumper: starting from ``prior_ranks``, repeatedly bump an axis
    ceiling by 1 until the mission becomes feasible.  Each bump records a
    designated rep (a random newly-admitted part) in ``RankBumperResult.reps``.

    Returns ``None`` if the location isn't capability-gated (tech-tree
    biome / starting inventory) OR if no kit reaches it within
    ``max_iterations`` bumps.
    """
    info = _parse_location(location_name)
    if info is None:
        return None
    diff = DIFFICULTY_PROFILES[difficulty]
    ranks = prior_ranks
    extras = dict(prior_extras or {})
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
            ranks, ctx,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
            precollected_names=precollected_names,
            reps_only=frozenset(reps_collected) if reps_only_mode else None,
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
                    pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                    precollected_names=precollected_names,
                )
                kept = set(reps_collected)
                _changed = True
                while _changed:
                    _changed = False
                    for _rep in sorted(kept - set(prior_reps)):
                        _trial = kept - {_rep}
                        _tf = _pre_pass_for_ranks(
                            ranks, ctx, reps_only=frozenset(_trial), **_pp)
                        if _evaluate(_tf, info, diff, mission_builder).feasible:
                            kept = _trial
                            _changed = True
                if kept != reps_collected:
                    reps_collected = kept
                    _new = prior_ranks
                    for _rep in reps_collected:
                        for _ax, _rk in rank_sig_for(_rep, ctx).axes:
                            if _rk > (_new.get(_ax) or 0):
                                _new = _new.with_axis(_ax, _rk)
                    ranks = _new
                    reps = {k: v for k, v in reps.items() if v in reps_collected}
                    flags = _pre_pass_for_ranks(
                        ranks, ctx, reps_only=frozenset(reps_collected), **_pp)
            delta_pairs: list[tuple[RankAxisKey, int]] = []
            for axis_key, ceil in ranks.upper_bounds:
                prior = prior_ranks.get(axis_key) or 0
                if ceil > prior:
                    delta_pairs.append((axis_key, ceil))
            delta = MinimumRanks(tuple(sorted(delta_pairs, key=lambda x: x[0].value)))
            prior_extras_d = dict(prior_extras or {})
            extras_delta = {
                k: v - prior_extras_d.get(k, 0)
                for k, v in extras.items()
                if v > prior_extras_d.get(k, 0)
            }
            return RankBumperResult(
                ranks=ranks,
                delta=delta,
                reps=reps,
                reps_collected=frozenset(reps_collected),
                flags=flags,
                profile_dv=result.launch_mass,
                extras=extras,
                extras_delta=extras_delta,
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
            cur_pad = extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0)
            if cur_pad < pad_cap_count:
                extras[PROGRESSIVE_LAUNCH_PAD_NAME] = cur_pad + 1
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
                    and p.size_class >= sd.min_tank_size_needed
                ]
                if not valid:
                    continue
                pick = rng.choice(sorted(valid))
                reps_collected.add(pick)
                for _ax, _rk in rank_sig_for(pick, ctx).axes:
                    if (_ax, _rk) not in reps:
                        reps[(_ax, _rk)] = pick
                    if _rk > (ranks.get(_ax) or 0):
                        ranks = ranks.with_axis(_ax, _rk)
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
                    if _rk > (ranks.get(_ax) or 0):
                        ranks = ranks.with_axis(_ax, _rk)
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
            result.blocking, ranks, ctx, rng,
            difficulty=difficulty,
            progressive_launch_pad=progressive_launch_pad,
            start_with_clamps=start_with_clamps,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
            precollected_names=precollected_names,
            mission_builder=mission_builder,
            info=info,
            diff=diff,
            reps_collected=reps_collected,
            reps_only_mode=reps_only_mode,
        )
        if axis is None:
            # Fall back to the catchall candidate set if stage_diag-targeted
            # axes are exhausted (e.g. non-NO_VIABLE_STAGE blockers only).
            axis = _pick_rank_bump(result.blocking, ranks, rng)
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
                f"{s.engine_count}x{s.engine_name}+{s.tank_count}tk "
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
            max_ranks_for_rescue = MinimumRanks(tuple(sorted(
                ((a, max_rank_for(a)) for a in RankAxisKey),
                key=lambda x: x[0].value,
            )))
            rescue_flags = _pre_pass_for_ranks(
                max_ranks_for_rescue, ctx,
                start_with_clamps=start_with_clamps,
                progressive_launch_pad=progressive_launch_pad,
                launch_pad_caps=mission_builder.launch_pad_caps,
                pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                precollected_names=precollected_names,
                reps_only=None,
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
                lifted_ranks = ranks
                for u in variant_parts:
                    sig = rank_sig_for(u, ctx)
                    for ax, rk in sig.axes:
                        cur = lifted_ranks.get(ax) or 0
                        if rk > cur:
                            lifted_ranks = lifted_ranks.with_axis(ax, rk)
                union_reps = reps_collected | variant_parts
                verify_flags = _pre_pass_for_ranks(
                    lifted_ranks, ctx,
                    start_with_clamps=start_with_clamps,
                    progressive_launch_pad=progressive_launch_pad,
                    launch_pad_caps=mission_builder.launch_pad_caps,
                    pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                    precollected_names=precollected_names,
                    reps_only=frozenset(union_reps),
                )
                verify_result = _evaluate(verify_flags, info, diff, mission_builder)
                kit_parts = variant_parts
                if not verify_result.feasible:
                    # Variant broke a cascade.  Fall back to the
                    # deterministic optimal kit and re-verify.
                    kit_parts = kit.all_parts()
                    lifted_ranks = ranks
                    for u in kit_parts:
                        sig = rank_sig_for(u, ctx)
                        for ax, rk in sig.axes:
                            cur = lifted_ranks.get(ax) or 0
                            if rk > cur:
                                lifted_ranks = lifted_ranks.with_axis(ax, rk)
                    union_reps = reps_collected | kit_parts
                    verify_flags = _pre_pass_for_ranks(
                        lifted_ranks, ctx,
                        start_with_clamps=start_with_clamps,
                        progressive_launch_pad=progressive_launch_pad,
                        launch_pad_caps=mission_builder.launch_pad_caps,
                        pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                        precollected_names=precollected_names,
                        reps_only=frozenset(union_reps),
                    )
                    verify_result = _evaluate(verify_flags, info, diff, mission_builder)
                if verify_result.feasible:
                    for u in kit_parts:
                        if u not in reps_collected:
                            reps_collected.add(u)
                            sig = rank_sig_for(u, ctx)
                            for ax, rk in sig.axes:
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
                            pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                            precollected_names=precollected_names,
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
                            lifted_ranks = prior_ranks
                            for _rep in reps_collected:
                                for _ax, _rk in rank_sig_for(_rep, ctx).axes:
                                    if _rk > (lifted_ranks.get(_ax) or 0):
                                        lifted_ranks = lifted_ranks.with_axis(_ax, _rk)
                            reps = {k: v for k, v in reps.items()
                                    if v in reps_collected}
                            verify_flags = _pre_pass_for_ranks(
                                lifted_ranks, ctx,
                                reps_only=frozenset(reps_collected), **_rpp)
                            verify_result = _evaluate(
                                verify_flags, info, diff, mission_builder)
                    delta_pairs: list[tuple[RankAxisKey, int]] = []
                    for axis_key, ceil in lifted_ranks.upper_bounds:
                        prior = prior_ranks.get(axis_key) or 0
                        if ceil > prior:
                            delta_pairs.append((axis_key, ceil))
                    delta = MinimumRanks(tuple(sorted(delta_pairs, key=lambda x: x[0].value)))
                    prior_extras_d = dict(prior_extras or {})
                    extras_delta = {
                        k: v - prior_extras_d.get(k, 0)
                        for k, v in extras.items()
                        if v > prior_extras_d.get(k, 0)
                    }
                    return RankBumperResult(
                        ranks=lifted_ranks, delta=delta, reps=reps,
                        reps_collected=frozenset(reps_collected),
                        flags=verify_flags,
                        profile_dv=verify_result.launch_mass,
                        extras=extras, extras_delta=extras_delta,
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
                            ranks, ctx,
                            start_with_clamps=start_with_clamps,
                            progressive_launch_pad=progressive_launch_pad,
                            launch_pad_caps=mission_builder.launch_pad_caps,
                            pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                            precollected_names=precollected_names,
                            reps_only=frozenset(new_reps),
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
                        ranks, ctx,
                        start_with_clamps=start_with_clamps,
                        progressive_launch_pad=progressive_launch_pad,
                        launch_pad_caps=mission_builder.launch_pad_caps,
                        pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                        precollected_names=precollected_names,
                        reps_only=frozenset(reps_collected),
                    )
                    delta_pairs: list[tuple[RankAxisKey, int]] = []
                    for axis_key, ceil in ranks.upper_bounds:
                        prior = prior_ranks.get(axis_key) or 0
                        if ceil > prior:
                            delta_pairs.append((axis_key, ceil))
                    delta = MinimumRanks(tuple(sorted(delta_pairs, key=lambda x: x[0].value)))
                    prior_extras_d = dict(prior_extras or {})
                    extras_delta = {
                        k: v - prior_extras_d.get(k, 0)
                        for k, v in extras.items()
                        if v > prior_extras_d.get(k, 0)
                    }
                    return RankBumperResult(
                        ranks=ranks, delta=delta, reps=reps,
                        reps_collected=frozenset(reps_collected),
                        flags=final_flags, profile_dv=m,
                        extras=extras, extras_delta=extras_delta,
                    )
            return None
        new_rank = (ranks.get(axis) or 0) + 1
        # Pick the part at this (axis, rank) that actually clears the most
        # blockers, not a random one — a random low-rank pick is often too
        # weak, forcing the bumper to over-raise the rank (orbit "needs" vac:5,
        # Pol "needs" launch:5 when a good rank-2 part flies it).
        rep_name = _pick_rank_rep_scored(
            axis, new_rank, ctx, rng,
            ranks=ranks, reps_collected=reps_collected, info=info, diff=diff,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=mission_builder.launch_pad_caps,
            pad_tier=extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
            precollected_names=precollected_names,
            mission_builder=mission_builder,
        )
        if rep_name is not None:
            reps[(axis, new_rank)] = rep_name
            reps_collected.add(rep_name)
        ranks = ranks.with_axis(axis, new_rank)
        # Co-axis lift.
        if rep_name is not None:
            sig = rank_sig_for(rep_name, ctx)
            for co_axis, co_rank in sig.axes:
                if co_axis == axis:
                    continue
                current = ranks.get(co_axis) or 0
                if co_rank > current:
                    ranks = ranks.with_axis(co_axis, co_rank)
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



def _pick_rank_bump_scored(blocking, ranks: MinimumRanks, ctx: RankContext,
                           rng: Random, *,
                           difficulty: str,
                           progressive_launch_pad: bool,
                           start_with_clamps: bool,
                           launch_pad_caps,
                           pad_tier: int,
                           precollected_names: frozenset[str],
                           mission_builder, info, diff,
                           reps_collected: Optional[set[str]] = None,
                           reps_only_mode: bool = True) -> Optional[RankAxisKey]:
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
    scored: list[tuple[int, float, int, int, float, RankAxisKey]] = []
    # Iterate in a deterministic, priority-group-aware order: lower
    # priority group first, then axis value within group.  ``set``
    # iteration is hash-randomized per Python process under the default
    # PYTHONHASHSEED — different solve-check workers were taking
    # different bumper paths on the same seed because each consumed the
    # ``rng`` in a different order.
    def _cand_sort_key(a: RankAxisKey) -> tuple[int, str]:
        for i, g in enumerate(RANK_PRIORITY_GROUPS):
            if a in g:
                return (i, a.value)
        return (len(RANK_PRIORITY_GROUPS), a.value)
    for cand in sorted(candidates, key=_cand_sort_key):
        new_rank = (ranks.get(cand) or 0) + 1
        trial_ranks = ranks.with_axis(cand, new_rank)
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
                    if (trial_ranks.get(_co_axis) or 0) < _co_rank:
                        trial_ranks = trial_ranks.with_axis(_co_axis, _co_rank)
            trial_reps_set = frozenset(trial_reps)
        trial_flags = _pre_pass_for_ranks(
            trial_ranks, ctx,
            start_with_clamps=start_with_clamps,
            progressive_launch_pad=progressive_launch_pad,
            launch_pad_caps=launch_pad_caps,
            pad_tier=pad_tier,
            precollected_names=precollected_names,
            reps_only=trial_reps_set,
        )
        trial_result = _evaluate(trial_flags, info, diff, mission_builder)
        feasibility_rank = 0 if trial_result.feasible else 1
        mass = trial_result.launch_mass or float("inf")
        group_idx = len(RANK_PRIORITY_GROUPS)
        for i, g in enumerate(RANK_PRIORITY_GROUPS):
            if cand in g:
                group_idx = i
                break
        # Sort order: feasibility > blocker reduction > mass.  Putting
        # n_blockers before mass: for infeasible candidates ``launch_mass``
        # comes back as the *payload* mass from stage failures, so a
        # tiny weak engine (ionEngine: 0.25 t) "looks lighter" than a
        # strong heavy engine (LV-T91: 4 t) and gets picked, leaving
        # the rep set unable to deliver hard missions like Moho / Bop.
        # Blocker-reduction is the real "closer to feasible" signal.
        scored.append((
            feasibility_rank, len(trial_result.blocking), mass,
            group_idx, rng.random(), cand,
        ))
    scored.sort()

    return scored[0][-1]


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
        info.spec.contract_type if info.spec else None,  # distinguishes contract payloads
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


def _inject_into_rocket(
    rocket: "MinimalRocket",
    inject: dict[str, int],
    prior_cumulative: dict[str, int],
) -> "MinimalRocket":
    """Return a copy of ``rocket`` with bookkeeping items merged in.

    Bookkeeping items (Progressive R&D, Progressive Science Instrument) are
    not bumped by the physics-driven bumper but must be chain-tracked.  This
    helper updates ``cumulative`` (element-wise max with the injection) and
    ``delta`` (the increase relative to ``prior_cumulative``) so Rule B and
    chain_required see the items naturally.
    """
    import dataclasses
    new_cum = dict(rocket.cumulative)
    new_delta = dict(rocket.delta)
    for name, count in inject.items():
        merged = max(new_cum.get(name, 0), count)
        new_cum[name] = merged
        new_in_delta = max(0, merged - prior_cumulative.get(name, 0))
        if new_in_delta > 0:
            new_delta[name] = max(new_delta.get(name, 0), new_in_delta)
    return dataclasses.replace(rocket, delta=new_delta, cumulative=new_cum)


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
    if info.spec is not None:
        # Apply the contract's edge modifier (polar/stationary at home) so the
        # ordering dv matches the runtime rule, not the base orbit.
        transform = info.spec.mission_transform(mission_builder)
        profiles = [transform(profile) for profile in profiles]
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
    return out


# Items injected into the cumulative kit of tech-tree anchor spheres.
# These are bookkeeping items (gate tech-tree access, not rocket physics);
# the chain walker merges them into the rocket's delta+cumulative after
# ``minimal_rocket_for`` builds the physics part.  Result: chain_required
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
    from .capability import compute_capability_from_items
    from .locations import TechTreeLocation, effective_tech_slots_per_node
    from .rules import bankable_science, effective_science_safety
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
    #
    # Uses ``bankable_science`` — the same gated computation the victory
    # rule uses.  If the two diverged, the ladder could mark a tier as
    # funded by a sphere where the rule sees zero science (e.g. a body
    # in orbit but with no relay or recover path), producing seeds the
    # rule rejects at fill time.
    home = world.mission_builder.home
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
        psi_tier = kit.get("Progressive Science Instrument", 0)
        sphere_science.append((sphere, bankable_science(cap, psi_tier, home) * safety))

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
    extra_names = set(world.location_builder.ksc_biome_names) | {f"{home} First Launch"}
    for loc in world.multiworld.get_locations(player):
        if loc.name in extra_names:
            loc.item_rule = local_only


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _compute_location_rank_ceilings(
    world: "KSP1World",
) -> dict[str, MinimumRanks]:
    """Compute the intrinsic per-location MinimumRanks ceiling.

    For each capability-gated player location, runs ``minimal_ranks_for``
    with an empty prior — the result is the minimum kit the rank-space
    bumper needs to satisfy that location.  Items whose rank exceeds any
    of these ceilings on any axis must not be placed at that location.

    Non-capability-gated locations (tech-tree, KSC biomes, starting
    inventory) are excluded — they're gated by other mechanisms.
    """
    ctx = getattr(world, "_rank_context", DEFAULT_CONTEXT)
    difficulty = ["casual", "normal", "expert", "insane"][
        world.options.difficulty.value
    ]
    pad_on = bool(world.options.progressive_launch_pad)
    clamps = bool(world.options.start_with_launch_clamps)
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    infeasible = world.model_infeasible_locations
    out: dict[str, MinimumRanks] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name in infeasible:
            continue
        info = _parse_location(loc.name)
        if info is None:
            continue
        rocket = minimal_ranks_for(
            loc.name, MinimumRanks.empty(), ctx,
            difficulty=difficulty,
            progressive_launch_pad=pad_on,
            start_with_clamps=clamps,
            rng=world.random,
            mission_builder=world.mission_builder,
            precollected_names=precollected_names,
        )
        if rocket is None:
            continue
        out[loc.name] = rocket.ranks
    return out


def _install_rank_ceiling_rule(
    world: "KSP1World",
    ceilings: dict[str, MinimumRanks],
    bootstrap_locations: set[str],
) -> None:
    """Per-location item_rule: admit only items whose rank on every axis
    is ≤ the location's ceiling on that axis.

    Composes with the location's existing item_rule (if any) by AND-ing.
    Bootstrap locations (KSC biomes, starting inv, First Launch) skip
    the rule — they're handled by local-only restrictions.
    """
    player = world.player
    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        ceiling = ceilings.get(loc.name)
        if ceiling is None:
            continue
        ceiling_dict = dict(ceiling.upper_bounds)
        existing = loc.item_rule

        def _rule(item, _ceil=ceiling_dict, _p=player, _orig=existing) -> bool:
            if _orig is not None and not _orig(item):
                return False
            if item.player != _p:
                return True
            sig = getattr(item, "rank_sig", None)
            if sig is None or not sig.axes:
                return True
            for axis_key, item_rank in sig.axes:
                cap = _ceil.get(axis_key)
                if cap is None or item_rank > cap:
                    return False
            return True

        loc.item_rule = _rule


def _demote_non_rep_parts(
    world: "KSP1World",
    rep_names: set[str],
    chain_cumulative: MinimumRanks,
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
    )
    from .ranks import ItemRankSig
    _KEEP_PROGRESSIVE: frozenset[str] = frozenset({
        PROGRESSIVE_RD_NAME,
        PROGRESSIVE_LAUNCH_PAD_NAME,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME,
    })
    ceiling = dict(chain_cumulative.upper_bounds)
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
        # (1a) Spare Launch Pad copies.  A pad copy gates progression only up
        # to the max tier the chain actually bumped (``chain_extras[Pad]``).
        # Copies beyond that gate nothing, but keeping them PROGRESSION clogs
        # the restrictive fill: a spare tier-3 pad with no reachable home left
        # aborts the whole fill ("No more spots to place").  Demote those
        # spares to USEFUL so they scatter freely; the chain-needed tiers stay
        # PROGRESSION.  (R&D / PSI keep the blanket rule -- their tier needs
        # aren't fully captured in extras, so demoting them risks solvability.)
        if item.name == PROGRESSIVE_LAUNCH_PAD_NAME and chain_extras is not None:
            tier = getattr(item, "_sphere_tier", None)
            if tier is not None and tier > chain_extras.get(item.name, 0):
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


def _record_rank_sphere_reps(world: "KSP1World", ladder: SphereLadder) -> None:
    """Walk the predictable sphere anchors in rank-space, recording the
    designated rep part name for every ``(RankAxisKey, rank)`` bump the
    bumper performs.

    Stores two attributes on the world:
      * ``_sphere_rank_reps``: ``dict[(RankAxisKey, int), str]`` — every
        rep encountered across the whole chain, latest one wins on
        conflict (ladder is walked S_launch → S_orbit → S_goal in that
        order, so later anchors override earlier).  Future phases will
        want a per-sphere structure; for now the flat dict is enough.
      * ``_sphere_rank_cumulative``: ``MinimumRanks`` — final ceiling
        after walking every predictable anchor.

    Defensive: any per-anchor failure is logged and skipped — the
    progressive walker is the placement authority in Phase 1.
    """
    home_atmo_bodies = {
        BodyName.KERBIN.value, BodyName.EVE.value,
        BodyName.DUNA.value, BodyName.LAYTHE.value,
    }
    ctx = RankContext(
        home_has_atmosphere=str(world.mission_builder.home) in home_atmo_bodies,
    )
    difficulty = ["casual", "normal", "expert", "insane"][
        world.options.difficulty.value
    ]
    progressive_launch_pad = bool(world.options.progressive_launch_pad)
    start_with_clamps = bool(world.options.start_with_launch_clamps)
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    reps: dict[tuple[RankAxisKey, int], str] = {}
    cumulative = MinimumRanks.empty()
    # Walk predictable spheres only (S_launch, S_orbit, S_goal*).
    # Intermediates depend on a chain that the rank walker doesn't fully
    # model yet (tech tree band funding, KSC biome locations); revisit
    # in Phase 2 once the rank walker is the source of truth.
    for sphere in ladder.spheres:
        if not sphere.is_predictable:
            continue
        rocket = minimal_ranks_for(
            sphere.location_name,
            cumulative,
            ctx,
            difficulty=difficulty,
            progressive_launch_pad=progressive_launch_pad,
            start_with_clamps=start_with_clamps,
            rng=world.random,
            mission_builder=world.mission_builder,
            precollected_names=precollected_names,
        )
        if rocket is None:
            # Non-fatal: scaffold can't reach this anchor.  Phase 1's
            # progressive walker still placed items correctly; this just
            # means the rank-space view is incomplete for this seed.
            continue
        for key, rep in rocket.reps.items():
            reps[key] = rep
        cumulative = rocket.ranks
    world._sphere_rank_reps = reps
    world._sphere_rank_cumulative = cumulative


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


def _make_bracket_rule(player: int, reps: tuple, extras: tuple):
    """Cheap reachability rule: the player has collected every
    PROGRESSION rep of the bracket sphere and met its counted-progressive
    thresholds.  Microsecond has/count checks — no capability physics."""
    def rule(state) -> bool:
        for r in reps:
            if not state.has(r, player):
                return False
        for name, count in extras:
            if state.count(name, player) < count:
                return False
        return True
    return rule


def _install_ladder_rules(
    world: "KSP1World",
    ladder: SphereLadder,
    location_min_ranks: dict[str, MinimumRanks],
    location_min_extras: dict[str, dict[str, int]],
    bootstrap_locations: set,
    save_original: bool = False,
) -> None:
    """Install BOTH the cheap access rule and the placement item_rule for
    every capability-gated location, driven by a SINGLE capability
    bracket — so access and placement can't disagree (the FillError-
    causing inconsistency the prototype surfaced).

    For each location L:
      * **Bracket** ``j`` = first chain sphere whose reps-only flags
        actually reach L's mission (capability, not rank coverage —
        deduped per mission so the scan runs ~40 times, not ~250).
      * **Access rule**: ``state.has_all(sphere[j].reps_collected)`` plus
        the sphere's counted-progressive thresholds.  L is reachable
        once the player holds the kit that the ladder proved reaches it.
      * **Placement item_rule**: a part is admitted at L iff its rank is
        within the PRIOR sphere's ceiling (``sphere[j-1].ranks``; empty
        for j=0).  This bans exactly the reps that unlock L's bracket
        sphere from landing at L — they must be collected earlier, which
        is what makes the cheap access rule cycle-free.  USEFUL parts are
        gated the same way, so a high-rank part still can't land before
        its sphere (the power-engine-too-early concern).
    """
    player = world.player
    spheres = ladder.spheres
    diff = DIFFICULTY_PROFILES[
        ["casual", "normal", "expert", "insane"][world.options.difficulty.value]
    ]
    mb = world.mission_builder

    # strict_ladder: keep the original capability access rule per
    # location so post_fill can swap it back in and independently
    # re-verify the cheap-rule fill is winnable under real capability.
    saved: dict[str, object] = {}
    bracket_by_mission: dict[tuple, Optional[int]] = {}
    # Per-location feasibility bracket (first sphere whose cumulative kit can
    # FLY the mission).  This is the single source of truth for a mission's
    # sphere — the placement rule reuses it instead of re-deriving via
    # rank-vector domination, which diverges from the chain (see
    # _install_unified_sphere_rules).
    bracket_by_loc: dict[str, int] = {}
    rebracketed = 0
    for loc in world.multiworld.get_locations(player):
        if loc.address is None or loc.name in bootstrap_locations:
            continue
        if loc.name not in location_min_ranks:
            continue  # not capability-gated (proxy) — leave rule
        info = _parse_location(loc.name)
        if info is None:
            continue  # tech anchor — gates on science, not capability
        mkey = (info.body, info.mission_type, info.crewed, info.threshold_km)
        if mkey in bracket_by_mission:
            j = bracket_by_mission[mkey]
        else:
            j = None
            for i, s in enumerate(spheres):
                if s.flags is not None and _evaluate(s.flags, info, diff, mb).feasible:
                    j = i
                    break
            bracket_by_mission[mkey] = j
        if j is None:
            # No sphere reaches this mission with its reps-only kit —
            # leave the capability rule as the (slow) fallback.
            continue
        bracket_by_loc[loc.name] = j
        if save_original:
            saved[loc.name] = loc.access_rule
        sphere = spheres[j]
        # Access rule: has the bracket sphere's full cumulative kit.
        loc.access_rule = _make_bracket_rule(
            player,
            tuple(sphere.reps_collected),
            tuple(sphere.extras.items()),
        )
        # No item_rule ban here.  The chicken-and-egg (a rep needed to reach
        # L sitting at L) is prevented by AP's restrictive fill, which never
        # places a progression item at a location unreachable without it —
        # the same protection strict_validation relies on.  Placement balance
        # is the unified sphere rule's job (_install_unified_sphere_rules).
        rebracketed += 1
    world._cheap_access_rebracketed = rebracketed
    world._cheap_access_bracket = bracket_by_loc
    if save_original:
        world._strict_ladder_saved_rules = saved


def _compute_location_priors(
    ladder: SphereLadder,
    location_min_ranks: dict[str, MinimumRanks],
    location_min_extras: dict[str, dict[str, int]],
    location_names_in_chain: set[str],
) -> tuple[dict[str, MinimumRanks], dict[str, dict[str, int]]]:
    """For each capability-gated location L, return ``(prior_ranks,
    prior_extras)`` — the cumulative kit at the chain sphere JUST BEFORE
    L becomes reachable.  Items belonging to that sphere or later
    (rank-delta items, R&D / Pad copies introduced there) must be banned
    at L: collecting them at L would be a chicken-and-egg dependency.

    Locations not covered by any sphere (post-goal / unfundable tech)
    are treated with the full chain cumulative as their prior — i.e.,
    only chain-irrelevant items can land at them.  This mirrors the
    legacy "sentinel-beyond-goal" path.
    """
    # Build per-sphere prior cumulatives (prior_index 0 = empty kit;
    # prior_index i = sphere_(i-1)'s cumulative).
    prior_ranks_per: list[MinimumRanks] = [MinimumRanks.empty()]
    prior_extras_per: list[dict[str, int]] = [{}]
    for sphere in ladder.spheres:
        prior_ranks_per.append(sphere.ranks)
        prior_extras_per.append(dict(sphere.extras))
    # Full chain cumulative for sentinel fallback.
    if ladder.spheres:
        full_ranks = ladder.spheres[-1].ranks
        full_extras = dict(ladder.spheres[-1].extras)
    else:
        full_ranks = MinimumRanks.empty()
        full_extras = {}
    loc_prior_ranks: dict[str, MinimumRanks] = {}
    loc_prior_extras: dict[str, dict[str, int]] = {}
    for loc_name, min_ranks in location_min_ranks.items():
        min_extras = location_min_extras.get(loc_name, {})
        found_idx: Optional[int] = None
        for i, sphere in enumerate(ladder.spheres):
            covers_ranks = all(
                (sphere.ranks.get(axis_key) or 0) >= rank
                for axis_key, rank in min_ranks.upper_bounds
            )
            covers_extras = all(
                sphere.extras.get(name, 0) >= count
                for name, count in min_extras.items()
            )
            if covers_ranks and covers_extras:
                found_idx = i
                break
        if found_idx is None:
            # Unreachable in this chain — sentinel: only items NOT used
            # anywhere in the chain can land here, so prior is the
            # full chain cumulative.
            loc_prior_ranks[loc_name] = full_ranks
            loc_prior_extras[loc_name] = dict(full_extras)
        else:
            loc_prior_ranks[loc_name] = prior_ranks_per[found_idx]
            loc_prior_extras[loc_name] = prior_extras_per[found_idx]
    return loc_prior_ranks, loc_prior_extras


def _make_placement_rule(player, ranks_dict, extras_dict, chain_full_extras,
                         existing, gate_parts: bool = True):
    """Rank-ceiling (PART items) + counted-progressive chain-ordering rule.

    ``gate_parts=False`` skips the PART rank ceiling, applying only the
    counted-progressive chain-ordering (used by the strict_ladder uniform
    progressive pass).
    """
    def _rule(item, _r=ranks_dict, _e=extras_dict, _p=player,
              _chain=chain_full_extras, _orig=existing, _gp=gate_parts) -> bool:
        if _orig is not None and not _orig(item):
            return False
        if item.player != _p:
            return True
        sig = getattr(item, "rank_sig", None)
        if sig is not None and sig.axes:
            if not _gp:
                return True
            for axis_key, rank in sig.axes:
                if rank > _r.get(axis_key, 0):
                    return False
            return True
        tier = getattr(item, "_sphere_tier", None)
        if tier is not None:
            if tier > _chain.get(item.name, 0):
                return True  # past-chain — admitted anywhere
            if tier > _e.get(item.name, 0):
                return False
        return True
    return _rule




def _install_placement_rules(
    world: "KSP1World",
    location_min_ranks: dict[str, MinimumRanks],
    loc_prior_extras: dict[str, dict[str, int]],
    bootstrap_locations: set[str],
    chain_full_extras: dict[str, int],
) -> None:
    """Per-location item_rule for Phase 2 placement.

    Two independent mechanisms, composed via AND with whatever rule the
    location already carries:

      * **Rank ceiling** for PART items (``rank_sig`` populated).
        A part with rank R on axis A is admitted iff R ≤
        ``location_min_ranks[L][A]`` — the location's intrinsic
        min-ranks from the bumper.  This mirrors the legacy "tier 1
        parts go anywhere reachable" property: low-rank items have
        many admitting locations; high-rank items are restricted to
        late-chain spots that need them.

      * **Chain-ordering** for counted progressives (``_sphere_tier``
        set, ``rank_sig`` empty: R&D / Pad / PSI).  Tier T banned at L
        iff T > ``loc_prior_extras[L][name]`` — replicates the legacy
        Rule B (3) R&D soft-lock and Pad self-ban.  Tiers past the
        chain's max for that counted progressive are admitted
        unconditionally (no chain position → no constraint).

    Splitting these matches the legacy design: ``_install_tier_ban_rule``
    banned ``(progressive_item, tier)`` pairs — i.e., wrappers, not
    individual parts.  Individual parts only saw the implicit ceiling
    via the wrapper.  In rank-space the wrappers are gone, so the parts
    use the location's intrinsic rank ceiling directly.
    """
    player = world.player
    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        loc_ranks = location_min_ranks.get(loc.name)
        loc_extras = loc_prior_extras.get(loc.name)
        if loc_ranks is None and loc_extras is None:
            continue
        ranks_dict = dict(loc_ranks.upper_bounds) if loc_ranks is not None else {}
        extras_dict = dict(loc_extras) if loc_extras is not None else {}
        loc.item_rule = _make_placement_rule(
            player, ranks_dict, extras_dict, chain_full_extras, loc.item_rule,
        )


_EMPTY_EXTRAS: dict[str, int] = {}


def _sphere_covers(s: "SphereBoundary", axes, extras) -> bool:
    """True iff sphere ``s`` admits an item/location with these rank
    ``axes`` (iterable of ``(axis, rank)``) and counted-progressive
    ``extras`` (``name -> count``).

    Same admission test as :func:`_rank_admits_item`: an axis absent from
    the sphere's ceiling is *unavailable* (cap 0), not unconstrained.
    """
    ranks = s.ranks
    for ax, rk in axes:
        cap = ranks.get(ax)
        if cap is None or rk > cap:
            return False
    if extras:
        s_extras = s.extras
        for nm, cnt in extras.items():
            if cnt > s_extras.get(nm, 0):
                return False
    return True


def _first_covering_sphere(spheres, axes, extras) -> int:
    """Ladder index of the first sphere that covers ``(axes, extras)``.

    Sphere rank ceilings and counted-progressive ``extras`` grow
    monotonically along the chain, so the first covering sphere is the
    ladder position.  Returns ``len(spheres)`` when no real sphere covers
    it (beyond the chain's reach).
    """
    for i, s in enumerate(spheres):
        if _sphere_covers(s, axes, extras):
            return i
    return len(spheres)


def _item_min_sphere(item, spheres) -> int:
    """Ladder position of an item — the first sphere at which it becomes
    available, and therefore the earliest location sphere it may sit at.

    * Parts use their ``rank_sig`` axes against sphere rank ceilings.
    * Counted progressives (R&D / Pad / PSI) use ``{name: tier}`` against
      sphere ``extras``.
    * Items the chain never requires — spare high-rank parts whose ranks
      exceed the goal, or counted progressives the goal never bumps — are
      unconstrained (sphere 0).  This reproduces the legacy "past-chain ⇒
      admitted anywhere" escape, so they remain free filler.
    """
    sig = getattr(item, "rank_sig", None)
    if sig is not None and sig.axes:
        idx = _first_covering_sphere(spheres, sig.axes, _EMPTY_EXTRAS)
        return 0 if idx == len(spheres) else idx
    tier = getattr(item, "_sphere_tier", None)
    if tier is not None:
        idx = _first_covering_sphere(spheres, (), {item.name: tier})
        return 0 if idx == len(spheres) else idx
    return 0


def _compute_cascade_lo(
    n: int, cap: dict[int, int], demand: dict[int, int],
) -> dict[int, int]:
    """Capacity-driven lower bound per advancement min_sphere.

    A chain rep new at sphere ``k`` belongs at its prerequisite sphere
    ``k-1``, but if that sphere (and the ones just below it) lack the room
    to hold every rep that targets the region, the window must expand
    *earlier*.  For each ``k`` with demand, walk back from ``k-1``
    accumulating capacity until the spheres ``[lo, k-1]`` comfortably hold
    the reps targeting that span (free room ≥ max(20%, 5)).  ``lo`` is how
    far back reps with min_sphere ``k`` may be placed.
    """
    lo: dict[int, int] = {}
    for k in range(1, n + 1):
        if demand.get(k, 0) == 0:
            continue
        target = k - 1
        need = 0
        avail = 0
        res = 0
        for s in range(target, -1, -1):
            avail += cap.get(s, 0)
            need += demand.get(s + 1, 0)  # reps targeting s (min_sphere s+1)
            margin = max(5, round(0.2 * avail))
            res = s
            if avail >= need + margin:
                break
        lo[k] = res
    return lo


def _install_unified_sphere_rules(
    world: "KSP1World",
    ladder: "SphereLadder",
    location_min_ranks: dict[str, MinimumRanks],
    location_min_extras: dict[str, dict[str, int]],
    bootstrap_locations: set[str],
) -> None:
    """The unified placement rule: every item is admissible at location L
    iff ``item.min_sphere <= L.sphere`` — one sphere-index comparison for
    parts, R&D, Pad and PSI alike.

    Replaces the rank-ceiling (parts) / extras-chain-ordering (counted
    progressives) split.  It is purely a balance / progression-ordering
    constraint; soundness is owned by the capability cross-check in
    ``post_fill``, not here.  Locations with no rank requirement (KSC,
    starting inventory, Victory) are left to their existing rule.
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
    for name, ranks in location_min_ranks.items():
        if name in cheap_bracket:
            loc_sphere[name] = cheap_bracket[name]
        else:
            loc_sphere[name] = _first_covering_sphere(
                spheres, ranks.upper_bounds, location_min_extras.get(name, _EMPTY_EXTRAS)
            )

    # Capacity-driven cascade.  A chain rep belongs at its prerequisite sphere
    # (min_sphere-1), but high-rank reps whose prerequisite sphere is empty /
    # thin are otherwise structurally unplaceable (banned below, chicken-and-egg
    # at/above).  ``lo`` expands the window earlier sphere-by-sphere until the
    # candidate spheres hold enough room.
    from collections import Counter
    cap: dict[int, int] = dict(Counter(loc_sphere.values()))
    chain_reps: set[str] = set()
    for s in spheres:
        chain_reps |= s.reps_collected
    demand: Counter = Counter()
    for it in world.multiworld.itempool:
        if it.player != player:
            continue
        if it.name in chain_reps or getattr(it, "_sphere_tier", None) is not None:
            demand[_item_min_sphere(it, spheres)] += 1
    cascade_lo = _compute_cascade_lo(len(spheres), cap, dict(demand))
    world._cascade_lo = cascade_lo  # expose for analysis

    for loc in world.multiworld.get_locations(player):
        if loc.name in bootstrap_locations:
            continue
        L = loc_sphere.get(loc.name)
        if L is None:
            continue  # ungated (KSC / starting inventory / event) — keep existing rule
        existing = loc.item_rule

        def _rule(item, _p=player, _L=L, _spheres=spheres, _orig=existing,
                  _lo=cascade_lo) -> bool:
            if _orig is not None and not _orig(item):
                return False
            if item.player != _p:
                return True
            ms = _item_min_sphere(item, _spheres)
            if item.advancement:
                # Minimization moves over-bumped high-rank parts out of the
                # chain into USEFUL (gated late), so the advancement set is the
                # genuine minimal/basic kit — safe to place reachably (AP's
                # fill handles reachability; the cross-check confirms).  The
                # capacity-cascade lower bound is kept as an env-gated
                # alternative while the rep-picker overpower is rooted out.
                return os.environ.get("KSP_ADV_PERMISSIVE", "1") == "1" or \
                    _L >= _lo.get(ms, max(0, ms - 1))
            return ms <= _L

        loc.item_rule = _rule


def _admitted_set_for_ranks(
    ranks: MinimumRanks, ctx: RankContext,
    precollected_names: frozenset[str],
) -> set[str]:
    """Return the set of PART_DB item names admitted by ``ranks``.
    Used to build the ``item_count_fn`` for rank-space capability
    queries."""
    admitted: set[str] = set()
    for item_name in PART_DB:
        if _rank_admits_item(item_name, ranks, ctx):
            admitted.add(item_name)
    admitted |= precollected_names & PART_DB.keys()
    return admitted


def _compute_tech_tier_signatures_rank(
    world: "KSP1World", ladder: SphereLadder, ctx: RankContext,
) -> tuple[dict[str, LocationSignature], dict[str, MinimumRanks],
           dict[str, dict[str, int]], dict[int, SphereBoundary]]:
    """Rank-space port of the legacy ``_compute_tech_tier_signatures``.

    Walks the chain, computes science accumulation per sphere from the
    capability the sphere's ``(ranks, extras)`` proves, and identifies
    the earliest sphere that funds each tech tier's cumulative cost.
    Returns:
      * ``signatures``  — tech-tree location → LocationSignature
      * ``min_ranks``   — tech-tree location → MinimumRanks
      * ``min_extras``  — tech-tree location → counted-progressive kit
        (PSI/Pad copies the player should have by this sphere)
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
    for sphere in ladder.spheres:
        admitted = (sphere.reps_collected
                    | (precollected_names & frozenset(PART_DB.keys())))
        extras = sphere.extras

        def _count(name: str, _adm=admitted, _ext=extras) -> int:
            if name in _ext:
                return _ext[name]
            return 1 if name in _adm else 0

        cap, _flags = compute_capability_from_items(
            _count, difficulty_name,
            start_with_clamps=clamps,
            mission_builder=world.mission_builder,
            progressive_launch_pad=pad_on,
        )
        from .items import PROGRESSIVE_SCIENCE_INSTRUMENT_NAME
        psi_tier = extras.get(PROGRESSIVE_SCIENCE_INSTRUMENT_NAME, 0)
        sphere_science.append((sphere, bankable_science(cap, psi_tier, home) * safety))

    sigs: dict[str, LocationSignature] = {}
    min_ranks_out: dict[str, MinimumRanks] = {}
    min_extras_out: dict[str, dict[str, int]] = {}
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
        # min_ranks for tech-tree locations is the funding sphere's
        # cumulative rank ceiling; R&D / PSI go in min_extras.
        extras_kit = dict(funding.extras)
        if rd_required > 0:
            extras_kit[PROGRESSIVE_RD_NAME] = max(
                extras_kit.get(PROGRESSIVE_RD_NAME, 0), rd_required
            )
        for node in TECH_NODES:
            if node.tier != tier:
                continue
            for slot in range(1, num_slots + 1):
                loc_name = str(TechTreeLocation(node.display_name, slot))
                sigs[loc_name] = sig
                min_ranks_out[loc_name] = funding.ranks
                min_extras_out[loc_name] = extras_kit
        if rd_required > 0:
            prev = band_funding.get(rd_required)
            if prev is None or (
                prev.signature is not None
                and funding.signature.dv < prev.signature.dv
            ):
                band_funding[rd_required] = funding
    return sigs, min_ranks_out, min_extras_out, band_funding


def apply_sphere_ladder(world: "KSP1World") -> None:
    """Build the rank-space sphere ladder for ``world`` and install
    fill-time placement guidance.  Phase 2 — the progressive walker has
    been retired.

    Scope:
      - Compute per-location MinimumRanks ceiling + LocationSignature.
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
    start_with_clamps = bool(world.options.start_with_launch_clamps)
    precollected_names = frozenset(
        it.name for it in world.multiworld.precollected_items[world.player]
    )
    home = str(world.mission_builder.home)
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
    _max_ranks = MinimumRanks(tuple(sorted(
        ((a, max_rank_for(a)) for a in RankAxisKey),
        key=lambda x: x[0].value,
    )))
    _pad_max = (len(world.mission_builder.launch_pad_caps) - 1
                if world.mission_builder.launch_pad_caps else 0)
    _max_flags = _pre_pass_for_ranks(
        _max_ranks, ctx,
        start_with_clamps=start_with_clamps,
        progressive_launch_pad=progressive_launch_pad,
        launch_pad_caps=world.mission_builder.launch_pad_caps,
        pad_tier=_pad_max,
        precollected_names=precollected_names,
        reps_only=None,
    )
    _diff = DIFFICULTY_PROFILES[difficulty]

    def _ranks_from_kit(kit) -> MinimumRanks:
        out = MinimumRanks.empty()
        for part_name in kit.all_parts():
            sig = rank_sig_for(part_name, ctx)
            for ax, rk in sig.axes:
                if rk > (out.get(ax) or 0):
                    out = out.with_axis(ax, rk)
        return out

    location_min_ranks: dict[str, MinimumRanks] = {}
    _ranks_by_mission: dict[tuple, Optional[MinimumRanks]] = {}
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None or loc.name in infeasible:
            continue
        info = _parse_location(loc.name)
        if info is None:
            continue
        # Canonical mission key — dedup locations that share one mission.
        mkey = (info.body, info.mission_type, info.crewed, info.threshold_km)
        if mkey in _ranks_by_mission:
            derived = _ranks_by_mission[mkey]
        else:
            # Minimal kit grown from nothing (the chain's own bumper) — the
            # intrinsic per-location requirement.  The previous max-flags kit
            # over-specified support gear (best antenna/capsule/SAS), pinning
            # every mission to the top sphere; this is the empty-prior minimum
            # the rest of the code already assumes (see the chain-walk
            # fallback comment below).
            rocket = minimal_ranks_for(
                loc.name, MinimumRanks.empty(), ctx,
                difficulty=difficulty,
                progressive_launch_pad=progressive_launch_pad,
                start_with_clamps=start_with_clamps,
                rng=world.random,
                mission_builder=world.mission_builder,
                precollected_names=precollected_names,
            )
            derived = rocket.ranks if rocket is not None else None
            _ranks_by_mission[mkey] = derived
        if derived is None:
            continue
        location_min_ranks[loc.name] = derived
        ladder.location_signatures[loc.name] = LocationSignature(
            dv=_goal_dv(loc.name, world.mission_builder),
            requirements=tuple(),
            body_chain_depth=_body_chain_depth(info.body, world.mission_builder.home),
        )

    # Predictable spheres provide the spine.
    predictable_labels = [(label, name) for label, name in _predictable_spheres(world)]
    predictable_names = {name for _, name in predictable_labels}

    # Look up dv of the three anchors (defaults if missing).
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
        min_rank_size = len(
            location_min_ranks.get(name, MinimumRanks.empty()).upper_bounds
        )
        return (group, min_rank_size, sig.dv, sig.body_chain_depth, name)

    all_sphere_names.sort(key=_sort_key)

    # Walk the chain in rank space.  Reps-only mode: each bumper call
    # uses ONLY the reps collected so far + precollected items for its
    # feasibility check, matching what fill places as PROGRESSION.
    cumulative_ranks = MinimumRanks.empty()
    cumulative_extras: dict[str, int] = {}
    cumulative_reps: frozenset[str] = frozenset()
    sphere_rank_reps: dict[tuple[RankAxisKey, int], str] = {}
    for label, location_name, is_pred in all_sphere_names:
        rocket = minimal_ranks_for(
            location_name, cumulative_ranks, ctx,
            difficulty=difficulty,
            progressive_launch_pad=progressive_launch_pad,
            start_with_clamps=start_with_clamps,
            rng=world.random,
            mission_builder=world.mission_builder,
            prior_extras=cumulative_extras,
            prior_reps=cumulative_reps,
            precollected_names=precollected_names,
        )
        if rocket is None:
            if is_pred:
                # Chain-walk failure on a predictable anchor.  Fall back
                # to the intrinsic per-location ranks (computed earlier
                # from an empty prior) and merge into cumulative.  If
                # even THAT is missing, the location is truly infeasible.
                intrinsic = location_min_ranks.get(location_name)
                if intrinsic is None:
                    raise OptionError(
                        f"Sphere ladder: predictable anchor {label} "
                        f"({location_name!r}) is unreachable under any rank kit."
                    )
                merged = cumulative_ranks.merged_max(intrinsic)
                rocket = RankBumperResult(
                    ranks=merged,
                    delta=intrinsic,
                    reps={},
                    flags=_pre_pass_for_ranks(
                        merged, ctx,
                        start_with_clamps=start_with_clamps,
                        progressive_launch_pad=progressive_launch_pad,
                        launch_pad_caps=world.mission_builder.launch_pad_caps,
                        pad_tier=cumulative_extras.get(PROGRESSIVE_LAUNCH_PAD_NAME, 0),
                        precollected_names=precollected_names,
                    ),
                    profile_dv=0.0,
                    extras=dict(cumulative_extras),
                    extras_delta={},
                )
            else:
                continue
        for key, rep in rocket.reps.items():
            sphere_rank_reps[key] = rep
        cumulative_ranks = rocket.ranks
        cumulative_extras = dict(rocket.extras)
        cumulative_reps = rocket.reps_collected
        ladder.spheres.append(SphereBoundary(
            name=label,
            location_name=location_name,
            is_predictable=is_pred,
            ranks=rocket.ranks,
            ranks_delta=rocket.delta,
            reps_collected=rocket.reps_collected,
            extras=dict(rocket.extras),
            extras_delta=dict(rocket.extras_delta),
            flags=rocket.flags,
            profile_dv=rocket.profile_dv,
            signature=ladder.location_signatures.get(location_name),
        ))

    world._sphere_ladder = ladder
    world._sphere_rank_reps = sphere_rank_reps
    world._sphere_rank_cumulative = cumulative_ranks

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
    tech_sigs, tech_min_ranks, tech_min_extras, band_funding = (
        _compute_tech_tier_signatures_rank(world, ladder, ctx)
    )
    ladder.location_signatures.update(tech_sigs)
    location_min_ranks.update(tech_min_ranks)
    # Mirror tech-tree min_extras alongside min_ranks so the chain-
    # ordering rule sees R&D / PSI requirements per tech location.
    location_min_extras: dict[str, dict[str, int]] = dict(tech_min_extras)

    # Inject R&D into the chain at each funding sphere so subsequent
    # spheres' cumulative_extras reflect "by sphere S, player has R&D=B".
    # Without this the chain-ordering rule treats every R&D copy as
    # post-goal and bans them everywhere reachable.
    for band, funding_sphere in band_funding.items():
        # Find funding sphere's index in ladder.spheres.
        idx = None
        for i, s in enumerate(ladder.spheres):
            if s is funding_sphere:
                idx = i
                break
        if idx is None:
            continue
        cur = funding_sphere.extras_delta.get(PROGRESSIVE_RD_NAME, 0)
        prior_at_funding = funding_sphere.extras.get(PROGRESSIVE_RD_NAME, 0) - cur
        new_count = max(band, funding_sphere.extras.get(PROGRESSIVE_RD_NAME, 0))
        funding_sphere.extras[PROGRESSIVE_RD_NAME] = new_count
        funding_sphere.extras_delta[PROGRESSIVE_RD_NAME] = new_count - prior_at_funding
        # Propagate forward to all later sphere cumulatives.
        for later in ladder.spheres[idx + 1:]:
            later.extras[PROGRESSIVE_RD_NAME] = max(
                later.extras.get(PROGRESSIVE_RD_NAME, 0), new_count,
            )

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
    _capsule_kit = MinimumRanks.empty().with_axis(RankAxisKey.CAPSULE, 1)
    _gate_early: dict[str, MinimumRanks] = {
        name: _capsule_kit for name in world.location_builder.ksc_biome_names
    }
    _gate_early[f"{home} First Launch"] = MinimumRanks.empty()
    for loc in world.multiworld.get_locations(world.player):
        if loc.address is None:
            continue
        if loc.name == "Splashdown":
            _gate_early[loc.name] = _capsule_kit
        elif loc.name.startswith("Starting Inventory"):
            _gate_early[loc.name] = MinimumRanks.empty()
    for _name, _ranks in _gate_early.items():
        location_min_ranks.setdefault(_name, _ranks)
        bootstrap_locations.discard(_name)

    # Per-location chain-ordering rule (replaces the previous rank-
    # ceiling rule).  Combines:
    #   * Self-ban — items in L's own min_kit can't land at L.
    #   * Chain-ordering — items added at the sphere covering L (or
    #     later) can't land at L; collecting them at L would be a
    #     chicken-and-egg.
    #   * R&D soft-lock — R&D=B banned at L if L isn't reachable with
    #     band B's funding sphere's cumulative kit.
    #   * Pad self-ban — Pad=K banned at L if L's min_extras need K.
    location_names_in_chain = {s.location_name for s in ladder.spheres}
    loc_prior_ranks, loc_prior_extras = _compute_location_priors(
        ladder, location_min_ranks, location_min_extras,
        location_names_in_chain,
    )
    chain_full_extras: dict[str, int] = {}
    for sphere in ladder.spheres:
        for name, count in sphere.extras.items():
            chain_full_extras[name] = max(chain_full_extras.get(name, 0), count)
    # Expose the finalized per-location rank/extras maps for analysis tooling.
    world._location_min_ranks = location_min_ranks
    world._location_min_extras = location_min_extras
    if _ACCESS_RULE_MODE in ("ladder", "strict_ladder"):
        # Cheap reachability: one capability-bracket per mission gives each
        # location a microsecond ``has_all(reps)`` access rule (validated
        # against real physics by the post_fill cross-check).
        _install_ladder_rules(
            world, ladder, location_min_ranks, location_min_extras,
            bootstrap_locations,
            save_original=(_ACCESS_RULE_MODE == "strict_ladder"),
        )
        # Unified placement: ONE rule for every item — admissible at L iff
        # its ladder position (min_sphere) ≤ L's.  Parts, R&D, Pad and PSI
        # share the same sphere-index comparison; soundness is the cross-
        # check's job, not this balance rule.
        _install_unified_sphere_rules(
            world, ladder, location_min_ranks, location_min_extras,
            bootstrap_locations,
        )
    else:
        _install_placement_rules(
            world, location_min_ranks, loc_prior_extras, bootstrap_locations,
            chain_full_extras,
        )

    # Demote everything that isn't in the chain's collected reps set,
    # and clear past-goal rank entries so above-ceiling items can
    # scatter freely.  ``cumulative_reps`` is the union of all parts
    # the bumper admitted across the chain — under the bulk-admit
    # extension it includes every part at any (axis, rank) the chain
    # touched.
    rep_part_names = set(cumulative_reps)
    rep_part_names |= set(sphere_rank_reps.values())  # belt-and-suspenders
    _demote_non_rep_parts(world, rep_part_names, cumulative_ranks,
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
                _dout.write(f'      ranks={list(_s.ranks.upper_bounds)}\n')
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
                _out.write(f'      ranks={list(_s.ranks.upper_bounds)}\n')
                _out.write(f'      extras={dict(_s.extras)}\n')
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
            launch_ceil = dict(launch_sphere.ranks.upper_bounds)
            if launch_ceil.get(_axis, 0) >= _rank:
                local_early[rep_name] = max(local_early.get(rep_name, 0), 1)

