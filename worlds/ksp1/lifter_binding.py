"""Offline lifter-chain binding tables (pad -> home low orbit).

The offline generator (``scripts/generate_lifter_chains.py``) grows randomized
part *chains* per ascent-regime cluster and binds each chain against real
physics per home body: for every payload threshold, the shortest chain prefix
whose parts close ``find_optimal_multistage_ascent`` at that mass, together
with the winning build (rungs).  At runtime the bumper consults the seed's
bound table instead of re-searching the ascent space.

Golden-Rule argument for serving a stored build: a rung is served only when
its chain prefix is a subset of the parts the evaluation admits, so the kit
can assemble the exact bind-time rocket; the stored rung was produced by the
real optimizer at a threshold >= the requested payload, so its mass only
overestimates.  Ceilings are computed under maximally permissive settings
(full pool, escalated bounds), so a payload above the ceiling is exactly
infeasible for ANY kit — the quick-reject is not an approximation.

This module never imports ``capability`` (capability imports it).  All part
resolution goes through the registry's stable offsets; anything pack-scoped
stays with the caller's PartManager.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

from .parts.registry import PART_REGISTRY
from .rocket_math import StageResult

if TYPE_CHECKING:
    from .bodies import BodyName, DifficultyProfile, MissionBuilder

# Stable identity for stored parts: PartMapping.offset <-> AP item name.
# Registry data is pack-independent; pack filtering happens at chain
# generation time (a chain only ever references parts of its pack set).
OFFSET_TO_NAME: dict[int, str] = {m.offset: m.ksp_name for m in PART_REGISTRY}
NAME_TO_OFFSET: dict[str, int] = {m.ksp_name: m.offset for m in PART_REGISTRY}


@dataclass(frozen=True)
class AscentConstraints:
    """Fingerprint of the non-kit ascent-build inputs a rung was bound under.

    A stored rung may serve a live request only if the bind-time constraints
    are at least as strict (see ``serves``).  Body-derived physics (gravity,
    atmosphere profile) is implied by the table's body key and not repeated
    here; ``pad_altitude_m`` is kept because the elevated-pad Isp credit is
    the one body input the generator could plausibly drift on (Eve mesa).
    """
    in_atmosphere: bool
    min_twr_liftoff: float
    requires_throttleable: bool
    srb_needs_rcs: bool
    needs_heat_shield: bool
    pad_altitude_m: float

    def serves(self, live: "AscentConstraints") -> bool:
        """True if a build bound under ``self`` is valid under ``live``.

        Direction-aware: bind-time may be stricter than the live request
        (extra TWR, throttle, RCS-gated SRBs, shields = conservative mass
        overestimates), never looser.

        ``require_gimbal`` is deliberately NOT a constraint field: the live
        gate fires only when the kit lacks the relevant control part (aero
        surface for atmospheric ascent, any wheel/RCS for the global-attitude
        fallback), and a bound build that leaned on an ungimballed engine
        always carries its control part in the serving prefix — so any kit
        the rung serves owns that part and the live gate cannot be raised.
        A gimballed bound build serves either way.
        """
        if self.in_atmosphere != live.in_atmosphere:
            return False
        if abs(self.pad_altitude_m - live.pad_altitude_m) > 1.0:
            return False
        if self.min_twr_liftoff < live.min_twr_liftoff - 1e-6:
            return False
        if live.requires_throttleable and not self.requires_throttleable:
            return False
        if live.srb_needs_rcs and not self.srb_needs_rcs:
            return False
        if live.needs_heat_shield and not self.needs_heat_shield:
            return False
        return True


@dataclass(frozen=True)
class LifterHint:
    """Architecture guide for one bound build: the dv split plus per-stage
    (n_boosters, booster_engines, engine_offset), bottom stage first.  Pins
    ``find_optimal_multistage_ascent``'s search to the bind-time winner so
    the REAL build is recomputed on demand at ~single-evaluation cost
    (measured ~150x vs the full search; the stored table needs no stage
    blueprints at all — hints intern heavily because architectures repeat
    across thresholds/chains/profiles)."""
    split: tuple[float, ...]
    stages: tuple[tuple[int, int, int], ...]

    def to_guide(self) -> tuple:
        """The ``guide=`` argument for ``find_optimal_multistage_ascent``."""
        return (self.split,
                tuple((nb, be, OFFSET_TO_NAME[eng])
                      for nb, be, eng in self.stages))


@dataclass(frozen=True)
class BoundRung:
    """One (payload threshold -> build) row of a bound ladder.  ``hint``
    rebuilds the exact bind-time rocket on demand; ``launch_mass`` is the
    bind-time result (rounded up), used for quick mass checks and as the
    drift tripwire against the rebuilt value."""
    threshold_t: float
    prefix_len: int
    launch_mass: float
    hint: LifterHint


@dataclass(frozen=True)
class BoundLadder:
    """All rungs of one (physics profile, required_dv variant), ascending by
    threshold.  ``ceiling_t`` is the full-pool max payload for this dv/profile
    (kit-independent: nothing can lift more, so payload > ceiling is exactly
    infeasible)."""
    dv: float
    ceiling_t: float
    rungs: tuple[BoundRung, ...]
    constraints: AscentConstraints


class ServeResult(Enum):
    SERVED = "served"
    PREFIX_MISSING = "prefix_missing"
    OVER_CEILING = "over_ceiling"
    NOT_COVERED = "not_covered"


@dataclass(frozen=True)
class LifterConsult:
    result: ServeResult
    hint: Optional[LifterHint] = None
    launch_mass: float = 0.0
    prefix_used: frozenset[str] = frozenset()
    missing_parts: tuple[str, ...] = ()
    ceiling_t: float = 0.0
    dv_bound: float = 0.0


# Requests within this many m/s above a bound dv still serve from it (float
# drift guard; a genuinely higher dv variant must fall through to live
# physics, never round DOWN onto a weaker bound).
_DV_EPSILON = 0.5


class BoundLifterTable:
    """Per-world runtime table: one chain (the seed's profile) plus its bound
    ladders for the seed's physics profile.  Chain parts are item names,
    resolved from offsets at load time."""

    def __init__(self, home: "BodyName", profile_id: int,
                 chain: tuple[str, ...],
                 ladders: dict[str, tuple[BoundLadder, ...]],
                 phys_profile: Optional[str] = None) -> None:
        self.home = home
        self.profile_id = profile_id
        self.chain = chain
        # The single physics profile this table was loaded for (the seed's).
        # The evaluator passes it to consult() so it need not re-derive the
        # name from the DifficultyProfile.  Defaults to the sole ladder key.
        self.phys_profile = phys_profile or next(iter(ladders), None)
        # dv-ascending per physics profile; consult picks the first ladder
        # with dv >= request.
        self.ladders = {
            prof: tuple(sorted(lads, key=lambda l: l.dv))
            for prof, lads in ladders.items()
        }
        # Cumulative prefix sets, index = prefix length.  Consult compares
        # subsets against these instead of re-slicing the chain per call.
        self._prefix_sets: list[frozenset[str]] = [frozenset()]
        for part in chain:
            self._prefix_sets.append(self._prefix_sets[-1] | {part})

    def prefix_set(self, length: int) -> frozenset[str]:
        return self._prefix_sets[min(length, len(self.chain))]


def hint_from_stages(stages: list[StageResult]) -> LifterHint:
    """Extract the architecture hint from an optimizer result (generator
    side).  The split ratios are NOT recoverable from achieved dv, so the
    generator canonicalizes: it re-runs the guided build per candidate split
    and stores the argmin — self-consistent with the runtime rebuild by
    construction."""
    return LifterHint(
        split=(),
        stages=tuple((s.n_boosters, s.booster_engines,
                      NAME_TO_OFFSET[s.engine_name]) for s in stages),
    )


def consult(table: BoundLifterTable, *, phys_profile: str, required_dv: float,
            payload_t: float, admitted_parts: frozenset[str],
            live: AscentConstraints) -> LifterConsult:
    """Answer a home-ascent build request from the bound table.

    Serve order: smallest bound dv >= request, then smallest rung threshold
    >= payload.  Every miss reason is typed so the caller can distinguish
    "fall back to live physics" (NOT_COVERED) from the two bumper-guidance
    outcomes (PREFIX_MISSING / OVER_CEILING)."""
    ladders = table.ladders.get(phys_profile, ())
    ladder: Optional[BoundLadder] = None
    for cand in ladders:
        if cand.dv >= required_dv - _DV_EPSILON:
            ladder = cand
            break
    if ladder is None or not ladder.constraints.serves(live):
        return LifterConsult(result=ServeResult.NOT_COVERED)

    # The seed is committed to ONE chain, so its capacity is what the FULL
    # chain can single-launch — the top rung's threshold — NOT the full-pool
    # ceiling (which a different kit could reach but this seed never will).
    # Above that is assembly territory: OVER_CEILING drives the multi-launch
    # split, exactly as a payload above the absolute pool ceiling would.
    # Reporting the pool ceiling here would be a lie about this seed's reach
    # and (the old bug) left a band [top_rung, pool_ceiling] that fell through
    # to live physics instead.
    chain_max = ladder.rungs[-1].threshold_t if ladder.rungs else 0.0
    if payload_t > chain_max:
        return LifterConsult(result=ServeResult.OVER_CEILING,
                             ceiling_t=chain_max, dv_bound=ladder.dv)

    # Launch mass is NOT monotone along the ladder: a higher rung's longer
    # prefix can build lighter (e.g. asparagus unlocks mid-chain).  Serve the
    # lightest admitted rung at or above the payload; if none is admitted,
    # guide toward the smallest one.
    first_missing: Optional[BoundRung] = None
    best: Optional[BoundRung] = None
    for rung in ladder.rungs:
        if rung.threshold_t < payload_t:
            continue
        if table.prefix_set(rung.prefix_len) <= admitted_parts:
            if best is None or rung.launch_mass < best.launch_mass:
                best = rung
        elif first_missing is None:
            first_missing = rung
    if best is not None:
        return LifterConsult(
            result=ServeResult.SERVED,
            hint=best.hint,
            launch_mass=best.launch_mass,
            prefix_used=table.prefix_set(best.prefix_len),
            dv_bound=ladder.dv,
        )
    if first_missing is not None:
        missing = tuple(p for p in table.chain[:first_missing.prefix_len]
                        if p not in admitted_parts)
        return LifterConsult(result=ServeResult.PREFIX_MISSING,
                             missing_parts=missing, dv_bound=ladder.dv)
    # payload <= chain_max and every covering rung's prefix is already owned
    # yet none served — the ladder has no rung at/above this payload despite
    # being within capacity (a gap in the rung set).  Rare; fall back honestly.
    return LifterConsult(result=ServeResult.NOT_COVERED)


def home_ascent_dv_variants(mission_builder: "MissionBuilder",
                            diff: "DifficultyProfile") -> tuple[float, ...]:
    """The required_dv values home-ascent builds are requested at, ascending.

    v1 binds two variants: the bare home ascent and ascent + the Apollo/
    assembly rendezvous surcharge (the maximum the assembly chunk lifters
    request).  Contract transforms that land between the two serve
    conservatively from the rendezvous variant via consult()'s
    smallest-dv->=-request rule; anything above falls through to live
    physics."""
    from .bodies import EdgeType, MissionType, effective_dv

    home = mission_builder.home
    ascent = None
    for mt in (MissionType.ORBIT, MissionType.RETURN, MissionType.SAMPLE_RETURN):
        for profile in mission_builder.profiles_for(home, mt) or ():
            for edge in profile:
                if (edge.body == home and edge.edge_type in
                        (EdgeType.ATMOSPHERIC_ASCENT, EdgeType.VACUUM_ASCENT)):
                    ascent = edge
                    break
            if ascent is not None:
                break
        if ascent is not None:
            break
    if ascent is None:
        raise ValueError(f"no home ascent edge found for {home}")
    base = effective_dv(ascent.base_dv, diff,
                        plane_change_dv=ascent.plane_change_dv)
    rendezvous = effective_dv(
        ascent.base_dv + type(mission_builder)._RESCUE_RENDEZVOUS_DV, diff,
        plane_change_dv=ascent.plane_change_dv)
    return (base, rendezvous)
