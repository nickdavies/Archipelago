"""Per-category rank scoring for KSP1 parts.

Each part participates on zero or more rank axes (engine launch/vac, tank by
fuel type, solar, etc.).  An axis maps a part's raw properties to a real-
valued score, then equal-frequency bucketing over the PART_DB at first use
discretizes the score into an integer rank ``1..buckets``.

Two directions are supported.  Both bucketize ascending — what differs is
the *meaning* of low buckets:

* :py:attr:`RankDirection.HIGHER_BETTER` — strong parts score high and land
  in high buckets, so early spheres (low ceilings) admit only weak parts.
* :py:attr:`RankDirection.LOWER_BETTER` — light parts score low and land in
  low buckets, so early spheres admit the lightest (best) parts; heavier
  parts come in later.

In both cases the placement rule is uniformly ``item_rank <= ceiling``.

The sphere ladder will consume these ranks to derive per-location ceilings;
``item_rule`` admits a part only if its rank on every applicable axis is
within the sphere's ceiling.  This module deliberately contains no AP imports
— it operates purely on PART_DB and is reusable from tests / tools.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from .parts import (
    ALL_PACKS,
    part_manager_for,
    AnyPart,
    CapabilityFlag,
    Decoupler,
    Engine,
    FuelTank,
    HeatShield,
    LandingLeg,
    MiscEquipment,
    Parachute,
    SolidBooster,
)
from .part_geometry import PartRole


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class RankAxisKey(StrEnum):
    LAUNCH_ENGINE = "launch_engine_rank"
    VAC_ENGINE = "vac_engine_rank"
    LFO_TANK = "lfo_tank_rank"
    LF_TANK = "lf_tank_rank"
    XENON_TANK = "xenon_tank_rank"
    MONOPROP_TANK = "monoprop_tank_rank"
    SRB = "srb_rank"
    HEAT_SHIELD = "heat_shield_rank"
    PARACHUTE = "parachute_rank"
    LANDING_LEG = "landing_leg_rank"
    STACK_DECOUPLER = "stack_decoupler_rank"
    RADIAL_DECOUPLER = "radial_decoupler_rank"
    SOLAR = "solar_rank"
    RELAY = "relay_rank"
    SAS = "sas_rank"
    CAPSULE = "capsule_rank"
    PROBE_SAS = "probe_sas_rank"


class RankDirection(StrEnum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


@dataclass(frozen=True)
class RankContext:
    """Per-world context that some scorers depend on.

    Currently the only axis that varies with world state is :py:attr:`SRB`:
    home bodies with an atmosphere (Kerbin, Eve, Duna, Laythe) score SRBs
    by ``atm_isp * fuel_mass`` because SRBs are predominantly atmospheric
    stagers there; vacuum homes score by ``vac_isp * fuel_mass`` instead.
    """
    home_has_atmosphere: bool
    # Which packs are enabled — determines the part universe the rank table is
    # bucketed over (a disabled pack's parts occupy no rank). Defaults to every
    # installed pack; Phase 2 binds the world's enabled set. The per-context
    # caches key on this, so each distinct pack-set is computed once.
    enabled_packs: frozenset[str] = ALL_PACKS
    # buildings_in_logic: whether home-system (moon) transfers require patched
    # conics / maneuver nodes (resolved from the HomeSystem* options + Difficulty).
    # Threaded through the ladder's capability path; irrelevant when the option
    # is off (default True is only consulted when buildings_in_logic is on).
    local_needs_conics: bool = True
    local_needs_nodes: bool = True


# Kerbin baseline.  Used when no per-world context has been bound (tests,
# tools, the module-load sanity pass).
DEFAULT_CONTEXT = RankContext(home_has_atmosphere=True)


# ---------------------------------------------------------------------------
# Scorers — pure functions ``(part, ctx) -> Optional[float]``.  Return None
# when the part is off this axis (e.g., a 1.25m FuelTank on XENON_TANK).
# ---------------------------------------------------------------------------

def _engine_launch(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, Engine):
        return None
    if p.atm_thrust <= 0:
        return None
    if p.fuel_type == "xenon":  # ion is out of logic — see _engine_vac
        return None
    return (
        0.4 * p.atm_isp
        + 0.3 * (p.atm_thrust / max(p.mass, 0.05))
        + (100.0 if p.has_gimbal else 0.0)
        + (30.0 if p.throttleable else 0.0)
    )


def _engine_vac(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, Engine):
        return None
    if p.vac_thrust <= 0:
        return None
    # Ion (xenon) is out of logic: its absurd Isp dominates every dv-bound
    # mission, which crowns it the required engine and kills variance.  It's
    # also never *needed* (every mission is reachable non-ion, just heavier).
    # So leave it off the engine axes entirely — capability ignores it too,
    # and it stays an out-of-logic bonus the player can fly if they collect
    # it.  See _filter_engines_for_ion.
    if p.fuel_type == "xenon":
        return None
    return (
        1.5 * p.vac_isp
        + 0.2 * math.log(1.0 + p.vac_thrust / max(p.mass, 0.05))
        + (20.0 if p.throttleable else 0.0)
        - 5.0 * p.mass
    )


def _make_tank_scorer(fuel_type: str) -> Callable[[AnyPart, RankContext], Optional[float]]:
    def _scorer(p: AnyPart, ctx: RankContext) -> Optional[float]:
        if not isinstance(p, FuelTank):
            return None
        if p.fuel_type != fuel_type:
            return None
        # Only spine-stackable tanks belong on the tank rank axis: the axis gates
        # the central fuel column, and a rep designated here must be a tank the
        # capability builder can actually stack.  Radial/coupler/single-node
        # tanks fall off the axis (they stay useful-pool bonus parts) — this is
        # the single source of truth shared with rocket_math's spine filter.
        if PartRole.SPINE not in p.roles:
            return None
        return p.dry_mass  # lower_better
    return _scorer


def _srb(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, SolidBooster):
        return None
    isp = p.atm_isp if ctx.home_has_atmosphere else p.vac_isp
    return isp * p.fuel_mass


def _heat_shield(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, HeatShield):
        return None
    return p.size_class


def _parachute(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, Parachute):
        return None
    bonus = 3.0 if p.is_radial else 1.0
    return (p.drag_area / max(p.mass, 0.01)) * bonus


def _landing_leg(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, LandingLeg):
        return None
    return p.mass  # lower_better


def _stack_decoupler(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, Decoupler) or p.kind != "stack":
        return None
    return p.size_class


def _radial_decoupler(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if isinstance(p, Decoupler) and p.kind == "radial":
        # The fuel-line is a MiscEquipment (FUEL_LINE), not a Decoupler;
        # handled below.  Radial Decouplers themselves bucket by mass —
        # bigger radial = more thrust capacity = higher rank.
        return p.mass
    if isinstance(p, MiscEquipment) and CapabilityFlag.FUEL_LINE in p.provides:
        # Fuel lines unlock asparagus; they should land at the top of the
        # axis.  Using a very high score keeps them in the last bucket
        # regardless of decoupler population.
        return math.inf
    return None


def _solar(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, MiscEquipment) or p.solar is None:
        return None
    # Both ec_per_sec and mass contribute positively: high-output panels rank
    # high (admitted late), and per the user's "tendency towards high mass"
    # heavier panels also rank higher (small rockets can't afford them early).
    # Output is the dominant signal; mass nudges the order on near-ties.
    return p.solar.charge_rate + 50.0 * p.mass


def _relay(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, MiscEquipment) or p.antenna is None:
        return None
    # INTERNAL is the 5kW transmitter built into pods/probes — not a real
    # antenna for relay purposes.
    if p.antenna.antenna_type not in ("DIRECT", "RELAY"):
        return None
    # Power ranges over ~6 orders of magnitude (5e5 → 1e11) — log compresses
    # the axis.  Subtract a mass nudge so lighter wins among similar-power
    # antennas.
    return math.log10(p.antenna.power) - 0.5 * p.mass


def _sas(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, MiscEquipment):
        return None
    if CapabilityFlag.REACTION_WHEEL not in p.provides:
        return None
    # Command modules (capsules and probe cores) carry embedded reaction
    # wheels but rank on their own axes — exclude them here.
    if CapabilityFlag.CAPSULE in p.provides or CapabilityFlag.PROBE_CORE in p.provides:
        return None
    return p.mass  # lower_better — "enough is enough" semantics.


def _capsule(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, MiscEquipment) or p.capsule is None:
        return None
    # Mobile Lab, Hitchhiker (``crewCabin``), etc. all have ``CapsuleSpec``
    # because their cfg lists ``CrewCapacity`` > 0, but their PartMapping
    # intentionally omits ``capsule`` from ``provides`` because they cannot
    # serve as terminal payload (no integrated command authority).  The
    # capability code only treats parts with the ``capsule`` provides flag
    # as valid pods, so the rank axis must mirror that — otherwise the
    # bumper picks a "rank-1 capsule" rep that doesn't satisfy
    # NO_CAPSULE, leading to repeated bumps and rank-ceiling inflation.
    if CapabilityFlag.CAPSULE not in p.provides:
        return None
    return p.mass - p.capsule.drainable_mass  # lower_better


def _probe_sas(p: AnyPart, ctx: RankContext) -> Optional[float]:
    if not isinstance(p, MiscEquipment) or p.probe_core is None:
        return None
    return float(p.probe_core.sas_level)


# ---------------------------------------------------------------------------
# Axis registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankAxis:
    key: RankAxisKey
    scorer: Callable[[AnyPart, RankContext], Optional[float]]
    direction: RankDirection
    # Rank count is data-driven: min(distinct score values in PART_DB,
    # _RANK_CAP).  See max_rank_for().  No per-axis hand-tuned count.


RANK_AXES: tuple[RankAxis, ...] = (
    RankAxis(RankAxisKey.LAUNCH_ENGINE,    _engine_launch,                RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.VAC_ENGINE,       _engine_vac,                   RankDirection.HIGHER_BETTER),
    # Tanks rank by dry mass but SMALL=early (HIGHER_BETTER on dry_mass puts
    # the heaviest/biggest tanks at the top rank).  Small tanks must be the
    # early admit: the optimizer stacks them to any fuel total, so exposing
    # only giant fuselages early (the old LOWER_BETTER flip) forced rockets
    # built from bad-ratio Mk3 parts (478t Mun landing).  Big tanks are a
    # late convenience, not an early gate.
    RankAxis(RankAxisKey.LFO_TANK,         _make_tank_scorer("lfo"),      RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.LF_TANK,          _make_tank_scorer("lf"),       RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.XENON_TANK,       _make_tank_scorer("xenon"),    RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.MONOPROP_TANK,    _make_tank_scorer("monoprop"), RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.SRB,              _srb,                          RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.HEAT_SHIELD,      _heat_shield,                  RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.PARACHUTE,        _parachute,                    RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.LANDING_LEG,      _landing_leg,                  RankDirection.LOWER_BETTER),
    RankAxis(RankAxisKey.STACK_DECOUPLER,  _stack_decoupler,              RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.RADIAL_DECOUPLER, _radial_decoupler,             RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.SOLAR,            _solar,                        RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.RELAY,            _relay,                        RankDirection.HIGHER_BETTER),
    RankAxis(RankAxisKey.SAS,              _sas,                          RankDirection.LOWER_BETTER),
    RankAxis(RankAxisKey.CAPSULE,          _capsule,                      RankDirection.LOWER_BETTER),
    RankAxis(RankAxisKey.PROBE_SAS,        _probe_sas,                    RankDirection.HIGHER_BETTER),
)

RANK_AXES_BY_KEY: dict[RankAxisKey, RankAxis] = {a.key: a for a in RANK_AXES}


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _bucketize(scores: list[float], buckets: int) -> list[int]:
    """Assign each input score a bucket in ``1..buckets``.

    Two strategies depending on score distribution:

    * If the number of *distinct* scores fits inside the bucket count, use
      **value-based mapping**: every distinct score value gets its own bucket
      (ascending), and items with identical scores land in the same bucket.
      This is the correct behavior for intrinsically discrete axes like
      ``PROBE_SAS`` (SAS levels 0..3) where equal-frequency bucketing would
      split tie groups across buckets.

    * Otherwise (more distinct values than buckets) fall back to
      **equal-frequency**: sort ascending and slice evenly across buckets.
    """
    if not scores:
        return []
    distinct = sorted(set(scores))
    n = len(scores)
    bs = max(1, min(buckets, n))
    if len(distinct) <= bs:
        # Distinct-value mapping — ties land in the same bucket.
        value_to_bucket = {v: i + 1 for i, v in enumerate(distinct)}
        return [value_to_bucket[s] for s in scores]
    order = sorted(range(n), key=lambda i: scores[i])
    out = [0] * n
    for pos, idx in enumerate(order):
        out[idx] = min(int(pos * bs / n) + 1, bs)
    return out


# ---------------------------------------------------------------------------
# Per-context computation + cache
# ---------------------------------------------------------------------------

# Per-item-name max-score on the axis (most items map to a single part,
# but multi-part items like multi-mount adapters can have more — take the
# extremum that best represents the item).  For HIGHER_BETTER axes we want
# the max; for LOWER_BETTER, the min.
def _item_score(parts: list[AnyPart], axis: RankAxis, ctx: RankContext) -> Optional[float]:
    best: Optional[float] = None
    for p in parts:
        s = axis.scorer(p, ctx)
        if s is None:
            continue
        if best is None:
            best = s
            continue
        if axis.direction == RankDirection.HIGHER_BETTER:
            best = max(best, s)
        else:
            best = min(best, s)
    return best


_RANK_CACHE: dict[RankContext, dict[RankAxisKey, dict[str, int]]] = {}

# Global cap on the rank-bucket count.  The *effective* bucket count for an
# axis is ``min(distinct score values, _RANK_CAP)``, computed from PART_DB at
# load — no per-axis hand-tuned ints.  Discrete axes (few distinct values,
# e.g. heat-shield coverage tiers, decoupler sizes) fall under the cap and get
# one rank per distinct value (correct resolution).  Continuous axes (engines/
# tanks, ~unique score per part) hit the cap and stay coarse so multiple parts
# share each rank — preserving the rep-picker's per-seed variance (with one
# rank per part the "next" part would be fully predictable).  Adapts
# automatically as PART_DB gains/loses parts (mods, DLC toggles).
_RANK_CAP = int(os.environ.get("KSP_RANK_CAP", "8"))

# Effective max rank per axis (= min(distinct, _RANK_CAP)); filled by
# _compute_ranks_for_context.  Single source of truth for "axis at cap".
_AXIS_MAX_RANK: dict[RankAxisKey, int] = {}

# Fungible axes get a *shallow* cap.  Tanks are ~interchangeable (LFO
# dry-fraction is 0.111-0.127 across the whole DB) and the optimizer just
# stacks small tanks to any total, so a deep rank ladder is meaningless and
# actively harmful: it parks big tanks at the top rank where they pile onto
# the final spheres.  Two ranks is enough — small tanks gate early (the
# building block), big tanks land low and place freely.  The launch-pad mass
# cap, not the tank rank, is the real size limiter.
_FUNGIBLE_AXIS_CAP = 2
_FUNGIBLE_AXES: frozenset[RankAxisKey] = frozenset({
    RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK,
    RankAxisKey.XENON_TANK, RankAxisKey.MONOPROP_TANK,
})


def _compute_ranks_for_context(ctx: RankContext) -> dict[RankAxisKey, dict[str, int]]:
    out: dict[RankAxisKey, dict[str, int]] = {}
    scores_out: dict[RankAxisKey, dict[str, float]] = {}
    part_db = part_manager_for(ctx.enabled_packs).parts
    for axis in RANK_AXES:
        scored: list[tuple[str, float]] = []
        for item_name, parts in part_db.items():
            s = _item_score(parts, axis, ctx)
            if s is not None:
                scored.append((item_name, s))
        cap = _FUNGIBLE_AXIS_CAP if axis.key in _FUNGIBLE_AXES else _RANK_CAP
        buckets = _bucketize([s for _, s in scored], cap)
        max_rank = max(buckets) if buckets else 1
        _AXIS_MAX_RANK[axis.key] = max_rank
        # For LOWER_BETTER axes, the bucketing above puts low scores in
        # low buckets (best parts in rank 1).  But the design intent for
        # rank-space placement is:  rank 1 = worst part (admitted early
        # as a forced choice), rank N = best part (admitted late as a
        # reward).  Flip the bucket index so the semantic is uniform
        # across direction.  Flip around the *effective* max rank
        # (min(distinct, _RANK_CAP)), not the cap, so value-based axes
        # (distinct < cap) flip correctly.
        if axis.direction == RankDirection.LOWER_BETTER:
            buckets = [max_rank + 1 - b for b in buckets]
        out[axis.key] = {name: b for (name, _), b in zip(scored, buckets)}
        # Stash raw scores so the bumper can rank candidates inside a
        # bucket — random picks within a bucket waste bumps when several
        # parts cluster on the boundary but only the top-scoring one
        # actually meets the mission's threshold.
        scores_out[axis.key] = {name: s for (name, s) in scored}
    # Single-side effect: populate the score cache for this context.
    _SCORE_CACHE[ctx] = scores_out
    return out


_SCORE_CACHE: dict[RankContext, dict[RankAxisKey, dict[str, float]]] = {}


def item_scores_for_context(
    ctx: RankContext = DEFAULT_CONTEXT,
) -> dict[RankAxisKey, dict[str, float]]:
    """Per-axis raw scores keyed by item name.  Populated as a side
    effect of ``ranks_for_context`` — call that first if you've never
    touched this context before.
    """
    if ctx not in _SCORE_CACHE:
        ranks_for_context(ctx)
    return _SCORE_CACHE[ctx]


def ranks_for_context(ctx: RankContext = DEFAULT_CONTEXT) -> dict[RankAxisKey, dict[str, int]]:
    """Return the per-axis rank table for every item in PART_DB.

    The result is a nested dict ``axis_key -> item_name -> int rank``.  Cached
    per context; subsequent calls with the same context are O(1).
    """
    cached = _RANK_CACHE.get(ctx)
    if cached is None:
        cached = _compute_ranks_for_context(ctx)
        _RANK_CACHE[ctx] = cached
    return cached


def max_rank_for(axis_key: RankAxisKey, ctx: RankContext = DEFAULT_CONTEXT) -> int:
    """Effective max rank (cap) for an axis = ``min(distinct scores, _RANK_CAP)``.

    Single source of truth for the bumper's "is this axis already at cap"
    check and for building max-rank vectors.  Data-driven, so it tracks
    PART_DB size automatically.
    """
    if axis_key not in _AXIS_MAX_RANK:
        ranks_for_context(ctx)
    return _AXIS_MAX_RANK.get(axis_key, _RANK_CAP)


# ---------------------------------------------------------------------------
# Per-item rank signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemRankSig:
    """Per-item view of all axes the item participates in.

    ``axes`` is sorted by ``RankAxisKey.value`` for stable hashing.  Empty
    tuple = the item is not on any rank axis (filler, R&D, Pad, PSI, etc.
    — those are gated separately).
    """
    axes: tuple[tuple[RankAxisKey, int], ...] = ()

    def rank_on(self, axis: RankAxisKey) -> Optional[int]:
        for k, v in self.axes:
            if k == axis:
                return v
        return None


_ITEM_RANK_SIG_CACHE: dict[tuple[str, RankContext], ItemRankSig] = {}


def rank_sig_for(item_name: str, ctx: RankContext = DEFAULT_CONTEXT) -> ItemRankSig:
    """Return the rank signature for an item name under the given context.

    Memoized per (item_name, ctx) pair so item_rule closures can share the
    same object reference and the equality check fast-paths to identity.
    """
    cache_key = (item_name, ctx)
    cached = _ITEM_RANK_SIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rankings = ranks_for_context(ctx)
    axes_for_item: list[tuple[RankAxisKey, int]] = []
    for axis in RANK_AXES:
        rank = rankings[axis.key].get(item_name)
        if rank is not None:
            axes_for_item.append((axis.key, rank))
    sig = ItemRankSig(tuple(sorted(axes_for_item, key=lambda x: x[0].value)))
    _ITEM_RANK_SIG_CACHE[cache_key] = sig
    return sig
