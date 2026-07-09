"""
Pure rocket-equation mathematics for KSP1 Archipelago.

All functions are side-effect-free and depend only on their arguments.
No imports from the rest of the KSP1 world package (aside from parts.py).

Golden rule: when there is ambiguity, be conservative — overestimate mass,
underestimate delta-v margin, underestimate asparagus benefit.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional

from ..parts import (
    Engine, FuelTank, SolidBooster, MultiMount,
    MAX_RADIAL_ENGINES,
)
from ..part_geometry import PartRole
from ..capability_reasons import StageDiagnostic, StageFailure

# Atmospheric-descent physics (entry bleed, chute ladder, touchdown bounds)
# lives in the aero submodule; exposed as rocket_math.aero.
from . import aero  # noqa: E402,F401  (import at end of header block is deliberate)

G0: float = 9.80665  # standard gravity, m/s²

def _isp_for_ascent(
    atm_isp: float,
    vac_isp: float,
    in_atmosphere: bool,
    atm_scale_height_m: float,
    atm_top_m: float,
    pad_altitude_m: float = 0.0,
) -> float:
    """Effective Isp for an ascent stage.

    KSP Isp is near-linear in atmospheric pressure: p(h) ≈ exp(-h/H) for
    scale height H, so Isp climbs from atm_isp at sea level to vac_isp
    near the top of the atmosphere.

    Vacuum bodies (or any call with ``in_atmosphere=False``) get ``vac_isp``
    directly.  Atmospheric bodies get a pressure-weighted average of the
    pressure fraction over the ascent column ``[pad_altitude_m, atm_top_m]``.
    An elevated launch site starts the burn where ambient pressure is already
    ``exp(-pad_altitude_m/H)`` of sea level, so the engines run nearer ``vac_isp``
    from the start — charging the sea-level column (pad=0) over-charges an
    elevated ascent (Eve's ~6.1 km mesa pad).  ``pad_altitude_m == 0`` (every
    sea-level pad) reduces to the original ``[0, atm_top_m]`` average, so this
    is byte-identical off the elevated-pad path.  All atmospheric parameters
    come from the Body — nothing here is Kerbin-specific.
    """
    if not in_atmosphere:
        return vac_isp
    if atm_scale_height_m <= 0.0 or atm_top_m <= 0.0:
        return atm_isp  # body has no atmospheric model — be conservative
    a = pad_altitude_m if pad_altitude_m > 0.0 else 0.0
    if a >= atm_top_m:
        return vac_isp  # pad above the modelled atmosphere → vacuum-Isp launch
    # Average pressure fraction over [a, atm_top_m]: ∫ exp(-h/H) dh / (top - a).
    avg_p = (atm_scale_height_m
             * (math.exp(-a / atm_scale_height_m)
                - math.exp(-atm_top_m / atm_scale_height_m))
             / (atm_top_m - a))
    return atm_isp * avg_p + vac_isp * (1.0 - avg_p)


# Stock-system constants — Kerbol's μ and Kerbin's solar orbital radius
# (KSP wiki values).  Used by ``hohmann_v_inf`` for interplanetary transfer
# math; ``bodies.planet_transfer_dv`` wraps it with per-Body lookups.
GM_SUN: float = 1.1723328e18                 # m³/s²
KERBIN_SOLAR_RADIUS_M: int = 13_599_840_256  # m


def hohmann_v_inf(r1: float, r2: float, mu: float) -> tuple[float, float]:
    """Hohmann transfer between circular orbits at radii ``r1`` and ``r2``
    around a body with gravitational parameter ``mu``.

    Returns the v∞ excess at each endpoint — the velocity above the local
    circular-orbit speed needed to enter (at r1) or leave (at r2) the
    transfer ellipse.  Both values are positive.  Plane-change cost is
    *not* included; the caller adds it as ``plane_change_dv`` on the
    transfer edge so the difficulty profile's ``plane_change_fraction``
    governs how much of it is actually paid.
    """
    if r1 == r2:
        return 0.0, 0.0
    a = (r1 + r2) / 2.0
    v_circ_1 = math.sqrt(mu / r1)
    v_trans_1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a))
    v_circ_2 = math.sqrt(mu / r2)
    v_trans_2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a))
    return abs(v_trans_1 - v_circ_1), abs(v_circ_2 - v_trans_2)

FILL_LEVELS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

# KSP symmetry tool modes. Radial boosters must use one of these counts.
# Parallel staging is now modelled for real by _find_optimal_parallel_stage
# (progressive radial shedding) — the old constant dry-mass factor is gone.
KSP_SYMMETRY_MODES: tuple[int, ...] = (2, 3, 4, 6, 8)


# Tanks whose fuel:dry ratios differ by less than this (relative to the best
# ratio) are the same "tier": ratio differences below the tier width are noise
# next to pack quantization, so within a tier the greedy pack prefers large
# tanks (realistic part counts).  Across tiers, better ratio always packs
# first — a materially worse tank (Mk3 fuselages, ~12.5% below the 8:1
# standard; oxidizer-drained LFO views at ~45% of native LF) must never lead
# the pack (bug 092).  Ordering, not a ratio *floor*: a floor relative to the
# best ratio makes packing non-monotone in the part set (adding a better tank
# drops existing tanks out of eligibility — with the old 0.15 floor, granting
# RCSTank1-2 at ratio 7.5 evicted every other monoprop tank), and a strictly
# larger kit must never lose a mission.
_TANK_RATIO_TIER_WIDTH = 0.05


def _tank_ratio(t):
    return t.fuel_mass / t.dry_mass if t.dry_mass > 0.0 else float("inf")


def _packable_tanks(tanks, engine, max_tank_size=None):
    """Return ``(pack_ctx, rho_star)``: the packing context for this engine
    and the best fuel:dry ratio among its mountable tanks.  ``pack_ctx`` is
    ``(packable, boundaries, cov_fuels, cov_tanks)`` — or ``None`` when the
    engine can mount nothing:

    * ``packable`` — the mountable, spine-stackable tanks, best ratio tier
      first, largest first within a tier.
    * ``boundaries`` — indices where ``packable`` crosses into a worse tier.
    * ``cov_fuels`` / ``cov_tanks`` — the covering-tank index: fuels sorted
      ascending, and for each position the lightest-dry tank among those
      holding at least that much (suffix-min).  ``bisect_left(cov_fuels,
      rem)`` answers "lightest tank that can hold ``rem``" in O(log n) —
      this is the hot leaf of the whole optimizer.

    Only tanks the ``engine`` can mount on survive — non-SPINE tanks (radial side
    tanks, single-node tanks like the FL-C1000, slanted/coupler adapters) can't
    form a central stackable column and are excluded, and the optimizer's size
    gate (``engine.size_class <= tank.size_class``) is applied.

    ``max_tank_size`` (set on shielded stages to the largest owned heat shield's
    size) excludes tanks WIDER than any shield can cover.  A reentry stage's
    widest tank must fit under a shield; a tank wider than every shield can never
    be shielded, so it can't be part of a coverable pack.  Without this the
    greedy pack (fuel-descending) could pick a wide un-coverable tank when a
    NARROWER pack of the same set was shieldable — and adding a mid-width tank to
    the kit would flip a feasible shielded stage infeasible (bug-092 shape,
    bugs/106).  Constraining the set keeps the pack coverable and monotone.

    Sort order is (ratio tier, fuel desc), NOT plain largest-first: a large
    tank whose ratio is materially worse (Mk3 fuselages, ~7:1 vs the 8:1
    standard) must not lead the greedy pack, or it drags every column's
    effective ratio down and can flip a high-dv stage infeasible even though a
    pack of better-ratio tanks in the SAME set closes it — a strictly larger
    kit losing a mission (bug 092, the monotonicity violation).  A real player
    packs the best-ratio tanks first; worse-ratio tanks (down to the
    oxidizer-drained LFO views a Nerv sees) only ever mop up remainders
    smaller than every better tank, or serve as the covering tank.  Tiers
    rather than raw ratios so a hairline ratio edge (the 0.3t toroid's 8.11 vs
    the big tanks' 8.00) can't force an 800-part pack of tiny tanks: within
    ``_TANK_RATIO_TIER_WIDTH`` the pack still prefers large tanks.

    ``boundaries`` marks the tier transitions: ``_pack_stop_plan`` may stop
    the greedy walk there and partial-fill a covering tank instead of letting
    a worse tier mop up (consuming a worse-ratio tank is not always cheaper
    than covering — and choosing wrongly is another way a larger kit packs
    heavier than a smaller one)."""
    mountable = [t for t in tanks
                 if PartRole.SPINE in t.roles and t.fuel_mass > 0.0
                 and engine.size_class <= t.size_class
                 and (max_tank_size is None or t.size_class <= max_tank_size)]
    if not mountable:
        return None, 0.0
    rho_star = max(_tank_ratio(t) for t in mountable)
    tier_span = rho_star * _TANK_RATIO_TIER_WIDTH
    def _tier(t):
        return int((rho_star - _tank_ratio(t)) / tier_span)
    mountable.sort(key=lambda t: (_tier(t), -t.fuel_mass))
    boundaries = tuple(i for i in range(1, len(mountable))
                       if _tier(mountable[i]) != _tier(mountable[i - 1]))
    # Covering index: walk fuel-descending keeping the running lightest-dry
    # tank (ties → smaller fuel: less over-capacity, the realistic pick), so
    # after reversing, position i holds the best cover among fuels >= fuel_i.
    by_fuel_desc = sorted(mountable, key=lambda t: -t.fuel_mass)
    cov_fuels: list[float] = []
    cov_tanks: list[FuelTank] = []
    best = None
    for t in by_fuel_desc:
        if (best is None or t.dry_mass < best.dry_mass
                or (t.dry_mass == best.dry_mass
                    and t.fuel_mass < best.fuel_mass)):
            best = t
        cov_fuels.append(t.fuel_mass)
        cov_tanks.append(best)
    cov_fuels.reverse()
    cov_tanks.reverse()
    # Plain float tuples for the hot walk — dataclass attribute access in
    # ``_pack_stop_plan`` (millions of calls/seed) measurably dominates it.
    fuels = tuple(t.fuel_mass for t in mountable)
    drys = tuple(t.dry_mass for t in mountable)
    cov_drys = tuple(t.dry_mass for t in cov_tanks)
    return (mountable, boundaries, tuple(cov_fuels), tuple(cov_tanks),
            fuels, drys, cov_drys), rho_star


def _merge_manifest(pack):
    """Collapse a pack ``[(count, tank), ...]`` to ``((count, name), ...)``,
    merging duplicate tank names and keeping largest-fuel-first order."""
    agg: dict[str, int] = {}
    rep: dict[str, FuelTank] = {}
    for n, t in pack:
        agg[t.name] = agg.get(t.name, 0) + n
        rep.setdefault(t.name, t)
    return tuple((agg[name], name)
                 for name in sorted(agg, key=lambda nm: rep[nm].fuel_mass,
                                    reverse=True))


def _pack_stop_plan(per_col, pack_ctx):
    """Choose the cheapest greedy stop point for ONE column, returning
    ``(stop_index, col_dry, cover)``: consume ``packable[:stop_index]``
    greedily (full tanks, in tier order), then partial-fill ``cover`` (or
    nothing when ``cover`` is None) to the leftover.

    Covering uses minimum DRY mass, not minimum fuel: partial fill makes
    spare capacity free, so a cover's dead mass is exactly its dry mass.
    Min-fuel was a correct proxy under uniform ratios, but with mixed ratios
    a smaller-fuel heavier-dry tank can slip in — and because min-dry is a
    pure min over the tank set, it is monotone in the kit (adding tanks can
    only improve the cover) where min-fuel is not.

    The greedy walk alone is not mass-minimal with mixed ratio tiers: letting
    a worse tier mop up a remainder can cost more dry mass than stopping at
    the tier boundary and partial-filling one better tank (a 4t ratio-7
    adapter consumed + covered costs 1.14t dry where one ratio-8 tank
    partial-filled costs 0.75t — the trial-60 regression).  So the stop point
    is the min of ``consumed_dry + cover_dry`` over the option points: before
    any consumption, at each tier boundary, and after the full walk.  A stop
    option only counts when some tank can actually hold the leftover — the
    end-of-walk option always can (greedy leaves the remainder below every
    scanned tank's fuel), so a plan always exists.  Evaluating stops ONLY at
    tier boundaries keeps this O(tiers) cover lookups per plan: within a tier
    ratios agree to ``_TANK_RATIO_TIER_WIDTH``, so mid-tier stops can't
    materially beat the boundary options, and the walk stays a single pass.
    This is the optimizer's hottest leaf (millions of calls per seed): cover
    lookups are O(log n) bisects on the precomputed index, boundary checks a
    cursor, and the ``remaining < fuel`` test skips the float division."""
    _, boundaries, cov_fuels, cov_tanks, fuels, drys, cov_drys = pack_ctx
    best_i = -1
    best_dry = float("inf")
    best_cov = None
    col_dry = 0.0
    remaining = per_col
    n = len(fuels)
    nb = len(boundaries)
    b = 0
    n_cov = len(cov_fuels)
    min_fuel = cov_fuels[0]  # sorted ascending — the smallest tank
    i = 0
    while True:
        if b < nb and boundaries[b] == i:
            b += 1
            at_option = True
        else:
            at_option = i == 0 or i == n
        # Once the leftover is below every tank's fuel, no later index can
        # consume anything — every remaining option point sees this exact
        # (col_dry, remaining), so evaluate it once here and stop instead of
        # walking the (long) tail of the list.
        exhausted = remaining < min_fuel
        if at_option or exhausted:
            if remaining <= 1e-12:
                if col_dry < best_dry:
                    best_i, best_dry, best_cov = i, col_dry, None
                break  # fully packed; consuming more only adds dry mass
            ci = bisect_left(cov_fuels, remaining)
            if ci != n_cov:
                total = col_dry + cov_drys[ci]
                if total < best_dry:
                    best_i, best_dry, best_cov = i, total, cov_tanks[ci]
            if exhausted:
                break
        if i == n:
            break
        f = fuels[i]
        if remaining >= f:
            k = int(remaining // f)
            col_dry += k * drys[i]
            remaining -= k * f
        i += 1
    return best_i, best_dry, best_cov


def _pack_columns(fuel_target, pack_ctx, cols):
    """Express ``fuel_target`` tonnes of fuel as the realistic tanks a player
    builds, returning ``(col, tank_dry)`` where ``col`` is ONE column's
    ``[(count, tank), ...]`` and tank_dry is the total (full) tank dry mass over
    all ``cols`` columns.  Each column holds ``fuel_target/cols`` of fuel as
    ``_pack_stop_plan`` prescribes: full tanks in packable order (best ratio
    first, largest within a tier) up to the chosen stop, plus one covering
    tank **partial-filled** to the leftover — so the packed fuel equals
    ``fuel_target`` exactly and the only dead dry mass is the covering
    tank's.

    ``cols`` (>=1) is the mounting floor: a non-radial multi-engine stage beyond
    adapter capacity replicates one column per engine, guaranteeing every engine
    has a tank to mount on.  The column list is built only on the committed
    winning stage; the sizing convergence uses the scalar ``_pack_dry``."""
    per_col = fuel_target / cols
    stop_i, col_dry, cov = _pack_stop_plan(per_col, pack_ctx)
    col: list[tuple[int, FuelTank]] = []
    remaining = per_col
    for t in pack_ctx[0][:stop_i]:
        k = int(remaining // t.fuel_mass)
        if k > 0:
            col.append((k, t))
            remaining -= k * t.fuel_mass
    if cov is not None:
        col.append((1, cov))
    return col, col_dry * cols


def _pack_dry(fuel_target, pack_ctx, cols):
    """Total full-tank dry mass for packing ``fuel_target`` into ``cols``
    columns — the allocation-free scalar the sizing convergence needs,
    identical to ``_pack_columns``'s dry by construction (both read the same
    ``_pack_stop_plan``)."""
    return _pack_stop_plan(fuel_target / cols, pack_ctx)[1] * cols


# Fixed-point fuel sizing converges geometrically; this bounds the rare slow
# case.  Non-convergence is treated as infeasible (conservative false negative).
_PACK_CONVERGE_ITERS = 8


def _size_and_pack(base, R_minus_1, isp_g0, sm_df, required_dv, rho_star,
                   pack_ctx, cols):
    """Size a stage's fuel to meet ``required_dv``, returning
    ``(fuel, tank_dry, actual_dv)`` or ``None`` if the tanks can't reach the dv.
    The caller builds the tank manifest from ``fuel`` only when this candidate
    wins — sizing itself stays allocation-free (``_pack_dry``).

    The mass and the dv come from the SAME pack, so they can't drift.  Sizing is
    a monotone fixed point: start optimistic (best ratio ``rho_star``, no dead
    dry), pack, measure the pack's true dv; if short, the fuel that would meet dv
    at the pack's *current* dry mass is strictly larger, so re-pack with it.
    Each step raises fuel (and dv) until dv is met.

    ``base`` = payload + engine mass; ``sm_df`` = staging dry factor (the dv sees
    reduced dry mass under asparagus/onion, but the assembled stage carries the
    full dry)."""
    denom = 1.0 - R_minus_1 * sm_df / rho_star
    if denom <= 0.0:
        return None  # best ratio still can't reach the dv
    fuel = R_minus_1 * base / denom
    for _ in range(_PACK_CONVERGE_ITERS):
        tank_dry = _pack_dry(fuel, pack_ctx, cols)
        verify_dry = base + tank_dry * sm_df
        if verify_dry <= 0.0:
            return None
        actual_dv = isp_g0 * math.log((verify_dry + fuel) / verify_dry)
        if actual_dv >= required_dv:
            return fuel, tank_dry, actual_dv
        # Fuel meeting dv at the current (real) dry mass.  The 1.0001 nudges the
        # target a hair past required so this monotone iteration *crosses* the
        # exact fixed point (where dv == required) instead of asymptoting to it
        # and stalling — critical when the pack is a single tank (dry constant).
        new_fuel = R_minus_1 * verify_dry * 1.0001
        if new_fuel <= fuel:
            return None  # not increasing → dv unreachable with these tanks
        fuel = new_fuel
    return None


# Parallel-unit fuel sizing.  parallel_stage_dv is monotone increasing in
# per-column fuel but has no closed inverse (segmented sum), so bisect — on a
# CHEAP closed-form dry estimate (dry ≈ fuel/rho_star, the best tank's ratio),
# then pack exactly only at the end and nudge up if the covering-tank dead mass
# under-shot.  This keeps the hot bisection allocation-free; packing exactly
# every iteration was the dominant cost (millions of _pack_dry calls).
_PARALLEL_BISECT_ITERS = 18  # on a tight ideal-seeded bracket → sub-0.001 precision
_PARALLEL_FUEL_GROW = 12     # 1.6x grows past the seed before declaring infeasible


def _size_parallel_unit(payload, e_mass, n_eng_core, n_eng_boost,
                        dec_mass, fl_mass, n_boost, mode,
                        required_dv, isp_g0, pack_ctx):
    """Size each identical column's fuel so the parallel unit (core +
    ``n_boost`` boosters) meets ``required_dv``, at minimum fuel.  Returns
    ``(col_fuel, col_tank_dry, actual_dv)`` or None if unreachable.

    Every column packs the same ``col_fuel`` from ``packable`` (so one
    column's tank dry mass is shared by core and boosters).  ``n_eng_boost``
    is 0 for drop-tank boosters (no engine), ``n_eng_core`` for engine
    boosters.  Each booster also carries ``dec_mass`` (radial decoupler) +
    ``fl_mass`` (fuel line).  Exact bisection of ``[0, hi]`` (grow ``hi`` from a
    rocket-equation seed until the dv is reached) so the returned build is the
    minimum-fuel one — the speedup comes from pruning the engine search, not
    from approximating the sizing."""
    def dv_exact(col_fuel):
        col_dry = _pack_dry(col_fuel, pack_ctx, 1)
        core_dry = n_eng_core * e_mass + col_dry
        booster_dry = n_eng_boost * e_mass + col_dry + dec_mass + fl_mass
        dv = parallel_stage_dv(isp_g0, payload, core_dry, col_fuel,
                               booster_dry, n_boost, mode)
        return dv

    R_minus_1 = math.exp(required_dv / isp_g0) - 1.0
    hi = max(R_minus_1 * (payload + e_mass * n_eng_core) / (n_boost + 1), 0.05)
    grow = 0
    dv_hi = dv_exact(hi)
    while dv_hi < required_dv and grow < _PARALLEL_FUEL_GROW:
        hi *= 2.0
        grow += 1
        dv_hi = dv_exact(hi)
    if dv_hi < required_dv:
        return None
    # Search DOWN from a sufficient hi to the true minimum.  ``dv_hi`` tracks the
    # dv at the current ``hi`` (always >= required), so the final hi's dv needs
    # no recompute — saves two dv_exact per call (grow-exit + return) vs always
    # re-evaluating, with byte-identical results.
    lo = 0.0
    for _ in range(_PARALLEL_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        dv_mid = dv_exact(mid)
        if dv_mid >= required_dv:
            hi, dv_hi = mid, dv_mid
        else:
            lo = mid
    col_dry = _pack_dry(hi, pack_ctx, 1)
    return hi, col_dry, dv_hi


# ---------------------------------------------------------------------------
# Core dataclass returned by find_optimal_stage
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result of a successful stage optimisation."""
    delta_v: float
    twr_at_ignition: float
    twr_at_burnout: float
    engine_is_throttleable: bool
    engine_has_gimbal: bool
    # Debug / propagation fields
    stage_mass_wet: float       # total wet mass of this stage (t), becomes payload for the next stage back
    stage_mass_dry: float       # total dry mass (t)
    engine_count: int
    fill_fraction: float
    engine_name: str
    # The exact tanks the solver tested and the player must build, as
    # ((count, tank_name), ...) largest-first.  Single source of truth for
    # this stage's tanks — display, kit, and sphere-ladder gating all read it,
    # so the reported parts and the verified mass can never drift apart.  Empty
    # for SRB stages (the booster is an integral engine+tank unit).
    tank_manifest: tuple[tuple[int, str], ...] = ()
    # Non-propulsion parts: [(count, part_id), ...]
    # Populated by _evaluate_profile after stage optimisation.
    equipment: list[tuple[int, str]] = field(default_factory=list)
    # Heat shield the optimizer charged for this stage (lightest shield
    # covering the chosen engine), if any.  Lets the kit capture the exact
    # shield used rather than the global biggest.
    heat_shield_name: Optional[str] = None
    # Parallel (asparagus/onion) staging: number of radial boosters dropped
    # progressively under this one "stage".  0 = a plain serial stage.  When
    # > 0, engine_count / tank_manifest are the FLATTENED totals across the
    # core + all booster columns (every column is identical), and equipment
    # carries the radial decouplers + fuel lines.  Display reads this for the
    # ``[ASPARAGUS - N boosters]`` tag.
    n_boosters: int = 0
    # Engines on EACH booster: 0 = drop-tank boosters (tank-only, fed to the
    # core), >0 = engine boosters (fire at liftoff for TWR, dropped with the
    # tank).  Disambiguates the flattened engine_count for display/build.
    booster_engines: int = 0
    # Per-candidate control surcharges the optimizer actually charged to this
    # winner (only ungimballed propulsion pays them).  The caller appends the
    # matching real parts to ``equipment`` iff the flag is set, so manifest
    # mass equals charged mass.  Presence of a wheel/fin in the KIT must never
    # force the charge — that made capability non-monotone (bug 092 family:
    # granting advSasModule flipped missions infeasible at the pad cap).
    carries_attitude_module: bool = False
    carries_aero_steering: bool = False


# ---------------------------------------------------------------------------
# Tsiolkovsky rocket equation — forward direction
# ---------------------------------------------------------------------------

def stage_delta_v(
    engine: Engine,
    engine_count: int,
    tank: FuelTank,
    tank_count: int,
    fill_fraction: float,
    payload_mass: float,           # tonnes
    in_atmosphere: bool = False,
) -> float:
    """Compute the delta-v produced by a single serial stage (Tsiolkovsky)."""
    isp = engine.atm_isp if in_atmosphere else engine.vac_isp
    if isp <= 0:
        return 0.0
    m_engine = engine.mass * engine_count
    m_tank_dry = tank.dry_mass * tank_count
    m_fuel = tank.fuel_mass * tank_count * fill_fraction

    m_dry = payload_mass + m_engine + m_tank_dry
    m_wet = m_dry + m_fuel
    if m_dry <= 0 or m_wet <= m_dry:
        return 0.0
    return isp * G0 * math.log(m_wet / m_dry)


def srb_delta_v(
    srb: SolidBooster,
    srb_count: int,
    payload_mass: float,
    in_atmosphere: bool = False,
) -> float:
    """Compute the delta-v produced by a cluster of SRBs."""
    isp = srb.atm_isp if in_atmosphere else srb.vac_isp
    if isp <= 0:
        return 0.0
    m_dry = payload_mass + srb.dry_mass * srb_count
    m_wet = m_dry + srb.fuel_mass * srb_count
    if m_dry <= 0 or m_wet <= m_dry:
        return 0.0
    return isp * G0 * math.log(m_wet / m_dry)


def parallel_stage_dv(
    isp_g0: float,          # isp * G0 (the effective ascent isp for the stage)
    payload: float,         # tonnes riding the core to the end of the burn
    core_dry: float,        # core column dry that burns to the end (core engines + core tank dry)
    col_fuel: float,        # fuel per column, tonnes (every column is identical)
    booster_dry: float,     # ONE booster's jettisoned dry: its engines (0 for drop-tanks) + tank dry + radial decoupler + fuel line
    n_boost: int,           # number of radial boosters (asparagus needs it even)
    mode: str,              # "asparagus" | "onion"
) -> float:
    """Vacuum delta-v of a parallel-staged unit: a central core column (carries
    the payload) ringed by ``n_boost`` identical radial booster columns, each
    holding ``col_fuel`` (the core holds ``col_fuel`` too).

    ``asparagus`` (fuel-line crossfeed, the chain C->A->Z): boosters drop in
    PAIRS.  The live outer pair feeds the whole stack, so its ``2*col_fuel`` is
    spent first while every inner tank stays full; the empty pair is then
    jettisoned.  Each pair's dry mass is therefore carried only until ``2*col_fuel``
    is burnt -- maximum progressive shedding, approaching the ideal as pairs grow
    (bounded in practice by the radial-decoupler + fuel-line mass folded into
    ``booster_dry``).

    ``onion`` (no crossfeed): every booster drains its own tank in step, so none
    can drop until they are ALL empty -- the whole ring is one coarse drop after
    ``n_boost*col_fuel`` is burnt, then the core.  Far less shedding, which is
    why it only wins when fuel lines (asparagus) are unavailable.

    Pure Tsiolkovsky; thrust/TWR is the caller's concern (asparagus fires every
    engine from liftoff, onion only the live ring).  Returns total dv (m/s)."""
    m = payload + core_dry + col_fuel + n_boost * (booster_dry + col_fuel)
    dv = 0.0
    if mode == "asparagus":
        for _ in range(n_boost // 2):
            m_end = m - 2.0 * col_fuel        # outer pair feeds the stack, drains
            if m_end <= 0.0:
                return 0.0
            dv += isp_g0 * math.log(m / m_end)
            m = m_end - 2.0 * booster_dry     # jettison the spent pair
            if m <= 0.0:
                return 0.0
    elif n_boost > 0:                          # onion: whole ring drops at once
        m_end = m - n_boost * col_fuel
        if m_end <= 0.0:
            return 0.0
        dv += isp_g0 * math.log(m / m_end)
        m = m_end - n_boost * booster_dry
        if m <= 0.0:
            return 0.0
    m_end = m - col_fuel                        # core burns last
    if m_end <= 0.0:
        return dv
    return dv + isp_g0 * math.log(m / m_end)


def parallel_stage_min_twr(
    thrust_per_eng: float,  # kN per engine (atmospheric or vacuum)
    gravity: float,         # m/s^2
    payload: float, core_dry: float, col_fuel: float, booster_dry: float,
    n_boost: int, n_eng_core: int, n_eng_boost: int, mode: str,
) -> float:
    """Minimum TWR across every phase of a parallel unit (the binding phase).

    Engine boosters are jettisoned with their tanks, so thrust steps DOWN each
    time a pair drops while the core engines alone carry the last phase.  TWR
    therefore is NOT necessarily worst at liftoff: a heavy payload on a few
    core engines can make the final core-only phase the tightest.  Every phase
    is checked at its heaviest instant (start, before that phase's fuel burns).
    Drop-tank boosters (n_eng_boost=0) keep thrust constant while mass falls,
    so liftoff binds — this returns that automatically.  0.0 if gravity<=0."""
    if gravity <= 0.0:
        return 0.0

    def twr_at(engines, mass):
        return engines * thrust_per_eng / (mass * gravity) if mass > 0.0 else 0.0

    m = payload + core_dry + col_fuel + n_boost * (booster_dry + col_fuel)
    live = n_boost
    worst = float("inf")
    if mode == "asparagus":
        for _ in range(n_boost // 2):
            worst = min(worst, twr_at(n_eng_core + live * n_eng_boost, m))
            m -= 2.0 * (col_fuel + booster_dry)
            live -= 2
    elif n_boost > 0:  # onion: the whole ring is one phase
        worst = min(worst, twr_at(n_eng_core + n_boost * n_eng_boost, m))
        m -= n_boost * (col_fuel + booster_dry)
    worst = min(worst, twr_at(n_eng_core, m))  # core-only phase
    return worst


# ---------------------------------------------------------------------------
# Inverse rocket equation — solve for required tank count
# ---------------------------------------------------------------------------

def required_tanks(
    engine: Engine,
    engine_count: int,
    tank: FuelTank,
    fill_fraction: float,
    delta_v: float,
    payload_mass: float,
    in_atmosphere: bool = False,
) -> int:
    """
    Return the minimum number of *tank* units to achieve *delta_v* with
    *engine_count* engines and *payload_mass* tonnes of payload.

    Returns -1 if the combination is physically impossible (e.g. the mass
    ratio required exceeds what the tank type can provide).
    """
    isp = engine.atm_isp if in_atmosphere else engine.vac_isp
    if isp <= 0 or delta_v <= 0:
        return -1
    R = math.exp(delta_v / (isp * G0))
    m_engine = engine.mass * engine_count

    numerator = (R - 1) * (payload_mass + m_engine)
    denominator = tank.fuel_mass * fill_fraction - (R - 1) * tank.dry_mass
    if denominator <= 0:
        return -1  # impossible: mass ratio exceeds tank capability
    return math.ceil(numerator / denominator)


# ---------------------------------------------------------------------------
# TWR
# ---------------------------------------------------------------------------

def twr(thrust_kn: float, mass_tonnes: float, gravity: float) -> float:
    """Thrust-to-weight ratio.  Returns 0 if mass is zero."""
    if mass_tonnes <= 0 or gravity <= 0:
        return 0.0
    return thrust_kn / (mass_tonnes * gravity)


# ---------------------------------------------------------------------------
# Terminal velocity (for parachute adequacy check)
# ---------------------------------------------------------------------------

def terminal_velocity(
    mass_tonnes: float,
    gravity: float,
    atm_density: float,             # kg/m³
    ship_cd: float,                 # dimensionless drag coefficient of craft body
    ship_cross_section: float,      # m², πr² of largest part
    total_chute_drag_area: float,   # m², sum of parachute fullyDeployedDrag values
) -> float:
    """
    Return the terminal velocity in m/s during parachute descent.

    ship_cd discounts the chute requirement by crediting some drag from the
    craft body itself.  Higher values are less conservative.
    """
    m_kg = mass_tonnes * 1000.0
    effective_drag = total_chute_drag_area + (ship_cd * ship_cross_section)
    if effective_drag <= 0 or atm_density <= 0:
        return float("inf")
    return math.sqrt((2.0 * m_kg * gravity) / (atm_density * effective_drag))


# ---------------------------------------------------------------------------
# Multi-mount adapter/plate helper
# ---------------------------------------------------------------------------

def _adapter_max_engines(e_size: float, t_size: float,
                         mounts: list[MultiMount]) -> int:
    """Best engine count from available adapters/plates for this engine+tank combo."""
    best = 0
    for mount in mounts:
        if mount.min_tank_size > 0 and t_size < mount.min_tank_size:
            continue
        # Find smallest engine_counts key >= e_size (pre-sorted)
        for max_size in mount._sorted_sizes:
            if e_size <= max_size:
                count = mount.engine_counts[max_size]
                if count > best:
                    best = count
                break
    return best


# ---------------------------------------------------------------------------
# Stage optimizer
# ---------------------------------------------------------------------------

# Asparagus/onion booster counts to search (radial symmetry; asparagus drops
# them in pairs).  Capped at 8 — the practical single-"stage" asparagus unit;
# beyond it the radial-decoupler + fuel-line overhead dominates anyway.
_PARALLEL_BOOSTER_COUNTS: tuple[int, ...] = (2, 4, 6, 8)
# Engines per column to escalate through for per-phase TWR before giving up.
_PARALLEL_MAX_ENG_PER_COL: int = 4

# Escalated build caps — the wider search space capability threads in ONLY for
# ascent edges named in ``data.feasibility.ESCALATED_ASCENT_EDGES``, and only
# inside Apollo-split evaluations.  Never the defaults: the common home-ascent
# path keeps the caps above (bug 094: a global K=3 cost 53-54% of all
# multistage builds for 0.3-1.3% mass wins and zero feasibility rescues — the
# rescues live on the hard tail these edges name, e.g. Eve's ~8 km/s ascent).
# Values are the smallest set measured to close Eve Return/Sample Return at
# max kit (probe 2026-07-02: uncrewed 124.5-355.2t, crewed 273.8-966.2t
# across zero/small/comfortable).
ESCALATED_BOOSTER_COUNTS: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16)
ESCALATED_MAX_ENG_PER_COL: int = 6
ESCALATED_MAX_ASCENT_STAGES: int = 3


def _lightest_covering_shield(
    size_class: float,
    heat_shields: tuple[tuple[float, float, str], ...],
) -> tuple[float, Optional[str]]:
    """Lightest shield whose size_class covers ``size_class``; (0.0, None) if
    none is needed/available."""
    best_m, best_n = float("inf"), None
    for ssz, sm, sn in heat_shields:
        if ssz >= size_class and sm < best_m:
            best_m, best_n = sm, sn
    return (best_m, best_n) if best_n is not None else (0.0, None)


def _find_optimal_parallel_stage(
    available_engines: list[Engine],
    required_dv: float,
    payload_mass: float,
    gravity: float,
    min_twr: float,
    mode: str,                       # "asparagus" | "onion"
    decoupler_mass: float,
    decoupler_name: str,
    fuel_line_mass: float,
    fuel_line_name: str,
    tanks_by_fuel_type: Optional[dict[str, list[FuelTank]]],
    available_tanks: list[FuelTank],
    in_atmosphere: bool = False,
    atm_scale_height_m: float = 0.0,
    atm_top_m: float = 0.0,
    pad_altitude_m: float = 0.0,
    requires_throttleable: bool = False,
    require_gimbal: bool = False,
    attitude_module_mass: float = 0.0,
    aero_steering_mass: float = 0.0,
    needs_heat_shield: bool = False,
    max_heat_shield_size: Optional[float] = None,
    heat_shields: tuple[tuple[float, float, str], ...] = (),
    best_wet_bound: float = float("inf"),
    booster_counts: tuple[int, ...] = _PARALLEL_BOOSTER_COUNTS,
    max_eng_per_col: int = _PARALLEL_MAX_ENG_PER_COL,
) -> Optional[StageResult]:
    """Lowest-wet-mass parallel-staged build (a core column ringed by identical
    radial boosters dropped progressively) for this stage, or None.

    Search: engine x booster-count x {drop-tank, engine-booster} x engines-per-
    column.  Drop-tanks are the default (shed dry mass for dv with no extra
    engines); engine-boosters escalate only when a phase is TWR-bound (the core-
    only phase usually binds — see ``parallel_stage_min_twr``).  Returns a
    FLATTENED StageResult: engine_count and tank_manifest sum the core + all
    booster columns (identical), equipment carries the radial decouplers + fuel
    lines, and ``n_boosters`` records the count.  ``best_wet_bound`` lets the
    caller pass the single-stage mass so we never return a heavier build."""
    if mode == "none" or required_dv <= 0:
        return None
    best: Optional[StageResult] = None
    best_wet = best_wet_bound
    has_twr = min_twr > 0 and gravity > 0
    twr_g = min_twr * gravity

    # Evaluate high-Isp engines first: their lower mass ratio gives the lightest
    # parallel build, which tightens ``best_wet`` early so the ideal-wet prune
    # below actually culls the rest (in arbitrary order the prune is inert —
    # asparagus beats the single-stage bound for almost every engine).
    for engine in sorted(available_engines, key=lambda e: -e.vac_isp):
        if needs_heat_shield and (max_heat_shield_size is None
                                  or engine.size_class > max_heat_shield_size):
            continue
        if require_gimbal and not engine.has_gimbal:
            continue
        if requires_throttleable and not engine.throttleable:
            continue
        isp = _isp_for_ascent(engine.atm_isp, engine.vac_isp, in_atmosphere,
                              atm_scale_height_m, atm_top_m, pad_altitude_m)
        if isp <= 0:
            continue
        isp_g0 = isp * G0
        e_mass = engine.mass
        thrust = engine.atm_thrust if in_atmosphere else engine.vac_thrust
        if has_twr and thrust <= 0:
            continue

        hs_mass, hs_name = (0.0, None)
        if needs_heat_shield:
            hs_mass, hs_name = _lightest_covering_shield(engine.size_class,
                                                         heat_shields)
            if hs_name is None:
                continue
        eng_payload = payload_mass + hs_mass + (
            attitude_module_mass + aero_steering_mass
            if not engine.has_gimbal else 0.0)

        # Free exact TWR gate: the unit always outweighs its payload, so its
        # liftoff TWR is strictly below total_thrust/(payload·g).  If even
        # the max-engine configuration (full core + engine boosters on every
        # column) can't hold the floor against the payload alone, no grid
        # point passes parallel_stage_min_twr — skip the engine.
        if has_twr and (max_eng_per_col * (1 + max(booster_counts))
                        * thrust) < twr_g * eng_payload:
            continue

        # Cheap, exact lower bound on this engine's lightest possible parallel
        # build: payload + one engine with ZERO effective dry (the unreachable
        # ideal of infinite shedding) needs wet = (payload+engine)·e^(dv/isp·g0).
        # A real build (tanks + boosters + decouplers, maybe more engines) is
        # strictly heavier, so if even this ideal can't beat the best serial
        # stage, no parallel build with this engine can — skip it.  Prunes the
        # low-Isp engines that dominate the 26-engine search cost.
        if (eng_payload + e_mass) * math.exp(required_dv / isp_g0) >= best_wet:
            continue

        if tanks_by_fuel_type is not None:
            compatible = tanks_by_fuel_type.get(engine.fuel_type, ())
        else:
            compatible = [t for t in available_tanks
                          if t.fuel_type == engine.fuel_type]
        if not compatible:
            continue
        if engine.size_class > max(t.size_class for t in compatible):
            continue
        # Shielded stage: exclude tanks wider than any owned shield can cover,
        # same monotone constraint as the serial path (bugs/106).
        _cap = (max_heat_shield_size
                if (needs_heat_shield and max_heat_shield_size is not None)
                else None)
        pack_ctx, _rho = _packable_tanks(compatible, engine, max_tank_size=_cap)
        if pack_ctx is None:
            continue

        # Reachability pre-gate: ONE sizing call at this engine's max-dv
        # configuration — most boosters, drop-tank columns, a single core
        # engine.  Every other grid point only adds dry mass (more core or
        # booster engines) or removes shed events (fewer boosters), so its
        # achievable dv is strictly lower.  If even this configuration can't
        # reach the target, no grid point can: skip the engine's whole
        # booster×type×count scan.  Result-identical by construction — it
        # prunes only provably-infeasible engines (the escalated
        # parallel_substages grids made exhaustive failure scans the
        # dominant cost, bug 093).
        if _size_parallel_unit(
                eng_payload, e_mass, 1, 0, decoupler_mass, fuel_line_mass,
                max(booster_counts), mode, required_dv, isp_g0,
                pack_ctx) is None:
            continue

        for n_boost in booster_counts:
            # Drop-tanks first (no booster engines); escalate to engine
            # boosters only if drop-tanks were TWR-bound.  If drop-tanks clear
            # TWR with a SINGLE core engine, the stage isn't thrust-bound, so
            # engine boosters (same fuel + extra engine mass) can only be
            # heavier — skip that pass entirely.
            droptank_single_eng = False
            for booster_has_engine in (False, True):
                if booster_has_engine and droptank_single_eng:
                    break
                for n in range(1, max_eng_per_col + 1):
                    n_be = n if booster_has_engine else 0
                    # Same exact TWR gate per config: liftoff TWR is bounded
                    # by these engines' thrust against the payload alone —
                    # skip the sizing bisection for configs that can't hold
                    # the floor (huge-payload sub-stage builds were sizing
                    # every config only to fail parallel_stage_min_twr).
                    if has_twr and (n + n_boost * n_be) * thrust < twr_g * eng_payload:
                        continue
                    res = _size_parallel_unit(
                        eng_payload, e_mass, n, n_be, decoupler_mass,
                        fuel_line_mass, n_boost, mode, required_dv, isp_g0,
                        pack_ctx)
                    if res is None:
                        continue
                    col_fuel, col_dry, actual_dv = res
                    core_dry = n * e_mass + col_dry
                    booster_dry = n_be * e_mass + col_dry + decoupler_mass + fuel_line_mass
                    m_wet = (eng_payload + core_dry + col_fuel
                             + n_boost * (booster_dry + col_fuel))
                    if m_wet >= best_wet:
                        break  # heavier than best; more core engines only worse
                    if has_twr:
                        mn_twr = parallel_stage_min_twr(
                            thrust, gravity, eng_payload, core_dry, col_fuel,
                            booster_dry, n_boost, n, n_be, mode)
                        if mn_twr < min_twr:
                            continue  # a phase is thrust-short → add engines
                    # Winner: flatten the identical columns into one manifest.
                    col0, _cd = _pack_columns(col_fuel, pack_ctx, 1)
                    manifest = _merge_manifest(
                        [(cnt * (1 + n_boost), t) for cnt, t in col0])
                    total_eng = n + n_boost * n_be
                    m_dry = m_wet - (1 + n_boost) * col_fuel
                    liftoff_thrust = total_eng * thrust
                    twr_ign = (liftoff_thrust / (m_wet * gravity)
                               if gravity > 0 else 0.0)
                    equip = [(n_boost, decoupler_name)]
                    if fuel_line_name:
                        equip.append((n_boost, fuel_line_name))
                    if n_be == 0 and n == 1:
                        droptank_single_eng = True  # not thrust-bound here
                    best_wet = m_wet
                    best = StageResult(
                        delta_v=actual_dv,
                        twr_at_ignition=twr_ign,
                        twr_at_burnout=twr_ign,  # liftoff binds; see min_twr
                        engine_is_throttleable=engine.throttleable,
                        engine_has_gimbal=engine.has_gimbal,
                        stage_mass_wet=m_wet,
                        stage_mass_dry=m_dry,
                        engine_count=total_eng,
                        fill_fraction=1.0,
                        engine_name=engine.name,
                        tank_manifest=manifest,
                        equipment=list(equip),
                        heat_shield_name=hs_name,
                        n_boosters=n_boost,
                        booster_engines=n_be,
                        carries_attitude_module=(
                            not engine.has_gimbal and attitude_module_mass > 0.0),
                        carries_aero_steering=(
                            not engine.has_gimbal and aero_steering_mass > 0.0),
                    )
                    break  # found the lowest-n feasible for this config
    return best


def _find_optimal_stage_uncached(
    available_engines: list[Engine],
    available_srbs: list[SolidBooster],
    available_tanks: list[FuelTank],
    required_dv: float,
    payload_mass: float,            # tonnes (terminal payload + equipment)
    gravity: float,                 # m/s² at the relevant body
    min_twr: float = 0.0,           # 0 = no TWR requirement
    requires_throttleable: bool = False,
    needs_heat_shield: bool = False,
    max_heat_shield_size: Optional[float] = None,
    heat_shields: tuple[tuple[float, float, str], ...] = (),  # (size_class, mass, name), per available shield
    in_atmosphere: bool = False,
    parallel_mode: str = "none",    # "none", "asparagus", or "onion"
    srb_needs_rcs: bool = True,     # if True, SRBs only valid when player has RCS
    player_has_rcs: bool = False,
    tanks_by_fuel_type: Optional[dict[str, list[FuelTank]]] = None,
    available_multi_mounts: Optional[list[MultiMount]] = None,
    require_gimbal: bool = False,
    attitude_module_mass: float = 0.0,  # added to payload only if chosen prop lacks gimbal
    aero_steering_mass: float = 0.0,    # ditto: fins for atmo gravity-turn steering
    diagnostic_out: Optional[list[StageDiagnostic]] = None,
    body_name: str = "",                  # for diagnostic reporting only
    launch_pad_mass_cap: float = float("inf"),  # for MASS_CAP_EXCEEDED diagnostic
    atm_scale_height_m: float = 0.0,    # body atmosphere model (0 = no atm)
    atm_top_m: float = 0.0,             # body atmosphere top in metres
    pad_altitude_m: float = 0.0,        # launch-site altitude (elevated-pad Isp credit)
    # Parallel-staging parts (as hashable primitives so the cache key stays
    # valid).  When present and parallel_mode != "none", a real radial
    # asparagus/onion build is tried alongside the serial search.
    radial_decoupler_mass: float = 0.0,
    radial_decoupler_name: str = "",
    fuel_line_mass: float = 0.0,
    fuel_line_name: str = "",
    # When False, skip the (expensive) real parallel build and return the
    # serial stage only.  The sphere-ladder bumper sets this so its many
    # probes use the conservative serial mass; the final build / display
    # leave it True for the exact asparagus rocket.
    run_parallel: bool = True,
    # Parallel-build search bounds.  Defaults are the hot-path caps; the
    # ESCALATED_* values arrive here only via capability's Apollo/eligible-
    # edge scoping.  Hashable, so they join the memo key automatically.
    booster_counts: tuple[int, ...] = _PARALLEL_BOOSTER_COUNTS,
    max_eng_per_col: int = _PARALLEL_MAX_ENG_PER_COL,
) -> Optional[StageResult]:
    """
    Find the minimum-mass engine+tank configuration that meets *required_dv*
    and all constraints.  Returns None if no valid configuration exists.

    Algorithm:
      For each eligible engine (sorted by Isp desc, mass asc):
        For each compatible tank (sorted by ratio desc, fuel_mass asc):
          For each fill level [1.0, 0.75, 0.5, 0.25]:
            Find min engine count satisfying TWR (analytical)
            Compute required tank count
            If valid, track minimum wet-mass configuration
            Use best-so-far as upper bound to skip unpromising combos
      Also evaluate SRBs (fixed config, no fill levels)
      Return best result or None

    When parallel_mode != "none", both parallel-benefit (symmetric engine
    counts with reduced tank dry mass) and non-parallel (all engine counts,
    full dry mass) are searched in a single pass.
    """
    best: Optional[StageResult] = None
    best_wet: float = float("inf")

    # Near-miss diagnostic accumulators. Inexpensive: a handful of scalar
    # updates per inner-loop iteration. Only synthesized at the end if no
    # result was found AND the caller asked for a diagnostic.
    _diag_engines_seen = 0          # engines that entered the loop
    _diag_engines_passed_filter = 0 # engines that survived size/throttle/gimbal/isp
    _diag_engines_hs_blocked = 0    # engines rejected for size > shield only
    _diag_engines_throttle_blocked = 0
    _diag_engines_gimbal_blocked = 0
    _diag_engines_isp_blocked = 0
    _diag_engines_no_tank = 0       # engines with empty compatible_tanks
    _diag_smallest_hs_blocked_size = float("inf")
    _diag_best_dv: float = 0.0
    _diag_best_twr: float = 0.0
    _diag_seen_dry_kills_ratio = False
    _diag_seen_engine_too_big = False
    _diag_min_too_big_engine_size = float("inf")  # smallest engine lacking a tank
    _diag_seen_twr_short = False
    _diag_engine_fuel_types: set[str] = set()
    _diag_mass_cap_blocked = False  # at least one combo would meet dv/twr but mass>cap

    # Base payload (heat shield is charged per-engine below, since the
    # lightest shield covering a given engine depends on the engine's size).
    full_payload = payload_mass

    # Per-engine heat shield: the LIGHTEST shield whose size_class covers
    # the engine.  Charging the lightest sufficient shield (rather than the
    # global biggest) keeps the model monotonic — adding a bigger, heavier
    # shield can never increase a stage's cost — and conservative (still a
    # real shield big enough for the part).  Returns (mass, name); memoized
    # by size_class (few discrete values) so the hot loop stays cheap.
    _hs_cache: dict[float, tuple[float, Optional[str]]] = {}
    def _hs_for(size_class: float) -> tuple[float, Optional[str]]:
        r = _hs_cache.get(size_class)
        if r is None:
            best_m, best_n = float("inf"), None
            for ssz, sm, sn in heat_shields:
                if ssz >= size_class and sm < best_m:
                    best_m, best_n = sm, sn
            r = (best_m, best_n) if best_n is not None else (0.0, None)
            _hs_cache[size_class] = r
        return r

    # No dry-mass factor: parallel-staging benefit is no longer faked here —
    # it is earned by the real radial drop-tank/booster unit in
    # _find_optimal_parallel_stage (called after this serial search).  This
    # path is always full-dry single-stage; ``_parallel`` only still enables
    # the symmetric engine-count option (radial engine clusters mounted on one
    # stage for TWR — distinct from asparagus, which drops tanks).
    _parallel = parallel_mode != "none"

    # Sub-modes: (dry_factor, use_symmetric_counts).  dry_factor is always 1.0
    # now; the two entries differ only in whether symmetric counts are tried.
    _sub_modes: list[tuple[float, bool]] = []
    if _parallel:
        _sub_modes.append((1.0, True))
    _sub_modes.append((1.0, False))

    # Pre-check: TWR denom component that depends only on engine
    has_twr = min_twr > 0 and gravity > 0
    twr_g = min_twr * gravity  # reused per engine

    # On a shielded stage every packed tank must fit under a shield: the widest
    # coverable tank is the largest owned shield's size.  Wider tanks can never
    # be shielded, so excluding them from the pack keeps it coverable AND
    # monotone (a wide un-coverable tank in the kit can't flip a shielded stage
    # infeasible — bugs/106).  ``None`` off the shielded path (no constraint).
    _shield_tank_cap = (max_heat_shield_size
                        if (needs_heat_shield
                            and max_heat_shield_size is not None)
                        else None)

    # Local refs to avoid repeated global/attribute lookups in hot loop
    _exp = math.exp
    _log = math.log
    _ceil = math.ceil
    _mounts = available_multi_mounts or []
    _max_radial = MAX_RADIAL_ENGINES
    # Cache adapter lookups: only ~36 unique (e_size, t_size) combos
    _adapter_cache: dict[tuple[float, float], int] = {}
    # Cache packable tank set per (fuel_type, engine size): engines sharing both
    # pack from the identical eligible set, so it's sorted/scanned once.
    _packable_cache: dict[tuple[str, float], tuple] = {}

    # -----------------------------------------------------------------------
    # Evaluate liquid engines + tanks
    # -----------------------------------------------------------------------
    for engine in available_engines:
        _diag_engines_seen += 1
        _diag_engine_fuel_types.add(engine.fuel_type)
        # Heat shield filter: engine must fit under the player's shield
        if needs_heat_shield:
            if max_heat_shield_size is None:
                _diag_engines_hs_blocked += 1
                if engine.size_class < _diag_smallest_hs_blocked_size:
                    _diag_smallest_hs_blocked_size = engine.size_class
                continue  # no shield at all
            if engine.size_class > max_heat_shield_size:
                _diag_engines_hs_blocked += 1
                if engine.size_class < _diag_smallest_hs_blocked_size:
                    _diag_smallest_hs_blocked_size = engine.size_class
                continue

        # Gimbal filter (atmospheric gravity turn without aero surfaces)
        if require_gimbal and not engine.has_gimbal:
            _diag_engines_gimbal_blocked += 1
            continue

        # Throttle filter
        if requires_throttleable and not engine.throttleable:
            _diag_engines_throttle_blocked += 1
            continue

        # Per-candidate control surcharges. When the caller passes a non-zero
        # `attitude_module_mass` (this stage needs an on-stage attitude
        # source) or `aero_steering_mass` (atmospheric gravity-turn steering
        # via fins), gimballed engines provide the control for free and
        # ungimballed engines must carry the real parts the caller selected.
        # Adding them only to ungimballed candidates lets the optimizer trade
        # "ungimballed + parts" against "gimballed alone" inside the same
        # search loop — presence of the parts in the kit never forces the
        # charge (bug 092: a mandated charge makes capability non-monotone).
        hs_mass_e, hs_name_e = _hs_for(engine.size_class) if needs_heat_shield else (0.0, None)
        eng_payload = full_payload + hs_mass_e + (
            attitude_module_mass + aero_steering_mass
            if not engine.has_gimbal else 0.0
        )

        e_mass = engine.mass
        e_size = engine.size_class
        thrust_per_eng = engine.atm_thrust if in_atmosphere else engine.vac_thrust

        # Engine lower bound: even with 1 engine + 1 smallest tank, can't beat best?
        if eng_payload + e_mass >= best_wet:
            continue

        # Compute mass ratio R once per engine (the key optimisation: avoids
        # redundant math.exp inside the tank/fill/engine-count loops).
        # For atmospheric ascent, use pressure-weighted Isp across the
        # atmospheric column rather than flat atm_isp — the burn spans
        # both regimes and Isp climbs to vacuum near the top of the column.
        isp = _isp_for_ascent(engine.atm_isp, engine.vac_isp, in_atmosphere,
                              atm_scale_height_m, atm_top_m, pad_altitude_m)
        if isp <= 0:
            _diag_engines_isp_blocked += 1
            continue
        _diag_engines_passed_filter += 1
        R = _exp(required_dv / (isp * G0))
        R_minus_1 = R - 1
        isp_g0 = isp * G0

        # TWR: denom that depends only on engine
        # thrust_per_eng - min_twr * gravity * engine.mass
        twr_eng_denom = thrust_per_eng - twr_g * e_mass if has_twr else 0.0

        # Use pre-indexed tanks if available, otherwise filter inline
        if tanks_by_fuel_type is not None:
            compatible_tanks = tanks_by_fuel_type.get(engine.fuel_type, ())
        else:
            compatible_tanks = [t for t in available_tanks if t.fuel_type == engine.fuel_type]
        if not compatible_tanks:
            _diag_engines_no_tank += 1
            continue

        # Engine bigger than EVERY compatible tank (the exact set the optimizer
        # pairs it with — already includes oxidizer-drained LFO views for LF
        # engines): it fits no tank at all.  Only this genuine case is
        # ENGINE_TOO_BIG_FOR_TANK; flag it, record the smallest such engine's
        # size (the min tank size a fix must satisfy), and skip the tank loop.
        # An engine that fits a big tank but not a small one is NOT too big —
        # that's handled per-tank below and must not mis-trip the classifier.
        if e_size > max(t.size_class for t in compatible_tanks):
            _diag_seen_engine_too_big = True
            if e_size < _diag_min_too_big_engine_size:
                _diag_min_too_big_engine_size = e_size
            continue

        # Pack-native tank selection.  Instead of searching single tank types,
        # the packer chooses the realistic tanks that hit the fuel target, and
        # the SAME pack yields the mass the dv/TWR are checked against — so the
        # reported parts and the tested mass are one object and can't drift.
        # Memoized per (fuel_type, size_class): every engine sharing those gets
        # the same eligible tank set, so we sort/scan it once per call.
        _pk_key = (engine.fuel_type, e_size)
        cached = _packable_cache.get(_pk_key)
        if cached is None:
            cached = _packable_tanks(compatible_tanks, engine,
                                     max_tank_size=_shield_tank_cap)
            _packable_cache[_pk_key] = cached
        pack_ctx, rho_star = cached
        if pack_ctx is None:
            _diag_engines_no_tank += 1
            continue
        if has_twr and twr_eng_denom <= 0:
            continue  # engine can't even lift its own weight at this TWR

        # Adapter capacity for this engine (widest mountable tank = most
        # permissive plate).  Non-radial stack engines beyond it each need their
        # own tank column.
        _widest_tank_size = max(t.size_class for t in pack_ctx[0])
        _akey = (e_size, _widest_tank_size)
        adapter_max = _adapter_cache.get(_akey)
        if adapter_max is None:
            adapter_max = _adapter_max_engines(e_size, _widest_tank_size, _mounts)
            _adapter_cache[_akey] = adapter_max
        if engine.radial_mountable:
            max_eng_base = _max_radial
            min_eng = 2  # radial engines need >= 2 for symmetric thrust
            use_radial_tank_constraint = False
        else:
            max_eng_base = max(adapter_max, _max_radial)
            min_eng = 1
            use_radial_tank_constraint = True
        # Analytical TWR floor: a lower bound on engine count from the fixed
        # payload+engine mass (fuel only makes TWR harder, so this never skips a
        # feasible higher count); the loop refines upward via the TWR check.
        if has_twr:
            min_eng = max(min_eng, _ceil(twr_g * eng_payload / twr_eng_denom))

        for sm_df, sm_symmetric in _sub_modes:
            if rho_star <= R_minus_1 * sm_df:
                _diag_seen_dry_kills_ratio = True
                continue  # best ratio can't reach the dv under this staging
            sm_max = (max(max_eng_base, 1 + KSP_SYMMETRY_MODES[-1])
                      if (_parallel and sm_symmetric) else max_eng_base)
            if min_eng > sm_max:
                # TWR floor needs more engines than can be mounted → thrust-short.
                if has_twr:
                    _diag_seen_twr_short = True
                continue
            if sm_symmetric:
                eng_counts = [1 + s for s in KSP_SYMMETRY_MODES
                              if min_eng <= 1 + s <= sm_max]
            else:
                eng_counts = range(min_eng, sm_max + 1)

            for n_eng in eng_counts:
                base = eng_payload + e_mass * n_eng
                if base >= best_wet:
                    break  # payload+engines alone already >= best
                # Mounting floor: non-radial stack engines past adapter capacity
                # get one tank column each so every engine has a mount.
                cols = (n_eng if (use_radial_tank_constraint and n_eng > 1
                                  and n_eng > adapter_max) else 1)
                res = _size_and_pack(base, R_minus_1, isp_g0, sm_df,
                                     required_dv, rho_star, pack_ctx, cols)
                if res is None:
                    continue
                m_fuel, m_tank_dry, actual_dv = res
                stage_hs_name = hs_name_e
                # The shield must cover the widest part of the stage facing entry
                # — max(engine, widest tank).  The shield was first sized to the
                # engine; if a wider tank (a radial engine ringing a big tank, or
                # a small stack engine under a big tank) needs more, re-size the
                # stage with the heavier shield folded into the payload so mass
                # and dv stay consistent.  Normal stacks (engine >= tank) are
                # unchanged — the wider-tank branch simply doesn't trigger.
                if needs_heat_shield and hs_name_e is not None:
                    _col0, _ = _pack_columns(m_fuel, pack_ctx, cols)
                    _wid = max((t.size_class for _n, t in _col0), default=0.0)
                    r_mass, r_name = _hs_for(_wid)
                    if r_name is None:
                        continue  # widest tank exceeds every shield → can't enter
                    if r_mass > hs_mass_e:
                        base = base + (r_mass - hs_mass_e)
                        res = _size_and_pack(base, R_minus_1, isp_g0, sm_df,
                                             required_dv, rho_star, pack_ctx,
                                             cols)
                        if res is None:
                            continue
                        m_fuel, m_tank_dry, actual_dv = res
                        stage_hs_name = r_name
                if actual_dv > _diag_best_dv:
                    _diag_best_dv = actual_dv
                m_dry = base + m_tank_dry
                m_wet = m_dry + m_fuel
                if m_wet >= best_wet:
                    break  # heavier than best; more engines only worse

                thrust = thrust_per_eng * n_eng
                if has_twr and thrust < twr_g * m_wet:
                    twr_actual = thrust / (m_wet * gravity) if gravity > 0 else 0.0
                    if twr_actual > _diag_best_twr:
                        _diag_best_twr = twr_actual
                    _diag_seen_twr_short = True
                    continue  # too little thrust → add engines

                if m_wet > launch_pad_mass_cap:
                    _diag_mass_cap_blocked = True
                twr_ign = thrust / (m_wet * gravity) if gravity > 0 else 0.0
                twr_bur = thrust / (m_dry * gravity) if gravity > 0 else 0.0
                # Build the manifest only now, on the committed winner — its dry
                # mass equals m_tank_dry (shared covering logic), so mass and
                # parts stay one consistent build.
                _col, _ = _pack_columns(m_fuel, pack_ctx, cols)
                manifest = _merge_manifest([(n * cols, t) for n, t in _col])
                best = StageResult(
                    delta_v=actual_dv,
                    twr_at_ignition=twr_ign,
                    twr_at_burnout=twr_bur,
                    engine_is_throttleable=engine.throttleable,
                    engine_has_gimbal=engine.has_gimbal,
                    stage_mass_wet=m_wet,
                    stage_mass_dry=m_dry,
                    engine_count=n_eng,
                    fill_fraction=1.0,
                    engine_name=engine.name,
                    tank_manifest=manifest,
                    heat_shield_name=stage_hs_name,
                    carries_attitude_module=(
                        not engine.has_gimbal and attitude_module_mass > 0.0),
                    carries_aero_steering=(
                        not engine.has_gimbal and aero_steering_mass > 0.0),
                )
                best_wet = m_wet
                break  # lightest TWR-passing count for this engine

    # -----------------------------------------------------------------------
    # Evaluate SRBs
    # -----------------------------------------------------------------------
    for srb in available_srbs:
        # SRBs can't throttle — skip throttle-required edges
        if requires_throttleable:
            continue
        # Gimbal filter (atmospheric gravity turn without aero surfaces)
        if require_gimbal and not srb.has_gimbal:
            continue
        # At casual/normal, SRBs need RCS for attitude/fine control
        if srb_needs_rcs and not player_has_rcs:
            continue
        # Heat shield filter
        if needs_heat_shield:
            if max_heat_shield_size is None:
                continue
            if srb.size_class > max_heat_shield_size:
                continue

        if srb.radial_mountable:
            max_srb = _max_radial
        else:
            max_srb = 1

        # Same pressure-weighted Isp treatment as the engine loop.
        srb_isp = _isp_for_ascent(srb.atm_isp, srb.vac_isp, in_atmosphere,
                                  atm_scale_height_m, atm_top_m, pad_altitude_m)
        if srb_isp <= 0:
            continue
        srb_thrust = (srb.atm_thrust if in_atmosphere else srb.vac_thrust)

        # Per-candidate control surcharges (see engine loop above).
        hs_mass_s, hs_name_s = _hs_for(srb.size_class) if needs_heat_shield else (0.0, None)
        srb_payload = full_payload + hs_mass_s + (
            attitude_module_mass + aero_steering_mass
            if not srb.has_gimbal else 0.0
        )

        srb_counts = range(1, max_srb + 1)

        for n_srb in srb_counts:
            # Inline srb_delta_v
            m_dry = srb_payload + srb.dry_mass * n_srb
            m_wet = m_dry + srb.fuel_mass * n_srb
            if m_dry <= 0 or m_wet <= m_dry:
                continue
            dv = srb_isp * G0 * _log(m_wet / m_dry)
            if dv < required_dv:
                continue

            thrust = srb_thrust * n_srb
            if has_twr and thrust < twr_g * m_wet:
                continue

            twr_ign = thrust / (m_wet * gravity) if gravity > 0 else 0.0
            twr_bur = thrust / (m_dry * gravity) if gravity > 0 else 0.0

            if best is None or m_wet < best_wet:
                best = StageResult(
                    delta_v=dv,
                    twr_at_ignition=twr_ign,
                    twr_at_burnout=twr_bur,
                    engine_is_throttleable=False,
                    engine_has_gimbal=srb.has_gimbal,
                    stage_mass_wet=m_wet,
                    stage_mass_dry=m_dry,
                    engine_count=n_srb,
                    fill_fraction=1.0,
                    engine_name=srb.name,
                    # SRB is an integral engine+tank unit — no tanks to pack.
                    tank_manifest=(),
                    heat_shield_name=hs_name_s,
                    carries_attitude_module=(
                        not srb.has_gimbal and attitude_module_mass > 0.0),
                    carries_aero_steering=(
                        not srb.has_gimbal and aero_steering_mass > 0.0),
                )
                best_wet = m_wet
            break  # found minimum SRB count, no need to try more

    # Real parallel (asparagus/onion) staging.  Build the radial drop-tank /
    # engine-booster unit and adopt it only if lighter than the best serial
    # stage.  The benefit is earned by progressively shedding real booster
    # columns (charged with radial decouplers + fuel lines), not the old flat
    # factor — and only when the player actually has those parts.  Run before
    # the diagnostic so it is synthesized only if BOTH serial and parallel fail.
    # Needs a radial decoupler to shed boosters; the fuel line (crossfeed) is
    # required for asparagus but absent for onion (capability only selects
    # asparagus when the player has fuel lines, so it'll be present there).
    if run_parallel and parallel_mode != "none" and radial_decoupler_name:
        par = _find_optimal_parallel_stage(
            available_engines=available_engines,
            required_dv=required_dv,
            payload_mass=full_payload,
            gravity=gravity,
            min_twr=min_twr,
            mode=parallel_mode,
            decoupler_mass=radial_decoupler_mass,
            decoupler_name=radial_decoupler_name,
            fuel_line_mass=fuel_line_mass,
            fuel_line_name=fuel_line_name,
            tanks_by_fuel_type=tanks_by_fuel_type,
            available_tanks=available_tanks,
            in_atmosphere=in_atmosphere,
            atm_scale_height_m=atm_scale_height_m,
            atm_top_m=atm_top_m,
            pad_altitude_m=pad_altitude_m,
            requires_throttleable=requires_throttleable,
            require_gimbal=require_gimbal,
            attitude_module_mass=attitude_module_mass,
            aero_steering_mass=aero_steering_mass,
            needs_heat_shield=needs_heat_shield,
            max_heat_shield_size=max_heat_shield_size,
            heat_shields=heat_shields,
            best_wet_bound=best.stage_mass_wet if best is not None else float("inf"),
            booster_counts=booster_counts,
            max_eng_per_col=max_eng_per_col,
        )
        if par is not None:
            best = par

    if best is None and diagnostic_out is not None:
        diagnostic_out.append(_synthesize_diagnostic(
            body_name=body_name,
            in_atmosphere=in_atmosphere,
            needs_heat_shield=needs_heat_shield,
            require_gimbal=require_gimbal,
            requires_throttleable=requires_throttleable,
            max_heat_shield_size=max_heat_shield_size or 0.0,
            min_twr=min_twr,
            gravity=gravity,
            required_dv=required_dv,
            payload_mass=full_payload,
            launch_pad_mass_cap=launch_pad_mass_cap,
            engines_seen=_diag_engines_seen,
            engines_passed_filter=_diag_engines_passed_filter,
            engines_hs_blocked=_diag_engines_hs_blocked,
            engines_throttle_blocked=_diag_engines_throttle_blocked,
            engines_gimbal_blocked=_diag_engines_gimbal_blocked,
            engines_no_tank=_diag_engines_no_tank,
            smallest_hs_blocked_size=_diag_smallest_hs_blocked_size,
            best_dv=_diag_best_dv,
            best_twr=_diag_best_twr,
            seen_dry_kills_ratio=_diag_seen_dry_kills_ratio,
            seen_engine_too_big=_diag_seen_engine_too_big,
            min_too_big_engine_size=_diag_min_too_big_engine_size,
            seen_twr_short=_diag_seen_twr_short,
            engine_fuel_types=tuple(sorted(_diag_engine_fuel_types)),
            mass_cap_blocked=_diag_mass_cap_blocked,
        ))

    return best


# ---------------------------------------------------------------------------
# find_optimal_stage cache
# ---------------------------------------------------------------------------
# Profile alternatives for the same body often share an early-stage edge
# (e.g. two Mun-orbit profiles both start with "Kerbin ascent to LKO" at
# the same payload).  Each shared edge invokes ``find_optimal_stage`` with
# identical arguments.  Caching dedupes those calls.
#
# Structural safety: the cache key is built from every signature parameter
# except those in ``_FOS_EXCLUDED_PARAMS``.  At module load
# ``_build_fos_key_spec`` validates that ``_FOS_EXCLUDED_PARAMS`` and
# ``_FOS_NORMALIZERS`` only reference names that exist in the function's
# signature.  Adding a new parameter to ``_find_optimal_stage_uncached``
# without updating either set causes the new parameter to enter the cache
# key automatically — correct by default.  Adding a parameter with an
# unhashable type produces a ``TypeError`` on the first call.

import inspect as _inspect


_FOS_CACHE: dict[tuple, "tuple[Optional[StageResult], tuple]"] = {}
_FOS_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
# Memory cap: in the rank-bumper world a single seed can produce ~80k
# unique cache keys (cumulative across spheres + bump iterations).  Each
# entry retains a StageResult + tuple of StageDiagnostic frames, ~3-5 KiB
# in practice.  Uncapped, the cache holds ~400 MiB worth of Python
# objects per worker; with 15 workers in a solve-check sweep that's
# enough to OOM a 30 GiB machine.  Cap at 16k entries with simple FIFO
# (popitem(last=False) is O(1) on dict in CPython 3.7+ insertion-order
# semantics).  Hit rate stays high because the bumper's hot loop hammers
# a small working set; older entries are cold.
_FOS_CACHE_MAX: int = 16_000
_FOS_SENTINEL = object()


def _fos_norm_part_list(parts):
    """Normalize a list of Engine/SolidBooster/FuelTank to a tuple of names."""
    return tuple(p.name for p in parts)


def _fos_norm_tanks_by_fuel_type(d):
    if d is None:
        return None
    return tuple(sorted(
        (k, tuple(t.name for t in v)) for k, v in d.items()
    ))


def _fos_norm_multi_mounts(mounts):
    # MultiMount has no .name field (instances live in MULTI_MOUNT_TABLE
    # keyed by string, but the field isn't carried).  Canonicalize by
    # the identity-defining content.  Order is preserved because the
    # underlying function iterates the list and order can affect tie-
    # breaking in candidate engine-mounting decisions.
    if mounts is None:
        return None
    return tuple(
        (m.min_tank_size, tuple(sorted(m.engine_counts.items())))
        for m in mounts
    )


# Parameters that aren't part of the input (output/identity-only).
# ``diagnostic_out`` is a mutable output channel; the wrapper captures
# and replays its content.
_FOS_EXCLUDED_PARAMS: frozenset[str] = frozenset({"diagnostic_out"})

# Normalizers convert non-hashable parameter values into a hashable form
# that preserves identity-relevant content.
_FOS_NORMALIZERS: dict = {
    "available_engines": _fos_norm_part_list,
    "available_srbs": _fos_norm_part_list,
    "available_tanks": _fos_norm_part_list,
    "tanks_by_fuel_type": _fos_norm_tanks_by_fuel_type,
    "available_multi_mounts": _fos_norm_multi_mounts,
}


def _build_fos_key_spec():
    """Validate ``_FOS_EXCLUDED_PARAMS`` and ``_FOS_NORMALIZERS`` against
    the signature of ``_find_optimal_stage_uncached``, then precompute a
    per-cacheable-parameter dispatch tuple (positional_index, name,
    normalizer_or_None, default).

    Raises ``ImportError`` at module load if either set names a parameter
    that no longer exists.  This is the structural check that catches a
    signature change that would otherwise silently drift the cache key.
    """
    sig = _inspect.signature(_find_optimal_stage_uncached)
    sig_names = set(sig.parameters.keys())
    unknown_excl = _FOS_EXCLUDED_PARAMS - sig_names
    if unknown_excl:
        raise ImportError(
            f"_FOS_EXCLUDED_PARAMS has names not in "
            f"_find_optimal_stage_uncached signature: {sorted(unknown_excl)}"
        )
    unknown_norm = set(_FOS_NORMALIZERS) - sig_names
    if unknown_norm:
        raise ImportError(
            f"_FOS_NORMALIZERS has names not in "
            f"_find_optimal_stage_uncached signature: {sorted(unknown_norm)}"
        )
    spec = []
    diag_index = -1
    diag_default = None
    for i, (name, p) in enumerate(sig.parameters.items()):
        if name == "diagnostic_out":
            diag_index = i
            diag_default = p.default if p.default is not _inspect.Parameter.empty else None
        if name in _FOS_EXCLUDED_PARAMS:
            continue
        default = (
            p.default if p.default is not _inspect.Parameter.empty else _FOS_SENTINEL
        )
        spec.append((i, name, _FOS_NORMALIZERS.get(name), default))
    return spec, diag_index, diag_default


_FOS_KEY_SPEC, _FOS_DIAG_INDEX, _FOS_DIAG_DEFAULT = _build_fos_key_spec()


def clear_find_optimal_stage_cache() -> None:
    """Reset the find_optimal_stage cache and its hit/miss counters."""
    _FOS_CACHE.clear()
    for k in _FOS_CACHE_STATS:
        _FOS_CACHE_STATS[k] = 0


def get_find_optimal_stage_cache_stats() -> dict[str, int]:
    """Snapshot the cache hit/miss counters."""
    return dict(_FOS_CACHE_STATS)


def find_optimal_stage(*args, **kwargs):
    """Cache wrapper around ``_find_optimal_stage_uncached``.  See that
    function for the underlying behavior.

    ``diagnostic_out`` is excluded from the cache key but its content is
    captured and replayed: the inner function is always invoked with our
    own internal list, and any diagnostics it appends are both stored in
    the cache and forwarded to the caller's ``diagnostic_out`` (if any).
    """
    # Build cache key from every cacheable parameter.
    key_parts = []
    for idx, name, norm, default in _FOS_KEY_SPEC:
        if idx < len(args):
            v = args[idx]
        elif name in kwargs:
            v = kwargs[name]
        else:
            v = default
        key_parts.append(norm(v) if norm is not None else v)
    key = tuple(key_parts)

    # Locate the caller's diagnostic_out (may be positional, kwarg, or absent).
    if _FOS_DIAG_INDEX != -1 and _FOS_DIAG_INDEX < len(args):
        caller_diag = args[_FOS_DIAG_INDEX]
    else:
        caller_diag = kwargs.get("diagnostic_out", _FOS_DIAG_DEFAULT)

    cached = _FOS_CACHE.get(key, _FOS_SENTINEL)
    if cached is not _FOS_SENTINEL:
        _FOS_CACHE_STATS["hits"] += 1
        result, cached_diag = cached
        if caller_diag is not None and cached_diag:
            caller_diag.extend(cached_diag)
        return result

    _FOS_CACHE_STATS["misses"] += 1
    # Always run with our own internal diagnostic_out so we can capture
    # any appends.  This means the inner function always treats
    # diagnostic_out as non-None — synthesis cost on infeasibility is
    # ~3μs and only runs on the failure path, so this is fine.
    internal_diag: list = []
    if _FOS_DIAG_INDEX != -1 and _FOS_DIAG_INDEX < len(args):
        new_args = list(args)
        new_args[_FOS_DIAG_INDEX] = internal_diag
        result = _find_optimal_stage_uncached(*new_args, **kwargs)
    else:
        new_kwargs = dict(kwargs)
        new_kwargs["diagnostic_out"] = internal_diag
        result = _find_optimal_stage_uncached(*args, **new_kwargs)

    cached_diag = tuple(internal_diag)
    if len(_FOS_CACHE) >= _FOS_CACHE_MAX:
        # FIFO eviction: pop the oldest entry.  In CPython 3.7+, dicts
        # preserve insertion order, so iter(_FOS_CACHE) yields the
        # earliest key first.
        oldest = next(iter(_FOS_CACHE))
        del _FOS_CACHE[oldest]
        _FOS_CACHE_STATS["evictions"] += 1
    _FOS_CACHE[key] = (result, cached_diag)
    if caller_diag is not None and internal_diag:
        caller_diag.extend(internal_diag)
    return result


# ---------------------------------------------------------------------------
# Multi-stage atmospheric ascent (F4)
# ---------------------------------------------------------------------------
# Replaces the single-stage ascent assumption for atmospheric groups.
# Real KSP launches use 2-3 stages from launchpad to LKO; the Tsiolkovsky
# product-of-mass-ratios across K stages beats a single-stage mass ratio
# by 30-50% on heavy missions.
#
# Per-stage role and floors (planet-agnostic; same shape for any
# atmospheric body — Kerbin, Eve, Laythe, Duna):
#   Stage 1 (bottom, in atmosphere): atm TWR floor 1.2, weighted Isp.
#   Middle stages (vacuum, sustainer): TWR floor 1.0, vac Isp.
#   Top stage (circularisation): TWR floor 0.8 (horizontal burn near
#     apoapsis, gravity drag cosine; not zero because still suborbital).
#
# K is gated by the number of stack decouplers the player has unlocked:
# K-1 interstages are needed, each charged at the lightest stack
# decoupler's real mass.

# Δv split fractions to grid-search per K.  Bottom stage first; each row
# sums to 1.0.  Hand-picked spread covering equal-mass-ratio (high
# fraction on low-Isp stages) through vacuum-heavy splits.
_F4_DV_SPLITS: dict[int, tuple[tuple[float, ...], ...]] = {
    1: ((1.0,),),
    2: (
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
    ),
    # K=3 is reachable only via ``max_ascent_stages`` (the escalated bound —
    # _F4_MAX_K stays 2 on the hot path, bug 094).  Grid covers bottom-heavy
    # through top-heavy allocations (index 0 = bottom/launch stage); the
    # coarse 6-entry grid of the old K=3 era undervalued staging on high-dv
    # ascents (bug 094 cause #3).
    3: (
        (0.2, 0.3, 0.5),
        (0.25, 0.25, 0.5),
        (0.3, 0.3, 0.4),
        (0.3, 0.4, 0.3),
        (0.34, 0.33, 0.33),
        (0.4, 0.3, 0.3),
        (0.5, 0.25, 0.25),
        (0.5, 0.3, 0.2),
    ),
}

# Per-stage TWR floor.  Stage 1 (liftoff) uses whatever the caller passes
# in ``min_twr_liftoff`` — atmospheric ascent on Kerbin is ~1.2, vacuum
# ascent on Mun is ~1.2 (different body but same numeric floor in normal
# difficulty), Moho ascent is similar.  Middle and top stages are
# hardcoded universal values per role.
_F4_TWR_MIDDLE: float = 1.0          # sustainer (vacuum, already moving)
_F4_TWR_TOP_CIRCULARIZE: float = 0.8 # circularisation (near-orbital, horizontal burn)

# Max ascent stages.  Capped at 2: measured across 50k+ ascent evaluations,
# K=3 rescued feasibility 0 times (K≤2 is the feasibility frontier) and beat
# K=2 on launch mass only ~1% of the time, while accounting for ~54% of
# multi-stage stage-builds.  Dropping K=3 is feasibility-identical and at most
# ~1% mass-conservative (the Golden-Rule-safe direction).
_F4_MAX_K: int = 2


def find_optimal_multistage_ascent(
    required_dv: float,
    payload_mass: float,
    gravity: float,
    *,
    in_atmosphere: bool,         # stage 1 in atm? (False for vacuum-body ascent)
    min_twr_liftoff: float,      # TWR floor for stage 1 (caller's body+difficulty)
    available_engines: list[Engine],
    available_tanks: list[FuelTank],
    available_srbs: list[SolidBooster],
    tanks_by_fuel_type: Optional[dict[str, list[FuelTank]]] = None,
    available_multi_mounts: Optional[list[MultiMount]] = None,
    stack_decoupler: Optional[Decoupler],
    staging_tier: int,
    needs_heat_shield: bool = False,
    max_heat_shield_size: Optional[float] = None,
    heat_shields: tuple[tuple[float, float], ...] = (),
    requires_throttleable: bool = False,
    require_gimbal: bool = False,
    srb_needs_rcs: bool = True,
    player_has_rcs: bool = False,
    attitude_module_mass: float = 0.0,
    aero_steering_mass: float = 0.0,
    body_name: str = "",
    launch_pad_mass_cap: float = float("inf"),
    atm_scale_height_m: float = 0.0,
    atm_top_m: float = 0.0,
    pad_altitude_m: float = 0.0,  # elevated launch-site altitude (Isp credit)
    parallel_mode: str = "none",  # asparagus/onion — K=1 only unless parallel_substages
    radial_decoupler_mass: float = 0.0,
    radial_decoupler_name: str = "",
    fuel_line_mass: float = 0.0,
    fuel_line_name: str = "",
    run_parallel: bool = True,
    diagnostic_out: Optional[list[StageDiagnostic]] = None,
    partial_stages_out: Optional[list[StageResult]] = None,
    # Escalated search bounds (capability's Apollo/eligible-edge scoping).
    # ``max_ascent_stages`` None = the hot-path default: _F4_MAX_K, further
    # bounded by staging_tier+1.  An explicit value bypasses the tier bound
    # (a search-cost choice, not physics — stack decouplers permit arbitrary
    # serial depth) but never the no-stack-decoupler ⇒ K=1 kit gate.
    max_ascent_stages: Optional[int] = None,
    booster_counts: tuple[int, ...] = _PARALLEL_BOOSTER_COUNTS,
    max_eng_per_col: int = _PARALLEL_MAX_ENG_PER_COL,
    parallel_substages: bool = False,
    # Architecture guide (offline lifter-chain bindings): pins the search to
    # one known-good architecture — (dv_split, per-stage
    # (n_boosters, booster_engines, engine_name)) — turning the K x splits x
    # engines x booster-grid search into K single-architecture stage sizings
    # (~150x measured).  The guided result is a real optimizer build; only
    # the SEARCH is skipped.
    guide: Optional[tuple[tuple[float, ...],
                          tuple[tuple[int, int, str], ...]]] = None,
) -> Optional[list[StageResult]]:
    """Find lowest-total-wet K-stage ascent architecture for an atmospheric
    body.  Returns a list of StageResults from BOTTOM (launch) to TOP
    (circularisation), or None if no architecture is feasible.

    Each stage is independently optimised by ``find_optimal_stage``;
    payloads chain top-down, decoupler mass charged on every interstage.

    ``parallel_mode`` (asparagus/onion) applies to K=1 always; with
    ``parallel_substages`` each serial sub-stage may ALSO build as a real
    radial cluster (bug 093) — the architecture real high-dv ascents fly
    (an asparagus launcher delivering an asparagus cruise stage).  Since
    bf9a2c51 the parallel model sizes each cluster exactly (real column
    masses, real drop schedule), so composing it with serial staging
    double-counts nothing: each sub-stage's shed is its own hardware, the
    interstage decoupler drops the whole sub-stage.  Default False is a
    SEARCH-COST bound, not physics — every sub-stage build widens to the
    full parallel-unit search when enabled.
    """
    if required_dv <= 0:
        return None
    # K cap from kit: no stack decoupler ⇒ K=1; otherwise up to _F4_MAX_K
    # (or the caller's escalated bound).
    if stack_decoupler is None:
        max_K = 1
    elif max_ascent_stages is not None:
        max_K = max(1, max_ascent_stages)
    else:
        max_K = min(_F4_MAX_K, max(1, staging_tier + 1))
    if guide is not None:
        guide_split, guide_stages = guide
        # The no-stack-decoupler ⇒ K=1 gate is physics, never bypassed.
        if stack_decoupler is None and len(guide_stages) > 1:
            return None
    deco_mass = stack_decoupler.mass if stack_decoupler else 0.0

    best: Optional[list[StageResult]] = None
    best_launch_wet = float("inf")

    # Track the closest infeasible attempt for diagnostics.  "Closest" =
    # most stages built before the binding stage failed, tie-broken by the
    # smallest dv shortfall on that failing stage.  Lets the caller surface
    # a real StageDiagnostic + partial rocket for an otherwise-opaque
    # NO_VIABLE_STAGE on the multistage ascent path.
    closest_diag: Optional[StageDiagnostic] = None
    closest_partial: list[StageResult] = []
    closest_score: tuple[int, float] = (-1, -float("inf"))  # (stages_built, -dv_gap)

    if guide is not None:
        k_splits: list[tuple[int, tuple[tuple[float, ...], ...]]] = [
            (len(guide_stages), (guide_split,))]
    else:
        k_splits = [(K, _F4_DV_SPLITS[K]) for K in range(1, max_K + 1)]
    for K, splits in k_splits:
        for split in splits:
            # Build top-down: top stage first (carries mission payload),
            # lower stages carry wet of stages above + decoupler.
            stages_top_to_bot: list[StageResult] = []
            current_payload = payload_mass
            ok = True
            for i in range(K - 1, -1, -1):  # K-1 (top) ... 0 (bottom)
                # Exact best-so-far prune: every stage's wet mass exceeds
                # its payload, so once the running payload alone matches the
                # best launch mass found, no lower stage can improve on it —
                # abandon this split before paying its remaining builds.
                if current_payload >= best_launch_wet:
                    ok = False
                    break
                stage_dv = required_dv * split[i]
                # Stage 1 (i==0): launch (uses the caller's body-aware
                # in_atmosphere flag — True for atm bodies, False for vac).
                # Stages above: always vacuum (post-stage-1 altitude reached).
                stage_in_atm = in_atmosphere and (i == 0)
                # TWR floor by role.  Stage 1 uses caller's body+difficulty
                # floor; middle/top use universal sustainer/circularisation.
                if i == 0:
                    twr_floor = min_twr_liftoff
                elif i == K - 1:
                    twr_floor = _F4_TWR_TOP_CIRCULARIZE
                else:
                    twr_floor = _F4_TWR_MIDDLE
                # Heat shield mass charge only on the bottom stage (it
                # carries the shield up through atmosphere).  Upper stages
                # see it propagated via current_payload (no double-charge).
                stage_heat_shields = heat_shields if i == 0 else ()
                stage_needs_heat_shield = needs_heat_shield if i == 0 else False
                stage_engines = available_engines
                stage_srbs = available_srbs if stage_in_atm else []
                stage_bc = booster_counts
                stage_mec = max_eng_per_col
                stage_pmode = (parallel_mode
                               if (K == 1 or parallel_substages) else "none")
                if guide is not None:
                    g_nb, g_beng, g_eng = guide_stages[i]
                    if g_nb > 0:
                        # This stage WAS a parallel cluster at bind time —
                        # force the parallel build regardless of the
                        # parallel_substages search-cost bound (the guide
                        # already picked the architecture, so there is no
                        # search to bound).
                        stage_bc = (g_nb,)
                        stage_pmode = parallel_mode
                        if g_beng:
                            stage_mec = min(max_eng_per_col, max(1, g_beng))
                    else:
                        stage_pmode = "none"
                    if any(s.name == g_eng for s in stage_srbs):
                        stage_srbs = [s for s in stage_srbs
                                      if s.name == g_eng]
                        stage_engines = []
                    else:
                        stage_engines = [e for e in stage_engines
                                         if e.name == g_eng]
                        stage_srbs = []
                    if not stage_engines and not stage_srbs:
                        # Guide references a part outside this kit — the
                        # caller's prefix guarantee was violated; fail the
                        # architecture rather than silently widening.
                        ok = False
                        break
                inner_diag: list[StageDiagnostic] = []
                stage = find_optimal_stage(
                    available_engines=stage_engines,
                    available_srbs=stage_srbs,
                    available_tanks=available_tanks,
                    required_dv=stage_dv,
                    payload_mass=current_payload,
                    gravity=gravity,
                    min_twr=twr_floor,
                    requires_throttleable=requires_throttleable,
                    needs_heat_shield=stage_needs_heat_shield,
                    max_heat_shield_size=max_heat_shield_size,
                    heat_shields=stage_heat_shields,
                    in_atmosphere=stage_in_atm,
                    # K=1 may always build a real radial asparagus/onion unit.
                    # Sub-stages of a serial stack may too when the caller
                    # opts in (parallel_substages, bug 093): per-cluster
                    # sizing is exact, so nothing double-counts — the flag is
                    # a search-cost bound only.
                    parallel_mode=stage_pmode,
                    radial_decoupler_mass=radial_decoupler_mass,
                    radial_decoupler_name=radial_decoupler_name,
                    fuel_line_mass=fuel_line_mass,
                    fuel_line_name=fuel_line_name,
                    run_parallel=run_parallel,
                    booster_counts=stage_bc,
                    max_eng_per_col=stage_mec,
                    srb_needs_rcs=srb_needs_rcs,
                    player_has_rcs=player_has_rcs,
                    tanks_by_fuel_type=tanks_by_fuel_type,
                    available_multi_mounts=available_multi_mounts,
                    require_gimbal=require_gimbal and stage_in_atm,
                    # The wheel/RCS bundle rides the TOP stage (it serves the
                    # whole flight); fins matter on the sub-stages that fly
                    # through atmosphere.
                    attitude_module_mass=attitude_module_mass if i == K - 1 else 0.0,
                    aero_steering_mass=aero_steering_mass if stage_in_atm else 0.0,
                    body_name=body_name,
                    launch_pad_mass_cap=launch_pad_mass_cap,
                    atm_scale_height_m=atm_scale_height_m if stage_in_atm else 0.0,
                    atm_top_m=atm_top_m if stage_in_atm else 0.0,
                    pad_altitude_m=pad_altitude_m if stage_in_atm else 0.0,
                    diagnostic_out=inner_diag,
                )
                if stage is None:
                    ok = False
                    fail_diag = inner_diag[0] if inner_diag else None
                    dv_gap = (
                        fail_diag.required_dv - fail_diag.best_dv_achieved
                        if fail_diag else float("inf")
                    )
                    score = (len(stages_top_to_bot), -dv_gap)
                    if score > closest_score:
                        closest_score = score
                        closest_diag = fail_diag
                        # Stages built so far are top-to-bottom; reverse to
                        # bottom-to-top so the partial matches the success
                        # convention (launch stage first).
                        closest_partial = list(reversed(stages_top_to_bot))
                    break
                stages_top_to_bot.append(stage)
                if i > 0:
                    current_payload = stage.stage_mass_wet + deco_mass
                else:
                    current_payload = stage.stage_mass_wet  # launch mass
            if not ok:
                continue
            launch_wet = stages_top_to_bot[-1].stage_mass_wet
            if launch_wet < best_launch_wet:
                best_launch_wet = launch_wet
                # Store bottom-to-top: reverse so the caller iterates
                # launch first, ending at circularisation.
                best = list(reversed(stages_top_to_bot))
                # Annotate each stage's equipment with the interstage
                # decoupler that drops it (bottom of stage K-1 carries
                # the decoupler that releases stage K, etc.). Bottom
                # stage carries no decoupler below it (it sits on the pad).
                if K > 1 and stack_decoupler is not None:
                    for stage_idx in range(K - 1):
                        # Stage at index stage_idx carries the decoupler
                        # that drops the stage above (stage_idx + 1).
                        best[stage_idx].equipment.append(
                            (1, stack_decoupler.name)
                        )
    if best is None:
        if diagnostic_out is not None and closest_diag is not None:
            diagnostic_out.append(closest_diag)
        if partial_stages_out is not None:
            partial_stages_out.extend(closest_partial)
    return best


def _synthesize_diagnostic(
    *,
    body_name: str,
    in_atmosphere: bool,
    needs_heat_shield: bool,
    require_gimbal: bool,
    requires_throttleable: bool,
    max_heat_shield_size: float,
    min_twr: float,
    gravity: float,
    required_dv: float,
    payload_mass: float,
    launch_pad_mass_cap: float,
    engines_seen: int,
    engines_passed_filter: int,
    engines_hs_blocked: int,
    engines_throttle_blocked: int,
    engines_gimbal_blocked: int,
    engines_no_tank: int,
    smallest_hs_blocked_size: float,
    best_dv: float,
    best_twr: float,
    seen_dry_kills_ratio: bool,
    seen_engine_too_big: bool,
    min_too_big_engine_size: float,
    seen_twr_short: bool,
    engine_fuel_types: tuple[str, ...],
    mass_cap_blocked: bool,
) -> StageDiagnostic:
    """Classify the dominant near-miss from accumulator state.

    Order matters — pick the *earliest* failure mode encountered, since
    fixing earlier ones is a prerequisite for the later ones.
    """
    # No engine ever survived basic filters.
    if engines_passed_filter == 0:
        # Specific filter reason — which one dominated?
        if engines_hs_blocked and engines_hs_blocked == engines_seen:
            return StageDiagnostic(
                failure=StageFailure.HEAT_SHIELD_TOO_SMALL,
                body=body_name, in_atmosphere=in_atmosphere,
                group_needs_heat_shield=needs_heat_shield,
                smallest_filtered_engine_size=(
                    smallest_hs_blocked_size
                    if smallest_hs_blocked_size != float("inf") else 0.0
                ),
                current_max_shield_size=max_heat_shield_size,
                required_dv=required_dv, payload_mass=payload_mass,
                engine_fuel_types_attempted=engine_fuel_types,
            )
        if require_gimbal and engines_gimbal_blocked == engines_seen:
            return StageDiagnostic(
                failure=StageFailure.REQUIRE_GIMBAL_NONE,
                body=body_name, in_atmosphere=in_atmosphere,
                required_dv=required_dv, payload_mass=payload_mass,
                engine_fuel_types_attempted=engine_fuel_types,
            )
        if requires_throttleable and engines_throttle_blocked == engines_seen:
            return StageDiagnostic(
                failure=StageFailure.REQUIRE_THROTTLE_NONE,
                body=body_name, in_atmosphere=in_atmosphere,
                required_dv=required_dv, payload_mass=payload_mass,
                engine_fuel_types_attempted=engine_fuel_types,
            )
        return StageDiagnostic(
            failure=StageFailure.NO_ENGINES_AFTER_FILTER,
            body=body_name, in_atmosphere=in_atmosphere,
            group_needs_heat_shield=needs_heat_shield,
            required_dv=required_dv, payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
        )

    # Engines passed filters but had no tanks.
    if engines_no_tank > 0 and engines_no_tank == engines_passed_filter:
        return StageDiagnostic(
            failure=StageFailure.NO_TANK_FOR_FUEL_TYPE,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv, payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
        )

    # Engines + tanks paired but build couldn't satisfy.
    if mass_cap_blocked and best_dv >= required_dv:
        return StageDiagnostic(
            failure=StageFailure.MASS_CAP_EXCEEDED,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv, best_dv_achieved=best_dv,
            payload_mass=payload_mass, mass_cap=launch_pad_mass_cap,
            engine_fuel_types_attempted=engine_fuel_types,
        )
    if best_dv > 0 and best_dv < required_dv:
        return StageDiagnostic(
            failure=StageFailure.DV_SHORT,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv, best_dv_achieved=best_dv,
            twr_floor=min_twr, best_twr_achieved=best_twr,
            payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
        )
    if seen_twr_short:
        return StageDiagnostic(
            failure=StageFailure.TWR_SHORT,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv,
            twr_floor=min_twr, best_twr_achieved=best_twr,
            payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
        )
    if seen_dry_kills_ratio:
        return StageDiagnostic(
            failure=StageFailure.DRY_MASS_KILLS_RATIO,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv,
            payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
        )
    if seen_engine_too_big:
        return StageDiagnostic(
            failure=StageFailure.ENGINE_TOO_BIG_FOR_TANK,
            body=body_name, in_atmosphere=in_atmosphere,
            required_dv=required_dv,
            payload_mass=payload_mass,
            engine_fuel_types_attempted=engine_fuel_types,
            min_tank_size_needed=(
                min_too_big_engine_size
                if min_too_big_engine_size != float("inf") else 0.0
            ),
        )
    # Catchall — shouldn't normally fire if all the rejection paths above
    # are instrumented. If it does, return a generic DV_SHORT with zero
    # best_dv so the bumper still has something to act on.
    return StageDiagnostic(
        failure=StageFailure.DV_SHORT,
        body=body_name, in_atmosphere=in_atmosphere,
        required_dv=required_dv, best_dv_achieved=best_dv,
        payload_mass=payload_mass,
        engine_fuel_types_attempted=engine_fuel_types,
    )


# ---------------------------------------------------------------------------
# CommNet range calculations
# ---------------------------------------------------------------------------

#: Deep Space Network power by tracking station level (watts).
DSN_POWER: dict[int, float] = {1: 2e9, 2: 50e9, 3: 250e9}

#: Antenna power ratings from KSP game data (watts), keyed by part internal name.
ANTENNA_POWER: dict[str, float] = {
    "SurfAntenna": 500_000,             # Communotron 16-S
    "longAntenna": 500_000,             # Communotron 16
    "HighGainAntenna5_v2": 5_000_000,   # HG-5 High Gain Antenna
    "RelayAntenna5": 2e9,               # RA-2 Relay Antenna
    "mediumDishAntenna": 2e9,           # Communotron DTS-M1
    "HighGainAntenna": 15e9,            # Communotron HG-55
    "RelayAntenna50": 15e9,             # RA-15 Relay Antenna
    "commDish": 100e9,                  # Communotron 88-88
    "RelayAntenna100": 100e9,           # RA-100 Relay Antenna
}

#: 1 Kerbin AU in meters (Kerbin's orbital radius).
KERBIN_AU_M: float = 13_599_840_256.0


def commnet_range(power_a: float, power_b: float) -> float:
    """Max CommNet range in meters between two antenna powers."""
    return math.sqrt(power_a * power_b)


def max_body_distance_m(solar_distance_au: float) -> float:
    """Worst-case distance from Kerbin to a body (opposition, circular orbits)."""
    return (solar_distance_au + 1.0) * KERBIN_AU_M


# ---------------------------------------------------------------------------
# Stage group merging helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergedEdgeGroup:
    """
    A group of mission edges that will be evaluated as a single stage.

    Constraints are the union (strictest) of all constituent edges.
    Delta-v is the sum.
    """
    total_dv: float
    body: str                       # primary body (first edge's body)
    in_atmosphere: bool             # True if any edge is atmospheric
    min_twr: float
    requires_throttleable: bool
    requires_attitude_control: bool
    needs_heat_shield: bool
    needs_landing_legs: bool
    needs_ladder: bool
    plane_change_dv: float          # sum of plane-change components


def merge_edge_groups(
    edges: list,                    # list[MissionEdge]
) -> list[MergedEdgeGroup]:
    """
    Merge a list of MissionEdge objects into a single MergedEdgeGroup.
    Constraints are the union (strictest) of all constituent edges.
    """
    from ..bodies import EdgeType

    atmo_types = {
        EdgeType.ATMOSPHERIC_ASCENT,
        EdgeType.ATMO_LANDING,
        EdgeType.AEROBRAKE_CAPTURE,
    }

    return [MergedEdgeGroup(
        total_dv=sum(e.base_dv for e in edges),
        body=edges[0].body,
        in_atmosphere=any(e.edge_type in atmo_types for e in edges),
        min_twr=max(e.min_twr for e in edges),
        requires_throttleable=any(e.requires_throttleable for e in edges),
        requires_attitude_control=any(e.requires_attitude_control for e in edges),
        needs_heat_shield=any(e.needs_heat_shield for e in edges),
        needs_landing_legs=any(e.needs_landing_legs for e in edges),
        needs_ladder=any(e.needs_ladder for e in edges),
        plane_change_dv=sum(e.plane_change_dv for e in edges),
    )]
