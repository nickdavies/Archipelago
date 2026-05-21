"""
Pure rocket-equation mathematics for KSP1 Archipelago.

All functions are side-effect-free and depend only on their arguments.
No imports from the rest of the KSP1 world package (aside from parts.py).

Golden rule: when there is ambiguity, be conservative — overestimate mass,
underestimate delta-v margin, underestimate asparagus benefit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .parts import (
    Engine, FuelTank, SolidBooster, MultiMount,
    MAX_RADIAL_ENGINES,
)
from .capability_reasons import StageDiagnostic, StageFailure

G0: float = 9.80665  # standard gravity, m/s²

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

# Parallel staging (asparagus/onion) reduces effective tank dry mass.
# A constant-factor model under-models true multi-stage Tsiolkovsky by
# 10-25% — even at factor 0.0 the single-stage formula can't reach the
# product-of-mass-ratios that real asparagus achieves. Accepting that
# under-model in exchange for simple per-iteration math.
ASPARAGUS_DRY_MASS_FACTOR: float = 0.25
ONION_DRY_MASS_FACTOR: float = 0.75

# KSP symmetry tool modes. Radial boosters must use one of these counts.
KSP_SYMMETRY_MODES: tuple[int, ...] = (2, 3, 4, 6, 8)

# Lookup: parallel_mode string → dry mass factor
_PARALLEL_DRY_FACTORS: dict[str, float] = {
    "none": 1.0,
    "asparagus": ASPARAGUS_DRY_MASS_FACTOR,
    "onion": ONION_DRY_MASS_FACTOR,
}


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
    tank_count: int
    fill_fraction: float
    engine_name: str
    tank_name: str
    # Non-propulsion parts: [(count, part_id), ...]
    # Populated by _evaluate_profile after stage optimisation.
    equipment: list[tuple[int, str]] = field(default_factory=list)


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
    parallel_mode: str = "none",
) -> float:
    """Compute the delta-v produced by a single stage."""
    isp = engine.atm_isp if in_atmosphere else engine.vac_isp
    if isp <= 0:
        return 0.0
    m_engine = engine.mass * engine_count
    m_tank_dry = tank.dry_mass * tank_count
    m_fuel = tank.fuel_mass * tank_count * fill_fraction

    # Parallel staging: side boosters are dropped mid-burn → reduced dry mass.
    dry_factor = _PARALLEL_DRY_FACTORS[parallel_mode]
    m_tank_dry *= dry_factor

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
    parallel_mode: str = "none",
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

    dry_mass_factor = _PARALLEL_DRY_FACTORS[parallel_mode]
    effective_dry = tank.dry_mass * dry_mass_factor

    numerator = (R - 1) * (payload_mass + m_engine)
    denominator = tank.fuel_mass * fill_fraction - (R - 1) * effective_dry
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
    heat_shield_mass: float = 0.0,
    in_atmosphere: bool = False,
    parallel_mode: str = "none",    # "none", "asparagus", or "onion"
    srb_needs_rcs: bool = True,     # if True, SRBs only valid when player has RCS
    player_has_rcs: bool = False,
    tanks_by_fuel_type: Optional[dict[str, list[FuelTank]]] = None,
    available_multi_mounts: Optional[list[MultiMount]] = None,
    require_gimbal: bool = False,
    attitude_module_mass: float = 0.0,  # added to payload only if chosen prop lacks gimbal
    diagnostic_out: Optional[list[StageDiagnostic]] = None,
    body_name: str = "",                  # for diagnostic reporting only
    launch_pad_mass_cap: float = float("inf"),  # for MASS_CAP_EXCEEDED diagnostic
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
    _diag_seen_twr_short = False
    _diag_engine_fuel_types: set[str] = set()
    _diag_mass_cap_blocked = False  # at least one combo would meet dv/twr but mass>cap

    # Full payload = caller-supplied payload + heat shield (if needed)
    full_payload = payload_mass + (heat_shield_mass if needs_heat_shield else 0.0)

    # Parallel staging dry-mass factor (applied to tank dry mass)
    _parallel = parallel_mode != "none"
    dry_factor = _PARALLEL_DRY_FACTORS[parallel_mode]

    # Sub-modes: (dry_factor, use_symmetric_counts)
    # When parallel, try symmetric counts with dry benefit first (usually
    # finds a good bound), then try all counts without benefit.
    _sub_modes: list[tuple[float, bool]] = []
    if _parallel:
        _sub_modes.append((dry_factor, True))
    _sub_modes.append((1.0, False))

    # Pre-check: TWR denom component that depends only on engine
    has_twr = min_twr > 0 and gravity > 0
    twr_g = min_twr * gravity  # reused per engine

    # Local refs to avoid repeated global/attribute lookups in hot loop
    _exp = math.exp
    _log = math.log
    _ceil = math.ceil
    _fills = FILL_LEVELS
    _mounts = available_multi_mounts or []
    _max_radial = MAX_RADIAL_ENGINES
    # Cache adapter lookups: only ~36 unique (e_size, t_size) combos
    _adapter_cache: dict[tuple[float, float], int] = {}

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

        # Per-candidate attitude module charge. When the caller passes a
        # non-zero `attitude_module_mass`, this stage needs an on-stage
        # attitude-control source. Gimballed engines provide it for free;
        # ungimballed engines must carry the lightest reaction-wheel / RCS
        # bundle the caller selected. Adding it only to ungimballed
        # candidates lets the optimizer trade "ungimballed + module" against
        # "gimballed alone" inside the same search loop.
        eng_payload = full_payload + (
            attitude_module_mass if not engine.has_gimbal else 0.0
        )

        e_mass = engine.mass
        e_size = engine.size_class
        thrust_per_eng = engine.atm_thrust if in_atmosphere else engine.vac_thrust

        # Engine lower bound: even with 1 engine + 1 smallest tank, can't beat best?
        if eng_payload + e_mass >= best_wet:
            continue

        # Compute mass ratio R once per engine (the key optimisation: avoids
        # redundant math.exp inside the tank/fill/engine-count loops).
        isp = engine.atm_isp if in_atmosphere else engine.vac_isp
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

        for tank in compatible_tanks:
            t_dry = tank.dry_mass
            t_fuel = tank.fuel_mass
            t_size = tank.size_class

            # Lower bound: 1 engine + 1 tank at lowest fill can't beat best?
            lb = eng_payload + e_mass + t_dry + t_fuel * 0.25
            if lb >= best_wet:
                continue

            # Early exit: if even fill=1.0 can't achieve the dv, skip tank.
            # denominator = fuel*fill - R_minus_1 * dry*dry_factor
            # At fill=1.0 this is maximised; if still <= 0, impossible.
            effective_dry = t_dry * dry_factor
            if t_fuel - R_minus_1 * effective_dry <= 0:
                _diag_seen_dry_kills_ratio = True
                continue

            # Engine must fit under or on the tank
            if e_size > t_size:
                _diag_seen_engine_too_big = True
                continue

            # Three mounting modes:
            # 1. Radial engine: mounts directly on tank side (cap 8)
            # 2. Adapter/plate: multiple engines under shared tank stack
            # 3. Radial tank: each engine needs its own tank (cap 8)
            _size_key = (e_size, t_size)
            adapter_max = _adapter_cache.get(_size_key)
            if adapter_max is None:
                adapter_max = _adapter_max_engines(e_size, t_size, _mounts)
                _adapter_cache[_size_key] = adapter_max

            if engine.radial_mountable:
                max_eng_base = _max_radial
                use_radial_tank_constraint = False
            else:
                # Best of adapter mode or radial-tank mode
                max_eng_base = max(adapter_max, _max_radial)
                use_radial_tank_constraint = True

            max_eng_parallel = (max(max_eng_base, 1 + KSP_SYMMETRY_MODES[-1])
                                if _parallel else max_eng_base)

            for fill in _fills:
                # TWR minimum engine count (independent of dry factor)
                # Radial-only engines need >= 2 for symmetric thrust
                min_engines = 2 if engine.radial_mountable else 1
                if has_twr:
                    if twr_eng_denom <= 0:
                        continue  # engine too heavy for this TWR at any count
                    m_tank_1 = t_dry + t_fuel * fill
                    numer = twr_g * (eng_payload + m_tank_1)
                    min_engines = max(min_engines, _ceil(numer / twr_eng_denom))

                for sm_df, sm_symmetric in _sub_modes:
                    sm_max = max_eng_parallel if sm_symmetric else max_eng_base
                    if min_engines > sm_max:
                        _diag_seen_twr_short = True
                        continue

                    # Denominator for required tank count
                    sm_ed = t_dry * sm_df
                    denom = t_fuel * fill - R_minus_1 * sm_ed
                    if denom <= 0:
                        continue

                    if sm_symmetric:
                        eng_counts = [1 + s for s in KSP_SYMMETRY_MODES
                                      if min_engines <= 1 + s <= sm_max]
                    else:
                        eng_counts = range(min_engines, sm_max + 1)

                    for n_eng in eng_counts:
                        m_engine = e_mass * n_eng
                        n_tanks = _ceil(R_minus_1 * (eng_payload + m_engine) / denom)
                        if n_tanks <= 0:
                            continue
                        if tank.max_count > 0 and n_tanks > tank.max_count:
                            continue

                        # Radial-tank constraint: each stack engine needs its own
                        # tank unless an adapter/plate covers this engine count
                        if (use_radial_tank_constraint and n_eng > 1
                                and n_eng > adapter_max):
                            n_tanks = max(n_tanks, n_eng)

                        m_tank_dry = t_dry * n_tanks
                        m_fuel = t_fuel * n_tanks * fill
                        m_dry = eng_payload + m_engine + m_tank_dry
                        m_wet = m_dry + m_fuel

                        # Skip if can't beat current best
                        if m_wet >= best_wet:
                            break  # more engines only adds mass

                        # Verify TWR at ignition with the actual tank count
                        thrust = thrust_per_eng * n_eng
                        if has_twr and thrust < twr_g * m_wet:
                            twr_actual = thrust / (m_wet * gravity) if gravity > 0 else 0.0
                            if twr_actual > _diag_best_twr:
                                _diag_best_twr = twr_actual
                            _diag_seen_twr_short = True
                            continue

                        # Verify delta-v (required_tanks rounds up, so check actual)
                        # Inline stage_delta_v to avoid function call overhead.
                        verify_dry = m_tank_dry * sm_df if sm_df != 1.0 else m_tank_dry
                        v_dry = eng_payload + m_engine + verify_dry
                        v_wet = v_dry + m_fuel
                        if v_dry <= 0 or v_wet <= v_dry:
                            continue
                        actual_dv = isp_g0 * _log(v_wet / v_dry)
                        if actual_dv > _diag_best_dv:
                            _diag_best_dv = actual_dv
                        if actual_dv < required_dv:
                            continue
                        # Mass-cap check (informational; the caller enforces).
                        if m_wet > launch_pad_mass_cap:
                            _diag_mass_cap_blocked = True

                        twr_ign = thrust / (m_wet * gravity) if gravity > 0 else 0.0
                        twr_bur = thrust / (m_dry * gravity) if gravity > 0 else 0.0

                        best = StageResult(
                            delta_v=actual_dv,
                            twr_at_ignition=twr_ign,
                            twr_at_burnout=twr_bur,
                            engine_is_throttleable=engine.throttleable,
                            engine_has_gimbal=engine.has_gimbal,
                            stage_mass_wet=m_wet,
                            stage_mass_dry=m_dry,
                            engine_count=n_eng,
                            tank_count=n_tanks,
                            fill_fraction=fill,
                            engine_name=engine.name,
                            tank_name=tank.name,
                        )
                        best_wet = m_wet

                        # Once we've found a valid n_eng, no need to try more
                        break

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

        srb_isp = srb.atm_isp if in_atmosphere else srb.vac_isp
        if srb_isp <= 0:
            continue
        srb_thrust = (srb.atm_thrust if in_atmosphere else srb.vac_thrust)

        # Per-candidate attitude module charge (see engine loop above).
        srb_payload = full_payload + (
            attitude_module_mass if not srb.has_gimbal else 0.0
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
                    tank_count=0,
                    fill_fraction=1.0,
                    engine_name=srb.name,
                    tank_name="(SRB integral)",
                )
                best_wet = m_wet
            break  # found minimum SRB count, no need to try more

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
_FOS_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}
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
    _FOS_CACHE[key] = (result, cached_diag)
    if caller_diag is not None and internal_diag:
        caller_diag.extend(internal_diag)
    return result


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
    from .bodies import EdgeType

    atmo_types = {
        EdgeType.ATMOSPHERIC_ASCENT,
        EdgeType.ATMO_LANDING_PROPULSIVE,
        EdgeType.ATMO_LANDING_AERO,
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
