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

G0: float = 9.80665  # standard gravity, m/s²

FILL_LEVELS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

# Parallel staging (asparagus/onion) reduces effective tank dry mass.
# We model only a fraction of the theoretical benefit (golden rule).
ASPARAGUS_DRY_MASS_FACTOR: float = 0.5
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

def find_optimal_stage(
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
        # Heat shield filter: engine must fit under the player's shield
        if needs_heat_shield:
            if max_heat_shield_size is None:
                continue  # no shield at all
            if engine.size_class > max_heat_shield_size:
                continue

        # Gimbal filter (atmospheric gravity turn without aero surfaces)
        if require_gimbal and not engine.has_gimbal:
            continue

        # Throttle filter
        if requires_throttleable and not engine.throttleable:
            continue

        e_mass = engine.mass
        e_size = engine.size_class
        thrust_per_eng = engine.atm_thrust if in_atmosphere else engine.vac_thrust

        # Engine lower bound: even with 1 engine + 1 smallest tank, can't beat best?
        if full_payload + e_mass >= best_wet:
            continue

        # Compute mass ratio R once per engine (the key optimisation: avoids
        # redundant math.exp inside the tank/fill/engine-count loops).
        isp = engine.atm_isp if in_atmosphere else engine.vac_isp
        if isp <= 0:
            continue
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

        for tank in compatible_tanks:
            t_dry = tank.dry_mass
            t_fuel = tank.fuel_mass
            t_size = tank.size_class

            # Lower bound: 1 engine + 1 tank at lowest fill can't beat best?
            lb = full_payload + e_mass + t_dry + t_fuel * 0.25
            if lb >= best_wet:
                continue

            # Early exit: if even fill=1.0 can't achieve the dv, skip tank.
            # denominator = fuel*fill - R_minus_1 * dry*dry_factor
            # At fill=1.0 this is maximised; if still <= 0, impossible.
            effective_dry = t_dry * dry_factor
            if t_fuel - R_minus_1 * effective_dry <= 0:
                continue

            # Engine must fit under or on the tank
            if e_size > t_size:
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
                    numer = twr_g * (full_payload + m_tank_1)
                    min_engines = max(min_engines, _ceil(numer / twr_eng_denom))

                for sm_df, sm_symmetric in _sub_modes:
                    sm_max = max_eng_parallel if sm_symmetric else max_eng_base
                    if min_engines > sm_max:
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
                        n_tanks = _ceil(R_minus_1 * (full_payload + m_engine) / denom)
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
                        m_dry = full_payload + m_engine + m_tank_dry
                        m_wet = m_dry + m_fuel

                        # Skip if can't beat current best
                        if m_wet >= best_wet:
                            break  # more engines only adds mass

                        # Verify TWR at ignition with the actual tank count
                        thrust = thrust_per_eng * n_eng
                        if has_twr and thrust < twr_g * m_wet:
                            continue

                        # Verify delta-v (required_tanks rounds up, so check actual)
                        # Inline stage_delta_v to avoid function call overhead.
                        verify_dry = m_tank_dry * sm_df if sm_df != 1.0 else m_tank_dry
                        v_dry = full_payload + m_engine + verify_dry
                        v_wet = v_dry + m_fuel
                        if v_dry <= 0 or v_wet <= v_dry:
                            continue
                        actual_dv = isp_g0 * _log(v_wet / v_dry)
                        if actual_dv < required_dv:
                            continue

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

        srb_counts = range(1, max_srb + 1)

        for n_srb in srb_counts:
            # Inline srb_delta_v
            m_dry = full_payload + srb.dry_mass * n_srb
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

    return best


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
