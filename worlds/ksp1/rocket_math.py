"""
Pure rocket-equation mathematics for KSP1 Archipelago.

All functions are side-effect-free and depend only on their arguments.
No imports from the rest of the KSP1 world package (aside from parts.py).

Golden rule: when there is ambiguity, be conservative — overestimate mass,
underestimate delta-v margin, underestimate asparagus benefit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .parts import (
    Engine, FuelTank, SolidBooster,
    max_engine_count, AnyPart,
)

G0: float = 9.80665  # standard gravity, m/s²

FILL_LEVELS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

# Asparagus staging reduces effective tank dry mass.
# We model only 50% of the theoretical benefit (golden rule).
ASPARAGUS_DRY_MASS_FACTOR: float = 0.5


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
    asparagus: bool = False,
) -> float:
    """Compute the delta-v produced by a single stage."""
    isp = engine.atm_isp if in_atmosphere else engine.vac_isp
    if isp <= 0:
        return 0.0
    m_engine = engine.mass * engine_count
    m_tank_dry = tank.dry_mass * tank_count
    m_fuel = tank.fuel_mass * tank_count * fill_fraction

    # Asparagus: side boosters are dropped mid-burn.  Model as reduced dry mass.
    if asparagus:
        m_tank_dry *= ASPARAGUS_DRY_MASS_FACTOR

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
    asparagus: bool = False,
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

    dry_mass_factor = ASPARAGUS_DRY_MASS_FACTOR if asparagus else 1.0
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
    asparagus: bool = False,
    srb_needs_rcs: bool = True,     # if True, SRBs only valid when player has RCS
    player_has_rcs: bool = False,
) -> Optional[StageResult]:
    """
    Find the minimum-mass engine+tank configuration that meets *required_dv*
    and all constraints.  Returns None if no valid configuration exists.

    Algorithm:
      For each eligible engine:
        For each compatible tank:
          For each fill level [1.0, 0.75, 0.5, 0.25]:
            Find min engine count satisfying TWR (if required)
            Compute required tank count
            If valid, track minimum wet-mass configuration
      Also evaluate SRBs (fixed config, no fill levels)
      Return best result or None
    """
    best: Optional[StageResult] = None

    # Full payload = caller-supplied payload + heat shield (if needed)
    full_payload = payload_mass + (heat_shield_mass if needs_heat_shield else 0.0)

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

        # Throttle filter
        if requires_throttleable and not engine.throttleable:
            continue

        # ION engine (xenon) special handling: requires xenon tanks
        for tank in available_tanks:
            if tank.fuel_type != engine.fuel_type:
                continue

            for fill in FILL_LEVELS:
                max_eng = max_engine_count(tank.size_class, engine.size_class)
                if max_eng == 0:
                    continue

                # Find minimum engine count that satisfies TWR
                min_engines = 1
                if min_twr > 0 and gravity > 0:
                    # We need: thrust / (m_wet * gravity) >= min_twr
                    # Estimate wet mass with 1 tank (conservative starting point)
                    # Use a single-tank wet mass for the TWR floor search;
                    # we'll verify with the actual tank count after.
                    for n_eng in range(1, max_eng + 1):
                        # Approximate wet mass with 1 tank to check if TWR is
                        # achievable at all for this engine count.  The actual
                        # check below uses the real tank count.
                        est_wet = full_payload + engine.mass * n_eng + tank.dry_mass + tank.fuel_mass * fill
                        thrust = (engine.atm_thrust if in_atmosphere else engine.vac_thrust) * n_eng
                        if twr(thrust, est_wet, gravity) >= min_twr:
                            min_engines = n_eng
                            break
                    else:
                        continue  # even max engines can't meet TWR on a 1-tank stage

                for n_eng in range(min_engines, max_eng + 1):
                    n_tanks = required_tanks(
                        engine, n_eng, tank, fill, required_dv, full_payload,
                        in_atmosphere=in_atmosphere, asparagus=asparagus,
                    )
                    if n_tanks <= 0:
                        continue

                    m_engine = engine.mass * n_eng
                    m_tank_dry = tank.dry_mass * n_tanks
                    m_fuel = tank.fuel_mass * n_tanks * fill
                    m_dry = full_payload + m_engine + m_tank_dry
                    m_wet = m_dry + m_fuel

                    # Verify TWR at ignition with the actual tank count
                    thrust = (engine.atm_thrust if in_atmosphere else engine.vac_thrust) * n_eng
                    if min_twr > 0 and twr(thrust, m_wet, gravity) < min_twr:
                        continue

                    # Verify delta-v is actually achieved (required_tanks rounds up)
                    actual_dv = stage_delta_v(
                        engine, n_eng, tank, n_tanks, fill, full_payload,
                        in_atmosphere=in_atmosphere, asparagus=asparagus,
                    )
                    if actual_dv < required_dv:
                        continue

                    twr_ign = twr(thrust, m_wet, gravity)
                    twr_bur = twr(thrust, m_dry, gravity)

                    result = StageResult(
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

                    if best is None or m_wet < best.stage_mass_wet:
                        best = result

                    # Once we've found a valid n_eng, no need to try more
                    break

    # -----------------------------------------------------------------------
    # Evaluate SRBs
    # -----------------------------------------------------------------------
    for srb in available_srbs:
        # SRBs can't throttle — skip throttle-required edges
        if requires_throttleable:
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

        # SRBs don't have separate tank types — they self-contain fuel
        # Use a single 1.25m LFO tank's size as the "tank" for count lookup
        max_srb = max_engine_count(srb.size_class, srb.size_class)
        max_srb = max(max_srb, 1)

        for n_srb in range(1, max_srb + 1):
            dv = srb_delta_v(srb, n_srb, full_payload, in_atmosphere=in_atmosphere)
            if dv < required_dv:
                continue

            m_dry = full_payload + srb.dry_mass * n_srb
            m_wet = m_dry + srb.fuel_mass * n_srb

            thrust = (srb.atm_thrust if in_atmosphere else srb.vac_thrust) * n_srb
            if min_twr > 0 and twr(thrust, m_wet, gravity) < min_twr:
                continue

            twr_ign = twr(thrust, m_wet, gravity)
            twr_bur = twr(thrust, m_dry, gravity)

            result = StageResult(
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

            if best is None or m_wet < best.stage_mass_wet:
                best = result
            break  # found minimum SRB count, no need to try more

    return best


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
