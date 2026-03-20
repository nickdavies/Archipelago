"""
Celestial body data, mission graph, and difficulty profiles for KSP1 Archipelago.

Delta-v values sourced from the KSP-DeltaV-Planner (planets.ts / kerbin.ts).
Physics values from the KSP wiki.  All values are conservative estimates that
err toward overestimating the required delta-v (golden rule).

Mission profile structure
-------------------------
MISSION_PROFILES maps (body_name, mission_type) to a list of profile
alternatives.  Each alternative is an ordered list of MissionEdge objects
describing the journey from kerbin_surface to the destination (and back for
return missions).  The capability engine tries every alternative; if any one
succeeds the mission is considered achievable.

Node naming convention:
  "kerbin_surface", "kerbin_low_orbit", "kerbin_soi"
  "{body}_surface", "{body}_low_orbit", "{body}_soi", "{body}_intercept"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Difficulty profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DifficultyProfile:
    fixed_margin: float         # m/s added to every dv budget
    percent_margin: float       # fraction multiplied on top (0.15 = 15%)
    plane_change_fraction: float # fraction of worst-case plane-change dv included
    min_twr_atmo: float         # minimum TWR for atmospheric ascent/landing
    min_twr_vac: float          # minimum TWR for vacuum ascent/landing
    ship_cd: float              # ship body drag coefficient (parachute discount)
    srb_needs_rcs: bool         # True = SRBs require RCS for throttle-mode edges


DIFFICULTY_PROFILES: dict[str, DifficultyProfile] = {
    "casual": DifficultyProfile(
        fixed_margin=200, percent_margin=0.30, plane_change_fraction=0.50,
        min_twr_atmo=1.5, min_twr_vac=1.2,
        ship_cd=0.0, srb_needs_rcs=True,
    ),
    "normal": DifficultyProfile(
        fixed_margin=100, percent_margin=0.15, plane_change_fraction=0.25,
        min_twr_atmo=1.5, min_twr_vac=1.2,
        ship_cd=0.1, srb_needs_rcs=True,
    ),
    "expert": DifficultyProfile(
        fixed_margin=50, percent_margin=0.05, plane_change_fraction=0.10,
        min_twr_atmo=1.3, min_twr_vac=1.1,
        ship_cd=0.2, srb_needs_rcs=False,
    ),
    "insane": DifficultyProfile(
        fixed_margin=0, percent_margin=0.00, plane_change_fraction=0.00,
        min_twr_atmo=1.2, min_twr_vac=1.0,
        ship_cd=0.2, srb_needs_rcs=False,
    ),
}


def effective_dv(base_dv: float, profile: DifficultyProfile,
                 plane_change_dv: float = 0.0) -> float:
    """Return the margin-adjusted dv budget for an edge."""
    pc = plane_change_dv * profile.plane_change_fraction
    return (base_dv + pc + profile.fixed_margin) * (1.0 + profile.percent_margin)


# ---------------------------------------------------------------------------
# Body dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyDeltaV:
    """Raw delta-v components from the planner graph.  None = not applicable."""
    dvGL: float                  # Ground <-> Low Orbit
    dvLE: Optional[float]        # Low Orbit <-> SOI edge (planets only)
    dvEI: Optional[float]        # SOI edge <-> Intercept (planets only)
    dvK: Optional[float]         # Intercept <-> Kerbin elliptical (planets)
    dvLI: Optional[float]        # Low Orbit <-> Intercept (direct, moons)
    dvPL: Optional[float]        # Intercept <-> Parent Low Orbit (Kerbin moons)
    dvPE: Optional[float]        # Intercept <-> Parent Elliptical (other moons)
    dvPlaneChange: float         # Worst-case plane change (0 if negligible)


@dataclass(frozen=True)
class Body:
    name: str
    parent: Optional[str]           # None for planets, parent name for moons
    surface_gravity: float          # m/s²
    has_atmosphere: bool
    atm_pressure_kpa: float         # sea-level, 0 for vacuum bodies
    atm_density_kg_m3: float        # sea-level air density, 0 for vacuum
    can_land: bool
    low_orbit_alt_km: float         # defines "low orbit" for location checks
    solar_distance_au: float        # Kerbin = 1.0, used for ION/solar logic
    landing_leg_tier: int           # minimum leg tier required for landing
    min_relay_tier: int             # 0=none, 1=t1, 2=t2, 3=t3
    power_requirement: str          # "solar" | "solar_marginal" | "rtg"
    eva_jetpack_twr: float          # precomputed: 0.5/(0.09375*surface_gravity)
    dv: BodyDeltaV

    # --- Location generation ---
    check_scale: int = 1            # location checks per event (1/2/3)

    # --- Science budget (for tech-tree access rules) ---
    has_ocean: bool = False         # body has splashable liquid surface
    num_biomes: int = 1             # distinct landed biomes
    num_splash_biomes: int = 0      # distinct ocean/splash biomes
    space_low_mult: float = 1.0     # InSpaceLow science multiplier
    space_high_mult: float = 1.0    # InSpaceHigh science multiplier
    fly_low_mult: float = 0.0       # FlyingLow multiplier (0 = no atmosphere)
    fly_high_mult: float = 0.0      # FlyingHigh multiplier (0 = no atmosphere)
    landed_mult: float = 0.0        # Landed multiplier (0 = can't land)
    splashed_mult: float = 0.0      # Splashed multiplier (0 = no ocean)


def _jetpack_twr(g: float) -> float:
    """Jetpack thrust ~0.5 kN, Kerbal mass ~0.09375 t."""
    return 0.5 / (0.09375 * g)


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(Enum):
    ATMOSPHERIC_ASCENT      = auto()  # surface -> orbit, atmospheric body
    VACUUM_ASCENT           = auto()  # surface -> orbit, airless body
    PURE_VACUUM             = auto()  # orbital transfer, no TWR requirement
    PLANET_TRANSFER         = auto()  # interplanetary, plane change fraction applied
    VACUUM_LANDING          = auto()  # orbit -> surface, airless body
    ATMO_LANDING_PROPULSIVE = auto()  # orbit -> surface, propulsive through atmo
    ATMO_LANDING_AERO       = auto()  # orbit -> surface, heat shield + parachutes
    AEROBRAKE_CAPTURE       = auto()  # SOI capture using atmosphere (not landing)


# ---------------------------------------------------------------------------
# Mission edge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionEdge:
    source: str
    destination: str
    edge_type: EdgeType
    base_dv: float                      # m/s, nominal delta-v
    body: str                           # body name for physics lookups
    plane_change_dv: float = 0.0        # worst-case plane change
    min_twr: float = 0.0                # 0.0 = no TWR requirement
    requires_throttleable: bool = False
    requires_attitude_control: bool = False  # gimbal OR rcs OR reaction_wheel
    needs_heat_shield: bool = False
    needs_landing_legs: bool = False
    needs_ladder: bool = False          # set at profile build time if eva_twr < 1.05


# ---------------------------------------------------------------------------
# Body database
# ---------------------------------------------------------------------------

KERBIN = Body(
    name="Kerbin", parent=None,
    surface_gravity=9.81, has_atmosphere=True,
    atm_pressure_kpa=101.325, atm_density_kg_m3=1.225,
    can_land=True, low_orbit_alt_km=80,
    solar_distance_au=1.0,
    landing_leg_tier=2, min_relay_tier=0,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(9.81),
    dv=BodyDeltaV(
        dvGL=3400, dvLE=950, dvEI=None, dvK=None,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=0,
    ),
    check_scale=1,  # Kerbin has its own special event list; scale unused
    has_ocean=True, num_biomes=9, num_splash_biomes=2,
    space_low_mult=1.5, space_high_mult=1.0,
    fly_low_mult=1.0, fly_high_mult=0.7,
    landed_mult=0.3, splashed_mult=0.4,
)

MUN = Body(
    name="Mun", parent="Kerbin",
    surface_gravity=1.63, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=14,
    solar_distance_au=1.0,
    landing_leg_tier=2, min_relay_tier=0,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(1.63),
    dv=BodyDeltaV(
        dvGL=580, dvLE=None, dvEI=None, dvK=None,
        dvLI=310, dvPL=860, dvPE=None, dvPlaneChange=0,
    ),
    check_scale=1,
    num_biomes=7,
    space_low_mult=4.0, space_high_mult=2.0,
    landed_mult=9.0,
)

MINMUS = Body(
    name="Minmus", parent="Kerbin",
    surface_gravity=0.491, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=1.0,
    landing_leg_tier=1, min_relay_tier=0,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(0.491),
    dv=BodyDeltaV(
        dvGL=180, dvLE=None, dvEI=None, dvK=None,
        dvLI=160, dvPL=930, dvPE=None, dvPlaneChange=340,
    ),
    check_scale=1,
    num_biomes=9,
    space_low_mult=5.0, space_high_mult=2.5,
    landed_mult=12.0,
)

MOHO = Body(
    name="Moho", parent=None,
    surface_gravity=2.70, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=20,
    solar_distance_au=0.34,
    landing_leg_tier=2, min_relay_tier=1,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(2.70),
    dv=BodyDeltaV(
        dvGL=870, dvLE=None, dvEI=None, dvK=760,
        dvLI=2410, dvPL=None, dvPE=None, dvPlaneChange=2520,
    ),
    check_scale=2,
    num_biomes=6,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=9.0,
)

EVE = Body(
    name="Eve", parent=None,
    surface_gravity=16.7, has_atmosphere=True,
    atm_pressure_kpa=506.625, atm_density_kg_m3=5.0,
    can_land=True, low_orbit_alt_km=90,
    solar_distance_au=0.72,
    landing_leg_tier=2, min_relay_tier=1,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(16.7),
    dv=BodyDeltaV(
        dvGL=8000, dvLE=1330, dvEI=80, dvK=90,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=430,
    ),
    check_scale=2,
    has_ocean=True, num_biomes=8, num_splash_biomes=3,
    space_low_mult=8.0, space_high_mult=4.0,
    fly_low_mult=2.0, fly_high_mult=1.5,
    landed_mult=8.0, splashed_mult=8.0,
)

GILLY = Body(
    name="Gilly", parent="Eve",
    surface_gravity=0.049, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=6,
    solar_distance_au=0.72,
    landing_leg_tier=1, min_relay_tier=1,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(0.049),
    dv=BodyDeltaV(
        dvGL=30, dvLE=None, dvEI=None, dvK=None,
        dvLI=410, dvPL=None, dvPE=60, dvPlaneChange=0,
    ),
    check_scale=1,
    num_biomes=3,
    space_low_mult=9.0, space_high_mult=4.5,
    landed_mult=12.0,
)

DUNA = Body(
    name="Duna", parent=None,
    surface_gravity=2.94, has_atmosphere=True,
    atm_pressure_kpa=6.755, atm_density_kg_m3=0.096,
    can_land=True, low_orbit_alt_km=50,
    solar_distance_au=1.52,
    landing_leg_tier=2, min_relay_tier=1,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(2.94),
    dv=BodyDeltaV(
        dvGL=1450, dvLE=360, dvEI=250, dvK=130,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=10,
    ),
    check_scale=2,
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    fly_low_mult=1.5, fly_high_mult=1.2,
    landed_mult=8.0,
)

IKE = Body(
    name="Ike", parent="Duna",
    surface_gravity=1.10, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=1.52,
    landing_leg_tier=2, min_relay_tier=1,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(1.10),
    dv=BodyDeltaV(
        dvGL=390, dvLE=None, dvEI=None, dvK=None,
        dvLI=180, dvPL=None, dvPE=30, dvPlaneChange=0,
    ),
    check_scale=1,
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=8.0,
)

DRES = Body(
    name="Dres", parent=None,
    surface_gravity=2.94, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=25,
    solar_distance_au=2.65,
    landing_leg_tier=2, min_relay_tier=2,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(2.94),
    dv=BodyDeltaV(
        dvGL=430, dvLE=None, dvEI=None, dvK=610,
        dvLI=1290, dvPL=None, dvPE=None, dvPlaneChange=1010,
    ),
    check_scale=2,
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=8.0,
)

JOOL = Body(
    name="Jool", parent=None,
    surface_gravity=7.85, has_atmosphere=True,
    atm_pressure_kpa=1519.88, atm_density_kg_m3=10.0,
    can_land=False, low_orbit_alt_km=210,
    solar_distance_au=5.20,
    landing_leg_tier=0, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=14000, dvLE=2810, dvEI=160, dvK=980,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=270,
    ),
    check_scale=2,
    num_biomes=0,
    space_low_mult=12.0, space_high_mult=6.0,
    fly_low_mult=6.0, fly_high_mult=4.0,
)

LAYTHE = Body(
    name="Laythe", parent="Jool",
    surface_gravity=7.85, has_atmosphere=True,
    atm_pressure_kpa=60.795, atm_density_kg_m3=0.73,
    can_land=True, low_orbit_alt_km=60,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=2900, dvLE=None, dvEI=None, dvK=None,
        dvLI=1070, dvPL=None, dvPE=930, dvPlaneChange=0,
    ),
    check_scale=3,
    has_ocean=True, num_biomes=9, num_splash_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    fly_low_mult=4.0, fly_high_mult=3.0,
    landed_mult=14.0, splashed_mult=10.0,
)

VALL = Body(
    name="Vall", parent="Jool",
    surface_gravity=2.31, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=15,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(2.31),
    dv=BodyDeltaV(
        dvGL=860, dvLE=None, dvEI=None, dvK=None,
        dvLI=910, dvPL=None, dvPE=620, dvPlaneChange=0,
    ),
    check_scale=3,
    num_biomes=9,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

TYLO = Body(
    name="Tylo", parent="Jool",
    surface_gravity=7.85, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=30,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=2270, dvLE=None, dvEI=None, dvK=None,
        dvLI=1100, dvPL=None, dvPE=400, dvPlaneChange=0,
    ),
    check_scale=3,
    num_biomes=6,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

BOP = Body(
    name="Bop", parent="Jool",
    surface_gravity=0.589, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=5.20,
    landing_leg_tier=1, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(0.589),
    dv=BodyDeltaV(
        dvGL=230, dvLE=None, dvEI=None, dvK=None,
        dvLI=900, dvPL=None, dvPE=220, dvPlaneChange=2440,
    ),
    check_scale=2,
    num_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

POL = Body(
    name="Pol", parent="Jool",
    surface_gravity=0.373, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=6,
    solar_distance_au=5.20,
    landing_leg_tier=1, min_relay_tier=2,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(0.373),
    dv=BodyDeltaV(
        dvGL=130, dvLE=None, dvEI=None, dvK=None,
        dvLI=820, dvPL=None, dvPE=160, dvPlaneChange=700,
    ),
    check_scale=2,
    num_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

EELOO = Body(
    name="Eeloo", parent=None,
    surface_gravity=1.72, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=6.0,
    landing_leg_tier=2, min_relay_tier=3,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(1.72),
    dv=BodyDeltaV(
        dvGL=620, dvLE=None, dvEI=None, dvK=1140,
        dvLI=1370, dvPL=None, dvPE=None, dvPlaneChange=1330,
    ),
    check_scale=3,
    num_biomes=7,
    space_low_mult=15.0, space_high_mult=7.5,
    landed_mult=15.0,
)

KERBOL = Body(
    name="Kerbol", parent=None,
    surface_gravity=17.1, has_atmosphere=True,
    atm_pressure_kpa=16200.0, atm_density_kg_m3=350.0,
    can_land=False, low_orbit_alt_km=1000,
    solar_distance_au=0.0,
    landing_leg_tier=0, min_relay_tier=0,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(17.1),
    dv=BodyDeltaV(
        dvGL=67000, dvLE=None, dvEI=None, dvK=6000,
        dvLI=13700, dvPL=None, dvPE=None, dvPlaneChange=0,
    ),
    check_scale=1,
    num_biomes=0,
    space_low_mult=2.0, space_high_mult=1.0,
)

# Authoritative list of all bodies
ALL_BODIES: list[Body] = [
    KERBIN, MUN, MINMUS,
    MOHO, EVE, GILLY,
    DUNA, IKE, DRES,
    JOOL, LAYTHE, VALL, TYLO, BOP, POL,
    EELOO, KERBOL,
]

BODY_BY_NAME: dict[str, Body] = {b.name: b for b in ALL_BODIES}


# ---------------------------------------------------------------------------
# Mission profile builder helpers
# ---------------------------------------------------------------------------

def _E(  # shorthand for MissionEdge
    src: str, dst: str, et: EdgeType, dv: float, body: str,
    pc: float = 0.0, min_twr: float = 0.0,
    throttle: bool = False, attitude: bool = False,
    heat: bool = False, legs: bool = False,
) -> MissionEdge:
    return MissionEdge(
        source=src, destination=dst, edge_type=et,
        base_dv=dv, body=body, plane_change_dv=pc,
        min_twr=min_twr, requires_throttleable=throttle,
        requires_attitude_control=attitude,
        needs_heat_shield=heat, needs_landing_legs=legs,
    )


AT = EdgeType.ATMOSPHERIC_ASCENT
VA = EdgeType.VACUUM_ASCENT
PV = EdgeType.PURE_VACUUM
PT = EdgeType.PLANET_TRANSFER
VL = EdgeType.VACUUM_LANDING
ALP = EdgeType.ATMO_LANDING_PROPULSIVE
ALA = EdgeType.ATMO_LANDING_AERO
AB = EdgeType.AEROBRAKE_CAPTURE

# Kerbin ascent — always the first edge of every profile
_KERBIN_ASCENT = _E("kerbin_surface", "kerbin_low_orbit", AT, 3400, "Kerbin",
                    min_twr=1.5, throttle=True, attitude=True)
# Kerbin escape to SOI edge
_KERBIN_ESCAPE = _E("kerbin_low_orbit", "kerbin_soi", PV, 950, "Kerbin",
                    attitude=True)
# Kerbin aero reentry (return missions)
_KERBIN_REENTRY = _E("kerbin_intercept", "kerbin_surface", ALA, 100, "Kerbin",
                     heat=True)


def _planet_transfer(dv_k: float, dv_ei: float, pc: float, planet: str) -> list[MissionEdge]:
    """Two edges for a Kerbin SOI → planet intercept → planet SOI transfer."""
    return [
        _E("kerbin_soi", f"{planet}_intercept", PT, dv_k, "Kerbin", pc=pc, attitude=True),
        _E(f"{planet}_intercept", f"{planet}_soi", PV, dv_ei, planet, attitude=True),
    ]


def _kerbin_return_transfer(dv_k: float, pc: float, planet: str) -> list[MissionEdge]:
    """Two edges for planet SOI → Kerbin intercept transfer."""
    return [
        _E(f"{planet}_soi", "kerbin_intercept", PT, dv_k, planet, pc=pc, attitude=True),
    ]


# ---------------------------------------------------------------------------
# Mission profiles
# ---------------------------------------------------------------------------
# Keys: (body_name_lowercase, mission_type)
# Values: list of profile alternatives (each alternative is list[MissionEdge])

MissionProfiles = dict[tuple[str, str], list[list[MissionEdge]]]

MISSION_PROFILES: MissionProfiles = {}


def _add(body: str, mission: str, *profiles: list[MissionEdge]) -> None:
    MISSION_PROFILES[(body, mission)] = list(profiles)


# ===========================================================================
# Kerbin (orbit only — landing/return handled as starting body)
# ===========================================================================
_add("Kerbin", "orbit", [_KERBIN_ASCENT])


# ===========================================================================
# Mun
# ===========================================================================

_MUN_ORBIT = [
    _KERBIN_ASCENT,
    _E("kerbin_low_orbit", "mun_intercept", PV, 860, "Kerbin", attitude=True),
    _E("mun_intercept", "mun_low_orbit", PV, 310, "Mun", attitude=True),
]

_MUN_LAND = _MUN_ORBIT + [
    _E("mun_low_orbit", "mun_surface", VL, 580, "Mun",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MUN_RETURN = _MUN_LAND + [
    _E("mun_surface", "mun_low_orbit", VA, 580, "Mun",
       min_twr=1.2, throttle=True, attitude=True),
    _E("mun_low_orbit", "kerbin_intercept", PV, 1170, "Mun", attitude=True),
    _KERBIN_REENTRY,
]

_MUN_SAMPLE_RETURN = _MUN_RETURN  # ladder check applied dynamically in _assess_body

_add("Mun", "orbit",         _MUN_ORBIT)
_add("Mun", "land",          _MUN_LAND)
_add("Mun", "return",        _MUN_RETURN)
_add("Mun", "sample_return", _MUN_SAMPLE_RETURN)


# ===========================================================================
# Minmus
# ===========================================================================

_MINMUS_ORBIT = [
    _KERBIN_ASCENT,
    _E("kerbin_low_orbit", "minmus_intercept", PV, 930, "Kerbin",
       pc=340, attitude=True),
    _E("minmus_intercept", "minmus_low_orbit", PV, 160, "Minmus", attitude=True),
]

_MINMUS_LAND = _MINMUS_ORBIT + [
    _E("minmus_low_orbit", "minmus_surface", VL, 180, "Minmus",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MINMUS_RETURN = _MINMUS_LAND + [
    _E("minmus_surface", "minmus_low_orbit", VA, 180, "Minmus",
       min_twr=1.2, throttle=True, attitude=True),
    _E("minmus_low_orbit", "kerbin_intercept", PV, 1090, "Minmus", attitude=True),
    _KERBIN_REENTRY,
]

_add("Minmus", "orbit",         _MINMUS_ORBIT)
_add("Minmus", "land",          _MINMUS_LAND)
_add("Minmus", "return",        _MINMUS_RETURN)
_add("Minmus", "sample_return", _MINMUS_RETURN)


# ===========================================================================
# Moho  (no atmosphere, very high dv, large plane change)
# ===========================================================================

_MOHO_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "moho_intercept", PT, 760, "Kerbin",
       pc=2520, attitude=True),
    _E("moho_intercept", "moho_low_orbit", PV, 2410, "Moho", attitude=True),
]

_MOHO_ORBIT = _MOHO_COMMON

_MOHO_LAND = _MOHO_COMMON + [
    _E("moho_low_orbit", "moho_surface", VL, 870, "Moho",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MOHO_RETURN = _MOHO_LAND + [
    _E("moho_surface", "moho_low_orbit", VA, 870, "Moho",
       min_twr=1.2, throttle=True, attitude=True),
    _E("moho_low_orbit", "kerbin_intercept", PV, 3170, "Moho",
       pc=2520, attitude=True),
    _KERBIN_REENTRY,
]

_add("Moho", "orbit",         _MOHO_ORBIT)
_add("Moho", "land",          _MOHO_LAND)
_add("Moho", "return",        _MOHO_RETURN)
_add("Moho", "sample_return", _MOHO_RETURN)


# ===========================================================================
# Eve  (thick atmosphere — land is one-way; return is extremely hard)
# ===========================================================================

_EVE_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "eve_intercept", PT, 90, "Kerbin", pc=430, attitude=True),
    _E("eve_intercept", "eve_soi", PV, 80, "Eve", attitude=True),
]

# Aero capture into Eve orbit
_EVE_ORBIT_AERO = _EVE_COMMON + [
    _E("eve_soi", "eve_low_orbit", AB, 100, "Eve", heat=True),
]

# Propulsive capture into Eve orbit
_EVE_ORBIT_PROP = _EVE_COMMON + [
    _E("eve_soi", "eve_low_orbit", PV, 1330, "Eve", attitude=True),
]

# Eve land — aero descent (only realistic option)
_EVE_LAND_AERO = _EVE_ORBIT_AERO + [
    _E("eve_low_orbit", "eve_surface", ALA, 100, "Eve", heat=True, legs=True),
]

# Eve land — propulsive descent (brute force, very expensive)
_EVE_LAND_PROP = _EVE_ORBIT_PROP + [
    _E("eve_low_orbit", "eve_surface", ALP, 1330, "Eve",
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

# Eve return from surface (atmosphere is thick — 8000 m/s ascent!)
_EVE_RETURN_AERO = _EVE_LAND_AERO + [
    _E("eve_surface", "eve_low_orbit", AT, 8000, "Eve",
       min_twr=1.5, throttle=True, attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, "Eve",
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_EVE_RETURN_PROP = _EVE_LAND_PROP + [
    _E("eve_surface", "eve_low_orbit", AT, 8000, "Eve",
       min_twr=1.5, throttle=True, attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, "Eve",
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_add("Eve", "orbit",         _EVE_ORBIT_AERO, _EVE_ORBIT_PROP)
_add("Eve", "land",          _EVE_LAND_AERO,  _EVE_LAND_PROP)
_add("Eve", "return",        _EVE_RETURN_AERO, _EVE_RETURN_PROP)
_add("Eve", "sample_return", _EVE_RETURN_AERO, _EVE_RETURN_PROP)


# ===========================================================================
# Gilly  (Eve moon, extremely low gravity)
# ===========================================================================

_GILLY_COMMON = _EVE_ORBIT_AERO + [
    _E("eve_low_orbit", "gilly_intercept", PV, 60, "Eve", attitude=True),
    _E("gilly_intercept", "gilly_low_orbit", PV, 410, "Gilly", attitude=True),
]

_GILLY_ORBIT = _GILLY_COMMON

_GILLY_LAND = _GILLY_COMMON + [
    _E("gilly_low_orbit", "gilly_surface", VL, 30, "Gilly",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_GILLY_RETURN = _GILLY_LAND + [
    _E("gilly_surface", "gilly_low_orbit", VA, 30, "Gilly",
       min_twr=1.2, throttle=True, attitude=True),
    _E("gilly_low_orbit", "eve_low_orbit", PV, 470, "Gilly", attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, "Eve",
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_add("Gilly", "orbit",         _GILLY_ORBIT)
_add("Gilly", "land",          _GILLY_LAND)
_add("Gilly", "return",        _GILLY_RETURN)
_add("Gilly", "sample_return", _GILLY_RETURN)


# ===========================================================================
# Duna  (atmosphere — multiple landing strategies)
# ===========================================================================

_DUNA_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "duna_intercept", PT, 130, "Kerbin", pc=10, attitude=True),
    _E("duna_intercept", "duna_soi", PV, 250, "Duna", attitude=True),
]

# Propulsive capture + propulsive landing
_DUNA_ORBIT_PROP = _DUNA_COMMON + [
    _E("duna_soi", "duna_low_orbit", PV, 360, "Duna", attitude=True),
]

# Aerobrake into orbit
_DUNA_ORBIT_AERO = _DUNA_COMMON + [
    _E("duna_soi", "duna_low_orbit", AB, 100, "Duna", heat=True),
]

_DUNA_LAND_PROP = _DUNA_ORBIT_PROP + [
    _E("duna_low_orbit", "duna_surface", ALP, 1450, "Duna",
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

_DUNA_LAND_AERO = _DUNA_ORBIT_AERO + [
    _E("duna_low_orbit", "duna_surface", ALA, 200, "Duna",
       heat=True, legs=True),
]

_DUNA_RETURN_PROP = _DUNA_LAND_PROP + [
    _E("duna_surface", "duna_low_orbit", AT, 1450, "Duna",
       min_twr=1.5, throttle=True, attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, "Duna",
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_DUNA_RETURN_AERO = _DUNA_LAND_AERO + [
    _E("duna_surface", "duna_low_orbit", AT, 1450, "Duna",
       min_twr=1.5, throttle=True, attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, "Duna",
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_add("Duna", "orbit",         _DUNA_ORBIT_PROP, _DUNA_ORBIT_AERO)
_add("Duna", "land",          _DUNA_LAND_PROP,  _DUNA_LAND_AERO)
_add("Duna", "return",        _DUNA_RETURN_PROP, _DUNA_RETURN_AERO)
_add("Duna", "sample_return", _DUNA_RETURN_PROP, _DUNA_RETURN_AERO)


# ===========================================================================
# Ike  (Duna moon, airless)
# ===========================================================================

_IKE_COMMON = _DUNA_ORBIT_PROP + [
    _E("duna_low_orbit", "ike_intercept", PV, 30, "Duna", attitude=True),
    _E("ike_intercept", "ike_low_orbit", PV, 180, "Ike", attitude=True),
]

_IKE_ORBIT = _IKE_COMMON

_IKE_LAND = _IKE_COMMON + [
    _E("ike_low_orbit", "ike_surface", VL, 390, "Ike",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_IKE_RETURN = _IKE_LAND + [
    _E("ike_surface", "ike_low_orbit", VA, 390, "Ike",
       min_twr=1.2, throttle=True, attitude=True),
    _E("ike_low_orbit", "duna_low_orbit", PV, 210, "Ike", attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, "Duna",
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_add("Ike", "orbit",         _IKE_ORBIT)
_add("Ike", "land",          _IKE_LAND)
_add("Ike", "return",        _IKE_RETURN)
_add("Ike", "sample_return", _IKE_RETURN)


# ===========================================================================
# Dres  (airless, significant plane change)
# ===========================================================================

_DRES_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "dres_intercept", PT, 610, "Kerbin",
       pc=1010, attitude=True),
    _E("dres_intercept", "dres_low_orbit", PV, 1290, "Dres", attitude=True),
]

_DRES_ORBIT = _DRES_COMMON

_DRES_LAND = _DRES_COMMON + [
    _E("dres_low_orbit", "dres_surface", VL, 430, "Dres",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_DRES_RETURN = _DRES_LAND + [
    _E("dres_surface", "dres_low_orbit", VA, 430, "Dres",
       min_twr=1.2, throttle=True, attitude=True),
    _E("dres_low_orbit", "kerbin_intercept", PV, 1900, "Dres",
       pc=1010, attitude=True),
    _KERBIN_REENTRY,
]

_add("Dres", "orbit",         _DRES_ORBIT)
_add("Dres", "land",          _DRES_LAND)
_add("Dres", "return",        _DRES_RETURN)
_add("Dres", "sample_return", _DRES_RETURN)


# ===========================================================================
# Jool  (cannot land; orbit only)
# ===========================================================================

_JOOL_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "jool_intercept", PT, 980, "Kerbin", pc=270, attitude=True),
    _E("jool_intercept", "jool_soi", PV, 160, "Jool", attitude=True),
]

# Aerobrake into Jool orbit
_JOOL_ORBIT_AERO = _JOOL_COMMON + [
    _E("jool_soi", "jool_low_orbit", AB, 100, "Jool", heat=True),
]

# Propulsive capture (expensive)
_JOOL_ORBIT_PROP = _JOOL_COMMON + [
    _E("jool_soi", "jool_low_orbit", PV, 2810, "Jool", attitude=True),
]

_JOOL_RETURN_AERO = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_JOOL_RETURN_PROP = _JOOL_ORBIT_PROP + [
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Jool", "orbit",  _JOOL_ORBIT_AERO, _JOOL_ORBIT_PROP)
_add("Jool", "return", _JOOL_RETURN_AERO, _JOOL_RETURN_PROP)


# ===========================================================================
# Laythe  (Jool moon, atmosphere)
# ===========================================================================

_LAYTHE_COMMON = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "laythe_intercept", PV, 930, "Jool", attitude=True),
    _E("laythe_intercept", "laythe_low_orbit", PV, 1070, "Laythe", attitude=True),
]

_LAYTHE_LAND_AERO = _LAYTHE_COMMON + [
    _E("laythe_low_orbit", "laythe_surface", ALA, 200, "Laythe",
       heat=True, legs=True),
]

_LAYTHE_LAND_PROP = _LAYTHE_COMMON + [
    _E("laythe_low_orbit", "laythe_surface", ALP, 2900, "Laythe",
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

_LAYTHE_RETURN_AERO = _LAYTHE_LAND_AERO + [
    _E("laythe_surface", "laythe_low_orbit", AT, 2900, "Laythe",
       min_twr=1.5, throttle=True, attitude=True),
    _E("laythe_low_orbit", "jool_low_orbit", PV, 2000, "Laythe", attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_LAYTHE_RETURN_PROP = _LAYTHE_LAND_PROP + [
    _E("laythe_surface", "laythe_low_orbit", AT, 2900, "Laythe",
       min_twr=1.5, throttle=True, attitude=True),
    _E("laythe_low_orbit", "jool_low_orbit", PV, 2000, "Laythe", attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Laythe", "orbit",         _LAYTHE_COMMON)
_add("Laythe", "land",          _LAYTHE_LAND_AERO, _LAYTHE_LAND_PROP)
_add("Laythe", "return",        _LAYTHE_RETURN_AERO, _LAYTHE_RETURN_PROP)
_add("Laythe", "sample_return", _LAYTHE_RETURN_AERO, _LAYTHE_RETURN_PROP)


# ===========================================================================
# Vall  (Jool moon, airless)
# ===========================================================================

_VALL_COMMON = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "vall_intercept", PV, 620, "Jool", attitude=True),
    _E("vall_intercept", "vall_low_orbit", PV, 910, "Vall", attitude=True),
]

_VALL_LAND = _VALL_COMMON + [
    _E("vall_low_orbit", "vall_surface", VL, 860, "Vall",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_VALL_RETURN = _VALL_LAND + [
    _E("vall_surface", "vall_low_orbit", VA, 860, "Vall",
       min_twr=1.2, throttle=True, attitude=True),
    _E("vall_low_orbit", "jool_low_orbit", PV, 1530, "Vall", attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Vall", "orbit",         _VALL_COMMON)
_add("Vall", "land",          _VALL_LAND)
_add("Vall", "return",        _VALL_RETURN)
_add("Vall", "sample_return", _VALL_RETURN)


# ===========================================================================
# Tylo  (Jool moon, airless, high gravity — hardest landing in the system)
# ===========================================================================

_TYLO_COMMON = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "tylo_intercept", PV, 400, "Jool", attitude=True),
    _E("tylo_intercept", "tylo_low_orbit", PV, 1100, "Tylo", attitude=True),
]

_TYLO_LAND = _TYLO_COMMON + [
    _E("tylo_low_orbit", "tylo_surface", VL, 2270, "Tylo",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_TYLO_RETURN = _TYLO_LAND + [
    _E("tylo_surface", "tylo_low_orbit", VA, 2270, "Tylo",
       min_twr=1.2, throttle=True, attitude=True),
    _E("tylo_low_orbit", "jool_low_orbit", PV, 1500, "Tylo", attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Tylo", "orbit",         _TYLO_COMMON)
_add("Tylo", "land",          _TYLO_LAND)
_add("Tylo", "return",        _TYLO_RETURN)
_add("Tylo", "sample_return", _TYLO_RETURN)


# ===========================================================================
# Bop  (Jool moon, airless, high inclination)
# ===========================================================================

_BOP_COMMON = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "bop_intercept", PV, 220, "Jool",
       pc=2440, attitude=True),
    _E("bop_intercept", "bop_low_orbit", PV, 900, "Bop", attitude=True),
]

_BOP_LAND = _BOP_COMMON + [
    _E("bop_low_orbit", "bop_surface", VL, 230, "Bop",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_BOP_RETURN = _BOP_LAND + [
    _E("bop_surface", "bop_low_orbit", VA, 230, "Bop",
       min_twr=1.2, throttle=True, attitude=True),
    _E("bop_low_orbit", "jool_low_orbit", PV, 1120, "Bop",
       pc=2440, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Bop", "orbit",         _BOP_COMMON)
_add("Bop", "land",          _BOP_LAND)
_add("Bop", "return",        _BOP_RETURN)
_add("Bop", "sample_return", _BOP_RETURN)


# ===========================================================================
# Pol  (Jool moon, airless, inclined)
# ===========================================================================

_POL_COMMON = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "pol_intercept", PV, 160, "Jool",
       pc=700, attitude=True),
    _E("pol_intercept", "pol_low_orbit", PV, 820, "Pol", attitude=True),
]

_POL_LAND = _POL_COMMON + [
    _E("pol_low_orbit", "pol_surface", VL, 130, "Pol",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_POL_RETURN = _POL_LAND + [
    _E("pol_surface", "pol_low_orbit", VA, 130, "Pol",
       min_twr=1.2, throttle=True, attitude=True),
    _E("pol_low_orbit", "jool_low_orbit", PV, 980, "Pol",
       pc=700, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, "Jool",
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add("Pol", "orbit",         _POL_COMMON)
_add("Pol", "land",          _POL_LAND)
_add("Pol", "return",        _POL_RETURN)
_add("Pol", "sample_return", _POL_RETURN)


# ===========================================================================
# Eeloo  (distant, icy, no atmosphere)
# ===========================================================================

_EELOO_COMMON = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "eeloo_intercept", PT, 1140, "Kerbin",
       pc=1330, attitude=True),
    _E("eeloo_intercept", "eeloo_low_orbit", PV, 1370, "Eeloo", attitude=True),
]

_EELOO_LAND = _EELOO_COMMON + [
    _E("eeloo_low_orbit", "eeloo_surface", VL, 620, "Eeloo",
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_EELOO_RETURN = _EELOO_LAND + [
    _E("eeloo_surface", "eeloo_low_orbit", VA, 620, "Eeloo",
       min_twr=1.2, throttle=True, attitude=True),
    _E("eeloo_low_orbit", "kerbin_intercept", PV, 2510, "Eeloo",
       pc=1330, attitude=True),
    _KERBIN_REENTRY,
]

_add("Eeloo", "orbit",         _EELOO_COMMON)
_add("Eeloo", "land",          _EELOO_LAND)
_add("Eeloo", "return",        _EELOO_RETURN)
_add("Eeloo", "sample_return", _EELOO_RETURN)


# ===========================================================================
# Kerbol  (orbit only — cannot land)
# ===========================================================================

_KERBOL_ORBIT = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "kerbol_low_orbit", PT, 6000, "Kerbol",
       attitude=True),
]

_KERBOL_RETURN = _KERBOL_ORBIT + [
    _E("kerbol_low_orbit", "kerbin_intercept", PV, 6000, "Kerbol", attitude=True),
    _KERBIN_REENTRY,
]

_add("Kerbol", "orbit",  _KERBOL_ORBIT)
_add("Kerbol", "return", _KERBOL_RETURN)


# ---------------------------------------------------------------------------
# Science budget estimator (used by tech-tree access rules)
# ---------------------------------------------------------------------------

def science_budget(
    body: Body,
    has_thermometer: bool,
    has_barometer: bool,
    has_capsule: bool,
    can_land_crewed: bool,
) -> float:
    """
    Estimate the total science collectible from *body* given the player's
    current instrument and crew capabilities.

    Conservative (golden rule): uses min(fly_low, fly_high) for flying
    situations, does not count surface-sample crew value without crewed landing.

    Base instrument values (from KSP science definitions):
      Thermometer: 8   Barometer: 12   Crew Report: 5   EVA Report: 8
      Surface Sample: 30

    Science = base_value * situation_multiplier * recovery_factor
    The recovery_factor (0.25 for transmit) is NOT applied here — we assume
    the player physically recovers the data, giving full science value.
    This is the upper bound; safety factors in can_afford_tier() discount it.
    """
    instrument_val: float = 0.0
    if has_thermometer:
        instrument_val += 8.0
    if has_barometer:
        instrument_val += 12.0

    crew_orbital_val: float = (5.0 + 8.0) if has_capsule else 0.0
    crew_surface_val: float = (5.0 + 8.0 + 30.0) if (has_capsule and can_land_crewed) else 0.0

    # Orbital science (global, not per-biome)
    orbital = (instrument_val + crew_orbital_val) * (body.space_low_mult + body.space_high_mult)

    # Flying science (atmosphere only; use min to underestimate)
    flying = 0.0
    if body.has_atmosphere and body.fly_low_mult > 0 and body.fly_high_mult > 0:
        flying = (instrument_val + crew_orbital_val) * min(body.fly_low_mult, body.fly_high_mult)

    # Landed science (scales with biome count)
    landed = 0.0
    if body.can_land and body.num_biomes > 0:
        landed = (
            instrument_val * body.landed_mult
            + crew_surface_val * body.landed_mult
        ) * body.num_biomes

    # Splashed science (ocean biomes only)
    splashed = 0.0
    if body.has_ocean and body.num_splash_biomes > 0:
        splashed = (
            instrument_val * body.splashed_mult
            + crew_surface_val * body.splashed_mult
        ) * body.num_splash_biomes

    return orbital + flying + landed + splashed


# ---------------------------------------------------------------------------
# Parent gating helper
# ---------------------------------------------------------------------------

def parent_chain(body: Body) -> list[str]:
    """Return [body.name, parent.name, grandparent.name, ...] up to a planet."""
    chain: list[str] = []
    current: Optional[Body] = body
    while current is not None:
        chain.append(current.name)
        current = BODY_BY_NAME.get(current.parent) if current.parent else None
    return chain
