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
from enum import Enum, StrEnum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Body names — single source of truth, catches typos at import time
# ---------------------------------------------------------------------------

class MissionType(StrEnum):
    """Mission profile types — keys into MISSION_PROFILES."""
    ORBIT = "orbit"
    LAND = "land"
    RETURN = "return"
    SAMPLE_RETURN = "sample_return"
    FLAG_PLANT = "flag_plant"
    ESCAPE = "escape"
    # Kerbin-only mission types (not in MISSION_PROFILES)
    SOUNDING = "sounding"
    FIRST_LAUNCH = "first_launch"
    FIRST_LANDING = "first_landing"
    FIRST_STAGING = "first_staging"
    SPLASHDOWN = "splashdown"


class BodyName(StrEnum):
    KERBIN = "Kerbin"
    MUN = "Mun"
    MINMUS = "Minmus"
    MOHO = "Moho"
    EVE = "Eve"
    GILLY = "Gilly"
    DUNA = "Duna"
    IKE = "Ike"
    DRES = "Dres"
    JOOL = "Jool"
    LAYTHE = "Laythe"
    VALL = "Vall"
    TYLO = "Tylo"
    BOP = "Bop"
    POL = "Pol"
    EELOO = "Eeloo"
    KERBOL = "Kerbol"


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
    name: BodyName
    parent: Optional[BodyName]      # None for planets, parent name for moons
    surface_gravity: float          # m/s²
    has_atmosphere: bool
    atm_pressure_kpa: float         # sea-level, 0 for vacuum bodies
    atm_density_kg_m3: float        # sea-level air density, 0 for vacuum
    can_land: bool
    low_orbit_alt_km: float         # defines "low orbit" for location checks
    solar_distance_au: float        # Kerbin = 1.0, used for ION/solar logic
    landing_leg_tier: int           # minimum leg tier required for landing
    min_relay_tier: int             # 0=none, 1=local, 2=inner, 3=mid, 4=outer
    power_requirement: str          # "solar" | "solar_marginal" | "rtg"
    eva_jetpack_twr: float          # precomputed: 0.5/(0.09375*surface_gravity)
    dv: BodyDeltaV

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
    all_parts_proxy: bool = False    # True = return rules use all-parts proxy (can't model ascent)


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
    body: BodyName                      # body name for physics lookups
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
    name=BodyName.KERBIN, parent=None,
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
    has_ocean=True, num_biomes=9, num_splash_biomes=2,
    space_low_mult=1.5, space_high_mult=1.0,
    fly_low_mult=1.0, fly_high_mult=0.7,
    landed_mult=0.3, splashed_mult=0.4,
)

MUN = Body(
    name=BodyName.MUN, parent=BodyName.KERBIN,
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
    num_biomes=7,
    space_low_mult=4.0, space_high_mult=2.0,
    landed_mult=9.0,
)

MINMUS = Body(
    name=BodyName.MINMUS, parent=BodyName.KERBIN,
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
    num_biomes=9,
    space_low_mult=5.0, space_high_mult=2.5,
    landed_mult=12.0,
)

MOHO = Body(
    name=BodyName.MOHO, parent=None,
    surface_gravity=2.70, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=20,
    solar_distance_au=0.34,
    landing_leg_tier=2, min_relay_tier=2,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(2.70),
    dv=BodyDeltaV(
        dvGL=870, dvLE=None, dvEI=None, dvK=760,
        dvLI=2410, dvPL=None, dvPE=None, dvPlaneChange=2520,
    ),
    num_biomes=6,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=9.0,
)

EVE = Body(
    name=BodyName.EVE, parent=None,
    surface_gravity=16.7, has_atmosphere=True,
    atm_pressure_kpa=506.625, atm_density_kg_m3=5.0,
    can_land=True, low_orbit_alt_km=90,
    solar_distance_au=0.72,
    landing_leg_tier=2, min_relay_tier=3,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(16.7),
    dv=BodyDeltaV(
        dvGL=8000, dvLE=1330, dvEI=80, dvK=90,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=430,
    ),
    has_ocean=True, num_biomes=8, num_splash_biomes=3,
    space_low_mult=8.0, space_high_mult=4.0,
    fly_low_mult=2.0, fly_high_mult=1.5,
    landed_mult=8.0, splashed_mult=8.0,
    all_parts_proxy=True,
)

GILLY = Body(
    name=BodyName.GILLY, parent=BodyName.EVE,
    surface_gravity=0.049, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=6,
    solar_distance_au=0.72,
    landing_leg_tier=1, min_relay_tier=3,
    power_requirement="solar",
    eva_jetpack_twr=_jetpack_twr(0.049),
    dv=BodyDeltaV(
        dvGL=30, dvLE=None, dvEI=None, dvK=None,
        dvLI=410, dvPL=None, dvPE=60, dvPlaneChange=0,
    ),
    num_biomes=3,
    space_low_mult=9.0, space_high_mult=4.5,
    landed_mult=12.0,
)

DUNA = Body(
    name=BodyName.DUNA, parent=None,
    surface_gravity=2.94, has_atmosphere=True,
    atm_pressure_kpa=6.755, atm_density_kg_m3=0.096,
    can_land=True, low_orbit_alt_km=50,
    solar_distance_au=1.52,
    landing_leg_tier=2, min_relay_tier=3,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(2.94),
    dv=BodyDeltaV(
        dvGL=1450, dvLE=360, dvEI=250, dvK=130,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=10,
    ),
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    fly_low_mult=1.5, fly_high_mult=1.2,
    landed_mult=8.0,
)

IKE = Body(
    name=BodyName.IKE, parent=BodyName.DUNA,
    surface_gravity=1.10, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=1.52,
    landing_leg_tier=2, min_relay_tier=3,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(1.10),
    dv=BodyDeltaV(
        dvGL=390, dvLE=None, dvEI=None, dvK=None,
        dvLI=180, dvPL=None, dvPE=30, dvPlaneChange=0,
    ),
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=8.0,
)

DRES = Body(
    name=BodyName.DRES, parent=None,
    surface_gravity=2.94, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=25,
    solar_distance_au=2.65,
    landing_leg_tier=2, min_relay_tier=3,
    power_requirement="solar_marginal",
    eva_jetpack_twr=_jetpack_twr(2.94),
    dv=BodyDeltaV(
        dvGL=430, dvLE=None, dvEI=None, dvK=610,
        dvLI=1290, dvPL=None, dvPE=None, dvPlaneChange=1010,
    ),
    num_biomes=5,
    space_low_mult=8.0, space_high_mult=4.0,
    landed_mult=8.0,
)

JOOL = Body(
    name=BodyName.JOOL, parent=None,
    surface_gravity=7.85, has_atmosphere=True,
    atm_pressure_kpa=1519.88, atm_density_kg_m3=10.0,
    can_land=False, low_orbit_alt_km=210,
    solar_distance_au=5.20,
    landing_leg_tier=0, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=14000, dvLE=2810, dvEI=160, dvK=980,
        dvLI=None, dvPL=None, dvPE=None, dvPlaneChange=270,
    ),
    num_biomes=0,
    space_low_mult=12.0, space_high_mult=6.0,
    fly_low_mult=6.0, fly_high_mult=4.0,
)

LAYTHE = Body(
    name=BodyName.LAYTHE, parent=BodyName.JOOL,
    surface_gravity=7.85, has_atmosphere=True,
    atm_pressure_kpa=60.795, atm_density_kg_m3=0.73,
    can_land=True, low_orbit_alt_km=60,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=2900, dvLE=None, dvEI=None, dvK=None,
        dvLI=1070, dvPL=None, dvPE=930, dvPlaneChange=0,
    ),
    has_ocean=True, num_biomes=9, num_splash_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    fly_low_mult=4.0, fly_high_mult=3.0,
    landed_mult=14.0, splashed_mult=10.0,
    all_parts_proxy=True,
)

VALL = Body(
    name=BodyName.VALL, parent=BodyName.JOOL,
    surface_gravity=2.31, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=15,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(2.31),
    dv=BodyDeltaV(
        dvGL=860, dvLE=None, dvEI=None, dvK=None,
        dvLI=910, dvPL=None, dvPE=620, dvPlaneChange=0,
    ),
    num_biomes=9,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

TYLO = Body(
    name=BodyName.TYLO, parent=BodyName.JOOL,
    surface_gravity=7.85, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=30,
    solar_distance_au=5.20,
    landing_leg_tier=2, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(7.85),
    dv=BodyDeltaV(
        dvGL=2270, dvLE=None, dvEI=None, dvK=None,
        dvLI=1100, dvPL=None, dvPE=400, dvPlaneChange=0,
    ),
    num_biomes=6,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
    all_parts_proxy=True,
)

BOP = Body(
    name=BodyName.BOP, parent=BodyName.JOOL,
    surface_gravity=0.589, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=5.20,
    landing_leg_tier=1, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(0.589),
    dv=BodyDeltaV(
        dvGL=230, dvLE=None, dvEI=None, dvK=None,
        dvLI=900, dvPL=None, dvPE=220, dvPlaneChange=2440,
    ),
    num_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

POL = Body(
    name=BodyName.POL, parent=BodyName.JOOL,
    surface_gravity=0.373, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=6,
    solar_distance_au=5.20,
    landing_leg_tier=1, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(0.373),
    dv=BodyDeltaV(
        dvGL=130, dvLE=None, dvEI=None, dvK=None,
        dvLI=820, dvPL=None, dvPE=160, dvPlaneChange=700,
    ),
    num_biomes=4,
    space_low_mult=12.0, space_high_mult=6.0,
    landed_mult=12.0,
)

EELOO = Body(
    name=BodyName.EELOO, parent=None,
    surface_gravity=1.72, has_atmosphere=False,
    atm_pressure_kpa=0, atm_density_kg_m3=0,
    can_land=True, low_orbit_alt_km=10,
    solar_distance_au=6.0,
    landing_leg_tier=2, min_relay_tier=4,
    power_requirement="rtg",
    eva_jetpack_twr=_jetpack_twr(1.72),
    dv=BodyDeltaV(
        dvGL=620, dvLE=None, dvEI=None, dvK=1140,
        dvLI=1370, dvPL=None, dvPE=None, dvPlaneChange=1330,
    ),
    num_biomes=7,
    space_low_mult=15.0, space_high_mult=7.5,
    landed_mult=15.0,
)

KERBOL = Body(
    name=BodyName.KERBOL, parent=None,
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

BODY_BY_NAME: dict[BodyName, Body] = {b.name: b for b in ALL_BODIES}


# ---------------------------------------------------------------------------
# Mission profile builder helpers
# ---------------------------------------------------------------------------

def _E(  # shorthand for MissionEdge
    src: str, dst: str, et: EdgeType, dv: float, body: BodyName,
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
_KERBIN_ASCENT = _E("kerbin_surface", "kerbin_low_orbit", AT, 3400, BodyName.KERBIN,
                    min_twr=1.5, throttle=True, attitude=True)
# Kerbin escape to SOI edge
_KERBIN_ESCAPE = _E("kerbin_low_orbit", "kerbin_soi", PV, 950, BodyName.KERBIN,
                    attitude=True)
# Kerbin aero reentry (return missions)
_KERBIN_REENTRY = _E("kerbin_intercept", "kerbin_surface", ALA, 100, BodyName.KERBIN,
                     heat=True)


def _planet_transfer(dv_k: float, dv_ei: float, pc: float, planet: BodyName) -> list[MissionEdge]:
    """Two edges for a Kerbin SOI → planet intercept → planet SOI transfer."""
    return [
        _E("kerbin_soi", f"{planet}_intercept", PT, dv_k, BodyName.KERBIN, pc=pc, attitude=True),
        _E(f"{planet}_intercept", f"{planet}_soi", PV, dv_ei, planet, attitude=True),
    ]


def _kerbin_return_transfer(dv_k: float, pc: float, planet: BodyName) -> list[MissionEdge]:
    """Two edges for planet SOI → Kerbin intercept transfer."""
    return [
        _E(f"{planet}_soi", "kerbin_intercept", PT, dv_k, planet, pc=pc, attitude=True),
    ]


# ---------------------------------------------------------------------------
# Mission profiles
# ---------------------------------------------------------------------------
# Keys: (body_name_lowercase, mission_type)
# Values: list of profile alternatives (each alternative is list[MissionEdge])

MissionProfiles = dict[tuple[BodyName, str], list[list[MissionEdge]]]

MISSION_PROFILES: MissionProfiles = {}


def _add(body: BodyName, mission: MissionType, *profiles: list[MissionEdge]) -> None:
    MISSION_PROFILES[(body, mission)] = list(profiles)


# ===========================================================================
# Kerbin
# ===========================================================================
# Deorbit: reentry from low orbit (heat shield + parachutes)
_KERBIN_DEORBIT = _E("kerbin_low_orbit", "kerbin_surface", ALA, 100, BodyName.KERBIN,
                     heat=True)

_add(BodyName.KERBIN, MissionType.ORBIT,  [_KERBIN_ASCENT])
_add(BodyName.KERBIN, MissionType.ESCAPE, [_KERBIN_ASCENT, _KERBIN_ESCAPE])
_add(BodyName.KERBIN, MissionType.LAND,   [_KERBIN_ASCENT, _KERBIN_DEORBIT])
_add(BodyName.KERBIN, MissionType.FLAG_PLANT,    [])  # 0 dv — walk out and plant
_add(BodyName.KERBIN, MissionType.RETURN,        [_KERBIN_ASCENT, _KERBIN_DEORBIT])
# WARNING: sample_return MUST stay empty — kerbal EVAs from the launchpad,
# takes a surface sample, and recovers. No rocket needed. Do not add edges.
_add(BodyName.KERBIN, MissionType.SAMPLE_RETURN, [])


# ===========================================================================
# Mun
# ===========================================================================

_MUN_TRANSFER = [
    _KERBIN_ASCENT,
    _E("kerbin_low_orbit", "mun_intercept", PV, 860, BodyName.KERBIN, attitude=True),
    _E("mun_intercept", "mun_soi", PV, 0, BodyName.MUN, attitude=True),  # SOI entry
]

_MUN_ORBIT = _MUN_TRANSFER + [
    _E("mun_soi", "mun_low_orbit", PV, 310, BodyName.MUN, attitude=True),  # orbit insertion
]

_MUN_LAND = _MUN_ORBIT + [
    _E("mun_low_orbit", "mun_surface", VL, 580, BodyName.MUN,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MUN_RETURN = _MUN_LAND + [
    _E("mun_surface", "mun_low_orbit", VA, 580, BodyName.MUN,
       min_twr=1.2, throttle=True, attitude=True),
    _E("mun_low_orbit", "kerbin_intercept", PV, 1170, BodyName.MUN, attitude=True),
    _KERBIN_REENTRY,
]

_MUN_SAMPLE_RETURN = _MUN_RETURN  # ladder check applied dynamically in _assess_body

_add(BodyName.MUN, MissionType.ESCAPE,        _MUN_TRANSFER)
_add(BodyName.MUN, MissionType.ORBIT,         _MUN_ORBIT)
_add(BodyName.MUN, MissionType.LAND,          _MUN_LAND)
_add(BodyName.MUN, MissionType.RETURN,        _MUN_RETURN)
_add(BodyName.MUN, MissionType.SAMPLE_RETURN, _MUN_SAMPLE_RETURN)


# ===========================================================================
# Minmus
# ===========================================================================

_MINMUS_TRANSFER = [
    _KERBIN_ASCENT,
    _E("kerbin_low_orbit", "minmus_intercept", PV, 930, BodyName.KERBIN,
       pc=340, attitude=True),
    _E("minmus_intercept", "minmus_soi", PV, 0, BodyName.MINMUS, attitude=True),  # SOI entry
]

_MINMUS_ORBIT = _MINMUS_TRANSFER + [
    _E("minmus_soi", "minmus_low_orbit", PV, 160, BodyName.MINMUS, attitude=True),  # orbit insertion
]

_MINMUS_LAND = _MINMUS_ORBIT + [
    _E("minmus_low_orbit", "minmus_surface", VL, 180, BodyName.MINMUS,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MINMUS_RETURN = _MINMUS_LAND + [
    _E("minmus_surface", "minmus_low_orbit", VA, 180, BodyName.MINMUS,
       min_twr=1.2, throttle=True, attitude=True),
    _E("minmus_low_orbit", "kerbin_intercept", PV, 1090, BodyName.MINMUS, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.MINMUS, MissionType.ESCAPE,        _MINMUS_TRANSFER)
_add(BodyName.MINMUS, MissionType.ORBIT,         _MINMUS_ORBIT)
_add(BodyName.MINMUS, MissionType.LAND,          _MINMUS_LAND)
_add(BodyName.MINMUS, MissionType.RETURN,        _MINMUS_RETURN)
_add(BodyName.MINMUS, MissionType.SAMPLE_RETURN, _MINMUS_RETURN)


# ===========================================================================
# Moho  (no atmosphere, very high dv, large plane change)
# ===========================================================================

_MOHO_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "moho_intercept", PT, 760, BodyName.KERBIN,
       pc=2520, attitude=True),
    _E("moho_intercept", "moho_soi", PV, 0, BodyName.MOHO, attitude=True),  # SOI entry
]

_MOHO_ORBIT = _MOHO_TRANSFER + [
    _E("moho_soi", "moho_low_orbit", PV, 2410, BodyName.MOHO, attitude=True),  # orbit insertion
]

_MOHO_LAND = _MOHO_ORBIT + [
    _E("moho_low_orbit", "moho_surface", VL, 870, BodyName.MOHO,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_MOHO_RETURN = _MOHO_LAND + [
    _E("moho_surface", "moho_low_orbit", VA, 870, BodyName.MOHO,
       min_twr=1.2, throttle=True, attitude=True),
    _E("moho_low_orbit", "kerbin_intercept", PV, 3170, BodyName.MOHO,
       pc=2520, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.MOHO, MissionType.ESCAPE,        _MOHO_TRANSFER)
_add(BodyName.MOHO, MissionType.ORBIT,         _MOHO_ORBIT)
_add(BodyName.MOHO, MissionType.LAND,          _MOHO_LAND)
_add(BodyName.MOHO, MissionType.RETURN,        _MOHO_RETURN)
_add(BodyName.MOHO, MissionType.SAMPLE_RETURN, _MOHO_RETURN)


# ===========================================================================
# Eve  (thick atmosphere — land is one-way; return is extremely hard)
# ===========================================================================

_EVE_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "eve_intercept", PT, 90, BodyName.KERBIN, pc=430, attitude=True),
    _E("eve_intercept", "eve_soi", PV, 80, BodyName.EVE, attitude=True),
]

# Aero capture into Eve orbit
_EVE_ORBIT_AERO = _EVE_TRANSFER + [
    _E("eve_soi", "eve_low_orbit", AB, 100, BodyName.EVE, heat=True),
]

# Propulsive capture into Eve orbit
_EVE_ORBIT_PROP = _EVE_TRANSFER + [
    _E("eve_soi", "eve_low_orbit", PV, 1330, BodyName.EVE, attitude=True),
]

# Eve land — aero descent (only realistic option)
_EVE_LAND_AERO = _EVE_ORBIT_AERO + [
    _E("eve_low_orbit", "eve_surface", ALA, 100, BodyName.EVE, heat=True, legs=True),
]

# Eve land — propulsive descent (brute force, very expensive)
_EVE_LAND_PROP = _EVE_ORBIT_PROP + [
    _E("eve_low_orbit", "eve_surface", ALP, 1330, BodyName.EVE,
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

# Eve return from surface (atmosphere is thick — 8000 m/s ascent!)
_EVE_RETURN_AERO = _EVE_LAND_AERO + [
    _E("eve_surface", "eve_low_orbit", AT, 8000, BodyName.EVE,
       min_twr=1.5, throttle=True, attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, BodyName.EVE,
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_EVE_RETURN_PROP = _EVE_LAND_PROP + [
    _E("eve_surface", "eve_low_orbit", AT, 8000, BodyName.EVE,
       min_twr=1.5, throttle=True, attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, BodyName.EVE,
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.EVE, MissionType.ESCAPE,        _EVE_TRANSFER)
_add(BodyName.EVE, MissionType.ORBIT,         _EVE_ORBIT_AERO, _EVE_ORBIT_PROP)
_add(BodyName.EVE, MissionType.LAND,          _EVE_LAND_AERO,  _EVE_LAND_PROP)
_add(BodyName.EVE, MissionType.RETURN,        _EVE_RETURN_AERO, _EVE_RETURN_PROP)
_add(BodyName.EVE, MissionType.SAMPLE_RETURN, _EVE_RETURN_AERO, _EVE_RETURN_PROP)


# ===========================================================================
# Gilly  (Eve moon, extremely low gravity)
# ===========================================================================

_GILLY_TRANSFER = _EVE_ORBIT_AERO + [
    _E("eve_low_orbit", "gilly_intercept", PV, 60, BodyName.EVE, attitude=True),
    _E("gilly_intercept", "gilly_soi", PV, 0, BodyName.GILLY, attitude=True),  # SOI entry
]

_GILLY_ORBIT = _GILLY_TRANSFER + [
    _E("gilly_soi", "gilly_low_orbit", PV, 410, BodyName.GILLY, attitude=True),  # orbit insertion
]

_GILLY_LAND = _GILLY_ORBIT + [
    _E("gilly_low_orbit", "gilly_surface", VL, 30, BodyName.GILLY,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_GILLY_RETURN = _GILLY_LAND + [
    _E("gilly_surface", "gilly_low_orbit", VA, 30, BodyName.GILLY,
       min_twr=1.2, throttle=True, attitude=True),
    _E("gilly_low_orbit", "eve_low_orbit", PV, 470, BodyName.GILLY, attitude=True),
    _E("eve_low_orbit", "kerbin_intercept", PV, 1420, BodyName.EVE,
       pc=430, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.GILLY, MissionType.ESCAPE,        _GILLY_TRANSFER)
_add(BodyName.GILLY, MissionType.ORBIT,         _GILLY_ORBIT)
_add(BodyName.GILLY, MissionType.LAND,          _GILLY_LAND)
_add(BodyName.GILLY, MissionType.RETURN,        _GILLY_RETURN)
_add(BodyName.GILLY, MissionType.SAMPLE_RETURN, _GILLY_RETURN)


# ===========================================================================
# Duna  (atmosphere — multiple landing strategies)
# ===========================================================================

_DUNA_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "duna_intercept", PT, 130, BodyName.KERBIN, pc=10, attitude=True),
    _E("duna_intercept", "duna_soi", PV, 250, BodyName.DUNA, attitude=True),  # SOI entry
]

# Propulsive capture + propulsive landing
_DUNA_ORBIT_PROP = _DUNA_TRANSFER + [
    _E("duna_soi", "duna_low_orbit", PV, 360, BodyName.DUNA, attitude=True),
]

# Aerobrake into orbit
_DUNA_ORBIT_AERO = _DUNA_TRANSFER + [
    _E("duna_soi", "duna_low_orbit", AB, 100, BodyName.DUNA, heat=True),
]

_DUNA_LAND_PROP = _DUNA_ORBIT_PROP + [
    _E("duna_low_orbit", "duna_surface", ALP, 1450, BodyName.DUNA,
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

_DUNA_LAND_AERO = _DUNA_ORBIT_AERO + [
    _E("duna_low_orbit", "duna_surface", ALA, 200, BodyName.DUNA,
       heat=True, legs=True),
]

_DUNA_RETURN_PROP = _DUNA_LAND_PROP + [
    _E("duna_surface", "duna_low_orbit", AT, 1450, BodyName.DUNA,
       min_twr=1.5, throttle=True, attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, BodyName.DUNA,
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_DUNA_RETURN_AERO = _DUNA_LAND_AERO + [
    _E("duna_surface", "duna_low_orbit", AT, 1450, BodyName.DUNA,
       min_twr=1.5, throttle=True, attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, BodyName.DUNA,
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.DUNA, MissionType.ESCAPE,        _DUNA_TRANSFER)
_add(BodyName.DUNA, MissionType.ORBIT,         _DUNA_ORBIT_PROP, _DUNA_ORBIT_AERO)
_add(BodyName.DUNA, MissionType.LAND,          _DUNA_LAND_PROP,  _DUNA_LAND_AERO)
_add(BodyName.DUNA, MissionType.RETURN,        _DUNA_RETURN_PROP, _DUNA_RETURN_AERO)
_add(BodyName.DUNA, MissionType.SAMPLE_RETURN, _DUNA_RETURN_PROP, _DUNA_RETURN_AERO)


# ===========================================================================
# Ike  (Duna moon, airless)
# ===========================================================================

_IKE_TRANSFER = _DUNA_ORBIT_PROP + [
    _E("duna_low_orbit", "ike_intercept", PV, 30, BodyName.DUNA, attitude=True),
    _E("ike_intercept", "ike_soi", PV, 0, BodyName.IKE, attitude=True),  # SOI entry
]

_IKE_ORBIT = _IKE_TRANSFER + [
    _E("ike_soi", "ike_low_orbit", PV, 180, BodyName.IKE, attitude=True),  # orbit insertion
]

_IKE_LAND = _IKE_ORBIT + [
    _E("ike_low_orbit", "ike_surface", VL, 390, BodyName.IKE,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_IKE_RETURN = _IKE_LAND + [
    _E("ike_surface", "ike_low_orbit", VA, 390, BodyName.IKE,
       min_twr=1.2, throttle=True, attitude=True),
    _E("ike_low_orbit", "duna_low_orbit", PV, 210, BodyName.IKE, attitude=True),
    _E("duna_low_orbit", "kerbin_intercept", PV, 610, BodyName.DUNA,
       pc=10, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.IKE, MissionType.ESCAPE,        _IKE_TRANSFER)
_add(BodyName.IKE, MissionType.ORBIT,         _IKE_ORBIT)
_add(BodyName.IKE, MissionType.LAND,          _IKE_LAND)
_add(BodyName.IKE, MissionType.RETURN,        _IKE_RETURN)
_add(BodyName.IKE, MissionType.SAMPLE_RETURN, _IKE_RETURN)


# ===========================================================================
# Dres  (airless, significant plane change)
# ===========================================================================

_DRES_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "dres_intercept", PT, 610, BodyName.KERBIN,
       pc=1010, attitude=True),
    _E("dres_intercept", "dres_soi", PV, 0, BodyName.DRES, attitude=True),  # SOI entry
]

_DRES_ORBIT = _DRES_TRANSFER + [
    _E("dres_soi", "dres_low_orbit", PV, 1290, BodyName.DRES, attitude=True),  # orbit insertion
]

_DRES_LAND = _DRES_ORBIT + [
    _E("dres_low_orbit", "dres_surface", VL, 430, BodyName.DRES,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_DRES_RETURN = _DRES_LAND + [
    _E("dres_surface", "dres_low_orbit", VA, 430, BodyName.DRES,
       min_twr=1.2, throttle=True, attitude=True),
    _E("dres_low_orbit", "kerbin_intercept", PV, 1900, BodyName.DRES,
       pc=1010, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.DRES, MissionType.ESCAPE,        _DRES_TRANSFER)
_add(BodyName.DRES, MissionType.ORBIT,         _DRES_ORBIT)
_add(BodyName.DRES, MissionType.LAND,          _DRES_LAND)
_add(BodyName.DRES, MissionType.RETURN,        _DRES_RETURN)
_add(BodyName.DRES, MissionType.SAMPLE_RETURN, _DRES_RETURN)


# ===========================================================================
# Jool  (cannot land; orbit only)
# ===========================================================================

_JOOL_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "jool_intercept", PT, 980, BodyName.KERBIN, pc=270, attitude=True),
    _E("jool_intercept", "jool_soi", PV, 160, BodyName.JOOL, attitude=True),  # SOI entry
]

# Aerobrake into Jool orbit
_JOOL_ORBIT_AERO = _JOOL_TRANSFER + [
    _E("jool_soi", "jool_low_orbit", AB, 100, BodyName.JOOL, heat=True),
]

# Propulsive capture (expensive)
_JOOL_ORBIT_PROP = _JOOL_TRANSFER + [
    _E("jool_soi", "jool_low_orbit", PV, 2810, BodyName.JOOL, attitude=True),
]

_JOOL_RETURN_AERO = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_JOOL_RETURN_PROP = _JOOL_ORBIT_PROP + [
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.JOOL, MissionType.ESCAPE, _JOOL_TRANSFER)
_add(BodyName.JOOL, MissionType.ORBIT,  _JOOL_ORBIT_AERO, _JOOL_ORBIT_PROP)
_add(BodyName.JOOL, MissionType.RETURN, _JOOL_RETURN_AERO, _JOOL_RETURN_PROP)


# ===========================================================================
# Laythe  (Jool moon, atmosphere)
# ===========================================================================

_LAYTHE_TRANSFER = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "laythe_intercept", PV, 930, BodyName.JOOL, attitude=True),
    _E("laythe_intercept", "laythe_soi", PV, 0, BodyName.LAYTHE, attitude=True),  # SOI entry
]

_LAYTHE_ORBIT = _LAYTHE_TRANSFER + [
    _E("laythe_soi", "laythe_low_orbit", PV, 1070, BodyName.LAYTHE, attitude=True),  # orbit insertion
]

_LAYTHE_LAND_AERO = _LAYTHE_ORBIT + [
    _E("laythe_low_orbit", "laythe_surface", ALA, 200, BodyName.LAYTHE,
       heat=True, legs=True),
]

_LAYTHE_LAND_PROP = _LAYTHE_ORBIT + [
    _E("laythe_low_orbit", "laythe_surface", ALP, 2900, BodyName.LAYTHE,
       min_twr=1.5, throttle=True, attitude=True, legs=True),
]

_LAYTHE_RETURN_AERO = _LAYTHE_LAND_AERO + [
    _E("laythe_surface", "laythe_low_orbit", AT, 2900, BodyName.LAYTHE,
       min_twr=1.5, throttle=True, attitude=True),
    _E("laythe_low_orbit", "jool_low_orbit", PV, 2000, BodyName.LAYTHE, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_LAYTHE_RETURN_PROP = _LAYTHE_LAND_PROP + [
    _E("laythe_surface", "laythe_low_orbit", AT, 2900, BodyName.LAYTHE,
       min_twr=1.5, throttle=True, attitude=True),
    _E("laythe_low_orbit", "jool_low_orbit", PV, 2000, BodyName.LAYTHE, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.LAYTHE, MissionType.ESCAPE,        _LAYTHE_TRANSFER)
_add(BodyName.LAYTHE, MissionType.ORBIT,         _LAYTHE_ORBIT)
_add(BodyName.LAYTHE, MissionType.LAND,          _LAYTHE_LAND_AERO, _LAYTHE_LAND_PROP)
_add(BodyName.LAYTHE, MissionType.RETURN,        _LAYTHE_RETURN_AERO, _LAYTHE_RETURN_PROP)
_add(BodyName.LAYTHE, MissionType.SAMPLE_RETURN, _LAYTHE_RETURN_AERO, _LAYTHE_RETURN_PROP)


# ===========================================================================
# Vall  (Jool moon, airless)
# ===========================================================================

_VALL_TRANSFER = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "vall_intercept", PV, 620, BodyName.JOOL, attitude=True),
    _E("vall_intercept", "vall_soi", PV, 0, BodyName.VALL, attitude=True),  # SOI entry
]

_VALL_ORBIT = _VALL_TRANSFER + [
    _E("vall_soi", "vall_low_orbit", PV, 910, BodyName.VALL, attitude=True),  # orbit insertion
]

_VALL_LAND = _VALL_ORBIT + [
    _E("vall_low_orbit", "vall_surface", VL, 860, BodyName.VALL,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_VALL_RETURN = _VALL_LAND + [
    _E("vall_surface", "vall_low_orbit", VA, 860, BodyName.VALL,
       min_twr=1.2, throttle=True, attitude=True),
    _E("vall_low_orbit", "jool_low_orbit", PV, 1530, BodyName.VALL, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.VALL, MissionType.ESCAPE,        _VALL_TRANSFER)
_add(BodyName.VALL, MissionType.ORBIT,         _VALL_ORBIT)
_add(BodyName.VALL, MissionType.LAND,          _VALL_LAND)
_add(BodyName.VALL, MissionType.RETURN,        _VALL_RETURN)
_add(BodyName.VALL, MissionType.SAMPLE_RETURN, _VALL_RETURN)


# ===========================================================================
# Tylo  (Jool moon, airless, high gravity — hardest landing in the system)
# ===========================================================================

_TYLO_TRANSFER = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "tylo_intercept", PV, 400, BodyName.JOOL, attitude=True),
    _E("tylo_intercept", "tylo_soi", PV, 0, BodyName.TYLO, attitude=True),  # SOI entry
]

_TYLO_ORBIT = _TYLO_TRANSFER + [
    _E("tylo_soi", "tylo_low_orbit", PV, 1100, BodyName.TYLO, attitude=True),  # orbit insertion
]

_TYLO_LAND = _TYLO_ORBIT + [
    _E("tylo_low_orbit", "tylo_surface", VL, 2270, BodyName.TYLO,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_TYLO_RETURN = _TYLO_LAND + [
    _E("tylo_surface", "tylo_low_orbit", VA, 2270, BodyName.TYLO,
       min_twr=1.2, throttle=True, attitude=True),
    _E("tylo_low_orbit", "jool_low_orbit", PV, 1500, BodyName.TYLO, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.TYLO, MissionType.ESCAPE,        _TYLO_TRANSFER)
_add(BodyName.TYLO, MissionType.ORBIT,         _TYLO_ORBIT)
_add(BodyName.TYLO, MissionType.LAND,          _TYLO_LAND)
_add(BodyName.TYLO, MissionType.RETURN,        _TYLO_RETURN)
_add(BodyName.TYLO, MissionType.SAMPLE_RETURN, _TYLO_RETURN)


# ===========================================================================
# Bop  (Jool moon, airless, high inclination)
# ===========================================================================

_BOP_TRANSFER = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "bop_intercept", PV, 220, BodyName.JOOL,
       pc=2440, attitude=True),
    _E("bop_intercept", "bop_soi", PV, 0, BodyName.BOP, attitude=True),  # SOI entry
]

_BOP_ORBIT = _BOP_TRANSFER + [
    _E("bop_soi", "bop_low_orbit", PV, 900, BodyName.BOP, attitude=True),  # orbit insertion
]

_BOP_LAND = _BOP_ORBIT + [
    _E("bop_low_orbit", "bop_surface", VL, 230, BodyName.BOP,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_BOP_RETURN = _BOP_LAND + [
    _E("bop_surface", "bop_low_orbit", VA, 230, BodyName.BOP,
       min_twr=1.2, throttle=True, attitude=True),
    _E("bop_low_orbit", "jool_low_orbit", PV, 1120, BodyName.BOP,
       pc=2440, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.BOP, MissionType.ESCAPE,        _BOP_TRANSFER)
_add(BodyName.BOP, MissionType.ORBIT,         _BOP_ORBIT)
_add(BodyName.BOP, MissionType.LAND,          _BOP_LAND)
_add(BodyName.BOP, MissionType.RETURN,        _BOP_RETURN)
_add(BodyName.BOP, MissionType.SAMPLE_RETURN, _BOP_RETURN)


# ===========================================================================
# Pol  (Jool moon, airless, inclined)
# ===========================================================================

_POL_TRANSFER = _JOOL_ORBIT_AERO + [
    _E("jool_low_orbit", "pol_intercept", PV, 160, BodyName.JOOL,
       pc=700, attitude=True),
    _E("pol_intercept", "pol_soi", PV, 0, BodyName.POL, attitude=True),  # SOI entry
]

_POL_ORBIT = _POL_TRANSFER + [
    _E("pol_soi", "pol_low_orbit", PV, 820, BodyName.POL, attitude=True),  # orbit insertion
]

_POL_LAND = _POL_ORBIT + [
    _E("pol_low_orbit", "pol_surface", VL, 130, BodyName.POL,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_POL_RETURN = _POL_LAND + [
    _E("pol_surface", "pol_low_orbit", VA, 130, BodyName.POL,
       min_twr=1.2, throttle=True, attitude=True),
    _E("pol_low_orbit", "jool_low_orbit", PV, 980, BodyName.POL,
       pc=700, attitude=True),
    _E("jool_low_orbit", "kerbin_intercept", PV, 3790, BodyName.JOOL,
       pc=270, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.POL, MissionType.ESCAPE,        _POL_TRANSFER)
_add(BodyName.POL, MissionType.ORBIT,         _POL_ORBIT)
_add(BodyName.POL, MissionType.LAND,          _POL_LAND)
_add(BodyName.POL, MissionType.RETURN,        _POL_RETURN)
_add(BodyName.POL, MissionType.SAMPLE_RETURN, _POL_RETURN)


# ===========================================================================
# Eeloo  (distant, icy, no atmosphere)
# ===========================================================================

_EELOO_TRANSFER = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "eeloo_intercept", PT, 1140, BodyName.KERBIN,
       pc=1330, attitude=True),
    _E("eeloo_intercept", "eeloo_soi", PV, 0, BodyName.EELOO, attitude=True),  # SOI entry
]

_EELOO_ORBIT = _EELOO_TRANSFER + [
    _E("eeloo_soi", "eeloo_low_orbit", PV, 1370, BodyName.EELOO, attitude=True),  # orbit insertion
]

_EELOO_LAND = _EELOO_ORBIT + [
    _E("eeloo_low_orbit", "eeloo_surface", VL, 620, BodyName.EELOO,
       min_twr=1.2, throttle=True, attitude=True, legs=True),
]

_EELOO_RETURN = _EELOO_LAND + [
    _E("eeloo_surface", "eeloo_low_orbit", VA, 620, BodyName.EELOO,
       min_twr=1.2, throttle=True, attitude=True),
    _E("eeloo_low_orbit", "kerbin_intercept", PV, 2510, BodyName.EELOO,
       pc=1330, attitude=True),
    _KERBIN_REENTRY,
]

_add(BodyName.EELOO, MissionType.ESCAPE,        _EELOO_TRANSFER)
_add(BodyName.EELOO, MissionType.ORBIT,         _EELOO_ORBIT)
_add(BodyName.EELOO, MissionType.LAND,          _EELOO_LAND)
_add(BodyName.EELOO, MissionType.RETURN,        _EELOO_RETURN)
_add(BodyName.EELOO, MissionType.SAMPLE_RETURN, _EELOO_RETURN)


# ===========================================================================
# Kerbol  (orbit only — cannot land)
# ===========================================================================

_KERBOL_ORBIT = [
    _KERBIN_ASCENT,
    _KERBIN_ESCAPE,
    _E("kerbin_soi", "kerbol_low_orbit", PT, 6000, BodyName.KERBOL,
       attitude=True),
]

_KERBOL_RETURN = _KERBOL_ORBIT + [
    _E("kerbol_low_orbit", "kerbin_intercept", PV, 6000, BodyName.KERBOL, attitude=True),
    _KERBIN_REENTRY,
]

# Kerbol has no flyby / SOI-leave checks (you start inside its SOI), so
# no ESCAPE profile is registered. ORBIT/RETURN are the only Kerbol events.
_add(BodyName.KERBOL, MissionType.ORBIT,  _KERBOL_ORBIT)
_add(BodyName.KERBOL, MissionType.RETURN, _KERBOL_RETURN)

# ESCAPE/flyby profiles are now declared explicitly per body above.
# (Historically these were auto-generated by stripping the last edge from
#  the ORBIT profile, but that silently dropped destination relay-tier
#  checks when SOI entry and orbit insertion were merged into one edge.
#  See bug 075 / plans/ssr_variance_investigation.md for context.)
# Sanity check: every body that has an ORBIT profile and exposes
# flyby/SOI-leave checks also has an explicit ESCAPE profile. Kerbol is
# excluded — the player is already inside its SOI so no flyby is possible.
for _body_name, _mission_type in list(MISSION_PROFILES):
    if (_mission_type == MissionType.ORBIT
            and _body_name != BodyName.KERBOL
            and (_body_name, MissionType.ESCAPE) not in MISSION_PROFILES):
        raise RuntimeError(
            f"{_body_name} has an ORBIT profile but no ESCAPE profile. "
            "Add an explicit `_add(body, MissionType.ESCAPE, _<BODY>_TRANSFER)`."
        )
del _body_name, _mission_type

# Auto-generate flag-plant profiles from landing profiles.
# Flag plant is the same mission as a crewed landing — if no explicit
# FLAG_PLANT profile exists, copy the LAND profile.
# Kerbin already has an explicit empty FLAG_PLANT entry — skip it.
for _body_name, _mission_type in list(MISSION_PROFILES):
    if _mission_type == MissionType.LAND and (_body_name, MissionType.FLAG_PLANT) not in MISSION_PROFILES:
        MISSION_PROFILES[(_body_name, MissionType.FLAG_PLANT)] = MISSION_PROFILES[(_body_name, MissionType.LAND)]
del _body_name, _mission_type


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
