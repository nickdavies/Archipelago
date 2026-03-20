"""
Part data models and dummy part database for KSP1 Archipelago.

All dataclasses are frozen (hashable, immutable).  The PART_DB dict maps
item-name strings (matching ITEM_TABLE in items.py) to lists of part objects.
The pre-pass in capability.py iterates the player's collected items and looks
up their corresponding parts here.

Values are conservative estimates suitable for logic calculations, not exact
KSP figures.  They will be replaced with real PartDumper output once that
pipeline is complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Part dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Engine:
    name: str
    vac_isp: float          # seconds
    atm_isp: float          # seconds
    vac_thrust: float       # kN
    atm_thrust: float       # kN
    mass: float             # tonnes (dry engine mass only)
    throttleable: bool
    has_gimbal: bool
    size_class: float       # metres: 0.625, 1.25, 2.5, 3.75, 5.0
    fuel_type: str          # "lfo" | "lf" | "xenon"


@dataclass(frozen=True)
class FuelTank:
    name: str
    dry_mass: float         # tonnes
    fuel_mass: float        # tonnes at 100% fill
    fuel_type: str          # "lfo" | "lf" | "xenon" | "monoprop"
    size_class: float       # metres


@dataclass(frozen=True)
class SolidBooster:
    """SRB modeled as a combined engine+tank unit.  Always 100% fill."""
    name: str
    vac_isp: float
    atm_isp: float
    vac_thrust: float       # kN
    atm_thrust: float       # kN
    dry_mass: float         # tonnes (empty casing)
    fuel_mass: float        # tonnes (propellant)
    has_gimbal: bool        # always False for stock SRBs
    size_class: float


@dataclass(frozen=True)
class HeatShield:
    name: str
    mass: float             # tonnes
    size_class: float       # metres — covers parts up to this diameter


@dataclass(frozen=True)
class Parachute:
    name: str
    mass: float             # tonnes
    drag_area: float        # effective drag area (m²), from KSP fullyDeployedDrag
    is_drogue: bool         # drogue chutes are NEVER counted in logic


@dataclass(frozen=True)
class LandingLeg:
    name: str
    mass: float             # tonnes
    tier: int               # 1=small, 2=medium, 3=heavy
    count_per_set: int      # legs per item (typically 1; buy 4 for a set)


@dataclass(frozen=True)
class Decoupler:
    name: str
    mass: float
    kind: str               # "stack" | "radial"
    size_class: float


@dataclass(frozen=True)
class MiscEquipment:
    name: str
    mass: float
    provides: frozenset[str]    # capability flags — see below
    # Known flag strings:
    #   "probe_core"            unmanned command
    #   "capsule"               crewed command
    #   "reaction_wheel"        attitude control without gimbal/RCS
    #   "rcs"                   RCS thrusters (attitude + fine dv)
    #   "solar_fixed"           fixed panels, destroyed by aero
    #   "solar_retractable"     retractable panels, survive aero
    #   "solar_array_large"     large array, enables ION engine
    #   "rtg"                   power regardless of distance/orientation
    #   "relay_t1"              Communotron-class relay
    #   "relay_t2"              HG-5-class relay
    #   "relay_t3"              RA-2+-class relay
    #   "battery_small"         small EC storage
    #   "battery_large"         large EC storage (sustains ION burns)
    #   "docking_port"          enables docking-based staging (tier 3)
    #   "fuel_line"             enables asparagus staging
    #   "ladder"                crew ladder for EVA on low-gravity bodies
    #   "launch_clamp"          hold-down + fuelling (interplanetary gate)
    #   "isru"                  in-situ resource utilisation


# ---------------------------------------------------------------------------
# Engine cluster sizing table
# (tank_size_class, engine_size_class) -> max simultaneous engines
# Conservative estimates capped at 6.  Verified values to be updated
# after in-game testing.
# ---------------------------------------------------------------------------

ENGINE_COUNT_TABLE: dict[tuple[float, float], int] = {
    (0.625, 0.625): 1,
    (1.25,  0.625): 2,
    (1.25,  1.25):  1,
    (2.5,   0.625): 4,
    (2.5,   1.25):  4,
    (2.5,   2.5):   1,
    (3.75,  0.625): 6,
    (3.75,  1.25):  6,
    (3.75,  2.5):   3,
    (3.75,  3.75):  1,
    (5.0,   0.625): 6,
    (5.0,   1.25):  6,
    (5.0,   2.5):   4,
    (5.0,   3.75):  2,
    (5.0,   5.0):   1,
}


def max_engine_count(tank_size_class: float, engine_size_class: float) -> int:
    """Return the maximum number of engines of *engine_size_class* that fit
    on a stage whose main tank is *tank_size_class*.  Returns 0 if the engine
    is larger than the tank."""
    if engine_size_class > tank_size_class:
        return 0
    return ENGINE_COUNT_TABLE.get((tank_size_class, engine_size_class), 1)


# ---------------------------------------------------------------------------
# Type alias for any part
# ---------------------------------------------------------------------------

AnyPart = Union[Engine, FuelTank, SolidBooster, HeatShield, Parachute,
                LandingLeg, Decoupler, MiscEquipment]


# ---------------------------------------------------------------------------
# Dummy part database
# Keys are the canonical item names used in ITEM_TABLE (items.py).
# Values are lists because one item may unlock multiple variants or because
# the same item represents a "set" (e.g., four landing legs).
# ---------------------------------------------------------------------------

# --- Engines ---------------------------------------------------------------

_RELIANT = Engine(
    name="LV-T30 Reliant",
    vac_isp=310, atm_isp=265,
    vac_thrust=215, atm_thrust=205,
    mass=1.25, throttleable=True, has_gimbal=False,
    size_class=1.25, fuel_type="lfo",
)

_SWIVEL = Engine(
    name="LV-T45 Swivel",
    vac_isp=320, atm_isp=270,
    vac_thrust=200, atm_thrust=167,
    mass=1.5, throttleable=True, has_gimbal=True,
    size_class=1.25, fuel_type="lfo",
)

_TERRIER = Engine(
    name="LV-909 Terrier",
    vac_isp=345, atm_isp=85,
    vac_thrust=60, atm_thrust=14,
    mass=0.5, throttleable=True, has_gimbal=True,
    size_class=1.25, fuel_type="lfo",
)

_POODLE = Engine(
    name="RE-L10 Poodle",
    vac_isp=350, atm_isp=90,
    vac_thrust=250, atm_thrust=65,
    mass=1.75, throttleable=True, has_gimbal=True,
    size_class=2.5, fuel_type="lfo",
)

_MAINSAIL = Engine(
    name="RE-M3 Mainsail",
    vac_isp=310, atm_isp=285,
    vac_thrust=1500, atm_thrust=1379,
    mass=6.0, throttleable=True, has_gimbal=True,
    size_class=2.5, fuel_type="lfo",
)

_NERV = Engine(
    name="LV-N Nerv",
    vac_isp=800, atm_isp=185,
    vac_thrust=60, atm_thrust=14,
    mass=3.0, throttleable=True, has_gimbal=True,
    size_class=1.25, fuel_type="lf",
)

_DAWN = Engine(
    name="IX-6315 Dawn",
    vac_isp=4200, atm_isp=100,
    vac_thrust=2, atm_thrust=0.5,
    mass=0.25, throttleable=True, has_gimbal=False,
    size_class=0.625, fuel_type="xenon",
)

_RHINO = Engine(
    name="KS-25x4 Rhino",
    vac_isp=340, atm_isp=205,
    vac_thrust=2000, atm_thrust=1205,
    mass=9.0, throttleable=True, has_gimbal=True,
    size_class=3.75, fuel_type="lfo",
)

_MAMMOTH = Engine(
    name="S3 KS-25x4 Mammoth",
    vac_isp=315, atm_isp=295,
    vac_thrust=4000, atm_thrust=3746,
    mass=15.0, throttleable=True, has_gimbal=True,
    size_class=3.75, fuel_type="lfo",
)

# --- SRBs ------------------------------------------------------------------

_HAMMER = SolidBooster(
    name="Rockomax BACC Thumper",
    vac_isp=210, atm_isp=175,
    vac_thrust=300, atm_thrust=250,
    dry_mass=0.75, fuel_mass=6.5625,
    has_gimbal=False, size_class=1.25,
)

# --- Fuel Tanks ------------------------------------------------------------

_OSCAR_B = FuelTank(
    name="Oscar-B",
    dry_mass=0.025, fuel_mass=0.225,
    fuel_type="lfo", size_class=0.625,
)

_FL_T400 = FuelTank(
    name="FL-T400",
    dry_mass=0.25, fuel_mass=2.25,
    fuel_type="lfo", size_class=1.25,
)

_FL_T800 = FuelTank(
    name="FL-T800",
    dry_mass=0.5, fuel_mass=4.5,
    fuel_type="lfo", size_class=1.25,
)

_X200_32 = FuelTank(
    name="Rockomax X200-32",
    dry_mass=2.0, fuel_mass=18.0,
    fuel_type="lfo", size_class=2.5,
)

_JUMBO_64 = FuelTank(
    name="Rockomax Jumbo-64",
    dry_mass=4.0, fuel_mass=36.0,
    fuel_type="lfo", size_class=2.5,
)

_MK1_LF = FuelTank(
    name="Mk1 LF Tank",
    dry_mass=0.25, fuel_mass=2.25,
    fuel_type="lf", size_class=1.25,
)

_S3_3600 = FuelTank(
    name="Kerbodyne S3-3600",
    dry_mass=2.25, fuel_mass=36.0,
    fuel_type="lfo", size_class=3.75,
)

_PB_X50R = FuelTank(
    name="PB-X50R Xenon",
    dry_mass=0.13, fuel_mass=0.72,
    fuel_type="xenon", size_class=0.625,
)

_FL_R10 = FuelTank(
    name="FL-R10 Monoprop",
    dry_mass=0.025, fuel_mass=0.2,
    fuel_type="monoprop", size_class=0.625,
)

# --- Heat Shields ----------------------------------------------------------

_SHIELD_125 = HeatShield(name="1.25m Heat Shield", mass=0.15, size_class=1.25)
_SHIELD_25 = HeatShield(name="2.5m Heat Shield",  mass=0.4,  size_class=2.5)
_SHIELD_375 = HeatShield(name="3.75m Heat Shield", mass=0.9,  size_class=3.75)

# --- Parachutes ------------------------------------------------------------

_MK16 = Parachute(
    name="Mk16 Parachute",
    mass=0.1,
    drag_area=400.0,    # m², conservative estimate from KSP fullyDeployedDrag
    is_drogue=False,
)

_MK2R_DROGUE = Parachute(
    name="Mk2-R Drogue",
    mass=0.05,
    drag_area=20.0,
    is_drogue=True,     # excluded from all logic
)

# --- Landing Legs ----------------------------------------------------------

_LT1 = LandingLeg(name="LT-1 Landing Struts",  mass=0.05, tier=1, count_per_set=1)
_LT2 = LandingLeg(name="LT-2 Landing Strut",   mass=0.1,  tier=2, count_per_set=1)

# --- Decouplers ------------------------------------------------------------

_TR18A = Decoupler(name="TR-18A Stack Decoupler", mass=0.05,  kind="stack",  size_class=1.25)
_TT38K = Decoupler(name="TT-38K Radial Decoupler", mass=0.025, kind="radial", size_class=1.25)

# --- Misc Equipment --------------------------------------------------------

_PROBE_CORE = MiscEquipment(
    name="HECS Probe Core",
    mass=0.1,
    provides=frozenset({"probe_core", "reaction_wheel"}),
)

_OKTO2 = MiscEquipment(
    name="Probodobodyne OKTO2",
    mass=0.04,
    provides=frozenset({"probe_core"}),
)

_COMMAND_POD = MiscEquipment(
    name="Mk1 Command Pod",
    mass=0.84,
    provides=frozenset({"capsule", "reaction_wheel"}),
)

_REACTION_WHEEL = MiscEquipment(
    name="Advanced Inline Stabilizer",
    mass=0.1,
    provides=frozenset({"reaction_wheel"}),
)

_OX_STAT = MiscEquipment(
    name="OX-STAT Solar",
    mass=0.005,
    provides=frozenset({"solar_fixed"}),
)

_OX_4 = MiscEquipment(
    name="OX-4 Retractable Solar",
    mass=0.0175,
    provides=frozenset({"solar_retractable"}),
)

_SOLAR_ARRAY = MiscEquipment(
    name="Gigantor XL Solar Array",
    mass=0.3,
    provides=frozenset({"solar_retractable", "solar_array_large"}),
)

_RTG = MiscEquipment(
    name="PB-NUK RTG",
    mass=0.08,
    provides=frozenset({"rtg"}),
)

_COMM16 = MiscEquipment(
    name="Communotron 16",
    mass=0.005,
    provides=frozenset({"relay_t1"}),
)

_HG5 = MiscEquipment(
    name="HG-5 High Gain",
    mass=0.07,
    provides=frozenset({"relay_t2"}),
)

_RA2 = MiscEquipment(
    name="RA-2 Relay Antenna",
    mass=0.15,
    provides=frozenset({"relay_t3"}),
)

_RCS = MiscEquipment(
    name="RV-105 RCS Thruster Block",
    mass=0.05,
    provides=frozenset({"rcs"}),
)

_FUEL_LINE = MiscEquipment(
    name="FTX-2 External Fuel Duct",
    mass=0.05,
    provides=frozenset({"fuel_line"}),
)

_LADDER = MiscEquipment(
    name="Pegasus I Mobility Enhancer",
    mass=0.005,
    provides=frozenset({"ladder"}),
)

_LAUNCH_CLAMP = MiscEquipment(
    name="TT18-A Launch Stability Enhancer",
    mass=0.1,
    provides=frozenset({"launch_clamp"}),
)

_BATTERY_LARGE = MiscEquipment(
    name="Z-4K Rechargeable Battery",
    mass=0.1,
    provides=frozenset({"battery_large", "battery_small"}),
)

_DOCKING_PORT = MiscEquipment(
    name="Clamp-O-Tron Docking Port",
    mass=0.05,
    provides=frozenset({"docking_port"}),
)

# Science instruments — no capability provides; checked separately by the
# science heuristic in rules.py via state.has("Thermometer", player).
_THERMOMETER = MiscEquipment(
    name="2HOT Thermometer",
    mass=0.005,
    provides=frozenset(),
)

_BAROMETER = MiscEquipment(
    name="PresMat Barometer",
    mass=0.005,
    provides=frozenset(),
)

# Structural — no capability provides; precollected for all seeds.
_STRUT = MiscEquipment(
    name="EAS-4 Strut Connector",
    mass=0.05,
    provides=frozenset(),
)


# ---------------------------------------------------------------------------
# PART_DB: canonical item name -> list of part objects
# ---------------------------------------------------------------------------
# Item names here are the authoritative strings.  items.py ITEM_TABLE must
# use the same strings.  Each value is a list so that one "item unlock" can
# grant multiple part variants (e.g., both sizes of a part family).

PART_DB: dict[str, list[AnyPart]] = {
    # Engines
    "Reliant Engine":   [_RELIANT],
    "Swivel Engine":    [_SWIVEL],
    "Terrier Engine":   [_TERRIER],
    "Poodle Engine":    [_POODLE],
    "Mainsail Engine":  [_MAINSAIL],
    "Nerv Engine":      [_NERV],
    "Dawn Ion Engine":  [_DAWN],
    "Rhino Engine":     [_RHINO],
    "Mammoth Engine":   [_MAMMOTH],
    "Hammer SRB":       [_HAMMER],

    # Fuel tanks
    "Oscar-B Tank":         [_OSCAR_B],
    "FL-T400 Tank":         [_FL_T400],
    "FL-T800 Tank":         [_FL_T800],
    "Rockomax X200-32":     [_X200_32],
    "Rockomax Jumbo-64":    [_JUMBO_64],
    "Mk1 LF Tank":          [_MK1_LF],
    "Kerbodyne S3-3600":    [_S3_3600],
    "PB-X50R Xenon Tank":   [_PB_X50R],
    "FL-R10 Monoprop Tank": [_FL_R10],

    # Heat shields
    "1.25m Heat Shield":  [_SHIELD_125],
    "2.5m Heat Shield":   [_SHIELD_25],
    "3.75m Heat Shield":  [_SHIELD_375],

    # Parachutes
    "Mk16 Parachute":  [_MK16],
    "Mk2-R Drogue":    [_MK2R_DROGUE],

    # Landing legs
    "LT-1 Landing Legs":  [_LT1],
    "LT-2 Landing Strut": [_LT2],

    # Decouplers
    "TR-18A Decoupler":      [_TR18A],
    "TT-38K Radial Decoupler": [_TT38K],

    # Misc equipment
    "Probe Core":            [_PROBE_CORE],
    "Probodobodyne OKTO2":   [_OKTO2],
    "Command Pod":           [_COMMAND_POD],
    "Reaction Wheel":        [_REACTION_WHEEL],
    "OX-STAT Solar":         [_OX_STAT],
    "OX-4 Solar":            [_OX_4],
    "Gigantor Solar Array":  [_SOLAR_ARRAY],
    "RTG":                   [_RTG],
    "Communotron 16":        [_COMM16],
    "HG-5 Relay":            [_HG5],
    "RA-2 Relay":            [_RA2],
    "RCS Thruster":          [_RCS],
    "FTX-2 Fuel Line":       [_FUEL_LINE],
    "Crew Ladder":           [_LADDER],
    "Launch Clamp":          [_LAUNCH_CLAMP],
    "Z-4K Battery":          [_BATTERY_LARGE],
    "Docking Port":          [_DOCKING_PORT],

    # Science instruments (gate tech-tree science heuristic)
    "Thermometer":           [_THERMOMETER],
    "Barometer":             [_BAROMETER],

    # Structural (precollected for all seeds)
    "Struts":                [_STRUT],
}
