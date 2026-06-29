"""Frozen part dataclasses, capability flags, and multi-mount adapter tables.

Pure data models with no I/O. The JSON-backed database that instantiates these
lives in the private ``_raw`` module and is exposed via ``PartManager``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union

from ..part_geometry import PartRole


# ---------------------------------------------------------------------------
# Capability flags — single source of truth for MiscEquipment.provides values
# ---------------------------------------------------------------------------

class CapabilityFlag(StrEnum):
    PROBE_CORE = "probe_core"
    CAPSULE = "capsule"
    REACTION_WHEEL = "reaction_wheel"
    RCS = "rcs"
    SOLAR_FIXED = "solar_fixed"
    SOLAR_RETRACTABLE = "solar_retractable"
    SOLAR_ARRAY_LARGE = "solar_array_large"
    RTG = "rtg"
    BATTERY_SMALL = "battery_small"
    BATTERY_LARGE = "battery_large"
    DOCKING_PORT = "docking_port"
    FUEL_LINE = "fuel_line"
    LADDER = "ladder"
    LAUNCH_CLAMP = "launch_clamp"
    ISRU = "isru"
    MULTI_MOUNT = "multi_mount"
    THERMOMETER = "thermometer"
    BAROMETER = "barometer"
    WHEEL = "wheel"
    AERO_CONTROL = "aero_control"
    SCIENCE_INSTRUMENT = "science_instrument"
    RELAY_T1 = "relay_t1"
    RELAY_T2 = "relay_t2"
    RELAY_T3 = "relay_t3"
    RELAY_T4 = "relay_t4"


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
    fuel_type: str          # "lfo" | "lf" | "xenon" (derived tag)
    radial_mountable: bool = False  # True only for srf-ONLY (purpose-built radial) engines
    # Stock-resource names the engine consumes (excluding ElectricCharge),
    # sorted. Used for generic tank/engine compatibility.
    propellants: tuple[str, ...] = ()


@dataclass(frozen=True)
class FuelTank:
    name: str
    dry_mass: float         # tonnes
    fuel_mass: float        # tonnes at 100% fill (sum of all carried propellants)
    fuel_type: str          # "lfo" | "lf" | "xenon" | "monoprop" (derived tag)
    size_class: float       # metres
    max_count: int = 0      # 0 = unlimited; >0 caps optimizer tank count (adapters)
    # Structural roles derived from attach-node geometry (see part_geometry).
    # SPINE = can be a stage's central stackable column; RADIAL_MOUNT = side/drop
    # booster only.  Replaces the old bulkhead-only is_radial, which mis-modelled
    # single-node/slanted/coupler tanks (FL-C1000, slant adapters) as spines.
    roles: frozenset[PartRole] = field(default_factory=frozenset)
    # Per-propellant mass at 100% fill (tonnes), e.g. {"LiquidFuel": 0.5,
    # "Oxidizer": 0.5} for an LFO tank. Lets an engine that needs only a
    # subset of the carried propellants drain the rest — except MonoPropellant,
    # which cannot be drained (dedicated resource).
    fuel_masses: tuple[tuple[str, float], ...] = ()


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
    radial_mountable: bool = False  # True if SRB has "srf" in bulkhead_profiles


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
    is_radial: bool = False # radial chutes mount on sides (unlimited); inline cap at 1


@dataclass(frozen=True)
class LandingLeg:
    name: str
    mass: float             # tonnes
    tier: int               # 1=small, 2=medium, 3=heavy


@dataclass(frozen=True)
class Decoupler:
    name: str
    mass: float
    kind: str               # "stack" | "radial"
    size_class: float


# Typed sub-specs for MiscEquipment parts that participate in rank axes.
# These default to None on MiscEquipment; the loader populates them from
# parts.json fields produced by scripts/extract_parts.py.

@dataclass(frozen=True)
class SolarSpec:
    """Solar panel data sourced from ModuleDeployableSolarPanel."""
    charge_rate: float       # EC/sec at 1 AU sun distance
    tracking: bool           # True = deployable/sun-tracking; False = fixed (OX-STAT)


@dataclass(frozen=True)
class AntennaSpec:
    """Antenna data sourced from ModuleDataTransmitter.

    Only ``DIRECT`` and ``RELAY`` antennas count for the RELAY rank axis;
    ``INTERNAL`` (the 5kW transmitter built into pods/probes) is excluded.
    """
    power: float             # raw antennaPower (large numbers — log/quantize at scoring time)
    combinable: bool
    antenna_type: str        # "DIRECT" | "RELAY" | "INTERNAL"


@dataclass(frozen=True)
class CapsuleSpec:
    """Crew-pod data sourced from top-level ``CrewCapacity`` + drainable
    resource mass.  ``effective_dry_mass = part.mass - drainable_propellant``
    is the rank ordering quantity; capsule pods carry MonoPropellant /
    LiquidFuel / Oxidizer that can be drained pre-launch."""
    crew_capacity: int
    drainable_mass: float    # tonnes of removable propellant at 100% fill


@dataclass(frozen=True)
class ProbeCoreSpec:
    """Probe-core SAS service level from ``ModuleSAS.SASServiceLevel`` (0..3).

    Pods carry ModuleSAS too but their primary rank axis is CAPSULE; this
    spec is only attached when ``provides`` contains :py:attr:`CapabilityFlag.PROBE_CORE`.
    """
    sas_level: int


@dataclass(frozen=True)
class MiscEquipment:
    name: str
    mass: float
    provides: frozenset[CapabilityFlag]  # see CapabilityFlag enum for valid values
    crew_capacity: int = 0               # seats; >0 for pods/cabins/lab
    # Diameter (metres), from bulkhead_profiles.  Used to size the reentry heat
    # shield to the capsule it protects — a 1.25m pod needs a 1.25m shield, not
    # the lightest available.  0.0 when the part has no meaningful diameter.
    size_class: float = 0.0
    solar: Optional[SolarSpec] = None
    antenna: Optional[AntennaSpec] = None
    capsule: Optional[CapsuleSpec] = None
    probe_core: Optional[ProbeCoreSpec] = None


# ---------------------------------------------------------------------------
# Multi-mount adapter/coupler/plate table
# ---------------------------------------------------------------------------

# Cap for radial-mount engines/SRBs and radial-tank-mounting (KSP symmetry
# modes: 1x, 2x, 3x, 4x, 6x, 8x).
MAX_RADIAL_ENGINES: int = 8


@dataclass(frozen=True)
class MultiMount:
    """An adapter, coupler, or engine plate that enables multi-engine stages.

    min_tank_size: minimum tank size_class above the adapter (0.0 = no limit,
                   e.g. engine plates attach anywhere).
    engine_counts: mapping of engine_size_class → max engines mountable.
                   Lookup: find the smallest key >= the engine's size_class.
    _sorted_sizes: pre-sorted engine_counts keys for fast lookup (auto-set).
    """
    min_tank_size: float
    engine_counts: dict[float, int]
    _sorted_sizes: tuple[float, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_sorted_sizes',
                           tuple(sorted(self.engine_counts)))


MULTI_MOUNT_TABLE: dict[str, MultiMount] = {
    # Stock couplers (output 1.25m nodes)
    "stackBiCoupler.v2":     MultiMount(1.25, {1.25: 2}),
    "stackTriCoupler.v2":    MultiMount(1.25, {1.25: 3}),
    "stackQuadCoupler":      MultiMount(1.25, {1.25: 4}),
    # Stock adapters (2.5m input → 1.25m output)
    "mk2.1m.Bicoupler":     MultiMount(2.5,  {1.25: 2}),
    "adapterLargeSmallBi":   MultiMount(2.5,  {1.25: 2}),
    "adapterLargeSmallTri":  MultiMount(2.5,  {1.25: 3}),
    "adapterLargeSmallQuad": MultiMount(2.5,  {1.25: 4}),
    # Engine plates (Making History, no min_tank_size restriction)
    "EnginePlate5":   MultiMount(0.0, {0.625: 3}),                                       # EP-12
    "EnginePlate1p5": MultiMount(0.0, {0.625: 7}),                                       # EP-18
    "EnginePlate2":   MultiMount(0.0, {0.625: 9, 1.25: 3}),                              # EP-25
    "EnginePlate3":   MultiMount(0.0, {0.625: 9, 1.25: 7, 1.875: 3}),                    # EP-37
    "EnginePlate4":   MultiMount(0.0, {0.625: 9, 1.25: 9, 1.875: 7, 2.5: 3}),            # EP-50
    # Dual-purpose adapter (Making History) — also a fuel tank
    "Size4.EngineAdapter.01": MultiMount(3.75, {2.5: 5, 3.75: 1}),
}


# ---------------------------------------------------------------------------
# Type alias for any part
# ---------------------------------------------------------------------------

AnyPart = Union[Engine, FuelTank, SolidBooster, HeatShield, Parachute,
                LandingLeg, Decoupler, MiscEquipment]
