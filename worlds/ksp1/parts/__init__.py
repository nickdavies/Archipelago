"""Part data models + JSON-backed database for KSP1 Archipelago.

Public surface re-exported from the split submodules. The raw database is built
in the private ``_raw`` module; pack-aware access comes through ``PartManager``.
"""
from .types import (
    CapabilityFlag, Engine, FuelTank, SolidBooster, HeatShield, Parachute,
    LandingLeg, Decoupler, MiscEquipment, SolarSpec, AntennaSpec,
    CapsuleSpec, ProbeCoreSpec, AnyPart, MultiMount, MULTI_MOUNT_TABLE,
    MAX_RADIAL_ENGINES,
)
from .registry import PartMapping, PART_REGISTRY
from ._raw import usable_fuel_mass, UNDRAINABLE_PROPELLANTS
from .categories import (
    PartCategory, CONTRACT_PART_CATEGORIES, CONTRACT_CATEGORY_MEMBERS,
    PART_TO_CONTRACT_CATEGORIES,
)
from .manager import (
    PartManager, part_manager_for, DEFAULT_PART_MANAGER, ALL_PACKS,
)
from . import packs
