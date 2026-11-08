"""AP item name groups: static, named subsets of ``ITEM_NAME_TO_ID``.

Groups ship in the per-game data package, so they cover the full item universe
(every pack, every contract); hints and ``/item_groups`` only ever resolve the
items a seed actually placed.  Part groups are built from the typed part DB:
the part dataclass for the structural roles, ``CapabilityFlag`` for
``MiscEquipment``.  The ~150 MiscEquipment parts with no capability flag
(wings, intakes, cargo bays, structural pieces, lights) are only in "Parts".
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType

from .contracts import all_possible_contract_specs
from .items import (
    BUFF_ITEM_NAMES, CONSUMABLE_ITEM_NAMES, DISCOVER_ITEM_BY_BODY, ITEM_TABLE,
    PROGRESSIVE_ASTRONAUT_COMPLEX_NAME, PROGRESSIVE_LAUNCH_PAD_NAME,
    PROGRESSIVE_MISSION_CONTROL_NAME, PROGRESSIVE_RD_NAME,
    PROGRESSIVE_SCIENCE_INSTRUMENT_NAME, PROGRESSIVE_TRACKING_STATION_NAME,
    PROGRESSIVE_VAB_NAME,
    SCIENCE_PACK_NAMES, TRAP_ITEM_NAMES,
)
from .parts import (
    CONTRACT_CATEGORY_MEMBERS, DEFAULT_PART_MANAGER, CapabilityFlag, Decoupler,
    Engine, FuelTank, HeatShield, LandingLeg, Parachute, SolidBooster,
    part_manager_for, packs,
)

_PART_TYPE_GROUPS: dict[type, str] = {
    Engine: "Engines",
    SolidBooster: "Solid Boosters",
    FuelTank: "Fuel Tanks",
    HeatShield: "Heat Shields",
    Parachute: "Parachutes",
    LandingLeg: "Landing Legs",
    Decoupler: "Decouplers",
}

_FLAG_GROUPS: dict[CapabilityFlag, str] = {
    CapabilityFlag.CAPSULE: "Command Pods",
    CapabilityFlag.PROBE_CORE: "Probe Cores",
    CapabilityFlag.REACTION_WHEEL: "Reaction Wheels",
    CapabilityFlag.RCS: "RCS",
    CapabilityFlag.SOLAR_FIXED: "Power",
    CapabilityFlag.SOLAR_RETRACTABLE: "Power",
    CapabilityFlag.SOLAR_ARRAY_LARGE: "Power",
    CapabilityFlag.RTG: "Power",
    CapabilityFlag.BATTERY_SMALL: "Batteries",
    CapabilityFlag.BATTERY_LARGE: "Batteries",
    # Every standalone antenna carries a comms tier flag, DIRECT and RELAY
    # alike (pods' built-in antennas carry none).
    CapabilityFlag.RELAY_T1: "Antennas",
    CapabilityFlag.RELAY_T2: "Antennas",
    CapabilityFlag.RELAY_T3: "Antennas",
    CapabilityFlag.RELAY_T4: "Antennas",
    CapabilityFlag.DOCKING_PORT: "Docking Ports",
    CapabilityFlag.WHEEL: "Wheels",
    CapabilityFlag.AERO_CONTROL: "Control Surfaces",
    CapabilityFlag.MULTI_MOUNT: "Engine Mounts",
    CapabilityFlag.SCIENCE_INSTRUMENT: "Science Instruments",
    CapabilityFlag.THERMOMETER: "Science Instruments",
    CapabilityFlag.BAROMETER: "Science Instruments",
    CapabilityFlag.ISRU: "Mining",
}

#: Drills and ore tanks carry no capability flag; the contract categories
#: already curate them.
_MINING_CATEGORIES: tuple[str, ...] = ("drill", "ore_tank")


def _build() -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = defaultdict(set)

    groups["Parts"] = set(ITEM_TABLE)
    for name, parts in DEFAULT_PART_MANAGER.parts.items():
        for part in parts:
            if type(part) in _PART_TYPE_GROUPS:
                groups[_PART_TYPE_GROUPS[type(part)]].add(name)
            for flag in getattr(part, "provides", ()):
                if flag in _FLAG_GROUPS:
                    groups[_FLAG_GROUPS[flag]].add(name)
    for category in _MINING_CATEGORIES:
        groups["Mining"] |= CONTRACT_CATEGORY_MEMBERS[category]
    groups["Science Instruments"].add(PROGRESSIVE_SCIENCE_INSTRUMENT_NAME)

    stock_only = set(part_manager_for(frozenset()).parts)
    for pack in packs.OPTIONAL_PACKS:
        groups[f"{pack} Parts"] = set(part_manager_for(frozenset({pack})).parts) - stock_only

    groups["Contracts"] = {spec.item_name for spec in all_possible_contract_specs()}
    groups["Discover Bodies"] = set(DISCOVER_ITEM_BY_BODY.values())
    groups["Buildings"] = {
        PROGRESSIVE_LAUNCH_PAD_NAME, PROGRESSIVE_RD_NAME, PROGRESSIVE_VAB_NAME,
        PROGRESSIVE_TRACKING_STATION_NAME, PROGRESSIVE_ASTRONAUT_COMPLEX_NAME,
        PROGRESSIVE_MISSION_CONTROL_NAME,
    }
    groups["Science Packs"] = set(SCIENCE_PACK_NAMES)
    groups["Traps"] = set(TRAP_ITEM_NAMES)
    groups["Buffs"] = set(BUFF_ITEM_NAMES | CONSUMABLE_ITEM_NAMES)
    return {name: frozenset(members) for name, members in groups.items()}


#: Group name → item names.  ``KSP1World.item_name_groups``.
ITEM_NAME_GROUPS: Mapping[str, frozenset[str]] = MappingProxyType(_build())
