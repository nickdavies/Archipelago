"""AP location name groups: static, named subsets of ``LocationCatalog``.

AP ships groups in the per-game data package, so they are fixed per apworld
version and cannot depend on a seed.  That is fine in practice: every consumer
(``!missing`` / ``!checked`` filters, ``!hint_location``) intersects a group
with the slot's own locations, so e.g. "Home Milestones" (every possible
home's specials) acts as "my home's milestones" in a live game.

Group names must not equal a string players already filter by.  ``!missing X``
switches from a substring match to group membership whenever ``X`` is a group
name, so a group called "Landing" would hide the "Landing N" tech-node slots
from ``!missing Landing``.  Event groups are plural for that reason, and
``test_name_groups`` enforces it.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType
from typing import assert_never

from .catalog import LocationCatalog
from .defs import (
    ContractLocation, HomeLocation, KSP1LocationDef, KscBiomeLocation,
    MissionLocation, SplashdownLocation, StartingInventoryLocation,
    TechTreeLocation, ThresholdLocation,
)
from .events import EventName

MISSIONS = "Missions"
CONTRACTS = "Contracts"
HOME_MILESTONES = "Home Milestones"
TECH_TREE = "Tech Tree"
KSC_SCIENCE = "KSC Science"
STARTING_INVENTORY = "Starting Inventory"
CONTRACT_THRESHOLDS = "Contract Thresholds"

#: Per-event group name (plural, so it never equals the event's own name).
EVENT_GROUP_NAMES: Mapping[EventName, str] = MappingProxyType({
    EventName.FLYBY: "Flybys",
    EventName.SOI_LEAVE: "SOI Leaves",
    EventName.UNMANNED_FLYBY: "Unmanned Flybys",
    EventName.SOI_RETURN: "SOI Returns",
    EventName.ORBIT: "Orbits",
    EventName.EVA_IN_ORBIT: "EVAs in Orbit",
    EventName.ORBITAL_PROBE: "Orbital Probes",
    EventName.ORBIT_RETURN: "Orbit Returns",
    EventName.LANDING: "Landings",
    EventName.CREWED_LANDING: "Crewed Landings",
    EventName.FLAG_PLANT: "Flag Plants",
    EventName.RETURN: "Returns",
    EventName.SAMPLE_RETURN: "Sample Returns",
})
assert set(EVENT_GROUP_NAMES) == set(EventName), "every EventName needs a group name"


def groups_of(defn: KSP1LocationDef) -> tuple[str, ...]:
    """The group names ``defn`` belongs to.  Body-scoped locations (missions,
    home specials, contracts) also join their body's group, named after the
    body."""
    match defn:
        case MissionLocation():
            return str(defn.body), MISSIONS, EVENT_GROUP_NAMES[defn.event]
        case HomeLocation():
            return str(defn.body), HOME_MILESTONES
        case ContractLocation():
            return str(defn.body), CONTRACTS
        case TechTreeLocation():
            return (TECH_TREE,)
        case KscBiomeLocation():
            return (KSC_SCIENCE,)
        case StartingInventoryLocation():
            return (STARTING_INVENTORY,)
        case ThresholdLocation():
            return (CONTRACT_THRESHOLDS,)
        case SplashdownLocation():
            # A single body-agnostic check, and "Splashdown" is already its
            # location name, so it can't also be a group name.
            return ()
        case _:
            assert_never(defn)


def _build() -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for defn in LocationCatalog.all():
        for group in groups_of(defn):
            groups[group].add(str(defn))
    return {name: frozenset(members) for name, members in groups.items()}


#: Group name → location names.  ``KSP1World.location_name_groups``.
LOCATION_NAME_GROUPS: Mapping[str, frozenset[str]] = MappingProxyType(_build())
