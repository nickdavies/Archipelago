"""Universal Tracker map tab: where KSP1 locations and entrances sit on the
Kerbonaut map pack (github.com/1Kerbonaut/KSP-Style-Flags, ``UT Maps/``).

The pack's marker files define the nodes that exist: on every map one per
body, an entry/exit pair per body, KSC, the four first-milestone markers,
Altitude and one Science marker per tech tier; on the ocean-home maps also
Splashdown.  This module maps the world's typed locations and its UT-only
entrances onto those node labels; ``scripts/generate_ut_map.py`` combines that
with the marker coordinates into ``tracker/locations.json``.  Starting
inventory and contract thresholds have no node and are deliberately off-map —
the pack defines what can be mapped.

Archipelago-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from .bodies import BodyName, MissionType
from .locations.catalog import LANDABLE_HOMES
from .locations.defs import (
    ContractLocation, HomeLocation, HomeMilestone, KSP1LocationDef,
    KscBiomeLocation, MissionLocation, SplashdownLocation,
    StartingInventoryLocation, TechTreeLocation, ThresholdLocation,
)
from .locations.events import EventName


@dataclass(frozen=True)
class UtEntrance:
    """A UT-only, dead-end entrance per body, shown on the body's entry or
    exit node and labelled with the mission's raw profile dV.  It mirrors the
    logic of the ``event`` mission location, so it is only created where that
    location is emitted."""
    mission: MissionType
    event: EventName
    on_exit_node: bool

    def name(self, body: BodyName) -> str:
        return f"{body} - {self.mission}"

    def node(self, body: BodyName) -> str:
        return f"{body}_{'exit' if self.on_exit_node else 'entry'}"


UT_ENTRANCES: tuple[UtEntrance, ...] = (
    UtEntrance(MissionType.ORBIT, EventName.ORBIT, on_exit_node=False),
    UtEntrance(MissionType.LAND, EventName.LANDING, on_exit_node=False),
    UtEntrance(MissionType.RETURN, EventName.RETURN, on_exit_node=True),
    UtEntrance(MissionType.ESCAPE, EventName.SOI_LEAVE, on_exit_node=True),
    UtEntrance(MissionType.ORBIT_RETURN, EventName.ORBIT_RETURN, on_exit_node=True),
    UtEntrance(MissionType.SOI_RETURN, EventName.SOI_RETURN, on_exit_node=True),
)

#: Map tab order (``maps.json``): one map per possible home, alphabetical.
UT_MAP_HOMES: tuple[BodyName, ...] = tuple(sorted(LANDABLE_HOMES, key=str))

# The pack's sidebar marker labels for the single-instance home milestones.
_MILESTONE_NODE: dict[HomeMilestone, str] = {
    HomeMilestone.FIRST_LAUNCH: "First_Launch",
    HomeMilestone.FIRST_LANDING: "First_Landing",
    HomeMilestone.FIRST_CRASH: "First_Crash",
    HomeMilestone.FIRST_STAGING: "First_Staging",
}


def location_map(defn: KSP1LocationDef) -> BodyName | None:
    """The one map a location is drawn on, or ``None`` for every map that has
    its node.  Home specials belong to their home's map: UT filters sections by
    the connected slot, not by map, so listing them on every map would draw
    the slot's home specials on every tab."""
    return defn.home if isinstance(defn, HomeLocation) else None


def location_node(defn: KSP1LocationDef) -> str | None:
    """The map node label a location is listed under, or ``None`` if it is
    off-map."""
    match defn:
        case MissionLocation() | ContractLocation():
            return str(defn.body)
        case HomeLocation(milestone=HomeMilestone.ALTITUDE):
            return "Altitude"
        case HomeLocation():
            return _MILESTONE_NODE[defn.milestone]
        case TechTreeLocation():
            return f"Science_{defn.node.tier - 1}"
        case KscBiomeLocation():
            return "KSC"
        case SplashdownLocation():
            return "Splashdown"   # marker only on the ocean-home maps
        case StartingInventoryLocation() | ThresholdLocation():
            return None
        case _:
            assert_never(defn)
