"""The physics meaning of a capability-gated location.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from ..bodies import BodyName, MissionType
from .defs import (
    ContractLocation, HomeLocation, KSP1LocationDef, KscBiomeLocation,
    MissionLocation, SplashdownLocation, StartingInventoryLocation,
    TechTreeLocation, ThresholdLocation,
)
from .events import EventName

if TYPE_CHECKING:
    from ..contracts import ContractSpec


@dataclass(frozen=True)
class LocationDescriptor:
    """The physics meaning of a capability-gated location, carried as a real
    object on its ``KSP1Location`` (``loc.descriptor``).

    Derived from the location's typed def (``descriptor_for``) so the sphere
    ladder never recovers structure by decoding the display name.
    Slot-agnostic: the event slots that share a mission (e.g. Mun Landing 1/2/3)
    map to one descriptor.
    """
    body: BodyName
    mission_type: MissionType
    crewed: bool | None
    threshold_km: float | None = None
    # Contract-completion locations carry their ContractSpec; the delivery
    # payload is sized per-rung from it (see contract_payload_parts).  ``None``
    # for ordinary (non-contract) missions.
    spec: ContractSpec | None = None
    # Whether the mission requires a Kerbal EVA — drives the curated
    # Astronaut-Complex ``can_eva`` gate (buildings_in_logic).  ``None`` means
    # "let the evaluator derive it from mission_type".
    requires_eva: bool | None = None
    # The specific event (ORBIT and EVA_IN_ORBIT stay distinct even though they
    # share ``mission_type=ORBIT``).  ``None`` for home specials and contracts —
    # used by the mission-only graph-walk to skip non-mission descriptors.
    event: EventName | None = None

    @classmethod
    def from_mission(cls, ml: MissionLocation) -> LocationDescriptor:
        event_def = ml.event_def
        return cls(
            body=ml.body,
            mission_type=event_def.mission_type,
            crewed=event_def.crewed,
            threshold_km=None,
            requires_eva=event_def.requires_eva,
            event=ml.event,
        )

    @classmethod
    def from_home(cls, hloc: HomeLocation) -> LocationDescriptor:
        return cls(
            body=hloc.home,
            mission_type=hloc.mission_type,
            crewed=None,
            threshold_km=hloc.threshold_km,
        )

    @classmethod
    def from_spec(cls, spec: ContractSpec) -> LocationDescriptor:
        # Physics-gated like a mission of the contract's base type, but with the
        # required equipment as delivered payload (sized per-rung from the spec).
        td = spec.type_def
        return cls(
            body=spec.body,
            mission_type=td.base_mission_type,
            crewed=td.crewed,
            threshold_km=None,
            spec=spec,
        )


def descriptor_for(defn: KSP1LocationDef) -> LocationDescriptor | None:
    """The physics descriptor for a location, or ``None`` when its access isn't
    capability-gated by a single body's mission.

    Splashdown has no single body to drive the bumper's mission-centric work;
    its requirements are dominated by the per-body LAND missions the bumper
    already handles.
    """
    match defn:
        case MissionLocation():
            return LocationDescriptor.from_mission(defn)
        case HomeLocation():
            return LocationDescriptor.from_home(defn)
        case ContractLocation():
            return LocationDescriptor.from_spec(defn.spec)
        case (StartingInventoryLocation() | KscBiomeLocation() | SplashdownLocation()
              | TechTreeLocation() | ThresholdLocation()):
            return None
        case _:
            assert_never(defn)
