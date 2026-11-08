"""Per-body mission events: the event vocabulary and its metadata table.

``EventDef`` rows are the single source of truth for every per-body mission
event — location slot counts, the mission profile each event maps to, crew /
landing / EVA requirements.  Locations, rules, capability and the client
(via slot_data ``event_scales``) all derive from this table.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..bodies import BodyName, MissionType


class EventName(StrEnum):
    """Canonical event names — use these instead of string literals."""
    FLYBY = "Flyby"
    SOI_LEAVE = "SOI Leave"
    UNMANNED_FLYBY = "Unmanned Flyby"
    SOI_RETURN = "SOI Return"
    ORBIT = "Orbit"
    EVA_IN_ORBIT = "EVA in Orbit"
    ORBITAL_PROBE = "Orbital Probe"
    ORBIT_RETURN = "Orbit Return"
    LANDING = "Landing"
    CREWED_LANDING = "Crewed Landing"
    FLAG_PLANT = "Flag Plant"
    RETURN = "Return"
    SAMPLE_RETURN = "Sample Return"


@dataclass(frozen=True)
class EventDef:
    """Metadata for a body mission event type.

    One row per event — single source of truth. All consumers (locations,
    rules, capability, CLI) derive their needs from this table.
    """
    name: EventName
    scale: int                            # location slots per body for this event
    mission_type: MissionType             # key into MissionBuilder.profiles_for
    crewed: bool | None                   # None=try both, True=crewed only, False=unmanned only
    requires_landing: bool                # only applies to landable bodies
    # Whether completing this event requires a Kerbal EVA (walk out of the
    # craft).  Source of truth for the curated Astronaut-Complex ``can_eva``
    # gate (buildings_in_logic).  FLAG_PLANT / SAMPLE_RETURN imply it from
    # their mission_type, but EVA-in-orbit shares the plain ORBIT type and so
    # needs this explicit flag.  Default False.
    requires_eva: bool = False
    # Whether this event is NOT emitted for the seed's HOME body.  A *home*
    # flyby is the weird "returning from interplanetary and passing home"
    # special case; the probe-only UNMANNED_FLYBY variant skips it (home orbit
    # is fine — that's a first-satellite check).  SOI_RETURN is excluded for the
    # same reason: "leave your home SOI and come back" is hard to explain to
    # players and reads as a chore, and home's return tiers are inverted anyway
    # (its surface is the starting point).  If the location budget ever needs
    # those slots back, Kerbol is the reserve source — see ``get_body_events``.
    # Applied event-side in ``ConfiguredLocations.from_settings`` because the
    # ``unachievable`` set is keyed by (body, mission_type) and can't single out
    # one event of a shared type.
    home_excluded: bool = False


ALL_EVENTS: tuple[EventDef, ...] = (
    EventDef(EventName.ORBIT,          1, MissionType.ORBIT,         None,  False),
    EventDef(EventName.EVA_IN_ORBIT,   1, MissionType.ORBIT,         True,  False, requires_eva=True),
    EventDef(EventName.ORBITAL_PROBE,  1, MissionType.ORBIT,         False, False),
    EventDef(EventName.FLYBY,          1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.SOI_LEAVE,      1, MissionType.ESCAPE,        None,  False),
    EventDef(EventName.UNMANNED_FLYBY, 1, MissionType.ESCAPE,        False, False, home_excluded=True),
    # The three return tiers.  requires_landing=False on the two shallow ones so
    # Jool (no surface) still gets a flyby-return and a capture-return check.
    EventDef(EventName.SOI_RETURN,     2, MissionType.SOI_RETURN,    None,  False, home_excluded=True),
    EventDef(EventName.ORBIT_RETURN,   2, MissionType.ORBIT_RETURN,  None,  False),
    EventDef(EventName.LANDING,        2, MissionType.LAND,          None,  True),
    EventDef(EventName.CREWED_LANDING, 2, MissionType.LAND,          True,  True),
    EventDef(EventName.FLAG_PLANT,     2, MissionType.FLAG_PLANT,    True,  True,  requires_eva=True),
    EventDef(EventName.RETURN,         3, MissionType.RETURN,        None,  True),
    EventDef(EventName.SAMPLE_RETURN,  3, MissionType.SAMPLE_RETURN, True,  True,  requires_eva=True),
)

EVENT_BY_NAME: dict[str, EventDef] = {e.name: e for e in ALL_EVENTS}


def get_body_events(body) -> tuple[EventName, ...]:
    """Return the AP event list for a body."""
    if body.name == BodyName.KERBOL:
        # Root body — can't flyby/escape, orbit is infeasible.  Kerbol is also
        # the reserve source of extra mission locations: it is excluded wholesale
        # here AND in ``MissionBuilder._build_destination_profiles``, so if the
        # location budget ever runs short, ``Kerbol Orbit Return`` /
        # ``Kerbol SOI Return`` can be turned on by relaxing both of those.
        return ()
    if body.can_land:
        return tuple(e.name for e in ALL_EVENTS)
    return tuple(e.name for e in ALL_EVENTS if not e.requires_landing)
