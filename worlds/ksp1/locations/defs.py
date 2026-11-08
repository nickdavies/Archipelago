"""Typed location definitions: one frozen dataclass per location family.

A location's display name is a boundary concern (AP's name-keyed registry and
the client wire).  Inside the world a location is one of these objects, and
``__str__`` is the ONLY place its name is formed.  Consumers dispatch on the
type (``match`` + ``assert_never``) and read structured fields — never parse a
name or test a prefix.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from ..bodies import BodyName, MissionType
from ..tech_tree import TechNode
from .events import EVENT_BY_NAME, EventDef, EventName

if TYPE_CHECKING:
    from ..contracts import ContractSpec


@dataclass(frozen=True)
class StartingInventoryLocation:
    """Zero-requirement bootstrap slot, auto-checked on connect."""
    slot: int  # 1-based

    def __str__(self) -> str:
        return f"Starting Inventory {self.slot}"


class KscBiome(Enum):
    """KSC science biomes: (KSP biome key, building display name, kerbin_only).

    Order matters for the client's sub-biome fallback: it iterates the
    slot_data map in this order and falls back to ``startswith`` on the key
    when no exact match is found (e.g. ``VABMainBuilding`` → ``VAB``).  The
    bare ``KSC`` key is a catch-all prefix and MUST stay last so more-specific
    keys win.

    ``kerbin_only``: the Grounds biome (KSP's catch-all ``KSC`` key — the grass
    and water around the buildings) only exists on Kerbin.  Kerbal Konstructs
    places the named buildings on alien homes, but the surrounding terrain
    doesn't report as ``KSC`` there, so an alien-home world can't emit it.
    """
    LAUNCH_PAD = ("LaunchPad", "KSC LaunchPad", False)
    RUNWAY = ("Runway", "KSC Runway", False)
    ADMINISTRATION = ("Administration", "KSC Administration", False)
    ASTRONAUT_COMPLEX = ("AstronautComplex", "KSC Astronaut Complex", False)
    FLAG_POLE = ("FlagPole", "KSC Flag Pole (Astronaut Complex)", False)
    SPH = ("SPH", "KSC SPH", False)
    VAB = ("VAB", "KSC VAB", False)
    TRACKING_STATION = ("TrackingStation", "KSC Tracking Station", False)
    MISSION_CONTROL = ("MissionControl", "KSC Mission Control", False)
    CRAWLERWAY = ("Crawlerway", "KSC Crawlerway", False)
    RND = ("R&D", "KSC R&D", False)
    GROUNDS = ("KSC", "KSC Grounds", True)

    def __init__(self, key: str, display: str, kerbin_only: bool) -> None:
        self.key = key
        self.display = display
        self.kerbin_only = kerbin_only

    def available_on(self, home: BodyName) -> bool:
        return home == BodyName.KERBIN or not self.kerbin_only


@dataclass(frozen=True)
class KscBiomeLocation:
    """Science collected at a KSC building / biome."""
    biome: KscBiome

    def __str__(self) -> str:
        return f"Science from {self.biome.display}"


class HomeMilestone(Enum):
    """The home-body special checks.  ``value`` is the name suffix."""
    FIRST_LAUNCH = "First Launch"
    FIRST_LANDING = "First Landing"
    FIRST_CRASH = "First Crash"
    ALTITUDE = "Altitude"
    FIRST_STAGING = "First Staging"

    @property
    def mission_type(self) -> MissionType:
        return _MILESTONE_MISSION_TYPE[self]


_MILESTONE_MISSION_TYPE: dict[HomeMilestone, MissionType] = {
    HomeMilestone.FIRST_LAUNCH: MissionType.FIRST_LAUNCH,
    HomeMilestone.FIRST_LANDING: MissionType.FIRST_LANDING,
    # A crash is detected as a sounding flight that gets off the pad.
    HomeMilestone.FIRST_CRASH: MissionType.SOUNDING,
    HomeMilestone.ALTITUDE: MissionType.SOUNDING,
    HomeMilestone.FIRST_STAGING: MissionType.FIRST_STAGING,
}


@dataclass(frozen=True)
class HomeLocation:
    """A home-body special: first launch / landing / crash / staging, or an
    altitude milestone (``altitude_km`` set iff ``milestone is ALTITUDE``)."""
    home: BodyName
    milestone: HomeMilestone
    altitude_km: int | None = None

    #: Sounding altitude a First Crash requires: leave the pad at all.
    FIRST_CRASH_KM: ClassVar[float] = 0.1

    def __post_init__(self) -> None:
        if (self.altitude_km is not None) != (self.milestone is HomeMilestone.ALTITUDE):
            raise ValueError(
                f"altitude_km must be set exactly for ALTITUDE milestones: {self!r}")

    @property
    def body(self) -> BodyName:
        return self.home

    @property
    def mission_type(self) -> MissionType:
        return self.milestone.mission_type

    @property
    def threshold_km(self) -> float | None:
        """Sounding altitude the check needs, for the SOUNDING milestones."""
        if self.milestone is HomeMilestone.ALTITUDE:
            return float(self.altitude_km)
        if self.milestone is HomeMilestone.FIRST_CRASH:
            return self.FIRST_CRASH_KM
        return None

    def __str__(self) -> str:
        if self.milestone is HomeMilestone.ALTITUDE:
            return f"{self.home} {self.altitude_km}km Altitude"
        return f"{self.home} {self.milestone.value}"


@dataclass(frozen=True)
class SplashdownLocation:
    """Body-agnostic: one check for splashing down on any ocean body
    (Kerbin / Eve / Laythe), present in every world."""

    #: Sounding altitude a home-ocean splashdown needs before the safe descent.
    SOUNDING_KM: ClassVar[float] = 1.0

    @property
    def mission_type(self) -> MissionType:
        return MissionType.SPLASHDOWN

    def __str__(self) -> str:
        return "Splashdown"


@dataclass(frozen=True)
class MissionLocation:
    """A per-body mission event slot, e.g. "Mun Orbit 1"."""
    body: BodyName
    event: EventName
    slot: int  # 1-based

    @property
    def event_def(self) -> EventDef:
        return EVENT_BY_NAME[self.event]

    def __str__(self) -> str:
        return f"{self.body} {self.event} {self.slot}"


@dataclass(frozen=True)
class TechTreeLocation:
    """A tech-tree node slot, e.g. "Basic Rocketry 3"."""
    node: TechNode
    slot: int  # 1-based

    def __str__(self) -> str:
        return f"{self.node.display_name} {self.slot}"


@dataclass(frozen=True)
class ContractLocation:
    """A contract reward location.  Goal contracts have exactly one, unsuffixed
    (``slot is None``); non-goal contracts have numbered slots sharing one gate
    item and one access rule."""
    spec: ContractSpec
    slot: int | None

    @classmethod
    def for_spec(cls, spec: ContractSpec, slot_count: int) -> tuple[ContractLocation, ...]:
        """The reward location(s) ``spec`` checks.  ``slot_count`` is ignored
        for goal contracts (always the single unsuffixed location)."""
        if spec.is_goal:
            return (cls(spec, None),)
        return tuple(cls(spec, i) for i in range(1, slot_count + 1))

    @property
    def body(self) -> BodyName:
        return self.spec.body

    def __str__(self) -> str:
        if self.slot is None:
            return self.spec.display_name
        return f"{self.spec.display_name} {self.slot}"


@dataclass(frozen=True)
class ThresholdLocation:
    """A goal-mode contract-count threshold (count / progressive_unlock):
    pre-filled, reported by the client when enough contracts are complete."""
    index: int  # 1-based

    def __str__(self) -> str:
        return f"Contract Threshold {self.index}"


KSP1LocationDef = (
    StartingInventoryLocation | KscBiomeLocation | HomeLocation | SplashdownLocation
    | MissionLocation | TechTreeLocation | ContractLocation | ThresholdLocation
)
