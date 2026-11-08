"""The locations a world emits as a pure function of its resolved settings.

Inputs are plain values — the resolved home, the per-seed slot counts, and the
``unachievable`` mission set (the checked-in feasibility table plus curated
bans, both functions of settings) — never the AP options object, so this layer
stays Archipelago-free and UT regen reproduces it exactly from slot_data.

Contracts and thresholds are rolled later in generation; see ``seed.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..bodies import BodyName, MissionType
from .catalog import LocationCatalog
from .defs import (
    HomeLocation, HomeMilestone, KSP1LocationDef, KscBiomeLocation,
    MissionLocation, SplashdownLocation, StartingInventoryLocation,
    TechTreeLocation,
)
from .events import EventName


@dataclass(frozen=True)
class ConfiguredLocations:
    """Settings-derived emitted locations.  Build with ``from_settings``."""
    home: BodyName
    starting_inventory: tuple[StartingInventoryLocation, ...]
    ksc_biomes: tuple[KscBiomeLocation, ...]
    home_specials: tuple[HomeLocation, ...]
    splashdown: SplashdownLocation
    tech_slots: int
    tech: tuple[TechTreeLocation, ...]
    missions: tuple[MissionLocation, ...]
    # Missions this world deliberately does not emit (the world's
    # ``model_infeasible_locations``).
    excluded_missions: frozenset[MissionLocation]

    @classmethod
    def from_settings(
        cls, *, home: BodyName, starting_inv_count: int, tech_slots: int,
        unachievable: frozenset[tuple[BodyName, MissionType]],
    ) -> ConfiguredLocations:
        """``unachievable`` is every (body, mission_type) the capability model
        can't verify (physics-infeasible ∪ curated ban).  A world emits only
        the mission locations it can reach, so ``accessibility=full`` holds —
        the mission case of a general invariant: a location exists iff its
        access rule is satisfiable.  The offline feasibility generator passes
        an empty set so it keeps probing true maximal capability."""

        def emitted(ml: MissionLocation) -> bool:
            ed = ml.event_def
            if (ml.body, ed.mission_type) in unachievable:
                return False
            if ed.home_excluded and ml.body == home:
                return False
            return True

        all_missions = LocationCatalog.missions()
        return cls(
            home=home,
            starting_inventory=LocationCatalog.starting_inventory()[:starting_inv_count],
            ksc_biomes=tuple(loc for loc in LocationCatalog.ksc_biomes()
                             if loc.biome.available_on(home)),
            home_specials=LocationCatalog.home_specials(home),
            splashdown=LocationCatalog.splashdown(),
            tech_slots=tech_slots,
            tech=tuple(loc for loc in LocationCatalog.tech() if loc.slot <= tech_slots),
            missions=tuple(ml for ml in all_missions if emitted(ml)),
            excluded_missions=frozenset(ml for ml in all_missions if not emitted(ml)),
        )

    def emitted(self) -> tuple[KSP1LocationDef, ...]:
        return (
            *self.starting_inventory, *self.ksc_biomes, *self.home_specials,
            self.splashdown, *self.tech, *self.missions,
        )

    def emitted_mission(self, body: BodyName, event: EventName,
                        slot: int = 1) -> MissionLocation | None:
        """The emitted mission for ``(body, event, slot)``, or ``None`` when this
        world doesn't emit it (the mission is unreachable)."""
        ml = MissionLocation(body, event, slot)
        return ml if ml in self._mission_set else None

    def home_special(self, milestone: HomeMilestone) -> HomeLocation:
        """This home's special for a single-instance milestone (not ALTITUDE)."""
        if milestone is HomeMilestone.ALTITUDE:
            raise ValueError("ALTITUDE has several milestones; use altitude_milestones")
        return next(loc for loc in self.home_specials if loc.milestone is milestone)

    @property
    def altitude_milestones(self) -> tuple[HomeLocation, ...]:
        return tuple(loc for loc in self.home_specials
                     if loc.milestone is HomeMilestone.ALTITUDE)

    @cached_property
    def _mission_set(self) -> frozenset[MissionLocation]:
        return frozenset(self.missions)
