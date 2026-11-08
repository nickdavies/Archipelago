"""The static location universe: every location that can ever exist.

This is what the AP data package advertises (``location_name_to_id``), what
the Universal Tracker map and the CLI can name, and what keeps location ids
stable across seeds.  A single world emits a subset — see ``configured.py``
(settings-derived) and ``seed.py`` (generation-derived).

Ids are assigned in fixed offset blocks.  The order within each block is
load-bearing: changing it shifts ids, which breaks existing seeds and the
client.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from ..bodies import ALL_BODIES, BODY_BY_NAME, BodyName, home_altitude_milestones
from ..contracts import (
    GOAL_CONTRACT_TYPES, MAX_LOCATIONS_PER_CONTRACT, all_possible_contract_specs,
)
from ..tech_tree import TECH_NODES
from .defs import (
    ContractLocation, HomeLocation, HomeMilestone, KSP1LocationDef, KscBiome,
    KscBiomeLocation, MissionLocation, SplashdownLocation,
    StartingInventoryLocation, TechTreeLocation, ThresholdLocation,
)
from .events import EVENT_BY_NAME, EventName, get_body_events

KSP1_BASE_ID = 7_700_000

# Offset blocks (items use 0–1999).
_STARTING_INV_OFFSET = 2000   # 2000-2019
_SPLASHDOWN_OFFSET = 2079     # one id, any ocean body
_KSC_BIOME_OFFSET = 2080      # 2080-2099
_KERBIN_HOME_OFFSET = 2100    # Kerbin home specials (legacy block, 11 used)
_MISSION_OFFSET = 2200        # per-body mission events, 2200-2999
_TECH_OFFSET = 3000           # tech tree, 3000-3999
_ALT_HOME_OFFSET = 4000       # non-Kerbin home specials, 14 bodies × 11
_THRESHOLD_OFFSET = 19_000    # goal-mode thresholds, 19_000-19_099
_CONTRACT_OFFSET = 20_000     # contract rewards, large dedicated block

MAX_STARTING_INV = 20
MAX_TECH_SLOTS = 4
# Held constant across homes so every home contributes exactly 11 specials
# (4 fixed + 7 altitude milestones) and the per-home id blocks stay regular.
HOME_ALTITUDE_MILESTONE_COUNT = 7

#: Landable bodies (possible homes), in ``ALL_BODIES`` order.
LANDABLE_HOMES: tuple[BodyName, ...] = tuple(
    BodyName(b.name) for b in ALL_BODIES if b.can_land)


def _home_specials(home: BodyName) -> tuple[HomeLocation, ...]:
    milestones = home_altitude_milestones(
        BODY_BY_NAME[home], n=HOME_ALTITUDE_MILESTONE_COUNT)
    return (
        HomeLocation(home, HomeMilestone.FIRST_LAUNCH),
        HomeLocation(home, HomeMilestone.FIRST_LANDING),
        HomeLocation(home, HomeMilestone.FIRST_CRASH),
        *(HomeLocation(home, HomeMilestone.ALTITUDE, km) for km in milestones),
        HomeLocation(home, HomeMilestone.FIRST_STAGING),
    )


_STARTING_INV = tuple(StartingInventoryLocation(i) for i in range(1, MAX_STARTING_INV + 1))
_SPLASHDOWN = SplashdownLocation()
_KSC = tuple(KscBiomeLocation(b) for b in KscBiome)
_HOME_SPECIALS: Mapping[BodyName, tuple[HomeLocation, ...]] = MappingProxyType(
    {home: _home_specials(home) for home in LANDABLE_HOMES})
# Body order in ALL_BODIES, then events, then slots.
_MISSIONS = tuple(
    MissionLocation(body.name, event, slot)
    for body in ALL_BODIES
    for event in get_body_events(body)
    for slot in range(1, EVENT_BY_NAME[event].scale + 1)
)
_TECH = tuple(
    TechTreeLocation(node, slot)
    for node in TECH_NODES
    for slot in range(1, MAX_TECH_SLOTS + 1)
)
# One threshold per possible goal achievement (each goal contract type on every
# body it can target) — derived, so adding bodies/types can't overflow.
_THRESHOLDS = tuple(
    ThresholdLocation(i) for i in range(1, 1 + sum(
        1 for spec in all_possible_contract_specs()
        if spec.contract_type in GOAL_CONTRACT_TYPES))
)
# Every reward location a contract could ever register: per (type, body), the
# bare goal form plus every non-goal slot up to the Contract Repeats ceiling.
# Registering the universe maximum keeps ids independent of the seed's
# per-contract count.  Sorted by contract_id so the block is stable.
_CONTRACTS = tuple(
    loc
    for spec in sorted(all_possible_contract_specs(), key=lambda s: s.contract_id)
    for loc in (
        ContractLocation(spec, None),
        *(ContractLocation(spec, i) for i in range(1, MAX_LOCATIONS_PER_CONTRACT + 1)),
    )
)


def _build_ids() -> dict[KSP1LocationDef, int]:
    blocks: list[tuple[int, tuple[KSP1LocationDef, ...]]] = [
        (_STARTING_INV_OFFSET, _STARTING_INV),
        (_SPLASHDOWN_OFFSET, (_SPLASHDOWN,)),
        (_KSC_BIOME_OFFSET, _KSC),
        (_KERBIN_HOME_OFFSET, _HOME_SPECIALS[BodyName.KERBIN]),
        (_MISSION_OFFSET, _MISSIONS),
        (_TECH_OFFSET, _TECH),
        (_ALT_HOME_OFFSET, tuple(
            loc for home in LANDABLE_HOMES if home != BodyName.KERBIN
            for loc in _HOME_SPECIALS[home])),
        (_THRESHOLD_OFFSET, _THRESHOLDS),
        (_CONTRACT_OFFSET, _CONTRACTS),
    ]
    ids: dict[KSP1LocationDef, int] = {}
    for offset, defs in blocks:
        for i, defn in enumerate(defs):
            ids[defn] = KSP1_BASE_ID + offset + i
    return ids


_IDS = _build_ids()
_BY_NAME: Mapping[str, KSP1LocationDef] = MappingProxyType({str(d): d for d in _IDS})
_NAME_TO_ID: Mapping[str, int] = MappingProxyType({str(d): i for d, i in _IDS.items()})
assert len(_BY_NAME) == len(_IDS), "two location defs share a display name"
assert len(set(_IDS.values())) == len(_IDS), "two location defs share an id"


class LocationCatalog:
    """Every location that can ever exist.  Classmethods only; all data is
    built eagerly at import and immutable."""

    _ALL: ClassVar[tuple[KSP1LocationDef, ...]] = tuple(_IDS)

    @classmethod
    def all(cls) -> tuple[KSP1LocationDef, ...]:
        """Every location, in id order."""
        return cls._ALL

    @classmethod
    def name_to_id(cls) -> Mapping[str, int]:
        """The AP data-package table."""
        return _NAME_TO_ID

    @classmethod
    def id_of(cls, defn: KSP1LocationDef) -> int:
        return _NAME_TO_ID[str(defn)]

    @classmethod
    def by_name(cls, name: str) -> KSP1LocationDef | None:
        """The def behind an AP location name, or ``None`` if no such location
        exists.  For the name-keyed boundaries (AP registry, UT, CLI) only."""
        return _BY_NAME.get(name)

    @classmethod
    def starting_inventory(cls) -> tuple[StartingInventoryLocation, ...]:
        return _STARTING_INV

    @classmethod
    def splashdown(cls) -> SplashdownLocation:
        return _SPLASHDOWN

    @classmethod
    def ksc_biomes(cls) -> tuple[KscBiomeLocation, ...]:
        return _KSC

    @classmethod
    def home_specials(cls, home: BodyName) -> tuple[HomeLocation, ...]:
        return _HOME_SPECIALS[home]

    @classmethod
    def missions(cls) -> tuple[MissionLocation, ...]:
        return _MISSIONS

    @classmethod
    def missions_for(cls, body: BodyName, event: EventName) -> tuple[MissionLocation, ...]:
        """Every slot of one body's event, whether or not a world emits it."""
        return tuple(MissionLocation(body, event, slot)
                     for slot in range(1, EVENT_BY_NAME[event].scale + 1))

    @classmethod
    def tech(cls) -> tuple[TechTreeLocation, ...]:
        return _TECH

    @classmethod
    def thresholds(cls) -> tuple[ThresholdLocation, ...]:
        return _THRESHOLDS

    @classmethod
    def contracts(cls) -> tuple[ContractLocation, ...]:
        return _CONTRACTS
