"""
Contract definitions for the KSP1 Archipelago contracts-as-items pacing system.

A contract is an AP item + AP location pair. The item paces progression (it is
placed into the run like any other item); completing the contract checks the
location, releasing an item to the multiworld. A contract location becomes
reachable only when THREE independent gates hold:

  1. the player has the contract item            (state.has(item))
  2. the player has the required parts           (category presence)
  3. physics can deliver the contract's kit       (capability profile eval)

This module is the single source of truth for the contract catalog, the
per-type required-part categories + client-facing KSP parameter trees, the seed
contract generator, and — critically — the feasibility check that is shared by
both generation (which contracts to place) and the runtime access rule (when a
contract is completable). Those two MUST call identical logic: a generation
false-positive (placing a contract that can never be completed) produces an
unsolvable seed.

The client is a dumb actuator: it receives the contract item, looks up the
metadata blob in slot_data, and builds a native KSP contract from the parameter
tree. New contract types that recombine existing parameter primitives need no
client release.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, TYPE_CHECKING

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, MissionType, DifficultyProfile,
    DIFFICULTY_PROFILES, MissionBuilder,
)
from .parts import CONTRACT_CATEGORY_MEMBERS, MiscEquipment

if TYPE_CHECKING:
    from .capability import EquipmentFlags
    from .world import KSP1World


# Bumped when the parameter wire format or primitive vocabulary changes. The
# client rejects contracts whose schema it doesn't understand. v5 added the
# variable reward-slot count (Contract Repeats): a non-goal contract's
# ``locations`` array may now hold more than 2 entries, so a v4 client that
# assumed exactly 2 must reject rather than silently drop the extras.
CONTRACT_SCHEMA_VERSION = 5

# Base number of reward locations a non-goal contract awards, sharing ONE gate
# item: completing the contract checks every slot, so each non-goal contract is
# net (slots - 1) locations of slack for the multiworld. Goal contracts stay 1:1
# (a single, unsuffixed location) — see ContractSpec.location_names.
NON_GOAL_SLOT_COUNT = 2

# Upper bound on the Contract Repeats option (extra reward slots per non-goal
# contract beyond the base 2). The data package registers every possible slot
# name up to NON_GOAL_SLOT_COUNT + MAX_CONTRACT_REPEATS so any option value has
# stable ids; a given seed creates only its resolved subset.
MAX_CONTRACT_REPEATS = 8
MAX_NON_GOAL_SLOT_COUNT = NON_GOAL_SLOT_COUNT + MAX_CONTRACT_REPEATS


def effective_contract_repeats(options) -> int:
    """Extra reward slots each non-goal contract yields beyond the base 2.
    Clamped to MAX_CONTRACT_REPEATS so a stale yaml can never exceed the
    registered slot-name universe. 0 (default) == today's exactly-2-slots
    behavior."""
    opt = getattr(options, "contract_repeats", None)
    if opt is None:
        return 0
    return max(0, min(int(opt.value), MAX_CONTRACT_REPEATS))


def non_goal_slot_count(options) -> int:
    """Reward-slot count per non-goal contract for this world:
    ``NON_GOAL_SLOT_COUNT + contract_repeats``."""
    return NON_GOAL_SLOT_COUNT + effective_contract_repeats(options)

# Event item locked on each non-goal contract's "Contract Complete: ..." event
# location (address None). state.has(this, X) == "X contracts completable in
# logic", which paces the count/progressive_unlock threshold locations.
CONTRACT_COMPLETED_EVENT = "Contract Completed"

# Mine Ore contract: fixed ore quantity to extract. 50 fits in the smallest ore
# tank (RadialOreTank holds 75), so a single tank suffices — 100 would force a
# second tank for no logic benefit.
MINE_ORE_UNITS = 50

# Space Station contract: required crew CAPACITY (seats). Fixed (not random) so
# the requirement is predictable; delivered as empty cabins to orbit.
STATION_CREW = 5

# Orbit-variant contracts: orbit-match tolerance passed to stock
# SpecificOrbitParameter (degrees / the param's deviation window). 10 is the stock
# satellite-contract default — generous enough to be achievable by hand.
ORBIT_DEVIATION = 10.0

# Part categories already guaranteed reachable by a progression chain (or
# standalone progression): power (Progressive Solar Panel + RTG), relay
# (Progressive Relay), crew cabins (Progressive Capsule command pods). The
# contract's category gate is satisfied by the chain rep, so promoting a
# specific part for these is redundant and only adds inert pool pressure.
_CHAIN_GUARANTEED_CATEGORIES = frozenset({"power", "relay", "crew_cabin"})


# ---------------------------------------------------------------------------
# Wire-format parameter primitives
# ---------------------------------------------------------------------------
# Each primitive maps 1:1 onto a compile-checked C# actuator wrapping a stock /
# custom KSP ContractParameter (see the design-plan appendix). KSP supplies
# OR/XOR and implicit-AND, so a flat list of these is an implicit-AND success
# condition. ``to_json`` is the slot_data wire form the client consumes.

@dataclass(frozen=True)
class SituationParam:
    """Vessel must reach ``situation`` (landed/orbiting/...) at ``body``.
    Wraps stock ReachSituation / LocationAndSituationParameter on the client."""
    situation: str
    body: str

    def to_json(self) -> dict:
        return {"kind": "situation", "situation": self.situation, "body": str(self.body)}


@dataclass(frozen=True)
class ResourceParam:
    """Vessel must hold >= ``minimum`` units of ``resource``.
    Wraps stock ResourcePossessionParameter on the client."""
    resource: str
    minimum: float

    def to_json(self) -> dict:
        return {"kind": "resource", "resource": self.resource, "min": self.minimum}


@dataclass(frozen=True)
class HasAnyPartParam:
    """Vessel must carry at least one of ``parts`` (AvailablePart.name). The
    server resolves a part CATEGORY to this explicit list so the client stays
    dumb. ``label`` is the category name, for display only. Wraps the mod's
    VesselHasPartParameter on the client."""
    parts: tuple[str, ...]
    label: str = ""

    def to_json(self) -> dict:
        return {"kind": "has_any_part", "parts": list(self.parts), "label": self.label}


@dataclass(frozen=True)
class HasSystemParam:
    """Vessel must carry a part providing ``system``, checked via KSP's native
    contract-objective system (stock VesselSystemsParameter on the client).
    ``system`` is a ContractObjectiveType (``"Generator"`` = any solar/RTG/fuel
    cell) or a PartModule class name (``"ModuleScienceLab"``). Preferred over
    has_any_part where a native objective exists — DLC/mod-robust and reads like
    a stock contract. ``label`` is the human description."""
    system: str
    label: str = ""

    def to_json(self) -> dict:
        return {"kind": "has_system", "system": self.system, "label": self.label}


@dataclass(frozen=True)
class CrewCapacityParam:
    """Vessel must have crew CAPACITY (seats, occupied or not) >= ``minimum``.
    Wraps stock CrewCapacityParameter on the client."""
    minimum: int

    def to_json(self) -> dict:
        return {"kind": "crew_capacity", "min": self.minimum}


@dataclass(frozen=True)
class PlantFlagParam:
    """Plant a flag on ``body``. Wraps stock PlantFlag on the client.
    (Needs a client primitive — orbit uses the existing 'situation'.)"""
    body: str

    def to_json(self) -> dict:
        return {"kind": "plant_flag", "body": str(self.body)}


@dataclass(frozen=True)
class SampleReturnParam:
    """Recover a surface sample from ``body`` back at the home world. Wraps a
    recover/collect-science parameter on the client (new primitive)."""
    body: str

    def to_json(self) -> dict:
        return {"kind": "sample_return", "body": str(self.body)}


@dataclass(frozen=True)
class RescueParam:
    """Rescue a stranded Kerbal from orbit of ``body`` and return them home. The
    client SPAWNS the stranded Kerbal (a small pod in low orbit around ``body``)
    when the contract is accepted, and completes when that Kerbal is recovered.
    Unlike every other primitive, this one creates world state rather than just
    watching the player's vessel — see the client RescuePrimitive."""
    body: str

    def to_json(self) -> dict:
        return {"kind": "rescue", "body": str(self.body)}


@dataclass(frozen=True)
class SpecificOrbitParam:
    """Match a specific target orbit around ``body``. Wraps stock
    SpecificOrbitParameter (the satellite-contract orbit param) on the client; the
    client renders the blue target orbit and completes when the active vessel
    matches within ``deviation``. The orbit is a deterministic function of
    (orbit_type, body) — circular at the body's low-orbit altitude — so nothing
    random crosses the wire. ``lan``/``arg_pe``/``mna``/``epoch`` are 0 for a
    circular orbit and defaulted client-side, so they're not sent."""
    body: str
    orbit_type: str          # FinePrint.Utilities.OrbitType name, e.g. "EQUATORIAL"
    inclination: float
    eccentricity: float
    sma: float               # semi-major axis, metres
    deviation: float

    def to_json(self) -> dict:
        return {
            "kind": "specific_orbit", "body": str(self.body),
            "orbit_type": self.orbit_type, "inclination": self.inclination,
            "eccentricity": self.eccentricity, "sma": self.sma,
            "deviation": self.deviation,
        }


@dataclass(frozen=True)
class CollectScienceParam:
    """Recover OR transmit science from ``body`` at ``location`` (space|surface).
    Wraps stock CollectScience on the client, which credits on transmit *or*
    recovery (GameEvents.OnScienceRecieved / OnTriggeredDataTransmission) — so the
    space variant is the cheap 'phone home' contract, no round trip."""
    body: str
    location: str            # "space" | "surface"

    def to_json(self) -> dict:
        return {"kind": "collect_science", "body": str(self.body), "location": self.location}


def _min_crew_combo(crew_parts, min_crew: int):
    """Cheapest (least-mass) set of crew parts whose seats total >= min_crew, as
    a tuple of parts. Considers single part types (ceil(min/seats) copies); a
    mixed combo could be marginally lighter, so this slightly OVERestimates —
    conservative (golden rule). None if no crew part is available."""
    best = None  # (total_mass, parts_tuple)
    for p in crew_parts:
        if p.crew_capacity <= 0:
            continue
        n = math.ceil(min_crew / p.crew_capacity)
        mass = n * p.mass
        if best is None or mass < best[0]:
            best = (mass, (p,) * n)
    return best[1] if best else None


# Union of all parameter primitives (extend as new primitives land).
ContractParam = SituationParam  # | ResourceParam | HasAnyPartParam | ...


# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------

class ContractType(StrEnum):
    MINE_ORE = "mine_ore"
    SURFACE_BASE = "surface_base"
    SPACE_STATION = "space_station"
    # Migrated mission types (additive non-goal pacing contracts; no extra kit).
    FLAG_PLANT = "flag_plant"
    SAMPLE_RETURN = "sample_return"
    ORBIT = "orbit"
    # Orbit-variant contracts (all wrap stock SpecificOrbitParameter; differ in
    # target orbit + the extra-dv their mission transform injects at the home body).
    EQUATORIAL_ORBIT = "equatorial_orbit"   # circular low equatorial orbit
    POLAR_ORBIT = "polar_orbit"             # circular low polar orbit (+v_rot ascent at home)
    STATIONARY_ORBIT = "stationary_orbit"   # synchronous orbit (+raise dv at home)
    RANDOM_ORBIT = "random_orbit"           # seeded inclined/eccentric satellite orbit
    TRANSMIT_SCIENCE = "transmit_science"   # phone home from a body's space (CollectScience)
    KERBAL_RESCUE = "kerbal_rescue"         # rescue a stranded Kerbal from orbit + return
    # Goal-only types — used when a goal achievement is one of these missions.
    RETURN = "return"
    FLYBY = "flyby"


# Non-goal contract types (the pool of weighted pacing contracts). RETURN/FLYBY
# are goal-only and never placed as non-goal contracts.
NON_GOAL_TYPES: tuple[ContractType, ...] = (
    ContractType.MINE_ORE, ContractType.SURFACE_BASE, ContractType.SPACE_STATION,
    ContractType.FLAG_PLANT, ContractType.SAMPLE_RETURN, ContractType.ORBIT,
    ContractType.EQUATORIAL_ORBIT, ContractType.POLAR_ORBIT,
    ContractType.STATIONARY_ORBIT, ContractType.RANDOM_ORBIT,
    ContractType.TRANSMIT_SCIENCE, ContractType.KERBAL_RESCUE,
)


# Part categories with a native KSP contract-objective check. "Generator" covers
# any solar panel / RTG / fuel cell; "ModuleScienceLab" matches the lab by module
# class. Categories absent here (battery, relay) have no precise native objective
# — battery is a plain resource, and "Antenna" would lose relay's range tiering —
# so they stay explicit part lists.
_CATEGORY_TO_SYSTEM: dict[str, tuple[str, str]] = {
    "power": ("Generator", "power generation"),
    "science_lab": ("ModuleScienceLab", "science lab"),
}


def _category_param(cat: str):
    """The success-condition param for a required part category: a native
    has_system check where one exists, else an explicit has_any_part list."""
    if cat in _CATEGORY_TO_SYSTEM:
        system, label = _CATEGORY_TO_SYSTEM[cat]
        return HasSystemParam(system=system, label=label)
    parts = tuple(sorted(CONTRACT_CATEGORY_MEMBERS.get(cat, frozenset())))
    return HasAnyPartParam(parts, label=cat)


# ---------------------------------------------------------------------------
# Contract requirements (the logic gate)
# ---------------------------------------------------------------------------
# A contract's required kit is a tuple of Requirement objects.  Each resolves to
# the delivery payload the sphere ladder charges and the runtime access rule
# checks.  The baseline catalog uses AnyOf exclusively — any available member of
# a part category satisfies it — derived automatically from each type's
# ``required_categories``.  Requirement is the typed extension point: new kinds
# (a specific part, a minimum rank) are added as subclasses here together with a
# resolver branch in required_part_breakdown / required_part_names_for, which
# fail closed on any kind they don't yet handle.

@dataclass(frozen=True)
class Requirement:
    """Base for a contract's logic-gate requirements."""


@dataclass(frozen=True)
class AnyOf(Requirement):
    """Satisfied by any available member of ``category`` (e.g. AnyOf('relay') —
    any antenna); the seed's lightest available member is the representative."""
    category: str


@dataclass(frozen=True)
class ContractTypeDef:
    """Static definition of a contract type.

    ``base_mission_type`` is the physics profile the required kit must be
    delivered through (LAND for surface contracts, ORBIT for orbital ones).
    ``required_categories`` are the part categories that must be present for the
    contract to be completable (the logic gate); they also drive which parts get
    promoted to progression in a seed that contains this type.
    """
    contract_type: ContractType
    location_noun: str               # "Mine Ore" -> "Contract: Mine Ore on Mun"
    base_mission_type: MissionType
    crewed: Optional[bool]           # None = try crewed and uncrewed
    required_categories: tuple[str, ...]
    title_fmt: str
    synopsis_fmt: str
    # Preposition joining noun and body in the location name, e.g. "Orbit
    # around Bop", "Sample Return from Dres", "Flyby of Jool". Default "on"
    # suits surface contracts.
    location_prep: str = "on"
    crew_requirement: int = 0        # >0 => the crew_cabin category must total N seats
    # True if this type makes sense on the HOME body. Orbit / station / satellite
    # content around home is good early-game; "go land/mine elsewhere" types are
    # silly at home. Generation only places a contract on the home body when this
    # is True (see the candidates loop).
    home_safe: bool = False
    # The logic-gate requirements (see Requirement). Defaults to one AnyOf per
    # required_categories entry; new types may author this explicitly — then it
    # is taken as given.
    requirements: tuple[Requirement, ...] = ()

    def __post_init__(self):
        if not self.requirements:
            object.__setattr__(
                self, "requirements",
                tuple(AnyOf(cat) for cat in self.required_categories))

    def requires_landing(self) -> bool:
        return self.base_mission_type in (
            MissionType.LAND, MissionType.FLAG_PLANT, MissionType.SAMPLE_RETURN,
            MissionType.RETURN)

    def body_compatible(self, body) -> bool:
        """True if this type can target ``body`` at all (before feasibility)."""
        if self.requires_landing():
            return body.can_land
        # All remaining types are orbital — they need a body that can be orbited
        # at all, which excludes the star (Kerbol). Gas giants (Jool) qualify.
        if not body.is_orbitable:
            return False
        if self.contract_type == ContractType.STATIONARY_ORBIT:
            # A synchronous orbit must exist above the surface and inside the
            # SOI — false for tidally-locked moons whose sync altitude is beyond
            # their SOI (no geostationary orbit there).
            return body.has_stationary_orbit
        return True

    def transform_mission(self, target_body, home_body, edges, mission_builder):
        """Contract-specific mission modifier — rewrite the base mission's edge
        sequence (the profile-level hook passed to evaluate_mission_detailed).
        Default identity. Orbit variants inject their extra delta-v here, and
        only at the HOME body: POLAR pays the rotation-assist loss as an ascent
        penalty; STATIONARY appends the low-orbit→sync raise burn; RANDOM pays
        the inclination rotation loss + a raise to its apoapsis. All are free at
        remote bodies (capture straight into the target plane / a high orbit),
        so they return ``edges`` unchanged off home. (RESCUE's rendezvous margin
        is baked into its profile at build time, not here — it's at the target
        body mid-trajectory, so it can't be appended without disconnecting.)"""
        if target_body != home_body:
            return edges
        home = BODY_BY_NAME[home_body]
        if self.contract_type == ContractType.POLAR_ORBIT:
            return mission_builder.add_ascent_penalty(
                edges, home_body, home.surface_rotation_velocity)
        if self.contract_type == ContractType.STATIONARY_ORBIT:
            return list(edges) + [
                mission_builder.make_raise_edge(home_body, home.stationary_raise_dv)]
        if self.contract_type == ContractType.RANDOM_ORBIT:
            params = mission_builder.random_orbit_params.get(target_body)
            if params is None:
                return edges  # defensive: no orbit assigned -> base orbit
            # Inclination rotation loss: the eastward assist you forgo, scaling
            # from 0 (equatorial) to the full surface-rotation velocity (polar),
            # i.e. v_rot * (1 - cos i). Charged on the ascent edge like POLAR.
            incl_penalty = home.surface_rotation_velocity * (
                1.0 - math.cos(math.radians(params.inclination_deg)))
            edges = mission_builder.add_ascent_penalty(
                edges, home_body, incl_penalty)
            # Apoapsis raise: conservatively model the eccentric orbit as a
            # circular orbit at its apoapsis (>= the actual eccentric orbit's
            # cost). raise_dv is 0 when apoapsis sits at low orbit.
            raise_dv = home.raise_dv(params.apoapsis_m)
            if raise_dv > 0.0:
                edges = list(edges) + [
                    mission_builder.make_raise_edge(home_body, raise_dv)]
            return edges
        return edges

    def build_parameters(self, body: BodyName, mission_builder=None) -> list:
        # ``mission_builder`` is required only for RANDOM_ORBIT (it owns the
        # per-body seeded target orbit); other types ignore it.
        if self.contract_type == ContractType.MINE_ORE:
            return [
                SituationParam("landed", body),
                ResourceParam("Ore", MINE_ORE_UNITS),
            ]
        if self.contract_type == ContractType.SURFACE_BASE:
            # Landed at the body with each required system on the vessel (lab +
            # battery + power + relay). Native objective checks where they exist
            # (lab/power), explicit part lists otherwise (battery/relay).
            params = [SituationParam("landed", body)]
            params += [_category_param(cat) for cat in self.required_categories]
            return params
        if self.contract_type == ContractType.SPACE_STATION:
            # In orbit with crew capacity >= N (stock CrewCapacityParameter, which
            # implies the cabins) plus battery + power + relay systems present.
            params = [SituationParam("orbiting", body),
                      CrewCapacityParam(self.crew_requirement)]
            params += [_category_param(cat) for cat in ("battery", "power", "relay")]
            return params
        if self.contract_type == ContractType.ORBIT:
            return [SituationParam("orbiting", body)]
        if self.contract_type in (ContractType.EQUATORIAL_ORBIT,
                                  ContractType.POLAR_ORBIT,
                                  ContractType.STATIONARY_ORBIT):
            # A circular target orbit, deterministic per (type, body). Equatorial
            # / polar at the body's low orbit (inc 0 / 90); stationary at the
            # synchronous radius. The mission transform (not here) adds the extra
            # delta-v polar/stationary need at the home body.
            b = BODY_BY_NAME[body]
            if self.contract_type == ContractType.STATIONARY_ORBIT:
                # A circular equatorial orbit at the synchronous radius IS a
                # stationary orbit; send EQUATORIAL + the explicit sync SMA so the
                # stock param can't recompute the altitude from an OrbitType.
                sma, inc, otype = b.sync_orbit_radius_m, 0.0, "EQUATORIAL"
            elif self.contract_type == ContractType.POLAR_ORBIT:
                sma, inc, otype = b.lo_radius_m, 90.0, "POLAR"
            else:
                sma, inc, otype = b.lo_radius_m, 0.0, "EQUATORIAL"
            return [SpecificOrbitParam(
                body=body, orbit_type=otype, inclination=inc,
                eccentricity=0.0, sma=sma, deviation=ORBIT_DEVIATION)]
        if self.contract_type == ContractType.RANDOM_ORBIT:
            # The seeded target orbit (inclination / apoapsis / eccentricity)
            # lives on the mission_builder; the client renders it via the stock
            # SpecificOrbitParameter and the player matches it within deviation.
            if mission_builder is None:
                raise ValueError("RANDOM_ORBIT build_parameters needs mission_builder")
            params = mission_builder.random_orbit_params.get(body)
            if params is None:
                raise ValueError(f"RANDOM_ORBIT on {body} has no assigned orbit")
            return [SpecificOrbitParam(
                body=body, orbit_type="EQUATORIAL",
                inclination=params.inclination_deg,
                eccentricity=params.eccentricity, sma=params.sma_m,
                deviation=ORBIT_DEVIATION)]
        if self.contract_type == ContractType.TRANSMIT_SCIENCE:
            # Gather + phone home science from the body's space. CollectScience
            # credits on transmit OR recover; the relay category is the antenna +
            # the range gate (remoteness cap keys on "relay" in required_categories).
            return [CollectScienceParam(body, "space"), _category_param("relay")]
        if self.contract_type == ContractType.FLAG_PLANT:
            return [PlantFlagParam(body)]
        if self.contract_type == ContractType.SAMPLE_RETURN:
            return [SampleReturnParam(body)]
        if self.contract_type == ContractType.KERBAL_RESCUE:
            # The client spawns the stranded Kerbal in orbit of `body` and
            # completes when they are recovered. The crew-cabin free seat is a
            # separate has_any_part objective so the player must actually have
            # room to bring the Kerbal home.
            params = [RescueParam(body)]
            params += [_category_param(cat) for cat in self.required_categories]
            return params
        if self.contract_type == ContractType.FLYBY:
            return [SituationParam("flyby", body)]      # stock EnterSOI(body)
        if self.contract_type == ContractType.RETURN:
            return [SampleReturnParam(body)]            # reach body then recover home
        raise NotImplementedError(
            f"build_parameters not implemented for {self.contract_type}")

    def title(self, body: BodyName) -> str:
        return self.title_fmt.format(
            body=body, units=MINE_ORE_UNITS, crew=self.crew_requirement)

    def synopsis(self, body: BodyName) -> str:
        return self.synopsis_fmt.format(
            body=body, units=MINE_ORE_UNITS, crew=self.crew_requirement)


CONTRACT_TYPE_DEFS: dict[ContractType, ContractTypeDef] = {
    ContractType.MINE_ORE: ContractTypeDef(
        contract_type=ContractType.MINE_ORE,
        location_noun="Mine Ore",
        base_mission_type=MissionType.LAND,
        crewed=None,                          # a drill rig can be probe-controlled
        required_categories=("drill", "ore_tank"),
        title_fmt="Mine {units} ore on {body}",
        synopsis_fmt="Extract {units} units of ore from the surface of {body}.",
    ),
    ContractType.SURFACE_BASE: ContractTypeDef(
        contract_type=ContractType.SURFACE_BASE,
        location_noun="Surface Base",
        base_mission_type=MissionType.LAND,
        crewed=None,                          # can be delivered uncrewed
        required_categories=("science_lab", "battery", "power", "relay"),
        title_fmt="Build a surface base on {body}",
        synopsis_fmt="Land a science base (Mobile Lab + power + relay) on {body}.",
    ),
    ContractType.SPACE_STATION: ContractTypeDef(
        contract_type=ContractType.SPACE_STATION,
        location_noun="Space Station",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,                          # empty cabins delivered to orbit
        required_categories=("crew_cabin", "battery", "power", "relay"),
        crew_requirement=STATION_CREW,
        home_safe=True,
        title_fmt="Build a space station in orbit of {body}",
        synopsis_fmt="Assemble a {crew}-crew station (power + relay) in {body} orbit.",
    ),
    # Migrated mission types — no extra kit; feasibility == the base mission.
    ContractType.ORBIT: ContractTypeDef(
        contract_type=ContractType.ORBIT,
        location_noun="Orbit",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=(),
        home_safe=True,
        title_fmt="Reach orbit of {body}",
        synopsis_fmt="Establish a stable orbit around {body}.",
    ),
    ContractType.EQUATORIAL_ORBIT: ContractTypeDef(
        contract_type=ContractType.EQUATORIAL_ORBIT,
        location_noun="Equatorial Orbit",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=(),
        home_safe=True,
        title_fmt="Reach an equatorial orbit of {body}",
        synopsis_fmt="Circularise an equatorial orbit around {body}.",
    ),
    ContractType.POLAR_ORBIT: ContractTypeDef(
        contract_type=ContractType.POLAR_ORBIT,
        location_noun="Polar Orbit",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=(),
        home_safe=True,
        title_fmt="Reach a polar orbit of {body}",
        synopsis_fmt="Circularise a polar orbit around {body}.",
    ),
    ContractType.STATIONARY_ORBIT: ContractTypeDef(
        contract_type=ContractType.STATIONARY_ORBIT,
        location_noun="Stationary Orbit",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=(),
        home_safe=True,
        title_fmt="Reach a stationary orbit of {body}",
        synopsis_fmt="Establish a synchronous (stationary) orbit around {body}.",
    ),
    ContractType.RANDOM_ORBIT: ContractTypeDef(
        contract_type=ContractType.RANDOM_ORBIT,
        location_noun="Satellite Orbit",
        location_prep="around",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=(),
        home_safe=True,
        title_fmt="Deploy a satellite around {body}",
        synopsis_fmt="Place a satellite into the assigned orbit around {body}.",
    ),
    ContractType.TRANSMIT_SCIENCE: ContractTypeDef(
        contract_type=ContractType.TRANSMIT_SCIENCE,
        location_noun="Transmit Science",
        location_prep="from",
        base_mission_type=MissionType.ORBIT,
        crewed=None,
        required_categories=("relay",),
        home_safe=True,
        title_fmt="Transmit science from {body}",
        synopsis_fmt="Gather and transmit science from space around {body}.",
    ),
    ContractType.KERBAL_RESCUE: ContractTypeDef(
        contract_type=ContractType.KERBAL_RESCUE,
        location_noun="Crew Rescue",
        location_prep="around",
        base_mission_type=MissionType.RESCUE,
        crewed=None,
        # A free seat to bring the stranded Kerbal home (delivered as payload,
        # like a station's crew cabins). crewed=None lets a probe-controlled
        # craft with an empty cabin do it (lightest), or a crewed capsule.
        required_categories=("crew_cabin",),
        crew_requirement=1,
        home_safe=True,
        title_fmt="Rescue a stranded Kerbal in orbit of {body}",
        synopsis_fmt="Rendezvous with a stranded Kerbal in orbit of {body} and "
                     "bring them home safely.",
    ),
    ContractType.FLAG_PLANT: ContractTypeDef(
        contract_type=ContractType.FLAG_PLANT,
        location_noun="Flag Plant",
        base_mission_type=MissionType.FLAG_PLANT,
        crewed=True,
        required_categories=(),
        title_fmt="Plant a flag on {body}",
        synopsis_fmt="Land a kerbal on {body} and plant a flag.",
    ),
    ContractType.SAMPLE_RETURN: ContractTypeDef(
        contract_type=ContractType.SAMPLE_RETURN,
        location_noun="Sample Return",
        location_prep="from",
        base_mission_type=MissionType.SAMPLE_RETURN,
        crewed=True,
        required_categories=(),
        title_fmt="Return a surface sample from {body}",
        synopsis_fmt="Collect a surface sample from {body} and bring it home.",
    ),
    # Goal-only types (used when a goal achievement is a return/flyby).
    ContractType.RETURN: ContractTypeDef(
        contract_type=ContractType.RETURN,
        location_noun="Return",
        location_prep="from",
        base_mission_type=MissionType.RETURN,
        crewed=None,
        required_categories=(),
        title_fmt="Return from {body}",
        synopsis_fmt="Travel to {body} and return safely home.",
    ),
    ContractType.FLYBY: ContractTypeDef(
        contract_type=ContractType.FLYBY,
        location_noun="Flyby",
        location_prep="of",
        base_mission_type=MissionType.ESCAPE,
        crewed=None,
        required_categories=(),
        title_fmt="Fly by {body}",
        synopsis_fmt="Perform a flyby of {body}.",
    ),
}


@dataclass(frozen=True)
class ContractSpec:
    """One concrete contract instance, fixed at generation time."""
    contract_type: ContractType
    body: BodyName
    is_goal: bool = False
    # Client-display title override (Mission Control contract name). Only set
    # for flavour cases like the random_contracts "free" goal ("Plant Flag on
    # Launch Pad"); None means derive from the type_def. Cosmetic only —
    # location_name / item_name never depend on it, so it can't affect logic.
    title_override: Optional[str] = None

    @property
    def type_def(self) -> ContractTypeDef:
        return CONTRACT_TYPE_DEFS[self.contract_type]

    @property
    def contract_id(self) -> str:
        # Stable per (type, body); the capability cache key.
        return f"{self.contract_type}:{self.body}"

    @property
    def display_name(self) -> str:
        td = self.type_def
        return f"Contract: {td.location_noun} {td.location_prep} {self.body}"

    def mission_transform(self, mission_builder: MissionBuilder):
        """The profile-level edge modifier for this contract (polar ascent
        penalty / stationary raise), bound to this contract's body and home.
        Pure — returns a new edge list per call, never mutating the shared base
        profiles. Shared by the runtime feasibility check (``evaluate_contract``)
        and the sphere-ladder signature so the two cannot drift."""
        td = self.type_def
        return lambda edges: td.transform_mission(
            self.body, mission_builder.home, edges, mission_builder)

    # AP item and location share the same descriptive string (separate namespaces).
    @property
    def item_name(self) -> str:
        return self.display_name

    def location_names(self, slot_count: int = NON_GOAL_SLOT_COUNT) -> tuple[str, ...]:
        """The reward location(s) this contract checks. Goal contracts stay 1:1
        (a single unsuffixed location, ignoring ``slot_count``); non-goal
        contracts award ``slot_count`` slot-suffixed locations ("... 1",
        "... 2", ...) that share one gate item and one access rule.
        ``slot_count`` defaults to the base 2 (today's behavior); a world threads
        ``NON_GOAL_SLOT_COUNT + contract_repeats`` through for repeating contracts
        and the data package threads ``MAX_NON_GOAL_SLOT_COUNT`` to register every
        possible slot name."""
        if self.is_goal:
            return (self.display_name,)
        return tuple(
            f"{self.display_name} {i}"
            for i in range(1, slot_count + 1)
        )

    @property
    def location_name(self) -> str:
        """The canonical / primary location (slot 1). Goal logic, /explain, and
        the client's binding key all use this; the suffixed siblings share its
        access rule. Independent of the seed's slot count — slot 1 always exists
        (>= the base 2 non-goal slots), so this is stable as repeats vary."""
        if self.is_goal:
            return self.display_name
        return f"{self.display_name} 1"

    def to_slot_dict(self, mission_builder=None,
                     slot_count: int = NON_GOAL_SLOT_COUNT) -> dict:
        """The self-describing manifest entry the dumb client actuates. Carries
        ``contract_type``/``body`` structurally so UT regen reconstructs the
        spec from fields, never by parsing the display name (the client ignores
        these two extra keys). ``locations`` is the full slot list (1 for goal,
        ``slot_count`` for non-goal); the client reports every entry on
        completion. ``mission_builder`` is required only for RANDOM_ORBIT (it owns
        the seeded target orbit the client renders)."""
        td = self.type_def
        d = {
            "item": self.item_name,
            "locations": list(self.location_names(slot_count)),
            "title": self.title_override if self.title_override else td.title(self.body),
            "synopsis": td.synopsis(self.body),
            "schema": CONTRACT_SCHEMA_VERSION,
            "is_goal": self.is_goal,
            "contract_type": str(self.contract_type),
            "body": str(self.body),
            "parameters": [p.to_json()
                           for p in td.build_parameters(self.body, mission_builder)],
        }
        if self.title_override:
            # Round-trips through UT regen so the flavour title survives.
            d["title_override"] = self.title_override
        return d

    @staticmethod
    def from_slot_dict(d: dict) -> "ContractSpec":
        """Rebuild a spec from a slot_data manifest entry (UT regen — never
        re-randomize). Reads the structured ``contract_type``/``body`` fields."""
        return ContractSpec(
            ContractType(d["contract_type"]),
            BodyName(d["body"]),
            is_goal=bool(d.get("is_goal", False)),
            title_override=d.get("title_override"),
        )


def _parse_exact_contract_name(name: str) -> Optional[ContractSpec]:
    """Match a name against the bare ``display_name`` of some (type, body), or
    None. Rebuilds each candidate and compares — no preposition/format coupling."""
    body_str = name.rsplit(None, 1)[-1]          # body is the final token
    try:
        body = BodyName(body_str)
    except ValueError:
        return None
    for ct in CONTRACT_TYPE_DEFS:
        spec = ContractSpec(ct, body)
        if spec.display_name == name:
            return spec
    return None


def parse_contract_location_name(name: str) -> Optional[ContractSpec]:
    """Return the ContractSpec for a contract location name, or None if it isn't
    one. Accepts BOTH the bare goal-contract form ("Contract: Mine Ore on Mun")
    and the non-goal slot-suffixed form ("Contract: Mine Ore on Mun 1" / "... 2").
    Used by the sphere ladder to give every contract slot a real signature — a
    silent None here un-gates the location and deadlocks fill (bug 086 / project
    memory). Threshold ("Contract Threshold N") and event ("Contract Complete:
    ...") names deliberately don't match (different prefix)."""
    if not name.startswith("Contract: "):
        return None
    spec = _parse_exact_contract_name(name)
    if spec is not None:
        return spec
    # Strip a trailing slot integer ("... 1") and retry against the bare form.
    base, _, last = name.rpartition(" ")
    if last.isdigit():
        return _parse_exact_contract_name(base)
    return None


# ---------------------------------------------------------------------------
# Part requirements & feasibility (shared by generation and access rules)
# ---------------------------------------------------------------------------

def required_part_breakdown(
    spec: ContractSpec, flags: "EquipmentFlags",
) -> list[tuple[str, Optional[tuple[MiscEquipment, ...]]]]:
    """Per required category, the available parts to deliver (the lightest part,
    or for a crew contract the cheapest combo reaching the seat count), or None
    for a requirement with no available part. Generic over contract types — a
    single iteration of ``requirements`` that BOTH the delivery manifest and the
    ``/explain`` Gate-2 breakdown read, so they cannot diverge."""
    td = spec.type_def
    out: list[tuple[str, Optional[tuple[MiscEquipment, ...]]]] = []
    for req in td.requirements:
        if not isinstance(req, AnyOf):
            raise NotImplementedError(f"requirement kind not handled: {req!r}")
        cat = req.category
        if cat == "crew_cabin" and td.crew_requirement:
            combo = _min_crew_combo(flags.available_crew_parts, td.crew_requirement)
            out.append((cat, combo))
        else:
            part = flags.category_lightest.get(cat)
            out.append((cat, (part,) if part is not None else None))
    return out


def required_part_manifest(
    spec: ContractSpec, flags: "EquipmentFlags",
) -> Optional[tuple[MiscEquipment, ...]]:
    """The available parts to deliver — the lightest per required category, plus
    for a crew contract the cheapest crew combo reaching the seat requirement.
    Returns None if any required category has no available part — the contract is
    then infeasible (fail-closed, per the golden rule)."""
    parts: list[MiscEquipment] = []
    for _cat, got in required_part_breakdown(spec, flags):
        if got is None:
            return None
        parts.extend(got)
    return tuple(parts)


def contract_payload_parts(
    spec: ContractSpec, flags: "EquipmentFlags",
) -> Optional[tuple[MiscEquipment, ...]]:
    """The delivery payload the sphere ladder charges for a contract, sized from
    ``flags`` so the ladder signature matches the runtime access rule.

    For the CHAIN-GUARANTEED categories (crew / relay / power) this is exactly
    ``required_part_manifest``: the parts come from this seed's progressive
    representatives carried in ``flags``, so the signature can't be optimistic
    about a lighter member the chain doesn't actually guarantee.

    The ladder's progressive-only kit can't grant the PROMOTED standalone
    categories (drill / ore_tank / battery / science_lab), so for those we fall
    back to the guaranteed representative — the lightest member, which is exactly
    what ``required_part_names_for`` promotes and what the runtime's
    ``category_lightest`` resolves to once that item is collected. Folding them
    in here (as pure payload mass) keeps the lab's crew seats and the drill's
    ISRU flag from leaking into the ladder's capability flags.

    Returns ``None`` if a chain-guaranteed category has no part at this kit — the
    contract is infeasible at this rung and the bumper bumps the relevant chain.
    """
    from .parts import CONTRACT_CATEGORY_MEMBERS, PART_DB
    parts: list[MiscEquipment] = []
    for cat, got in required_part_breakdown(spec, flags):
        if got is not None:
            parts.extend(got)
        elif cat in _CHAIN_GUARANTEED_CATEGORIES:
            return None  # chain rep not yet unlocked at this kit
        else:
            members = CONTRACT_CATEGORY_MEMBERS.get(cat, frozenset())
            if members:
                parts.append(min((PART_DB[n][0] for n in members),
                                 key=lambda p: p.mass))
    return tuple(parts)


def evaluate_contract(
    spec: ContractSpec,
    flags: "EquipmentFlags",
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
):
    """Full-kit mission evaluation for a contract with its required-part payload
    on the manifest, or None if a required category has no available part.
    Shared by the feasibility check and the harder-than-goal launch-mass cap, so
    they read the same numbers."""
    manifest = required_part_manifest(spec, flags)
    if manifest is None:
        return None
    # Local import keeps the contracts <-> capability cycle one-directional.
    from .capability import evaluate_mission_detailed
    td = spec.type_def
    return evaluate_mission_detailed(
        flags, diff, spec.body, td.base_mission_type, td.crewed,
        mission_builder, extra_payload_parts=manifest,
        mission_transform=spec.mission_transform(mission_builder),
    )


def can_complete_contract(
    spec: ContractSpec,
    flags: "EquipmentFlags",
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
) -> bool:
    """The single shared feasibility check. Used by BOTH the generator's
    ever-achievable filter and the runtime access rule, so they cannot drift."""
    result = evaluate_contract(spec, flags, diff, mission_builder)
    return result is not None and result.feasible


def compute_contract_access(
    specs,
    flags: "EquipmentFlags",
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
) -> dict[str, bool]:
    """Per-contract feasibility map cached on RocketCapability. Called once per
    cold capability state from compute_capability_from_items."""
    return {
        s.contract_id: can_complete_contract(s, flags, diff, mission_builder)
        for s in specs
    }


def required_part_names_for(specs) -> frozenset[str]:
    """The part ksp_names to promote to progression for THIS seed: ONE
    representative — the lightest — per required category over the contracts
    actually generated. AP only guarantees progression items reachable, so we
    must guarantee a usable part per category; promoting only the lightest (the
    one capability selects as ``category_lightest``) avoids bloating the
    advancement pool with every variant, which adds fill pressure that can
    strand goal-path items. Empty when no contract uses a category (e.g. mine
    disabled) — unused mining parts then keep their normal classification."""
    from .parts import PART_DB
    cats: set[str] = set()
    for s in specs:
        for req in s.type_def.requirements:
            if not isinstance(req, AnyOf):
                raise NotImplementedError(f"requirement kind not handled: {req!r}")
            cats.add(req.category)
    # Categories already guaranteed reachable by a progression chain (solar/relay
    # antennas, command pods). Promoting a *specific* part for them is redundant
    # AND adds an inert progression item fill can strand on a hard location,
    # increasing pool pressure for no gate benefit — the chain rep already
    # satisfies category_lightest. Only promote contract-specific equipment with
    # no chain (drill, ore tank, lab, battery).
    cats -= _CHAIN_GUARANTEED_CATEGORIES
    reps: set[str] = set()
    for cat in cats:
        members = CONTRACT_CATEGORY_MEMBERS.get(cat, frozenset())
        if members:
            reps.add(min(members, key=lambda n: PART_DB[n][0].mass))
    return frozenset(reps)


# ---------------------------------------------------------------------------
# Universe (stable item/location name registration) + generation
# ---------------------------------------------------------------------------

def all_possible_contract_specs() -> list[ContractSpec]:
    """Every (type, body) a contract could ever target, independent of seed.
    items.py / locations.py register names from this so the name set is stable."""
    out: list[ContractSpec] = []
    for ct in ContractType:
        td = CONTRACT_TYPE_DEFS[ct]
        for body in ALL_BODIES:
            if td.body_compatible(body):
                out.append(ContractSpec(ct, body.name))
    return out


def _full_kit_flags(world: "KSP1World") -> "EquipmentFlags":
    """EquipmentFlags with every part/tier unlocked — the upper bound used to
    decide whether a contract is EVER achievable in this seed."""
    from .capability import _pre_pass
    return _pre_pass(
        lambda name: 99,
        start_with_clamps=bool(world.options.start_with_launch_clamps.value),
        progressive_launch_pad=bool(world.options.progressive_launch_pad.value),
        launch_pad_caps=world.mission_builder.launch_pad_caps,
    )


def _difficulty(world: "KSP1World") -> DifficultyProfile:
    name = ["casual", "normal", "expert", "insane"][world.options.difficulty.value]
    return DIFFICULTY_PROFILES[name]


# Auto (-1) total non-goal contract count, by difficulty index. Harder settings
# get fewer (tighter fill, faster gen). Calibrated further once data exists.
_AUTO_COUNT_BY_DIFFICULTY = (12, 10, 8, 6)


def _resolve_count(world: "KSP1World") -> int:
    raw = world.options.contracts_available.value
    if raw < 0:
        return _AUTO_COUNT_BY_DIFFICULTY[world.options.difficulty.value]
    return raw


# Goal body-list attribute -> the mission type it implies. Used to compute the
# goal's intrinsic difficulty from the STRUCTURED goal_spec (no name parsing).
_GOAL_BODY_LISTS: tuple[tuple[str, MissionType], ...] = (
    ("flag_bodies", MissionType.FLAG_PLANT),
    ("return_bodies", MissionType.RETURN),
    ("sample_return_bodies", MissionType.SAMPLE_RETURN),
    ("orbit_bodies", MissionType.ORBIT),
    ("flyby_bodies", MissionType.ESCAPE),
)


def _goal_achievements(goal_spec) -> set:
    """The (body, mission_type) pairs the goal already requires. Migrated-type
    contracts skip these so the goal stays untouched (no contract gating the goal
    mission, no redundant contract on a goal body)."""
    out = set()
    for attr, mtype in _GOAL_BODY_LISTS:
        for body in getattr(goal_spec, attr, ()):
            out.add((body, mtype))
    return out


# Goal mission type -> the contract type that represents it.
_GOAL_MISSION_TO_CONTRACT: dict[MissionType, ContractType] = {
    MissionType.FLAG_PLANT: ContractType.FLAG_PLANT,
    MissionType.RETURN: ContractType.RETURN,
    MissionType.SAMPLE_RETURN: ContractType.SAMPLE_RETURN,
    MissionType.ORBIT: ContractType.ORBIT,
    MissionType.ESCAPE: ContractType.FLYBY,
}

# The contract types that can represent a goal achievement (the only types that
# ever become goal contracts). Used to size the threshold-location registry.
GOAL_CONTRACT_TYPES: frozenset[ContractType] = frozenset(
    _GOAL_MISSION_TO_CONTRACT.values())


def _goal_contract_specs(goal_spec) -> list:
    """One goal-contract per goal body achievement. Always generated (a goal is
    mandatory) — no ever-achievable filter; model-infeasible goal bodies fall
    back to the all-parts proxy in the access rule (see rules._set_contract_rules)."""
    # The random_contracts "free" goal is a single home-body flag plant; give it
    # a celebratory title instead of the bare "Plant Flag on Kerbin".
    free = getattr(goal_spec, "free_goal", False)
    out = []
    for body, mtype in sorted(_goal_achievements(goal_spec)):
        ct = _GOAL_MISSION_TO_CONTRACT.get(mtype)
        if ct is not None:
            title_override = (
                "Plant Flag on Launch Pad"
                if free and ct == ContractType.FLAG_PLANT else None
            )
            out.append(ContractSpec(ct, body, is_goal=True,
                                    title_override=title_override))
    return out


# Inverse of (_GOAL_BODY_LISTS composed with _GOAL_MISSION_TO_CONTRACT): a goal
# contract's type -> the custom-goal body-list option attribute it came from.
_CONTRACT_TO_GOAL_BODY_LIST: dict[ContractType, str] = {
    _GOAL_MISSION_TO_CONTRACT[mtype]: attr
    for attr, mtype in _GOAL_BODY_LISTS
}


def goal_body_lists_from_specs(specs) -> dict[str, set[str]]:
    """Invert ``_goal_contract_specs``: recover the custom-goal body-list option
    values (``{option_attr: {body_name, ...}}``) from a list of goal ContractSpecs.

    UT regen restores a custom goal with this — the goal contracts are the only
    structured record of the body lists that survives slot_data (``goal_locations``
    holds contract *display* names, which don't round-trip to a body+mission-type).
    Non-goal specs are ignored. Every option attr is present (empty set when
    unused) so the caller can assign unconditionally."""
    out: dict[str, set[str]] = {attr: set() for attr, _ in _GOAL_BODY_LISTS}
    for spec in specs:
        if not spec.is_goal:
            continue
        attr = _CONTRACT_TO_GOAL_BODY_LIST.get(spec.contract_type)
        if attr is not None:
            out[attr].add(str(spec.body))
    return out


def _goal_max_mass(goal_spec, flags, diff: DifficultyProfile,
                   mission_builder: MissionBuilder) -> float:
    """The hardest goal mission's full-kit launch mass — the difficulty ceiling
    for the harder-than-goal cap. Payload-aware (unlike bare trajectory dv): each
    goal achievement is evaluated as its equivalent goal contract, so candidate
    contracts compare apples-to-apples (a crewed station's payload counts toward
    difficulty). 0.0 when the goal has no body missions (e.g. complete_tech_tree)
    — callers then skip the cap (nothing to compare against)."""
    mass = 0.0
    for spec in _goal_contract_specs(goal_spec):
        result = evaluate_contract(spec, flags, diff, mission_builder)
        if result is not None and result.feasible:
            mass = max(mass, result.launch_mass)
    return mass


def _goal_max_relay_tier(goal_spec, mission_builder: MissionBuilder) -> int:
    """The highest relay tier any goal body needs to reach home — the remoteness
    ceiling for the harder-than-goal cap. A relay-requiring contract (station,
    base) at a body beyond this tier would need comms infrastructure the goal
    never does (a return mission brings everything home; an unattended station
    must phone home), so it's 'harder than goal' on the remoteness axis even when
    its launch mass is not. Distinct from mass: it catches far airless bodies
    (e.g. Eeloo) that carry no aerobrake/dv penalty."""
    rt = mission_builder.relay_tier_by_body
    tier = 0
    for attr, _mtype in _GOAL_BODY_LISTS:
        for body in getattr(goal_spec, attr, ()):
            tier = max(tier, rt.get(body, 0))
    return tier


# ---------------------------------------------------------------------------
# Synthetic difficulty anchors for goals with no natural body mission
# (random_contracts and complete_tech_tree). Both need a contract-difficulty
# cap, but neither has a goal mission whose mass/relay-tier to compare against:
#   - random_contracts: anchor at a difficulty-scaled percentile of the mission
#     Δv scale, so contracts ramp up to (but not past) a sensible ceiling.
#   - complete_tech_tree: anchor at the science-funding body returns (the
#     missions the player must do to unlock the tree), so contracts are no harder
#     than the tree itself demands.
# ---------------------------------------------------------------------------

# random_contracts: fraction up the RETURN-mass scale, indexed by difficulty
# (casual / normal / expert / insane). Harder settings allow harder contracts.
_RANDOM_CONTRACTS_PERCENTILE: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)


def _return_mass_scale(flags, diff, mb) -> list[tuple[float, BodyName]]:
    """``(launch_mass, body)`` for a RETURN from every landable body but home,
    feasible-only, ascending. The intrinsic mission-difficulty ladder used to
    anchor goals that carry no body mission of their own."""
    out: list[tuple[float, BodyName]] = []
    for b in ALL_BODIES:
        if not b.can_land or b.name == mb.home:
            continue
        result = evaluate_contract(ContractSpec(ContractType.RETURN, b.name), flags, diff, mb)
        if result is not None and result.feasible:
            out.append((result.launch_mass, b.name))
    out.sort(key=lambda t: (t[0], t[1].value))
    return out


def _anchor_cap(anchor_bodies, flags, diff, mb) -> tuple[float, int]:
    """``(max RETURN launch mass, max relay tier)`` over the anchor bodies —
    the synthetic harder-than-goal ceiling. 0.0 mass means no cap."""
    rt = mb.relay_tier_by_body
    mass = 0.0
    tier = 0
    for body in anchor_bodies:
        result = evaluate_contract(ContractSpec(ContractType.RETURN, body), flags, diff, mb)
        if result is not None and result.feasible:
            mass = max(mass, result.launch_mass)
        tier = max(tier, rt.get(body, 0))
    return mass, tier


def _random_contracts_anchor_body(world, flags, diff, mb):
    """The body whose RETURN sits at the difficulty-scaled percentile of the
    mission Δv scale; random_contracts caps contracts at its difficulty. None if
    no feasible interplanetary return exists (cap then disabled)."""
    scale = _return_mass_scale(flags, diff, mb)
    if not scale:
        return None
    pct = _RANDOM_CONTRACTS_PERCENTILE[world.options.difficulty.value]
    idx = min(len(scale) - 1, round(pct * (len(scale) - 1)))
    return scale[idx][1]


def _tech_tree_anchor_bodies(world, flags, diff, mb) -> list[BodyName]:
    """The interplanetary RETURN bodies whose science funds the whole tech tree,
    cheapest-mass first until the tier-MAX science target is met. Deterministic —
    a difficulty cap, not the sphere ladder's seed-varied anchor set, but the same
    greedy science accounting (see sphere_ladder._pick_tech_tree_anchors). Empty
    when home-system science alone funds the tree (cap stays at home difficulty)."""
    from .bodies import BODY_BY_NAME, home_system_bodies, science_budget
    from .tech_tree import cumulative_tier_cost, MAX_TIER
    from .rules import effective_science_safety

    safety = effective_science_safety(world.options, world.options.difficulty.value)
    target = cumulative_tier_cost(MAX_TIER) / safety
    home = mb.home
    home_set = home_system_bodies(home)

    def body_yield(body) -> float:
        return science_budget(
            body, has_thermometer=True, has_barometer=True, has_capsule=True,
            can_land_crewed=body.can_land, home=home, psi_tier=3)

    accumulated = sum(body_yield(BODY_BY_NAME[bn]) for bn in home_set)
    picked: list[BodyName] = []
    for _mass, body in _return_mass_scale(flags, diff, mb):
        if accumulated >= target:
            break
        if body in home_set:
            continue
        accumulated += body_yield(BODY_BY_NAME[body])
        picked.append(body)
    return picked


def _difficulty_cap(world, flags, diff, mb) -> tuple[float, int | None]:
    """Resolve the harder-than-goal cap as ``(goal_mass, goal_relay_tier)``.

    ``allow_missions_harder_than_goal`` (on) disables the cap. Otherwise a goal
    with body missions caps at its hardest mission; random_contracts and the tech
    tree cap at their synthetic anchors (above)."""
    if world.options.allow_missions_harder_than_goal.value:
        return 0.0, None
    if world.goal_spec.free_goal:
        body = _random_contracts_anchor_body(world, flags, diff, mb)
        if body is None:
            return 0.0, None
        return _anchor_cap([body], flags, diff, mb)
    if world.goal_spec.complete_tech_tree:
        bodies = _tech_tree_anchor_bodies(world, flags, diff, mb)
        if not bodies:
            return 0.0, None
        return _anchor_cap(bodies, flags, diff, mb)
    goal_mass = _goal_max_mass(world.goal_spec, flags, diff, mb)
    goal_relay_tier = _goal_max_relay_tier(world.goal_spec, mb) if goal_mass > 0 else None
    return goal_mass, goal_relay_tier


def goal_contracts_easiest_first(world: "KSP1World") -> list[ContractSpec]:
    """This world's goal contracts ordered easiest-first by full-kit launch mass
    (model-infeasible / proxy goals sort last via an infinite key; ties broken by
    contract_id for determinism). progressive_unlock awards them in this order —
    the easiest goal mission unlocks at the lowest contract-count threshold."""
    flags = _full_kit_flags(world)
    diff = _difficulty(world)
    mb = world.mission_builder

    def sort_key(spec: ContractSpec) -> tuple[float, str]:
        result = evaluate_contract(spec, flags, diff, mb)
        mass = (result.launch_mass
                if result is not None and result.feasible else float("inf"))
        return (mass, spec.contract_id)

    return sorted(world.goal_contract_specs, key=sort_key)


def generate_contracts(world: "KSP1World") -> tuple[list[ContractSpec], list[ContractSpec]]:
    """Pick this seed's contracts. Returns (non_goal, goal).

    Non-goal: every ever-achievable (type, body) is a candidate, weighted-sampled
    without replacement by the per-type weight (0 disables), up to the configured
    count, deterministically from the world seed.

    Goal: one mandatory goal-contract per goal achievement (see
    _goal_contract_specs) — always generated, no ever-achievable filter, so the
    goal/victory missions are contract locations the player must complete.
    """
    rng = world.random
    diff = _difficulty(world)
    mb = world.mission_builder
    home = mb.home
    full = _full_kit_flags(world)
    weights: dict = dict(world.options.contract_type_weights.value)

    # Home-system-local invariant (NOT optional): a goal that stays within the
    # home neighbourhood must never put a contract out of system — logic must
    # never make the player leave home for a home-local run. Otherwise every
    # body is eligible. (sorted() keeps the home-local branch deterministic.)
    from .bodies import home_system_bodies, BODY_BY_NAME
    eligible_bodies = (
        [BODY_BY_NAME[n] for n in sorted(home_system_bodies(home))]
        if world.goal_spec.is_home_system_only(home) else ALL_BODIES
    )

    # Harder-than-goal cap, on two independent axes (see _difficulty_cap):
    #  - launch mass: a contract whose full-kit launch mass exceeds the goal's
    #    hardest mission is too heavy (payload-aware, not bare trajectory dv, so a
    #    crewed station's mass counts). Catches heavy-at-similar-distance.
    #  - relay tier: a relay-requiring contract (station/base) at a body more
    #    remote than the goal's farthest needs comms infrastructure the goal never
    #    does. Catches too-far — including airless bodies (Eeloo) that carry no
    #    mass penalty.
    # goal_mass == 0 (allow_missions_harder_than_goal, or no anchor) disables both.
    # random_contracts / complete_tech_tree have no goal mission, so they cap at a
    # synthetic anchor instead.
    goal_mass, goal_relay_tier = _difficulty_cap(world, full, diff, mb)
    goal_achievements = _goal_achievements(world.goal_spec)

    candidates: list[ContractSpec] = []
    for ct in NON_GOAL_TYPES:
        if weights.get(str(ct), 0) <= 0:
            continue
        td = CONTRACT_TYPE_DEFS[ct]
        for body in eligible_bodies:
            # The home body is eligible only for home-safe types (orbital /
            # satellite / station content); "go land/mine elsewhere" types skip it.
            if (body.name == home and not td.home_safe) or not td.body_compatible(body):
                continue
            # Don't contractize a mission the goal already requires — the goal
            # mission becomes a GOAL-contract instead (below), so a non-goal
            # contract here would collide / double up.
            if (body.name, td.base_mission_type) in goal_achievements:
                continue
            # Remoteness cap (cheap, so it gates before the optimizer eval).
            if (goal_relay_tier is not None and "relay" in td.required_categories
                    and mb.relay_tier_by_body.get(body.name, 0) > goal_relay_tier):
                continue  # needs comms beyond the goal's reach (option-gated)
            spec = ContractSpec(ct, body.name)
            result = evaluate_contract(spec, full, diff, mb)
            if result is None or not result.feasible:
                continue  # missing a required part, or can't be delivered at all
            if goal_mass > 0 and result.launch_mass > goal_mass:
                continue  # harder than the goal mission, by launch mass (option-gated)
            candidates.append(spec)

    chosen = _weighted_sample_without_replacement(
        rng, candidates, weights, _resolve_count(world))
    # Goals become contracts: one goal-contract per goal achievement, mandatory.
    return chosen, _goal_contract_specs(world.goal_spec)


def _weighted_sample_without_replacement(rng, candidates, weights, k):
    pool = list(candidates)
    chosen: list[ContractSpec] = []
    while pool and len(chosen) < k:
        ws = [max(weights.get(str(c.contract_type), 0), 0) for c in pool]
        if sum(ws) <= 0:
            break
        idx = rng.choices(range(len(pool)), weights=ws, k=1)[0]
        chosen.append(pool.pop(idx))
    return chosen
