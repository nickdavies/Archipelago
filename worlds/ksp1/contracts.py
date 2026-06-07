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

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, TYPE_CHECKING

from .bodies import (
    ALL_BODIES, BodyName, MissionType, DifficultyProfile, DIFFICULTY_PROFILES,
    MissionBuilder,
)
from .parts import CONTRACT_CATEGORY_MEMBERS, MiscEquipment

if TYPE_CHECKING:
    from .capability import EquipmentFlags
    from .world import KSP1World


# Bumped when the parameter wire format or primitive vocabulary changes. The
# client rejects contracts whose schema it doesn't understand.
CONTRACT_SCHEMA_VERSION = 1

# Mine Ore contract: fixed ore quantity to extract. 50 fits in the smallest ore
# tank (RadialOreTank holds 75), so a single tank suffices — 100 would force a
# second tank for no logic benefit.
MINE_ORE_UNITS = 50


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


# Union of all parameter primitives (extend as new primitives land).
ContractParam = SituationParam  # | ResourceParam | HasAnyPartParam | ...


# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------

class ContractType(StrEnum):
    MINE_ORE = "mine_ore"
    # Phase 2/3 add: SURFACE_BASE, SPACE_STATION, FLAG_PLANT, SAMPLE_RETURN, ORBIT.


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

    def requires_landing(self) -> bool:
        return self.base_mission_type == MissionType.LAND

    def body_compatible(self, body) -> bool:
        """True if this type can target ``body`` at all (before feasibility)."""
        if self.requires_landing():
            return body.can_land
        # Orbital types: any body that can be orbited (Kerbol/Sun excluded).
        return body.can_land or body.name == BodyName.JOOL

    def build_parameters(self, body: BodyName) -> list:
        if self.contract_type == ContractType.MINE_ORE:
            return [
                SituationParam("landed", body),
                ResourceParam("Ore", MINE_ORE_UNITS),
            ]
        raise NotImplementedError(
            f"build_parameters not implemented for {self.contract_type}")

    def title(self, body: BodyName) -> str:
        return self.title_fmt.format(body=body, units=MINE_ORE_UNITS)

    def synopsis(self, body: BodyName) -> str:
        return self.synopsis_fmt.format(body=body, units=MINE_ORE_UNITS)


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
}


@dataclass(frozen=True)
class ContractSpec:
    """One concrete contract instance, fixed at generation time."""
    contract_type: ContractType
    body: BodyName
    is_goal: bool = False

    @property
    def type_def(self) -> ContractTypeDef:
        return CONTRACT_TYPE_DEFS[self.contract_type]

    @property
    def contract_id(self) -> str:
        # Stable per (type, body); the capability cache key.
        return f"{self.contract_type}:{self.body}"

    @property
    def display_name(self) -> str:
        return f"Contract: {self.type_def.location_noun} on {self.body}"

    # AP item and location share the same descriptive string (separate namespaces).
    @property
    def item_name(self) -> str:
        return self.display_name

    @property
    def location_name(self) -> str:
        return self.display_name

    def to_slot_dict(self) -> dict:
        """The self-describing manifest entry the dumb client actuates."""
        td = self.type_def
        return {
            "item": self.item_name,
            "location": self.location_name,
            "title": td.title(self.body),
            "synopsis": td.synopsis(self.body),
            "schema": CONTRACT_SCHEMA_VERSION,
            "parameters": [p.to_json() for p in td.build_parameters(self.body)],
        }

    @staticmethod
    def from_slot_dict(d: dict) -> "ContractSpec":
        """Rebuild a spec from a slot_data manifest entry (UT regen — never
        re-randomize). The (type, body) pair is recovered from the location
        name's ``Contract: <noun> on <body>`` form."""
        name = d["location"]
        body = BodyName(name.rsplit(" on ", 1)[1])
        noun = name[len("Contract: "):].rsplit(" on ", 1)[0]
        ct = _CONTRACT_TYPE_BY_NOUN[noun]
        return ContractSpec(ct, body)


_CONTRACT_TYPE_BY_NOUN: dict[str, ContractType] = {
    td.location_noun: td.contract_type for td in CONTRACT_TYPE_DEFS.values()
}


def parse_contract_location_name(name: str) -> Optional[ContractSpec]:
    """Return the ContractSpec for a contract item/location name (they share the
    "Contract: <noun> on <body>" string), or None if it isn't one. Used by the
    sphere ladder to give contract locations a real signature."""
    if not name.startswith("Contract: "):
        return None
    try:
        noun, body_str = name[len("Contract: "):].rsplit(" on ", 1)
        ct = _CONTRACT_TYPE_BY_NOUN.get(noun)
        if ct is None:
            return None
        return ContractSpec(ct, BodyName(body_str))
    except (ValueError, KeyError):
        return None


def canonical_payload_parts(spec: ContractSpec) -> tuple:
    """The canonical required-equipment parts a contract must DELIVER — the
    lightest registered part per required category (matches the progression
    representative promoted for the seed). Deterministic and collection-state
    independent, so the sphere ladder can size the contract's delivery kit and
    Rule B can protect the contract location from bootstrap items."""
    from .parts import CONTRACT_CATEGORY_MEMBERS, PART_DB
    parts = []
    for cat in spec.type_def.required_categories:
        members = CONTRACT_CATEGORY_MEMBERS.get(cat, frozenset())
        if members:
            lightest = min(members, key=lambda n: PART_DB[n][0].mass)
            parts.append(PART_DB[lightest][0])
    return tuple(parts)


# ---------------------------------------------------------------------------
# Part requirements & feasibility (shared by generation and access rules)
# ---------------------------------------------------------------------------

def required_part_manifest(
    spec: ContractSpec, flags: "EquipmentFlags",
) -> Optional[tuple[MiscEquipment, ...]]:
    """The lightest available part for each required category, as the payload to
    deliver. Returns None if any required category has no available part — the
    contract is then infeasible (fail-closed, per the golden rule)."""
    parts: list[MiscEquipment] = []
    for cat in spec.type_def.required_categories:
        part = flags.category_lightest.get(cat)
        if part is None:
            return None
        parts.append(part)
    return tuple(parts)


def can_complete_contract(
    spec: ContractSpec,
    flags: "EquipmentFlags",
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
) -> bool:
    """The single shared feasibility check. Used by BOTH the generator's
    ever-achievable filter and the runtime access rule, so they cannot drift."""
    manifest = required_part_manifest(spec, flags)
    if manifest is None:
        return False
    # Local import keeps the contracts <-> capability cycle one-directional.
    from .capability import evaluate_mission_detailed
    td = spec.type_def
    result = evaluate_mission_detailed(
        flags, diff, spec.body, td.base_mission_type, td.crewed,
        mission_builder, extra_payload_parts=manifest,
    )
    return result.feasible


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
        cats.update(s.type_def.required_categories)
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
        rep_names=frozenset(),
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
    raw = world.options.non_goal_contract_count.value
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


def _intrinsic_dv(mission_builder: MissionBuilder, body: BodyName,
                  mission_type: MissionType) -> float:
    """Cheapest-profile delta-v for a (body, mission_type) — the sphere ladder's
    own difficulty scalar. 0.0 if no profile exists."""
    profiles = mission_builder.profiles_for(body, mission_type)
    if not profiles:
        return 0.0
    return min(sum(e.base_dv for e in profile) for profile in profiles)


def _goal_max_dv(goal_spec, mission_builder: MissionBuilder) -> float:
    """The hardest goal mission's intrinsic dv. 0.0 when the goal has no body
    missions (e.g. complete_tech_tree) — callers then skip the harder-than-goal
    filter (there is no goal mission to compare against)."""
    dv = 0.0
    for attr, mtype in _GOAL_BODY_LISTS:
        for body in getattr(goal_spec, attr, ()):
            dv = max(dv, _intrinsic_dv(mission_builder, body, mtype))
    return dv


def generate_contracts(world: "KSP1World") -> tuple[list[ContractSpec], list[ContractSpec]]:
    """Pick this seed's contracts. Returns (non_goal, goal).

    Non-goal: every ever-achievable (type, body) is a candidate, weighted-sampled
    without replacement by the per-type weight (0 disables), up to the configured
    count, deterministically from the world seed.

    Goal contracts are phase 3 — returns [] for now (the existing goal/victory
    system is untouched).
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

    # Harder-than-goal cap (option, default ON = no cap): when off, contracts
    # whose intrinsic mission dv exceeds the goal's hardest mission are excluded.
    # goal_dv == 0 (e.g. complete_tech_tree, no body missions) disables the cap.
    allow_harder = bool(world.options.allow_missions_harder_than_goal.value)
    goal_dv = 0.0 if allow_harder else _goal_max_dv(world.goal_spec, mb)

    candidates: list[ContractSpec] = []
    for ct in ContractType:
        if weights.get(str(ct), 0) <= 0:
            continue
        td = CONTRACT_TYPE_DEFS[ct]
        for body in eligible_bodies:
            if body.name == home or not td.body_compatible(body):
                continue
            if goal_dv > 0 and _intrinsic_dv(mb, body.name, td.base_mission_type) > goal_dv:
                continue  # harder than the goal mission (option-gated)
            spec = ContractSpec(ct, body.name)
            if can_complete_contract(spec, full, diff, mb):
                candidates.append(spec)

    chosen = _weighted_sample_without_replacement(
        rng, candidates, weights, _resolve_count(world))
    return chosen, []


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
