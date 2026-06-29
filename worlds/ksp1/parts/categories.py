"""Contract part categories: generic, resolvable groups a contract can require.

``PartCategory.resolve`` maps a category to the ksp_names that satisfy it,
optionally narrowed to an ``allowed`` subset (used for pack filtering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .registry import PART_REGISTRY
from ._raw import _PARTS_JSON


# ---------------------------------------------------------------------------
# Part categories — generic, resolvable groups a contract can require
# ---------------------------------------------------------------------------
# A contract's part requirement is a *category*, not a hardcoded part. Some
# categories derive from real metadata (e.g. "stores Ore") and auto-pick-up
# any matching part; others must be a curated list because parts.json carries
# no PartModule info to distinguish them (drills, labs, crew cabins). Both the
# generation-time feasibility logic and the client-facing ``has_any_part``
# parameter resolve through this one source of truth, so they can never drift.

@dataclass(frozen=True)
class PartCategory:
    """A named group of parts. Membership is the union of three matchers, any of
    which may be empty: an explicit cfg_name set, a metadata predicate over the
    raw parts.json cfg dict, and a set of capability ``provides`` flags (matches
    any part whose provides intersect it — the clean way to name battery / power
    / relay, which already carry provides flags). ``resolve`` returns the
    matching ksp_names (= AvailablePart.name on the client), optionally narrowed
    to an ``allowed`` subset for future DLC / part-pool filtering.
    """
    key: str
    members: frozenset[str] = frozenset()                 # explicit cfg_names
    predicate: Optional[Callable[[dict], bool]] = None    # over raw cfg dict
    provides_any: frozenset[str] = frozenset()            # capability provides flags
    exclude: frozenset[str] = frozenset()                 # cfg_names to never match
    description: str = ""

    def _matches(self, cfg_name: str, cfg: dict, provides: frozenset[str]) -> bool:
        if cfg_name in self.exclude:
            return False
        if cfg_name in self.members:
            return True
        if self.predicate is not None and self.predicate(cfg):
            return True
        return bool(self.provides_any and (provides & self.provides_any))

    def resolve(self, allowed: Optional[frozenset[str]] = None) -> frozenset[str]:
        out: set[str] = set()
        for mapping in PART_REGISTRY:
            if allowed is not None and mapping.ksp_name not in allowed:
                continue
            cfg = _PARTS_JSON.get(mapping.cfg_name)
            if cfg is None:
                continue
            provides = mapping.overrides.get("provides", frozenset())
            if self._matches(mapping.cfg_name, cfg, provides):
                out.add(mapping.ksp_name)
        return frozenset(out)


def _stores_resource(resource: str) -> Callable[[dict], bool]:
    def pred(cfg: dict) -> bool:
        return (cfg.get("resources") or {}).get(resource, 0) > 0
    return pred


# The categories the contract system can require. Add new entries here as new
# contract types arrive (lab, crew_cabin, relay, power, …).
CONTRACT_PART_CATEGORIES: dict[str, PartCategory] = {
    # Curated: parts.json has no harvester module info to derive these.
    "drill": PartCategory(
        "drill", members=frozenset({"MiniDrill", "RadialDrill"}),
        description="ore mining drill"),
    # Metadata-derived: any part that stores Ore (RadialOreTank/Small/LargeTank).
    "ore_tank": PartCategory(
        "ore_tank", predicate=_stores_resource("Ore"),
        description="ore storage tank"),
    # Curated single part — the Mobile Processing Lab (MPL-LG-2).
    "science_lab": PartCategory(
        "science_lab", members=frozenset({"Large_Crewed_Lab"}),
        description="mobile science lab"),
    # Provides-flag derived. Battery via the dedicated battery flags (NOT
    # "stores ElectricCharge" — pods/probes carry EC too). Power = solar or RTG.
    "battery": PartCategory(
        "battery", provides_any=frozenset({"battery_small", "battery_large"}),
        description="rechargeable battery"),
    "power": PartCategory(
        "power", provides_any=frozenset({
            "solar_fixed", "solar_retractable", "solar_array_large", "rtg"}),
        description="power generation (solar/RTG)"),
    "relay": PartCategory(
        "relay", provides_any=frozenset({
            "relay_t1", "relay_t2", "relay_t3", "relay_t4"}),
        description="antenna able to relay home"),
    # Metadata-derived: any part with crew seats (pods/cabins/lab), minus the
    # exposed external command seat — a "station" of lawn chairs is degenerate
    # (and the capability system already excludes it from real capsules).
    "crew_cabin": PartCategory(
        "crew_cabin", predicate=lambda cfg: (cfg.get("crew_capacity") or 0) > 0,
        exclude=frozenset({"seatExternalCmd"}),
        description="crewed pod or cabin (provides seats)"),
    # Curated single part — the stranded Kerbal must jetpack across to the rescue
    # craft, so a crew rescue is impossible without one (they'd float). Promoted
    # to progression per-seed (not chain-guaranteed) so it's a real gate.
    "eva_jetpack": PartCategory(
        "eva_jetpack", members=frozenset({"evaJetpack"}),
        description="EVA jetpack (kerbal orbital maneuvering for rescue)"),
}

# Resolved ksp_name membership per category, computed once over the full part
# universe. The client-facing ``has_any_part`` part list and the capability
# pre-pass both read this. (Pass a narrower ``allowed`` to PartCategory.resolve
# when DLC/part-pool filtering lands.)
CONTRACT_CATEGORY_MEMBERS: dict[str, frozenset[str]] = {
    key: cat.resolve() for key, cat in CONTRACT_PART_CATEGORIES.items()
}

# NOTE: which parts get promoted to progression is decided PER-SEED, not here —
# a category's parts are only promoted if a contract that requires it was
# actually generated (see contracts.required_part_names_for). Promoting drills
# when zero mine contracts exist would needlessly bloat the advancement pool.


def _invert_category_members() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for key, names in CONTRACT_CATEGORY_MEMBERS.items():
        for name in names:
            out.setdefault(name, []).append(key)
    return {name: tuple(keys) for name, keys in out.items()}


# Inverse of CONTRACT_CATEGORY_MEMBERS: ksp_name → the categories it belongs to.
# Lets the capability pre-pass set lightest-per-category in O(1) per part without
# scanning every category.
PART_TO_CONTRACT_CATEGORIES: dict[str, tuple[str, ...]] = _invert_category_members()
