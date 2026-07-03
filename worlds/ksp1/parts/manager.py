"""PartManager — the pack-aware source of truth for the parts a world may use.

The World constructs one (like ``MissionBuilder``), passing the set of enabled
optional packs; Stock is always included. Every part-derived structure the
generator needs — the part map, the lightest part providing a capability, the
contract-category membership — is served from here, filtered to the enabled
packs, so a disabled pack's parts are absent everywhere consistently (one
source of truth, no second filter to keep in sync).

Instances are immutable and memoized per enabled-pack set (``part_manager_for``),
so the heavy derivations compute once per process for the common all-default
sweep, preserving the import-time-cache performance of the pre-refactor globals.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from . import packs
from .types import AnyPart, CapabilityFlag, Decoupler, FuelTank, MiscEquipment
from ._raw import _RAW_PART_DB, _RAW_PART_PACK
from .categories import CONTRACT_PART_CATEGORIES

# The full installed universe — every extracted pack. Phase-1 call sites that
# don't yet receive a world's enabled set default to this (so behavior is
# identical to the pre-refactor global PART_DB).
ALL_PACKS: frozenset[str] = frozenset(_RAW_PART_PACK.values()) | {packs.STOCK}


class PartManager:
    """Pack-filtered view over the raw part universe. Construct with the set of
    enabled optional packs (Stock is added automatically)."""

    def __init__(self, enabled_packs: frozenset[str]) -> None:
        self.enabled_packs: frozenset[str] = (
            frozenset(enabled_packs) | {packs.STOCK})
        # Filtered part map; preserves _RAW_PART_DB (registry) iteration order so
        # order-sensitive "first match" selections are unchanged when all packs
        # are enabled.
        self.parts: dict[str, list[AnyPart]] = {
            nm: ps for nm, ps in _RAW_PART_DB.items()
            if _RAW_PART_PACK.get(nm, packs.STOCK) in self.enabled_packs
        }
        self._lightest: dict[CapabilityFlag, Optional[str]] = {}
        self._fuel_line: Optional[tuple[Optional[str], float]] = None
        self._category_members: Optional[dict[str, frozenset[str]]] = None
        self._part_to_categories: Optional[dict[str, tuple[str, ...]]] = None

    # --- part-of-a-capability selection ------------------------------------
    # (mirrors the old module-level capability globals, now pack-aware)

    def lightest_providing(self, flag: CapabilityFlag) -> Optional[str]:
        """Item name of the lightest part (by that part's own mass) that
        provides ``flag`` among enabled parts; None if none."""
        if flag not in self._lightest:
            best: Optional[tuple[str, float]] = None
            for nm, parts in self.parts.items():
                for p in parts:
                    if flag in getattr(p, "provides", ()):
                        if best is None or p.mass < best[1]:
                            best = (nm, p.mass)
                        break
            self._lightest[flag] = best[0] if best else None
        return self._lightest[flag]

    def lightest_decoupler(self, kind: str) -> Optional[str]:
        """Item name of the lightest decoupler of ``kind`` ("stack"/"radial")
        among enabled parts; None if none."""
        best: Optional[tuple[str, float]] = None
        for nm, parts in self.parts.items():
            for p in parts:
                if isinstance(p, Decoupler) and p.kind == kind:
                    if best is None or p.mass < best[1]:
                        best = (nm, p.mass)
        return best[0] if best else None

    def docking_gear_candidates(self) -> dict[str, frozenset[str]]:
        """Per-role candidate part names for the Apollo docking gear under
        these packs: every docking port, RCS thruster, and monoprop tank.

        No command-part role: the mission's own kit always carries one
        (capsule for crewed, probe core for uncrewed) and the parked stack
        reuses it — capability's ``_apollo_split_for`` flies the lightest
        the player owns.  Consumers pick ONE candidate per role per seed
        (variance — no part is hardcoded into every run); capability then
        accepts whichever suitable parts are actually collected.
        """
        def _providing(flag: CapabilityFlag) -> frozenset[str]:
            return frozenset(
                nm for nm, parts in self.parts.items()
                if any(flag in getattr(p, "provides", ()) for p in parts))

        return {
            "docking_port": _providing(CapabilityFlag.DOCKING_PORT),
            "rcs_thruster": _providing(CapabilityFlag.RCS),
            "monoprop_tank": frozenset(
                nm for nm, parts in self.parts.items()
                if any(isinstance(p, FuelTank) and p.fuel_type == "monoprop"
                       for p in parts)),
        }

    def _fuel_line_info(self) -> tuple[Optional[str], float]:
        if self._fuel_line is None:
            part: Optional[str] = None
            mass = 0.0
            for nm, parts in self.parts.items():
                fl = next((p for p in parts if isinstance(p, MiscEquipment)
                           and CapabilityFlag.FUEL_LINE in p.provides), None)
                if fl is not None:
                    part, mass = nm, fl.mass
                    break
            self._fuel_line = (part, mass)
        return self._fuel_line

    @property
    def fuel_line_part(self) -> Optional[str]:
        """First enabled item (registry order) providing FUEL_LINE — the
        asparagus enabler — or None."""
        return self._fuel_line_info()[0]

    @property
    def fuel_line_mass(self) -> float:
        return self._fuel_line_info()[1]

    # --- contract part categories ------------------------------------------

    @property
    def category_members(self) -> dict[str, frozenset[str]]:
        """Contract-category key -> the enabled ksp_names that satisfy it."""
        if self._category_members is None:
            allowed = frozenset(self.parts)
            self._category_members = {
                key: cat.resolve(allowed=allowed)
                for key, cat in CONTRACT_PART_CATEGORIES.items()
            }
        return self._category_members

    @property
    def part_to_categories(self) -> dict[str, tuple[str, ...]]:
        """Inverse of ``category_members``: ksp_name -> categories it belongs
        to, in category-declaration order (matches the legacy inversion)."""
        if self._part_to_categories is None:
            out: dict[str, list[str]] = {}
            for key, names in self.category_members.items():
                for nm in names:
                    out.setdefault(nm, []).append(key)
            self._part_to_categories = {
                nm: tuple(keys) for nm, keys in out.items()}
        return self._part_to_categories

    # --- feasibility axis (consumed in Phase 2) ----------------------------

    def capability_relevant_packs(self) -> frozenset[str]:
        """Enabled packs that contribute a part able to change the dv / landing
        feasibility model (propulsion + descent hardware)."""
        out: set[str] = set()
        for nm, parts in self.parts.items():
            if any(type(p).__name__ in packs.CAPABILITY_RELEVANT_TYPE_NAMES
                   for p in parts):
                out.add(_RAW_PART_PACK.get(nm, packs.STOCK))
        return frozenset(out)


@lru_cache(maxsize=None)
def part_manager_for(enabled_packs: frozenset[str]) -> PartManager:
    """Memoized PartManager per enabled-pack set (process scope). The common
    all-default sweep builds (and derives from) it once."""
    return PartManager(enabled_packs)


# Default manager over the full installed universe — for Phase-1 call sites that
# don't yet receive a world's enabled set.
DEFAULT_PART_MANAGER: PartManager = part_manager_for(ALL_PACKS)
