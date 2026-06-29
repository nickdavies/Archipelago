"""Part-pack identity model — the single source of truth for which packs exist,
which are enabled by default, and how a GameData directory maps to a pack id.

This module is a LEAF: it imports nothing from the rest of the project (not even
the sibling part dataclasses), so the standalone parts extractor can load it by
file path without dragging in the part database. ``manager.py`` applies
``CAPABILITY_RELEVANT_TYPE_NAMES`` to concrete part objects.
"""
from __future__ import annotations

from typing import Optional

# --- Pack ids --------------------------------------------------------------

STOCK = "Stock"
MAKING_HISTORY = "MakingHistory"
BREAKING_GROUND = "BreakingGround"

# Stock is ALWAYS enabled and is never an optional toggle. ``OPTIONAL_PACKS`` is
# what a player can turn on/off; BREAKING_GROUND is intentionally absent for now
# (its parts aren't extracted and would only add filler — see
# plans/optional_part_packs.md).
OPTIONAL_PACKS: tuple[str, ...] = (MAKING_HISTORY,)

# What the player gets if they don't override the option.
DEFAULT_ENABLED_OPTIONAL_PACKS: frozenset[str] = frozenset({MAKING_HISTORY})


# --- GameData directory -> pack id ----------------------------------------
# Explicit mapping so the extractor can never silently mislabel or drop a pack:
# a part-containing GameData root not covered here is a HARD ERROR (the operator
# must add a mapping or explicitly --exclude it). Keys are GameData-relative,
# forward-slash paths.
KNOWN_PACK_ROOTS: dict[str, str] = {
    "Squad": STOCK,
    "SquadExpansion/MakingHistory": MAKING_HISTORY,
    "SquadExpansion/Serenity": BREAKING_GROUND,
}


def pack_for_gamedata_dir(rel_dir: str) -> Optional[str]:
    """Pack id for a GameData-relative directory, or None if no known root
    matches (the caller decides whether that's a hard error). Longest matching
    root wins so ``SquadExpansion/MakingHistory`` beats a hypothetical
    ``SquadExpansion`` entry."""
    rel = rel_dir.replace("\\", "/").strip("/")
    best_root: Optional[str] = None
    best_pack: Optional[str] = None
    for root, pack in KNOWN_PACK_ROOTS.items():
        if rel == root or rel.startswith(root + "/"):
            if best_root is None or len(root) > len(best_root):
                best_root, best_pack = root, pack
    return best_pack


# --- Capability relevance --------------------------------------------------
# Part-type names whose presence lets a pack change the dv / landing feasibility
# model (propulsion + descent hardware). A pack contributing only filler /
# cosmetic / robotic parts is NOT a feasibility axis. ``manager.py`` checks
# ``type(part).__name__`` against this set. Kept as names (not classes) to keep
# this module a dependency-free leaf.
CAPABILITY_RELEVANT_TYPE_NAMES: frozenset[str] = frozenset({
    "Engine", "FuelTank", "SolidBooster", "HeatShield",
    "Parachute", "LandingLeg", "Decoupler",
})
