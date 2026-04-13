"""
KSP1 stock tech tree data for Archipelago.

62 purchasable nodes across tiers 1-8 (excluding the free Start node).
Loaded from data/tech_tree.json, which is extracted from the stock
GameData/Squad/Resources/TechTree.cfg by scripts/extract_tech_tree.py.
"""
from __future__ import annotations

import json
import pkgutil
from dataclasses import dataclass


@dataclass(frozen=True)
class TechNode:
    node_id: str            # KSP internal ID (matches TechTree.cfg techID)
    display_name: str       # Human-readable label
    tier: int               # 1-8
    science_cost: int       # Science required to unlock
    parents: tuple[str, ...]    # Parent node IDs (or "start")
    any_to_unlock: bool     # True = only one parent needed, False = all parents


def _load_tech_nodes() -> list[TechNode]:
    """Load tech nodes from the bundled JSON data file."""
    raw = pkgutil.get_data("worlds.ksp1", "data/tech_tree.json")
    assert raw is not None, "data/tech_tree.json not found in package"
    data = json.loads(raw.decode("utf-8"))
    nodes = []
    for entry in data["nodes"]:
        nodes.append(TechNode(
            node_id=entry["id"],
            display_name=entry["title"],
            tier=entry["tier"],
            science_cost=entry["cost"],
            parents=tuple(entry.get("parents", [])),
            any_to_unlock=entry.get("any_to_unlock", False),
        ))
    return nodes


TECH_NODES: list[TechNode] = _load_tech_nodes()

assert len(TECH_NODES) == 62, f"Expected 62 tech nodes, got {len(TECH_NODES)}"

# ---------------------------------------------------------------------------
# Derived lookup structures
# ---------------------------------------------------------------------------

NODES_BY_TIER: dict[int, list[TechNode]] = {}
for _node in TECH_NODES:
    NODES_BY_TIER.setdefault(_node.tier, []).append(_node)

NODE_BY_ID: dict[str, TechNode] = {n.node_id: n for n in TECH_NODES}

MAX_TIER: int = max(n.tier for n in TECH_NODES)

# Cumulative science cost through each tier (inclusive).
# cumulative_tier_cost(T) = sum of all node science_costs for tiers 1..T
_CUMULATIVE: dict[int, int] = {}
_running = 0
for _tier in range(1, MAX_TIER + 1):
    _running += sum(n.science_cost for n in NODES_BY_TIER.get(_tier, []))
    _CUMULATIVE[_tier] = _running

TOTAL_TECH_COST: int = _CUMULATIVE[MAX_TIER]

# Progressive R&D bands — pairs of tiers locked behind the same R&D item.
TIER_TO_BAND: dict[int, int] = {
    1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3,
}
MAX_RD_BAND: int = 3


def cumulative_tier_cost(tier: int) -> int:
    """Return the total science needed to purchase all nodes through *tier*."""
    return _CUMULATIVE[tier]
