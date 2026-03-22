"""
KSP1 stock tech tree data for Archipelago.

43 purchasable nodes across tiers 1-9 (excluding the free Start node).
Science costs are scaled so cumulative costs roughly match:
  - Through tier 3: ~335 science
  - Through tier 5: ~1,883 science
  - Through tier 7: ~7,233 science
  - Through tier 9: ~18,233 science (≈ complete tech tree cost)

Source of truth: GameData/Squad/Resources/TechTree.cfg (verify against this).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechNode:
    node_id: str            # KSP internal ID (matches TechTree.cfg techID)
    display_name: str       # Human-readable label
    tier: int               # 1-9
    science_cost: int       # Science required to unlock


# ---------------------------------------------------------------------------
# Node definitions — 43 nodes across tiers 1-9
# ---------------------------------------------------------------------------

TECH_NODES: list[TechNode] = [
    # -----------------------------------------------------------------------
    # Tier 1  (1 node, cumulative: 5)
    # -----------------------------------------------------------------------
    TechNode("basicRocketry", "Basic Rocketry", tier=1, science_cost=5),

    # -----------------------------------------------------------------------
    # Tier 2  (3 nodes, cumulative: 65)
    # -----------------------------------------------------------------------
    TechNode("generalRocketry", "General Rocketry", tier=2, science_cost=20),
    TechNode("survivability", "Survivability", tier=2, science_cost=20),
    TechNode("stability", "Stability", tier=2, science_cost=20),

    # -----------------------------------------------------------------------
    # Tier 3  (5 nodes, cumulative: 335)
    # -----------------------------------------------------------------------
    TechNode("advRocketry", "Advanced Rocketry", tier=3, science_cost=45),
    TechNode("spaceExploration", "Space Exploration", tier=3, science_cost=45),
    TechNode("advConstruction", "Advanced Construction", tier=3, science_cost=45),
    TechNode("fieldScience", "Field Science", tier=3, science_cost=90),
    TechNode("basicScience", "Basic Science", tier=3, science_cost=45),

    # -----------------------------------------------------------------------
    # Tier 4  (6 nodes, cumulative: 835)
    # -----------------------------------------------------------------------
    TechNode("propulsionSystems", "Propulsion Systems", tier=4, science_cost=80),
    TechNode("advExploration", "Advanced Exploration", tier=4, science_cost=80),
    TechNode("landing", "Landing", tier=4, science_cost=60),
    TechNode("advAerodynamics", "Advanced Aerodynamics", tier=4, science_cost=100),
    TechNode("scienceTech", "Science Tech", tier=4, science_cost=120),
    TechNode("generalConstruction", "General Construction", tier=4, science_cost=60),

    # -----------------------------------------------------------------------
    # Tier 5  (7 nodes, cumulative: 1883)
    # -----------------------------------------------------------------------
    TechNode("heavyRocketry", "Heavy Rocketry", tier=5, science_cost=130),
    TechNode("highAltitudeFlight", "High Altitude Flight", tier=5, science_cost=130),
    TechNode("advLanding", "Advanced Landing", tier=5, science_cost=150),
    TechNode("actuators", "Actuators", tier=5, science_cost=180),
    TechNode("electronics", "Electronics", tier=5, science_cost=180),
    TechNode("ionPropulsion", "Ion Propulsion", tier=5, science_cost=150),
    TechNode("precisionEngineering", "Precision Engineering", tier=5, science_cost=128),

    # -----------------------------------------------------------------------
    # Tier 6  (8 nodes, cumulative: 4033)
    # -----------------------------------------------------------------------
    TechNode("heavierRocketry", "Heavier Rocketry", tier=6, science_cost=200),
    TechNode("nuclearPropulsion", "Nuclear Propulsion", tier=6, science_cost=600),
    TechNode("specializedControl", "Specialized Control", tier=6, science_cost=300),
    TechNode("unmannedTech", "Unmanned Tech", tier=6, science_cost=200),
    TechNode("advScienceTech", "Advanced Science Tech", tier=6, science_cost=250),
    TechNode("specializedConstruction", "Specialized Construction", tier=6, science_cost=200),
    TechNode("largeElectrics", "Large Electrics", tier=6, science_cost=200),
    TechNode("composites", "Composites", tier=6, science_cost=200),

    # -----------------------------------------------------------------------
    # Tier 7  (6 nodes, cumulative: 7233)
    # -----------------------------------------------------------------------
    TechNode("veryHeavyRocketry", "Very Heavy Rocketry", tier=7, science_cost=550),
    TechNode("experimentalElectrics", "Experimental Electrics", tier=7, science_cost=600),
    TechNode("highPerformanceFuelSystems", "High Performance Fuel Systems", tier=7, science_cost=600),
    TechNode("advUnmannedTech", "Advanced Unmanned Tech", tier=7, science_cost=600),
    TechNode("robotics", "Robotics", tier=7, science_cost=500),
    TechNode("experimentalMotors", "Experimental Motors", tier=7, science_cost=350),

    # -----------------------------------------------------------------------
    # Tier 8  (5 nodes, cumulative: 15733)
    # -----------------------------------------------------------------------
    TechNode("aerospaceComposites", "Aerospace Composites", tier=8, science_cost=1800),
    TechNode("fieldResearch", "Field Research", tier=8, science_cost=1800),
    TechNode("nanolathing", "Nanolathing", tier=8, science_cost=1700),
    TechNode("advancedMotors", "Advanced Motors", tier=8, science_cost=1600),
    TechNode("highPerformanceSystems", "High Performance Systems", tier=8, science_cost=1600),

    # -----------------------------------------------------------------------
    # Tier 9  (2 nodes, cumulative: 18233)
    # -----------------------------------------------------------------------
    TechNode("metaMaterials", "Meta-Materials", tier=9, science_cost=1400),
    TechNode("ultimateRocketry", "Ultimate Rocketry", tier=9, science_cost=1100),
]

assert len(TECH_NODES) == 43, f"Expected 43 tech nodes, got {len(TECH_NODES)}"

# ---------------------------------------------------------------------------
# Derived lookup structures
# ---------------------------------------------------------------------------

NODES_BY_TIER: dict[int, list[TechNode]] = {}
for _node in TECH_NODES:
    NODES_BY_TIER.setdefault(_node.tier, []).append(_node)

NODE_BY_ID: dict[str, TechNode] = {n.node_id: n for n in TECH_NODES}

# Cumulative science cost through each tier (inclusive).
# cumulative_tier_cost(T) = sum of all node science_costs for tiers 1..T
_CUMULATIVE: dict[int, int] = {}
_running = 0
for _tier in range(1, 10):
    _running += sum(n.science_cost for n in NODES_BY_TIER.get(_tier, []))
    _CUMULATIVE[_tier] = _running

TOTAL_TECH_COST: int = _CUMULATIVE[9]  # ~18,233


def cumulative_tier_cost(tier: int) -> int:
    """Return the total science needed to purchase all nodes through *tier*."""
    return _CUMULATIVE[tier]


# Location names for tech tree slots: "{display_name} {slot}" for slots 1-5.
TECH_TREE_LOCATION_NAMES: list[str] = [
    f"{node.display_name} {slot}"
    for node in TECH_NODES
    for slot in range(1, 6)
]

assert len(TECH_TREE_LOCATION_NAMES) == 215  # 43 nodes × 5 slots
