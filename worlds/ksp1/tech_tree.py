"""
KSP1 stock tech tree data for Archipelago.

43 purchasable nodes across tiers 1-9 (excluding the free Start node).
Science costs are scaled so cumulative costs roughly match:
  - Through tier 3: ~335 science
  - Through tier 5: ~1,883 science
  - Through tier 7: ~7,233 science
  - Through tier 9: ~18,233 science (≈ complete tech tree cost)

Part names reference PART_DB keys in parts.py.  Nodes without entries in
the current PART_DB still exist as locations; their part list will fill in
as PART_DB grows toward the full stock part list.

Source of truth: GameData/Squad/Resources/TechTree.cfg (verify against this).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TechNode:
    node_id: str            # KSP internal ID (matches TechTree.cfg techID)
    display_name: str       # Human-readable label
    tier: int               # 1-9
    science_cost: int       # Science required to unlock
    parts: tuple[str, ...]  # AP item names (PART_DB keys) living in this node


# ---------------------------------------------------------------------------
# Node definitions — 43 nodes across tiers 1-9
# ---------------------------------------------------------------------------

TECH_NODES: list[TechNode] = [
    # -----------------------------------------------------------------------
    # Tier 1  (1 node, cumulative: 5)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="basicRocketry",
        display_name="Basic Rocketry",
        tier=1, science_cost=5,
        parts=(
            "Command Pod",
            "Reliant Engine",
            "FL-T400 Tank",
            "Mk16 Parachute",
            "TR-18A Decoupler",
        ),
    ),

    # -----------------------------------------------------------------------
    # Tier 2  (3 nodes, cumulative: 65)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="generalRocketry",
        display_name="General Rocketry",
        tier=2, science_cost=20,
        parts=(
            "Oscar-B Tank",
            "FL-T800 Tank",
            "Hammer SRB",
            "TT-38K Radial Decoupler",
        ),
    ),
    TechNode(
        node_id="survivability",
        display_name="Survivability",
        tier=2, science_cost=20,
        parts=(
            "LT-1 Landing Legs",
            "Mk2-R Drogue",
        ),
    ),
    TechNode(
        node_id="stability",
        display_name="Stability",
        tier=2, science_cost=20,
        parts=(
            "OX-STAT Solar",
            "Communotron 16",
        ),
    ),

    # -----------------------------------------------------------------------
    # Tier 3  (5 nodes, cumulative: 335)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="advRocketry",
        display_name="Advanced Rocketry",
        tier=3, science_cost=45,
        parts=(
            "Swivel Engine",
            "FL-R10 Monoprop Tank",
            "RCS Thruster",
        ),
    ),
    TechNode(
        node_id="spaceExploration",
        display_name="Space Exploration",
        tier=3, science_cost=45,
        parts=(
            "Probe Core",
            "OX-4 Solar",
            "Z-4K Battery",
        ),
    ),
    TechNode(
        node_id="advConstruction",
        display_name="Advanced Construction",
        tier=3, science_cost=45,
        parts=(
            "FTX-2 Fuel Line",
            "Crew Ladder",
        ),
    ),
    TechNode(
        node_id="fieldScience",
        display_name="Field Science",
        tier=3, science_cost=90,
        parts=(
            "Probodobodyne OKTO2",
            "HG-5 Relay",
            "Thermometer",
            "Barometer",
        ),
    ),
    TechNode(
        node_id="basicScience",
        display_name="Basic Science",
        tier=3, science_cost=45,
        parts=(
            "Reaction Wheel",
            "Struts",
        ),
    ),

    # -----------------------------------------------------------------------
    # Tier 4  (6 nodes, cumulative: 835)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="propulsionSystems",
        display_name="Propulsion Systems",
        tier=4, science_cost=80,
        parts=(
            "Terrier Engine",
            "Rockomax X200-32",
            "Docking Port",
        ),
    ),
    TechNode(
        node_id="advExploration",
        display_name="Advanced Exploration",
        tier=4, science_cost=80,
        parts=(
            "1.25m Heat Shield",
        ),
    ),
    TechNode(
        node_id="landing",
        display_name="Landing",
        tier=4, science_cost=60,
        parts=(
            "LT-2 Landing Strut",
        ),
    ),
    TechNode(
        node_id="advAerodynamics",
        display_name="Advanced Aerodynamics",
        tier=4, science_cost=100,
        parts=(),
    ),
    TechNode(
        node_id="scienceTech",
        display_name="Science Tech",
        tier=4, science_cost=120,
        parts=(),
    ),
    TechNode(
        node_id="generalConstruction",
        display_name="General Construction",
        tier=4, science_cost=60,
        parts=(),
    ),

    # -----------------------------------------------------------------------
    # Tier 5  (7 nodes, cumulative: 1883)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="heavyRocketry",
        display_name="Heavy Rocketry",
        tier=5, science_cost=130,
        parts=(
            "Poodle Engine",
            "Rockomax Jumbo-64",
        ),
    ),
    TechNode(
        node_id="highAltitudeFlight",
        display_name="High Altitude Flight",
        tier=5, science_cost=130,
        parts=(),
    ),
    TechNode(
        node_id="advLanding",
        display_name="Advanced Landing",
        tier=5, science_cost=150,
        parts=(),
    ),
    TechNode(
        node_id="actuators",
        display_name="Actuators",
        tier=5, science_cost=180,
        parts=(),
    ),
    TechNode(
        node_id="electronics",
        display_name="Electronics",
        tier=5, science_cost=180,
        parts=(),
    ),
    TechNode(
        node_id="ionPropulsion",
        display_name="Ion Propulsion",
        tier=5, science_cost=150,
        parts=(
            "Dawn Ion Engine",
            "PB-X50R Xenon Tank",
            "Gigantor Solar Array",
        ),
    ),
    TechNode(
        node_id="precisionEngineering",
        display_name="Precision Engineering",
        tier=5, science_cost=128,
        parts=(
            "2.5m Heat Shield",
        ),
    ),

    # -----------------------------------------------------------------------
    # Tier 6  (8 nodes, cumulative: 4033)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="heavierRocketry",
        display_name="Heavier Rocketry",
        tier=6, science_cost=200,
        parts=(
            "Mainsail Engine",
        ),
    ),
    TechNode(
        node_id="nuclearPropulsion",
        display_name="Nuclear Propulsion",
        tier=6, science_cost=600,
        parts=(
            "Nerv Engine",
            "Mk1 LF Tank",
        ),
    ),
    TechNode(
        node_id="specializedControl",
        display_name="Specialized Control",
        tier=6, science_cost=300,
        parts=(),
    ),
    TechNode(
        node_id="unmannedTech",
        display_name="Unmanned Tech",
        tier=6, science_cost=200,
        parts=(
            "RA-2 Relay",
            "RTG",
        ),
    ),
    TechNode(
        node_id="advScienceTech",
        display_name="Advanced Science Tech",
        tier=6, science_cost=250,
        parts=(),
    ),
    TechNode(
        node_id="specializedConstruction",
        display_name="Specialized Construction",
        tier=6, science_cost=200,
        parts=(),
    ),
    TechNode(
        node_id="largeElectrics",
        display_name="Large Electrics",
        tier=6, science_cost=200,
        parts=(),
    ),
    TechNode(
        node_id="composites",
        display_name="Composites",
        tier=6, science_cost=200,
        parts=(),
    ),

    # -----------------------------------------------------------------------
    # Tier 7  (6 nodes, cumulative: 7233)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="veryHeavyRocketry",
        display_name="Very Heavy Rocketry",
        tier=7, science_cost=550,
        parts=(
            "Rhino Engine",
            "Kerbodyne S3-3600",
        ),
    ),
    TechNode(
        node_id="experimentalElectrics",
        display_name="Experimental Electrics",
        tier=7, science_cost=600,
        parts=(),
    ),
    TechNode(
        node_id="highPerformanceFuelSystems",
        display_name="High Performance Fuel Systems",
        tier=7, science_cost=600,
        parts=(),
    ),
    TechNode(
        node_id="advUnmannedTech",
        display_name="Advanced Unmanned Tech",
        tier=7, science_cost=600,
        parts=(),
    ),
    TechNode(
        node_id="robotics",
        display_name="Robotics",
        tier=7, science_cost=500,
        parts=(),
    ),
    TechNode(
        node_id="experimentalMotors",
        display_name="Experimental Motors",
        tier=7, science_cost=350,
        parts=(),
    ),

    # -----------------------------------------------------------------------
    # Tier 8  (5 nodes, cumulative: 15733)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="aerospaceComposites",
        display_name="Aerospace Composites",
        tier=8, science_cost=1800,
        parts=(
            "Mammoth Engine",
            "3.75m Heat Shield",
        ),
    ),
    TechNode(
        node_id="fieldResearch",
        display_name="Field Research",
        tier=8, science_cost=1800,
        parts=(),
    ),
    TechNode(
        node_id="nanolathing",
        display_name="Nanolathing",
        tier=8, science_cost=1700,
        parts=(),
    ),
    TechNode(
        node_id="advancedMotors",
        display_name="Advanced Motors",
        tier=8, science_cost=1600,
        parts=(),
    ),
    TechNode(
        node_id="highPerformanceSystems",
        display_name="High Performance Systems",
        tier=8, science_cost=1600,
        parts=(),
    ),

    # -----------------------------------------------------------------------
    # Tier 9  (2 nodes, cumulative: 18233)
    # -----------------------------------------------------------------------
    TechNode(
        node_id="metaMaterials",
        display_name="Meta-Materials",
        tier=9, science_cost=1400,
        parts=(),
    ),
    TechNode(
        node_id="ultimateRocketry",
        display_name="Ultimate Rocketry",
        tier=9, science_cost=1100,
        parts=(),
    ),
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
