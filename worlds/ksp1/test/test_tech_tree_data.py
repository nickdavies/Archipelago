"""
Validation tests for the tech tree data pipeline.

Ensures tech_tree.json and TECH_NODES stay in sync with the stock KSP TechTree.cfg.
"""
import json
import pkgutil
import unittest

from worlds.ksp1.tech_tree import (
    MAX_TIER,
    NODES_BY_TIER,
    NODE_BY_ID,
    TECH_NODES,
    TOTAL_TECH_COST,
    TechNode,
    cumulative_tier_cost,
)


def _load_json() -> dict:
    raw = pkgutil.get_data("worlds.ksp1", "data/tech_tree.json")
    assert raw is not None
    return json.loads(raw.decode("utf-8"))


# IDs that were fabricated in the old hardcoded tech_tree.py (never existed in stock).
_OLD_FABRICATED_IDS = frozenset({
    "advUnmannedTech", "robotics", "aerospaceComposites",
    "fieldResearch", "highPerformanceSystems", "ultimateRocketry",
})


class TestJsonPythonConsistency(unittest.TestCase):
    """JSON data and Python TECH_NODES must agree on everything."""

    def test_node_count(self) -> None:
        data = _load_json()
        self.assertEqual(data["_meta"]["node_count"], 62)
        self.assertEqual(len(data["nodes"]), 62)
        self.assertEqual(len(TECH_NODES), 62)

    def test_ids_match(self) -> None:
        data = _load_json()
        json_ids = [n["id"] for n in data["nodes"]]
        py_ids = [n.node_id for n in TECH_NODES]
        self.assertEqual(json_ids, py_ids)

    def test_names_match(self) -> None:
        data = _load_json()
        for entry in data["nodes"]:
            node = NODE_BY_ID[entry["id"]]
            self.assertEqual(
                node.display_name, entry["title"],
                f"Display name mismatch for {entry['id']}",
            )

    def test_costs_match(self) -> None:
        data = _load_json()
        for entry in data["nodes"]:
            node = NODE_BY_ID[entry["id"]]
            self.assertEqual(
                node.science_cost, entry["cost"],
                f"Cost mismatch for {entry['id']}",
            )

    def test_tiers_match(self) -> None:
        data = _load_json()
        for entry in data["nodes"]:
            node = NODE_BY_ID[entry["id"]]
            self.assertEqual(
                node.tier, entry["tier"],
                f"Tier mismatch for {entry['id']}",
            )


class TestNoFabricatedIds(unittest.TestCase):
    """None of the old fabricated IDs should exist in the new data."""

    def test_no_fabricated_ids(self) -> None:
        current_ids = {n.node_id for n in TECH_NODES}
        for bad_id in _OLD_FABRICATED_IDS:
            self.assertNotIn(
                bad_id, current_ids,
                f"Fabricated ID '{bad_id}' should not be in TECH_NODES",
            )


class TestTierStructure(unittest.TestCase):
    """Verify tier boundaries and node counts per tier."""

    def test_max_tier(self) -> None:
        self.assertEqual(MAX_TIER, 8)

    def test_tiers_are_1_through_8(self) -> None:
        self.assertEqual(set(NODES_BY_TIER.keys()), {1, 2, 3, 4, 5, 6, 7, 8})

    def test_node_counts_per_tier(self) -> None:
        expected = {1: 2, 2: 3, 3: 5, 4: 10, 5: 13, 6: 12, 7: 12, 8: 5}
        for tier, count in expected.items():
            self.assertEqual(
                len(NODES_BY_TIER[tier]), count,
                f"Tier {tier}: expected {count} nodes, got {len(NODES_BY_TIER[tier])}",
            )

    def test_total_matches_sum(self) -> None:
        total = sum(1 for _ in TECH_NODES)
        self.assertEqual(total, 62)
        tier_total = sum(len(nodes) for nodes in NODES_BY_TIER.values())
        self.assertEqual(tier_total, 62)


class TestCumulativeCosts(unittest.TestCase):
    """Cumulative costs must be monotonically increasing and correct."""

    def test_monotonically_increasing(self) -> None:
        prev = 0
        for tier in range(1, MAX_TIER + 1):
            cost = cumulative_tier_cost(tier)
            self.assertGreater(
                cost, prev,
                f"Cumulative cost for tier {tier} ({cost}) not > tier {tier-1} ({prev})",
            )
            prev = cost

    def test_total_tech_cost(self) -> None:
        expected = sum(n.science_cost for n in TECH_NODES)
        self.assertEqual(TOTAL_TECH_COST, expected)
        self.assertEqual(TOTAL_TECH_COST, cumulative_tier_cost(MAX_TIER))


class TestParentReferences(unittest.TestCase):
    """All parent IDs must reference valid nodes or 'start'."""

    def test_all_parents_valid(self) -> None:
        valid_ids = {n.node_id for n in TECH_NODES} | {"start"}
        for node in TECH_NODES:
            for parent_id in node.parents:
                self.assertIn(
                    parent_id, valid_ids,
                    f"Node '{node.node_id}' references unknown parent '{parent_id}'",
                )

    def test_all_nodes_have_parents(self) -> None:
        """Every node must have at least one parent."""
        for node in TECH_NODES:
            self.assertGreater(
                len(node.parents), 0,
                f"Node '{node.node_id}' has no parents",
            )



class TestLocationNames(unittest.TestCase):
    """Tech tree location names must be consistent."""

    def test_location_count(self) -> None:
        from worlds.ksp1.locations import MAX_TECH_SLOTS, TECH_TREE_LOCATION_NAMES
        self.assertEqual(MAX_TECH_SLOTS, 4)
        self.assertEqual(len(TECH_TREE_LOCATION_NAMES), 62 * 4)

    def test_location_ids_in_range(self) -> None:
        from worlds.ksp1.locations import LOCATION_TABLE
        tech_locs = {
            name: offset for name, offset in LOCATION_TABLE.items()
            if offset >= 3000 and offset < 4000
        }
        self.assertEqual(len(tech_locs), 62 * 4)


if __name__ == "__main__":
    unittest.main()
