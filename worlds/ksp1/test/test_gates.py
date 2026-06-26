"""Unit tests for the unified accumulation-gate access face (`gates.py`).

The gate's runtime rule is the single predicate both the science threshold and
the contract-completion threshold use: "does the player's current supply meet
the amount?". These pin that semantics (including the integer-count case, which
must stay equivalent to the old ``state.has(event, count)`` check).
"""

import unittest

from ..gates import AccumulationGate, Resource


class _FakeState:
    """Stand-in for CollectionState — the gate only ever calls the injected
    measure, never the state directly, so any object works."""

    def __init__(self, supply):
        self.supply = supply


class TestAccumulationGate(unittest.TestCase):
    def test_resources_present(self):
        self.assertEqual(
            {r.value for r in Resource},
            {"science", "contract_completion"},
        )

    def test_rule_passes_at_or_above_amount(self):
        gate = AccumulationGate(Resource.SCIENCE, 100.0)
        rule = gate.runtime_rule(lambda s: s.supply)
        self.assertFalse(rule(_FakeState(99.9)))
        self.assertTrue(rule(_FakeState(100.0)))
        self.assertTrue(rule(_FakeState(250.0)))

    def test_contract_count_equivalent_to_has_count(self):
        # For an integer resource, `measure(state) >= amount` must match the
        # old `state.has(event, amount)` semantics exactly.
        gate = AccumulationGate(Resource.CONTRACT_COMPLETION, 3)
        rule = gate.runtime_rule(lambda s: s.supply)
        for have in range(6):
            self.assertEqual(rule(_FakeState(have)), have >= 3)

    def test_zero_amount_always_satisfied(self):
        rule = AccumulationGate(Resource.CONTRACT_COMPLETION, 0).runtime_rule(
            lambda s: s.supply
        )
        self.assertTrue(rule(_FakeState(0)))

    def test_gate_is_frozen_hashable(self):
        g1 = AccumulationGate(Resource.SCIENCE, 100.0)
        g2 = AccumulationGate(Resource.SCIENCE, 100.0)
        self.assertEqual(g1, g2)
        self.assertEqual(hash(g1), hash(g2))


if __name__ == "__main__":
    unittest.main()
