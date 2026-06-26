"""Regression: every item an access rule gates on must stay PROGRESSION.

A location's access rule that gates on ``state.has(X)`` is only honoured by
AP's advancement-only beatability sweep if X is PROGRESSION — a demoted
(USEFUL/FILLER) gate item is never collected, so the location is
unreachable-in-logic and any PROGRESSION item fill placed there strands.

This bit us repeatedly: the sphere-ladder classification pass demoted non-goal
contract gate items to USEFUL in ``findable`` mode (the ``count`` /
``progressive_unlock`` keep-list never covered ``findable``), so a Progressive
Launch Pad copy landing on a non-goal contract location stranded — capping the
pad tier and making deep Laythe sample-returns unsolvable.

The fix records every gated item via the ``rules.require_item(s)`` chokepoint
(``world.logic_required_items``) and keeps them PROGRESSION; the
``_assert_gate_items_progression`` backstop fails generation otherwise. These
tests exercise that invariant across modes/homes so it can't silently regress.
"""
import unittest

from BaseClasses import ItemClassification

from worlds.ksp1.test.base import KSP1TestBase


class _GateItemInvariantMixin:
    """Assert no logic-gated pooled item was demoted below PROGRESSION."""

    needs_real_pre_fill = True  # classification happens in apply_sphere_ladder

    def test_no_gate_item_demoted_below_progression(self):
        required = getattr(self.world, "logic_required_items", set())
        self.assertTrue(
            required,
            "expected some logic-required gate items (contracts present)",
        )
        demoted = sorted(
            it.name for it in self.multiworld.itempool
            if it.player == self.player and it.name in required
            and not (it.classification & ItemClassification.progression)
        )
        self.assertEqual(
            demoted, [],
            f"access rules gate on these items but they were demoted below "
            f"PROGRESSION (fill can strand progression behind them): {demoted}",
        )


class TestGateItemsLaytheFindable(_GateItemInvariantMixin, KSP1TestBase):
    """The original failure: Laythe-home standard_sample_returns, findable mode.
    Non-goal contract gate items were demoted, stranding a Progressive Launch
    Pad copy on a contract location."""
    options = {
        "goal": "standard_sample_returns",
        "starting_body": "laythe",
        "goal_contract_mode": "findable",
    }


class TestGateItemsKerbinFindable(_GateItemInvariantMixin, KSP1TestBase):
    """Same invariant on the common Kerbin home — the demotion is mode-driven,
    not home-driven, so it must hold here too."""
    options = {
        "goal": "standard_sample_returns",
        "starting_body": "kerbin",
        "goal_contract_mode": "findable",
    }


if __name__ == "__main__":
    unittest.main()
