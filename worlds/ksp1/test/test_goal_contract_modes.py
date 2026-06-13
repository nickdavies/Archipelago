"""Goal contract mode tests (findable / starting / count / progressive_unlock).

The fill-based classes (run_default_tests=True) inherit WorldTestBase's
all-state-reachable + beatable checks; the custom methods assert the mode's
structural invariants (goal items out of pool, thresholds pre-filled, balance).
Tech-tree-goal classes skip beatability (the tech-tree goal is solve-rate exempt)
and only assert the Progressive-R&D distribution.
"""
import unittest

from Options import OptionError
from test.general import gen_steps, setup_multiworld

from worlds.ksp1.world import KSP1World
from worlds.ksp1.items import PROGRESSIVE_RD_NAME, PROGRESSIVE_RD_COUNT
from worlds.ksp1.test.base import KSP1TestBase


def _unfilled_real_locations(mw, player):
    return [
        loc for loc in mw.get_locations(player)
        if loc.address is not None and loc.item is None
    ]


class _ModeChecks:
    """Mixin (NOT collected on its own — no Test prefix, not a TestCase) adding
    structural assertions on top of the WorldTestBase fill + beatability suite."""
    run_default_tests = True
    needs_real_pre_fill = True

    def test_goal_items_not_in_pool(self):
        mode = self.options.get("goal_contract_mode")
        if mode not in ("count", "progressive_unlock", "starting"):
            return
        goal_items = {s.item_name for s in self.world.goal_contract_specs}
        pool = {i.name for i in self.multiworld.itempool}
        self.assertEqual(
            goal_items & pool, set(),
            f"goal contract items leaked into the pool in {mode} mode",
        )

    def test_pool_balances_unfilled_locations(self):
        unfilled = _unfilled_real_locations(self.multiworld, self.player)
        self.assertEqual(len(self.multiworld.itempool), len(unfilled))

    def test_thresholds_prefilled_with_correct_items(self):
        for loc_name, _count, item_name in self.world.contract_threshold_defs:
            loc = self.world.get_location(loc_name)
            self.assertIsNotNone(loc.item, f"{loc_name} should be pre-filled")
            self.assertEqual(loc.item.name, item_name)


class TestDunaCount(_ModeChecks, KSP1TestBase):
    options = {"goal": "duna_return", "goal_contract_mode": "count",
               "contracts_available": 10}


class TestDunaProgressive(_ModeChecks, KSP1TestBase):
    options = {"goal": "duna_return", "goal_contract_mode": "progressive_unlock",
               "contracts_available": 10}


class TestDunaStarting(_ModeChecks, KSP1TestBase):
    options = {"goal": "duna_return", "goal_contract_mode": "starting",
               "contracts_available": 8}


class TestRandomContractsCount(_ModeChecks, KSP1TestBase):
    options = {"goal": "random_contracts", "goal_contract_mode": "count",
               "contracts_available": 10}


class TestRandomContractsProgressive(_ModeChecks, KSP1TestBase):
    options = {"goal": "random_contracts", "goal_contract_mode": "progressive_unlock",
               "contracts_available": 10}


class TestFlagEveryBodyProgressive(_ModeChecks, KSP1TestBase):
    options = {"goal": "flag_every_body", "goal_contract_mode": "progressive_unlock",
               "contracts_available": 12}

    def test_one_threshold_per_goal_contract(self):
        self.assertEqual(
            len(self.world.contract_threshold_defs),
            len(self.world.goal_contract_specs),
        )

    def test_progressive_thresholds_strictly_increasing(self):
        counts = [c for _l, c, _i in self.world.contract_threshold_defs]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], self.world.contracts_required)


class _TechModeChecks:
    """Tech-tree goal: Progressive R&D copies fill the thresholds. Structure only
    (the tech-tree goal is solve-rate exempt, so beatability isn't asserted)."""
    needs_real_pre_fill = False
    expected_locked = 0

    def test_rd_distribution(self):
        pool_rd = sum(1 for i in self.multiworld.itempool
                      if i.name == PROGRESSIVE_RD_NAME)
        locked_rd = sum(1 for _l, _c, it in self.world.contract_threshold_defs
                        if it == PROGRESSIVE_RD_NAME)
        precollected_rd = sum(1 for i in self.multiworld.precollected_items[self.player]
                              if i.name == PROGRESSIVE_RD_NAME)
        self.assertEqual(locked_rd, self.expected_locked)
        # Every copy is accounted for exactly once across pool / locked / starting.
        self.assertEqual(pool_rd + locked_rd + precollected_rd, PROGRESSIVE_RD_COUNT)


class TestTechCount(_TechModeChecks, KSP1TestBase):
    options = {"goal": "complete_tech_tree", "goal_contract_mode": "count",
               "contracts_available": 10}
    expected_locked = 1  # only the final R&D copy is gated behind the threshold


class TestTechProgressive(_TechModeChecks, KSP1TestBase):
    options = {"goal": "complete_tech_tree", "goal_contract_mode": "progressive_unlock",
               "contracts_available": 10}
    expected_locked = PROGRESSIVE_RD_COUNT  # all copies spread across thresholds


class TestTechStarting(_TechModeChecks, KSP1TestBase):
    options = {"goal": "complete_tech_tree", "goal_contract_mode": "starting",
               "contracts_available": 8}
    expected_locked = 0  # all R&D copies precollected, none locked, none pooled

    def test_rd_precollected(self):
        precollected = [
            i.name for i in self.multiworld.precollected_items[self.player]
        ]
        self.assertEqual(precollected.count(PROGRESSIVE_RD_NAME), PROGRESSIVE_RD_COUNT)


class TestGoalModeValidation(unittest.TestCase):
    """Fail-fast OptionError matrix (raised in generate_early)."""

    def _gen(self, **opts):
        return setup_multiworld(KSP1World, gen_steps, seed=1, options=opts)

    def test_random_contracts_findable_raises(self):
        with self.assertRaises(OptionError):
            self._gen(goal="random_contracts", goal_contract_mode="findable")

    def test_random_contracts_starting_raises(self):
        with self.assertRaises(OptionError):
            self._gen(goal="random_contracts", goal_contract_mode="starting")

    def test_x_exceeds_y_raises(self):
        with self.assertRaises(OptionError):
            self._gen(goal="duna_return", goal_contract_mode="count",
                      contracts_available=5, contracts_required_for_goal=10)

    def test_zero_contracts_count_raises(self):
        with self.assertRaises(OptionError):
            self._gen(goal="duna_return", goal_contract_mode="count",
                      contracts_available=0)
