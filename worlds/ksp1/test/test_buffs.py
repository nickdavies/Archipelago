"""
Buff item tests: datapackage id stability, classification, the pure clamp
function, per-density pool composition, and the logic-invisibility guarantee.
"""
from __future__ import annotations

import unittest

from BaseClasses import ItemClassification

from worlds.ksp1.buffs import (
    BUFF_DEFS,
    BUFF_NAME_TO_DEF,
    BUFF_TIER_COUNTS,
    BUFF_TIER_PERCENT,
    BUFF_TIER_PERCENT_OVERRIDE,
    BuffTier,
    BuffType,
    build_buff_pool,
    tier_percent,
    type_ceiling,
)
from worlds.ksp1.capability import CAPABILITY_ITEMS
from worlds.ksp1.items import BUFF_ITEM_NAMES, ITEM_NAME_TO_ID, SCIENCE_PACK_NAMES

from worlds.ksp1.test.base import KSP1TestBase


# Frozen at first release: buff ids are a datapackage contract.  If this test
# fails, an id moved — that is a breaking change, not a test to update.
EXPECTED_BUFF_IDS: dict[str, int] = {
    "Buff: Engine Efficiency I": 7_712_000,
    "Buff: Engine Efficiency II": 7_712_001,
    "Buff: Engine Efficiency III": 7_712_002,
    "Buff: Engine Thrust I": 7_712_010,
    "Buff: Engine Thrust II": 7_712_011,
    "Buff: Engine Thrust III": 7_712_012,
    "Buff: Heat Tolerance I": 7_712_020,
    "Buff: Heat Tolerance II": 7_712_021,
    "Buff: Heat Tolerance III": 7_712_022,
    "Buff: Structural Integrity I": 7_712_030,
    "Buff: Structural Integrity II": 7_712_031,
    "Buff: Structural Integrity III": 7_712_032,
    "Buff: Control Authority I": 7_712_040,
    "Buff: Control Authority II": 7_712_041,
    "Buff: Control Authority III": 7_712_042,
    "Buff: Power Generation I": 7_712_050,
    "Buff: Power Generation II": 7_712_051,
    "Buff: Power Generation III": 7_712_052,
}

_FULL_BLOCK = len(BuffType) * sum(BUFF_TIER_COUNTS.values())  # 6 * 6 = 36


class TestBuffIds(unittest.TestCase):
    """Static table checks — no world build needed."""

    def test_ids_frozen(self):
        for name, ap_id in EXPECTED_BUFF_IDS.items():
            self.assertEqual(
                ITEM_NAME_TO_ID.get(name), ap_id,
                f"buff id drift for {name!r} — ids are frozen at release",
            )

    def test_universe_complete(self):
        self.assertEqual(BUFF_ITEM_NAMES, frozenset(EXPECTED_BUFF_IDS))
        self.assertEqual(len(BUFF_DEFS), len(BuffType) * len(BuffTier))
        self.assertEqual(len(BUFF_NAME_TO_DEF), len(BUFF_DEFS))

    def test_buffs_are_invisible_to_logic(self):
        """The load-bearing correctness property.

        A buff absent from CAPABILITY_ITEMS makes world.collect_item return
        None for it, so it never enters state.prog_items and state.count() on
        it is identically 0 — the physics model cannot observe a buff even if
        a rule tried to read one.  Filler classification alone would NOT be
        enough: USEFUL-classified parts are capability-relevant.
        """
        for name in BUFF_ITEM_NAMES:
            self.assertNotIn(
                name, CAPABILITY_ITEMS,
                f"{name!r} leaked into CAPABILITY_ITEMS — buffs must never "
                f"reach the physics model",
            )


class TestBuffMagnitudes(unittest.TestCase):
    """Guards the numbers the option docstrings and player doc quote.

    The CLIENT (Buffs/BuffDefs.cs) is the authority on effect size; this table
    exists so the docs can state ceilings. If these drift apart the player is
    told the wrong number, so the ceilings are pinned here.
    """

    def test_default_ladder(self):
        self.assertEqual(BUFF_TIER_PERCENT[BuffTier.SMALL], 1)
        self.assertEqual(BUFF_TIER_PERCENT[BuffTier.MEDIUM], 3)
        self.assertEqual(BUFF_TIER_PERCENT[BuffTier.LARGE], 5)

    def test_structural_is_the_only_override(self):
        self.assertEqual(set(BUFF_TIER_PERCENT_OVERRIDE), {BuffType.STRUCTURAL})
        self.assertEqual(
            BUFF_TIER_PERCENT_OVERRIDE[BuffType.STRUCTURAL],
            {BuffTier.SMALL: 5, BuffTier.MEDIUM: 15, BuffTier.LARGE: 25},
        )

    def test_tier_percent_honours_override(self):
        self.assertEqual(tier_percent(BuffType.ISP, BuffTier.LARGE), 5)
        self.assertEqual(tier_percent(BuffType.STRUCTURAL, BuffTier.LARGE), 25)

    def test_documented_ceilings(self):
        """The exact figures in the BuffDensity docstring and player doc."""
        normal = BUFF_TIER_COUNTS
        light = {BuffTier.SMALL: 1, BuffTier.MEDIUM: 1, BuffTier.LARGE: 0}
        heavy = {BuffTier.SMALL: 5, BuffTier.MEDIUM: 3, BuffTier.LARGE: 2}
        for buff_type, counts, expected in (
            (BuffType.ISP, light, 4),
            (BuffType.ISP, normal, 14),
            (BuffType.ISP, heavy, 24),
            (BuffType.STRUCTURAL, light, 20),
            (BuffType.STRUCTURAL, normal, 70),
            (BuffType.STRUCTURAL, heavy, 120),
        ):
            with self.subTest(buff=buff_type.name, expected=expected):
                self.assertEqual(type_ceiling(buff_type, counts), expected)


class TestBuildBuffPool(unittest.TestCase):
    """The clamp, tested as a pure function.

    Deliberately not tested through a built world: that would tie the budgets
    to a seed's location count, which drifts every time a location is added.
    """

    ALL = tuple(BuffType)

    def test_full_budget_returns_whole_block(self):
        got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, 10_000)
        self.assertEqual(len(got), _FULL_BLOCK)

    def test_exact_budget_returns_whole_block(self):
        got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, _FULL_BLOCK)
        self.assertEqual(len(got), _FULL_BLOCK)

    def test_budget_is_used_in_full(self):
        for budget in (0, 1, 5, 19, 35, _FULL_BLOCK):
            with self.subTest(budget=budget):
                got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, budget)
                self.assertEqual(len(got), min(budget, _FULL_BLOCK))

    def test_non_positive_budget_is_empty(self):
        for budget in (0, -1, -100):
            with self.subTest(budget=budget):
                self.assertEqual(build_buff_pool(BUFF_TIER_COUNTS, self.ALL, budget), [])

    def test_no_enabled_types_is_empty(self):
        self.assertEqual(build_buff_pool(BUFF_TIER_COUNTS, (), 100), [])

    def test_zero_counts_is_empty(self):
        zero = {t: 0 for t in BuffTier}
        self.assertEqual(build_buff_pool(zero, self.ALL, 100), [])

    def test_monotone_in_budget(self):
        """A larger budget yields a superset — and the same prefix."""
        prev = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, 0)
        for budget in range(1, _FULL_BLOCK + 1):
            got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, budget)
            self.assertEqual(got[:len(prev)], prev, f"prefix changed at budget {budget}")
            prev = got

    def test_large_tiers_survive_truncation(self):
        """Strongest-first: a tight budget keeps the +5% copies, not the +1%."""
        got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, len(BuffType))
        for name in got:
            self.assertEqual(BUFF_NAME_TO_DEF[name][1], BuffTier.LARGE)

    def test_types_stay_even_under_truncation(self):
        """Copy-major emit: two enabled types differ by at most one copy."""
        for budget in range(1, _FULL_BLOCK + 1):
            got = build_buff_pool(BUFF_TIER_COUNTS, self.ALL, budget)
            per_type = {t: 0 for t in BuffType}
            for name in got:
                per_type[BUFF_NAME_TO_DEF[name][0]] += 1
            with self.subTest(budget=budget):
                self.assertLessEqual(
                    max(per_type.values()) - min(per_type.values()), 1,
                    f"uneven type stocking at budget {budget}: {per_type}",
                )

    def test_disabled_types_absent(self):
        enabled = (BuffType.ISP, BuffType.POWER)
        got = build_buff_pool(BUFF_TIER_COUNTS, enabled, 10_000)
        self.assertEqual({BUFF_NAME_TO_DEF[n][0] for n in got}, set(enabled))
        self.assertEqual(len(got), len(enabled) * sum(BUFF_TIER_COUNTS.values()))


class _BuffPoolMixin:
    """Helpers over the generated item pool."""

    def _own_pool(self):
        return [it for it in self.multiworld.itempool if it.player == self.player]

    def _buffs(self):
        return [it for it in self._own_pool() if it.name in BUFF_ITEM_NAMES]

    def _science_packs(self):
        return [it for it in self._own_pool() if it.name in SCIENCE_PACK_NAMES]

    def _unfilled_addressed_locations(self):
        # Same predicate create_all_items pads against: pre-filled locations
        # (goal-mode thresholds carry a locked item) are excluded, because
        # their item was never in the pool.
        return sum(
            1 for loc in self.multiworld.get_locations(self.player)
            if loc.address is not None and loc.item is None
        )

    def assert_pool_balanced(self):
        """Items must exactly fill the addressed locations."""
        self.assertEqual(len(self._own_pool()), self._unfilled_addressed_locations())


class TestBuffsOff(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "none"}

    def test_no_buffs_in_pool(self):
        self.assertEqual(self._buffs(), [])
        self.assertGreater(len(self._science_packs()), 0)
        self.assert_pool_balanced()


class TestBuffsNormal(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "normal"}

    def test_exact_per_type_per_tier_counts(self):
        seen: dict[tuple[BuffType, BuffTier], int] = {}
        for item in self._buffs():
            key = BUFF_NAME_TO_DEF[item.name]
            seen[key] = seen.get(key, 0) + 1
        for buff_type in BuffType:
            for tier, expected in BUFF_TIER_COUNTS.items():
                with self.subTest(buff=buff_type.name, tier=tier.name):
                    self.assertEqual(seen.get((buff_type, tier), 0), expected)

    def test_block_size(self):
        self.assertEqual(len(self._buffs()), _FULL_BLOCK)

    def test_classification_is_filler(self):
        for item in self._buffs():
            with self.subTest(item=item.name):
                self.assertEqual(item.classification, ItemClassification.filler)
                self.assertFalse(item.advancement)
                self.assertFalse(item.trap)

    def test_pool_balanced(self):
        self.assert_pool_balanced()

    def test_buffs_displace_science_packs_not_locations(self):
        """Buffs come out of the filler budget — the pool size is unchanged."""
        self.assert_pool_balanced()
        self.assertGreater(len(self._science_packs()), 0)


class TestBuffsLight(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "light"}

    def test_light_block_size(self):
        # 1 small + 1 medium per type, no large.
        self.assertEqual(len(self._buffs()), len(BuffType) * 2)
        tiers = {BUFF_NAME_TO_DEF[it.name][1] for it in self._buffs()}
        self.assertNotIn(BuffTier.LARGE, tiers)
        self.assert_pool_balanced()


class TestBuffTypesSubset(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "normal", "buff_types": ["isp", "power"]}

    def test_only_enabled_types_appear(self):
        got = {BUFF_NAME_TO_DEF[it.name][0] for it in self._buffs()}
        self.assertEqual(got, {BuffType.ISP, BuffType.POWER})
        self.assertEqual(len(self._buffs()), 2 * sum(BUFF_TIER_COUNTS.values()))
        self.assert_pool_balanced()
