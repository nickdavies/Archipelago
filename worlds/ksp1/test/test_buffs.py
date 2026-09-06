"""
Buff item tests, permanent and consumable: datapackage id stability,
classification, the pure clamp functions, per-density pool composition, and the
logic-invisibility guarantee.
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
    CONSUMABLE_DEFS,
    CONSUMABLE_DENSITY_COUNTS,
    CONSUMABLE_NAME_TO_TYPE,
    BuffTier,
    BuffType,
    ConsumableType,
    build_buff_pool,
    build_consumable_pool,
    tier_percent,
    type_ceiling,
)
from worlds.ksp1.capability import CAPABILITY_ITEMS
from worlds.ksp1.items import (
    BUFF_ITEM_NAMES,
    CONSUMABLE_ITEM_NAMES,
    ITEM_NAME_TO_ID,
    SCIENCE_PACK_NAMES,
)
from worlds.ksp1.options import BuffDensity

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

# Frozen for the same reason, in the 12100-12199 consumable sub-band.  An id
# moving here is a breaking datapackage change, not a test to update.
EXPECTED_CONSUMABLE_IDS: dict[str, int] = {
    "Buff: Mid-Air Refuel": 7_712_100,
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


class TestConsumableIds(unittest.TestCase):
    """Static table checks for the one-shot block — no world build needed."""

    def test_ids_frozen(self):
        for name, ap_id in EXPECTED_CONSUMABLE_IDS.items():
            self.assertEqual(
                ITEM_NAME_TO_ID.get(name), ap_id,
                f"consumable id drift for {name!r} — ids are frozen at release",
            )

    def test_universe_complete(self):
        self.assertEqual(CONSUMABLE_ITEM_NAMES, frozenset(EXPECTED_CONSUMABLE_IDS))
        self.assertEqual(len(CONSUMABLE_DEFS), len(ConsumableType))
        self.assertEqual(len(CONSUMABLE_NAME_TO_TYPE), len(CONSUMABLE_DEFS))

    def test_no_overlap_with_permanent_buffs(self):
        """Two tables, one id space: names and ids must stay disjoint."""
        self.assertEqual(CONSUMABLE_ITEM_NAMES & BUFF_ITEM_NAMES, frozenset())
        self.assertEqual(
            {ITEM_NAME_TO_ID[n] for n in CONSUMABLE_ITEM_NAMES}
            & {ITEM_NAME_TO_ID[n] for n in BUFF_ITEM_NAMES},
            set(),
        )

    def test_density_keys_match_the_option(self):
        """buffs.py owns the counts, keyed by BuffDensity's key names.

        Pins the two halves of that contract together: adding a density rung
        without a charge count (or vice versa) fails here instead of raising
        KeyError mid-generation.
        """
        self.assertEqual(
            set(CONSUMABLE_DENSITY_COUNTS),
            set(BuffDensity.name_lookup.values()),
        )

    def test_documented_charge_counts(self):
        """The exact figures the BuffDensity docstring and player doc quote."""
        self.assertEqual(CONSUMABLE_DENSITY_COUNTS["none"], 0)
        self.assertEqual(CONSUMABLE_DENSITY_COUNTS["light"], 1)
        self.assertEqual(CONSUMABLE_DENSITY_COUNTS["normal"], 3)
        self.assertEqual(CONSUMABLE_DENSITY_COUNTS["heavy"], 5)

    def test_consumables_are_invisible_to_logic(self):
        """Same load-bearing property the permanent block has.

        A consumable is spent at the player's discretion — logic must never be
        able to assume a charge was banked, and absence from CAPABILITY_ITEMS
        is what makes that structural rather than a convention.
        """
        for name in CONSUMABLE_ITEM_NAMES:
            self.assertNotIn(
                name, CAPABILITY_ITEMS,
                f"{name!r} leaked into CAPABILITY_ITEMS — consumables must "
                f"never reach the physics model",
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


class TestBuildConsumablePool(unittest.TestCase):
    """The one-shot clamp, tested as a pure function.

    Same rationale as TestBuildBuffPool: going through a built world would tie
    the budgets to a seed's location count, which drifts whenever a location is
    added.
    """

    ALL = tuple(ConsumableType)
    NORMAL = CONSUMABLE_DENSITY_COUNTS["normal"]

    def _full_block(self, count_per_type: int) -> int:
        return count_per_type * len(ConsumableType)

    def test_full_budget_returns_whole_block(self):
        got = build_consumable_pool(self.NORMAL, self.ALL, 10_000)
        self.assertEqual(len(got), self._full_block(self.NORMAL))

    def test_exact_budget_returns_whole_block(self):
        size = self._full_block(self.NORMAL)
        self.assertEqual(len(build_consumable_pool(self.NORMAL, self.ALL, size)), size)

    def test_budget_is_used_in_full(self):
        size = self._full_block(self.NORMAL)
        for budget in (0, 1, 2, size - 1, size, size + 7):
            with self.subTest(budget=budget):
                got = build_consumable_pool(self.NORMAL, self.ALL, budget)
                self.assertEqual(len(got), min(max(budget, 0), size))

    def test_non_positive_budget_is_empty(self):
        for budget in (0, -1, -100):
            with self.subTest(budget=budget):
                self.assertEqual(build_consumable_pool(self.NORMAL, self.ALL, budget), [])

    def test_no_enabled_types_is_empty(self):
        self.assertEqual(build_consumable_pool(self.NORMAL, (), 100), [])

    def test_zero_count_is_empty(self):
        self.assertEqual(build_consumable_pool(0, self.ALL, 100), [])

    def test_every_density_count(self):
        for key, count in CONSUMABLE_DENSITY_COUNTS.items():
            with self.subTest(density=key):
                got = build_consumable_pool(count, self.ALL, 10_000)
                self.assertEqual(len(got), self._full_block(count))
                for consumable in ConsumableType:
                    name = CONSUMABLE_DEFS[consumable][0]
                    self.assertEqual(got.count(name), count)

    def test_monotone_in_budget(self):
        """A larger budget yields a superset — and the same prefix."""
        size = self._full_block(self.NORMAL)
        prev = build_consumable_pool(self.NORMAL, self.ALL, 0)
        for budget in range(1, size + 1):
            got = build_consumable_pool(self.NORMAL, self.ALL, budget)
            self.assertEqual(got[:len(prev)], prev, f"prefix changed at budget {budget}")
            prev = got

    def test_types_stay_even_under_truncation(self):
        """Copy-major emit: two enabled types differ by at most one charge."""
        size = self._full_block(self.NORMAL)
        for budget in range(1, size + 1):
            got = build_consumable_pool(self.NORMAL, self.ALL, budget)
            per_type = {t: 0 for t in ConsumableType}
            for name in got:
                per_type[CONSUMABLE_NAME_TO_TYPE[name]] += 1
            with self.subTest(budget=budget):
                self.assertLessEqual(
                    max(per_type.values()) - min(per_type.values()), 1,
                    f"uneven type stocking at budget {budget}: {per_type}",
                )

    def test_only_known_names_emitted(self):
        got = build_consumable_pool(self.NORMAL, self.ALL, 10_000)
        self.assertEqual(set(got), set(CONSUMABLE_NAME_TO_TYPE))

    def test_disabled_types_absent(self):
        """A type left out of buff_types contributes nothing."""
        for consumable in ConsumableType:
            enabled = [t for t in ConsumableType if t is not consumable]
            with self.subTest(disabled=consumable.name):
                got = build_consumable_pool(self.NORMAL, enabled, 10_000)
                self.assertNotIn(CONSUMABLE_DEFS[consumable][0], got)
                self.assertEqual(len(got), self._full_block(self.NORMAL) - self.NORMAL)


class _BuffPoolMixin:
    """Helpers over the generated item pool."""

    def _own_pool(self):
        return [it for it in self.multiworld.itempool if it.player == self.player]

    def _buffs(self):
        return [it for it in self._own_pool() if it.name in BUFF_ITEM_NAMES]

    def _consumables(self):
        return [it for it in self._own_pool() if it.name in CONSUMABLE_ITEM_NAMES]

    def assert_consumable_charges(self, density_key: str):
        """Every enabled consumable type carries exactly its density's charges.

        Asserted per type rather than as a total so a future second consumable
        cannot be stocked unevenly without failing.  The count comes from
        CONSUMABLE_DENSITY_COUNTS, never a literal, so the test tracks the
        single source of truth.
        """
        expected = CONSUMABLE_DENSITY_COUNTS[density_key]
        seen: dict[ConsumableType, int] = {t: 0 for t in ConsumableType}
        for item in self._consumables():
            seen[CONSUMABLE_NAME_TO_TYPE[item.name]] += 1
        for consumable, count in seen.items():
            with self.subTest(consumable=consumable.name, density=density_key):
                self.assertEqual(count, expected)

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

    def test_no_consumables_in_pool(self):
        self.assertEqual(self._consumables(), [])


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

    def test_consumable_charges(self):
        self.assert_consumable_charges("normal")

    def test_consumable_classification_is_filler(self):
        """``useful`` would move these into usefulitempool and reorder fill."""
        for item in self._consumables():
            with self.subTest(item=item.name):
                self.assertEqual(item.classification, ItemClassification.filler)
                self.assertFalse(item.advancement)
                self.assertFalse(item.trap)

    def test_both_blocks_fit_inside_the_filler_budget(self):
        """The pool invariant with both blocks present.

        The consumable block is sized off the budget the permanent block left,
        so an arithmetic slip in create_all_items shows up here as an
        items-vs-locations imbalance.
        """
        self.assertGreater(len(self._consumables()), 0)
        self.assertGreater(len(self._buffs()), 0)
        self.assert_pool_balanced()


class TestBuffsLight(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "light"}

    def test_light_block_size(self):
        # 1 small + 1 medium per type, no large.
        self.assertEqual(len(self._buffs()), len(BuffType) * 2)
        tiers = {BUFF_NAME_TO_DEF[it.name][1] for it in self._buffs()}
        self.assertNotIn(BuffTier.LARGE, tiers)
        self.assert_pool_balanced()

    def test_consumable_charges(self):
        self.assert_consumable_charges("light")


class TestBuffsHeavy(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "heavy"}

    def test_heavy_block_size(self):
        self.assertEqual(len(self._buffs()), len(BuffType) * (5 + 3 + 2))
        self.assert_pool_balanced()

    def test_consumable_charges(self):
        self.assert_consumable_charges("heavy")


class TestBuffTypesSubset(_BuffPoolMixin, KSP1TestBase):
    options = {"buff_density": "normal", "buff_types": ["isp", "power"]}

    def test_only_enabled_types_appear(self):
        got = {BUFF_NAME_TO_DEF[it.name][0] for it in self._buffs()}
        self.assertEqual(got, {BuffType.ISP, BuffType.POWER})
        self.assertEqual(len(self._buffs()), 2 * sum(BUFF_TIER_COUNTS.values()))
        self.assert_pool_balanced()

    def test_consumables_disabled_by_omission(self):
        """buff_types without a consumable name switches it off entirely."""
        self.assertEqual(self._consumables(), [])


class TestConsumablesOnly(_BuffPoolMixin, KSP1TestBase):
    """The other half of the shared option set: consumables with no permanents."""

    options = {"buff_density": "normal", "buff_types": ["refuel"]}

    def test_no_permanent_buffs(self):
        self.assertEqual(self._buffs(), [])

    def test_consumable_charges(self):
        self.assert_consumable_charges("normal")
        self.assert_pool_balanced()
