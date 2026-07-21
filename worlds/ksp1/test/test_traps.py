"""
Trap item tests: datapackage id stability, classification, filler
substitution by density/weights, and a full-density generation smoke.
"""
from __future__ import annotations

import unittest

from BaseClasses import ItemClassification

from worlds.ksp1.items import ITEM_NAME_TO_ID, SCIENCE_PACK_NAMES, TRAP_ITEM_NAMES
from worlds.ksp1.locations import STARTING_INV_NAMES, effective_starting_inv_count
from worlds.ksp1.traps import TRAP_DEFS, TrapType

from worlds.ksp1.test.base import KSP1TestBase


# Frozen at first release: trap ids are a datapackage contract.  If this test
# fails, an id moved — that is a breaking change, not a test to update.
EXPECTED_TRAP_IDS: dict[str, int] = {
    "Trap: Stage Fright": 7_700_300,
    "Trap: Gravity Storm": 7_700_301,
    "Trap: Spin Cycle": 7_700_302,
    "Trap: Radio Silence": 7_700_303,
    "Trap: Short Circuit": 7_700_304,
    "Trap: Thermal Runaway": 7_700_305,
    "Trap: Loose Bolts": 7_700_306,
    "Trap: Mandatory Spacewalk": 7_700_307,
    "Trap: Time Slip": 7_700_308,
    "Trap: Sticky Throttle": 7_700_309,
    "Trap: Minor Kraken Attack": 7_700_310,
}


class TestTrapIds(unittest.TestCase):
    """Static table checks — no world build needed."""

    def test_ids_frozen(self):
        for name, ap_id in EXPECTED_TRAP_IDS.items():
            self.assertEqual(
                ITEM_NAME_TO_ID.get(name), ap_id,
                f"trap id drift for {name!r} — ids are frozen at release",
            )

    def test_universe_complete(self):
        self.assertEqual(TRAP_ITEM_NAMES, frozenset(EXPECTED_TRAP_IDS))
        self.assertEqual(len(TRAP_DEFS), len(TrapType))


class _TrapPoolMixin:
    """Helpers over the generated item pool."""

    def _own_pool(self):
        return [it for it in self.multiworld.itempool if it.player == self.player]

    def _traps(self):
        return [it for it in self._own_pool() if it.name in TRAP_ITEM_NAMES]

    def _science_packs(self):
        return [it for it in self._own_pool() if it.name in SCIENCE_PACK_NAMES]


class TestTrapsOff(_TrapPoolMixin, KSP1TestBase):
    options = {"trap_density": "none"}

    def test_no_traps_in_pool(self):
        self.assertEqual(self._traps(), [])
        self.assertGreater(len(self._science_packs()), 0)


class TestTrapsHell(_TrapPoolMixin, KSP1TestBase):
    options = {"trap_density": "hell"}

    def test_every_filler_slot_is_a_trap(self):
        traps = self._traps()
        self.assertGreater(len(traps), 0)
        self.assertEqual(self._science_packs(), [])

    def test_trap_classification(self):
        for it in self._traps():
            self.assertEqual(it.classification, ItemClassification.trap)
            self.assertTrue(it.trap)
            self.assertFalse(it.advancement)

    def test_traps_banned_from_starting_inventory_rule(self):
        # Starting-inventory items are granted at connect, before the first
        # flight — the item rule must reject every trap there while still
        # accepting ordinary items (Science Pack 25 isn't early-banned).
        num = effective_starting_inv_count(
            self.world.options, self.world.options.difficulty.value)
        self.assertGreater(num, 0)
        allowed = self.world.create_item("Science Pack 25")
        for name in STARTING_INV_NAMES[:num]:
            loc = self.multiworld.get_location(name, self.player)
            for trap_name in sorted(TRAP_ITEM_NAMES):
                self.assertFalse(
                    loc.item_rule(self.world.create_item(trap_name)),
                    f"{trap_name!r} must be banned from {name!r}",
                )
            self.assertTrue(loc.item_rule(allowed))

    def test_pool_matches_unfilled_locations(self):
        # The padding law: items == locations that still need filling.  Trap
        # substitution must not perturb the balance.
        unfilled = sum(
            1 for loc in self.multiworld.get_locations(self.player)
            if loc.address is not None and loc.item is None
        )
        self.assertEqual(len(self._own_pool()), unfilled)


class TestTrapWeightsSingle(_TrapPoolMixin, KSP1TestBase):
    # Absent keys weigh 0 (same consumption semantics as contract weights),
    # so listing only ``staging`` makes Stage Fright the whole trap pool.
    options = {
        "trap_density": "hell",
        "trap_type_weights": {"staging": 1},
    }

    def test_only_weighted_trap_appears(self):
        traps = self._traps()
        self.assertGreater(len(traps), 0)
        self.assertEqual({it.name for it in traps}, {"Trap: Stage Fright"})


class TestTrapWeightsAllZero(_TrapPoolMixin, KSP1TestBase):
    # Density hell + all-zero weights degrades silently to science packs.
    options = {
        "trap_density": "hell",
        "trap_type_weights": {str(t): 0 for t in TrapType},
    }

    def test_degrades_to_science_packs(self):
        self.assertEqual(self._traps(), [])
        self.assertGreater(len(self._science_packs()), 0)


class TestTrapsHellFill(KSP1TestBase):
    """Generation smoke: a full-density world still fills and beats.

    Hell is the worst case for the starting-inventory trap ban: every KSP
    filler item is a trap, so leftover SI slots can only be rescued by the
    remaining_fill swap (traps share the call with useful/filler parts).
    """
    options = {"trap_density": "hell"}
    run_default_tests = True
    needs_real_pre_fill = True

    def test_no_trap_lands_on_starting_inventory(self):
        from Fill import distribute_items_restrictive
        distribute_items_restrictive(self.multiworld)
        for name in STARTING_INV_NAMES:
            try:
                loc = self.multiworld.get_location(name, self.player)
            except KeyError:
                continue
            self.assertIsNotNone(loc.item, f"{name!r} left unfilled")
            self.assertNotIn(loc.item.name, TRAP_ITEM_NAMES, f"trap on {name!r}")
