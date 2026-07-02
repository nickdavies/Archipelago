"""EnabledPartPacks option: disabling a pack removes its parts from the world,
scopes the rank table, and the model-infeasible table resolves per the enabled
capability-relevant packs (base ⊕ delta)."""
import random
import unittest
from argparse import Namespace

import worlds  # noqa: F401  (registers the KSP1 world)
import worlds.AutoWorld as AutoWorld
from BaseClasses import CollectionState, MultiWorld

from worlds.ksp1.parts.packs import STOCK, MAKING_HISTORY
from worlds.ksp1.data.feasibility import BASE_RELEVANT_PACKS

GAME = "Kerbal Space Program 1"


def _build_world(enabled_part_packs):
    """Run generate_early for a Kerbin/mun_flag seed with the given
    enabled_part_packs (None = option default)."""
    mw = MultiWorld(1)
    mw.game[1] = GAME
    mw.player_name = {1: "Tester"}
    mw.set_seed(0xC0FFEE)
    random.seed(mw.seed)
    args = Namespace()
    opt_types = AutoWorld.AutoWorldRegister.world_types[
        GAME].options_dataclass.type_hints
    overrides = {"goal": "mun_flag", "starting_body": "kerbin",
                 "difficulty": "normal"}
    if enabled_part_packs is not None:
        overrides["enabled_part_packs"] = enabled_part_packs
    for name, option in opt_types.items():
        setattr(args, name,
                {1: option.from_any(overrides.get(name, option.default))})
    mw.set_options(args)
    mw.state = CollectionState(mw)
    AutoWorld.call_all(mw, "generate_early")
    return mw.worlds[1]


class TestEnabledPartPacks(unittest.TestCase):
    def test_default_is_making_history_on(self):
        w = _build_world(None)
        self.assertEqual(w.part_manager.enabled_packs,
                         frozenset({STOCK, MAKING_HISTORY}))
        # The default config is exactly the base the feasibility table was
        # generated for, so the seed uses MODEL_INFEASIBLE_BASE directly.
        self.assertEqual(
            tuple(sorted(w.part_manager.capability_relevant_packs())),
            BASE_RELEVANT_PACKS)

    def test_disabling_making_history_drops_parts(self):
        on = _build_world(["MakingHistory"])
        off = _build_world([])
        self.assertEqual(off.part_manager.enabled_packs, frozenset({STOCK}))
        self.assertLess(len(off.part_manager.parts), len(on.part_manager.parts))
        # The rank table is scoped to the enabled set.
        self.assertEqual(off._rank_context.enabled_packs, frozenset({STOCK}))

    def test_disabled_pack_resolves_feasibility(self):
        off = _build_world([])
        # base ⊕ delta must yield a well-formed infeasible set (no crash).
        self.assertIsInstance(off.unachievable_missions, frozenset)

    def test_making_history_does_not_change_feasibility(self):
        # MH adds no frontier (empty delta), so the model-infeasible mission
        # set is identical with or without it. This is the runtime mirror of
        # the generator's empty MODEL_INFEASIBLE_DELTAS.
        on = _build_world(["MakingHistory"])
        off = _build_world([])
        self.assertEqual(off.unachievable_missions, on.unachievable_missions)

    def test_stock_only_contracts_emit_no_mh_has_any_part(self):
        """Stock-only seeds must not leak MH part names into has_any_part lists."""
        from worlds.ksp1.parts import part_manager_for
        from worlds.ksp1.parts.packs import MAKING_HISTORY, STOCK

        w = _build_world([])
        default_pm = part_manager_for(frozenset({STOCK, MAKING_HISTORY}))
        stock_pm = w.part_manager
        mh_only: set[str] = set()
        for cat, members in default_pm.category_members.items():
            mh_only |= members - stock_pm.category_members.get(cat, frozenset())

        has_any_part_names: set[str] = set()
        for contract in w.fill_slot_data()["contracts"]:
            for param in contract["parameters"]:
                if param.get("kind") == "has_any_part":
                    has_any_part_names.update(param["parts"])

        leaked = mh_only & has_any_part_names
        self.assertFalse(
            leaked,
            f"MH-only parts leaked into stock-only has_any_part: {sorted(leaked)}")


if __name__ == "__main__":
    unittest.main()
