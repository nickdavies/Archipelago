"""Strict capability cross-check backstop.

The strict-ladder fill gates each capability mission with a cheap
``has_all(reps)`` bracket rule for speed.  ``post_fill`` then re-verifies the
SAME seed under the REAL capability rules, sweeping from scratch — and if that
disagrees (a location the cheap fill treated as reachable is not reachable under
real physics, e.g. because capability is non-monotone in the part set), it pays
an expensive whole-seed fallback re-fill.

This test is the structural guard for that agreement: build a seed, run the real
strict-ladder fill, then install the saved capability rules and assert the seed
is still beatable under real physics WITHOUT the fallback.  A regression here is
a cheap-bracket vs capability disparity (bug 092 class) — fix the capability
model, never the bracket.

The from-scratch sweep is what matters: a non-monotonicity (e.g. the RTG
power-charge bug) bites mid-bootstrap, when a heavy item is collected before the
part that would offset it.  Comparing against the *full* pool would mask it (the
pool owns every offsetting part), so this drives the real cross-check instead.
"""
from __future__ import annotations

import random
from argparse import Namespace

import worlds  # noqa: F401  (registers all worlds, including KSP1)
import worlds.AutoWorld as AutoWorld
from BaseClasses import CollectionState, MultiWorld
from Generate import get_seed_name
from test.general import gen_steps
from Fill import distribute_items_restrictive

GAME = "Kerbal Space Program 1"

# Seeds known to exercise the cheap-bracket vs capability agreement.  The
# complete_tech_tree/kerbin seed is the exact case whose Mun-return RTG charge
# was non-monotone before capability._required_power_source — a direct
# regression guard.  flag_every_body/kerbin gives broad body/mission coverage.
_CONFIGS = [
    ("complete_tech_tree", "kerbin", "findable", 10655457994218381434),
    ("flag_every_body", "kerbin", "findable", 0xDEADBEEF),
]


def _build_and_fill(goal: str, home: str, mode: str, seed: int):
    opts = {"goal": goal, "starting_body": home, "difficulty": "normal",
            "accessibility": "minimal", "goal_contract_mode": mode}
    mw = MultiWorld(1)
    mw.game[1] = GAME
    mw.player_name = {1: "Tester"}
    mw.set_seed(seed)
    random.seed(mw.seed)
    mw.seed_name = get_seed_name(random)
    args = Namespace()
    world_type = AutoWorld.AutoWorldRegister.world_types[GAME]
    for name, option in world_type.options_dataclass.type_hints.items():
        setattr(args, name, {1: option.from_any(opts.get(name, option.default))})
    mw.set_options(args)
    mw.state = CollectionState(mw)
    for step in gen_steps:  # includes pre_fill -> apply_sphere_ladder
        AutoWorld.call_all(mw, step)
    distribute_items_restrictive(mw)
    return mw


class TestCapabilityCrossCheck:
    """Build + fill, then assert the seed is beatable under REAL capability."""

    def _check(self, goal, home, mode, seed):
        mw = _build_and_fill(goal, home, mode, seed)
        world = mw.worlds[1]
        saved = getattr(world, "_strict_ladder_saved_rules", None) or {}
        assert saved, "expected strict-ladder saved capability rules"
        # Swap the cheap bracket rules for the real capability rules and verify
        # the seed still beats from scratch (the post_fill cross-check).
        for loc in mw.get_locations(world.player):
            if loc.name in saved:
                loc.access_rule = saved[loc.name]
        assert mw.can_beat_game(), (
            f"{goal}/{home}/{mode} seed={seed}: NOT beatable under real capability "
            "after a cheap-bracket fill — a cheap-bracket vs capability disparity "
            "(non-monotonicity). Fix the capability model, not the bracket.")

    def test_complete_tech_tree_kerbin(self):
        self._check(*_CONFIGS[0])

    def test_flag_every_body_kerbin(self):
        self._check(*_CONFIGS[1])
