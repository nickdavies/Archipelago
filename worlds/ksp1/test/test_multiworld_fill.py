"""Multiworld fill + cross-check backstop.

Every other fill test (and the solve-check rig) is SINGLE-player.  That gap let
a multiworld-only defect ship: in a real multiworld AP scatters KSP's own
progression items into other games' locations, and the old post_fill fallback
re-fill — which cleared only KSP's own locations — could not relocate them, so
its "rescue" was a no-op that reported "0 unreachable progression" (empty
sample) while the seed stayed broken.

This test drives the real pipeline with KSP + filler players (ChecksFinder):
gen steps, the actual ``distribute_items_restrictive``, then ``post_fill`` (the
strict-ladder cross-check).  It asserts the multiworld path is genuinely
exercised — KSP progression items DID land in foreign slots — and that the
cross-check still proves the whole multiworld beatable under real capability,
with the cheap rules restored afterwards (the spoiler-perf contract).
"""
from __future__ import annotations

import random
from argparse import Namespace

import pytest

import worlds  # noqa: F401  (registers all worlds, including KSP1)
import worlds.AutoWorld as AutoWorld
from BaseClasses import CollectionState, ItemClassification, MultiWorld
from Generate import get_seed_name
from test.general import gen_steps
from Fill import distribute_items_restrictive

KSP = "Kerbal Space Program 1"
FILLER = "ChecksFinder"
N_FILLERS = 12

# Fixed seeds validated to scatter KSP advancement items into filler worlds
# (9-19 foreign placements each at 12 fillers).  duna_return/kerbin is the
# reported-failure config class; the "full" case pins the reporter's exact
# accessibility (non-gating for release, but deterministic here — a divergence
# appearing on this seed after a capability change is an early warning, not
# flake).
_CONFIGS = [
    ("duna_return", "kerbin", "minimal", 1),
    ("duna_return", "kerbin", "minimal", 2),
    ("duna_return", "kerbin", "full", 3),
]


def _build_multiworld(goal: str, home: str, accessibility: str, seed: int):
    n = 1 + N_FILLERS
    mw = MultiWorld(n)
    mw.game = {1: KSP, **{p: FILLER for p in range(2, n + 1)}}
    mw.player_name = {1: "KSP", **{p: f"cf{p}" for p in range(2, n + 1)}}
    mw.set_seed(seed)
    random.seed(mw.seed)
    mw.seed_name = get_seed_name(random)
    ksp_opts = {"goal": goal, "starting_body": home, "difficulty": "normal",
                "accessibility": accessibility}
    args = Namespace()
    for player, game in mw.game.items():
        hints = AutoWorld.AutoWorldRegister.world_types[game].options_dataclass.type_hints
        for name, option in hints.items():
            per_player = getattr(args, name, {})
            raw = ksp_opts.get(name, option.default) if game == KSP else option.default
            per_player[player] = option.from_any(raw)
            setattr(args, name, per_player)
    mw.set_options(args)
    mw.state = CollectionState(mw)
    for step in gen_steps:  # includes pre_fill -> apply_sphere_ladder
        AutoWorld.call_all(mw, step)
    return mw


class TestMultiworldFill:
    """KSP + fillers through the real fill and post_fill cross-check."""

    @pytest.mark.parametrize("goal,home,accessibility,seed", _CONFIGS)
    def test_multiworld_generation_clean(self, goal, home, accessibility, seed):
        mw = _build_multiworld(goal, home, accessibility, seed)
        world = mw.worlds[1]
        distribute_items_restrictive(mw)

        # The multiworld path must actually be exercised: at least one KSP
        # progression item in a foreign world's location.  If this ever drops
        # to zero the test has silently degenerated to single-player coverage.
        foreign = [
            loc for loc in mw.get_locations()
            if loc.item is not None and loc.item.player == 1 and loc.player != 1
            and (loc.item.classification & ItemClassification.progression)
        ]
        assert foreign, (
            f"{goal}/{home}/{accessibility} seed={seed}: no KSP progression "
            "items landed in filler worlds — the multiworld path is not being "
            "exercised (raise N_FILLERS or pick a different seed)")

        # The real post_fill: strict-ladder cross-check under real capability.
        # A divergence raises OptionError here (hard assert, no rescue) — so
        # completing this call IS the multiworld cross-check passing.
        AutoWorld.call_all(mw, "post_fill")

        # Ship gates, same semantics Main.py applies after fill.
        assert mw.can_beat_game(), (
            f"{goal}/{home}/{accessibility} seed={seed}: multiworld not "
            "beatable after post_fill")
        assert mw.fulfills_accessibility(), (
            f"{goal}/{home}/{accessibility} seed={seed}: accessibility rules "
            "not fulfilled after post_fill")

        # Pass-path contract: post_fill must restore the CHEAP rules so the
        # spoiler prune pass never re-pays the full get_capability cost.
        saved = getattr(world, "_strict_ladder_saved_rules", None) or {}
        assert saved, "expected strict-ladder saved capability rules"
        restored = [
            loc for loc in mw.get_locations(1)
            if loc.name in saved and loc.access_rule is saved[loc.name]
        ]
        assert not restored, (
            f"{goal}/{home}/{accessibility} seed={seed}: {len(restored)} "
            "locations still carry the REAL capability rule after a passing "
            "cross-check — the cheap-rule restore regressed "
            "(spoiler-perf contract, cd49406b)")
