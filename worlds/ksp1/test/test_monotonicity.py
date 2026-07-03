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
# regression guard.  duna_return/kerbin/count is the seed whose 3rd Progressive
# R&D stranded behind Tylo Sample Return until missions gained their transitive
# Counted(R&D, rd_avail(M_sphere)) requirement (sphere_ladder._install_ladder_rules)
# — guards the counted-progressive cycle.  flag_every_body/kerbin gives broad
# body/mission coverage.
_CONFIGS = [
    ("complete_tech_tree", "kerbin", "findable", 10655457994218381434, {}),
    ("duna_return", "kerbin", "count", 10727693526395107800, {}),
    ("flag_every_body", "kerbin", "findable", 0xDEADBEEF, {}),
    # Bug 092's deadlock seed: mid-sweep the real-rules cross-check collected
    # mk3FuselageLFO.50, whose largest-first pack poisoning lost Eve return
    # (a strictly larger kit losing a mission) → strand → fallback re-fill.
    # Guards the tank-pack monotonicity fix end to end.
    ("duna_return", "kerbin", "count", 17074417405164113416,
     {"buildings_in_logic": 1}),
    # Bugs/101's strand seed: the bracket scan proved missions against
    # spheres[i].flags — the defining mission's own kit, or a fallback
    # anchor's full rank-admit kit — while the installed gate enforces
    # has_all(reps_collected), a thinner kit nobody proved.  Here that
    # stranded 64 progression items behind a "Moho Orbit 1" gate whose
    # enforced kit couldn't fly the mission.  Guards SphereBoundary.flags
    # being built reps-only from exactly reps_collected.
    ("complete_tech_tree", "duna", "count", 6903591710897770246, {}),
    # Bugs/102's strand seed: the relay CHARGE ignored the relay GATE's
    # crewed exemption, so only kits that OWNED a tier-2 antenna paid its
    # mass on crewed missions — acquiring RelayAntenna5 pushed "Kerbin EVA
    # in Orbit 1" (from Moho) past the pad cap (a strictly larger kit
    # losing a mission).  Guards charge-mirrors-gate for support equipment.
    ("flag_every_body", "moho", "count", 10721509650795190484, {}),
    # Bugs/103's strand seed: the relay charge scanned tiers
    # range(required, 4), EXCLUDING tier 4 — a kit whose only adequate
    # antenna was the tier-4 dish charged NO antenna mass (under-charge)
    # while a kit that also owned the tier-3 antenna paid its real mass
    # (bigger kit, heavier rocket → over Moho's pad cap).  Guards the
    # scan-all-owned-tiers charge.
    ("mun_flag", "moho", "count", 11551505357908101103, {}),
]


def _build_and_fill(goal: str, home: str, mode: str, seed: int,
                    extra_opts: dict | None = None):
    opts = {"goal": goal, "starting_body": home, "difficulty": "normal",
            "accessibility": "minimal", "goal_contract_mode": mode}
    opts.update(extra_opts or {})
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
    # Run the real post_fill cross-check so ``_strict_ladder_fell_back``
    # reflects a genuine divergence (gen_steps stops at pre_fill).
    AutoWorld.call_all(mw, "post_fill")
    return mw


class TestCapabilityCrossCheck:
    """Build + fill, then assert the seed is beatable under REAL capability."""

    def _check(self, goal, home, mode, seed, extra_opts=None):
        mw = _build_and_fill(goal, home, mode, seed, extra_opts)
        world = mw.worlds[1]
        # The post_fill cross-check must not have needed the whole-seed
        # re-fill rescue: a fallback on these fixed seeds is a cheap-bracket
        # vs capability divergence (bug 092 class) even though the rescue
        # makes the seed solvable.
        assert not getattr(world, "_strict_ladder_fell_back", False), (
            f"{goal}/{home}/{mode} seed={seed}: strict-ladder fallback fired "
            "(cheap-bracket vs capability divergence)")
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

    def test_duna_return_kerbin_count(self):
        self._check(*_CONFIGS[1])

    def test_flag_every_body_kerbin(self):
        self._check(*_CONFIGS[2])

    def test_bug_092_duna_return_buildings(self):
        self._check(*_CONFIGS[3])

    def test_bug_101_complete_tech_tree_duna(self):
        self._check(*_CONFIGS[4])

    def test_bug_102_flag_every_body_moho(self):
        self._check(*_CONFIGS[5])

    def test_bug_103_mun_flag_moho(self):
        self._check(*_CONFIGS[6])
