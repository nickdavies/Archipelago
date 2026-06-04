"""
Tests for the Phase 1 rank-space sphere walker (scaffold).

These verify that ``minimal_ranks_for`` and ``_pre_pass_for_ranks`` produce
sensible output on the canonical predictable anchors.  The walker runs
alongside the progressive walker — its placement output is not consumed
yet, so these tests check the walker's internal correctness rather than
solve-rate impact.
"""
from __future__ import annotations

import random
import unittest

from worlds.ksp1.bodies import BodyName, MissionBuilder
from worlds.ksp1.parts import PART_DB
from worlds.ksp1.ranks import DEFAULT_CONTEXT, RankAxisKey, RankContext
from worlds.ksp1.sphere_ladder import (
    MinimumRanks,
    _pre_pass_for_ranks,
    _rank_admits_item,
    minimal_ranks_for,
)


MISSION_BUILDER = MissionBuilder(home=BodyName.KERBIN)


def _bumper(loc: str, seed: int = 42, prior: MinimumRanks = MinimumRanks.empty()):
    return minimal_ranks_for(
        loc, prior, DEFAULT_CONTEXT,
        difficulty="normal",
        progressive_launch_pad=False,
        start_with_clamps=True,
        rng=random.Random(seed),
        mission_builder=MISSION_BUILDER,
    )


class TestRankPrePass(unittest.TestCase):
    def test_empty_admits_only_non_ranked_parts(self) -> None:
        """An empty MinimumRanks ceiling rejects every item that has any
        rank — only non-ranked items pass.  Ion is the lone non-ranked
        engine (it's out of logic, so it's off the engine axes; see
        ``_engine_vac``), so it slips through the rank gate here — but
        capability's ``_filter_engines_for_ion`` strips it at evaluation
        time, keeping it out of logic in practice."""
        flags = _pre_pass_for_ranks(
            MinimumRanks.empty(), DEFAULT_CONTEXT,
            start_with_clamps=True, progressive_launch_pad=False,
            launch_pad_caps=None,
        )
        self.assertEqual([e.fuel_type for e in flags.available_engines],
                         ["xenon"] * len(flags.available_engines))
        self.assertEqual(flags.available_tanks, [])
        self.assertEqual(flags.available_srbs, [])

    def test_max_ranks_admits_full_part_db(self) -> None:
        """With every axis at its maximum bucket, every part participating
        in any rank axis is admitted."""
        from worlds.ksp1.ranks import RANK_AXES, max_rank_for
        all_max = MinimumRanks(tuple((a.key, max_rank_for(a.key)) for a in RANK_AXES))
        flags = _pre_pass_for_ranks(
            all_max, DEFAULT_CONTEXT,
            start_with_clamps=True, progressive_launch_pad=False,
            launch_pad_caps=None,
        )
        self.assertGreater(len(flags.available_engines), 20)
        self.assertGreater(len(flags.available_tanks), 30)
        self.assertGreater(len(flags.available_srbs), 5)
        # Derived binary gates set correctly.
        self.assertTrue(flags.has_launch_engine)
        self.assertTrue(flags.has_lfo_fuel)
        self.assertTrue(flags.has_srb_fuel)


class TestRankBumperFeasibility(unittest.TestCase):
    """Bumper must converge on every predictable Kerbin-home anchor."""

    def test_first_launch_feasible(self) -> None:
        r = _bumper("Kerbin First Launch")
        self.assertIsNotNone(r)

    def test_kerbin_orbit_feasible(self) -> None:
        r = _bumper("Kerbin Orbit 1")
        self.assertIsNotNone(r)
        self.assertGreater(len(r.reps), 0)

    def test_mun_landing_feasible(self) -> None:
        r = _bumper("Mun Landing 1")
        self.assertIsNotNone(r)
        # Landing missions should pick a landing-leg axis bump.
        axes = {k for k, _ in r.ranks.upper_bounds}
        self.assertIn(RankAxisKey.LANDING_LEG, axes)

    def test_duna_landing_picks_heat_shield(self) -> None:
        r = _bumper("Duna Landing 1")
        self.assertIsNotNone(r)
        axes = {k for k, _ in r.ranks.upper_bounds}
        self.assertIn(RankAxisKey.HEAT_SHIELD, axes,
                      "Duna landing must require a heat shield")


class TestRankBumperDeterminism(unittest.TestCase):
    """Same seed + same prior_ranks → same output."""

    def test_repeatable_with_same_seed(self) -> None:
        a = _bumper("Mun Landing 1", seed=1234)
        b = _bumper("Mun Landing 1", seed=1234)
        self.assertEqual(a.ranks, b.ranks)
        self.assertEqual(a.reps, b.reps)

    def test_different_seeds_diverge(self) -> None:
        a = _bumper("Mun Landing 1", seed=1)
        b = _bumper("Mun Landing 1", seed=2)
        # Different RNGs should pick different reps for at least one bump.
        # (They might converge on the same ranks though.)
        self.assertNotEqual(a.reps, b.reps,
                            "different seeds should pick different reps")


class TestRankBumperReps(unittest.TestCase):
    """Every recorded rep must name a real PART_DB item."""

    def test_every_rep_is_a_real_part(self) -> None:
        for loc in ["Kerbin First Launch", "Kerbin Orbit 1",
                    "Mun Landing 1", "Duna Landing 1"]:
            r = _bumper(loc)
            self.assertIsNotNone(r, f"bumper failed on {loc}")
            for (axis, rank), name in r.reps.items():
                with self.subTest(loc=loc, axis=axis, rank=rank):
                    self.assertIn(name, PART_DB,
                                  f"{loc}: rep {name!r} for ({axis}, {rank}) "
                                  f"not in PART_DB")


class TestRankBumperSRBContext(unittest.TestCase):
    """The SRB axis is home-aware — atmospheric homes use atm_isp, vacuum
    homes use vac_isp.  Bumping it should give the right rep on either."""

    def test_atmospheric_home_uses_atm_branch(self) -> None:
        # Kerbin context: atmospheric.
        atm_ctx = RankContext(home_has_atmosphere=True)
        vac_ctx = RankContext(home_has_atmosphere=False)
        # Same prior, same RNG seed, same location → potentially different
        # picks because the rank-table value (which rep is at which rank)
        # differs between contexts.  Just verify both produce results.
        from worlds.ksp1.sphere_ladder import _items_at_rank
        atm = _items_at_rank(atm_ctx)[RankAxisKey.SRB]
        vac = _items_at_rank(vac_ctx)[RankAxisKey.SRB]
        # Both maps must populate.
        self.assertGreater(sum(len(v) for v in atm.values()), 0)
        self.assertGreater(sum(len(v) for v in vac.values()), 0)


if __name__ == "__main__":
    unittest.main()
