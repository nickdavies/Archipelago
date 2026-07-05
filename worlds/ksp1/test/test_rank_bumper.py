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
from worlds.ksp1.locations import (
    EVENT_BY_NAME, LocationBuilder, LocationDescriptor, MissionLocation,
)
from worlds.ksp1.parts import DEFAULT_PART_MANAGER

PART_DB = DEFAULT_PART_MANAGER.parts
from worlds.ksp1.ranks import DEFAULT_CONTEXT, RankAxisKey, RankContext
from worlds.ksp1.requirements import Rank, Signature
from worlds.ksp1.sphere_ladder import (
    _pre_pass_for_ranks,
    _rank_admits_item,
    minimal_ranks_for,
)


MISSION_BUILDER = MissionBuilder(home=BodyName.KERBIN)


def _descriptor_for(name: str) -> LocationDescriptor:
    """Resolve a canonical location name to its descriptor the same way
    ``create_all_locations`` does — home specials from the LocationBuilder,
    per-body missions from the MissionLocation grammar.  Test scaffolding: the
    generator carries ``loc.descriptor`` off the real object; here we rebuild it
    from a hard-coded canonical name."""
    hloc = LocationBuilder.all_home_locations().get(name)
    if hloc is not None:
        return LocationDescriptor.from_home(hloc)
    ml = MissionLocation.parse(name)
    return LocationDescriptor.from_mission(ml, EVENT_BY_NAME[ml.event])


def _bumper(loc: str, seed: int = 42, prior: Signature = Signature.empty()):
    return minimal_ranks_for(
        _descriptor_for(loc), prior, DEFAULT_CONTEXT,
        difficulty="comfortable",
        progressive_launch_pad=False,
        start_with_clamps=True,
        rng=random.Random(seed),
        mission_builder=MISSION_BUILDER,
    )


class TestRankPrePass(unittest.TestCase):
    def test_empty_admits_only_non_ranked_parts(self) -> None:
        """An empty Signature ceiling rejects every item that has any rank —
        only NON-ranked items pass.  Non-ranked parts (the scorer returned None
        for them on their axis) carry no rank sig, so they slip through at every
        ceiling.  Today that's: the ion engine (off the engine axes — see
        ``_engine_vac``; capability's ``_filter_engines_for_ion`` strips it at
        evaluation time) and structural adapter "tanks" the tank scorers exclude
        (e.g. adapterMk3-Mk2).  The invariant: nothing RANKED is admitted."""
        from worlds.ksp1.ranks import ranks_for_context
        flags = _pre_pass_for_ranks(
            Signature.empty(), DEFAULT_CONTEXT,
            start_with_clamps=True, progressive_launch_pad=False,
            launch_pad_caps=None,
        )
        tank_axes = (RankAxisKey.LFO_TANK, RankAxisKey.LF_TANK,
                     RankAxisKey.XENON_TANK, RankAxisKey.MONOPROP_TANK)
        ranks = ranks_for_context(DEFAULT_CONTEXT)
        ranked_tanks = set().union(*(ranks.get(ax, {}) for ax in tank_axes))
        admitted_ranked = [t.name for t in flags.available_tanks
                           if t.name in ranked_tanks]
        # Only the ion engine is non-ranked on the engine axes.
        self.assertEqual([e.fuel_type for e in flags.available_engines],
                         ["xenon"] * len(flags.available_engines))
        # No RANKED tank may be admitted at the empty ceiling (non-ranked
        # adapters may slip through, like the ion engine).
        self.assertEqual(admitted_ranked, [],
                         f"empty ceiling admitted ranked tanks: {admitted_ranked}")
        self.assertEqual(flags.available_srbs, [])

    def test_max_ranks_admits_full_part_db(self) -> None:
        """With every axis at its maximum bucket, every part participating
        in any rank axis is admitted."""
        from worlds.ksp1.ranks import RANK_AXES, max_rank_for
        all_max = Signature.of(Rank(a.key, max_rank_for(a.key)) for a in RANK_AXES)
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
        axes = {rq.axis for rq in r.signature.rank_reqs}
        self.assertIn(RankAxisKey.LANDING_LEG, axes)

    def test_return_requires_heat_shield(self) -> None:
        """High-speed reentry (a RETURN to home) requires a heat shield —
        verifies the heat-shield rank axis is wired into the bumper.

        Note Duna *landing* does NOT require one: it has a propulsive profile
        (thin atmo, propulsive capture + descent) whose minimal kit the bumper
        prefers over the aero profile, so it gates on engines/tanks, not a heat
        shield.  Reentry to home is where the heat shield is unavoidable."""
        r = _bumper("Mun Return 1")
        self.assertIsNotNone(r)
        axes = {rq.axis for rq in r.signature.rank_reqs}
        self.assertIn(RankAxisKey.HEAT_SHIELD, axes,
                      "a return-to-home reentry must require a heat shield")


class TestRankBumperDeterminism(unittest.TestCase):
    """Same seed + same prior_ranks → same output."""

    def test_repeatable_with_same_seed(self) -> None:
        a = _bumper("Mun Landing 1", seed=1234)
        b = _bumper("Mun Landing 1", seed=1234)
        self.assertEqual(a.signature, b.signature)
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
                with self.subTest(loc=loc, axis=axis.name, rank=rank):
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
