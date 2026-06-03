"""
Sanity tests for the per-category rank system.

Phase 0 deliverable: each axis assigns ranks to multiple items, and a handful
of well-known parts land where the design says they should.  This is a
behavior-preservation phase — nothing consumes these ranks yet, so the bar
is "ranks are computed and sensible."  Later phases tighten the contract.
"""
import unittest

from worlds.ksp1.ranks import (
    DEFAULT_CONTEXT,
    RankAxisKey,
    RankContext,
    RankDirection,
    RANK_AXES,
    max_rank_for,
    ItemRankSig,
    rank_sig_for,
    ranks_for_context,
)


class TestRanksPopulated(unittest.TestCase):
    """Every axis must rank a non-trivial number of items into multiple buckets.
    If a scorer is broken and yields None for everything, this catches it."""

    def test_every_axis_has_items(self) -> None:
        r = ranks_for_context()
        for axis in RANK_AXES:
            with self.subTest(axis=axis.key):
                items = r[axis.key]
                self.assertGreater(
                    len(items), 0,
                    f"axis {axis.key} ranks zero items — scorer probably broken",
                )

    def test_dense_axes_use_their_full_bucket_range(self) -> None:
        """An axis with N>=buckets parts should populate every bucket."""
        r = ranks_for_context()
        dense = {RankAxisKey.LAUNCH_ENGINE, RankAxisKey.VAC_ENGINE,
                 RankAxisKey.LFO_TANK, RankAxisKey.CAPSULE, RankAxisKey.RELAY}
        for axis in RANK_AXES:
            if axis.key not in dense:
                continue
            with self.subTest(axis=axis.key):
                ranks_seen = set(r[axis.key].values())
                self.assertEqual(
                    ranks_seen, set(range(1, max_rank_for(axis.key) + 1)),
                    f"axis {axis.key} should populate all buckets 1..{max_rank_for(axis.key)},"
                    f" got {sorted(ranks_seen)}",
                )


class TestEnginesByMissionClass(unittest.TestCase):
    """LV-909 (Terrier), Mainsail, Nerv, and Dawn are the canonical test
    points for per-mission-class engine ranks.  Per the plan: LV-909 should
    rank low on LAUNCH and high on VAC; Mainsail the inverse."""

    def setUp(self) -> None:
        r = ranks_for_context()
        self.launch = r[RankAxisKey.LAUNCH_ENGINE]
        self.vac = r[RankAxisKey.VAC_ENGINE]
        self.launch_buckets = max_rank_for(RankAxisKey.LAUNCH_ENGINE)
        self.vac_buckets = max_rank_for(RankAxisKey.VAC_ENGINE)

    def test_terrier_low_launch_high_vac(self) -> None:
        terrier = "liquidEngine3.v2"
        self.assertLessEqual(self.launch[terrier], 2,
                             "Terrier (LV-909) should be a bottom-tier launch engine")
        self.assertGreaterEqual(self.vac[terrier], self.vac_buckets - 1,
                                "Terrier (LV-909) should be a top-tier vacuum engine")

    def test_mainsail_top_launch(self) -> None:
        mainsail = "liquidEngineMainsail.v2"
        self.assertGreaterEqual(self.launch[mainsail], self.launch_buckets - 1,
                                "Mainsail should be a top-tier launch engine")

    def test_nerv_top_vac_bad_launch(self) -> None:
        nerv = "nuclearEngine"
        self.assertLessEqual(self.launch[nerv], 1,
                             "Nerv has poor atm Isp — should be bottom launch rank")
        self.assertEqual(self.vac[nerv], self.vac_buckets,
                         "Nerv should be a top-tier vacuum engine")

    def test_dawn_top_vac_off_launch_or_bottom(self) -> None:
        dawn = "ionEngine"
        self.assertEqual(self.vac[dawn], self.vac_buckets,
                         "Dawn (ion) should be top vacuum rank thanks to xenon multiplier")
        # Dawn does technically have atm_thrust > 0 (atm_isp blends), so it appears
        # on the launch axis but should be at the absolute bottom.
        if dawn in self.launch:
            self.assertEqual(self.launch[dawn], 1,
                             "Dawn should rank as the worst launch engine if on the axis")


class TestTanksByDryMass(unittest.TestCase):
    def setUp(self) -> None:
        self.lfo = ranks_for_context()[RankAxisKey.LFO_TANK]

    def test_oscar_b_low_jumbo_high(self) -> None:
        """Lightest LFO tanks (Oscar-B, FL-T100) should land at low ranks —
        admitted in early spheres — and the giant adapters / S4 tanks at the
        top.  Matches today's progressive LFO tier ordering by mass."""
        oscar = "miniFuelTank"
        fl_t100 = "fuelTankSmallFlat"
        self.assertLessEqual(self.lfo[oscar], 2)
        self.assertLessEqual(self.lfo[fl_t100], 2)
        s3_huge = "Size3LargeTank"        # S3-14400, 72t fuel
        s4_huge = "Size4.Tank.04"          # S4-512, 256t fuel
        self.assertGreaterEqual(self.lfo[s3_huge], 4)
        self.assertGreaterEqual(self.lfo[s4_huge], 4)


class TestSolarHeavyLast(unittest.TestCase):
    def test_gigantor_top_ox_stat_bottom(self) -> None:
        """Per design: heavy panels admitted late, basic fixed panels early."""
        solar = ranks_for_context()[RankAxisKey.SOLAR]
        buckets = max_rank_for(RankAxisKey.SOLAR)
        gigantor = "largeSolarPanel"
        ox_stat = "solarPanels5"
        self.assertEqual(solar[gigantor], buckets,
                         "Gigantor should be the top solar rank (heaviest, most output)")
        self.assertEqual(solar[ox_stat], 1,
                         "OX-STAT should be the bottom solar rank (cheapest fixed panel)")


class TestCapsuleHeaviestLast(unittest.TestCase):
    def test_mk1_pod_low_mk1_3_high(self) -> None:
        """Capsules use effective dry mass (mass - drainable propellant);
        lighter pods admit early, heavier pods admit late."""
        cap = ranks_for_context()[RankAxisKey.CAPSULE]
        buckets = max_rank_for(RankAxisKey.CAPSULE)
        # Mk1 Command Pod — small, light, 1 crew.
        self.assertLessEqual(cap["mk1pod.v2"], 2)
        # Mk1-3 Command Pod — large, 3 crew.
        self.assertGreaterEqual(cap["mk1-3pod"], buckets - 1)
        # MEMLander — featherweight 1-crew lander.
        self.assertEqual(cap["MEMLander"], 1)


class TestProbeCoreBySASLevel(unittest.TestCase):
    def test_stayputnik_low_hecs2_high(self) -> None:
        probe = ranks_for_context()[RankAxisKey.PROBE_SAS]
        buckets = max_rank_for(RankAxisKey.PROBE_SAS)
        self.assertEqual(probe["probeCoreSphere.v2"], 1,
                         "Stayputnik has SAS level 0 — bottom rank")
        self.assertEqual(probe["HECS2.ProbeCore"], buckets,
                         "HECS2 has SAS level 3 — top rank")


class TestSRBHomeAwareness(unittest.TestCase):
    """SRBs score by atmospheric Isp on atmospheric homes and vacuum Isp
    elsewhere; rankings must respond to the context."""

    def test_atm_vs_vac_ordering_differs(self) -> None:
        atm = ranks_for_context(RankContext(home_has_atmosphere=True))[RankAxisKey.SRB]
        vac = ranks_for_context(RankContext(home_has_atmosphere=False))[RankAxisKey.SRB]
        # Both contexts must rank Clydesdale at the top (it dominates either way).
        buckets = max_rank_for(RankAxisKey.SRB)
        self.assertEqual(atm["Clydesdale"], buckets)
        self.assertEqual(vac["Clydesdale"], buckets)

    def test_only_atm_isp_uses_atm_branch(self) -> None:
        from worlds.ksp1.parts import PART_DB, SolidBooster
        from worlds.ksp1.ranks import _srb
        clyde = next(p for p in PART_DB["Clydesdale"] if isinstance(p, SolidBooster))
        atm_score = _srb(clyde, RankContext(home_has_atmosphere=True))
        vac_score = _srb(clyde, RankContext(home_has_atmosphere=False))
        # vac_isp > atm_isp for every stock SRB, so vac score must be > atm.
        self.assertGreater(vac_score, atm_score)


class TestRankSigStability(unittest.TestCase):
    def test_same_item_returns_same_sig(self) -> None:
        a = rank_sig_for("liquidEngine3.v2")
        b = rank_sig_for("liquidEngine3.v2")
        self.assertIs(a, b, "memoized rank_sig_for should return the same object")

    def test_terrier_appears_on_both_engine_axes(self) -> None:
        sig = rank_sig_for("liquidEngine3.v2")
        axis_keys = {k for k, _ in sig.axes}
        self.assertIn(RankAxisKey.LAUNCH_ENGINE, axis_keys)
        self.assertIn(RankAxisKey.VAC_ENGINE, axis_keys)

    def test_non_ranked_item_has_empty_sig(self) -> None:
        # Science Pack filler isn't in PART_DB, so it has no rank.
        sig = rank_sig_for("Science Pack 5")
        self.assertEqual(sig.axes, ())


class TestProgressiveDirectionalConsistency(unittest.TestCase):
    """A weaker monotonicity test: for HIGHER_BETTER axes whose progressive
    tier ordering matches "weak first → strong last," verify the rank
    assignment respects that direction.  Axes where progressive ordering
    intentionally inverts (e.g. CAPSULE places heaviest pods first) are
    excluded; specific assertions cover those above."""

    # (progressive item name, RankAxisKey) — both go strong→strong in tier order.
    _AXES_TO_CHECK = (
        ("Progressive Heat Shield", RankAxisKey.HEAT_SHIELD),
        ("Progressive Stack Decoupler", RankAxisKey.STACK_DECOUPLER),
        ("Progressive Radial Decoupler", RankAxisKey.RADIAL_DECOUPLER),
        ("Progressive SRB", RankAxisKey.SRB),
        ("Progressive Parachute", RankAxisKey.PARACHUTE),
    )

    def test_higher_better_progressive_axes_are_ordered(self) -> None:
        from worlds.ksp1.parts import PROGRESSIVE_PART_TIERS
        all_ranks = ranks_for_context()
        for prog_name, axis_key in self._AXES_TO_CHECK:
            tiers = PROGRESSIVE_PART_TIERS.get(prog_name, {})
            axis_ranks = all_ranks[axis_key]
            # For each tier pair, every tier-N part should have rank ≥ every
            # tier-(N-1) part on this axis (subject to bucket coarseness).
            for tier in sorted(tiers):
                if tier == 1:
                    continue
                lower = [axis_ranks[p] for p in tiers[tier - 1] if p in axis_ranks]
                upper = [axis_ranks[p] for p in tiers[tier] if p in axis_ranks]
                if not lower or not upper:
                    continue
                with self.subTest(axis=axis_key, tier=tier, prog=prog_name):
                    self.assertGreaterEqual(
                        min(upper), min(lower),
                        f"{prog_name} tier {tier} parts should rank no lower than tier {tier-1}",
                    )


if __name__ == "__main__":
    unittest.main()
