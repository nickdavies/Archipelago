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

    def test_dawn_off_engine_axes_out_of_logic(self) -> None:
        # Ion (Dawn) is out of logic: its ~4200s Isp would dominate every
        # dv-bound mission and flatten per-seed variance, and it's never
        # actually required (every mission is reachable non-ion).  So it's
        # left off the engine rank axes entirely -- capability ignores it
        # too -- and stays an out-of-logic bonus part.
        dawn = "ionEngine"
        self.assertNotIn(dawn, self.vac, "ion must be off the vacuum engine axis")
        self.assertNotIn(dawn, self.launch, "ion must be off the launch engine axis")


class TestTanksByDryMass(unittest.TestCase):
    def setUp(self) -> None:
        self.lfo = ranks_for_context()[RankAxisKey.LFO_TANK]

    def test_oscar_b_low_jumbo_high(self) -> None:
        """LFO is a FUNGIBLE axis, deliberately capped at 2 ranks
        (``_FUNGIBLE_AXIS_CAP``): the lightest tanks (Oscar-B, FL-T100) gate at
        rank 1 (the early building block) and the giant S3/S4 tanks at the top
        rank — there is intentionally no deeper ladder (small tanks just stack
        to any total; the launch-pad mass cap is the real size limiter)."""
        top = max_rank_for(RankAxisKey.LFO_TANK)  # 2 for fungible axes
        oscar = "miniFuelTank"
        fl_t100 = "fuelTankSmallFlat"
        self.assertEqual(self.lfo[oscar], 1)
        self.assertEqual(self.lfo[fl_t100], 1)
        s3_huge = "Size3LargeTank"        # S3-14400, 72t fuel
        s4_huge = "Size4.Tank.04"          # S4-512, 256t fuel
        self.assertEqual(self.lfo[s3_huge], top)
        self.assertEqual(self.lfo[s4_huge], top)


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


class TestCapsuleLightestBest(unittest.TestCase):
    def test_lightest_pods_rank_highest(self) -> None:
        """Capsules score by effective dry mass (mass - drainable propellant),
        LOWER_BETTER: the lightest pods are the "best" parts and land at the TOP
        ranks (admitted late, as a reward), while heavy multi-crew pods sit at
        low ranks (the forced early bootstrap). This is the same
        lower-dry-mass-is-better pattern shared with tanks, landing legs, and
        SAS modules."""
        cap = ranks_for_context()[RankAxisKey.CAPSULE]
        buckets = max_rank_for(RankAxisKey.CAPSULE)
        # MEMLander — featherweight 1-crew lander — the lightest, so top rank.
        self.assertEqual(cap["MEMLander"], buckets)
        # Mk1 Command Pod — small, light, 1 crew — near the top.
        self.assertGreaterEqual(cap["mk1pod.v2"], buckets - 1)
        # Mk1-3 Command Pod — large, heavy, 3 crew — low rank (early bootstrap).
        self.assertLessEqual(cap["mk1-3pod"], 2)


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


class TestSparseAxisOrdering(unittest.TestCase):
    """Ordering guards for the sparse HIGHER_BETTER axes that no other test
    pins (engines / tanks / solar / capsule / probe-core have their own).

    These assert physically-meaningful orderings *derived from part
    properties* — a scorer sign flip, a bucket collapse, or a bad property
    read would silently reorder these axes, distorting bumper pacing and
    risking unsatisfiable early spheres.  Self-contained: no dependency on
    the retired progressive tiers (anchors are concrete PART_DB names).
    """

    def setUp(self) -> None:
        self.r = ranks_for_context()

    def test_heat_shield_bigger_ranks_higher(self) -> None:
        # scorer: size_class — a 0.625m shield must rank below a 3.75m shield.
        hs = self.r[RankAxisKey.HEAT_SHIELD]
        self.assertLess(hs["HeatShield0"], hs["HeatShield3"],
                        "smallest heat shield must rank below the largest")
        self.assertLess(hs["HeatShield1"], hs["HeatShield2"])

    def test_srb_bigger_impulse_ranks_higher(self) -> None:
        # scorer: atm_isp * fuel_mass (home-aware) — separatron below Clydesdale.
        srb = self.r[RankAxisKey.SRB]
        self.assertLess(srb["sepMotor1"], srb["Clydesdale"],
                        "tiny separator SRB must rank below the largest booster")
        self.assertLess(srb["solidBooster.v2"], srb["Thoroughbred"])

    def test_parachute_radial_outranks_inline(self) -> None:
        # scorer: (drag_area / mass) * radial_bonus — radials beat inline chutes.
        ch = self.r[RankAxisKey.PARACHUTE]
        self.assertGreater(ch["parachuteRadial"], ch["parachuteSingle"],
                           "radial chute should outrank the inline single chute")
        self.assertLess(ch["parachuteDrogue"], ch["parachuteRadial"])

    def test_radial_decoupler_fuelline_is_top(self) -> None:
        # crossfeed fuel line is the top bucket (enables asparagus staging).
        rd = self.r[RankAxisKey.RADIAL_DECOUPLER]
        self.assertEqual(rd["fuelLine"], max_rank_for(RankAxisKey.RADIAL_DECOUPLER),
                         "fuelLine must be the top radial-decoupler bucket")
        self.assertLess(rd["radialDecoupler"], rd["fuelLine"])

    def test_stack_decoupler_bigger_ranks_higher(self) -> None:
        # scorer: size_class — smallest stack decoupler below the largest.
        sd = self.r[RankAxisKey.STACK_DECOUPLER]
        self.assertLess(sd["Decoupler.0"], sd["Decoupler.4"],
                        "smallest stack decoupler must rank below the largest")


if __name__ == "__main__":
    unittest.main()
