"""
Contract system tests.

Covers the shared feasibility check (the single most safety-critical point — a
generation false positive yields an unsolvable seed), the required-part manifest,
slot_data round-trip, and the three-gate access rule under a full world setup.
"""
import math
import unittest

import random as _random

from worlds.ksp1 import contracts as C
from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, BODY_BY_NAME, MissionBuilder, DIFFICULTY_PROFILES,
    GameplayDifficulty, MissionType,
    generate_random_orbit_params,
    generate_rescue_orbit_params, generate_surface_rescue_site_lats,
    generate_survey_site_lats,
    generate_tourist_manifests, precision_landing_dv,
)
from worlds.ksp1.capability import _pre_pass
from worlds.ksp1.test.base import KSP1TestBase

MB = MissionBuilder(home=BodyName.KERBIN)
# RANDOM_ORBIT / KERBAL_RESCUE / SURFACE_RESCUE contracts read their seeded
# target orbit / site latitude off the mission_builder; populate them so every
# type can build parameters in tests.
MB.random_orbit_params = generate_random_orbit_params(_random.Random(0), ALL_BODIES)
MB.rescue_orbit_params = generate_rescue_orbit_params(_random.Random(0), ALL_BODIES)
MB.surface_rescue_site_lats = generate_surface_rescue_site_lats(
    _random.Random(0), ALL_BODIES)
MB.survey_site_lats = generate_survey_site_lats(_random.Random(0), ALL_BODIES)
MB.tourist_manifests = generate_tourist_manifests(
    _random.Random(0), ALL_BODIES)
DIFF = DIFFICULTY_PROFILES["comfortable"]


def _flags(item_count_fn):
    return _pre_pass(item_count_fn, start_with_clamps=True,
                     progressive_launch_pad=False,
                     launch_pad_caps=MB.launch_pad_caps)


FULL = _flags(lambda n: 99)
EMPTY = _flags(lambda n: 0)
# Drill + ore tank present, but no propulsion / command part → can't deliver.
DRILL_ONLY = _flags(lambda n: 1 if n in ("MiniDrill", "RadialOreTank") else 0)
# Drill but NO ore tank, full rocket otherwise.
NO_TANK = _flags(lambda n: 0 if n in ("RadialOreTank", "SmallTank", "LargeTank") else 99)

MUN_MINE = C.ContractSpec(C.ContractType.MINE_ORE, BodyName.MUN)


def _first_compatible_body(td):
    """A body the contract type can target — for tests that must instantiate
    every type without caring which body."""
    for body in ALL_BODIES:
        if td.body_compatible(body):
            return body.name
    return None


class TestPreciseOrbitAttitude(unittest.TestCase):
    """Precise-pointing contracts (specific orbits, stations, rescues) must hold a
    fixed attitude with the engine off, so on casual/normal difficulty they need a
    reaction wheel or RCS — engine gimbal alone isn't enough. Expert is trusted to
    fly them on gimbal. This is a GAMEPLAY-difficulty gate (base Difficulty),
    orthogonal to the physics profile — carried on ``mission_builder.gameplay``.
    Generic ORBIT (any shape) never needs it.
    """

    ASSIST = GameplayDifficulty(precise_pointing_needs_reaction_control=True,
                                srb_needs_rcs=True,
                                docking_needs_rcs=True)   # casual / normal
    NO_ASSIST = GameplayDifficulty(precise_pointing_needs_reaction_control=False,
                                   srb_needs_rcs=False,
                                   docking_needs_rcs=False)  # expert

    def _mb(self, gameplay):
        mb = MissionBuilder(home=BodyName.KERBIN)
        mb.gameplay = gameplay
        mb.random_orbit_params = generate_random_orbit_params(_random.Random(0), ALL_BODIES)
        mb.rescue_orbit_params = generate_rescue_orbit_params(_random.Random(0), ALL_BODIES)
        return mb

    def _gimbal_only(self):
        """A full kit (reaches any Kerbin orbit, gimballed engines available) but
        with NO reaction wheel and NO RCS — attitude comes solely from gimbal."""
        g = _flags(lambda n: 99)
        g.has_reaction_wheels = False
        g.has_rcs = False
        g.lightest_reaction_wheel = None
        g.lightest_rcs_thruster = None
        return g

    def test_precise_orbit_blocked_on_casual_normal_gimbal_only(self):
        g = self._gimbal_only()
        mb = self._mb(self.ASSIST)
        for ct in C.PRECISE_ORBIT_TYPES:
            spec = C.ContractSpec(ct, BodyName.KERBIN)
            self.assertFalse(
                C.can_complete_contract(spec, g, DIFF, mb),
                f"{ct} on gimbal-only should be blocked when the assist is on")

    def test_precise_orbit_allowed_on_expert_gimbal_only(self):
        # Proves the precise-attitude gate is the SOLE differentiator: the same
        # gimbal-only kit (same physics) flies these orbits once the assist is off.
        g = self._gimbal_only()
        mb = self._mb(self.NO_ASSIST)
        for ct in C.PRECISE_ORBIT_TYPES:
            spec = C.ContractSpec(ct, BodyName.KERBIN)
            self.assertTrue(
                C.can_complete_contract(spec, g, DIFF, mb),
                f"{ct} should fly on gimbal alone when the assist is off")

    def test_generic_orbit_never_needs_wheel(self):
        g = self._gimbal_only()
        spec = C.ContractSpec(C.ContractType.ORBIT, BodyName.KERBIN)
        for gp in (self.ASSIST, self.NO_ASSIST):
            self.assertTrue(
                C.can_complete_contract(spec, g, DIFF, self._mb(gp)),
                "generic (any-shape) ORBIT must never require a wheel/RCS")

    def test_wheel_or_rcs_restores_precise_orbit_on_casual(self):
        spec = C.ContractSpec(C.ContractType.EQUATORIAL_ORBIT, BodyName.KERBIN)
        mb = self._mb(self.ASSIST)
        with_wheel = self._gimbal_only()
        with_wheel.has_reaction_wheels = True
        self.assertTrue(C.can_complete_contract(spec, with_wheel, DIFF, mb))
        with_rcs = self._gimbal_only()
        with_rcs.has_rcs = True
        self.assertTrue(C.can_complete_contract(spec, with_rcs, DIFF, mb))

    def test_station_and_rescue_also_need_precise_pointing(self):
        """SPACE_STATION (large crewed vessel holding a service orbit) and
        KERBAL_RESCUE (fine approach) join the precise-pointing set — same
        wheel/RCS-on-assist, gimbal-when-off rule as the specific orbits.
        RESCUE additionally keeps its existing navigation (rendezvous) gate."""
        g = self._gimbal_only()
        for ct in (C.ContractType.SPACE_STATION, C.ContractType.KERBAL_RESCUE):
            self.assertIn(ct, C.PRECISE_POINTING_TYPES)
            body = _first_compatible_body(C.CONTRACT_TYPE_DEFS[ct])
            spec = C.ContractSpec(ct, body)
            self.assertFalse(
                C.can_complete_contract(spec, g, DIFF, self._mb(self.ASSIST)),
                f"{ct} on gimbal-only should be blocked when the assist is on")
            # With the assist off, gimbal alone flies it → attitude is the sole blocker.
            self.assertTrue(
                C.can_complete_contract(spec, g, DIFF, self._mb(self.NO_ASSIST)),
                f"{ct} should fly gimbal-only with the assist off (so the block "
                f"above is the precise-attitude gate, not some other missing part)")

    def test_generic_precise_pointing_excludes_plain_missions(self):
        """The precise-pointing set must NOT sweep in ordinary missions — a
        generic ORBIT / a mine-ore landing never needs a wheel."""
        for ct in (C.ContractType.ORBIT, C.ContractType.MINE_ORE,
                   C.ContractType.TRANSMIT_SCIENCE):
            self.assertNotIn(ct, C.PRECISE_POINTING_TYPES)


class TestRequiredPartManifest(unittest.TestCase):
    def test_full_kit_returns_lightest_per_category(self):
        manifest = C.required_part_manifest(MUN_MINE, FULL)
        self.assertIsNotNone(manifest)
        names = {p.name for p in manifest}
        # Lightest drill = MiniDrill (0.25t), lightest ore tank = RadialOreTank (0.125t).
        self.assertEqual(names, {"MiniDrill", "RadialOreTank"})

    def test_missing_category_returns_none(self):
        self.assertIsNone(C.required_part_manifest(MUN_MINE, EMPTY))
        self.assertIsNone(C.required_part_manifest(MUN_MINE, NO_TANK))


class TestCanCompleteContract(unittest.TestCase):
    """The shared feasibility truth table."""

    def test_full_kit_can_mine_mun(self):
        self.assertTrue(C.can_complete_contract(MUN_MINE, FULL, DIFF, MB))

    def test_no_parts_cannot_mine(self):
        self.assertFalse(C.can_complete_contract(MUN_MINE, EMPTY, DIFF, MB))

    def test_drill_without_tank_cannot_mine(self):
        self.assertFalse(C.can_complete_contract(MUN_MINE, NO_TANK, DIFF, MB))

    def test_mining_kit_without_rocket_cannot_deliver(self):
        # Has drill + ore tank but no way to reach Mun's surface.
        self.assertFalse(C.can_complete_contract(MUN_MINE, DRILL_ONLY, DIFF, MB))


class TestSlotDataRoundTrip(unittest.TestCase):
    def test_to_from_slot_dict(self):
        d = MUN_MINE.to_slot_dict()
        self.assertEqual(d["item"], "Contract: Mine Ore on Mun")
        # Non-goal contracts ship two slot-suffixed reward locations.
        self.assertEqual(
            d["locations"],
            ["Contract: Mine Ore on Mun 1", "Contract: Mine Ore on Mun 2"],
        )
        self.assertEqual(d["schema"], C.CONTRACT_SCHEMA_VERSION)
        kinds = [p["kind"] for p in d["parameters"]]
        self.assertEqual(kinds, ["situation", "resource"])
        self.assertEqual(C.ContractSpec.from_slot_dict(d), MUN_MINE)

    def test_goal_contract_single_location(self):
        # Goal contracts stay 1:1 — one unsuffixed reward location.
        goal = C.ContractSpec(C.ContractType.RETURN, BodyName.DUNA, is_goal=True)
        d = goal.to_slot_dict()
        self.assertEqual(d["locations"], ["Contract: Return from Duna"])
        self.assertEqual(goal.location_name, "Contract: Return from Duna")

    def test_every_type_round_trips(self):
        # to_slot_dict/from_slot_dict must rebuild an identical spec for EVERY
        # contract type (UT regen reconstructs from fields, never re-randomizes),
        # and to_slot_dict must build a non-empty parameter tree for each type
        # without raising NotImplementedError.
        from worlds.ksp1.bodies import (
            ALL_BODIES, MissionBuilder, BodyName, generate_random_orbit_params,
            generate_rescue_orbit_params, generate_surface_rescue_site_lats,
            generate_survey_site_lats,
            generate_tourist_manifests,
        )
        import random as _random
        mb = MissionBuilder(home=BodyName.KERBIN)
        # Every seeded type reads its target orbit / site latitude / survey site
        # off the mb.
        mb.random_orbit_params = generate_random_orbit_params(
            _random.Random(1), ALL_BODIES)
        mb.rescue_orbit_params = generate_rescue_orbit_params(
            _random.Random(1), ALL_BODIES)
        mb.surface_rescue_site_lats = generate_surface_rescue_site_lats(
            _random.Random(1), ALL_BODIES)
        mb.survey_site_lats = generate_survey_site_lats(
            _random.Random(1), ALL_BODIES)
        mb.tourist_manifests = generate_tourist_manifests(
            _random.Random(1), ALL_BODIES)
        for ct, td in C.CONTRACT_TYPE_DEFS.items():
            with self.subTest(contract_type=ct.name):
                body = _first_compatible_body(td)
                self.assertIsNotNone(body, f"{ct} has no compatible body")
                spec = C.ContractSpec(ct, body)
                d = spec.to_slot_dict(mb)
                self.assertTrue(d["parameters"], f"{ct} emitted no parameters")
                self.assertEqual(C.ContractSpec.from_slot_dict(d), spec)

    def test_random_orbit_home_cost_modeled(self):
        # The extra orbit cost (inclination rotation loss + apoapsis raise) MUST
        # be modeled at the home body: a clearly inclined/raised home orbit costs
        # more than a plain home orbit.
        from worlds.ksp1.bodies import MissionBuilder, RandomOrbitParams
        mb = MissionBuilder(home=BodyName.KERBIN)
        kerbin = next(b for b in ALL_BODIES if b.name == BodyName.KERBIN)
        mb.random_orbit_params[BodyName.KERBIN] = RandomOrbitParams(
            inclination_deg=45.0, sma_m=kerbin.lo_radius_m * 1.5, eccentricity=0.2)
        base = C.evaluate_contract(
            C.ContractSpec(C.ContractType.ORBIT, BodyName.KERBIN), FULL, DIFF, mb)
        rand = C.evaluate_contract(
            C.ContractSpec(C.ContractType.RANDOM_ORBIT, BodyName.KERBIN), FULL, DIFF, mb)
        self.assertIsNotNone(rand)
        self.assertGreater(rand.launch_mass, base.launch_mass,
                           "inclined/raised home orbit must cost more than a plain orbit")

    def test_kerbal_rescue_params_and_free_seat(self):
        # Rescue emits the spawn primitive + a crew-cabin free-seat objective,
        # and is infeasible without a crew cabin to bring the Kerbal home.
        spec = C.ContractSpec(C.ContractType.KERBAL_RESCUE, BodyName.MUN)
        kinds = [p.to_json()["kind"] for p in spec.type_def.build_parameters(BodyName.MUN)]
        self.assertIn("rescue", kinds)
        self.assertIn("has_any_part", kinds)  # the crew_cabin free seat
        # Feasible with full kit; the crew_cabin category must resolve a part.
        self.assertTrue(C.can_complete_contract(spec, FULL, DIFF, MB))
        self.assertIsNotNone(C.required_part_manifest(spec, FULL))
        # No crew part available at all -> the free seat can't be delivered.
        no_crew = _flags(lambda n: 0)
        self.assertIsNone(C.required_part_manifest(spec, no_crew))

    def test_kerbal_rescue_no_target_landing(self):
        # The rescue trajectory reaches the target's LOW ORBIT and returns from
        # there -- it must never include the target's surface (no landing leg).
        from worlds.ksp1.bodies import MissionType
        for profile in MB.profiles_for(BodyName.MUN, MissionType.RESCUE):
            nodes = {e.source for e in profile} | {e.destination for e in profile}
            self.assertNotIn("mun_surface", nodes,
                             "rescue must not land at the target body")
            self.assertIn("mun_low_orbit", nodes)

    def test_rescue_orbit_is_collision_safe(self):
        # The seeded rescue orbit must clear every moon's full PeR..ApR range
        # (plus the moon's SOI) and sit above low orbit / below the SOI, across
        # many seeds, for planets with moons and moonless bodies alike.
        margin = 0.0  # allow R right at a band edge
        for seed in range(60):
            params = generate_rescue_orbit_params(_random.Random(seed), ALL_BODIES)
            for tgt in (BodyName.JOOL, BodyName.KERBIN, BodyName.DUNA,
                        BodyName.EVE, BodyName.MOHO, BodyName.LAYTHE):
                R = params[tgt]
                B = BODY_BY_NAME[tgt]
                self.assertGreaterEqual(R, B.lo_radius_m - 1.0,
                                        f"{tgt} R below low orbit (seed {seed})")
                self.assertLessEqual(R, B.soi_radius_km * 1000.0,
                                     f"{tgt} R outside SOI (seed {seed})")
                for m in ALL_BODIES:
                    if m.parent != tgt:
                        continue
                    lo = m.parent_periapsis_km * 1000.0 - m.soi_radius_km * 1000.0
                    hi = m.parent_apoapsis_km * 1000.0 + m.soi_radius_km * 1000.0
                    self.assertFalse(
                        lo - margin <= R <= hi + margin,
                        f"{tgt} rescue R={R:.0f} clips {m.name}'s "
                        f"[{lo:.0f},{hi:.0f}] band (seed {seed})")

    def test_rescue_dv_regime_general_vs_child_parent(self):
        # transform_mission charges the round trip to the seeded orbit: the
        # general case as 2*raise_dv(low<->R); the child->parent case as a
        # 2*Hohmann between the home moon's orbital radius and R (much cheaper,
        # since you never descend to the parent's low orbit).
        td = C.CONTRACT_TYPE_DEFS[C.ContractType.KERBAL_RESCUE]

        def bump(home, tgt, R):
            mb = MissionBuilder(home=home)
            mb.rescue_orbit_params = {tgt: R}
            base = list(mb.profiles_for(tgt, MissionType.RESCUE)[0])
            before = sum(e.base_dv for e in base)
            after = sum(e.base_dv for e in td.transform_mission(tgt, home, base, mb))
            return after - before

        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        jool = BODY_BY_NAME[BodyName.JOOL]
        # General (home target): 2*raise_dv.
        Rk = kerbin.lo_radius_m * 2.0
        self.assertAlmostEqual(bump(BodyName.KERBIN, BodyName.KERBIN, Rk),
                               2.0 * kerbin.raise_dv(Rk), places=3)
        # Child->parent (Laythe home, Jool target): 2*Hohmann(r_moon, R), and
        # strictly cheaper than the (wrong) 2*raise_dv-from-low-orbit would be.
        r_M = BODY_BY_NAME[BodyName.LAYTHE].parent_periapsis_km * 1000.0
        Rj = 16_000_000.0
        got = bump(BodyName.LAYTHE, BodyName.JOOL, Rj)
        self.assertAlmostEqual(got, 2.0 * jool.hohmann_dv(r_M, Rj), places=3)
        self.assertLess(got, 2.0 * jool.raise_dv(Rj),
                        "child->parent must be cheaper than raise-from-low-orbit")

    def test_random_orbit_offhome_vacuum_free(self):
        # At a VACUUM target, a propulsive capture into a higher orbit costs no
        # more than the low-orbit capture the base profile already charges, so a
        # remote random orbit is free (modeled as the base orbit).
        from worlds.ksp1.bodies import MissionBuilder, RandomOrbitParams
        mb = MissionBuilder(home=BodyName.KERBIN)
        mun = next(b for b in ALL_BODIES if b.name == BodyName.MUN)
        mb.random_orbit_params[BodyName.MUN] = RandomOrbitParams(
            inclination_deg=80.0, sma_m=mun.lo_radius_m * 2.0, eccentricity=0.3)
        base = C.evaluate_contract(
            C.ContractSpec(C.ContractType.ORBIT, BodyName.MUN), FULL, DIFF, mb)
        rand = C.evaluate_contract(
            C.ContractSpec(C.ContractType.RANDOM_ORBIT, BodyName.MUN), FULL, DIFF, mb)
        self.assertAlmostEqual(rand.launch_mass, base.launch_mass, places=3)

    def test_random_orbit_offhome_atmospheric_charged(self):
        # At an ATMOSPHERIC target (Jool), the base profile aerobrakes cheaply into
        # LOW orbit — but you cannot aerobrake into a high orbit, so raising to the
        # seeded orbit is a real propulsive burn that MUST be charged off-home.
        # (Regression: this was previously free, under-charging high Jool/Eve/Duna
        # orbits by hundreds-to-~2000 m/s.)
        from worlds.ksp1.bodies import MissionBuilder, RandomOrbitParams
        mb = MissionBuilder(home=BodyName.KERBIN)
        jool = BODY_BY_NAME[BodyName.JOOL]
        r_lo = jool.lo_radius_m
        mb.random_orbit_params[BodyName.JOOL] = RandomOrbitParams(
            inclination_deg=0.0, sma_m=r_lo * 2.0, eccentricity=0.0)  # circular 2x low
        td = C.CONTRACT_TYPE_DEFS[C.ContractType.RANDOM_ORBIT]
        base = mb.profiles_for(BodyName.JOOL, MissionType.ORBIT)[0]
        extra = (sum(e.base_dv for e in td.transform_mission(
                     BodyName.JOOL, BodyName.KERBIN, base, mb))
                 - sum(e.base_dv for e in base))
        expected = jool.transfer_circular_to_ellipse_dv(r_lo, r_lo * 2.0, r_lo * 2.0)
        self.assertGreater(expected, 100.0)            # a real burn, not noise
        self.assertAlmostEqual(extra, expected, places=3)

    def test_orbit_moon_to_parent_charges_descent(self):
        # CASE C: orbiting the parent from a moon home (Laythe -> Jool) must charge
        # the in-well descent from the moon's orbital radius down to the target
        # orbit. The base graph mis-charges that descent as free (bug 099); the
        # contract transform corrects it. (Plain low Jool orbit from Laythe was
        # under-charged by ~3 km/s.)
        mb = MissionBuilder(home=BodyName.LAYTHE)
        jool = BODY_BY_NAME[BodyName.JOOL]
        laythe = BODY_BY_NAME[BodyName.LAYTHE]
        td = C.CONTRACT_TYPE_DEFS[C.ContractType.ORBIT]
        base = mb.profiles_for(BodyName.JOOL, MissionType.ORBIT)[0]
        extra = (sum(e.base_dv for e in td.transform_mission(
                     BodyName.JOOL, BodyName.LAYTHE, base, mb))
                 - sum(e.base_dv for e in base))
        r_moon = laythe.parent_periapsis_km * 1000.0
        lo = jool.min_orbit_radius_m
        expected = jool.transfer_circular_to_ellipse_dv(r_moon, lo, lo)
        self.assertGreater(expected, 1000.0)           # a large, real descent
        self.assertAlmostEqual(extra, expected, places=3)

    def test_orbit_targets_clear_terrain(self):
        # Lumpy bodies: every contract orbit target (random, equatorial, polar)
        # must sit above the tallest terrain peak — never inside a mountain.
        gilly = BODY_BY_NAME[BodyName.GILLY]
        peak_radius = (gilly.radius_km + gilly.max_terrain_km) * 1000.0
        self.assertGreater(gilly.min_orbit_radius_m, peak_radius,
                           "Gilly orbit floor must clear its peaks")
        self.assertGreater(gilly.min_orbit_radius_m, gilly.lo_radius_m,
                           "Gilly is the lumpy case where terrain > low orbit")
        for seed in range(40):
            p = generate_random_orbit_params(_random.Random(seed), ALL_BODIES)[BodyName.GILLY]
            self.assertGreaterEqual(p.periapsis_m, gilly.min_orbit_radius_m - 1.0,
                                    f"random orbit periapsis below terrain (seed {seed})")
        for ct in (C.ContractType.EQUATORIAL_ORBIT, C.ContractType.POLAR_ORBIT):
            params = C.CONTRACT_TYPE_DEFS[ct].build_parameters(BodyName.GILLY)
            sma = next(p.to_json()["sma"] for p in params
                       if p.to_json()["kind"] == "specific_orbit")
            self.assertGreaterEqual(sma, gilly.min_orbit_radius_m - 1.0,
                                    f"{ct} target orbit below terrain")

    def test_random_orbit_is_collision_safe(self):
        # The seeded random orbit's full periapsis..apoapsis span must avoid every
        # moon's PeR..ApR (+SOI) band — an eccentric orbit that crosses a moon's
        # path is not flyable. Mirrors the rescue collision test for the eccentric
        # case (both periapsis and apoapsis live in one moon-free gap).
        for seed in range(60):
            params = generate_random_orbit_params(_random.Random(seed), ALL_BODIES)
            for tgt in (BodyName.JOOL, BodyName.KERBIN, BodyName.EVE, BodyName.DUNA):
                p = params[tgt]
                B = BODY_BY_NAME[tgt]
                self.assertGreaterEqual(p.periapsis_m, B.min_orbit_radius_m - 1.0,
                                        f"{tgt} periapsis below floor (seed {seed})")
                self.assertLessEqual(p.apoapsis_m, B.soi_radius_km * 1000.0,
                                     f"{tgt} apoapsis outside SOI (seed {seed})")
                for m in ALL_BODIES:
                    if m.parent != tgt:
                        continue
                    band_lo = m.parent_periapsis_km * 1000.0 - m.soi_radius_km * 1000.0
                    band_hi = m.parent_apoapsis_km * 1000.0 + m.soi_radius_km * 1000.0
                    overlaps = p.periapsis_m <= band_hi and band_lo <= p.apoapsis_m
                    self.assertFalse(
                        overlaps,
                        f"{tgt} orbit [{p.periapsis_m:.0f},{p.apoapsis_m:.0f}] clips "
                        f"{m.name}'s [{band_lo:.0f},{band_hi:.0f}] band (seed {seed})")

    def test_random_orbit_orientation_randomized(self):
        # LAN + argument of periapsis are randomized for visual variety: in
        # [0, 360), varying across bodies, ridden over the specific_orbit wire as
        # additive fields. They carry NO delta-v (orbit_reach_dv never sees them).
        params = generate_random_orbit_params(_random.Random(7), ALL_BODIES)
        lans, args = set(), set()
        for p in params.values():
            self.assertTrue(0.0 <= p.lan_deg < 360.0, f"LAN out of range: {p.lan_deg}")
            self.assertTrue(0.0 <= p.arg_pe_deg < 360.0, f"argPe out of range: {p.arg_pe_deg}")
            lans.add(round(p.lan_deg, 3))
            args.add(round(p.arg_pe_deg, 3))
        self.assertGreater(len(lans), 5, "LAN should vary across bodies")
        self.assertGreater(len(args), 5, "argument of periapsis should vary")
        # The seeded orientation rides the wire on a RANDOM_ORBIT contract.
        mb = MissionBuilder(home=BodyName.KERBIN)
        mb.random_orbit_params = params
        j = next(p.to_json() for p in C.CONTRACT_TYPE_DEFS[C.ContractType.RANDOM_ORBIT]
                 .build_parameters(BodyName.MUN, mb)
                 if p.to_json()["kind"] == "specific_orbit")
        self.assertEqual(j["lan"], params[BodyName.MUN].lan_deg)
        self.assertEqual(j["arg_pe"], params[BodyName.MUN].arg_pe_deg)

    def test_is_goal_flag_round_trips(self):
        # is_goal is the only field that distinguishes a goal contract in
        # slot_data (UT recategorizes on it), so it must survive the wire.
        goal = C.ContractSpec(C.ContractType.RETURN, BodyName.DUNA, is_goal=True)
        d = goal.to_slot_dict()
        self.assertTrue(d["is_goal"])
        restored = C.ContractSpec.from_slot_dict(d)
        self.assertTrue(restored.is_goal)
        self.assertEqual(restored, goal)
        # A non-goal spec stays non-goal across the round-trip.
        self.assertFalse(MUN_MINE.to_slot_dict()["is_goal"])
        self.assertFalse(C.ContractSpec.from_slot_dict(MUN_MINE.to_slot_dict()).is_goal)

    def test_required_part_names_per_seed(self):
        # One representative (the lightest) per required category when a mine
        # contract is present: lightest drill = MiniDrill, lightest ore tank =
        # RadialOreTank. Heavier variants are NOT promoted (avoids fill bloat).
        names = C.required_part_names_for([MUN_MINE])
        self.assertEqual(names, frozenset({"MiniDrill", "RadialOreTank"}))
        # Empty when no contract uses any category (e.g. mine disabled).
        self.assertEqual(C.required_part_names_for([]), frozenset())


class TestContractWorldIntegration(KSP1TestBase):
    """Default options (mine weight 1) generate mine contracts; verify the pool,
    the three-gate access rule, and slot_data emission under a real world."""
    # The contract access rule gates on has_all(_cheap_contract_reps) — a pre_fill
    # side effect; without the real ladder it's conservatively unreachable.
    needs_real_pre_fill = True

    def test_contracts_generated(self):
        self.assertTrue(self.world.contract_specs,
                        "default options should generate at least one contract")
        for spec in self.world.contract_specs:
            self.assertIn(spec.contract_type, set(C.ContractType))
            # The home body is only allowed for home_safe types (orbital content).
            if spec.body == self.world.mission_builder.home:
                self.assertTrue(
                    spec.type_def.home_safe,
                    f"{spec.contract_type} targets home but isn't home_safe")

    def test_contract_item_and_location_exist(self):
        spec = self.world.contract_specs[0]
        item_names = {i.name for i in self.multiworld.itempool}
        self.assertIn(spec.item_name, item_names)
        # Location attached to the world.
        loc = self.multiworld.get_location(spec.location_name, self.player)
        self.assertIsNotNone(loc)

    def test_location_gated_by_contract_item(self):
        spec = self.world.contract_specs[0]
        # Everything EXCEPT the contract item: capability is fully satisfied,
        # so only the item gate can block — the location must be unreachable.
        self.collect_all_but([spec.item_name])
        self.assertFalse(
            self.can_reach_location(spec.location_name),
            "contract location reachable without its contract item")
        # Granting the item opens it.
        self.collect_by_name([spec.item_name])
        self.assertTrue(
            self.can_reach_location(spec.location_name),
            "contract location still blocked after granting item + full capability")

    def test_slot_data_career_and_contracts(self):
        d = self.world.fill_slot_data()
        self.assertIn("career", d)
        self.assertTrue(d["career"]["infinite_funds"])
        self.assertEqual(len(d["career"]["building_levels"]), 9)
        # The contracts array carries both non-goal and goal contracts.
        self.assertEqual(
            len(d["contracts"]),
            len(self.world.contract_specs) + len(self.world.goal_contract_specs))
        # Goal contracts are flagged so UT can recategorize them.
        self.assertEqual(
            sum(1 for c in d["contracts"] if c.get("is_goal")),
            len(self.world.goal_contract_specs))
        entry = d["contracts"][0]
        self.assertIn("parameters", entry)
        self.assertIn("schema", entry)


class TestContractRequiredParts(KSP1TestBase):
    """mine_ore weighted to dominate the pool so a part-requiring contract is
    guaranteed (default options occasionally roll a contract set with no
    part-requiring type, which left this assertion seed-flaky). Verifies those
    required parts are promoted to progression."""
    options = {
        "contract_type_weights": {"mine_ore": 10},
        "contracts_available": 8,
    }

    def test_required_parts_promoted_to_progression(self):
        from BaseClasses import ItemClassification
        names = self.world.contract_required_part_names
        self.assertTrue(names)
        by_name = {i.name: i for i in self.multiworld.itempool}
        for n in names:
            if n in by_name:  # rep parts are removed from the pool
                self.assertEqual(by_name[n].classification,
                                 ItemClassification.progression,
                                 f"{n} should be progression for this seed")


class TestParamWireFormat(unittest.TestCase):
    """Lock the slot_data wire form of each non-trivial parameter primitive. The
    dumb client decodes these dicts by exact key, so a silent field rename or
    type change here breaks contract actuation with no Python-side error.
    (situation/resource are exercised by TestSlotDataRoundTrip.)"""

    def test_has_any_part(self):
        self.assertEqual(
            C.HasAnyPartParam(("MiniDrill", "RadialDrill"), label="drill").to_json(),
            {"kind": "has_any_part",
             "parts": ["MiniDrill", "RadialDrill"], "label": "drill"})

    def test_has_system(self):
        self.assertEqual(
            C.HasSystemParam("Generator", label="power generation").to_json(),
            {"kind": "has_system", "system": "Generator",
             "label": "power generation"})

    def test_crew_capacity(self):
        self.assertEqual(C.CrewCapacityParam(5).to_json(),
                         {"kind": "crew_capacity", "min": 5})

    def test_plant_flag(self):
        self.assertEqual(C.PlantFlagParam(BodyName.MUN).to_json(),
                         {"kind": "plant_flag", "body": "Mun"})

    def test_sample_return(self):
        self.assertEqual(C.SampleReturnParam(BodyName.DUNA).to_json(),
                         {"kind": "sample_return", "body": "Duna"})

    def test_specific_orbit(self):
        # Orientation defaults to 0 (deterministic types don't set it).
        self.assertEqual(
            C.SpecificOrbitParam(
                body=BodyName.KERBIN, orbit_type="EQUATORIAL", inclination=0.0,
                eccentricity=0.0, sma=700000.0, deviation=10.0).to_json(),
            {"kind": "specific_orbit", "body": "Kerbin",
             "orbit_type": "EQUATORIAL", "inclination": 0.0, "eccentricity": 0.0,
             "sma": 700000.0, "deviation": 10.0, "lan": 0.0, "arg_pe": 0.0})
        # A randomized orbit carries its orientation over the wire.
        self.assertEqual(
            C.SpecificOrbitParam(
                body=BodyName.MUN, orbit_type="EQUATORIAL", inclination=45.0,
                eccentricity=0.2, sma=300000.0, deviation=10.0,
                lan=120.0, arg_pe=275.0).to_json(),
            {"kind": "specific_orbit", "body": "Mun",
             "orbit_type": "EQUATORIAL", "inclination": 45.0, "eccentricity": 0.2,
             "sma": 300000.0, "deviation": 10.0, "lan": 120.0, "arg_pe": 275.0})

    def test_collect_science(self):
        self.assertEqual(C.CollectScienceParam(BodyName.MUN, "space").to_json(),
                         {"kind": "collect_science", "body": "Mun",
                          "location": "space"})


class TestStockBackedContractTypes(KSP1TestBase):
    """The stock-parameter orbit-variant + transmit-science contracts must
    generate (including on the home body) and stay reachable when weighted to
    dominate the pool."""
    options = {
        "goal": "standard_sample_returns",
        "difficulty": "normal",
        "contract_type_weights": {
            "equatorial_orbit": 5, "polar_orbit": 5,
            "stationary_orbit": 5, "transmit_science": 5,
        },
        "contracts_available": 20,
    }
    needs_real_pre_fill = True

    _NEW = ("Contract: Equatorial Orbit", "Contract: Polar Orbit",
            "Contract: Stationary Orbit", "Contract: Transmit Science")

    def test_new_types_generate_and_are_reachable(self):
        new_locs = [
            loc for loc in self.multiworld.get_locations(self.player)
            if loc.name.startswith(self._NEW)
        ]
        self.assertTrue(new_locs, "orbit-variant contracts not generated")
        state = self.multiworld.get_all_state(False)
        for loc in new_locs:
            self.assertTrue(loc.can_reach(state), f"{loc.name} unreachable")


class TestHomeBodyOrbitalContract(KSP1TestBase):
    """home_safe orbital contracts are allowed on the home body (good early
    content). Forcing a single home_safe orbital type at max count draws every
    candidate of that type (weighted sample-without-replacement), so the
    home-body instance is guaranteed regardless of seed — the un-pinned
    multi-type config sampled it only by luck."""
    options = {
        "contract_type_weights": {"equatorial_orbit": 1},
        "contracts_available": 40,
        "allow_missions_harder_than_goal": True,
    }

    def test_home_safe_orbital_contracts_on_home(self):
        home = self.world.mission_builder.home.value
        home_orbitals = [
            spec for spec in self.world.contract_specs
            if spec.contract_type == C.ContractType.EQUATORIAL_ORBIT
            and spec.body == home
        ]
        self.assertTrue(
            home_orbitals, f"no home-body ({home}) orbital contract generated")


class TestStationaryFeasibility(unittest.TestCase):
    def test_tidally_locked_moons_have_no_stationary_orbit(self):
        from worlds.ksp1.bodies import BODY_BY_NAME
        from worlds.ksp1.bodies import BodyName as BN
        td = C.CONTRACT_TYPE_DEFS[C.ContractType.STATIONARY_ORBIT]
        # Mun/Tylo (tidally locked) have sync altitude beyond their SOI.
        for bn in (BN.MUN, BN.TYLO, BN.IKE):
            self.assertFalse(td.body_compatible(BODY_BY_NAME[bn]),
                             f"{bn} should have no stationary orbit")
        # Kerbin's keostationary sits well inside its SOI.
        self.assertTrue(td.body_compatible(BODY_BY_NAME[BN.KERBIN]))


class TestDocking(unittest.TestCase):
    """DOCKING: dock two craft in orbit — event-based (repeatable), gated on a
    docking port + rendezvous capability, with an RCS logic-only requirement."""

    TD = C.CONTRACT_TYPE_DEFS[C.ContractType.DOCKING]

    def test_build_parameters(self):
        # A docking objective + the docking-port has_any_part; RCS is logic-only
        # (no in-game objective).
        params = self.TD.build_parameters(BodyName.KERBIN)
        kinds = [p.to_json()["kind"] for p in params]
        self.assertEqual(kinds, ["docking", "has_any_part"])
        self.assertEqual(params[0].to_json(), {"kind": "docking", "body": "Kerbin"})
        self.assertEqual(params[1].to_json()["label"], "docking_port")

    def test_rcs_is_logic_only(self):
        self.assertIn("rcs", self.TD.required_categories)
        self.assertIn("rcs", self.TD.logic_only_categories)
        self.assertNotIn("docking_port", self.TD.logic_only_categories)

    def test_requires_rendezvous_capability(self):
        from worlds.ksp1.effects import Capability
        self.assertTrue(self.TD.requires_rendezvous)
        mb = MissionBuilder(home=BodyName.KERBIN)
        spec = C.ContractSpec(C.ContractType.DOCKING, BodyName.KERBIN)
        needs = C.contract_logic_needs(spec, mb)
        self.assertIn(Capability.CAN_RENDEZVOUS, needs.capabilities)

    def test_phasing_edge_costs_more_than_plain_orbit(self):
        mb = MissionBuilder(home=BodyName.KERBIN)
        dock = C.evaluate_contract(
            C.ContractSpec(C.ContractType.DOCKING, BodyName.KERBIN), FULL, DIFF, mb)
        orbit = C.evaluate_contract(
            C.ContractSpec(C.ContractType.ORBIT, BodyName.KERBIN), FULL, DIFF, mb)
        self.assertTrue(dock.feasible)
        self.assertGreater(dock.launch_mass, orbit.launch_mass,
                           "docking (payload + phasing) must cost more than a bare orbit")

    def test_precise_pointing(self):
        self.assertIn(C.ContractType.DOCKING, C.PRECISE_POINTING_TYPES)


class TestDockingGeneration(KSP1TestBase):
    """DOCKING must generate (weighted to dominate) and stay reachable."""
    options = {
        "contract_type_weights": {"docking": 1},
        "contracts_available": 40,
        "allow_missions_harder_than_goal": True,
    }
    needs_real_pre_fill = True

    def test_docking_generates_and_is_reachable(self):
        locs = [loc for loc in self.multiworld.get_locations(self.player)
                if loc.name.startswith("Contract: Docking")]
        self.assertTrue(locs, "no docking contract generated")
        state = self.multiworld.get_all_state(False)
        for loc in locs:
            self.assertTrue(loc.can_reach(state), f"{loc.name} unreachable")


class TestSurfaceSurvey(unittest.TestCase):
    """SURFACE_SURVEY: run an experiment at a seeded surface waypoint. One-way
    landing (no return), precise touchdown, home-safe, thermometer logic-only."""

    TD = C.CONTRACT_TYPE_DEFS[C.ContractType.SURFACE_SURVEY]

    def _mb(self, lats):
        mb = MissionBuilder(home=BodyName.KERBIN)
        mb.survey_site_lats = lats
        return mb

    def test_build_parameters(self):
        params = self.TD.build_parameters(BodyName.MUN, MB)
        self.assertEqual(len(params), 1)
        j = params[0].to_json()
        self.assertEqual(j["kind"], "survey_waypoint")
        self.assertEqual(j["body"], "Mun")
        self.assertEqual(j["experiment"], C.SURVEY_EXPERIMENT)
        self.assertEqual(j["lat"], MB.survey_site_lats[BodyName.MUN])
        self.assertEqual(j["seed"], int(round(j["lat"] * 1_000_000)))

    def test_thermometer_logic_only(self):
        # Gates feasibility/promotion but emits no in-game objective.
        self.assertIn("thermometer", self.TD.required_categories)
        self.assertIn("thermometer", self.TD.logic_only_categories)
        self.assertNotIn("has_any_part",
                         [p.to_json()["kind"]
                          for p in self.TD.build_parameters(BodyName.MUN, MB)])

    def test_missing_site_raises(self):
        with self.assertRaises(ValueError):
            self.TD.build_parameters(BodyName.MUN, self._mb({}))

    def test_home_safe_and_landing_only(self):
        self.assertTrue(self.TD.home_safe)
        self.assertTrue(self.TD.requires_landing())
        # Only landable bodies are compatible.
        self.assertFalse(self.TD.body_compatible(BODY_BY_NAME[BodyName.JOOL]))
        self.assertTrue(self.TD.body_compatible(BODY_BY_NAME[BodyName.MUN]))

    def test_home_survey_charges_inclination(self):
        # A home survey at a nonzero site latitude costs more than a plain home
        # landing (the ascent must reach the site's latitude band).
        mb = self._mb({BodyName.KERBIN: 40.0})
        survey = C.evaluate_contract(
            C.ContractSpec(C.ContractType.SURFACE_SURVEY, BodyName.KERBIN),
            FULL, DIFF, mb)
        self.assertTrue(survey.feasible)
        flat = self._mb({BodyName.KERBIN: 0.0})
        base = C.evaluate_contract(
            C.ContractSpec(C.ContractType.SURFACE_SURVEY, BodyName.KERBIN),
            FULL, DIFF, flat)
        self.assertGreater(survey.launch_mass, base.launch_mass,
                           "inclined home survey must cost more than an equatorial one")


class TestSurfaceSurveyGeneration(KSP1TestBase):
    """SURFACE_SURVEY must generate (weighted to dominate) and stay reachable."""
    options = {
        "contract_type_weights": {"surface_survey": 1},
        "contracts_available": 40,
        "allow_missions_harder_than_goal": True,
    }
    needs_real_pre_fill = True

    def test_survey_generates_and_is_reachable(self):
        locs = [loc for loc in self.multiworld.get_locations(self.player)
                if loc.name.startswith("Contract: Surface Survey")]
        self.assertTrue(locs, "no surface survey contract generated")
        state = self.multiworld.get_all_state(False)
        for loc in locs:
            self.assertTrue(loc.can_reach(state), f"{loc.name} unreachable")


class TestTourism(unittest.TestCase):
    """TOURISM: fly tourists to a body's orbit and return. Priced as ORBIT_RETURN
    (reach orbit + return, no rendezvous); suborbit is approximated as orbit."""

    TD = C.CONTRACT_TYPE_DEFS[C.ContractType.TOURISM]

    def test_build_parameters_one_per_tourist(self):
        params = self.TD.build_parameters(BodyName.MUN, MB)
        self.assertEqual(len(params), C.TOURISM_CREW)
        for p in params:
            j = p.to_json()
            self.assertEqual(j["kind"], "tourist")
            self.assertEqual(j["body"], "Mun")
            self.assertIn(j["entry"], ("Suborbit", "Orbit"))
            self.assertIsInstance(j["female"], bool)

    def test_offhome_entry_is_orbit(self):
        # A suborbital hop only makes sense at home; off-home is always orbit.
        for body, manifest in MB.tourist_manifests.items():
            if body != BodyName.KERBIN:
                for t in manifest:
                    self.assertEqual(t.entry, "Orbit", f"{body} suborbit tourist")

    def test_missing_manifest_raises(self):
        mb = MissionBuilder(home=BodyName.KERBIN)
        with self.assertRaises(ValueError):
            self.TD.build_parameters(BodyName.MUN, mb)

    def test_orbit_return_is_cheaper_than_rescue(self):
        # No rendezvous phasing, so tourism (ORBIT_RETURN) costs <= a rescue
        # (RESCUE) to the same body, and both reach that body's orbit.
        from worlds.ksp1.capability import evaluate_mission_detailed
        mb = MB
        orbit_ret = evaluate_mission_detailed(
            FULL, DIFF, BodyName.MUN, MissionType.ORBIT_RETURN, None, mb)
        rescue = evaluate_mission_detailed(
            FULL, DIFF, BodyName.MUN, MissionType.RESCUE, None, mb)
        self.assertTrue(orbit_ret.feasible)
        self.assertLessEqual(orbit_ret.launch_mass, rescue.launch_mass)

    def test_home_and_offhome_feasible(self):
        for body in (BodyName.KERBIN, BodyName.MUN, BodyName.MINMUS):
            r = C.evaluate_contract(
                C.ContractSpec(C.ContractType.TOURISM, body), FULL, DIFF, MB)
            self.assertTrue(r.feasible, f"tourism to {body} infeasible on full kit")

    def test_targets_orbitable_not_star(self):
        self.assertTrue(self.TD.body_compatible(BODY_BY_NAME[BodyName.JOOL]))
        self.assertFalse(self.TD.body_compatible(BODY_BY_NAME[BodyName.KERBOL]))


class TestTourismGeneration(KSP1TestBase):
    """TOURISM must generate (weighted to dominate) and stay reachable."""
    options = {
        "contract_type_weights": {"tourism": 1},
        "contracts_available": 40,
        "allow_missions_harder_than_goal": True,
    }
    needs_real_pre_fill = True

    def test_tourism_generates_and_is_reachable(self):
        locs = [loc for loc in self.multiworld.get_locations(self.player)
                if loc.name.startswith("Contract: Tourism")]
        self.assertTrue(locs, "no tourism contract generated")
        state = self.multiworld.get_all_state(False)
        for loc in locs:
            self.assertTrue(loc.can_reach(state), f"{loc.name} unreachable")


class TestStarNotAMissionDestination(unittest.TestCase):
    """The star (Kerbol/Sun) is not a mission destination: no contract type can
    target it, it has no registered locations, and the orbit/flyby goal lists
    reject it at option validation and as defense-in-depth at gen time."""

    def test_no_contract_type_targets_the_star(self):
        from worlds.ksp1.bodies import BODY_BY_NAME
        sun = BODY_BY_NAME[BodyName.KERBOL]
        self.assertFalse(sun.is_orbitable)
        for ct in C.ContractType:
            self.assertFalse(
                C.CONTRACT_TYPE_DEFS[ct].body_compatible(sun),
                f"{ct} must not target the star")
        sun_specs = [s for s in C.all_possible_contract_specs()
                     if s.body == BodyName.KERBOL]
        self.assertEqual(sun_specs, [], f"star has contracts: {sun_specs}")

    def test_star_has_no_registered_locations(self):
        from worlds.ksp1.locations import LOCATION_NAME_TO_ID
        sun_locs = [n for n in LOCATION_NAME_TO_ID if BodyName.KERBOL in n]
        self.assertEqual(sun_locs, [], f"star has location checks: {sun_locs}")

    def test_orbit_flyby_options_reject_the_star(self):
        from worlds.ksp1.options import OrbitBodies, FlybyBodies
        from Options import OptionError
        for opt_cls in (OrbitBodies, FlybyBodies):
            with self.assertRaises(OptionError):
                opt_cls({BodyName.KERBOL, BodyName.MUN}).verify_keys()
        # A real orbitable body (incl. the gas giant) is accepted.
        OrbitBodies({BodyName.MUN, BodyName.JOOL}).verify_keys()

    def test_star_goal_body_fails_fast_not_keyerror(self):
        # Defense-in-depth: even if the star slips past option validation, gen
        # raises a clear OptionError, not a KeyError on the missing location.
        from test.general import setup_multiworld, call_all
        from worlds.ksp1.world import KSP1World
        from Options import OptionError
        mw = setup_multiworld(
            KSP1World, steps=(), seed=1,
            options={"goal": "custom", "orbit_bodies": {BodyName.KERBOL}})
        with self.assertRaises(OptionError):
            call_all(mw, "generate_early")


class TestExplainContractGeneric(unittest.TestCase):
    """`/explain Contract:` must work for EVERY contract type with no per-type
    handling — a future type that breaks the formatter fails here. Drives the
    formatter over each CONTRACT_TYPE_DEFS entry on a compatible body, with both
    a full kit and an empty kit, asserting it never raises and always renders the
    three gates and the parameter tree."""

    def test_every_contract_type_renders(self):
        from worlds.ksp1.capability_format import format_contract_output
        for ct, td in C.CONTRACT_TYPE_DEFS.items():
            body = _first_compatible_body(td)
            self.assertIsNotNone(body, f"{ct} has no compatible body")
            spec = C.ContractSpec(ct, body)
            for label, flags in (("full", FULL), ("empty", EMPTY)):
                with self.subTest(contract_type=ct.name, kit=label):
                    lines = format_contract_output(
                        spec, in_logic=False, item_held=False,
                        flags=flags, diff=DIFF, difficulty_name="comfortable",
                        mission_builder=MB, proxy=False,
                    )
                    text = "\n".join(lines)
                    self.assertIn(spec.display_name, text)
                    self.assertIn("Gate 1", text)
                    self.assertIn("Gate 2", text)
                    self.assertIn("Gate 3", text)
                    self.assertIn("Contract parameters", text)
                    # Every emitted parameter primitive must show its kind.
                    for p in td.build_parameters(body, MB):
                        self.assertIn(p.to_json()["kind"], text)

    def test_full_kit_mine_shows_required_parts_and_rocket(self):
        from worlds.ksp1.capability_format import format_contract_output
        text = "\n".join(format_contract_output(
            MUN_MINE, in_logic=True, item_held=True,
            flags=FULL, diff=DIFF, difficulty_name="comfortable",
            mission_builder=MB, proxy=False,
        ))
        # Gate 2 lists the lightest part per required category.
        self.assertIn("drill: HAVE", text)
        self.assertIn("MiniDrill", text)
        self.assertIn("ore_tank: HAVE", text)
        # Feasible -> the delivery rocket is rendered with the contract part on it.
        self.assertIn("Gate 3 - physics delivery: YES", text)
        self.assertIn("Delivery rocket", text)

    def test_empty_kit_mine_reports_missing_and_infeasible(self):
        from worlds.ksp1.capability_format import format_contract_output
        text = "\n".join(format_contract_output(
            MUN_MINE, in_logic=False, item_held=False,
            flags=EMPTY, diff=DIFF, difficulty_name="comfortable",
            mission_builder=MB, proxy=False,
        ))
        self.assertIn("drill: MISSING", text)
        self.assertIn("Gate 3 - physics delivery: NO", text)


class TestRequirementSeam(unittest.TestCase):
    """The typed Requirement union. Baseline contracts derive AnyOf from
    required_categories — behaviour-identical to the old category tuple — and
    the part-resolution consumers fail closed on any kind they don't handle, so
    a future Requirement subclass can't be silently dropped."""

    def test_every_def_derives_anyof_from_categories(self):
        for ct, td in C.CONTRACT_TYPE_DEFS.items():
            with self.subTest(contract_type=ct.name):
                self.assertEqual(
                    td.requirements,
                    tuple(C.AnyOf(cat) for cat in td.required_categories))

    def test_unhandled_requirement_kind_fails_closed(self):
        from types import SimpleNamespace
        unknown = C.Requirement()  # base class, no resolver branch
        td = C.ContractTypeDef(
            contract_type=C.ContractType.ORBIT, location_noun="X",
            base_mission_type=C.MissionType.ORBIT, crewed=None,
            required_categories=(), title_fmt="", synopsis_fmt="",
            requirements=(unknown,))
        with self.assertRaises(NotImplementedError):
            C.required_part_names_for([SimpleNamespace(type_def=td)])
        with self.assertRaises(NotImplementedError):
            C.required_part_breakdown(SimpleNamespace(type_def=td), FULL)


class TestSurfaceRescue(unittest.TestCase):
    """SURFACE_RESCUE: land NEAR a Kerbal stranded on the target's surface
    (precision-landing surcharge, difficulty-resolved) with a seeded
    site-latitude plane change on the ascent out, then bring them home.
    Home body excluded (stock RecoverAsset parity)."""

    TD = C.CONTRACT_TYPE_DEFS[C.ContractType.SURFACE_RESCUE]

    def _mb(self, lats):
        mb = MissionBuilder(home=BodyName.KERBIN)
        mb.surface_rescue_site_lats = lats
        return mb

    def test_profile_lands_and_returns(self):
        # Inverse of the orbital rescue's no-landing property: the surface
        # rescue MUST touch the target's surface, and end back home.
        profiles = MB.profiles_for(BodyName.MUN, MissionType.SURFACE_RESCUE)
        self.assertTrue(profiles)
        for profile in profiles:
            nodes = {e.source for e in profile} | {e.destination for e in profile}
            self.assertIn("mun_surface", nodes,
                          "surface rescue must land at the target")
            self.assertEqual(profile[-1].destination, "kerbin_surface")

    def test_home_excluded_and_body_compat(self):
        self.assertEqual(
            MB.profiles_for(BodyName.KERBIN, MissionType.SURFACE_RESCUE), [])
        self.assertFalse(self.TD.home_safe)
        self.assertTrue(self.TD.body_compatible(BODY_BY_NAME[BodyName.MUN]))
        self.assertFalse(self.TD.body_compatible(BODY_BY_NAME[BodyName.JOOL]),
                         "no solid surface -> no surface rescue")

    def test_transform_marks_precision_and_charges_plane(self):
        lat = 30.0
        mb = self._mb({BodyName.MUN: lat})
        base = list(mb.profiles_for(BodyName.MUN, MissionType.SURFACE_RESCUE)[0])
        out = self.TD.transform_mission(BodyName.MUN, BodyName.KERBIN, base, mb)
        land = [e for e in out
                if e.source == "mun_low_orbit" and e.destination == "mun_surface"]
        self.assertTrue(land)
        self.assertTrue(all(e.precision_landing for e in land))
        # Worst-case ascent plane reconcile: 2*v_LO*sin(lat/2) on the target
        # ascent edge's plane_change_dv (fraction-priced at evaluation).
        v_lo = BODY_BY_NAME[BodyName.MUN].lo_circular_velocity
        want = 2.0 * v_lo * math.sin(math.radians(lat) / 2.0)
        delta = (sum(e.plane_change_dv for e in out)
                 - sum(e.plane_change_dv for e in base))
        self.assertAlmostEqual(delta, want, places=3)
        # base_dv untouched: precision is difficulty-resolved at evaluation.
        self.assertAlmostEqual(sum(e.base_dv for e in out),
                               sum(e.base_dv for e in base), places=6)

    def test_precision_surcharge_arithmetic(self):
        mun_g = BODY_BY_NAME[BodyName.MUN].surface_gravity
        for name, hover in (("generous", 45.0), ("comfortable", 30.0),
                            ("small", 15.0), ("zero", 0.0)):
            p = DIFFICULTY_PROFILES[name]
            self.assertEqual(p.precision_hover_s, hover)
            want = 0.0 if hover == 0.0 else mun_g * hover + 60.0
            self.assertAlmostEqual(precision_landing_dv(mun_g, p), want,
                                   places=6, msg=name)

    def test_evaluator_charges_surcharge_zero_profile_free(self):
        # lat=0 kills the plane term, isolating the precision surcharge:
        # costs more than a plain RETURN on comfortable, identical on zero.
        from worlds.ksp1.capability import evaluate_mission_detailed
        mb = self._mb({BodyName.MUN: 0.0})

        def tf(edges):
            return self.TD.transform_mission(
                BodyName.MUN, BodyName.KERBIN, edges, mb)

        for name in ("comfortable", "zero"):
            diff = DIFFICULTY_PROFILES[name]
            base = evaluate_mission_detailed(
                FULL, diff, BodyName.MUN, MissionType.RETURN, None, mb)
            sr = evaluate_mission_detailed(
                FULL, diff, BodyName.MUN, MissionType.SURFACE_RESCUE, None, mb,
                mission_transform=tf)
            self.assertTrue(base.feasible and sr.feasible, name)
            if name == "zero":
                self.assertAlmostEqual(sr.launch_mass, base.launch_mass,
                                       places=6,
                                       msg="zero profile budgets no hover")
            else:
                self.assertGreater(sr.launch_mass, base.launch_mass)

    def test_mass_monotonic_in_site_latitude(self):
        from worlds.ksp1.capability import evaluate_mission_detailed
        diff = DIFFICULTY_PROFILES["generous"]   # pays the full plane change
        masses = []
        for lat in (5.0, 25.0, 45.0):
            mb = self._mb({BodyName.MUN: lat})

            def tf(edges, _mb=mb):
                return self.TD.transform_mission(
                    BodyName.MUN, BodyName.KERBIN, edges, _mb)

            r = evaluate_mission_detailed(
                FULL, diff, BodyName.MUN, MissionType.SURFACE_RESCUE, None,
                mb, mission_transform=tf)
            self.assertTrue(r.feasible, f"lat {lat}")
            masses.append(r.launch_mass)
        self.assertLessEqual(masses[0], masses[1])
        self.assertLessEqual(masses[1], masses[2])

    def test_paid_plane_change_scales_with_fraction(self):
        # Same 45-degree site: generous (fraction 1.0) pays a bigger mass
        # premium over its own lat-0 baseline than small (0.05) pays over its.
        from worlds.ksp1.capability import evaluate_mission_detailed

        def premium(name):
            diff = DIFFICULTY_PROFILES[name]
            out = []
            for lat in (45.0, 0.0):
                mb = self._mb({BodyName.MUN: lat})

                def tf(edges, _mb=mb):
                    return self.TD.transform_mission(
                        BodyName.MUN, BodyName.KERBIN, edges, _mb)

                r = evaluate_mission_detailed(
                    FULL, diff, BodyName.MUN, MissionType.SURFACE_RESCUE,
                    None, mb, mission_transform=tf)
                self.assertTrue(r.feasible, name)
                out.append(r.launch_mass)
            return out[0] - out[1]

        self.assertGreater(premium("generous"), premium("small"))
        self.assertGreater(premium("generous"), 0.0)

    def test_params_free_seat_and_fail_loud(self):
        spec = C.ContractSpec(C.ContractType.SURFACE_RESCUE, BodyName.MUN)
        params = self.TD.build_parameters(BodyName.MUN, MB)
        kinds = [p.to_json()["kind"] for p in params]
        self.assertIn("surface_rescue", kinds)
        self.assertIn("has_any_part", kinds)   # the crew_cabin free seat
        sr = next(p for p in params if isinstance(p, C.SurfaceRescueParam))
        self.assertEqual(sr.lat, MB.surface_rescue_site_lats[BodyName.MUN])
        # eva_jetpack is logic-only: gates the manifest, never an objective.
        self.assertIsNotNone(C.required_part_manifest(spec, FULL))
        self.assertIsNone(C.required_part_manifest(spec, _flags(lambda n: 0)))
        # A seeded-but-missing site latitude is a wiring bug — fail loud.
        with self.assertRaises(ValueError):
            self.TD.build_parameters(BodyName.MUN, self._mb({}))

    def test_slot_round_trip_schema(self):
        # Schema is pinned so a wire-format change is a conscious bump matched on
        # the client (ContractPrimitiveRegistry.Schema). v7 = docking + the rest
        # of the new-type wave.
        self.assertEqual(C.CONTRACT_SCHEMA_VERSION, 7)
        spec = C.ContractSpec(C.ContractType.SURFACE_RESCUE, BodyName.MUN)
        d = spec.to_slot_dict(mission_builder=MB)
        self.assertEqual(d["schema"], 7)
        self.assertIn("surface_rescue", [p["kind"] for p in d["parameters"]])
        self.assertEqual(C.ContractSpec.from_slot_dict(d), spec)

    def test_precise_pointing_gate(self):
        # Same wheel/RCS-on-assist, gimbal-when-off rule as the orbital rescue.
        self.assertIn(C.ContractType.SURFACE_RESCUE, C.PRECISE_POINTING_TYPES)
        g = _flags(lambda n: 99)
        g.has_reaction_wheels = False
        g.has_rcs = False
        g.lightest_reaction_wheel = None
        g.lightest_rcs_thruster = None
        spec = C.ContractSpec(C.ContractType.SURFACE_RESCUE, BodyName.MUN)
        for gameplay, want in ((TestPreciseOrbitAttitude.ASSIST, False),
                               (TestPreciseOrbitAttitude.NO_ASSIST, True)):
            mb = self._mb(generate_surface_rescue_site_lats(
                _random.Random(0), ALL_BODIES))
            mb.gameplay = gameplay
            self.assertEqual(C.can_complete_contract(spec, g, DIFF, mb), want)

    def test_logic_needs_eva_and_rendezvous(self):
        from worlds.ksp1.capability import mission_logic_needs
        from worlds.ksp1.effects import Capability
        needs = mission_logic_needs(
            BodyName.MUN, MissionType.SURFACE_RESCUE, None, None, MB)
        self.assertIn(Capability.CAN_EVA, needs.capabilities)
        self.assertIn(Capability.CAN_RENDEZVOUS, needs.capabilities)


if __name__ == "__main__":
    unittest.main()
