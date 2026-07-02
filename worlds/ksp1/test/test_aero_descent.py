"""Unit tests for rocket_math.aero — the staged atmospheric-descent physics.

Hand-computed vectors for every primitive plus behavioural checks on the
ladder walk.  End-to-end conservatism against an RK4 reference is validated
separately by scratchpad/sim_staged_landing.py; the per-body expectation
table lives in test_atmo_landing.py once capability wiring lands.
"""
import math
import unittest

from worlds.ksp1.rocket_math import aero


# Kerbin / Duna atmosphere constants (test vectors, mirroring bodies.py)
KERBIN = dict(rho0=1.225, H=5000.0, g=9.81, p0=101.325)
DUNA = dict(rho0=0.096, H=3000.0, g=2.94, p0=6.755)


class TestPrimitives(unittest.TestCase):
    def test_local_density(self) -> None:
        self.assertAlmostEqual(
            aero.local_density(1.225, 5000.0, 5000.0), 1.225 / math.e, places=6)
        self.assertEqual(aero.local_density(1.225, 5000.0, -10.0), 1.225)
        self.assertEqual(aero.local_density(0.0, 5000.0, 100.0), 0.0)

    def test_pressure_altitude(self) -> None:
        # Kerbin main-chute semi gate: 5000 * ln(1/0.04) = 16094.4 m
        h = aero.pressure_altitude(101.325, 5000.0, 0.04)
        self.assertAlmostEqual(h, 16094.4, delta=0.5)
        # Duna mains: 3000 * ln(0.06667/0.04) = 1532 m — everything happens low
        h = aero.pressure_altitude(6.755, 3000.0, 0.04)
        self.assertAlmostEqual(h, 1532.4, delta=1.0)
        # Body too thin to open the chute at all
        self.assertIsNone(aero.pressure_altitude(3.0, 5000.0, 0.04))
        self.assertIsNone(aero.pressure_altitude(0.0, 5000.0, 0.04))

    def test_local_terminal_velocity(self) -> None:
        # Mk16 (500) under a 1.5t craft at Kerbin sea level: 6.93 m/s
        vt = aero.local_terminal_velocity(1.5, 9.81, 1.225, 500.0)
        self.assertAlmostEqual(vt, 6.93, delta=0.01)
        self.assertEqual(
            aero.local_terminal_velocity(1.5, 9.81, 0.0, 500.0), math.inf)

    def test_bleed_speed_allen_eggers(self) -> None:
        # K = 0.096*3000*2.5 / (2*10000*0.5) = 0.072 -> 1000*exp(-0.072)
        v = aero.bleed_speed(1000.0, 10.0, 2.5, 0.096, 3000.0, 2.94)
        self.assertAlmostEqual(v, 1000.0 * math.exp(-0.072), delta=0.1)

    def test_bleed_speed_settle_floor(self) -> None:
        # Strong drag would decay to ~0; the floor is 1.3x local terminal
        v = aero.bleed_speed(300.0, 10.0, 2.5, 0.096, 3000.0, 2.94)
        floor = aero.SETTLE_FACTOR * aero.local_terminal_velocity(
            10.0, 2.94, 0.096, 2.5)
        self.assertAlmostEqual(v, floor, delta=0.1)

    def test_bleed_speed_no_drag_no_credit(self) -> None:
        self.assertEqual(
            aero.bleed_speed(1000.0, 10.0, 0.0, 0.096, 3000.0, 2.94), 1000.0)

    def test_drag_decayed_speed(self) -> None:
        # Kerbin main full-deploy: from 132 m/s at 1000m the chute pins the
        # craft to its (deploy-density) terminal velocity almost instantly.
        rho_dep = 1.225 * math.exp(-1000.0 / 5000.0)
        v = aero.drag_decayed_speed(132.0, 1.5, 500.0, rho_dep, 1000.0, 9.81)
        vt = aero.local_terminal_velocity(1.5, 9.81, rho_dep, 500.0)
        self.assertAlmostEqual(v, vt, delta=0.01)

    def test_drag_decayed_speed_no_drag_is_energy_gain(self) -> None:
        v = aero.drag_decayed_speed(100.0, 1.0, 0.0, 0.0, 1000.0, 9.81)
        self.assertAlmostEqual(v, math.sqrt(100.0 ** 2 + 2 * 9.81 * 1000.0),
                               places=6)

    def test_drag_decayed_from_below_terminal(self) -> None:
        # Below terminal the craft speeds toward it, never past it.
        rho = 0.0417
        vt = aero.local_terminal_velocity(10.0, 2.94, rho, 172.0)
        v = aero.drag_decayed_speed(vt * 0.3, 10.0, 172.0, rho, 5000.0, 2.94)
        self.assertLessEqual(v, vt + 1e-9)
        self.assertGreater(v, vt * 0.3)

    def test_decayed_speed_over_slabs_track_density(self) -> None:
        # Over multiple scale heights the slabbed bound must land near the
        # LOCAL terminal at the bottom, not the (huge) top-density terminal.
        v = aero.decayed_speed_over(272.0, 1.5, 3.46, 1.225, 5000.0,
                                    7500.0, 1000.0, 9.81)
        vt_bottom = aero.local_terminal_velocity(
            1.5, 9.81, aero.local_density(1.225, 5000.0, 1000.0), 3.46)
        self.assertLess(v, 1.35 * vt_bottom)
        self.assertGreaterEqual(v, vt_bottom)

    def test_burn_dv_gravity_loss(self) -> None:
        # Entry-angle burn at TWR 1.5: 1/(1-0.5/1.5) = 1.5x
        self.assertAlmostEqual(aero.burn_dv_for(100.0, 1.5, 0.5), 150.0)
        # Vertical suicide burn at TWR 1.5: 3x
        self.assertAlmostEqual(aero.burn_dv_for(93.0, 1.5, 1.0), 279.0)
        # TWR at/below the path component cannot brake
        self.assertEqual(aero.burn_dv_for(10.0, 0.9, 1.0), math.inf)
        self.assertEqual(aero.burn_dv_for(0.0, 1.5, 1.0), 0.0)

    def test_max_engage_speed(self) -> None:
        self.assertAlmostEqual(
            aero.max_engage_speed(8.8, 1.225), math.sqrt(17600.0 / 1.225),
            places=6)
        self.assertEqual(aero.max_engage_speed(8.8, 0.0), math.inf)

    def test_terminal_limited_count_solves_the_boundary(self) -> None:
        # The returned continuous count, plugged back into the terminal formula,
        # must give exactly v_safe (self-consistency of the closed-form solve).
        for is_radial, drag, mass in ((True, 500.0, 0.1), (False, 500.0, 0.1),
                                      (True, 170.0, 0.2)):
            for payload in (1.0, 5.0, 20.0):
                n = aero.terminal_limited_count(
                    payload_t=payload, chute_drag=drag, chute_mass_t=mass,
                    bleed_area=1.0, rho0=1.225, gravity=9.81, v_safe=6.0,
                    is_radial=is_radial)
                if not math.isfinite(n) or n <= 0:
                    continue
                s = (n ** 1.5) if is_radial else n
                area = 1.0 + drag * s
                vt = aero.local_terminal_velocity(payload + mass * n, 9.81, 1.225, area)
                self.assertAlmostEqual(vt, 6.0, delta=0.05,
                    msg=f"radial={is_radial} payload={payload}: n={n} -> vt={vt}")

    def test_terminal_limited_count_monotone_in_payload(self) -> None:
        # Heavier craft need more chutes to reach the same safe speed.
        prev = -1.0
        for payload in (1.0, 2.0, 5.0, 10.0, 20.0):
            n = aero.terminal_limited_count(payload, 500.0, 0.1, 1.0, 1.225,
                                            9.81, 6.0, is_radial=True)
            self.assertGreater(n, prev)
            prev = n

    def test_terminal_limited_count_already_safe(self) -> None:
        # A featherweight under a big bleed area is already safe with no chutes.
        n = aero.terminal_limited_count(0.05, 500.0, 0.1, 50.0, 1.225, 9.81,
                                        6.0, is_radial=True)
        self.assertEqual(n, 0.0)

    def test_terminal_limited_count_inline_unreachable(self) -> None:
        # Inline chute whose per-unit mass adds faster than its drag helps at the
        # safe speed → no finite count works (returns inf).
        n = aero.terminal_limited_count(50.0, 1.0, 5.0, 0.5, 0.01, 9.81, 6.0,
                                        is_radial=False)
        self.assertEqual(n, math.inf)


class TestChuteStages(unittest.TestCase):
    def test_kerbin_main_two_stages(self) -> None:
        stages = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                   KERBIN["p0"], KERBIN["H"], label="mk16")
        self.assertIsNotNone(stages)
        semi, full = stages
        self.assertAlmostEqual(semi.gate_altitude_m, 16094.4, delta=0.5)
        self.assertEqual(semi.added_drag_area, 1.0)
        self.assertAlmostEqual(semi.q_safe_kpa, 8.8 * aero.SEMI_DEPLOY_Q_MULT)
        self.assertEqual(semi.floor_altitude_m, 1000.0)
        self.assertEqual(full.gate_altitude_m, 1000.0)
        self.assertEqual(full.added_drag_area, 499.0)
        # Full deploy after a semi is a controlled reef-out on an already-open
        # canopy — not re-gated, so no spurious semi→full bridge burn.
        self.assertEqual(full.q_safe_kpa, math.inf)

    def test_semi_gate_below_deploy_collapses_to_full(self) -> None:
        # p0 barely above the pressure gate: single full stage low down
        stages = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                   4.2, 3000.0)
        self.assertEqual(len(stages), 1)
        self.assertLess(stages[0].gate_altitude_m, 1000.0)
        self.assertEqual(stages[0].added_drag_area, 500.0)

    def test_unopenable_on_thin_body(self) -> None:
        self.assertIsNone(
            aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04, 3.0, 5000.0))


class TestStagedDescent(unittest.TestCase):
    def _kerbin_capsule(self) -> aero.DescentPlan:
        """1.5t pod + 1.25m shield (occluded bleed ~2.46) + one Mk16."""
        stages = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                   KERBIN["p0"], KERBIN["H"])
        return aero.staged_descent(
            v_entry=2246.0, mass_t=1.5, bleed_area=2.46, stages=stages,
            rho0=KERBIN["rho0"], scale_height_m=KERBIN["H"],
            gravity=KERBIN["g"], twr=1.5, max_safe_touchdown=6.0)

    def test_kerbin_capsule_no_bridge_burn(self) -> None:
        """The canonical LKO capsule return: semi opens high, settles, mains
        full-deploy at the gate — no bridge burn.  Touchdown under a single
        Mk16 sits just above the 6 m/s line -> a few m/s flare at most."""
        plan = self._kerbin_capsule()
        self.assertEqual(plan.bridge_dv, 0.0)
        self.assertLess(plan.touchdown_speed, 9.0)
        self.assertLess(plan.finish_dv, 12.0)

    def test_kerbin_two_radial_mains_fully_passive(self) -> None:
        stages = aero.chute_stages(1414.0, 2.8, 8.8, 1000.0, 0.04,
                                   KERBIN["p0"], KERBIN["H"])
        plan = aero.staged_descent(2246.0, 1.5, 2.46, stages,
                                   KERBIN["rho0"], KERBIN["H"], KERBIN["g"],
                                   1.5, 6.0)
        self.assertEqual(plan.total_burn_dv, 0.0)
        self.assertLessEqual(plan.touchdown_speed, 6.0)

    def test_no_stages_is_bleed_plus_full_finish(self) -> None:
        plan = aero.staged_descent(1000.0, 10.0, 2.5, (), DUNA["rho0"],
                                   DUNA["H"], DUNA["g"], 1.5, 6.0)
        self.assertEqual(plan.bridge_dv, 0.0)
        self.assertAlmostEqual(plan.touchdown_speed,
                               1000.0 * math.exp(-0.072), delta=0.1)
        expected = aero.burn_dv_for(plan.touchdown_speed - 6.0, 1.5, 1.0)
        self.assertAlmostEqual(plan.finish_dv, expected, delta=0.1)

    def test_duna_mains_only_charges_real_burn(self) -> None:
        """10t under a single main set on Duna: mains alone can't brake the
        bulk in the thin atmosphere, so a real propulsive burn (~230 m/s) is
        charged — the old flat aero=100 false positive, corrected."""
        stages = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                   DUNA["p0"], DUNA["H"])
        plan = aero.staged_descent(869.0, 10.0, 2.46, stages,
                                   DUNA["rho0"], DUNA["H"], DUNA["g"],
                                   1.5, 6.0)
        self.assertGreater(plan.total_burn_dv, 150.0)
        self.assertTrue(plan.requires_burn)

    def test_duna_drogue_shrinks_the_burn(self) -> None:
        """Adding a drogue set must strictly reduce the propulsive shortfall
        (monotonicity in kit) — the enabler pattern."""
        mains = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                  DUNA["p0"], DUNA["H"], label="main")
        drogues = aero.chute_stages(850.0, 20.0, 70.4, 2500.0, 0.02,
                                    DUNA["p0"], DUNA["H"], label="drogue")
        base = aero.staged_descent(869.0, 10.0, 2.46, mains,
                                   DUNA["rho0"], DUNA["H"], DUNA["g"], 1.5, 6.0)
        with_drogue = aero.staged_descent(869.0, 10.0, 2.46, mains + drogues,
                                          DUNA["rho0"], DUNA["H"], DUNA["g"],
                                          1.5, 6.0)
        self.assertLess(with_drogue.total_burn_dv, base.total_burn_dv)
        # Doc anchor: a x5 drogue set bridges Duna entry for ~free
        self.assertLess(with_drogue.bridge_dv, 60.0)

    def test_duna_inflatable_shrinks_the_burn(self) -> None:
        mains = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                  DUNA["p0"], DUNA["H"])
        small = aero.staged_descent(869.0, 10.0, 2.46, mains, DUNA["rho0"],
                                    DUNA["H"], DUNA["g"], 1.5, 6.0)
        inflatable = aero.staged_descent(
            869.0, 10.0, aero.SHIELD_BLEED_OCCLUSION * 48.18, mains,
            DUNA["rho0"], DUNA["H"], DUNA["g"], 1.5, 6.0)
        self.assertLess(inflatable.total_burn_dv, small.total_burn_dv)

    def test_more_drag_never_costs_more(self) -> None:
        """Monotonicity: any added stage or bleed area may only reduce the
        total burn (superset kits never lose)."""
        mains = aero.chute_stages(500.0, 1.0, 8.8, 1000.0, 0.04,
                                  DUNA["p0"], DUNA["H"])
        for extra_bleed in (0.0, 1.0, 5.0, 20.0):
            a = aero.staged_descent(869.0, 10.0, 2.46 + extra_bleed, mains,
                                    DUNA["rho0"], DUNA["H"], DUNA["g"], 1.5, 6.0)
            b = aero.staged_descent(869.0, 10.0, 2.46 + extra_bleed + 1.0,
                                    mains, DUNA["rho0"], DUNA["H"], DUNA["g"],
                                    1.5, 6.0)
            self.assertLessEqual(b.total_burn_dv, a.total_burn_dv + 1e-6)

    def test_infeasible_twr_returns_inf(self) -> None:
        plan = aero.staged_descent(1000.0, 10.0, 2.5, (), DUNA["rho0"],
                                   DUNA["H"], DUNA["g"], 0.8, 6.0)
        self.assertEqual(plan.finish_dv, math.inf)


if __name__ == "__main__":
    unittest.main()
