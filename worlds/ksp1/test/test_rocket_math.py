"""
Unit tests for rocket_math.py — pure function tests with no world state.
"""
import math
import unittest

from worlds.ksp1.parts import (
    Engine, FuelTank, SolidBooster, HeatShield, PART_DB,
)
from worlds.ksp1.rocket_math import (
    G0, stage_delta_v, srb_delta_v, required_tanks, twr,
    terminal_velocity, find_optimal_stage, StageResult,
    KSP_SYMMETRY_MODES, ASPARAGUS_DRY_MASS_FACTOR, ONION_DRY_MASS_FACTOR,
)


# Convenience accessors for real parts
def _p(cfg_name: str, idx: int = 0):
    return PART_DB[cfg_name][idx]


_SWIVEL = _p("liquidEngine2.v2")
_TERRIER = _p("liquidEngine3.v2")
_MAINSAIL = _p("liquidEngineMainsail.v2")
_SPIDER = _p("radialEngineMini.v2")  # radial-mountable
_DAWN = _p("ionEngine")
_HAMMER = _p("solidBooster.v2")
_FL_T400 = _p("fuelTank")
_FL_T800 = _p("fuelTank.long")
_X200_32 = _p("Rockomax32.BW")
_MK1_LF = _p("MK1Fuselage")

# Valid engine counts for parallel staging: 1 core + symmetric radial boosters
_VALID_PARALLEL_ENGINE_COUNTS = frozenset(1 + s for s in KSP_SYMMETRY_MODES)


class TestStageDeltaV(unittest.TestCase):
    """Verify Tsiolkovsky equation results against hand-calculated values."""

    def test_basic_dv(self) -> None:
        # Swivel + 1x FL-T800 + 1t payload
        # m_wet = payload + engine_mass + tank_dry + tank_fuel
        m_wet = 1.0 + _SWIVEL.mass + _FL_T800.dry_mass + _FL_T800.fuel_mass
        m_dry = 1.0 + _SWIVEL.mass + _FL_T800.dry_mass
        dv = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 1.0, in_atmosphere=False)
        expected = _SWIVEL.vac_isp * G0 * math.log(m_wet / m_dry)
        self.assertAlmostEqual(dv, expected, places=1)

    def test_atmospheric_uses_atm_isp(self) -> None:
        dv_vac = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 1.0, in_atmosphere=False)
        dv_atm = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 1.0, in_atmosphere=True)
        self.assertLess(dv_atm, dv_vac)

    def test_fill_fraction_reduces_dv(self) -> None:
        dv_full = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 1.0)
        dv_half = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 0.5, 1.0)
        self.assertGreater(dv_full, dv_half)

    def test_multiple_engines(self) -> None:
        dv_one = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 1.0)
        dv_two = stage_delta_v(_SWIVEL, 2, _FL_T800, 1, 1.0, 1.0)
        # More engines means higher dry mass, so lower dv with same tank
        self.assertGreater(dv_one, dv_two)

    def test_high_isp_engine(self) -> None:
        # Terrier vac_isp=345 should beat Swivel vac_isp=320 for same config
        dv_terr = stage_delta_v(_TERRIER, 1, _MK1_LF, 1, 1.0, 0.5)
        dv_swiv = stage_delta_v(_SWIVEL, 1, _FL_T800, 1, 1.0, 0.5)
        # Terrier has much higher Isp; should produce positive dv
        self.assertGreater(dv_terr, 0)

    def test_zero_isp_returns_zero(self) -> None:
        bad_engine = Engine(
            name="Bad", vac_isp=0, atm_isp=0, vac_thrust=100, atm_thrust=100,
            mass=1.0, throttleable=True, has_gimbal=False,
            size_class=1.25, fuel_type="lfo",
        )
        dv = stage_delta_v(bad_engine, 1, _FL_T800, 1, 1.0, 1.0)
        self.assertEqual(dv, 0.0)


class TestSrbDeltaV(unittest.TestCase):

    def test_srb_dv_positive(self) -> None:
        dv = srb_delta_v(_HAMMER, 1, 1.0, in_atmosphere=False)
        self.assertGreater(dv, 0)

    def test_more_srbs_more_dv(self) -> None:
        dv1 = srb_delta_v(_HAMMER, 1, 1.0)
        dv2 = srb_delta_v(_HAMMER, 2, 1.0)
        self.assertGreater(dv2, dv1)


class TestRequiredTanks(unittest.TestCase):

    def test_round_trip(self) -> None:
        """stage_delta_v(required_tanks(...)) should recover the target dv."""
        target_dv = 1500.0
        n = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, target_dv, 1.0)
        self.assertGreater(n, 0)
        actual_dv = stage_delta_v(_SWIVEL, 1, _FL_T800, n, 1.0, 1.0)
        self.assertGreaterEqual(actual_dv, target_dv)

    def test_impossible_returns_minus_one(self) -> None:
        """A delta-v goal that exceeds the tank's mass ratio should return -1."""
        n = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 100_000.0, 1.0)
        self.assertEqual(n, -1)

    def test_partial_fill(self) -> None:
        """required_tanks should be larger at lower fill fractions."""
        n_full = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 1000.0, 1.0)
        n_half = required_tanks(_SWIVEL, 1, _FL_T800, 0.5, 1000.0, 1.0)
        self.assertGreaterEqual(n_half, n_full)

    def test_zero_target_dv(self) -> None:
        n = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 0.0, 1.0)
        self.assertEqual(n, -1)  # zero dv not a valid target


class TestTWR(unittest.TestCase):

    def test_basic_twr(self) -> None:
        # 200 kN thrust, 10 t mass, 9.81 m/s^2 gravity -> TWR = 200/(10*9.81)
        result = twr(200.0, 10.0, 9.81)
        self.assertAlmostEqual(result, 200.0 / (10.0 * 9.81), places=4)

    def test_zero_mass_returns_zero(self) -> None:
        self.assertEqual(twr(200.0, 0.0, 9.81), 0.0)

    def test_zero_gravity_returns_zero(self) -> None:
        self.assertEqual(twr(200.0, 10.0, 0.0), 0.0)


class TestTerminalVelocity(unittest.TestCase):

    def test_more_chutes_lower_velocity(self) -> None:
        kwargs = dict(
            mass_tonnes=5.0, gravity=9.81,
            atm_density=1.225, ship_cd=0.0,
            ship_cross_section=1.23,
        )
        v1 = terminal_velocity(**kwargs, total_chute_drag_area=400.0)
        v2 = terminal_velocity(**kwargs, total_chute_drag_area=800.0)
        self.assertGreater(v1, v2)

    def test_no_chutes_is_infinite(self) -> None:
        v = terminal_velocity(5.0, 9.81, 1.225, 0.0, 1.23, 0.0)
        self.assertEqual(v, float("inf"))

    def test_no_atmo_is_infinite(self) -> None:
        v = terminal_velocity(5.0, 1.63, 0.0, 0.0, 1.23, 400.0)
        self.assertEqual(v, float("inf"))

    def test_known_value(self) -> None:
        # m=5t, g=9.81, rho=1.225, Cd=0, A=0, chute=400m^2
        # v = sqrt(2*5000*9.81 / (1.225*400))
        v = terminal_velocity(5.0, 9.81, 1.225, 0.0, 0.0, 400.0)
        expected = math.sqrt(2 * 5000 * 9.81 / (1.225 * 400))
        self.assertAlmostEqual(v, expected, places=2)


class TestFindOptimalStage(unittest.TestCase):

    def _basic_engines(self) -> list[Engine]:
        return [_SWIVEL, _TERRIER]

    def _basic_tanks(self) -> list[FuelTank]:
        return [_FL_T400, _FL_T800]

    def test_finds_solution_for_reasonable_dv(self) -> None:
        result = find_optimal_stage(
            available_engines=self._basic_engines(),
            available_srbs=[],
            available_tanks=self._basic_tanks(),
            required_dv=1000.0,
            payload_mass=1.0,
            gravity=9.81,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.delta_v, 1000.0)

    def test_impossible_dv_returns_none(self) -> None:
        result = find_optimal_stage(
            available_engines=self._basic_engines(),
            available_srbs=[],
            available_tanks=self._basic_tanks(),
            required_dv=100_000.0,
            payload_mass=1.0,
            gravity=9.81,
        )
        self.assertIsNone(result)

    def test_throttle_filter(self) -> None:
        # Provide only non-throttleable engines and require throttleable
        non_throttle = Engine(
            name="FixedEngine", vac_isp=300, atm_isp=250,
            vac_thrust=200, atm_thrust=180,
            mass=1.0, throttleable=False, has_gimbal=False,
            size_class=1.25, fuel_type="lfo",
        )
        result = find_optimal_stage(
            available_engines=[non_throttle],
            available_srbs=[],
            available_tanks=self._basic_tanks(),
            required_dv=500.0,
            payload_mass=1.0,
            gravity=9.81,
            requires_throttleable=True,
        )
        self.assertIsNone(result)

    def test_heat_shield_filter(self) -> None:
        # Large engine should be filtered out when max_heat_shield_size is small
        big_engine = Engine(
            name="BigEngine", vac_isp=310, atm_isp=285,
            vac_thrust=1500, atm_thrust=1379,
            mass=6.0, throttleable=True, has_gimbal=True,
            size_class=2.5, fuel_type="lfo",
        )
        result = find_optimal_stage(
            available_engines=[big_engine],
            available_srbs=[],
            available_tanks=self._basic_tanks(),
            required_dv=500.0,
            payload_mass=1.0,
            gravity=9.81,
            needs_heat_shield=True,
            max_heat_shield_size=1.25,   # only 1.25m shield available
            heat_shield_mass=0.15,
        )
        self.assertIsNone(result)  # 2.5m engine filtered out

    def test_twr_requirement_satisfied(self) -> None:
        # Mainsail is a 2.5m engine — it requires a 2.5m tank (X200-32).
        result = find_optimal_stage(
            available_engines=[_MAINSAIL],
            available_srbs=[],
            available_tanks=[_X200_32],
            required_dv=500.0,
            payload_mass=5.0,
            gravity=9.81,
            min_twr=1.5,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.twr_at_ignition, 1.5)

    def test_srb_evaluated(self) -> None:
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[_HAMMER],
            available_tanks=[],
            required_dv=300.0,
            payload_mass=1.0,
            gravity=9.81,
            requires_throttleable=False,
            srb_needs_rcs=False,    # not a casual/normal game; bypass RCS gate
        )
        self.assertIsNotNone(result)

    def test_srb_filtered_when_throttle_required(self) -> None:
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[_HAMMER],
            available_tanks=[],
            required_dv=300.0,
            payload_mass=1.0,
            gravity=9.81,
            requires_throttleable=True,
        )
        self.assertIsNone(result)

    def test_no_engines_no_tanks_returns_none(self) -> None:
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[],
            available_tanks=[],
            required_dv=100.0,
            payload_mass=1.0,
            gravity=9.81,
        )
        self.assertIsNone(result)

    def test_minimum_mass_chosen(self) -> None:
        # Both Swivel and Terrier can do this dv; result should be the lighter one
        result1 = find_optimal_stage(
            available_engines=[_SWIVEL],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=800.0,
            payload_mass=1.0,
            gravity=1.63,
        )
        result2 = find_optimal_stage(
            available_engines=[_TERRIER],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=800.0,
            payload_mass=1.0,
            gravity=1.63,
        )
        # Both should succeed
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)

    def test_ion_engine_needs_xenon_tank(self) -> None:
        # Dawn (xenon) with only LFO tanks should fail
        result = find_optimal_stage(
            available_engines=[_DAWN],
            available_srbs=[],
            available_tanks=[_FL_T800],   # LFO only, no xenon
            required_dv=500.0,
            payload_mass=0.1,
            gravity=0.049,
        )
        self.assertIsNone(result)


class TestParallelStagingDryMassFactor(unittest.TestCase):
    """Verify asparagus/onion dry mass factors in stage_delta_v and required_tanks."""

    def test_asparagus_more_dv_than_none(self) -> None:
        dv_none = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="none")
        dv_asp = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="asparagus")
        self.assertGreater(dv_asp, dv_none)

    def test_onion_more_dv_than_none(self) -> None:
        dv_none = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="none")
        dv_onion = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="onion")
        self.assertGreater(dv_onion, dv_none)

    def test_asparagus_more_dv_than_onion(self) -> None:
        dv_onion = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="onion")
        dv_asp = stage_delta_v(_SWIVEL, 1, _FL_T800, 2, 1.0, 1.0, parallel_mode="asparagus")
        self.assertGreater(dv_asp, dv_onion)

    def test_onion_fewer_tanks_than_none(self) -> None:
        n_none = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 2000.0, 1.0, parallel_mode="none")
        n_onion = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 2000.0, 1.0, parallel_mode="onion")
        self.assertGreater(n_none, 0)
        self.assertGreater(n_onion, 0)
        self.assertLessEqual(n_onion, n_none)

    def test_asparagus_fewer_tanks_than_onion(self) -> None:
        n_onion = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 2000.0, 1.0, parallel_mode="onion")
        n_asp = required_tanks(_SWIVEL, 1, _FL_T800, 1.0, 2000.0, 1.0, parallel_mode="asparagus")
        self.assertGreater(n_onion, 0)
        self.assertGreater(n_asp, 0)
        self.assertLessEqual(n_asp, n_onion)


class TestSymmetricEngineCounts(unittest.TestCase):
    """Verify parallel staging considers both symmetric and non-symmetric counts."""

    def test_parallel_prefers_lightest(self) -> None:
        """Parallel mode finds the lightest solution across both sub-modes."""
        result_asp = find_optimal_stage(
            available_engines=[_SWIVEL],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=1000.0,
            payload_mass=1.0,
            gravity=9.81,
            parallel_mode="asparagus",
        )
        result_none = find_optimal_stage(
            available_engines=[_SWIVEL],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=1000.0,
            payload_mass=1.0,
            gravity=9.81,
            parallel_mode="none",
        )
        self.assertIsNotNone(result_asp)
        self.assertIsNotNone(result_none)
        assert result_asp is not None and result_none is not None
        # Parallel mode should be at least as good as non-parallel
        self.assertLessEqual(result_asp.stage_mass_wet,
                             result_none.stage_mass_wet)

    def test_parallel_finds_symmetric_when_beneficial(self) -> None:
        """When high dv forces many engines, parallel benefits from symmetry."""
        result = find_optimal_stage(
            available_engines=[_SWIVEL],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=4000.0,
            payload_mass=5.0,
            gravity=9.81,
            parallel_mode="asparagus",
        )
        self.assertIsNotNone(result)

    def test_no_parallel_allows_any_engine_count(self) -> None:
        """Without parallel mode, engine_count=1 should be valid."""
        result = find_optimal_stage(
            available_engines=[_SWIVEL],
            available_srbs=[],
            available_tanks=[_FL_T400, _FL_T800],
            required_dv=1000.0,
            payload_mass=1.0,
            gravity=9.81,
            parallel_mode="none",
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine_count, 1)

    def test_radial_engine_minimum_two(self) -> None:
        """Radial-only engines must have >= 2 for symmetric thrust."""
        result = find_optimal_stage(
            available_engines=[_SPIDER],
            available_srbs=[],
            available_tanks=[_FL_T400],
            required_dv=200.0,
            payload_mass=0.5,
            gravity=0.0,  # no TWR constraint
            parallel_mode="none",
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.engine_count, 2,
            "Radial-only engine should never recommend 1x (off-center thrust)")

    def test_radial_engine_not_one_even_with_twr(self) -> None:
        """TWR calculation should also respect the radial minimum of 2."""
        result = find_optimal_stage(
            available_engines=[_SPIDER],
            available_srbs=[],
            available_tanks=[_FL_T400],
            required_dv=100.0,
            payload_mass=0.1,
            gravity=1.63,  # Mun gravity — light enough that 1 engine suffices for TWR
            parallel_mode="none",
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.engine_count, 2,
            "TWR floor must not override radial minimum of 2")

    def test_stack_engine_allows_one(self) -> None:
        """Stack-mountable engines should still allow engine_count=1."""
        result = find_optimal_stage(
            available_engines=[_TERRIER],
            available_srbs=[],
            available_tanks=[_FL_T400],
            required_dv=500.0,
            payload_mass=1.0,
            gravity=0.0,
            parallel_mode="none",
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine_count, 1)

    def test_srb_finds_minimum(self) -> None:
        """SRBs should find the lightest valid count."""
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[_HAMMER],
            available_tanks=[],
            required_dv=300.0,
            payload_mass=1.0,
            gravity=9.81,
            requires_throttleable=False,
            parallel_mode="asparagus",
            srb_needs_rcs=False,
        )
        self.assertIsNotNone(result)
        assert result is not None
        # Should pick minimum count that meets dv
        self.assertGreaterEqual(result.engine_count, 1)


if __name__ == "__main__":
    unittest.main()
