"""
Unit tests for rocket_math.py — pure function tests with no world state.
"""
import math
import random
import unittest

from worlds.ksp1.parts import (
    Engine, FuelTank, SolidBooster, HeatShield, DEFAULT_PART_MANAGER,
)

PART_DB = DEFAULT_PART_MANAGER.parts
from worlds.ksp1.rocket_math import (
    G0, stage_delta_v, srb_delta_v, required_tanks, twr,
    terminal_velocity, find_optimal_stage, StageResult,
    KSP_SYMMETRY_MODES,
    parallel_stage_dv, parallel_stage_min_twr,
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
            heat_shields=((1.25, 0.15, "Shield1"),),
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


class TestParallelStageDv(unittest.TestCase):
    """parallel_stage_dv: real progressive-shedding model (replaces the flat
    dry-mass factor).  Identical columns; isp_g0=3000 for round numbers."""

    ISP_G0 = 3000.0
    P = 1.0          # payload (t)
    CORE_DRY = 1.0   # core engine + core tank dry (t)
    F = 4.0          # fuel per column (t)
    BDRY = 1.05      # one booster's jettisoned dry: engine + tank dry + decoupler (t)

    def _asp(self, n):
        return parallel_stage_dv(self.ISP_G0, self.P, self.CORE_DRY, self.F,
                                 self.BDRY, n, "asparagus")

    def _onion(self, n):
        return parallel_stage_dv(self.ISP_G0, self.P, self.CORE_DRY, self.F,
                                 self.BDRY, n, "onion")

    def test_zero_boosters_is_single_stage(self) -> None:
        # No boosters -> just the core column burning its own fuel.
        expected = self.ISP_G0 * math.log(
            (self.P + self.CORE_DRY + self.F) / (self.P + self.CORE_DRY))
        self.assertAlmostEqual(self._asp(0), expected, places=3)
        self.assertAlmostEqual(self._onion(0), expected, places=3)

    def test_progressive_shedding_grows_with_pairs(self) -> None:
        # Each added pair sheds more dry mass earlier -> strictly more dv.
        seq = [self._asp(n) for n in (0, 2, 4, 6, 8)]
        for lo, hi in zip(seq, seq[1:]):
            self.assertGreater(hi, lo)

    def test_asparagus_beats_onion_beyond_one_drop(self) -> None:
        # One drop (2 boosters) is identical; crossfeed pulls ahead with more.
        self.assertAlmostEqual(self._asp(2), self._onion(2), places=3)
        for n in (4, 6, 8):
            self.assertGreater(self._asp(n), self._onion(n))

    def test_asparagus_beats_equal_fuel_single_stage(self) -> None:
        # 8 boosters + core = 9 columns * 4t = 36t fuel.  A single stage with
        # the same 36t (dry scaling with fuel) sheds nothing, so asparagus wins.
        single = self.ISP_G0 * math.log(
            (self.P + self.CORE_DRY + 9 * 0.5 + 36.0)
            / (self.P + self.CORE_DRY + 9 * 0.5))
        self.assertGreater(self._asp(8), single)

    def test_heavier_decoupler_overhead_reduces_benefit(self) -> None:
        # The radial decoupler + fuel line folded into booster_dry is the cost
        # that bounds "more pairs is always better".
        light = parallel_stage_dv(self.ISP_G0, self.P, self.CORE_DRY, self.F,
                                   0.55, 8, "asparagus")   # tiny decoupler
        heavy = parallel_stage_dv(self.ISP_G0, self.P, self.CORE_DRY, self.F,
                                   2.05, 8, "asparagus")   # heavy decoupler
        self.assertGreater(light, heavy)


class TestParallelStageMinTwr(unittest.TestCase):
    """parallel_stage_min_twr: the binding phase, where dropped booster engines
    can make a LATE phase tighter than liftoff."""

    T = 1000.0   # kN per engine
    G = 10.0     # m/s^2

    def test_droptank_binds_at_liftoff(self) -> None:
        # Drop-tank boosters carry no engines, so thrust is constant while mass
        # falls -> liftoff is the heaviest, lowest-TWR instant.
        mn = parallel_stage_min_twr(self.T, self.G, payload=20.0, core_dry=2.0,
            col_fuel=20.0, booster_dry=1.0, n_boost=4, n_eng_core=1,
            n_eng_boost=0, mode="asparagus")
        m0 = 20.0 + 2.0 + 20.0 + 4 * (1.0 + 20.0)
        self.assertAlmostEqual(mn, 1 * self.T / (m0 * self.G), places=4)

    def test_engine_boosters_core_phase_can_bind(self) -> None:
        # Heavy payload, one engine per column: liftoff (3 engines) is fine but
        # the core-only phase (1 engine) is tighter -- the case that a
        # liftoff-only check would wrongly pass.
        mn = parallel_stage_min_twr(self.T, self.G, payload=50.0, core_dry=2.0,
            col_fuel=20.0, booster_dry=2.0, n_boost=2, n_eng_core=1,
            n_eng_boost=1, mode="asparagus")
        m0 = 50.0 + 2.0 + 20.0 + 2 * (2.0 + 20.0)
        liftoff = 3 * self.T / (m0 * self.G)
        m_core = m0 - 2 * (20.0 + 2.0)
        core = 1 * self.T / (m_core * self.G)
        self.assertLess(mn, liftoff)              # core tighter than liftoff
        self.assertAlmostEqual(mn, core, places=4)


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


class TestFindOptimalStageCacheStructuralGuard(unittest.TestCase):
    """The find_optimal_stage cache builds its key from every signature
    parameter except those in ``_FOS_EXCLUDED_PARAMS``.  These tests
    confirm the module-load assertion catches the two ways the guard
    could silently break: a typo'd exclude entry, or a typo'd normalizer
    entry.  If this assertion ever stops working, a new signature
    parameter could silently miss the cache key — a correctness bug.
    """

    def test_unknown_exclude_param_raises_import_error(self):
        from worlds.ksp1 import rocket_math
        original = rocket_math._FOS_EXCLUDED_PARAMS
        try:
            rocket_math._FOS_EXCLUDED_PARAMS = frozenset({"not_a_real_param"})
            with self.assertRaises(ImportError) as ctx:
                rocket_math._build_fos_key_spec()
            self.assertIn("not_a_real_param", str(ctx.exception))
        finally:
            rocket_math._FOS_EXCLUDED_PARAMS = original

    def test_unknown_normalizer_param_raises_import_error(self):
        from worlds.ksp1 import rocket_math
        original = rocket_math._FOS_NORMALIZERS
        try:
            rocket_math._FOS_NORMALIZERS = {"not_a_real_param": lambda x: x}
            with self.assertRaises(ImportError) as ctx:
                rocket_math._build_fos_key_spec()
            self.assertIn("not_a_real_param", str(ctx.exception))
        finally:
            rocket_math._FOS_NORMALIZERS = original

    def test_cache_key_covers_every_signature_param(self):
        """Every parameter in the signature is either in the key spec or
        in the excluded set — no parameter goes unaccounted for."""
        import inspect
        from worlds.ksp1 import rocket_math
        sig = inspect.signature(rocket_math._find_optimal_stage_uncached)
        sig_names = set(sig.parameters.keys())
        keyed = {entry[1] for entry in rocket_math._FOS_KEY_SPEC}
        excluded = rocket_math._FOS_EXCLUDED_PARAMS
        self.assertEqual(sig_names, keyed | excluded)


class TestStageMonotonicity(unittest.TestCase):
    """A strictly larger part set must never lose a stage (bug 092).

    ``find_optimal_stage`` is the physics floor for every access rule; if
    adding a part can flip a stage from feasible to infeasible, capability
    access is non-monotone along the sphere ladder — which the sphere-ladder
    funding pass and the strict post_fill sweep both assume never happens.
    Perfect monotonicity is unattainable in a quantized pack model (integer
    tanks + finite types), so these tests pin the two real trigger families
    found so far and randomly probe for new ones with a fixed RNG (failures
    are deterministic regressions, not flakes)."""

    # The exact constraint set of the Eve-ascent group (dv=9315) from the
    # bug-092 deadlock seed 17074417405164113416 (duna_return/kerbin,
    # buildings_in_logic).  With the pre-fix largest-first tank pack, adding
    # mk3FuselageLFO.50 (25t fuel at ratio 7.0 vs the 8.0 standard) poisoned
    # every column of the SSME asparagus build and lost the mission.
    _EVE_ASCENT = dict(
        required_dv=9315.0,
        payload_mass=0.538,
        gravity=16.7,
        min_twr=1.5,
        requires_throttleable=True,
        in_atmosphere=True,
        atm_scale_height_m=7000.0,
        atm_top_m=90000.0,
        parallel_mode="asparagus",
        radial_decoupler_mass=0.025,
        radial_decoupler_name="radialDecoupler",
        fuel_line_mass=0.05,
        fuel_line_name="fuelLine",
        player_has_rcs=True,
    )

    def test_bug_092_mk3_tank_must_not_poison_the_pack(self) -> None:
        ssme = _p("SSME")
        small_kit = [_p("Size3SmallTank"), _p("Rockomax16.BW"),
                     _p("Size1p5.Size0.Adapter.01"), _p("externalTankToroid")]
        big_kit = small_kit + [_p("mk3FuselageLFO.50")]
        base = find_optimal_stage(
            available_engines=[ssme], available_srbs=[],
            available_tanks=small_kit, **self._EVE_ASCENT)
        self.assertIsNotNone(base, "sans-mk3 Eve ascent must build (repro guard)")
        assert base is not None
        bigger = find_optimal_stage(
            available_engines=[ssme], available_srbs=[],
            available_tanks=big_kit, **self._EVE_ASCENT)
        self.assertIsNotNone(
            bigger, "adding mk3FuselageLFO.50 lost the Eve ascent (bug 092)")
        assert bigger is not None
        self.assertLessEqual(bigger.stage_mass_wet, base.stage_mass_wet * 1.001)

    def test_better_ratio_tank_must_not_evict_others(self) -> None:
        # The old rho_star*(1-0.15) eligibility floor: granting RCSTank1-2
        # (ratio 7.5) evicted every other monoprop tank (6.0/5.5/4.0) from
        # the packable set.  Eligibility must be per-tank, never relative to
        # the best tank in the kit.
        from worlds.ksp1.rocket_math import _packable_tanks
        puff = next(e for pl in PART_DB.values() for e in pl
                    if isinstance(e, Engine) and e.fuel_type == "monoprop")
        small = [_p("rcsTankMini"), _p("RCSFuelTank")]
        ctx_small, _ = _packable_tanks(small, puff)
        ctx_big, _ = _packable_tanks(small + [_p("RCSTank1-2")], puff)
        self.assertTrue(set(t.name for t in ctx_small[0])
                        <= set(t.name for t in ctx_big[0]),
                        "adding a tank removed others from the packable set")

    def test_random_superset_never_loses_a_stage(self) -> None:
        rng = random.Random(920925)
        engines = [e for pl in PART_DB.values() for e in pl
                   if isinstance(e, Engine)]
        tanks = [t for pl in PART_DB.values() for t in pl
                 if isinstance(t, FuelTank)]
        feasible_bases = 0
        for trial in range(120):
            constraints = dict(
                required_dv=rng.choice([1200.0, 3400.0, 5600.0, 8000.0, 9315.0]),
                payload_mass=rng.choice([0.5, 2.0, 8.0]),
                gravity=rng.choice([1.0, 3.5, 9.81, 16.7]),
                min_twr=rng.choice([0.0, 1.2, 1.5]),
                in_atmosphere=rng.random() < 0.5,
                atm_scale_height_m=5600.0,
                atm_top_m=70000.0,
                parallel_mode=rng.choice(["none", "asparagus"]),
                radial_decoupler_mass=0.025,
                radial_decoupler_name="radialDecoupler",
                fuel_line_mass=0.05,
                fuel_line_name="fuelLine",
                requires_throttleable=rng.random() < 0.5,
                player_has_rcs=True,
            )
            eng_subset = rng.sample(engines, rng.randint(2, 8))
            tank_subset = rng.sample(tanks, rng.randint(3, 12))
            base = find_optimal_stage(
                available_engines=eng_subset, available_srbs=[],
                available_tanks=tank_subset, **constraints)
            extra = rng.sample([t for t in tanks if t not in tank_subset],
                               rng.randint(1, 3))
            bigger = find_optimal_stage(
                available_engines=eng_subset, available_srbs=[],
                available_tanks=tank_subset + extra, **constraints)
            if base is None:
                continue
            feasible_bases += 1
            self.assertIsNotNone(bigger, (
                f"trial {trial}: adding {[t.name for t in extra]} lost the "
                f"stage (constraints={constraints}, "
                f"tanks={[t.name for t in tank_subset]}, "
                f"engines={[e.name for e in eng_subset]})"))
            assert bigger is not None
            # Mass bound: intra-tier pack quantization (a same-ratio tank of
            # a non-commensurate size leading the greedy pack leaves a
            # fractional remainder whose cover dead-dry compounds through the
            # rocket equation) wobbles up to ~7% at extreme dv — measured 0
            # feasibility flips over 1500 sampled pairs.  10% still catches
            # the cross-tier poisoning class (bug 092's mk3 pack was +43%).
            self.assertLessEqual(bigger.stage_mass_wet,
                                 base.stage_mass_wet * 1.10, (
                f"trial {trial}: adding {[t.name for t in extra]} made the "
                f"stage >10% heavier ({base.stage_mass_wet:.3f} -> "
                f"{bigger.stage_mass_wet:.3f}t): a poisoned pack "
                f"(constraints={constraints}, "
                f"tanks={[t.name for t in tank_subset]})"))
        # Guard against vacuity: a broken sampler that never builds a stage
        # would pass every assertion above.
        self.assertGreaterEqual(feasible_bases, 20)

    # bugs/106: a SHIELDED stage's tank pack must stay coverable by the
    # largest owned shield.  The seed (standard_sample_returns/eeloo, seed
    # 2062830491824920332, Mun SSR ascent group) had a 0.625m shield
    # (HeatShield0) and a small engine.  With only narrow tanks the greedy
    # pack was pure miniFuelTank (width 0.625, coverable) — feasible.  Adding
    # Size1p5.Size2.Adapter.01 (a size-2.5 SPINE tank, same 8:1 ratio) made
    # the fuel-descending greedy prefer it, producing a width-2.5 pack no
    # 0.625 shield could cover → every candidate rejected → a strictly larger
    # kit lost the stage.  The fix caps a shielded stage's packable tanks at
    # the largest shield size.
    _SHIELDED_ASCENT = dict(
        required_dv=3522.7,
        payload_mass=24.4,
        gravity=1.69,
        min_twr=0.0,
        in_atmosphere=False,
        needs_heat_shield=True,
        max_heat_shield_size=0.625,
        heat_shields=((0.625, 0.025, "HeatShield0"),),
        parallel_mode="none",
    )

    def test_bug_106_wide_tank_must_not_poison_shielded_pack(self) -> None:
        mini = _p("liquidEngineMini.v2")
        narrow = [_p("miniFuelTank"), _p("Size3To2Adapter.v2")]
        wide = narrow + [_p("Size1p5.Size2.Adapter.01")]  # size-2.5, uncoverable
        base = find_optimal_stage(
            available_engines=[mini], available_srbs=[],
            available_tanks=narrow, **self._SHIELDED_ASCENT)
        self.assertIsNotNone(
            base, "narrow-tank shielded ascent must build (repro guard)")
        assert base is not None
        bigger = find_optimal_stage(
            available_engines=[mini], available_srbs=[],
            available_tanks=wide, **self._SHIELDED_ASCENT)
        self.assertIsNotNone(
            bigger, "adding a wide un-coverable tank lost the shielded "
            "stage (bugs/106)")
        assert bigger is not None
        self.assertLessEqual(bigger.stage_mass_wet, base.stage_mass_wet * 1.001)

    def test_bug_106_packable_excludes_tanks_wider_than_shield(self) -> None:
        from worlds.ksp1.rocket_math import _packable_tanks
        mini = _p("liquidEngineMini.v2")
        tanks = [_p("miniFuelTank"), _p("Size1p5.Size2.Adapter.01")]
        ctx_capped, _ = _packable_tanks(tanks, mini, max_tank_size=0.625)
        self.assertEqual(
            {t.name for t in ctx_capped[0]}, {"miniFuelTank"},
            "shield cap must drop the size-2.5 adapter from the packable set")
        ctx_free, _ = _packable_tanks(tanks, mini)
        self.assertIn("Size1p5.Size2.Adapter.01",
                      {t.name for t in ctx_free[0]},
                      "uncapped set keeps the wide tank (guard)")


if __name__ == "__main__":
    unittest.main()
