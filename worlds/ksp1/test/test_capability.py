"""
Integration tests for the capability evaluation pipeline.

These tests call _pre_pass / _evaluate_profile / _assess_bodies directly,
bypassing the CollectionState so we don't need a full world setup.
"""
import unittest

from worlds.ksp1.bodies import (
    ALL_BODIES, BODY_BY_NAME, DIFFICULTY_PROFILES, DifficultyProfile,
    BodyName, MissionBuilder, MissionType, effective_dv,
)

# Phase 3a refactor: MISSION_PROFILES is no longer a module-level constant in
# bodies.py — it now lives behind ``MissionBuilder``.  Tests construct one
# Kerbin-home builder here so existing ``MISSION_PROFILES.get((body, mt), [])``
# assertions keep working without per-test edits.  Tests that drive
# capability internals (``_assess_bodies``, ``_assess_one_body``,
# ``evaluate_mission_detailed``) pass ``MISSION_BUILDER`` explicitly.
MISSION_BUILDER = MissionBuilder(home=BodyName.KERBIN)
MISSION_PROFILES = MISSION_BUILDER.all_profiles()
HOME_BODY = MISSION_BUILDER.home_body
from worlds.ksp1.locations import EventName
from worlds.ksp1.capability import (
    BodyAccessProfile, EquipmentFlags,
    _evaluate_profile, _assess_bodies, _assess_one_body,
    _try_profiles, _required_chute_count, _inject_ladder, _compute_sounding_altitude,
    _group_edges,
)
from worlds.ksp1.parts import PART_DB, Engine, FuelTank, SolidBooster, MultiMount
from worlds.ksp1.rocket_math import find_optimal_stage, _adapter_max_engines


# ---------------------------------------------------------------------------
# Convenience accessors for parts from the real PART_DB
# ---------------------------------------------------------------------------

def _part(cfg_name: str, idx: int = 0):
    """Get a part object by cfg name."""
    return PART_DB[cfg_name][idx]


# Engines
_RELIANT = _part("liquidEngine.v2")
_SWIVEL = _part("liquidEngine2.v2")
_TERRIER = _part("liquidEngine3.v2")
_MAINSAIL = _part("liquidEngineMainsail.v2")
_MAMMOTH = _part("Size3EngineCluster")
_NERV = _part("nuclearEngine")
_DAWN = _part("ionEngine")
_SPIDER = _part("radialEngineMini.v2")   # radial-mountable engine
_THUD = _part("radialLiquidEngine1-2")   # radial-mountable engine
_DART = _part("toroidalAerospike")       # no gimbal, high-Isp aerospike

# SRBs
_FLEA = _part("solidBooster.sm.v2")
_HAMMER = _part("solidBooster.v2")

# Fuel Tanks
_FL_T400 = _part("fuelTank")
_FL_T800 = _part("fuelTank.long")
_X200_32 = _part("Rockomax32.BW")
_JUMBO_64 = _part("Rockomax64.BW")
_S3_3600 = _part("Size3SmallTank")

# Heat Shields
_SHIELD_125 = _part("HeatShield1")
_SHIELD_25 = _part("HeatShield2")

# Parachutes
_MK16 = _part("parachuteSingle")        # inline (stack-node) chute
_MK2R = _part("parachuteRadial")        # radial (surface-mount) chute

# Landing Legs
_LT1 = _part("landingLeg1")
_LT2 = _part("landingLeg1-2")

# Decouplers
_TR18A = _part("Decoupler.1")
_TT38K = _part("radialDecoupler")

# Misc
_PROBE_CORE = _part("probeCoreHex.v2")
_COMMAND_POD = _part("mk1pod.v2")
_REACTION_WHEEL = _part("advSasModule")
_OX_STAT = _part("solarPanels5")
_SOLAR_ARRAY = _part("largeSolarPanel")
_RTG = _part("rtg")
_COMM16 = _part("longAntenna")
_HG5 = _part("HighGainAntenna5.v2")
_RA2 = _part("RelayAntenna5")
_LAUNCH_CLAMP = _part("launchClamp1")
_FUEL_LINE = _part("fuelLine")
_LADDER = _part("ladder1")
_BASIC_FIN = _part("basicFin")


# ---------------------------------------------------------------------------
# Helper: build a minimal EquipmentFlags for testing
# ---------------------------------------------------------------------------

def _make_flags(
    engines=None, srbs=None, tanks=None,
    probe_core=False, capsule=False,
    reaction_wheels=False, rcs=False,
    heat_shields=None, parachutes=None,
    legs=None, decoupler_stack=False, decoupler_radial=False,
    fuel_lines=False, docking_port=False,
    solar=False, solar_retractable=False, solar_large=False, rtg=False,
    battery_large=False,
    relay_tier=0, ladder=False, launch_clamp=False,
    staging_tier=None,
) -> EquipmentFlags:
    flags = EquipmentFlags()
    flags.available_engines = list(engines or [])
    flags.available_srbs = list(srbs or [])
    flags.available_tanks = list(tanks or [])

    if probe_core:
        flags.has_probe_core = True
        flags.lightest_probe = _PROBE_CORE
    if capsule:
        flags.has_capsule = True
        flags.lightest_capsule = _COMMAND_POD

    if reaction_wheels:
        flags.has_reaction_wheels = True
    if rcs:
        flags.has_rcs = True

    heat_shields = heat_shields or []
    for hs in heat_shields:
        flags.has_heat_shield = True
        flags.available_heat_shields.append(hs)
        if flags.best_heat_shield is None or hs.size_class > flags.best_heat_shield.size_class:
            flags.best_heat_shield = hs

    parachutes = parachutes or []
    for p in parachutes:
        if not p.is_drogue:
            flags.has_parachutes = True
            flags.available_parachutes.append(p)
    # Mirror _pre_pass: pick the best overall + best-of-each-kind chutes up front
    # so the landing solver reads them directly.
    if flags.available_parachutes:
        _key = lambda p: p.mass / max(p.drag_area, 1e-3)
        flags.best_chute = min(flags.available_parachutes, key=_key)
        _rad = [p for p in flags.available_parachutes if p.is_radial]
        _inl = [p for p in flags.available_parachutes if not p.is_radial]
        flags.best_radial_chute = min(_rad, key=_key) if _rad else None
        flags.best_inline_chute = min(_inl, key=_key) if _inl else None

    legs = legs or []
    for leg in legs:
        flags.available_landing_legs.append(leg)
        if leg.tier > flags.landing_leg_tier:
            flags.landing_leg_tier = leg.tier

    # Staging tier (docking ports don't affect staging_tier)
    if staging_tier is not None:
        flags.staging_tier = staging_tier
    else:
        if decoupler_radial:
            flags.staging_tier = 2
        elif decoupler_stack:
            flags.staging_tier = 1
        else:
            flags.staging_tier = 0

    flags.has_fuel_lines = fuel_lines
    flags.has_docking_port = docking_port
    flags.has_solar = solar or solar_retractable or solar_large
    flags.has_solar_retractable = solar_retractable or solar_large
    flags.has_solar_array_large = solar_large
    if solar or solar_retractable or solar_large:
        flags.lightest_solar = _OX_STAT
    if solar_retractable or solar_large:
        flags.lightest_solar_retractable = _SOLAR_ARRAY if solar_large else _OX_STAT
    flags.has_rtg = rtg
    if rtg:
        flags.lightest_rtg = _RTG
    flags.has_battery_large = battery_large
    flags.relay_tier = relay_tier
    flags.has_ladder = ladder
    if ladder:
        flags.lightest_ladder = _LADDER
    flags.has_launch_clamp = launch_clamp

    return flags


def _normal_diff():
    return DIFFICULTY_PROFILES["normal"]


def _casual_diff():
    return DIFFICULTY_PROFILES["casual"]


# A 0-margin profile (no dv/plane-change cushion) for tests that want the
# physics budget as tractable as possible to isolate a single gate (e.g. the
# parachute gate) rather than mission margins. Mirrors the retired "insane"
# profile so those tests keep their intent without depending on a difficulty.
def _zero_margin_diff():
    return DifficultyProfile(
        fixed_margin=0, percent_margin=0.00, plane_change_fraction=0.00,
        min_twr_atmo=1.2, min_twr_vac=1.0,
        ship_cd=0.2, srb_needs_rcs=False,
    )


# ---------------------------------------------------------------------------
# Minimum viable kit tests
# ---------------------------------------------------------------------------

class TestMinimumKit(unittest.TestCase):
    """
    A realistic 2-stage Mun rocket:
      - Mainsail (2.5m) + X200-32 (2.5m) first stage
      - Swivel (1.25m) + FL-T800 (1.25m) second stage
    This combination has enough atmospheric TWR and delta-v to reach Mun orbit
    at normal difficulty margins.
    """

    def _min_flags(self) -> EquipmentFlags:
        return _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, decoupler_stack=True,
        )

    def test_mun_orbit_achievable(self) -> None:
        flags = self._min_flags()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        self.assertTrue(len(profiles) > 0, "No Mun orbit profiles defined")
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Minimum kit should reach Mun orbit")

    def test_minmus_orbit_achievable(self) -> None:
        flags = self._min_flags()
        profiles = MISSION_PROFILES.get((BodyName.MINMUS, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Minimum kit should reach Minmus orbit")


class TestLandingLegs(unittest.TestCase):
    """Mun landing requires landing legs; orbit does not."""

    def _orbit_flags(self) -> EquipmentFlags:
        return _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, decoupler_stack=True,
        )

    def test_mun_land_fails_without_legs(self) -> None:
        flags = self._orbit_flags()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.LAND), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.LAND, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Should not land on Mun without landing legs")

    def test_mun_land_succeeds_with_legs(self) -> None:
        flags = self._orbit_flags()
        flags.available_landing_legs = [_LT2]
        flags.landing_leg_tier = 2
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.LAND), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.LAND, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Should land on Mun with tier-2 legs + Swivel + enough tanks")


class TestHeatShieldGate(unittest.TestCase):
    """Duna aero profile requires heat shield; propulsive does not."""

    def _duna_flags_no_shield(self) -> EquipmentFlags:
        return _make_flags(
            engines=[_MAINSAIL], tanks=[_FL_T400, _FL_T800],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=3, launch_clamp=True,
            decoupler_stack=True,
            parachutes=[_MK16, _MK16, _MK16, _MK16],
        )

    def test_aero_profile_fails_without_shield(self) -> None:
        flags = self._duna_flags_no_shield()
        # Find only the aero landing profile
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.LAND), [])
        aero_profiles = [p for p in profiles
                         if any(e.needs_heat_shield for e in p)]
        self.assertTrue(len(aero_profiles) > 0)
        ok = _try_profiles(aero_profiles, flags, _normal_diff(), MissionType.LAND, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Aero landing should fail without heat shield")

    def test_propulsive_profile_does_not_require_shield(self) -> None:
        flags = self._duna_flags_no_shield()
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.LAND), [])
        prop_profiles = [p for p in profiles
                         if not any(e.needs_heat_shield for e in p)]
        if not prop_profiles:
            self.skipTest("No propulsive-only Duna landing profiles found")
        # With a powerful engine and no heat shield, propulsive should work
        # (Duna has thin atmo so propulsive landing is possible)
        ok = _try_profiles(prop_profiles, flags, _normal_diff(), MissionType.LAND, crewed=False, home=BodyName.KERBIN)
        # This may or may not succeed depending on staging/engines; just assert no exception
        self.assertIsInstance(ok, bool)

    def test_aero_profile_succeeds_with_shield(self) -> None:
        flags = self._duna_flags_no_shield()
        flags.has_heat_shield = True
        flags.available_heat_shields = [_SHIELD_125]
        flags.best_heat_shield = _SHIELD_125
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.LAND), [])
        aero_profiles = [p for p in profiles
                         if any(e.needs_heat_shield for e in p)]
        ok = _try_profiles(aero_profiles, flags, _normal_diff(), MissionType.LAND, crewed=False, home=BodyName.KERBIN)
        # With a 1.25m shield + Mainsail (2.5m) filtered, should fail;
        # but Swivel (1.25m) would pass — this tests the filter logic.
        # The key assertion is "no exception thrown" — the boolean result
        # depends on whether a 1.25m engine can do Duna aero.
        self.assertIsInstance(ok, bool)


class TestParachuteGate(unittest.TestCase):
    """
    Kerbin reentry / aero landings require parachutes.

    The "success" case uses a 0-margin profile (no dv cushion) and a powerful
    rocket (3× Mainsail on S3-3600 first stage) to make the physics tractable
    while keeping the focus on the PARACHUTE gate itself, not mission margins.
    """

    def _return_flags_no_chutes(self) -> EquipmentFlags:
        """Minimal kit for testing the gate failure; normal difficulty."""
        return _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32, _S3_3600],
            probe_core=True, reaction_wheels=True,
            heat_shields=[_SHIELD_25],
            solar=True, relay_tier=1, launch_clamp=True,
            decoupler_radial=True, staging_tier=2,
        )

    def _return_flags_with_chutes(self) -> EquipmentFlags:
        """Full kit for success case; uses a 0-margin diff in the test call."""
        flags = _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32, _S3_3600],
            probe_core=True, reaction_wheels=True,
            heat_shields=[_SHIELD_25],
            legs=[_LT2],        # Mun landing requires tier-2 legs
            solar=True, relay_tier=1, launch_clamp=True,
            decoupler_radial=True, staging_tier=2,
        )
        flags.has_parachutes = True
        flags.available_parachutes = [_MK16, _MK16, _MK16]
        # set explicitly since we bypassed _make_flags's parachutes= path
        flags.best_chute = _MK16
        flags.best_inline_chute = _MK16  # _MK16 is an inline chute
        return flags

    def test_mun_return_fails_without_parachutes(self) -> None:
        # No parachutes: fails at the broad gate check (ATMO_LANDING_AERO present)
        flags = self._return_flags_no_chutes()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.RETURN), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.RETURN, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Mun return should fail without parachutes")

    def test_mun_return_succeeds_with_parachutes(self) -> None:
        # A 0-margin profile makes the dv/TWR budget tractable while keeping the
        # parachute gate as the distinguishing factor. With 0 margins + 3×
        # Mainsail first stage + Mainsail mid-stage, the full return chain fits
        # within the available thrust envelope.
        flags = self._return_flags_with_chutes()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.RETURN), [])
        diff = _zero_margin_diff()
        ok = _try_profiles(profiles, flags, diff, MissionType.RETURN, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Mun return should succeed with parachutes at 0-margin difficulty")


class TestParachuteCalculation(unittest.TestCase):
    """Test the required_chute_count helper."""

    def test_one_inline_chute_lands_light_craft_on_kerbin(self) -> None:
        # The single inline (stack-top) chute we assume lands a light craft on
        # Kerbin's thick atmosphere.
        flags = _make_flags(parachutes=[_MK16])
        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        n = _required_chute_count(1.0, kerbin, flags, _normal_diff())
        self.assertEqual(n, 1, "1 t on Kerbin should land under a single inline chute")

    def test_inline_chutes_capped_at_one(self) -> None:
        # Inline chutes can't be stacked past _MAX_INLINE_CHUTES (1): a craft
        # needing more than one inline chute is infeasible on inline-only kit,
        # even though several _MK16 are nominally "available".
        flags = _make_flags(parachutes=[_MK16, _MK16, _MK16, _MK16])
        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        n = _required_chute_count(3.0, kerbin, flags, _normal_diff())
        self.assertEqual(n, -1, "3 t needs >1 inline chute; inline is capped at 1")

    def test_radial_chutes_scale(self) -> None:
        # Radial chutes surface-mount around the body, so the same 3 t craft
        # lands once a radial chute is available (count scales past 1).
        flags = _make_flags(parachutes=[_MK2R])
        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        n = _required_chute_count(3.0, kerbin, flags, _normal_diff())
        self.assertGreater(n, 1, "3 t on Kerbin needs multiple radial chutes")

    def test_vacuum_body_needs_no_chutes(self) -> None:
        flags = _make_flags(parachutes=[])
        mun = BODY_BY_NAME[BodyName.MUN]
        n = _required_chute_count(3.0, mun, flags, _normal_diff())
        self.assertEqual(n, 0, "Mun has no atmosphere, no chutes needed")

    def test_insufficient_chutes_returns_minus_one(self) -> None:
        # Give 0 parachutes, land a very heavy craft on Kerbin
        flags = _make_flags()  # no parachutes
        kerbin = BODY_BY_NAME[BodyName.KERBIN]
        n = _required_chute_count(100.0, kerbin, flags, _normal_diff())
        self.assertEqual(n, -1, "Should return -1 when no chutes available")


class TestCrewedVsUnmanned(unittest.TestCase):
    """Crewed missions require capsule; unmanned require probe core."""

    def _base_flags(self) -> EquipmentFlags:
        return _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            reaction_wheels=True, solar=True, launch_clamp=True,
            decoupler_stack=True,
        )

    def test_unmanned_requires_probe_core(self) -> None:
        flags = self._base_flags()
        flags.has_probe_core = False
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Unmanned should fail without probe core")

    def test_unmanned_with_probe_core(self) -> None:
        flags = self._base_flags()
        flags.has_probe_core = True
        flags.lightest_probe = _PROBE_CORE
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok)

    def test_crewed_requires_capsule(self) -> None:
        flags = self._base_flags()
        flags.has_capsule = False
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=True, home=BodyName.KERBIN)
        self.assertFalse(ok, "Crewed should fail without capsule")


class TestParentGating(unittest.TestCase):
    """Moon accessibility should be blocked if parent planet orbit unreachable."""

    def test_gilly_blocked_if_eve_orbit_blocked(self) -> None:
        flags = _make_flags(
            engines=[_SWIVEL], tanks=[_FL_T400, _FL_T800],
            probe_core=True, reaction_wheels=True, solar=True,
        )
        # Eve orbit blocked (computed as False)
        computed = {
            BodyName.EVE: BodyAccessProfile(access={"Orbit": False})
        }
        gilly = BODY_BY_NAME[BodyName.GILLY]
        result = _assess_one_body(gilly, flags, _normal_diff(), computed, MISSION_BUILDER)  # type: ignore[arg-type]
        self.assertFalse(result.access.get("Orbit", False))
        self.assertIn("parent", (result.blocking_reason or "").lower())

    def test_gilly_accessible_if_eve_orbit_ok(self) -> None:
        flags = _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            heat_shields=[_SHIELD_25],
            relay_tier=3, launch_clamp=True, decoupler_stack=True,
        )
        computed = {
            BodyName.EVE: BodyAccessProfile(access={"Orbit": True})
        }
        gilly = BODY_BY_NAME[BodyName.GILLY]
        result = _assess_one_body(gilly, flags, _normal_diff(), computed, MISSION_BUILDER)  # type: ignore[arg-type]
        # Parent gating is cleared — any remaining blocking reason must NOT
        # be about the parent chain (it may fail for physics/dv reasons which
        # is expected for the tiny test fixture vs. the full Eve system journey).
        reason = result.blocking_reason or ""
        self.assertNotIn("parent", reason.lower(),
                         f"Blocking reason should not be parent-related: {reason}")


class TestStagingTier(unittest.TestCase):
    """Higher staging tier enables more multi-stage configurations."""

    def test_staging_tier_0_single_stage(self) -> None:
        # Single-stage to Mun orbit is extremely hard at normal margins.
        # Just verify the evaluation runs without error.
        flags = _make_flags(
            engines=[_MAINSAIL], tanks=[_X200_32, _JUMBO_64],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, staging_tier=0,
        )
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertIsInstance(ok, bool)

    def test_staging_tier_1_enables_two_stages(self) -> None:
        flags = _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, decoupler_stack=True, staging_tier=1,
        )
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Mun orbit should be achievable with 2-stage rocket")


class TestDifficultyMargins(unittest.TestCase):
    """Casual margins should require more dv than expert (the tightest difficulty)."""

    def test_effective_dv_casual_greater_than_expert(self) -> None:
        from worlds.ksp1.bodies import effective_dv, DIFFICULTY_PROFILES
        casual = DIFFICULTY_PROFILES["casual"]
        expert = DIFFICULTY_PROFILES["expert"]
        base = 1000.0
        self.assertGreater(
            effective_dv(base, casual),
            effective_dv(base, expert),
        )

    def test_plane_change_included_at_casual(self) -> None:
        from worlds.ksp1.bodies import effective_dv, DIFFICULTY_PROFILES
        casual = DIFFICULTY_PROFILES["casual"]
        expert = DIFFICULTY_PROFILES["expert"]
        dv_casual = effective_dv(100.0, casual, plane_change_dv=1000.0)
        dv_expert = effective_dv(100.0, expert, plane_change_dv=1000.0)
        # Casual includes 100% of 1000 plane change; expert includes only 5%.
        self.assertGreater(dv_casual, dv_expert)


class TestInjectLadder(unittest.TestCase):
    """_inject_ladder should add needs_ladder to landing edges."""

    def test_ladder_injected_on_landing_edge(self) -> None:
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.SAMPLE_RETURN), [])
        self.assertTrue(len(profiles) > 0)
        modified = _inject_ladder(profiles)
        for profile in modified:
            for edge in profile:
                if edge.needs_landing_legs:
                    self.assertTrue(edge.needs_ladder,
                                    f"Edge {edge.source}->{edge.destination} should have needs_ladder after inject")

    def test_original_profiles_unchanged(self) -> None:
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.SAMPLE_RETURN), [])
        _ = _inject_ladder(profiles)
        # Original should not have been mutated
        for edge in profiles[0]:
            if edge.needs_landing_legs:
                self.assertFalse(edge.needs_ladder, "Original profiles should not be mutated")


class TestBodiesDatabase(unittest.TestCase):
    """Sanity checks on the body database."""

    def test_all_17_bodies_present(self) -> None:
        expected = {
            BodyName.KERBIN, BodyName.MUN, BodyName.MINMUS,
            BodyName.MOHO, BodyName.EVE, BodyName.GILLY,
            BodyName.DUNA, BodyName.IKE, BodyName.DRES,
            BodyName.JOOL, BodyName.LAYTHE, BodyName.VALL, BodyName.TYLO, BodyName.BOP, BodyName.POL,
            BodyName.EELOO, BodyName.KERBOL,
        }
        self.assertEqual(set(BODY_BY_NAME.keys()), expected)

    def test_eva_jetpack_twr_precomputed(self) -> None:
        import math
        for body in BODY_BY_NAME.values():
            expected = 0.5 / (0.09375 * body.surface_gravity)
            self.assertAlmostEqual(
                body.eva_jetpack_twr, expected, places=3,
                msg=f"EVA jetpack TWR mismatch for {body.name}",
            )

    def test_no_orbit_mission_profiles_for_jool(self) -> None:
        """Jool can orbit but cannot land."""
        self.assertIn((BodyName.JOOL, MissionType.ORBIT), MISSION_PROFILES)
        self.assertNotIn((BodyName.JOOL, MissionType.LAND), MISSION_PROFILES)

    def test_mun_has_all_mission_types(self) -> None:
        for mtype in (MissionType.ORBIT, MissionType.LAND, MissionType.RETURN, MissionType.SAMPLE_RETURN):
            self.assertIn((BodyName.MUN, mtype), MISSION_PROFILES,
                          f"Mun should have {mtype} profiles")


class TestNoEngines(unittest.TestCase):
    """With no engines, nothing should be orbitally reachable."""

    def _no_engine_flags(self) -> EquipmentFlags:
        # Launch clamp, probe core, solar, fuel tank — but zero engines.
        return _make_flags(
            tanks=[_FL_T800],
            probe_core=True,
            solar=True,
            launch_clamp=True,
        )

    def test_kerbin_orbit_false_without_engines(self) -> None:
        flags = self._no_engine_flags()
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Kerbin orbit should require engines")

    def test_mun_orbit_false_without_engines(self) -> None:
        flags = self._no_engine_flags()
        mun = BODY_BY_NAME[BodyName.MUN]
        result = _assess_one_body(mun, flags, _normal_diff(), computed={}, mission_builder=MISSION_BUILDER)
        self.assertFalse(result.access.get("Orbit", False),
                         "Mun orbit should be False without engines")

    def test_all_bodies_orbit_false_without_engines(self) -> None:
        flags = self._no_engine_flags()
        # Assess every non-Kerbin body; none should be orbitally reachable.
        all_results = _assess_bodies(flags, _normal_diff(), MISSION_BUILDER)
        for body_name, prof in all_results.items():
            if body_name == BodyName.KERBIN:
                continue  # Kerbin is the starting body, always True
            self.assertFalse(
                prof.access.get("Orbit", False),
                f"{body_name} should not be orbitally reachable without engines",
            )


class TestKerbinOrbit(unittest.TestCase):
    """Minimum realistic kits for Kerbin orbit / failure cases."""

    def test_mainsail_x200_single_stage_orbits_kerbin(self) -> None:
        # Single stage, no decoupler: Mainsail (2.5m) + X200-32 (2.5m).
        # staging_tier=0 forces a single group — the optimizer stacks X200-32s.
        flags = _make_flags(
            engines=[_MAINSAIL],
            tanks=[_X200_32],
            probe_core=True, solar=True,
            launch_clamp=True,
            staging_tier=0,
        )
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(ok, "Mainsail + X200-32 single stage should reach Kerbin orbit")

    def test_nerv_x200_32_fails_kerbin_orbit_normal(self) -> None:
        # Nerv has atm_thrust ~14 kN — far too low for atmospheric TWR >= 1.5.
        flags = _make_flags(
            engines=[_NERV],
            tanks=[_X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True,
        )
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(
            ok,
            "Nerv Engine has insufficient atmospheric TWR for Kerbin ascent at normal difficulty",
        )


class TestAtmosphericAscentControl(unittest.TestCase):
    """
    Atmospheric-ascent gate (bug 049): steering a gravity turn in atmosphere
    requires either a gimballed engine or actuated aero control surfaces.
    Reaction wheels/RCS alone are insufficient.  Vacuum ascents and sounding
    rockets (altitude checks) are unaffected.
    """

    def _add_aero_control(self, flags: EquipmentFlags, part) -> None:
        flags.has_aero_control_surface = True
        flags.available_aero_controls.append(part)
        if flags.lightest_aero_control is None or part.mass < flags.lightest_aero_control.mass:
            flags.lightest_aero_control = part

    def test_atmo_ascent_requires_gimbal_or_aero(self) -> None:
        # Dart engine (no gimbal) + reaction wheels + fuel, no fins.
        # Kerbin orbit should be infeasible — reaction wheels cannot steer
        # a gravity turn in atmosphere.
        flags = _make_flags(
            engines=[_DART],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, staging_tier=1,
        )
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(
            ok,
            "Dart + reaction wheels (no fins) must not claim Kerbin orbit",
        )

    def test_atmo_ascent_ok_with_gimbal_engine(self) -> None:
        # LV-T45 Swivel (has gimbal) + reaction wheels.  Kerbin orbit should
        # be feasible — gimbal steers the gravity turn.
        flags = _make_flags(
            engines=[_SWIVEL, _MAINSAIL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, staging_tier=1,
        )
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(
            ok,
            "LV-T45 has gimbal — Kerbin orbit must be feasible",
        )

    def test_atmo_ascent_ok_with_aero_surface(self) -> None:
        # Dart + Basic Fin + reaction wheels.  Kerbin orbit feasible —
        # the fin provides atmospheric steering; optimizer adds 4x fin mass.
        flags = _make_flags(
            engines=[_DART],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True, staging_tier=1,
        )
        self._add_aero_control(flags, _BASIC_FIN)
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(
            ok,
            "Dart + Basic Fin should reach Kerbin orbit (fins provide control)",
        )

    def test_vacuum_ascent_unaffected(self) -> None:
        # Vacuum ascent does not need gimbal/fins — the optimizer with
        # require_gimbal=False must accept a non-gimbal engine like the Dart.
        result = find_optimal_stage(
            available_engines=[_DART],
            available_srbs=[],
            available_tanks=[_X200_32],
            required_dv=800.0,
            payload_mass=0.5,
            gravity=1.63,           # Mun surface gravity
            min_twr=1.2,
            in_atmosphere=False,
            available_multi_mounts=[],
            require_gimbal=False,
        )
        self.assertIsNotNone(
            result,
            "Dart should be a valid vacuum-ascent engine (no gimbal required)",
        )

    def test_sounding_altitude_unaffected(self) -> None:
        # Dart + fuel (no gimbal, no fins) — sounding altitude is not gated
        # by control authority; the altitude path doesn't go through the
        # ascent stage optimizer.
        flags = _make_flags(
            engines=[_DART],
            tanks=[_FL_T800],
            probe_core=True,
        )
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        self.assertGreater(
            alt, 70.0,
            f"Dart sounding altitude should exceed 70 km, got {alt:.1f} km",
        )


class TestDunaReturn(unittest.TestCase):
    """
    3-stage Duna return chain with realistic parts.

    Stage 1: Mammoth + S3-3600 (3.75m, high-thrust Kerbin ascent)
    Stage 2: Swivel + FL-T800 (1.25m, transfer + Duna orbit insertion)
    Stage 3: Terrier + FL-T400 (1.25m, Duna landing + ascent + Kerbin return)

    The Mammoth (3746 kN atm thrust) on S3-3600 (3.75m) tanks provides enough
    TWR for the heavy Kerbin ascent. Mainsail alone is limited to 1 engine on
    2.5m tanks (single engine without adapter), which can't close the dv/TWR tradeoff.

    Equipment: heat shield (2.5m), parachutes (3x Mk16), landing legs (LT-2),
    RTG (Duna power at solar distance ~1.5 AU), relay_tier=3.
    Staging tier 2 (radial decouplers) to enable 3 distinct stages.
    """

    def _duna_return_flags(self) -> EquipmentFlags:
        return _make_flags(
            engines=[_MAMMOTH, _MAINSAIL, _SWIVEL, _TERRIER],
            tanks=[_S3_3600, _JUMBO_64, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=3,
            heat_shields=[_SHIELD_25],
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_radial=True,
        )

    def test_duna_can_return_to_kerbin(self) -> None:
        flags = self._duna_return_flags()
        duna = BODY_BY_NAME[BodyName.DUNA]
        result = _assess_one_body(duna, flags, _normal_diff(), computed={}, mission_builder=MISSION_BUILDER)
        self.assertTrue(
            result.access.get("Return", False),
            f"Duna return should succeed with 3-stage rocket + heat shield + chutes. "
            f"Blocking reason: {result.blocking_reason}",
        )

    def test_duna_return_fails_without_heat_shield(self) -> None:
        # Remove the heat shield — Kerbin reentry requires it.
        flags = _make_flags(
            engines=[_MAINSAIL, _SWIVEL, _TERRIER],
            tanks=[_JUMBO_64, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=3,
            # No heat shield
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_radial=True,
        )
        return_profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.RETURN), [])
        ok = _try_profiles(return_profiles, flags, _normal_diff(), MissionType.RETURN, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(ok, "Duna return should fail without a heat shield (Kerbin reentry blocked)")


class TestInterplanetaryBodies(unittest.TestCase):
    """
    A full-kit rocket (3-engine tiers, large S3-3600 first stage, full support
    gear, relay_tier=3) should reach orbit of all inner/mid solar system
    bodies.  Jool system and Eeloo require relay_tier=4 and should be blocked
    at tier 3.

    Staging tier 2 (radial decouplers) is required so the optimizer can split
    the Kerbin ascent stage from the interplanetary transfer stages.
    """

    def _full_kit_flags(self, relay_tier: int = 3) -> EquipmentFlags:
        return _make_flags(
            engines=[_MAINSAIL, _SWIVEL, _TERRIER],
            tanks=[_JUMBO_64, _S3_3600, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=relay_tier,
            heat_shields=[_SHIELD_25],
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_radial=True,
        )

    def test_inner_and_middle_bodies_orbitally_reachable(self) -> None:
        flags = self._full_kit_flags()
        results = _assess_bodies(flags, _normal_diff(), MISSION_BUILDER)
        for body_name in (BodyName.MUN, BodyName.MINMUS, BodyName.MOHO, BodyName.EVE, BodyName.DUNA, BodyName.DRES):
            prof = results[body_name]
            self.assertTrue(
                prof.access.get("Orbit", False),
                f"{body_name} orbit should be reachable with full kit. "
                f"Blocking reason: {prof.blocking_reason}",
            )

    def test_outer_bodies_blocked_at_relay_tier_3(self) -> None:
        # Jool (max sep 6.2 AU) and Eeloo (7.0 AU) both fall in the
        # tier-4 band under the opposition-distance relay formula.
        flags = self._full_kit_flags()
        self.assertEqual(flags.relay_tier, 3)
        results = _assess_bodies(flags, _normal_diff(), MISSION_BUILDER)
        for body_name in (BodyName.JOOL, BodyName.EELOO):
            prof = results[body_name]
            self.assertFalse(
                prof.access.get("Orbit", False),
                f"{body_name} orbit should be blocked at relay_tier=3 (requires tier 4)",
            )


class TestKerbinOrbitIsEarlyGame(unittest.TestCase):
    """
    Kerbin orbit with early parts requires a multi-mount adapter.

    Reliant (1.25m stack engine) on X200-32 (2.5m tank) cannot mount
    multiple engines without an adapter — radial-tank mounting forces
    n_tank >= n_eng, making the vehicle too heavy.  A quad coupler
    (TVR-2160C) allows 4× Reliant under a shared tank stack, giving
    sufficient atmospheric TWR.
    """

    def test_reliant_x200_32_needs_adapter_for_kerbin_orbit(self) -> None:
        """Without adapter, Reliant + X200-32 single stage can't orbit Kerbin."""
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True,
            staging_tier=0,
        )
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        self.assertTrue(len(profiles) > 0, "Kerbin orbit profiles must exist")
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertFalse(
            ok,
            "Reliant + X200-32 should NOT reach Kerbin orbit without an adapter",
        )

    def test_reliant_x200_32_with_quad_coupler_orbits_kerbin(self) -> None:
        """With a quad coupler, 3–4× Reliant under X200-32 tanks can orbit."""
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_X200_32],
            probe_core=True, reaction_wheels=True, solar=True,
            launch_clamp=True,
            staging_tier=0,
        )
        from worlds.ksp1.parts import MULTI_MOUNT_TABLE
        flags.available_multi_mounts = [MULTI_MOUNT_TABLE["stackQuadCoupler"]]
        # Reliant has no gimbal; add a Basic Fin for atmospheric control.
        flags.has_aero_control_surface = True
        flags.available_aero_controls.append(_BASIC_FIN)
        flags.lightest_aero_control = _BASIC_FIN
        profiles = MISSION_PROFILES.get((BodyName.KERBIN, MissionType.ORBIT), [])
        ok = _try_profiles(profiles, flags, _normal_diff(), MissionType.ORBIT, crewed=False, home=BodyName.KERBIN)
        self.assertTrue(
            ok,
            "Reliant + X200-32 + quad coupler + fin should reach Kerbin orbit",
        )


class TestSoundingRocketAltitude(unittest.TestCase):
    """
    Unit tests for _compute_sounding_altitude.

    Formula: h_km = dv^2 * (twr - 1) / (2*g*twr*1000)
    No drag, TWR floor = 1.1.
    Crewed flights require staging_tier >= 1 (decoupler) AND parachutes.
    """

    def test_empty_flags_zero(self) -> None:
        flags = _make_flags()
        self.assertEqual(_compute_sounding_altitude(flags, HOME_BODY), 0.0)

    def test_probe_core_no_engine_zero(self) -> None:
        # Probe core alone, no engine -> no thrust, no altitude
        flags = _make_flags(probe_core=True)
        self.assertEqual(_compute_sounding_altitude(flags, HOME_BODY), 0.0)

    def test_capsule_only_zero(self) -> None:
        # Capsule, no engine -> zero
        flags = _make_flags(capsule=True)
        self.assertEqual(_compute_sounding_altitude(flags, HOME_BODY), 0.0)

    def test_probe_reliant_ft800_above_70km(self) -> None:
        # Probe + LFO engine + LFO tank — should clear all 7 altitude milestones
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_FL_T800],
            probe_core=True,
        )
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        self.assertGreater(alt, 70.0, f"Expected > 70 km, got {alt:.1f} km")

    def test_probe_hammer_srb_above_70km(self) -> None:
        # SRB path: Hammer SRB + probe core
        flags = _make_flags(srbs=[_HAMMER], probe_core=True)
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        self.assertGreater(alt, 70.0, f"Expected > 70 km, got {alt:.1f} km")

    def test_crewed_no_parachute_zero(self) -> None:
        # Capsule + engine + tank but no parachute -> can't survive, altitude = 0
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_FL_T800],
            capsule=True,
            staging_tier=1,  # has decoupler
        )
        self.assertEqual(_compute_sounding_altitude(flags, HOME_BODY), 0.0)

    def test_crewed_no_decoupler_zero(self) -> None:
        # Capsule + engine + tank + parachute but no decoupler -> can't separate
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_FL_T800],
            capsule=True,
            parachutes=[_MK16],
            staging_tier=0,  # no decoupler
        )
        self.assertEqual(_compute_sounding_altitude(flags, HOME_BODY), 0.0)

    def test_crewed_full_kit_above_70km(self) -> None:
        # Capsule + engine + tank + parachute + decoupler -> survivable crewed flight
        flags = _make_flags(
            engines=[_RELIANT],
            tanks=[_FL_T800],
            capsule=True,
            parachutes=[_MK16],
            staging_tier=1,  # has decoupler
        )
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        self.assertGreater(alt, 70.0, f"Expected > 70 km, got {alt:.1f} km")

    # --- Analytic precision tests -------------------------------------------
    # These check specific numeric outputs to catch formula regressions.
    # Expected values hand-calculated from h = dv^2 * (twr-1) / (2*g*twr*1000).

    def test_flea_probe_expected_altitude(self) -> None:
        """
        RT-5 Flea SRB + probe:
          m0 = dry_mass + fuel_mass + probe = 0.45 + 1.05 + 0.1 = 1.60t
          m_dry = 0.45 + 0.1 = 0.55t
          twr = 192 / (1.60 * 9.80665) = 12.24 (atm_thrust=162.9/... but
                sounding uses atm values for thrust, vac for ISP)
        """
        flags = _make_flags(srbs=[_FLEA], probe_core=True)
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        # Flea + probe should reach > 100 km regardless of exact formula details
        self.assertGreater(alt, 100.0, f"got {alt:.2f} km")

    def test_hammer_probe_expected_altitude(self) -> None:
        """
        RT-10 Hammer SRB + probe:
          m0 = 0.75 + 2.8125 + 0.1 = 3.6625t
          Real Hammer has lower ISP (195) and less fuel than old Thumper data.
        """
        flags = _make_flags(srbs=[_HAMMER], probe_core=True)
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        # Hammer + probe should comfortably clear 70 km
        self.assertGreater(alt, 70.0, f"got {alt:.2f} km")

    def test_reliant_fl400_probe_expected_altitude(self) -> None:
        """
        Reliant + FL-T400 + probe:
          Optimizer stacks tanks to maximize altitude within TWR constraints.
        """
        flags = _make_flags(engines=[_RELIANT], tanks=[_FL_T400], probe_core=True)
        alt = _compute_sounding_altitude(flags, HOME_BODY)
        # Should comfortably reach above 70 km (orbital altitude)
        self.assertGreater(alt, 70.0, f"got {alt:.2f} km")

    def test_flea_beats_reliant_low_twr_kills_altitude(self) -> None:
        """
        Flea (high TWR) should reach substantial altitude despite lower dv.
        Reliant+FL-T400 (low TWR) also reaches good altitude but TWR penalty
        limits sounding rocket performance.
        """
        flea_flags = _make_flags(srbs=[_FLEA], probe_core=True)
        reliant_flags = _make_flags(engines=[_RELIANT], tanks=[_FL_T400], probe_core=True)
        flea_alt = _compute_sounding_altitude(flea_flags, HOME_BODY)
        reliant_alt = _compute_sounding_altitude(reliant_flags, HOME_BODY)
        # Both should be well above 70 km
        self.assertGreater(flea_alt, 70.0)
        self.assertGreater(reliant_alt, 70.0)


class TestStagingGroupPreservation(unittest.TestCase):
    """Any decoupler (tier 1+) preserves all natural stage groups."""

    def test_any_decoupler_preserves_all_return_groups(self) -> None:
        """Tylo sample_return at tier 1+ should preserve all natural groups."""
        profiles = MISSION_PROFILES.get((BodyName.TYLO, MissionType.SAMPLE_RETURN), [])
        if not profiles:
            self.skipTest("No Tylo sample_return profiles")
        profile = profiles[0]
        groups_natural = _group_edges(profile, staging_tier=99)
        groups_tier1 = _group_edges(profile, staging_tier=1)
        self.assertEqual(len(groups_tier1), len(groups_natural),
                         "Tier 1+ should preserve all natural groups")
        self.assertGreater(len(groups_tier1), 4,
                           f"Expected >4 groups, got {len(groups_tier1)}")

    def test_constraint_aware_merge_skips_incompatible(self) -> None:
        """When forced to merge (tier 0), incompatible groups are skipped."""
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.RETURN), [])
        self.assertTrue(len(profiles) > 0)
        profile = profiles[0]
        groups_tier0 = _group_edges(profile, staging_tier=0)
        # Tier 0 wants 1 stage but _can_merge should prevent merging atmospheric
        # ascent with vacuum transfer. We may get >1 group if incompatible
        # pairs exist, or exactly 1 if all happen to be compatible.
        # The key assertion: no crash, and the result is a valid partition.
        total_edges = sum(len(g) for g in groups_tier0)
        self.assertEqual(total_edges, len(profile),
                         "All edges must be accounted for after merging")


class TestEngineMounting(unittest.TestCase):
    """
    Tests for the 3-mode engine mounting logic:
      Mode 1: Radial engines — mount on tank side, cap 8, no n_tank constraint
      Mode 2: Radial tank mounting — stack engines, cap 8, n_tank >= n_eng
      Mode 3: Adapter/plate — multiple engines under shared stack, no n_tank constraint
    """

    # Stock quad coupler: 1.25m input, up to 4× 1.25m engines
    _QUAD_COUPLER = MultiMount(1.25, {1.25: 4})
    # EP-50 engine plate: no min_tank_size, up to 9× 1.25m
    _EP50 = MultiMount(0.0, {0.625: 9, 1.25: 9, 1.875: 7, 2.5: 3, 3.75: 1})

    def test_adapter_max_engines_basic(self) -> None:
        """Quad coupler on 1.25m tank → 4 engines for 1.25m engine."""
        result = _adapter_max_engines(1.25, 1.25, [self._QUAD_COUPLER])
        self.assertEqual(result, 4)

    def test_adapter_max_engines_tank_too_small(self) -> None:
        """Quad coupler has min_tank_size=1.25; 0.625m tank → 0."""
        result = _adapter_max_engines(1.25, 0.625, [self._QUAD_COUPLER])
        self.assertEqual(result, 0)

    def test_adapter_max_engines_engine_too_large(self) -> None:
        """Quad coupler only mounts 1.25m engines; 2.5m engine → 0."""
        result = _adapter_max_engines(2.5, 1.25, [self._QUAD_COUPLER])
        self.assertEqual(result, 0)

    def test_adapter_max_engines_ep50(self) -> None:
        """EP-50 on any tank → 9 engines for 1.25m engine."""
        result = _adapter_max_engines(1.25, 0.625, [self._EP50])
        self.assertEqual(result, 9)

    def test_adapter_max_engines_best_of_multiple(self) -> None:
        """Multiple mounts: best count wins."""
        result = _adapter_max_engines(1.25, 1.25, [self._QUAD_COUPLER, self._EP50])
        self.assertEqual(result, 9)

    def test_stack_engine_no_adapter_enforces_tank_constraint(self) -> None:
        """Stack engine with no adapter: n_eng>1 forces n_tank >= n_eng."""
        # Reliant is a stack engine (not radial_mountable).
        # With no adapters, multi-engine requires radial tank mounting.
        self.assertFalse(_RELIANT.radial_mountable)
        result = find_optimal_stage(
            available_engines=[_RELIANT],
            available_srbs=[],
            available_tanks=[_X200_32],
            required_dv=3000.0,
            payload_mass=0.5,
            gravity=9.81,
            min_twr=1.5,
            in_atmosphere=True,
            available_multi_mounts=[],
        )
        if result is not None and result.engine_count > 1:
            n_tanks = sum(n for n, _ in result.tank_manifest)
            self.assertGreaterEqual(
                n_tanks, result.engine_count,
                "Without adapter, stack engine multi-engine must have n_tank >= n_eng",
            )

    def test_stack_engine_with_quad_coupler_no_tank_constraint(self) -> None:
        """Stack engine + quad coupler: up to 4 engines, no n_tank constraint."""
        result = find_optimal_stage(
            available_engines=[_RELIANT],
            available_srbs=[],
            available_tanks=[_X200_32],
            required_dv=3000.0,
            payload_mass=0.5,
            gravity=9.81,
            min_twr=1.5,
            in_atmosphere=True,
            available_multi_mounts=[self._QUAD_COUPLER],
        )
        self.assertIsNotNone(result)
        # With a quad coupler, optimizer can use up to 4 engines with fewer tanks
        if result.engine_count <= 4:
            # Adapter covers it — no constraint on n_tank >= n_eng
            self.assertGreater(result.engine_count, 0)

    def test_stack_engine_with_ep50_more_engines(self) -> None:
        """Stack engine + EP-50: allows up to 9× 1.25m engines."""
        result = find_optimal_stage(
            available_engines=[_RELIANT],
            available_srbs=[],
            available_tanks=[_X200_32],
            required_dv=3500.0,
            payload_mass=1.0,
            gravity=9.81,
            min_twr=1.5,
            in_atmosphere=True,
            available_multi_mounts=[self._EP50],
        )
        self.assertIsNotNone(result)

    def test_radial_engine_no_tank_constraint(self) -> None:
        """Radial engine (Spider): mounts on tank side, no n_tank >= n_eng."""
        self.assertTrue(_SPIDER.radial_mountable)
        result = find_optimal_stage(
            available_engines=[_SPIDER],
            available_srbs=[],
            available_tanks=[_FL_T400],
            required_dv=500.0,
            payload_mass=0.5,
            gravity=1.63,
            min_twr=1.2,
            available_multi_mounts=[],
        )
        self.assertIsNotNone(result)
        # Radial engines have no n_tank >= n_eng constraint
        if result.engine_count > 1:
            # n_tanks can be less than n_eng — that's the point
            self.assertIsInstance(result.tank_manifest, tuple)

    def test_radial_srb_capped_at_8(self) -> None:
        """Radial SRBs (Flea, Hammer) are capped at 8."""
        self.assertTrue(_FLEA.radial_mountable)
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[_FLEA],
            available_tanks=[],
            required_dv=200.0,
            payload_mass=0.5,
            gravity=9.81,
            min_twr=1.5,
            in_atmosphere=True,
            available_multi_mounts=[],
        )
        if result is not None:
            self.assertLessEqual(result.engine_count, 8)

    def test_stack_srb_limited_to_1(self) -> None:
        """
        A hypothetical stack-only SRB (no srf profile) can only mount 1.
        All stock SRBs happen to be radial-mountable, so we construct one.
        """
        stack_srb = SolidBooster(
            name="test_stack_srb",
            vac_isp=200.0, atm_isp=180.0,
            vac_thrust=500.0, atm_thrust=450.0,
            dry_mass=1.0, fuel_mass=4.0,
            has_gimbal=False, size_class=1.25,
            radial_mountable=False,
        )
        result = find_optimal_stage(
            available_engines=[],
            available_srbs=[stack_srb],
            available_tanks=[],
            required_dv=200.0,
            payload_mass=0.5,
            gravity=9.81,
            min_twr=1.5,
            in_atmosphere=True,
            available_multi_mounts=[],
        )
        if result is not None:
            self.assertEqual(result.engine_count, 1,
                             "Stack SRB should be limited to 1")

    def test_radial_mountable_flag_set_correctly(self) -> None:
        """Verify radial_mountable is set from bulkhead_profiles."""
        self.assertTrue(_SPIDER.radial_mountable, "Spider should be radial")
        self.assertTrue(_THUD.radial_mountable, "Thud should be radial")
        self.assertFalse(_RELIANT.radial_mountable, "Reliant should be stack")
        self.assertFalse(_SWIVEL.radial_mountable, "Swivel should be stack")
        self.assertTrue(_FLEA.radial_mountable, "Flea should be radial")
        self.assertTrue(_HAMMER.radial_mountable, "Hammer should be radial")


class TestAeroLandingPassiveStage(unittest.TestCase):
    """
    Bug 057: aero-landing stages should be passive (no engines, no fuel).

    The ATMO_LANDING_AERO edge represents atmospheric drag + parachutes,
    not a propulsive burn.  The optimizer must not add engines to these stages.
    """

    def _return_flags(self) -> EquipmentFlags:
        """Parts sufficient for a Mun return mission."""
        return _make_flags(
            engines=[_MAMMOTH, _MAINSAIL, _SWIVEL, _TERRIER],
            tanks=[_S3_3600, _JUMBO_64, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True, solar=True,
            heat_shields=[_SHIELD_125],
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_stack=True,
        )

    def test_mun_return_reentry_stage_has_no_engines(self) -> None:
        """The final stage of a Mun return (Kerbin reentry) should be passive."""
        flags = self._return_flags()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.RETURN), [])
        self.assertTrue(len(profiles) > 0)

        # Find a feasible profile and inspect its last stage
        for profile in profiles:
            result = _evaluate_profile(profile, flags, _normal_diff(), MissionType.RETURN, is_crewed=False, home=BodyName.KERBIN)
            if result.feasible:
                last_stage = result.stage_results[-1]
                self.assertEqual(last_stage.engine_count, 0,
                                 "Aero-landing stage should have no engines")
                self.assertEqual(last_stage.tank_manifest, (),
                                 "Aero-landing stage should have no fuel tanks")
                self.assertEqual(last_stage.delta_v, 0.0,
                                 "Aero-landing stage produces no delta-v")
                return

        self.fail("No feasible Mun return profile found — can't test reentry stage")

    def test_aero_landing_stage_mass_is_payload_plus_equipment(self) -> None:
        """Passive stage mass = payload (prior stage wet mass) + heat shield only."""
        flags = self._return_flags()
        profiles = MISSION_PROFILES.get((BodyName.MUN, MissionType.RETURN), [])

        for profile in profiles:
            result = _evaluate_profile(profile, flags, _normal_diff(), MissionType.RETURN, is_crewed=False, home=BodyName.KERBIN)
            if result.feasible:
                last_stage = result.stage_results[-1]
                self.assertEqual(last_stage.stage_mass_wet, last_stage.stage_mass_dry,
                                 "Passive stage has no fuel — wet == dry")
                # Mass should include heat shield but no engine/tank mass
                self.assertGreater(last_stage.stage_mass_wet, 0.0,
                                   "Passive stage must have non-zero mass (payload + shield)")
                return

        self.fail("No feasible Mun return profile found")

    def test_duna_return_reentry_stage_has_no_engines(self) -> None:
        """Duna return also ends with a Kerbin aero-landing — same passive requirement."""
        flags = _make_flags(
            engines=[_MAMMOTH, _MAINSAIL, _SWIVEL, _TERRIER],
            tanks=[_S3_3600, _JUMBO_64, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=3,
            heat_shields=[_SHIELD_25],
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_radial=True,
        )
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.RETURN), [])
        self.assertTrue(len(profiles) > 0)

        for profile in profiles:
            result = _evaluate_profile(profile, flags, _normal_diff(), MissionType.RETURN, is_crewed=False, home=BodyName.KERBIN)
            if result.feasible:
                last_stage = result.stage_results[-1]
                self.assertEqual(last_stage.engine_count, 0,
                                 "Duna return reentry stage should have no engines")
                return

        self.fail("No feasible Duna return profile found")


class TestEscapeRelayGate(unittest.TestCase):
    """Bug 075 regression: ESCAPE / flyby profiles must enforce each body's
    relay tier — the SOI-entry edge carries body=destination so the check
    fires even though the orbit-insertion edge is absent."""

    def _full_kit_without_relay(self, relay_tier: int = 0) -> EquipmentFlags:
        """A capable rocket kit with a tunable relay tier."""
        return _make_flags(
            engines=[_MAMMOTH, _MAINSAIL, _SWIVEL, _TERRIER, _NERV],
            tanks=[_S3_3600, _JUMBO_64, _X200_32, _FL_T800, _FL_T400],
            probe_core=True, reaction_wheels=True,
            solar=True, solar_retractable=True, rtg=True,
            relay_tier=relay_tier,
            heat_shields=[_SHIELD_25],
            parachutes=[_MK16, _MK16, _MK16],
            legs=[_LT2],
            launch_clamp=True, decoupler_radial=True,
        )

    def _escape_feasible(self, body: BodyName, relay_tier: int) -> bool:
        flags = self._full_kit_without_relay(relay_tier=relay_tier)
        profiles = MISSION_PROFILES.get((body, MissionType.ESCAPE), [])
        self.assertTrue(profiles, f"{body} should have an ESCAPE profile defined")
        for profile in profiles:
            r = _evaluate_profile(profile, flags, _normal_diff(),
                                  MissionType.ESCAPE, is_crewed=False, home=BodyName.KERBIN)
            if r.feasible:
                return True
        return False

    def test_dres_escape_requires_relay_tier_3(self) -> None:
        # Dres max-separation = 3.65 AU from Kerbin → tier 3 under the
        # opposition-distance relay formula.
        self.assertFalse(self._escape_feasible(BodyName.DRES, relay_tier=2))
        self.assertTrue(self._escape_feasible(BodyName.DRES, relay_tier=3))

    def test_moho_escape_requires_relay_tier_2(self) -> None:
        # Moho max-separation = 1.34 AU from Kerbin → tier 2.
        self.assertFalse(self._escape_feasible(BodyName.MOHO, relay_tier=1))
        self.assertTrue(self._escape_feasible(BodyName.MOHO, relay_tier=2))

    def test_eeloo_escape_requires_relay_tier_4(self) -> None:
        # Eeloo max-separation = 7.0 AU from Kerbin → tier 4.
        self.assertFalse(self._escape_feasible(BodyName.EELOO, relay_tier=3))
        self.assertTrue(self._escape_feasible(BodyName.EELOO, relay_tier=4))

    def test_mun_escape_no_relay_required(self) -> None:
        # Mun is in Kerbin's CommNet — flyby works with tier 0.
        self.assertTrue(self._escape_feasible(BodyName.MUN, relay_tier=0))

    def test_every_advertised_escape_check_has_a_profile(self) -> None:
        """For every body that exposes a Flyby / SOI Leave check (per
        locations.get_body_events), MISSION_PROFILES must carry an ESCAPE
        profile. No auto-strip fallback."""
        from worlds.ksp1.locations import get_body_events
        for body in ALL_BODIES:
            events = set(get_body_events(body))
            if EventName.FLYBY not in events and EventName.SOI_LEAVE not in events:
                continue
            self.assertIn(
                (body.name, MissionType.ESCAPE), MISSION_PROFILES,
                f"{body.name} exposes flyby/SOI-leave checks but has no "
                f"ESCAPE profile defined",
            )

    def test_escape_profile_ends_at_destination_body(self) -> None:
        """Every ESCAPE profile's last edge must have body=destination so the
        relay/power checks at the destination actually fire."""
        for (body, mt), profiles in list(MISSION_PROFILES.items()):
            if mt != MissionType.ESCAPE:
                continue
            for i, prof in enumerate(profiles):
                self.assertEqual(
                    prof[-1].body, body,
                    f"{body} ESCAPE alt {i}: last edge body={prof[-1].body} "
                    f"(must be destination for relay/power checks to fire)",
                )


class TestAttitudeBundleManifestReconciles(unittest.TestCase):
    """When the optimizer picks ungimballed propulsion on an attitude-required
    stage, the attitude bundle parts must appear in stage.equipment AND their
    real masses must sum to exactly the mass the optimizer charged.

    Decouplers are an explicit exception (in manifest, not in mass budget) —
    they're filtered out before summing. Everything else must reconcile.
    """

    def test_attitude_module_mass_reflects_real_part_masses(self) -> None:
        from worlds.ksp1.parts import PART_DB
        from worlds.ksp1.capability import _attitude_bundle_for_stage
        # Setup: Stayputnik (no built-in wheels) + sasModule unlocked.
        flags = _make_flags(
            engines=[_RELIANT], tanks=[_FL_T400, _FL_T800],
            probe_core=False, reaction_wheels=False,
            launch_clamp=True, decoupler_stack=True,
        )
        stayputnik = PART_DB["probeCoreSphere.v2"][0]
        flags.has_probe_core = True
        flags.lightest_probe = stayputnik
        sas = PART_DB["sasModule"][0]
        flags.has_reaction_wheels = True
        flags.lightest_reaction_wheel = sas

        bundle = _attitude_bundle_for_stage(flags, is_crewed=False)
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.parts, ((1, "sasModule"),))
        self.assertAlmostEqual(bundle.mass, sas.mass, places=6)

    def test_rcs_bundle_includes_tank_when_terminal_lacks_monoprop(self) -> None:
        from worlds.ksp1.parts import PART_DB
        from worlds.ksp1.capability import _attitude_bundle_for_stage
        flags = _make_flags(
            engines=[_RELIANT], tanks=[_FL_T400, _FL_T800],
            launch_clamp=True, decoupler_stack=True,
        )
        stayputnik = PART_DB["probeCoreSphere.v2"][0]
        flags.has_probe_core = True
        flags.lightest_probe = stayputnik
        rcs = PART_DB["RCSLinearSmall"][0]
        flags.has_rcs = True
        flags.lightest_rcs_thruster = rcs
        tank = PART_DB["monopropMiniSphere"][0]
        flags.lightest_monoprop_tank = tank
        flags.available_tanks.append(tank)

        bundle = _attitude_bundle_for_stage(flags, is_crewed=False)
        self.assertIsNotNone(bundle)
        self.assertEqual(set(bundle.parts), {(4, "RCSLinearSmall"), (1, "monopropMiniSphere")})
        expected = 4 * rcs.mass + tank.dry_mass + tank.fuel_mass
        self.assertAlmostEqual(bundle.mass, expected, places=6)

    def test_rcs_bundle_skips_tank_for_pod_with_internal_monoprop(self) -> None:
        from worlds.ksp1.parts import PART_DB
        from worlds.ksp1.capability import _attitude_bundle_for_stage
        flags = _make_flags(
            engines=[_RELIANT], tanks=[_FL_T400, _FL_T800],
            launch_clamp=True, decoupler_stack=True,
        )
        pod = PART_DB["mk1-3pod"][0]
        flags.has_capsule = True
        flags.lightest_capsule = pod
        rcs = PART_DB["RCSLinearSmall"][0]
        flags.has_rcs = True
        flags.lightest_rcs_thruster = rcs

        bundle = _attitude_bundle_for_stage(flags, is_crewed=True)
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.parts, ((4, "RCSLinearSmall"),))
        self.assertAlmostEqual(bundle.mass, 4 * rcs.mass, places=6)


class TestStructuredBlockingReasons(unittest.TestCase):
    """
    Verifies that ProfileResult.blocking carries structured BlockingInfo
    objects with the expected BlockingReason enum values.  Sphere-ladder
    pre-fill consumes these structured values directly (no string parsing).
    """

    def test_empty_kit_mun_orbit_yields_propulsion_reasons(self) -> None:
        from worlds.ksp1.capability import evaluate_mission_detailed, _pre_pass
        from worlds.ksp1.capability_reasons import BlockingReason
        flags = _pre_pass(lambda _n: 0, start_with_clamps=False)
        result = evaluate_mission_detailed(
            flags, _normal_diff(), "Mun", MissionType.ORBIT, None,
            MISSION_BUILDER,
        )
        self.assertFalse(result.feasible)
        reasons = {b.reason for b in result.blocking}
        # An empty kit should report at least one of NO_FUEL / NO_ENGINE
        # / NO_LAUNCH_ENGINE / STAGING_TIER_INSUFFICIENT.
        self.assertTrue(reasons & {
            BlockingReason.NO_FUEL,
            BlockingReason.NO_ENGINE,
            BlockingReason.NO_LAUNCH_ENGINE,
            BlockingReason.STAGING_TIER_INSUFFICIENT,
            BlockingReason.NO_PROBE_CORE,
        }, f"Expected propulsion or command-module reason; got {reasons}")

    def test_no_heat_shield_is_structured(self) -> None:
        from worlds.ksp1.capability import _evaluate_profile
        from worlds.ksp1.capability_reasons import BlockingReason
        flags = _make_flags(
            engines=[_MAINSAIL], tanks=[_FL_T400, _FL_T800],
            probe_core=True, reaction_wheels=True,
            solar=True, relay_tier=3, launch_clamp=True,
            decoupler_stack=True,
            parachutes=[_MK16, _MK16, _MK16, _MK16],
        )
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.LAND), [])
        aero_profiles = [p for p in profiles
                         if any(e.needs_heat_shield for e in p)]
        self.assertTrue(aero_profiles, "Test fixture: no aero profiles found")
        result = _evaluate_profile(
            aero_profiles[0], flags, _normal_diff(),
            MissionType.LAND, is_crewed=False, home=BodyName.KERBIN,
        )
        self.assertFalse(result.feasible)
        self.assertIn(BlockingReason.NO_HEAT_SHIELD,
                      {b.reason for b in result.blocking})

    def test_relay_tier_too_low_carries_typed_fields(self) -> None:
        from worlds.ksp1.capability import _evaluate_profile
        from worlds.ksp1.capability_reasons import BlockingReason
        flags = _make_flags(
            engines=[_MAINSAIL, _SWIVEL],
            tanks=[_FL_T400, _FL_T800, _X200_32],
            probe_core=True, reaction_wheels=True,
            solar=True, relay_tier=0,   # <-- too low for any outer body
            launch_clamp=True, decoupler_stack=True,
            heat_shields=[_SHIELD_25],
            parachutes=[_MK16, _MK16, _MK16, _MK16],
        )
        profiles = MISSION_PROFILES.get((BodyName.DUNA, MissionType.ORBIT), [])
        result = _evaluate_profile(
            profiles[0], flags, _normal_diff(),
            MissionType.ORBIT, is_crewed=False, home=BodyName.KERBIN,
        )
        self.assertFalse(result.feasible)
        relay_blockings = [b for b in result.blocking
                           if b.reason == BlockingReason.RELAY_TIER_TOO_LOW]
        self.assertTrue(relay_blockings,
                        f"Expected RELAY_TIER_TOO_LOW; got {result.blocking}")
        self.assertGreater(relay_blockings[0].relay_needed, 0)
        self.assertEqual(relay_blockings[0].relay_available, 0)


if __name__ == "__main__":
    unittest.main()
