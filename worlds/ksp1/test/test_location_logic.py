"""
Gold-standard location logic integration tests.

Two testing strategies:

  Real-items tests (TestKerbinEarlyLocations):
    Collects actual items and verifies location accessibility.
    Progressive items are used because individual parts absorbed into
    progressive chains can only be unlocked through their progressive tier
    mechanism — collecting the bare part name does nothing if the part's
    progressive parent hasn't been collected.

  Mocked-capability tests (TestKerbinReturnVsMunReturn, TestFlybyPerBodyWiring,
    TestCrewedEventsRequireBodyProfile, TestTechTreeBandGating):
    Patches worlds.ksp1.rules.get_capability to control exactly which
    bodies/events/science are accessible, without depending on rocket-math
    physics.  Directly catches the class of bug where a rule closure
    captures the wrong body_name.

Key item names for real-item tests:
  "Progressive Capsule"       → has_capsule = True
  "Progressive SRB"           → SRB available for sounding altitude calc
  "Progressive Probe Core"    → lightest_probe set (required for sounding)
  "parachuteSingle"           → has_parachutes (Mk16 Parachute, not in progressive chain)
  "Progressive Launch Engine" → has_throttleable_engine (tier-1 includes Swivel)
"""
import unittest
from unittest.mock import patch, MagicMock

from worlds.ksp1.locations import (
    MISSION_LOCATION_NAMES,
    LocationBuilder,
    KSC_BIOME_NAMES,
    EventName,
    MissionType,
)
from worlds.ksp1.bodies import ALL_BODIES, BodyName
from worlds.ksp1.items import PROGRESSIVE_RD_NAME
from worlds.ksp1.tech_tree import TECH_NODES, TIER_TO_BAND
from worlds.ksp1.test.base import KSP1TestBase as _SharedKSP1TestBase


class KSP1TestBase(_SharedKSP1TestBase):
    options = {"difficulty": 1}  # normal difficulty for determinism


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_body_cap(events) -> MagicMock:
    """Return a mock BodyAccessProfile with the given events set to True.

    All EventName values are present; unlisted ones default to False.
    """
    bp = MagicMock()
    bp.access = {e: (e in events) for e in EventName}
    return bp


def _all_false_bodies() -> dict:
    """Return a bodies dict with every body present and all events False."""
    return {body.name: _make_body_cap(set()) for body in ALL_BODIES}


def _make_zero_cap() -> MagicMock:
    """Return a RocketCapability mock with nothing accessible.

    All boolean flags False, sounding = 0, every body present but all events False.
    """
    cap = MagicMock()
    cap.bodies = _all_false_bodies()
    cap.has_capsule = False
    cap.has_parachutes = False
    cap.has_throttleable_engine = False
    cap.has_probe_core = False
    cap.has_wheel = False
    cap.power_profile = "none"
    cap.has_thermometer = False
    cap.has_barometer = False
    cap.staging_tier = 0
    cap.sounding_altitude_km = 0.0
    cap.relay_tier = 0
    return cap


# ---------------------------------------------------------------------------
# Class 1: Static structure assertions (no world instantiation needed)
# ---------------------------------------------------------------------------

class TestLocationSetStructure(unittest.TestCase):
    """Static assertions on locations.py data — catches wiring mistakes at import time."""

    def test_kerbin_sample_return_is_a_location(self):
        """Kerbin sample return must be in MISSION_LOCATION_NAMES.

        bodies.py registers it with an empty edge list (no rocket needed —
        launchpad EVA + surface sample + vessel recovery).  The capability system
        will always report it as achievable.  bodies.py is the source of truth;
        no special-casing elsewhere.
        """
        self.assertIn(
            "Kerbin Sample Return 1",
            MISSION_LOCATION_NAMES,
            "Kerbin Sample Return 1 must be in MISSION_LOCATION_NAMES "
            "(bodies.py defines it with an empty/always-achievable profile).",
        )

    def test_kerbin_return_is_a_location(self):
        """Kerbin Return is handled by the per-body mission system."""
        self.assertIn(
            "Kerbin Return 1",
            MISSION_LOCATION_NAMES,
            "Kerbin Return 1 must be in MISSION_LOCATION_NAMES.",
        )

    def test_kerbin_special_section_has_no_return_events(self):
        """Home location sets must not contain RETURN or SAMPLE_RETURN mission types.

        Those event types go through the per-body system (MissionLocation),
        not the home-body specials.
        """
        for loc in LocationBuilder(BodyName.KERBIN).locations:
            self.assertNotIn(
                loc.mission_type,
                (MissionType.RETURN, MissionType.SAMPLE_RETURN),
                f"Home location entry {loc.name!r} has mission_type "
                f"{loc.mission_type!r}; return events belong in the per-body system.",
            )

    def test_all_bodies_have_flyby_locations(self):
        """Every body except Kerbol must have a '<body> Flyby 1' location.

        Catches a silent gap if a body is accidentally skipped in _set_mission_rules.
        Kerbol is excluded — root body has no meaningful flyby/escape.
        """
        for body in ALL_BODIES:
            if body.name == BodyName.KERBOL:
                continue
            name = f"{body.name} Flyby 1"
            self.assertIn(
                name,
                MISSION_LOCATION_NAMES,
                f"Missing location '{name}' in MISSION_LOCATION_NAMES.",
            )

    def test_kerbol_has_no_locations(self):
        """Kerbol (root body) must not have any mission locations."""
        for name in MISSION_LOCATION_NAMES:
            self.assertFalse(
                name.startswith("Kerbol "),
                f"Unexpected Kerbol location: '{name}'",
            )

    def test_flyby_and_orbit_both_exist_per_body(self):
        """Every body except Kerbol must have both Flyby and Orbit locations.

        Both are generated by the same factory with different event keys;
        neither should be accidentally dropped.
        """
        for body in ALL_BODIES:
            if body.name == BodyName.KERBOL:
                continue
            for event in (EventName.FLYBY, EventName.ORBIT):
                name = f"{body.name} {event} 1"
                self.assertIn(
                    name,
                    MISSION_LOCATION_NAMES,
                    f"Missing location '{name}' in MISSION_LOCATION_NAMES.",
                )

    def test_crewed_events_only_on_landable_bodies(self):
        """FLAG_PLANT, SAMPLE_RETURN, CREWED_LANDING must only exist for landable bodies."""
        landable = {b.name for b in ALL_BODIES if b.can_land}
        for name in MISSION_LOCATION_NAMES:
            for event in (EventName.FLAG_PLANT, EventName.SAMPLE_RETURN,
                          EventName.CREWED_LANDING):
                marker = f" {event} "
                if marker in name:
                    body_name = name.split(marker)[0]
                    self.assertIn(
                        body_name, landable,
                        f"Location '{name}' has event '{event}' but "
                        f"body '{body_name}' is not landable.",
                    )

    def test_kerbin_special_location_count(self):
        """Kerbin home location set must have exactly 12 entries."""
        kerbin = LocationBuilder(BodyName.KERBIN).locations
        self.assertEqual(
            len(kerbin), 12,
            f"Expected 12 Kerbin home locations, got {len(kerbin)}.",
        )


# ---------------------------------------------------------------------------
# Class 2: Kerbin early locations — real items
# ---------------------------------------------------------------------------

class TestKerbinEarlyLocations(KSP1TestBase):
    """Kerbin special locations tested with real items.

    Progressive items used because individual parts in progressive chains are
    only unlocked through their tier mechanism — collecting 'Mk1 Command Pod'
    does nothing if 'Progressive Capsule' hasn't been collected first.
    """

    def test_nothing_accessible_without_items(self):
        """Fresh state: Kerbin mission locations and KSC biomes are not reachable."""
        self.assertFalse(self.can_reach_location("Kerbin First Launch"),
                         "First Launch should not be accessible without items")
        self.assertFalse(self.can_reach_location("Kerbin First Landing"),
                         "First Landing should not be accessible without items")
        self.assertFalse(self.can_reach_location("Kerbin First Crash"),
                         "First Crash should not be accessible without items")
        for biome in KSC_BIOME_NAMES:
            self.assertFalse(
                self.can_reach_location(biome),
                f"KSC biome '{biome}' should not be accessible without items",
            )

    def test_capsule_enables_first_launch_and_landing(self):
        """Progressive Capsule sets has_capsule; the capsule path enables First Launch and First Landing."""
        self.collect_by_name("Progressive Capsule")
        self.assertTrue(self.can_reach_location("Kerbin First Launch"),
                        "First Launch: has_capsule path must pass")
        self.assertTrue(self.can_reach_location("Kerbin First Landing"),
                        "First Landing: has_capsule path must pass")
        # Capsule alone gives sounding = 0 (no propulsion) — altitude check fails
        self.assertFalse(self.can_reach_location("Kerbin First Crash"),
                         "First Crash needs sounding ≥ 0.1 km; capsule alone gives sounding = 0")

    def test_capsule_enables_all_ksc_biomes(self):
        """has_capsule → KSC EVA science → all 12 KSC biome locations accessible."""
        self.collect_by_name("Progressive Capsule")
        for biome in KSC_BIOME_NAMES:
            self.assertTrue(
                self.can_reach_location(biome),
                f"KSC biome '{biome}' should be accessible with a capsule",
            )

    def test_srb_with_probe_enables_first_launch_and_crash(self):
        """SRB + probe core: sounding_altitude_km > 0 → First Launch YES, First Crash YES.

        An SRB alone cannot produce sounding altitude — the sounding model
        requires a payload (probe core or capsule + chute + decoupler) to compute.
        """
        self.collect_by_name("Progressive SRB")
        self.collect_by_name("Progressive Probe Core")
        self.assertTrue(
            self.can_reach_location("Kerbin First Launch"),
            "First Launch: sounding > 0 path must pass with SRB + probe",
        )
        self.assertTrue(
            self.can_reach_location("Kerbin First Crash"),
            "First Crash: sounding ≥ 0.1 km must pass with SRB + probe (tier-1 SRBs reach km-scale altitude)",
        )

    def test_srb_with_probe_no_landing_without_descent(self):
        """SRB + probe: sounding > 0 but no safe descent → First Landing NO, KSC biomes NO."""
        self.collect_by_name("Progressive SRB")
        self.collect_by_name("Progressive Probe Core")
        self.assertFalse(
            self.can_reach_location("Kerbin First Landing"),
            "First Landing requires parachute or throttleable engine; SRB + probe has neither",
        )
        for biome in KSC_BIOME_NAMES:
            self.assertFalse(
                self.can_reach_location(biome),
                f"KSC biome '{biome}' requires capsule or rover; SRB + probe provides neither",
            )

    def test_srb_plus_parachute_enables_first_landing(self):
        """SRB + probe + parachute: sounding > 0 AND has_parachutes → First Landing YES."""
        self.collect_by_name("Progressive SRB")
        self.collect_by_name("Progressive Probe Core")
        # Progressive Parachute tier 1 unlocks parachuteSingle/parachuteLarge
        # (the rep is auto-granted; we don't need to also collect the part name).
        self.collect_by_name("Progressive Parachute")
        self.assertTrue(
            self.can_reach_location("Kerbin First Landing"),
            "First Landing: sounding > 0 + parachutes must pass",
        )

    def test_srb_plus_throttleable_engine_enables_landing(self):
        """SRB + probe + throttleable engine: sounding > 0 AND has_throttleable_engine → First Landing YES."""
        self.collect_by_name("Progressive SRB")
        self.collect_by_name("Progressive Probe Core")
        # Progressive Launch Engine tier-1 includes LV-T45 Swivel (throttleable)
        self.collect_by_name("Progressive Launch Engine")
        self.assertTrue(
            self.can_reach_location("Kerbin First Landing"),
            "First Landing: sounding > 0 + throttleable engine must pass",
        )

    def test_splashdown_needs_altitude_and_descent_control(self):
        """Capsule alone (sounding = 0): Splashdown rule requires sounding ≥ 1.0 km."""
        self.collect_by_name("Progressive Capsule")
        self.assertFalse(
            self.can_reach_location("Kerbin Splashdown"),
            "Splashdown needs sounding ≥ 1 km; capsule alone gives sounding = 0",
        )

    def test_splashdown_accessible_with_srb_probe_parachute(self):
        """SRB + probe + parachute: sounding ≥ 1 km AND has_parachutes → Splashdown YES."""
        self.collect_by_name("Progressive SRB")
        self.collect_by_name("Progressive Probe Core")
        self.collect_by_name("Progressive Parachute")
        self.assertTrue(
            self.can_reach_location("Kerbin Splashdown"),
            "Splashdown: sounding ≥ 1 km + parachutes must pass (tier-1 SRBs easily reach 1 km)",
        )

    def test_capsule_enables_kerbin_eva_missions(self):
        """Progressive Capsule alone → all capsule-only Kerbin locations reachable.

        Kerbin Sample Return and Flag Plant have empty MISSION_PROFILES (no rocket
        needed — launchpad EVA). These must not be blocked by body-level gates
        that duplicate per-edge checks.
        """
        self.collect_by_name("Progressive Capsule")

        # Per-body mission events with empty profiles (always achievable with capsule)
        for loc in (
            "Kerbin Sample Return 1", "Kerbin Sample Return 2", "Kerbin Sample Return 3",
            "Kerbin Flag Plant 1", "Kerbin Flag Plant 2",
        ):
            self.assertTrue(
                self.can_reach_location(loc),
                f"'{loc}' must be reachable with just a capsule (empty profile = launchpad EVA)",
            )

        # Kerbin-specific locations that only need has_capsule
        self.assertTrue(self.can_reach_location("Kerbin First Launch"))
        self.assertTrue(self.can_reach_location("Kerbin First Landing"))

        # All 12 KSC biomes (already tested separately but included for completeness)
        for biome in KSC_BIOME_NAMES:
            self.assertTrue(self.can_reach_location(biome))

    def test_probe_core_does_not_enable_crewed_kerbin_missions(self):
        """Probe core alone must NOT unlock crewed Kerbin missions.

        Sample Return, Flag Plant, First Launch/Landing, and KSC biomes all
        require a capsule (crewed EVA). A probe core is not a substitute.
        """
        self.collect_by_name("Progressive Probe Core")

        for loc in (
            "Kerbin Sample Return 1", "Kerbin Sample Return 2", "Kerbin Sample Return 3",
            "Kerbin Flag Plant 1", "Kerbin Flag Plant 2",
            "Kerbin First Launch", "Kerbin First Landing",
        ):
            self.assertFalse(
                self.can_reach_location(loc),
                f"'{loc}' must NOT be reachable with just a probe core (crewed-only)",
            )
        for biome in KSC_BIOME_NAMES:
            self.assertFalse(
                self.can_reach_location(biome),
                f"KSC biome '{biome}' must NOT be reachable with just a probe core",
            )

    def test_fuel_tank_does_not_enable_crewed_kerbin_missions(self):
        """A fuel tank alone (no command module) must NOT unlock anything.

        Neither capsule nor probe core → no command authority at all.
        """
        self.collect_by_name("Progressive LFO Tank")

        for loc in (
            "Kerbin Sample Return 1", "Kerbin Sample Return 2", "Kerbin Sample Return 3",
            "Kerbin Flag Plant 1", "Kerbin Flag Plant 2",
            "Kerbin First Launch", "Kerbin First Landing",
        ):
            self.assertFalse(
                self.can_reach_location(loc),
                f"'{loc}' must NOT be reachable with just a fuel tank",
            )
        for biome in KSC_BIOME_NAMES:
            self.assertFalse(
                self.can_reach_location(biome),
                f"KSC biome '{biome}' must NOT be reachable with just a fuel tank",
            )

    def test_altitude_checks_gate_with_sounding(self):
        """Altitude check locations require strictly increasing sounding thresholds.

        Uses mocked capability to avoid depending on the SRB physics model.
        Altitudes are read from the Kerbin home location set so this stays
        correct if the milestone schedule shifts (Phase 3a moved from
        5/15/25/…/70 to 5/10/16/20/30/45/70).
        """
        altitudes = sorted(
            int(loc.threshold_km)
            for loc in LocationBuilder(BodyName.KERBIN).locations
            if loc.mission_type == MissionType.SOUNDING
            and loc.threshold_km is not None
            and loc.threshold_km >= 1.0  # exclude First Crash (0.1 km)
        )

        # sounding = 10 km: only the lowest milestones pass
        cap = _make_zero_cap()
        cap.sounding_altitude_km = 10.0
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            for km in altitudes:
                name = f"Kerbin {km}km Altitude"
                if km <= 10:
                    self.assertTrue(self.can_reach_location(name),
                                    f"{km} km check must pass with 10 km sounding")
                else:
                    self.assertFalse(self.can_reach_location(name),
                                     f"{km} km check must fail with 10 km sounding")

        # sounding = 50 km: everything below 50 km passes, top tier fails
        cap2 = _make_zero_cap()
        cap2.sounding_altitude_km = 50.0
        with patch("worlds.ksp1.rules.get_capability", return_value=cap2):
            for km in altitudes:
                name = f"Kerbin {km}km Altitude"
                if km <= 50:
                    self.assertTrue(self.can_reach_location(name),
                                    f"{km} km check must pass with 50 km sounding")
                else:
                    self.assertFalse(self.can_reach_location(name),
                                     f"{km} km check must fail with 50 km sounding")


# ---------------------------------------------------------------------------
# Class 3: Kerbin Return vs Mun Return — per-body closure identity
# ---------------------------------------------------------------------------

class TestKerbinReturnVsMunReturn(KSP1TestBase):
    """Each body's Return rule must capture its own body_name in the closure.

    Bug scenario: if _mission_rule_for_event reuses the same body_name variable
    (e.g. via a late-binding loop variable), all rules end up querying the same
    body's profile.  These three tests together prove the closures are distinct:
    if the Mun rule accidentally uses "Kerbin" as the key, test 2 would pass
    (Kerbin Return accessible) when it should fail.
    """

    def test_kerbin_return_accessible_when_only_kerbin_in_profile(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.KERBIN] = _make_body_cap({EventName.RETURN})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Kerbin Return 1"))
            self.assertFalse(
                self.can_reach_location("Mun Return 1"),
                "Mun Return 1 must be False when only Kerbin is in the profile — "
                "if True, the Mun closure is querying the Kerbin profile.",
            )

    def test_mun_return_accessible_when_only_mun_in_profile(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.RETURN})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun Return 1"))
            self.assertFalse(
                self.can_reach_location("Kerbin Return 1"),
                "Kerbin Return 1 must be False when only Mun is in the profile.",
            )

    def test_neither_reachable_when_no_body_profiles(self):
        cap = _make_zero_cap()
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertFalse(self.can_reach_location("Kerbin Return 1"))
            self.assertFalse(self.can_reach_location("Mun Return 1"))


# ---------------------------------------------------------------------------
# Class 4: Per-body flyby closure correctness
# ---------------------------------------------------------------------------

class TestFlybyPerBodyWiring(KSP1TestBase):
    """Regression tests for the cross-body flyby closure bug.

    The bug: _mission_rule_for_event's closure captures the loop variable
    by reference rather than by value, so all flyby rules end up querying
    the last body's profile (typically whichever body is last in ALL_BODIES).
    """

    def test_mun_flyby_uses_mun_profile(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.FLYBY})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun Flyby 1"))
            self.assertFalse(
                self.can_reach_location("Kerbin Flyby 1"),
                "Kerbin Flyby 1 must be False when only Mun has Flyby — "
                "if True, the Mun rule is querying the Kerbin profile.",
            )
            self.assertFalse(self.can_reach_location("Duna Flyby 1"))

    def test_kerbin_flyby_uses_kerbin_profile(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.KERBIN] = _make_body_cap({EventName.FLYBY})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Kerbin Flyby 1"))
            self.assertFalse(self.can_reach_location("Mun Flyby 1"))
            self.assertFalse(self.can_reach_location("Eeloo Flyby 1"))

    def test_duna_flyby_uses_duna_profile(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.DUNA] = _make_body_cap({EventName.FLYBY})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Duna Flyby 1"))
            self.assertFalse(self.can_reach_location("Kerbin Flyby 1"))
            self.assertFalse(self.can_reach_location("Mun Flyby 1"))

    def test_flyby_and_orbit_are_separate_events(self):
        """Orbit-only profile must not satisfy the Flyby rule (separate event keys)."""
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.ORBIT})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun Orbit 1"))
            self.assertFalse(
                self.can_reach_location("Mun Flyby 1"),
                "Mun Flyby 1 must be False when Mun only has Orbit access.",
            )

    def test_all_flyby_locations_reachable_with_full_profile(self):
        """All bodies with locations → every '<body> Flyby 1' is reachable.

        Catches a flyby rule that always returns False for a specific body.
        Kerbol excluded (root body, no locations).
        """
        cap = _make_zero_cap()
        cap.bodies = {body.name: _make_body_cap(EventName) for body in ALL_BODIES}
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            for body in ALL_BODIES:
                if body.name == BodyName.KERBOL:
                    continue
                loc_name = f"{body.name} Flyby 1"
                self.assertTrue(
                    self.can_reach_location(loc_name),
                    f"'{loc_name}' unreachable with full profile — "
                    f"flyby rule for '{body.name}' may always return False.",
                )


# ---------------------------------------------------------------------------
# Class 5: Crewed events read from the correct body profile
# ---------------------------------------------------------------------------

class TestCrewedEventsRequireBodyProfile(KSP1TestBase):
    """EVA_IN_ORBIT, FLAG_PLANT, and SAMPLE_RETURN must each read from the
    correct per-body profile, not a hardcoded or shared body_name."""

    def test_eva_in_orbit_reads_correct_body(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.EVA_IN_ORBIT})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun EVA in Orbit 1"))
            self.assertFalse(
                self.can_reach_location("Kerbin EVA in Orbit 1"),
                "Kerbin EVA in Orbit must be False when only Mun has EVA_IN_ORBIT access.",
            )

    def test_flag_plant_reads_correct_body(self):
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.FLAG_PLANT})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun Flag Plant 1"))
            self.assertFalse(
                self.can_reach_location("Duna Flag Plant 1"),
                "Duna Flag Plant must be False when only Mun has FLAG_PLANT access.",
            )

    def test_sample_return_reads_correct_body(self):
        """Mun sample return uses the normal bracket-access rule (not the all-parts proxy)."""
        cap = _make_zero_cap()
        cap.bodies[BodyName.MUN] = _make_body_cap({EventName.SAMPLE_RETURN})
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            self.assertTrue(self.can_reach_location("Mun Sample Return 1"))
            self.assertFalse(
                self.can_reach_location("Duna Sample Return 1"),
                "Duna Sample Return must be False when only Mun has SAMPLE_RETURN access.",
            )

    def test_crewed_events_absent_when_body_missing(self):
        """All-False bodies dict → standard crewed event locations not reachable.

        Confirms that all-False access profiles block crewed locations.
        Eve/Tylo/Laythe sample return use the all-parts proxy (not get_capability)
        and are also not reachable here since no progression items are collected.
        """
        cap = _make_zero_cap()
        sample_locs = [
            "Mun EVA in Orbit 1",
            "Mun Flag Plant 1",
            "Mun Sample Return 1",
            "Duna EVA in Orbit 1",
            "Duna Flag Plant 1",
            "Duna Sample Return 1",
            "Minmus Flag Plant 1",
            "Kerbin EVA in Orbit 1",
        ]
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            for loc_name in sample_locs:
                self.assertFalse(
                    self.can_reach_location(loc_name),
                    f"'{loc_name}' should not be reachable with empty bodies dict.",
                )


# ---------------------------------------------------------------------------
# Class 6: Tech tree band gating
# ---------------------------------------------------------------------------

class TestTechTreeBandGating(KSP1TestBase):
    """Tech tree locations require both the R&D band gate AND the science budget gate.

    Band mapping: tiers 1-2 → band 0, tiers 3-4 → band 1, tiers 5-6 → band 2,
    tiers 7-8 → band 3.  Each band requires that many Progressive R&D items.
    """

    def test_no_tech_nodes_accessible_without_items(self):
        """Fresh state (no science, no R&D): no tech tree locations accessible."""
        for node in TECH_NODES:
            loc_name = f"{node.display_name} 1"
            self.assertFalse(
                self.can_reach_location(loc_name),
                f"Tech node '{loc_name}' must not be accessible without any items.",
            )

    def test_science_budget_gate_blocks_without_science(self):
        """Max R&D collected + zero mock science → no tech nodes accessible.

        Proves the science check is a real gate, independent of having R&D items.
        """
        self.collect_by_name(PROGRESSIVE_RD_NAME)  # collects all 3 copies
        cap = _make_zero_cap()
        # all bodies present but all events False → _accessible_science returns 0
        with patch("worlds.ksp1.rules.get_capability", return_value=cap):
            for node in TECH_NODES:
                loc_name = f"{node.display_name} 1"
                self.assertFalse(
                    self.can_reach_location(loc_name),
                    f"Tech node '{loc_name}' must not be accessible with zero science "
                    f"even when max Progressive R&D is collected.",
                )

    def test_rd_band_gate_blocks_higher_tiers(self):
        """No R&D collected → band>0 nodes (tiers 3+) are blocked by the R&D gate.

        The R&D check fires before the science check, so this holds regardless
        of the science budget.
        """
        for node in TECH_NODES:
            if TIER_TO_BAND[node.tier] == 0:
                continue  # band 0 nodes don't require R&D — skip
            loc_name = f"{node.display_name} 1"
            self.assertFalse(
                self.can_reach_location(loc_name),
                f"Tier-{node.tier} node '{loc_name}' (band {TIER_TO_BAND[node.tier]}) "
                f"must not be accessible without Progressive R&D.",
            )

    def test_both_gates_required_for_band1_node(self):
        """A band-1 (tier-3) node requires BOTH 1× Progressive R&D AND science.

        Case A: high mock science, no R&D → blocked (R&D gate).
        Case B: R&D present, zero mock science → blocked (science gate).
        """
        tier3_node = next(n for n in TECH_NODES if n.tier == 3)
        loc_name = f"{tier3_node.display_name} 1"

        # High-science mock: all bodies with orbit + crewed landing + instruments
        cap_with_science = _make_zero_cap()
        cap_with_science.has_capsule = True
        cap_with_science.has_thermometer = True
        cap_with_science.has_barometer = True
        cap_with_science.bodies = {
            body.name: _make_body_cap({EventName.ORBIT, EventName.CREWED_LANDING})
            for body in ALL_BODIES
        }

        # Case A: science present but no R&D → R&D gate blocks
        with patch("worlds.ksp1.rules.get_capability", return_value=cap_with_science):
            self.assertFalse(
                self.can_reach_location(loc_name),
                f"Tier-3 node '{loc_name}' must be blocked without R&D "
                f"even when science is abundant.",
            )

        # Case B: R&D present but no science → science gate blocks
        self.collect_by_name(PROGRESSIVE_RD_NAME)  # collect all 3
        cap_no_science = _make_zero_cap()
        cap_no_science.bodies = {}
        with patch("worlds.ksp1.rules.get_capability", return_value=cap_no_science):
            self.assertFalse(
                self.can_reach_location(loc_name),
                f"Tier-3 node '{loc_name}' must be blocked without science "
                f"even when max Progressive R&D is present.",
            )


if __name__ == "__main__":
    unittest.main()
