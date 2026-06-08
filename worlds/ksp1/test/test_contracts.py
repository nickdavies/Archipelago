"""
Contract system tests.

Covers the shared feasibility check (the single most safety-critical point — a
generation false positive yields an unsolvable seed), the required-part manifest,
slot_data round-trip, and the three-gate access rule under a full world setup.
"""
import unittest

from worlds.ksp1 import contracts as C
from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, MissionBuilder, DIFFICULTY_PROFILES,
)
from worlds.ksp1.capability import _pre_pass
from worlds.ksp1.test.base import KSP1TestBase

MB = MissionBuilder(home=BodyName.KERBIN)
DIFF = DIFFICULTY_PROFILES["normal"]


def _flags(item_count_fn):
    return _pre_pass(item_count_fn, start_with_clamps=True, rep_names=frozenset(),
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
        self.assertEqual(d["location"], "Contract: Mine Ore on Mun")
        self.assertEqual(d["schema"], C.CONTRACT_SCHEMA_VERSION)
        kinds = [p["kind"] for p in d["parameters"]]
        self.assertEqual(kinds, ["situation", "resource"])
        self.assertEqual(C.ContractSpec.from_slot_dict(d), MUN_MINE)

    def test_every_type_round_trips(self):
        # to_slot_dict/from_slot_dict must rebuild an identical spec for EVERY
        # contract type (UT regen reconstructs from fields, never re-randomizes),
        # and to_slot_dict must build a non-empty parameter tree for each type
        # without raising NotImplementedError.
        for ct, td in C.CONTRACT_TYPE_DEFS.items():
            with self.subTest(contract_type=ct):
                body = _first_compatible_body(td)
                self.assertIsNotNone(body, f"{ct} has no compatible body")
                spec = C.ContractSpec(ct, body)
                d = spec.to_slot_dict()
                self.assertTrue(d["parameters"], f"{ct} emitted no parameters")
                self.assertEqual(C.ContractSpec.from_slot_dict(d), spec)

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
        "non_goal_contract_count": 8,
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
        self.assertEqual(
            C.SpecificOrbitParam(
                body=BodyName.KERBIN, orbit_type="EQUATORIAL", inclination=0.0,
                eccentricity=0.0, sma=700000.0, deviation=10.0).to_json(),
            {"kind": "specific_orbit", "body": "Kerbin",
             "orbit_type": "EQUATORIAL", "inclination": 0.0, "eccentricity": 0.0,
             "sma": 700000.0, "deviation": 10.0})

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
        "non_goal_contract_count": 20,
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

    def test_home_safe_orbital_contracts_on_home(self):
        # home_safe orbit types are allowed on the home body (good early content).
        home = self.world.mission_builder.home.value
        home_orbitals = [
            loc.name for loc in self.multiworld.get_locations(self.player)
            if loc.name.startswith(self._NEW) and loc.name.endswith(home)
        ]
        self.assertTrue(
            home_orbitals, f"no home-body ({home}) orbital contracts generated")


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
                with self.subTest(contract_type=ct, kit=label):
                    lines = format_contract_output(
                        spec, in_logic=False, item_held=False,
                        flags=flags, diff=DIFF, difficulty_name="normal",
                        mission_builder=MB, proxy=False,
                    )
                    text = "\n".join(lines)
                    self.assertIn(spec.display_name, text)
                    self.assertIn("Gate 1", text)
                    self.assertIn("Gate 2", text)
                    self.assertIn("Gate 3", text)
                    self.assertIn("Contract parameters", text)
                    # Every emitted parameter primitive must show its kind.
                    for p in td.build_parameters(body):
                        self.assertIn(p.to_json()["kind"], text)

    def test_full_kit_mine_shows_required_parts_and_rocket(self):
        from worlds.ksp1.capability_format import format_contract_output
        text = "\n".join(format_contract_output(
            MUN_MINE, in_logic=True, item_held=True,
            flags=FULL, diff=DIFF, difficulty_name="normal",
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
            flags=EMPTY, diff=DIFF, difficulty_name="normal",
            mission_builder=MB, proxy=False,
        ))
        self.assertIn("drill: MISSING", text)
        self.assertIn("Gate 3 - physics delivery: NO", text)


if __name__ == "__main__":
    unittest.main()
