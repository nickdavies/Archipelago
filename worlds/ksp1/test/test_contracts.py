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


class TestExplainContractGeneric(unittest.TestCase):
    """`/explain Contract:` must work for EVERY contract type with no per-type
    handling — a future type that breaks the formatter fails here. Drives the
    formatter over each CONTRACT_TYPE_DEFS entry on a compatible body, with both
    a full kit and an empty kit, asserting it never raises and always renders the
    three gates and the parameter tree."""

    def _first_compatible_body(self, td):
        for body in ALL_BODIES:
            if td.body_compatible(body):
                return body.name
        return None

    def test_every_contract_type_renders(self):
        from worlds.ksp1.capability_format import format_contract_output
        for ct, td in C.CONTRACT_TYPE_DEFS.items():
            body = self._first_compatible_body(td)
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
