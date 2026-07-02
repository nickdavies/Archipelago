"""
Validation tests for the real KSP part data pipeline.

Ensures parts.json, PART_REGISTRY, and PART_DB are consistent and sane.
"""
import json
import pkgutil
import unittest

from worlds.ksp1.parts import (
    DEFAULT_PART_MANAGER, PART_REGISTRY, PartMapping, CapabilityFlag,
    Engine, FuelTank, SolidBooster, HeatShield,
    Parachute, LandingLeg, Decoupler, MiscEquipment,
)
# Internal data-integrity test: reaches into the private loader for the raw
# universe + the dual-purpose table it asserts on.
from worlds.ksp1.parts._raw import _DUAL_PURPOSE

PART_DB = DEFAULT_PART_MANAGER.parts


def _load_json() -> dict:
    raw = pkgutil.get_data("worlds.ksp1", "data/parts.json")
    assert raw is not None
    return json.loads(raw.decode("utf-8"))


class TestRegistryJsonConsistency(unittest.TestCase):
    """Every cfg_name in PART_REGISTRY must exist in parts.json."""

    def test_all_cfg_names_exist_in_json(self) -> None:
        parts_json = _load_json()
        for mapping in PART_REGISTRY:
            self.assertIn(
                mapping.cfg_name, parts_json,
                f"PART_REGISTRY references {mapping.cfg_name!r} but it's "
                f"not in parts.json (AP item: {mapping.title!r})",
            )


class TestNoDuplicateCfgNames(unittest.TestCase):
    """Each cfg name should be mapped at most once."""

    def test_unique_cfg_names(self) -> None:
        seen: dict[str, str] = {}
        for mapping in PART_REGISTRY:
            if mapping.cfg_name in seen:
                self.fail(
                    f"cfg_name {mapping.cfg_name!r} mapped twice: "
                    f"{seen[mapping.cfg_name]!r} and {mapping.title!r}"
                )
            seen[mapping.cfg_name] = mapping.title


class TestPartDbPopulated(unittest.TestCase):
    """Every AP item in PART_REGISTRY produces a non-empty list in PART_DB."""

    def test_all_registry_items_in_part_db(self) -> None:
        for mapping in PART_REGISTRY:
            self.assertIn(
                mapping.ksp_name, PART_DB,
                f"ksp_name {mapping.ksp_name!r} ({mapping.title!r}) "
                f"not found in PART_DB",
            )
            self.assertTrue(
                len(PART_DB[mapping.ksp_name]) > 0,
                f"PART_DB[{mapping.ksp_name!r}] is empty",
            )

    def test_part_types_match_registry(self) -> None:
        dual_cfg_names = set(_DUAL_PURPOSE)
        for mapping in PART_REGISTRY:
            parts = PART_DB.get(mapping.ksp_name, [])
            for part in parts:
                # Dual-purpose parts add a MiscEquipment alongside the
                # primary type — skip MiscEquipment extras for those.
                if (mapping.cfg_name in dual_cfg_names
                        and isinstance(part, MiscEquipment)):
                    continue
                self.assertIsInstance(
                    part, mapping.part_type,
                    f"PART_DB[{mapping.ksp_name!r}] contains {type(part).__name__} "
                    f"but registry declares {mapping.part_type.__name__}",
                )


class TestEngineSanity(unittest.TestCase):
    """All engines must have positive, physically sensible values."""

    def _engines(self):
        for name, parts in PART_DB.items():
            for part in parts:
                if isinstance(part, Engine):
                    yield name, part

    def test_positive_isp(self) -> None:
        for name, eng in self._engines():
            self.assertGreater(eng.vac_isp, 0, f"{name}: vac_isp must be > 0")
            self.assertGreater(eng.atm_isp, 0, f"{name}: atm_isp must be > 0")

    def test_vac_isp_gte_atm_isp(self) -> None:
        for name, eng in self._engines():
            self.assertGreaterEqual(
                eng.vac_isp, eng.atm_isp,
                f"{name}: vac_isp ({eng.vac_isp}) < atm_isp ({eng.atm_isp})",
            )

    def test_positive_thrust(self) -> None:
        for name, eng in self._engines():
            self.assertGreater(eng.vac_thrust, 0, f"{name}: vac_thrust must be > 0")
            self.assertGreater(eng.atm_thrust, 0, f"{name}: atm_thrust must be > 0")

    def test_positive_mass(self) -> None:
        for name, eng in self._engines():
            self.assertGreater(eng.mass, 0, f"{name}: mass must be > 0")

    def test_size_class_valid(self) -> None:
        valid_sizes = {0.625, 1.25, 1.875, 2.5, 3.75, 5.0}
        for name, eng in self._engines():
            self.assertIn(
                eng.size_class, valid_sizes,
                f"{name}: size_class {eng.size_class} not in valid set",
            )


class TestFuelTankSanity(unittest.TestCase):
    """All fuel tanks must have positive dry and fuel mass."""

    def _tanks(self):
        for name, parts in PART_DB.items():
            for part in parts:
                if isinstance(part, FuelTank):
                    yield name, part

    def test_positive_dry_mass(self) -> None:
        for name, tank in self._tanks():
            self.assertGreater(tank.dry_mass, 0, f"{name}: dry_mass must be > 0")

    def test_positive_fuel_mass(self) -> None:
        for name, tank in self._tanks():
            self.assertGreater(tank.fuel_mass, 0, f"{name}: fuel_mass must be > 0")

    def test_fuel_type_valid(self) -> None:
        valid_types = {"lfo", "lf", "xenon", "monoprop"}
        for name, tank in self._tanks():
            self.assertIn(
                tank.fuel_type, valid_types,
                f"{name}: fuel_type {tank.fuel_type!r} not valid",
            )


class TestSolidBoosterSanity(unittest.TestCase):
    """SRBs must have positive mass and ISP."""

    def _srbs(self):
        for name, parts in PART_DB.items():
            for part in parts:
                if isinstance(part, SolidBooster):
                    yield name, part

    def test_positive_fuel_mass(self) -> None:
        for name, srb in self._srbs():
            self.assertGreater(srb.fuel_mass, 0, f"{name}: fuel_mass must be > 0")

    def test_positive_dry_mass(self) -> None:
        for name, srb in self._srbs():
            self.assertGreater(srb.dry_mass, 0, f"{name}: dry_mass must be > 0")


class TestPartDbItemTableSync(unittest.TestCase):
    """Every PART_DB key must exist in ITEM_TABLE and vice versa."""

    def test_part_db_keys_in_item_table(self) -> None:
        from worlds.ksp1.items import ITEM_TABLE
        for name in PART_DB:
            self.assertIn(
                name, ITEM_TABLE,
                f"PART_DB key {name!r} not in ITEM_TABLE",
            )

    def test_item_table_part_keys_in_part_db(self) -> None:
        from worlds.ksp1.items import ITEM_TABLE
        # Every ITEM_TABLE entry that came from PART_DB should still be there
        for name in ITEM_TABLE:
            self.assertIn(
                name, PART_DB,
                f"ITEM_TABLE key {name!r} not found in PART_DB",
            )


class TestProvidesFlags(unittest.TestCase):
    """All provides flags in PART_REGISTRY must be recognized by the system."""

    _KNOWN_FLAGS: frozenset[str] = frozenset(CapabilityFlag)

    def test_all_provides_flags_known(self) -> None:
        for mapping in PART_REGISTRY:
            if mapping.part_type is not MiscEquipment:
                continue
            provides = mapping.overrides.get("provides", frozenset())
            for flag in provides:
                self.assertIn(
                    flag, self._KNOWN_FLAGS,
                    f"Unknown provides flag {flag!r} on {mapping.title!r}",
                )

    def test_precollected_items_in_part_db(self) -> None:
        from worlds.ksp1.items import ALWAYS_PRECOLLECTED, CLAMP_PRECOLLECTED
        for name in ALWAYS_PRECOLLECTED + CLAMP_PRECOLLECTED:
            self.assertIn(
                name, PART_DB,
                f"Precollected item {name!r} not found in PART_DB",
            )


class TestSpotCheckValues(unittest.TestCase):
    """Spot-check a few well-known parts against KSP wiki values."""

    def test_reliant(self) -> None:
        eng = PART_DB["liquidEngine.v2"][0]
        assert isinstance(eng, Engine)
        self.assertAlmostEqual(eng.vac_isp, 310.0, places=0)
        self.assertAlmostEqual(eng.vac_thrust, 240.0, places=0)
        self.assertFalse(eng.has_gimbal)
        self.assertEqual(eng.size_class, 1.25)

    def test_swivel(self) -> None:
        eng = PART_DB["liquidEngine2.v2"][0]
        assert isinstance(eng, Engine)
        self.assertTrue(eng.has_gimbal)
        self.assertAlmostEqual(eng.vac_isp, 320.0, places=0)

    def test_flea_srb(self) -> None:
        srb = PART_DB["solidBooster.sm.v2"][0]
        assert isinstance(srb, SolidBooster)
        self.assertAlmostEqual(srb.vac_isp, 165.0, places=0)
        self.assertAlmostEqual(srb.vac_thrust, 192.0, places=0)

    def test_fl_t400(self) -> None:
        tank = PART_DB["fuelTank"][0]
        assert isinstance(tank, FuelTank)
        # LF=180, Ox=220 -> fuel_mass = 180*0.005 + 220*0.005 = 2.0
        self.assertAlmostEqual(tank.fuel_mass, 2.0, places=2)
        self.assertEqual(tank.fuel_type, "lfo")


class TestStableIdRanges(unittest.TestCase):
    """All item and location IDs use non-overlapping, in-range offsets."""

    def test_part_offsets_unique_and_in_range(self) -> None:
        seen: dict[int, str] = {}
        for mapping in PART_REGISTRY:
            self.assertGreaterEqual(
                mapping.offset, 1000,
                f"{mapping.title!r} offset {mapping.offset} < 1000",
            )
            self.assertLess(
                mapping.offset, 2000,
                f"{mapping.title!r} offset {mapping.offset} >= 2000",
            )
            if mapping.offset in seen:
                self.fail(
                    f"Duplicate offset {mapping.offset}: "
                    f"{seen[mapping.offset]!r} and {mapping.title!r}"
                )
            seen[mapping.offset] = mapping.title

    def test_filler_offsets_in_range(self) -> None:
        from worlds.ksp1.items import _FILLER_ITEMS
        for name, (offset, _) in _FILLER_ITEMS.items():
            self.assertGreaterEqual(offset, 100, f"{name} offset {offset} < 100")
            self.assertLess(offset, 200, f"{name} offset {offset} >= 200")

    def test_victory_offset(self) -> None:
        from worlds.ksp1.items import _VICTORY_ITEM
        offset, _ = _VICTORY_ITEM["Victory"]
        self.assertEqual(offset, 0)

    def test_no_item_offset_collisions(self) -> None:
        from worlds.ksp1.items import ITEM_NAME_TO_ID
        ids = list(ITEM_NAME_TO_ID.values())
        self.assertEqual(len(ids), len(set(ids)), "Duplicate item IDs detected")

    def test_location_offsets_in_range(self) -> None:
        from worlds.ksp1.locations import LOCATION_TABLE
        # Location offsets occupy distinct bands: 2000-4199 for the legacy
        # buckets (starting inventory, KSC biomes, Kerbin home specials,
        # mission events, tech tree) plus the 14 non-Kerbin home sets, a small
        # 19_000-block for goal-mode threshold locations, and a large dedicated
        # 20_000+ block for contract completion locations (3 names per spec).
        for name, offset in LOCATION_TABLE.items():
            in_legacy = 2000 <= offset < 4200
            in_thresholds = 19_000 <= offset < 19_100
            in_contracts = 20_000 <= offset < 30_000
            self.assertTrue(
                in_legacy or in_thresholds or in_contracts,
                f"Location {name!r} offset {offset} outside known bands",
            )

    def test_no_location_offset_collisions(self) -> None:
        from worlds.ksp1.locations import LOCATION_NAME_TO_ID
        ids = list(LOCATION_NAME_TO_ID.values())
        self.assertEqual(len(ids), len(set(ids)), "Duplicate location IDs detected")

    def test_item_and_location_ranges_no_overlap(self) -> None:
        from worlds.ksp1.items import ITEM_NAME_TO_ID
        from worlds.ksp1.locations import LOCATION_NAME_TO_ID
        item_ids = set(ITEM_NAME_TO_ID.values())
        loc_ids = set(LOCATION_NAME_TO_ID.values())
        overlap = item_ids & loc_ids
        self.assertEqual(len(overlap), 0, f"Overlapping IDs: {overlap}")


class TestDescentModelPartData(unittest.TestCase):
    """Fields feeding the staged atmospheric-descent model: chute deploy
    envelopes (q_safe / gates / semi drag) and heat-shield deployed drag areas
    from the baked drag cubes."""

    def _chute(self, ksp_name: str) -> Parachute:
        part = next(p for p in PART_DB[ksp_name] if isinstance(p, Parachute))
        return part

    def _shield(self, ksp_name: str) -> HeatShield:
        return next(p for p in PART_DB[ksp_name] if isinstance(p, HeatShield))

    def test_main_chutes_use_baseline_q_safe(self) -> None:
        for name in ("parachuteSingle", "parachuteLarge", "parachuteRadial"):
            chute = self._chute(name)
            self.assertFalse(chute.is_drogue)
            self.assertAlmostEqual(chute.q_safe_kpa, 8.8, places=3)
            self.assertEqual(chute.deploy_altitude_m, 1000.0)
            self.assertAlmostEqual(chute.min_pressure_atm, 0.04)
            self.assertGreater(chute.semi_drag_area, 0.0)

    def test_drogues_get_higher_q_envelope(self) -> None:
        mk25 = self._chute("parachuteDrogue")
        mk12r = self._chute("radialDrogue")
        for d in (mk25, mk12r):
            self.assertTrue(d.is_drogue)
            self.assertEqual(d.deploy_altitude_m, 2500.0)
            self.assertAlmostEqual(d.min_pressure_atm, 0.02)
        # Mk25: (1600/650)/0.25 = 9.85 -> capped at x8 = 70.4 kPa
        self.assertAlmostEqual(mk25.q_safe_kpa, 70.4, places=1)
        # Mk12-R: (1100/650)/0.5 = 3.38 -> 29.8 kPa
        self.assertAlmostEqual(mk12r.q_safe_kpa, 29.78, places=1)
        # Golden rule: the cap bounds every derived envelope
        for d in (mk25, mk12r):
            self.assertLessEqual(d.q_safe_kpa, 8.8 * 8.0 + 1e-9)

    def test_all_shields_have_cube_drag_area(self) -> None:
        shields = [p for parts in PART_DB.values() for p in parts
                   if isinstance(p, HeatShield)]
        self.assertGreaterEqual(len(shields), 6)
        for s in shields:
            self.assertGreater(
                s.drag_area, 0.0,
                f"{s.name} has no deployed drag area — cube extraction broke; "
                f"the descent model would credit zero aero bleed",
            )
        # Bigger shields present more face: drag area is monotone in size class
        by_size = sorted(shields, key=lambda s: s.size_class)
        for a, b in zip(by_size, by_size[1:]):
            self.assertLessEqual(a.drag_area, b.drag_area + 1e-9)

    def test_inflatable_shield_deployed_state(self) -> None:
        infl = self._shield("InflatableHeatShield")
        # 10m across when inflated: covers wide craft AND dominates entry drag.
        self.assertEqual(infl.size_class, 10.0)
        # 0.8 * cd_y * area_y of the inflated cube "A" (72.75 m2, cd 0.8279)
        self.assertAlmostEqual(infl.drag_area, 48.18, places=1)

    def test_rigid_shield_uses_clean_cube(self) -> None:
        hs2 = self._shield("HeatShield2")
        self.assertAlmostEqual(hs2.drag_area, 3.79, places=1)
        self.assertEqual(hs2.size_class, 2.5)

    def test_airbrake_raw_data_extracted(self) -> None:
        """airbrake1 stays MiscEquipment (Phase E decides inclusion) but the
        raw aero data must be in parts.json for the calibration."""
        parts_json = _load_json()
        ab = parts_json["airbrake1"]
        self.assertIn("aero_surface", ab)
        self.assertAlmostEqual(ab["aero_surface"]["deflection_lift_coeff"], 0.38)
        self.assertEqual(ab["aero_surface"]["lifting_surface_curve"], "SpeedBrake")
        self.assertIn("drag_cubes", ab)
        self.assertIn("fullDeflectionPos", ab["drag_cubes"])


if __name__ == "__main__":
    unittest.main()
