"""
Validation tests for the real KSP part data pipeline.

Ensures parts.json, PART_REGISTRY, and PART_DB are consistent and sane.
"""
import json
import pkgutil
import unittest

from worlds.ksp1.parts import (
    PART_DB, PART_REGISTRY, PartMapping, CapabilityFlag, _DUAL_PURPOSE,
    Engine, FuelTank, SolidBooster, HeatShield,
    Parachute, LandingLeg, Decoupler, MiscEquipment,
)


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
        for name, offset in LOCATION_TABLE.items():
            self.assertGreaterEqual(
                offset, 2000,
                f"Location {name!r} offset {offset} < 2000",
            )
            self.assertLess(
                offset, 4000,
                f"Location {name!r} offset {offset} >= 4000",
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


if __name__ == "__main__":
    unittest.main()
