"""Structural invariants of the location model (``worlds/ksp1/locations/``).

These pin the properties the design relies on: each lifetime is a subset of the
one above, every AP location a world creates is backed by a typed def, and the
core modules stay Archipelago-free.
"""
import ast
import unittest
from pathlib import Path

from worlds.ksp1.locations.ap import KSP1Location
from worlds.ksp1.locations.catalog import LocationCatalog
from worlds.ksp1.test.base import KSP1TestBase

_PACKAGE = Path(__file__).parent.parent / "locations"

#: Modules (top-level, relative to worlds/) that core location modules must not
#: import: Archipelago itself, and the AP-coupled parts of the world.
_FORBIDDEN_IMPORTS = (
    "BaseClasses", "Options", "worlds.AutoWorld", "settings", "Utils", "NetUtils",
    "Fill", "worlds.generic",
)
_FORBIDDEN_SIBLINGS = ("world", "rules", "regions", "capability", "items", "sphere_ladder")


def _imported_modules(path: Path) -> list[str]:
    """Absolute or parent-relative module names a file imports, anywhere in it."""
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # ``from ..x import y`` (level 2) is a worlds.ksp1 sibling.
            prefix = "ksp1." if node.level == 2 else ""
            out.append(prefix + (node.module or ""))
    return out


class TestArchipelagoFree(unittest.TestCase):
    """Everything in ``locations/`` except ``ap.py`` must not import AP, so the
    model can later move out of ``worlds/`` as plain files."""

    def test_core_modules_do_not_import_archipelago(self):
        for path in sorted(_PACKAGE.glob("*.py")):
            if path.name == "ap.py":
                continue
            for module in _imported_modules(path):
                with self.subTest(file=path.name, module=module):
                    self.assertFalse(
                        any(module == f or module.startswith(f + ".") for f in _FORBIDDEN_IMPORTS),
                        f"{path.name} imports Archipelago module {module!r}")
                    self.assertNotIn(
                        module.removeprefix("ksp1."), _FORBIDDEN_SIBLINGS,
                        f"{path.name} imports AP-coupled world module {module!r}")


class TestCatalog(unittest.TestCase):
    def test_every_def_round_trips_by_name(self):
        for defn in LocationCatalog.all():
            self.assertIs(LocationCatalog.by_name(str(defn)), defn)

    def test_ids_are_unique(self):
        ids = list(LocationCatalog.name_to_id().values())
        self.assertEqual(len(ids), len(set(ids)))


class _WorldInvariants:
    def test_emitted_subset_of_catalog(self):
        seed = self.world.seed_locations
        for defn in seed.emitted():
            self.assertIsNotNone(LocationCatalog.by_name(str(defn)), str(defn))
        self.assertTrue(set(seed.configured.emitted()) <= set(seed.emitted()))

    def test_every_addressed_location_has_a_def(self):
        """Every AP location with an id is a KSP1Location backed by the emitted
        def of the same name (event locations have no address)."""
        emitted = {str(d): d for d in self.world.seed_locations.emitted()}
        addressed = [loc for loc in self.multiworld.get_locations(self.player)
                     if loc.address is not None]
        self.assertEqual({loc.name for loc in addressed}, set(emitted))
        for loc in addressed:
            self.assertIsInstance(loc, KSP1Location)
            self.assertEqual(loc.definition, emitted[loc.name], loc.name)
            self.assertEqual(loc.address, LocationCatalog.id_of(loc.definition))


class TestWorldInvariantsDefault(_WorldInvariants, KSP1TestBase):
    pass


class TestWorldInvariantsAlienHomeThresholds(_WorldInvariants, KSP1TestBase):
    options = {"starting_body": "mun", "goal": "complete_tech_tree",
               "goal_contract_mode": "progressive_unlock"}
