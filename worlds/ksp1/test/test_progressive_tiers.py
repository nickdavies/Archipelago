"""
Validation tests for progressive part tier assignments.

These tests verify that the tier assignments produce the expected mission
reachability at each progressive level.  They use compute_capability_from_items()
with mock item_count_fn to simulate "player has progressive tier N + all
individual items" without requiring production code changes.
"""
import unittest
from typing import Callable

from worlds.ksp1.parts import (
    PART_DB, PROGRESSIVE_PART_TIERS, PROGRESSIVE_PART_NAMES,
    PROGRESSIVE_PART_COUNTS,
)
from worlds.ksp1.capability import compute_capability_from_items


def _make_item_count_fn(
    progressive_levels: dict[str, int],
    include_all_individual: bool = True,
) -> Callable[[str], int]:
    """
    Build a mock item_count_fn for testing.

    progressive_levels: {progressive_name: tier_level} — unlocks tiers 1..N
    include_all_individual: if True, all non-absorbed items return count=1
    """
    # Build set of unlocked part names from progressive tiers
    unlocked: set[str] = set()
    for prog_name, level in progressive_levels.items():
        tiers = PROGRESSIVE_PART_TIERS.get(prog_name, {})
        for t in range(1, level + 1):
            unlocked.update(tiers.get(t, []))

    def item_count_fn(name: str) -> int:
        # Progressive item itself: return count for fingerprint
        if name in PROGRESSIVE_PART_COUNTS:
            return progressive_levels.get(name, 0)
        # Part unlocked by progressive tier
        if name in unlocked:
            return 1
        # Part absorbed but not yet unlocked
        if name in PROGRESSIVE_PART_NAMES:
            return 0
        # Non-absorbed individual item
        if include_all_individual and name in PART_DB:
            return 1
        return 0

    return item_count_fn


class TestTierZeroGating(unittest.TestCase):
    """With zero progressive items, nothing should be reachable."""

    def test_no_orbit_without_progressive(self):
        """Zero progressive items → cannot reach Kerbin orbit."""
        fn = _make_item_count_fn({}, include_all_individual=True)
        cap, flags = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertFalse(
            kerbin is not None and kerbin.access.get("Orbit", False),
            "Should NOT be able to orbit Kerbin with zero progressive items "
            f"(engines={len(flags.available_engines)}, tanks={len(flags.available_tanks)}, "
            f"srbs={len(flags.available_srbs)})",
        )


class TestTierOneCapability(unittest.TestCase):
    """Tier 1 of all categories: should reach Kerbin orbit."""

    def _tier1_all(self) -> dict[str, int]:
        return {name: 1 for name in PROGRESSIVE_PART_COUNTS}

    def test_kerbin_orbit_achievable(self):
        fn = _make_item_count_fn(self._tier1_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertIsNotNone(kerbin)
        self.assertTrue(
            kerbin.access.get("Orbit", False),
            f"Tier 1 should reach Kerbin orbit. Blocking: {kerbin.blocking_reason}",
        )

    def test_mun_flyby_likely(self):
        fn = _make_item_count_fn(self._tier1_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        mun = cap.bodies.get("Mun")
        # Flyby = can_escape from Kerbin is enough to reach Mun SOI
        kerbin = cap.bodies.get("Kerbin")
        self.assertTrue(
            kerbin is not None and kerbin.access.get("Flyby", False),
            f"Tier 1 should at least be able to escape Kerbin. "
            f"Blocking: {kerbin.blocking_reason if kerbin else 'no Kerbin profile'}",
        )

    def test_not_too_powerful(self):
        """Tier 1 should NOT reach Duna orbit (too generous if so)."""
        fn = _make_item_count_fn(self._tier1_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        duna = cap.bodies.get("Duna")
        if duna is not None:
            self.assertFalse(
                duna.access.get("Orbit", False),
                "Tier 1 reaching Duna orbit means tier 1 is too generous",
            )


class TestTierTwoCapability(unittest.TestCase):
    """Tier 2 of all categories: Mun/Minmus landing + return."""

    def _tier2_all(self) -> dict[str, int]:
        return {name: min(2, cnt) for name, cnt in PROGRESSIVE_PART_COUNTS.items()}

    def test_mun_return(self):
        fn = _make_item_count_fn(self._tier2_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        mun = cap.bodies.get("Mun")
        self.assertIsNotNone(mun)
        self.assertTrue(
            mun.access.get("Return", False),
            f"Tier 2 should enable Mun return. Blocking: {mun.blocking_reason}",
        )

    def test_minmus_return(self):
        fn = _make_item_count_fn(self._tier2_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        minmus = cap.bodies.get("Minmus")
        self.assertIsNotNone(minmus)
        self.assertTrue(
            minmus.access.get("Return", False),
            f"Tier 2 should enable Minmus return. Blocking: {minmus.blocking_reason}",
        )


class TestTierThreeCapability(unittest.TestCase):
    """Tier 3+ of all categories: outer system access."""

    def _tier3_all(self) -> dict[str, int]:
        return {name: min(3, cnt) for name, cnt in PROGRESSIVE_PART_COUNTS.items()}

    def test_duna_return(self):
        fn = _make_item_count_fn(self._tier3_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        duna = cap.bodies.get("Duna")
        self.assertIsNotNone(duna)
        self.assertTrue(
            duna.access.get("Return", False),
            f"Tier 3 should enable Duna return. Blocking: {duna.blocking_reason}",
        )

    def test_duna_orbit(self):
        """Tier 3 should enable Duna orbit (Jool may need T4 tanks)."""
        fn = _make_item_count_fn(self._tier3_all())
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        duna = cap.bodies.get("Duna")
        self.assertIsNotNone(duna)
        self.assertTrue(
            duna.access.get("Orbit", False),
            f"Tier 3 should enable Duna orbit. Blocking: {duna.blocking_reason}",
        )


class TestAllTiersMaxed(unittest.TestCase):
    """All tiers maxed ≈ equivalent to current all-parts behavior."""

    def _all_max(self) -> dict[str, int]:
        return dict(PROGRESSIVE_PART_COUNTS)

    def test_equivalent_to_all_parts(self):
        """Maxed progressive tiers + individual items should match all-parts reach."""
        # All-parts baseline
        fn_all = _make_item_count_fn({}, include_all_individual=False)

        def all_items_fn(name: str) -> int:
            return 1 if name in PART_DB else 0

        cap_all, _ = compute_capability_from_items(all_items_fn, "normal", start_with_clamps=True)
        all_orbitals = {
            b for b, bp in cap_all.bodies.items() if bp.access.get("Orbit", False)
        }

        # Progressive maxed
        fn_prog = _make_item_count_fn(self._all_max(), include_all_individual=True)
        cap_prog, _ = compute_capability_from_items(fn_prog, "normal", start_with_clamps=True)
        prog_orbitals = {
            b for b, bp in cap_prog.bodies.items() if bp.access.get("Orbit", False)
        }

        # Progressive should reach at least as many bodies
        missing = all_orbitals - prog_orbitals
        self.assertEqual(
            missing, set(),
            f"Bodies reachable with all parts but not with maxed progressive: {missing}",
        )


class TestMixedTierCombos(unittest.TestCase):
    """Mixed tier combos should produce reasonable intermediate capability."""

    def test_launch_t2_vacuum_t1_tank_t1(self):
        """Launch T2 + vacuum T1 + tank T1 = more than just Kerbin orbit."""
        levels = {name: 1 for name in PROGRESSIVE_PART_COUNTS}
        levels["Progressive Launch Engine"] = 2
        fn = _make_item_count_fn(levels)
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertTrue(kerbin and kerbin.access.get("Orbit", False))
        # With bigger engines but small tanks, should at least reach Kerbin escape
        self.assertTrue(kerbin and kerbin.access.get("Flyby", False))

    def test_engines_without_tanks_or_individual(self):
        """Engines T1 + no tanks + no individual items = no orbit."""
        levels = {
            "Progressive Launch Engine": 1,
            "Progressive Stack Decoupler": 1,
            "Progressive Capsule": 1,
            "Progressive Probe Core": 1,
        }
        fn = _make_item_count_fn(levels, include_all_individual=False)
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertFalse(
            kerbin is not None and kerbin.access.get("Orbit", False),
            "Should not orbit with engines but zero fuel tanks/SRBs",
        )

    def test_srbs_only_no_liquid(self):
        """SRBs T1 + stack decoupler T1 + capsule T1 = sounding rockets, maybe orbit."""
        levels = {
            "Progressive SRB": 1,
            "Progressive Stack Decoupler": 1,
            "Progressive Capsule": 1,
            "Progressive Probe Core": 1,
            "Progressive Solar Panel": 1,
            "Progressive Relay": 1,
        }
        fn = _make_item_count_fn(levels, include_all_individual=True)
        cap, _ = compute_capability_from_items(fn, "normal", start_with_clamps=True)
        # With Flea+Mite+Shrimp, should at least achieve sounding altitude
        self.assertGreater(
            cap.sounding_altitude_km, 0,
            "SRBs T1 should achieve some sounding altitude",
        )


class TestDifficultyInteraction(unittest.TestCase):
    """Tier assignments should work across difficulty levels."""

    def _tier1_all(self) -> dict[str, int]:
        return {name: 1 for name in PROGRESSIVE_PART_COUNTS}

    def test_tier1_orbit_casual(self):
        fn = _make_item_count_fn(self._tier1_all())
        cap, _ = compute_capability_from_items(fn, "casual", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertTrue(
            kerbin and kerbin.access.get("Orbit", False),
            "Tier 1 must reach Kerbin orbit even on casual difficulty",
        )

    def test_tier1_orbit_expert(self):
        fn = _make_item_count_fn(self._tier1_all())
        cap, _ = compute_capability_from_items(fn, "expert", start_with_clamps=True)
        kerbin = cap.bodies.get("Kerbin")
        self.assertTrue(
            kerbin and kerbin.access.get("Orbit", False),
            f"Tier 1 must reach Kerbin orbit on expert. Blocking: "
            f"{kerbin.blocking_reason if kerbin else 'no profile'}",
        )


if __name__ == "__main__":
    unittest.main()
