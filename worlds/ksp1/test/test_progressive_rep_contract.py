"""
Tests for the progressive representative contract in _pre_pass.

The client only unlocks:
  1. The chosen representative for each tier.
  2. Any non-rep tier parts that arrived as individual items.

_pre_pass must mirror this: for progressive-absorbed parts, only the rep gets
count=1 automatically; other tier parts need item_count_fn(name) > 0.
"""
import unittest

from worlds.ksp1.capability import _pre_pass
from worlds.ksp1.parts import PART_DB, PROGRESSIVE_PART_TIERS


# ---------------------------------------------------------------------------
# Probe core tier-1 parts and masses (verified against PART_DB)
# ---------------------------------------------------------------------------
# probeCoreOcto2.v2  OKTO2       0.04t  (lightest)
# probeCoreSphere.v2 Stayputnik  0.05t
# probeCoreCube      QBE         0.07t
# probeCoreOcto.v2   OKTO        0.10t
# roverBody.v2       RoveMate    0.15t  (heaviest)

_TIER1_PROBES = PROGRESSIVE_PART_TIERS["Progressive Probe Core"][1]


def _probe_item_count_fn(prog_count: int, extra_individual: set[str] | None = None):
    """Build item_count_fn for probe core tests only (no engines/tanks/etc.)."""
    extra = extra_individual or set()

    def fn(name: str) -> int:
        if name == "Progressive Probe Core":
            return prog_count
        if name in extra:
            return 1
        return 0

    return fn


class TestRepOnlyAvailableWhenTierUnlocked(unittest.TestCase):
    """Only the rep is available at tier unlock; non-reps are excluded."""

    def test_qbe_rep_only_qbe_selected(self):
        """Rep=QBE (0.07t): lightest_probe must be QBE, not OKTO2 (0.04t)."""
        rep = "probeCoreCube"
        rep_names = frozenset({rep})
        flags = _pre_pass(_probe_item_count_fn(1), start_with_clamps=False, rep_names=rep_names)

        self.assertTrue(flags.has_probe_core, "Rep probe core should be available")
        self.assertIsNotNone(flags.lightest_probe)
        self.assertAlmostEqual(
            flags.lightest_probe.mass, 0.07, places=3,
            msg=f"Expected QBE (0.07t) but got {flags.lightest_probe.name} ({flags.lightest_probe.mass}t)",
        )

    def test_okto2_rep_okto2_selected(self):
        """Rep=OKTO2 (0.04t): lightest_probe is OKTO2."""
        rep = "probeCoreOcto2.v2"
        rep_names = frozenset({rep})
        flags = _pre_pass(_probe_item_count_fn(1), start_with_clamps=False, rep_names=rep_names)

        self.assertTrue(flags.has_probe_core)
        self.assertAlmostEqual(flags.lightest_probe.mass, 0.04, places=3)

    def test_non_rep_probes_excluded(self):
        """With rep=QBE and no individual items, the 4 other tier-1 probes must be absent."""
        rep = "probeCoreCube"
        rep_names = frozenset({rep})
        flags = _pre_pass(_probe_item_count_fn(1), start_with_clamps=False, rep_names=rep_names)

        # Only QBE (0.07t) should be the lightest; OKTO2 (0.04t) must not appear
        self.assertGreater(
            flags.lightest_probe.mass, 0.04 + 1e-6,
            "OKTO2 (0.04t) should not be available — it wasn't received",
        )

    def test_no_progressive_no_probe(self):
        """Zero progressive count → no probe cores at all."""
        rep_names = frozenset({"probeCoreCube"})
        flags = _pre_pass(_probe_item_count_fn(0), start_with_clamps=False, rep_names=rep_names)

        self.assertFalse(flags.has_probe_core)
        self.assertIsNone(flags.lightest_probe)


class TestNonRepIncludedWhenIndividuallyReceived(unittest.TestCase):
    """Non-rep tier parts participate only when item_count_fn returns > 0."""

    def test_individually_received_non_rep_included(self):
        """Rep=QBE; Stayputnik (0.05t) received individually → both available, lightest wins."""
        rep = "probeCoreCube"
        rep_names = frozenset({rep})
        extra = {"probeCoreSphere.v2"}  # Stayputnik (0.05t)
        flags = _pre_pass(
            _probe_item_count_fn(1, extra), start_with_clamps=False, rep_names=rep_names
        )

        self.assertTrue(flags.has_probe_core)
        self.assertAlmostEqual(
            flags.lightest_probe.mass, 0.05, places=3,
            msg=f"Expected Stayputnik (0.05t) but got {flags.lightest_probe.name}",
        )

    def test_unreceived_non_rep_still_excluded(self):
        """Rep=QBE; OKTO2 NOT received individually → only QBE available."""
        rep = "probeCoreCube"
        rep_names = frozenset({rep})
        # probeCoreSphere.v2 received but NOT OKTO2
        extra = {"probeCoreSphere.v2"}
        flags = _pre_pass(
            _probe_item_count_fn(1, extra), start_with_clamps=False, rep_names=rep_names
        )

        # OKTO2 (0.04t) should NOT be available; lightest must be ≥ 0.05t
        self.assertGreater(
            flags.lightest_probe.mass, 0.04 + 1e-6,
            "OKTO2 should not appear — it was never individually received",
        )


class TestProgressiveSolarRepContract(unittest.TestCase):
    """Equivalent fix verification for a different chain (Progressive Solar Panel)."""

    def _solar_item_count_fn(self, prog_count: int, extra: set[str] | None = None):
        extra = extra or set()

        def fn(name: str) -> int:
            if name == "Progressive Solar Panel":
                return prog_count
            if name in extra:
                return 1
            return 0

        return fn

    def test_only_rep_solar_available(self):
        """Rep=OX-STAT (fixed panel); retractable panels must not appear."""
        tier1_solars = PROGRESSIVE_PART_TIERS["Progressive Solar Panel"][1]
        rep = tier1_solars[0]  # solarPanels5 = OX-STAT
        rep_names = frozenset({rep})
        flags = _pre_pass(self._solar_item_count_fn(1), start_with_clamps=False, rep_names=rep_names)

        # has_solar should be set (rep is a fixed solar panel)
        self.assertTrue(flags.has_solar, "Rep solar panel should grant has_solar")
        # has_solar_retractable should NOT be set (tier-2 parts weren't received)
        self.assertFalse(
            flags.has_solar_retractable,
            "Retractable panels should not be available when only fixed-panel rep was received",
        )

    def test_solar_excluded_when_no_progressive(self):
        """Zero Progressive Solar Panel → no solar at all."""
        rep_names = frozenset({"solarPanels5"})
        flags = _pre_pass(self._solar_item_count_fn(0), start_with_clamps=False, rep_names=rep_names)

        self.assertFalse(flags.has_solar)
        self.assertFalse(flags.has_solar_retractable)


class TestBugReproduction(unittest.TestCase):
    """Reproduce the exact bug: Progressive Probe Core=1, rep=QBE, OKTO2 suggested."""

    def test_bug_without_fix_would_give_okto2(self):
        """
        Before the fix, _pre_pass gave count=1 to ALL tier-1 probes when the
        progressive was unlocked, so OKTO2 (lightest at 0.04t) was always chosen.

        This test verifies the fixed behavior: with rep=QBE and no rep_names
        bypassed, only the rep participates.
        """
        rep = "probeCoreCube"
        rep_names = frozenset({rep})

        def item_count_fn(name: str) -> int:
            # Simulate a player who received:
            #   - Progressive Probe Core x1
            # but has NOT individually received any tier-1 probes
            if name == "Progressive Probe Core":
                return 1
            return 0

        flags = _pre_pass(item_count_fn, start_with_clamps=False, rep_names=rep_names)

        # The fixed behavior: QBE is the only probe available
        self.assertTrue(flags.has_probe_core)
        self.assertAlmostEqual(
            flags.lightest_probe.mass, 0.07, places=3,
            msg=(
                f"Bug: lightest_probe is {flags.lightest_probe.name} ({flags.lightest_probe.mass}t), "
                f"expected QBE (0.07t). OKTO2 (0.04t) must not appear unless individually received."
            ),
        )
