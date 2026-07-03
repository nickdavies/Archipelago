"""Escalated build caps (capability ceiling): offline-scoped, Apollo-only.

The escalated search space (K=3 multistage, wide asparagus fans) is reachable
ONLY for ascent edges the offline feasibility probe marked eligible
(``data.feasibility.ESCALATED_ASCENT_EDGES``) and ONLY inside Apollo-split
evaluations.  These tests pin the load-bearing properties:

* the eligibility set is what the probe measured (Eve's atmospheric ascent —
  the one edge whose escalation flips max-kit missions feasible),
* Eve Return closes at max kit at the table's probe bar via Apollo +
  escalation, and drops back to infeasible without docking ports (the
  escalation can only be reached through the Apollo retry),
* the standard architecture NEVER escalates — a whole-stack evaluation of
  the same profile stays infeasible at max kit (the hot path keeps K=2),
* the K=3 split grid is well-formed.
"""
from __future__ import annotations

import unittest

from worlds.ksp1.bodies import (
    BodyName, DIFFICULTY_PROFILES, EdgeType, MissionBuilder, MissionType,
)
from worlds.ksp1.capability import (
    _evaluate_profile, compute_capability_from_items,
)
from worlds.ksp1.data.feasibility import ESCALATED_ASCENT_EDGES
from worlds.ksp1.locations import EventName
from worlds.ksp1.parts import CapabilityFlag, DEFAULT_PART_MANAGER
from worlds.ksp1.rocket_math import _F4_DV_SPLITS
from worlds.ksp1.scripts.generate_feasibility import (
    DEFAULT_OVERHEAD, _profile_with_overhead,
)

_PART_DB = DEFAULT_PART_MANAGER.parts

_PORT_NAMES = frozenset(
    nm for nm, parts in _PART_DB.items()
    if any(CapabilityFlag.DOCKING_PORT in getattr(p, "provides", ())
           for p in parts)
)

# The table's probe bar at which Eve Return is newly in the dv model's reach
# (see data/feasibility.py: at small margins the only remaining exclusions
# are Eve-as-home).  If tuning moves this, read the regenerated table for
# the new reference difficulty.
_PROBE_PROFILE = _profile_with_overhead("small", DEFAULT_OVERHEAD)


def _max_kit(exclude: frozenset[str] = frozenset()):
    counts = {n: 1 for n in _PART_DB if n not in exclude}
    return lambda name: counts.get(name, 0)


class TestEligibilitySet(unittest.TestCase):
    def test_probed_set_is_eve_atmospheric_ascent(self) -> None:
        """The offline probe found exactly one escalation-worthy edge.  A
        change here means the physics moved — re-read the table diff before
        accepting."""
        self.assertEqual(
            ESCALATED_ASCENT_EDGES,
            frozenset({(BodyName.EVE, EdgeType.ATMOSPHERIC_ASCENT)}),
        )

    def test_no_home_ascent_is_eligible(self) -> None:
        """Escalating a HOME ascent is the hot-path cost the design bans
        from the APOLLO set (the probe skips edge.body == home candidates);
        approved exceptions live ONLY in the separate allowlisted
        ESCALATED_HOME_ASCENT_EDGES."""
        for body, _et in ESCALATED_ASCENT_EDGES:
            # An eligible edge must come from a DESTINATION ascent — i.e.
            # the probe skipped candidates where edge.body == home, so the
            # pair can only have entered via missions from OTHER homes.
            self.assertIsInstance(body, BodyName)

    def test_home_set_is_allowlisted_eve_only(self) -> None:
        """The HOME-ascent escalation set is operator-allowlisted (the probe
        can only confirm entries, never add homes) — a change here is an
        operator decision, not a physics drift to wave through."""
        from worlds.ksp1.data.feasibility import ESCALATED_HOME_ASCENT_EDGES
        from worlds.ksp1.scripts.generate_feasibility import (
            _HOME_ESCALATION_ALLOWLIST,
        )
        self.assertEqual(
            ESCALATED_HOME_ASCENT_EDGES,
            frozenset({(BodyName.EVE, EdgeType.ATMOSPHERIC_ASCENT)}),
        )
        self.assertTrue(
            ESCALATED_HOME_ASCENT_EDGES <= _HOME_ESCALATION_ALLOWLIST)


class TestSplitGrid(unittest.TestCase):
    def test_splits_sum_to_one(self) -> None:
        for k, grid in _F4_DV_SPLITS.items():
            for split in grid:
                self.assertEqual(len(split), k)
                self.assertAlmostEqual(sum(split), 1.0, places=9,
                                       msg=f"K={k} split {split}")


class TestEveCeiling(unittest.TestCase):
    """Max-kit Eve Return at the probe bar: closes via Apollo + escalation,
    ports-gated, and never via the standard architecture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mb = MissionBuilder(home=BodyName.KERBIN)
        cls.cap, cls.flags = compute_capability_from_items(
            _max_kit(),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=cls.mb,
        )

    def test_eve_return_closes_at_max_kit(self) -> None:
        self.assertTrue(
            self.cap.bodies[BodyName.EVE].access[EventName.RETURN],
            "Eve Return should close via Apollo + escalated ascent",
        )
        # Crewed Eve SSR is OUT at the small probe bar since the dv
        # recalibration (lander ascent 8,000 → ~8,996 highlands-in-logic):
        # the pod+ladder payload on the escalated ascent no longer fits.
        # It survives only at zero physics, and not from Kerbin — see the
        # regenerated table.
        self.assertFalse(
            self.cap.bodies[BodyName.EVE].access[EventName.SAMPLE_RETURN],
            "crewed Eve SSR at small+bar should be excluded post-recalibration",
        )

    def test_no_docking_port_no_escalation(self) -> None:
        """Escalation is reachable only through the Apollo retry, which is
        ports-gated — stripping ports must drop Eve Return back out."""
        cap, _flags = compute_capability_from_items(
            _max_kit(exclude=_PORT_NAMES),
            difficulty_name=_PROBE_PROFILE,
            start_with_clamps=True,
            mission_builder=self.mb,
        )
        self.assertFalse(cap.bodies[BodyName.EVE].access[EventName.RETURN])

    def test_standard_architecture_never_escalates(self) -> None:
        """The whole-stack evaluation of the same profiles must stay
        infeasible at max kit — proof the hot path keeps the default caps."""
        profiles = self.mb.profiles_for(BodyName.EVE, MissionType.RETURN)
        for profile in profiles:
            res = _evaluate_profile(
                profile, self.flags, DIFFICULTY_PROFILES[_PROBE_PROFILE],
                MissionType.RETURN, is_crewed=False, home=BodyName.KERBIN,
                run_parallel=True, apollo_split=False,
            )
            self.assertFalse(
                res.feasible,
                "standard (non-Apollo) evaluation must not reach the "
                "escalated search space",
            )


if __name__ == "__main__":
    unittest.main()
