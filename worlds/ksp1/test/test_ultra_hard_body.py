"""Fire-drill for the "future harder planet" guarantee.

A body whose missions exceed the max-kit capability ceiling must flow
through the feasibility table into graceful exclusion — every one of its
locations marked EXCLUDED (filler-only), goals and contracts routed away,
seed still fillable and beatable.  It must NEVER leave progression-eligible
locations behind a permanently-false access rule (the strand that kills a
seed).

There is no shipped ultra-hard body, so this simulates one by injecting a
full body subtree's ``(body, mission_type)`` missions into
``MODEL_INFEASIBLE_BASE`` — the exact shape ``scripts/generate_feasibility.py``
emits for a real ceiling-exceeding body: it probes every event of every body, and parent
gating cascades an unreachable parent into all-False access on its moons,
so the whole subtree lands in the table together.  The Jool system stands
in for the hypothetical body here.
"""
from __future__ import annotations

from unittest.mock import patch

from BaseClasses import LocationProgressType
from worlds.ksp1.bodies import BodyName
from worlds.ksp1.data.feasibility import MODEL_INFEASIBLE_BASE
from worlds.ksp1.locations import (
    EVENT_BY_NAME, MISSION_LOCATIONS, MissionLocation,
)
from worlds.ksp1.test.base import KSP1TestBase

# The simulated ultra-hard body and its (parent-gated) subtree.
_SYNTH_BODIES = frozenset({
    BodyName.JOOL, BodyName.LAYTHE, BodyName.VALL,
    BodyName.TYLO, BodyName.BOP, BodyName.POL,
})

_SYNTH_MISSIONS = frozenset(
    (ml.body, EVENT_BY_NAME[ml.event].mission_type)
    for ml in MISSION_LOCATIONS if ml.body in _SYNTH_BODIES
)


class TestUltraHardBodyGracefulExclusion(KSP1TestBase):
    # Inherited default tests (fill + the all-reachable/beatable override in
    # KSP1TestBase) are the no-strand assertion: fill must complete and the
    # seed must be beatable with the entire subtree excluded.
    run_default_tests = True
    needs_real_pre_fill = True
    # A goal outside the simulated ban, as a real seed's goal resolution
    # would pick (bodies in the table are filtered from goal candidates).
    options = {"goal": "mun_flag", "starting_body": "kerbin"}

    def setUp(self) -> None:
        patched = {
            diff: {home: names | _SYNTH_MISSIONS
                   for home, names in homes.items()}
            for diff, homes in MODEL_INFEASIBLE_BASE.items()
        }
        # world.py reads the table once during world construction, so the
        # patch only needs to cover setUp.
        with patch("worlds.ksp1.world.MODEL_INFEASIBLE_BASE", patched):
            super().setUp()

    def _world(self):
        return self.multiworld.worlds[self.player]

    def test_subtree_locations_excluded(self) -> None:
        """Every location of the ultra-hard subtree is EXCLUDED and in the
        world's model-infeasible set — fill can only place filler there."""
        world = self._world()
        seen = 0
        for location in self.multiworld.get_locations(self.player):
            ml = MissionLocation.parse(location.name)
            if ml is None or ml.body not in _SYNTH_BODIES:
                continue
            seen += 1
            self.assertEqual(
                location.progress_type, LocationProgressType.EXCLUDED,
                f"{location.name} not EXCLUDED",
            )
            self.assertIn(location.name, world.model_infeasible_locations)
        # The Jool system exposes locations for 5 landable moons + Jool's
        # orbital events; if this is 0 the simulation itself is broken.
        self.assertGreater(seen, 50)

    def test_no_contracts_on_subtree(self) -> None:
        """Contract generation must route away from the infeasible bodies."""
        world = self._world()
        for spec in (*world.contract_specs, *world.goal_contract_specs):
            self.assertNotIn(
                spec.body, _SYNTH_BODIES,
                f"contract {spec.item_name} targets banned body {spec.body}",
            )
