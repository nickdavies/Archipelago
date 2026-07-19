"""Shared base for KSP1 world tests.

The KSP1 world runs ``apply_sphere_ladder`` at the end of ``set_rules`` — a
multi-second operation that builds the sphere ladder, demotes spare progressive
copies, installs per-location placement rules, and pins the exact home-orbit
kit to the player's own world via ``options.local_items``. The bulk of the test
suite's runtime is dominated by this step being re-run for every test method.

Most tests don't read any of the sphere-ladder side effects (``world.
_sphere_ladder``, the demoted item classifications, the bootstrap-local KSC
rules, the per-copy tier ban, or ``options.local_items``); they only inspect
``itempool`` or do their own reachability sweeps.

This base stubs ``apply_sphere_ladder`` to a no-op so world setup completes
through the fast gen steps. Tests that depend on sphere-ladder side effects opt
back into the real implementation via ``needs_real_pre_fill = True`` at the
class level, or ``with self.real_pre_fill():`` per test. (The ``pre_fill``
naming is retained for the opt-in API even though the build now happens in
``set_rules``.)
"""
from __future__ import annotations

import contextlib
from unittest.mock import patch

from test.bases import WorldTestBase


class KSP1TestBase(WorldTestBase):
    game = "Kerbal Space Program 1"
    run_default_tests = False  # default; overridden by TestFill et al.

    # Whole-class opt-in: when True, setUp runs the real apply_sphere_ladder
    # immediately after world_setup so every test method sees a fully-built
    # sphere ladder. Use this when most/all tests in the class read
    # ``world._sphere_ladder``, ``local_early_items``, or depend on Rule A/B
    # placement rules. For ad-hoc per-test opt-in, use the ``real_pre_fill``
    # context manager instead.
    needs_real_pre_fill: bool = False

    _real_pre_fill_done: bool = False  # internal: per-test idempotency

    def setUp(self) -> None:
        # Skip the expensive sphere-ladder build during world_setup (it runs in
        # set_rules). Patch the source function — set_rules imports it at call
        # time. Tests that need it opt back in via ``needs_real_pre_fill = True``
        # or the ``real_pre_fill`` context manager.
        with patch(
            "worlds.ksp1.sphere_ladder.apply_sphere_ladder",
            new=lambda world: None,
        ):
            super().setUp()
        self._real_pre_fill_done = False
        # WorldTestBase.world_setup skips construction for default-test
        # method names when run_default_tests is False (e.g. test_fill on a
        # class that didn't opt back in), so self.world may not exist.
        if self.needs_real_pre_fill and hasattr(self, "world"):
            from worlds.ksp1.sphere_ladder import apply_sphere_ladder
            apply_sphere_ladder(self.world)
            self._real_pre_fill_done = True

    @contextlib.contextmanager
    def real_pre_fill(self):
        """Run the real ``apply_sphere_ladder`` on the already-built world.

        Idempotent within a test — calling twice is a no-op the second time.
        Effects are observable until the next test (setUp rebuilds the
        world). Use this for ad-hoc per-test opt-in; for whole-class
        opt-in, set ``needs_real_pre_fill = True`` at the class level.
        """
        if not self._real_pre_fill_done:
            from worlds.ksp1.sphere_ladder import apply_sphere_ladder
            apply_sphere_ladder(self.world)
            self._real_pre_fill_done = True
        yield

    def test_all_state_can_reach_everything(self):
        """KSP1 override of WorldTestBase's default reachability check.

        Model-infeasible missions are no longer emitted at all (the world only
        creates reachable mission locations), so they need no exemption — if one
        ever leaks into the location set this test fails, which is the point.

        The one remaining unreachable-by-design case: contracts with no
        sphere-ladder bracket (and the curated goal-contract proxies) gate on
        the all-parts proxy ``has_all(every part)``, which a location-short pool
        can't satisfy — structurally unreachable filler, not a logic path.  (A
        follow-up will stop emitting these too, closing the last gap for
        ``accessibility=full``.)

        Everything else must still be reachable, and the seed must be beatable.
        """
        if not (self.run_default_tests and self.constructed):
            return
        world = self.multiworld.worlds[self.player]
        # No exemptions.  Every contract now brackets (off its foundational
        # reach kit), infeasible missions aren't emitted, and infeasible goals
        # are rejected — so there is no longer any all-parts / EXCLUDED-but-
        # unreachable location.  The invariant is now strict: every emitted
        # addressed location must be reachable with the full item pool.
        exempt: set[str] = set()
        with self.subTest("Game", game=self.game, seed=self.multiworld.seed):
            state = self.multiworld.get_all_state(False)
            for location in self.multiworld.get_locations():
                if location.name in exempt:
                    continue
                with self.subTest("Location should be reached",
                                  location=location.name):
                    self.assertTrue(location.can_reach(state),
                                    f"{location.name} unreachable")
            with self.subTest("Beatable"):
                self.multiworld.state = state
                self.assertBeatable(True)
