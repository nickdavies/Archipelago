"""Shared base for KSP1 world tests.

The KSP1 world's ``pre_fill`` runs ``apply_sphere_ladder`` — a multi-second
operation that builds the sphere ladder, demotes spare progressive copies,
and installs per-location placement rules. The bulk of the test suite's
runtime is dominated by this step being re-run for every test method.

Most tests don't read any of the sphere-ladder side effects:
``pre_fill`` populates ``world._sphere_ladder``, mutates item
classifications via ``_demote_non_rep_parts``, installs Rule A
(bootstrap-local) on KSC biomes, installs Rule B (per-copy tier ban) on
non-bootstrap locations, and registers ``S_launch.delta`` as
``multiworld.local_early_items``. Tests that only inspect ``itempool``,
``progressive_representatives``, or do their own reachability sweeps
don't need any of that.

This base stubs ``pre_fill`` to a no-op so world setup completes through
the other (fast) gen steps. Tests that depend on sphere-ladder side
effects opt back into the real implementation via ``with self.real_pre_fill():``.
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
        # Skip the expensive sphere-ladder build during world_setup. Tests
        # that need it opt back in via ``needs_real_pre_fill = True`` or
        # the ``real_pre_fill`` context manager.
        with patch(
            "worlds.ksp1.world.KSP1World.pre_fill",
            new=lambda self: None,
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
