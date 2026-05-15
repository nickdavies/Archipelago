"""Invariant tests for the sphere ladder pre-fill mechanism.

These tests build a world (running gen_steps up through pre_fill) and
check structural properties of the resulting ``SphereLadder``: predictable
spheres are present, each sphere's delta actually unlocks its target
location given the cumulative kit, and Rule B doesn't ban the items
that landed at the sphere boundary itself.
"""
from __future__ import annotations

import unittest

from worlds.ksp1.capability import (
    _pre_pass, evaluate_mission_detailed, _evaluate_sounding,
)
from worlds.ksp1.bodies import DIFFICULTY_PROFILES, MissionType
from worlds.ksp1.sphere_ladder import (
    _parse_location, _strict_less, _reqs_subset,
)
from worlds.ksp1.test.base import KSP1TestBase


class TestPredictableSpheres(KSP1TestBase):
    """The launch / orbit / goal spine must always be present."""
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_launch_and_orbit_present(self) -> None:
        ladder = self.world._sphere_ladder
        names = {s.location_name for s in ladder.spheres}
        self.assertIn("Kerbin First Launch", names,
                      "S_launch must always be in the predictable spine")
        self.assertIn("Kerbin Orbit 1", names,
                      "S_orbit must always be in the predictable spine")

    def test_at_least_one_goal_sphere(self) -> None:
        ladder = self.world._sphere_ladder
        goal_spheres = [s for s in ladder.spheres if s.name.startswith("S_goal")]
        self.assertGreaterEqual(
            len(goal_spheres), 1,
            "At least one goal sphere should anchor the ladder",
        )


class TestDeltaPassesCheck(KSP1TestBase):
    """Every accepted sphere's delta + prior cumulative kit must
    actually make the sphere's target location reachable, per
    ``evaluate_mission_detailed``.  This is the core ladder invariant.
    """
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_each_sphere_satisfies_check(self) -> None:
        ladder = self.world._sphere_ladder
        rep_names = frozenset(
            rep
            for tiers in self.world.progressive_representatives.values()
            for rep in tiers.values()
        )
        difficulty_name = ["casual", "normal", "expert", "insane"][
            self.world.options.difficulty.value
        ]
        diff = DIFFICULTY_PROFILES[difficulty_name]

        precollected_names = frozenset(
            it.name for it in self.multiworld.precollected_items[self.player]
        )

        def _count_fn_factory(kit):
            def fn(name, _k=kit, _pre=precollected_names):
                from worlds.ksp1.sphere_ladder import PROGRESSIVE_CAPS
                if name in PROGRESSIVE_CAPS:
                    return _k.get(name, 0)
                if name in _pre:
                    return 1
                return 0
            return fn

        for sphere in ladder.spheres:
            kit = sphere.rocket.cumulative
            flags = _pre_pass(
                _count_fn_factory(kit),
                start_with_clamps=bool(self.world.options.start_with_launch_clamps),
                rep_names=rep_names,
                progressive_launch_pad=bool(self.world.options.progressive_launch_pad),
            )
            info = _parse_location(sphere.location_name)
            self.assertIsNotNone(info,
                f"Sphere {sphere.name} has unparseable location")
            if info.mission_type == MissionType.SOUNDING:
                result = _evaluate_sounding(flags, info.threshold_km or 0.0)
            else:
                result = evaluate_mission_detailed(
                    flags, diff, info.body, info.mission_type,
                    info.crewed, info.threshold_km,
                )
            self.assertTrue(
                result.feasible,
                f"Sphere {sphere.name} ({sphere.location_name}) infeasible "
                f"with its own cumulative kit; blocking: "
                f"{[str(b) for b in result.blocking[:3]]}",
            )


class TestRuleBNotBannedAtSphereLocation(KSP1TestBase):
    """The items in a sphere's delta should be *accepted* at the sphere
    boundary location itself.  If Rule B bans them there, the ladder
    is self-contradictory.
    """
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_sphere_boundary_accepts_one_of_its_delta_items(self) -> None:
        from worlds.ksp1.items import create_item

        ladder = self.world._sphere_ladder
        for sphere in ladder.spheres:
            if not sphere.rocket.delta:
                continue  # nothing to verify
            loc = self.multiworld.get_location(
                sphere.location_name, self.player,
            )
            # The boundary location does need its delta items to be
            # reachable BEFORE itself, but a location may still ACCEPT
            # such items if they don't form a self-cycle on this location.
            # Practically: at least one delta item should NOT be in the
            # location's own min_kit (banned by per-location self-ban).
            # If all delta items self-ban here, we have an unsatisfiable
            # constraint — the test surfaces that.
            accepts_any = False
            for name in sphere.rocket.delta:
                test_item = create_item(self.world, name)
                if loc.item_rule(test_item):
                    accepts_any = True
                    break
            # Goal locations naturally self-ban their entire delta — the
            # delta IS their min_kit.  That's the design.  Skip goal
            # sphere boundaries; only intermediate/launch/orbit need
            # this check.
            if sphere.name.startswith("S_goal"):
                continue
            self.assertTrue(
                accepts_any,
                f"Sphere {sphere.name} ({sphere.location_name}) bans every "
                f"item in its own delta — circular constraint",
            )


class TestSphereChainOrdered(KSP1TestBase):
    """The ladder is sorted by (group, min_kit_size, dv, ...) where group
    pins S_launch first, S_orbit second, S_goal* last, and intermediates
    in the middle. The semantic invariant we care about: each sphere's
    cumulative kit is a superset of the prior sphere's cumulative kit
    — the chain only grows, never shrinks or contradicts itself.
    """
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_cumulative_kit_monotone(self) -> None:
        ladder = self.world._sphere_ladder
        prev_cum: dict[str, int] = {}
        for s in ladder.spheres:
            cum = s.rocket.cumulative
            for name, prev_count in prev_cum.items():
                self.assertGreaterEqual(
                    cum.get(name, 0), prev_count,
                    f"Sphere {s.name} regressed {name}: "
                    f"prior cumulative had {prev_count}, this sphere has "
                    f"{cum.get(name, 0)} — ladder must only grow.",
                )
            prev_cum = cum

    def test_chain_groups_anchored(self) -> None:
        """S_launch / S_orbit lead; S_goal* always trail; intermediates middle."""
        ladder = self.world._sphere_ladder
        spheres = ladder.spheres
        if not spheres:
            return
        self.assertEqual(spheres[0].name, "S_launch")
        seen_goal = False
        for s in spheres:
            if s.name.startswith("S_goal"):
                seen_goal = True
            elif seen_goal:
                self.fail(
                    f"Non-goal sphere {s.name} appears after a S_goal sphere"
                )


class TestSmallGoalMunFlag(KSP1TestBase):
    """The mun_flag goal has S_goal close to S_orbit; the ladder must
    still construct a valid chain without raising.
    """
    options = {"goal": "mun_flag", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_chain_constructed(self) -> None:
        ladder = self.world._sphere_ladder
        self.assertGreaterEqual(len(ladder.spheres), 2)
        names = {s.location_name for s in ladder.spheres}
        self.assertIn("Kerbin First Launch", names)
        self.assertIn("Kerbin Orbit 1", names)


class TestSphereLadderLocalEarlyItems(KSP1TestBase):
    """Items in S_launch's delta should be registered as
    ``local_early_items`` so AP places them at sphere-0 locations.
    """
    options = {"goal": "duna_return", "difficulty": "normal"}
    needs_real_pre_fill = True

    def test_launch_delta_in_local_early(self) -> None:
        ladder = self.world._sphere_ladder
        launch = next(
            (s for s in ladder.spheres
             if s.location_name == "Kerbin First Launch"),
            None,
        )
        self.assertIsNotNone(launch)
        local_early = self.multiworld.local_early_items[self.player]
        for name, count in launch.rocket.delta.items():
            self.assertGreaterEqual(
                local_early.get(name, 0), count,
                f"{name} (×{count}) from S_launch delta missing from "
                f"local_early_items (got {local_early.get(name, 0)})",
            )


class TestRequirementsSubset(unittest.TestCase):
    """Unit tests for the tier-aware requirements subset comparison."""

    def test_simple_subset(self) -> None:
        a = (("has_capsule", 1),)
        b = (("has_capsule", 1), ("has_solar", 1))
        self.assertTrue(_reqs_subset(a, b))
        self.assertFalse(_reqs_subset(b, a))

    def test_tier_level_comparison(self) -> None:
        low_relay = (("relay_tier", 2),)
        high_relay = (("relay_tier", 4),)
        # Needing tier 2 is a subset of needing tier 4 (4 covers 2's need)
        self.assertTrue(_reqs_subset(low_relay, high_relay))
        self.assertFalse(_reqs_subset(high_relay, low_relay))

    def test_missing_key_fails_subset(self) -> None:
        a = (("relay_tier", 2),)
        b = (("has_solar", 1),)
        self.assertFalse(_reqs_subset(a, b))


if __name__ == "__main__":
    unittest.main()
