"""Unit tests for the capability-effects translation layer (`effects.py`).

These pin the behavior-preservation contract for the launch-pad mass cap (the
only effect consumed by capability today) and the inverse seam used by the
sphere layer.
"""

import math
import unittest

from ..bodies import BodyName, progressive_launch_pad_caps_for
from ..effects import (
    Building,
    Effect,
    building_effects,
    max_effects,
    min_building_level_for,
    pad_mass_limit_from_caps,
)

HOMES = (BodyName.KERBIN, BodyName.MUN, BodyName.DUNA)


class TestLaunchPadIdentity(unittest.TestCase):
    """building_effects(LAUNCH_PAD, level) == raw cap table entry."""

    def test_pad_level_identity_all_levels(self):
        for home in HOMES:
            caps = progressive_launch_pad_caps_for(home)
            for level in range(len(caps)):
                got = building_effects(Building.LAUNCH_PAD, level, home=home)
                self.assertEqual(
                    got[Effect.PAD_MASS_LIMIT], caps[level],
                    f"home={home} level={level}")

    def test_pad_level_clamps_above_top(self):
        # A level beyond the table clamps to the top (infinite) cap.
        for home in HOMES:
            caps = progressive_launch_pad_caps_for(home)
            got = building_effects(Building.LAUNCH_PAD, len(caps) + 5, home=home)
            self.assertEqual(got[Effect.PAD_MASS_LIMIT], caps[-1])

    def test_pad_helper_matches_building_effects(self):
        # The thin caps-tuple helper used by capability._pre_pass produces the
        # same value as the full building_effects path.
        for home in HOMES:
            caps = progressive_launch_pad_caps_for(home)
            for level in range(len(caps) + 2):
                self.assertEqual(
                    pad_mass_limit_from_caps(caps, level),
                    building_effects(Building.LAUNCH_PAD, level, home=home)[Effect.PAD_MASS_LIMIT])


class TestMaxEffects(unittest.TestCase):
    def test_pad_mass_limit_is_top_cap(self):
        for home in HOMES:
            caps = progressive_launch_pad_caps_for(home)
            eff = max_effects(home)
            self.assertIn(Effect.PAD_MASS_LIMIT, eff)
            self.assertEqual(eff[Effect.PAD_MASS_LIMIT], caps[-1])

    def test_max_effects_covers_every_effect(self):
        eff = max_effects(BodyName.KERBIN)
        for member in Effect:
            self.assertIn(member, eff, f"max_effects missing {member}")


class TestMinBuildingLevelFor(unittest.TestCase):
    def test_pad_returns_cheapest_level_meeting_threshold(self):
        home = BodyName.KERBIN
        caps = progressive_launch_pad_caps_for(home)
        for target in range(len(caps)):
            want_cap = caps[target]
            if want_cap == float("inf"):
                continue
            building, level = min_building_level_for(
                Effect.PAD_MASS_LIMIT, want_cap, home=home)
            self.assertIs(building, Building.LAUNCH_PAD)
            # Cheapest level whose cap >= the threshold.
            self.assertLessEqual(level, target)
            self.assertGreaterEqual(caps[level], want_cap)
            if level > 0:
                self.assertLess(caps[level - 1], want_cap)

    def test_pad_below_first_cap_is_level_zero(self):
        home = BodyName.KERBIN
        caps = progressive_launch_pad_caps_for(home)
        building, level = min_building_level_for(
            Effect.PAD_MASS_LIMIT, caps[0] - 1.0, home=home)
        self.assertEqual((building, level), (Building.LAUNCH_PAD, 0))

    def test_pad_unreachable_threshold_returns_max_level(self):
        home = BodyName.KERBIN
        caps = progressive_launch_pad_caps_for(home)
        # Nothing exceeds infinity; ask for more than any finite cap and below
        # infinity -> the inf level satisfies, returns that level.
        building, level = min_building_level_for(
            Effect.PAD_MASS_LIMIT, math.inf, home=home)
        self.assertIs(building, Building.LAUNCH_PAD)
        self.assertEqual(caps[level], math.inf)

    def test_vessel_mass_inverse(self):
        building, level = min_building_level_for(
            Effect.VESSEL_MASS_LIMIT, 100.0, home=BodyName.KERBIN)
        self.assertIs(building, Building.VAB)
        # 30t (lvl0) < 100 <= 140t (lvl1)
        self.assertEqual(level, 1)


class TestEnumMembers(unittest.TestCase):
    def test_effect_members(self):
        names = {e.name for e in Effect}
        self.assertEqual(names, {
            "PAD_MASS_LIMIT", "CAN_EVA", "DSN_POWER",
            "VESSEL_MASS_LIMIT", "VESSEL_PART_LIMIT", "CREW_RANK",
        })

    def test_building_members(self):
        names = {b.name for b in Building}
        self.assertEqual(names, {
            "LAUNCH_PAD", "VAB", "SPH", "TRACKING_STATION", "ASTRONAUT_COMPLEX",
        })


if __name__ == "__main__":
    unittest.main()
