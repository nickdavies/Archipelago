"""Tests for the DSN / Tracking-Station comms model (``comms.py``).

Pins the physics of the buildings_in_logic Tracking-Station gate: the max DSN
level reproduces the antenna-only relay table (option-off no-op), a weaker DSN
raises the required antenna tier, home-system bodies are never gated, and far
bodies escalate past the best antenna (forcing a TS upgrade) at low DSN.
"""

import math
import unittest

from ..bodies import BodyName, min_relay_tier, relay_tier_table_for
from ..comms import (
    DSN_MAX_LEVEL,
    DSN_POWER_MAX,
    dsn_required_relay_table,
    dsn_sep_scale,
    min_dsn_level_for,
)
from ..effects import DSN_POWER_BY_LEVEL


class TestDsnScale(unittest.TestCase):
    def test_max_power_is_unity(self):
        # At max DSN power the separation is unscaled -> today's behavior.
        self.assertEqual(dsn_sep_scale(DSN_POWER_MAX), 1.0)

    def test_lower_power_scales_up(self):
        prev = dsn_sep_scale(DSN_POWER_MAX)
        for power in sorted(DSN_POWER_BY_LEVEL, reverse=True)[1:]:
            s = dsn_sep_scale(power)
            self.assertGreater(s, prev)
            prev = s

    def test_scale_matches_sqrt_power_ratio(self):
        # range = sqrt(P_antenna * P_dsn) -> separation scale = sqrt(P_max/P).
        for power in DSN_POWER_BY_LEVEL:
            want = math.sqrt(DSN_POWER_BY_LEVEL[-1] / power)
            self.assertAlmostEqual(dsn_sep_scale(power), want)

    def test_clamps_out_of_range(self):
        self.assertEqual(dsn_sep_scale(DSN_POWER_MAX * 10), dsn_sep_scale(DSN_POWER_MAX))
        self.assertEqual(dsn_sep_scale(0.0), dsn_sep_scale(DSN_POWER_BY_LEVEL[0]))


class TestDsnRequiredTable(unittest.TestCase):
    def test_max_power_reproduces_antenna_table_kerbin(self):
        # Option-off path: at max DSN the required tier equals the plain antenna
        # table for a home whose farthest target is within tier-4 reach.
        home = BodyName.KERBIN
        self.assertEqual(
            dsn_required_relay_table(home, DSN_POWER_MAX),
            relay_tier_table_for(home))

    def test_home_system_never_gated(self):
        home = BodyName.KERBIN
        for power in DSN_POWER_BY_LEVEL:
            table = dsn_required_relay_table(home, power)
            self.assertEqual(table[BodyName.MUN], 0)
            self.assertEqual(table[BodyName.MINMUS], 0)

    def test_lower_power_never_lowers_requirement(self):
        home = BodyName.KERBIN
        top = dsn_required_relay_table(home, DSN_POWER_MAX)
        for power in DSN_POWER_BY_LEVEL[:-1]:
            lower = dsn_required_relay_table(home, power)
            for body, tier in top.items():
                self.assertGreaterEqual(lower[body], tier, f"{body} @ {power:.0e}W")

    def test_far_body_exceeds_max_antenna_at_low_power(self):
        # At the lowest DSN, no real antenna (max tier 4) reaches Jool/Eeloo, so
        # the required tier exceeds 4 -> the gate must charge a TS upgrade.
        low = dsn_required_relay_table(BodyName.KERBIN, DSN_POWER_BY_LEVEL[0])
        self.assertGreater(low[BodyName.JOOL], 4)
        self.assertGreater(low[BodyName.EELOO], 4)


class TestMinDsnLevelFor(unittest.TestCase):
    def test_home_system_is_level_zero(self):
        self.assertEqual(min_dsn_level_for(BodyName.MUN, BodyName.KERBIN, 0), 0)

    def test_max_dsn_reaches_with_base_antenna(self):
        # The antenna the mission already needs (min_relay_tier) always reaches
        # by the max DSN level, and the returned level actually satisfies.
        home = BodyName.KERBIN
        for body in (BodyName.DUNA, BodyName.EVE, BodyName.JOOL,
                     BodyName.EELOO, BodyName.MOHO):
            antenna = min_relay_tier(body, home)
            lvl = min_dsn_level_for(body, home, antenna)
            self.assertLessEqual(lvl, DSN_MAX_LEVEL)
            self.assertLessEqual(
                dsn_required_relay_table(home, DSN_POWER_BY_LEVEL[lvl])[body],
                antenna)

    def test_interplanetary_needs_upgrade(self):
        home = BodyName.KERBIN
        lvl = min_dsn_level_for(
            BodyName.JOOL, home, min_relay_tier(BodyName.JOOL, home))
        self.assertGreater(lvl, 0)


class TestEvaRescueGate(unittest.TestCase):
    def test_rescue_and_surface_eva_missions_require_eva(self):
        from ..bodies import MissionType
        from ..capability import MISSION_TYPES_REQUIRING_EVA
        for mt in (MissionType.RESCUE, MissionType.FLAG_PLANT,
                   MissionType.SAMPLE_RETURN):
            self.assertIn(mt, MISSION_TYPES_REQUIRING_EVA)


if __name__ == "__main__":
    unittest.main()
