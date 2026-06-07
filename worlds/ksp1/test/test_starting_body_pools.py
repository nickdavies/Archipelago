"""
Tests for the pre-canned StartingBody pool options (atmospheric,
standard, planets, all).

Pool keys are resolved at generation time to a concrete body via the
seed RNG.  After resolution every downstream consumer must see a
single body — slot_data, ``mission_builder.home``, the option value
itself — so a regen from slot_data reproduces the same world even
though the original yaml asked for a random pool.
"""
import unittest

from test.general import setup_multiworld

from worlds.ksp1.bodies import BodyName
from worlds.ksp1.options import STARTING_BODY_POOLS, StartingBody
from worlds.ksp1.world import KSP1World


_STEPS = ("generate_early",)


def _gen(seed: int, starting_body: str, goal: str = "complete_tech_tree") -> KSP1World:
    mw = setup_multiworld(
        KSP1World,
        steps=_STEPS,
        seed=seed,
        options={"starting_body": starting_body, "goal": goal},
    )
    return mw.worlds[1]


class TestPoolContents(unittest.TestCase):
    def test_all_pool_excludes_eve(self):
        # The user's hard rule: opt-in Full Random must never auto-pick Eve.
        self.assertNotIn(BodyName.EVE, STARTING_BODY_POOLS["all"])

    def test_all_pool_has_every_other_landable_body(self):
        from worlds.ksp1.bodies import ALL_BODIES
        expected = {
            BodyName(b.name) for b in ALL_BODIES
            if b.can_land and b.name != BodyName.EVE
        }
        self.assertEqual(STARTING_BODY_POOLS["all"], expected)

    def test_pool_keys_match_option_names(self):
        # Every pool key must have a matching option_<key> on StartingBody,
        # otherwise resolution can't reach the pool entry from yaml.
        for key in STARTING_BODY_POOLS:
            self.assertTrue(
                hasattr(StartingBody, f"option_{key}"),
                f"StartingBody.option_{key} missing for pool {key!r}",
            )

    def test_pool_keys_disjoint_from_concrete_body_keys(self):
        concrete = {
            name.replace("option_", "")
            for name in vars(StartingBody)
            if name.startswith("option_")
        } - set(STARTING_BODY_POOLS)
        self.assertTrue(concrete.isdisjoint(set(STARTING_BODY_POOLS)))


class TestPoolResolution(unittest.TestCase):
    def test_each_pool_resolves_to_a_member(self):
        for key, pool in STARTING_BODY_POOLS.items():
            with self.subTest(pool=key):
                w = _gen(seed=1, starting_body=key)
                self.assertIn(w.mission_builder.home, pool)

    def test_resolution_is_deterministic(self):
        # Same seed + same pool key -> same picked body.
        for key in STARTING_BODY_POOLS:
            with self.subTest(pool=key):
                a = _gen(seed=12345, starting_body=key)
                b = _gen(seed=12345, starting_body=key)
                self.assertEqual(a.mission_builder.home, b.mission_builder.home)

    def test_option_value_mutated_to_concrete_body(self):
        # After generate_early the option must point at a concrete body
        # option, so any code reading ``options.starting_body.current_key``
        # post-resolution sees a normal body key.
        w = _gen(seed=42, starting_body="atmospheric")
        key = w.options.starting_body.current_key
        self.assertNotIn(key, STARTING_BODY_POOLS)
        self.assertEqual(BodyName(key.title()), w.mission_builder.home)

    def test_concrete_key_unaffected(self):
        # Regression: passing a concrete body key still routes straight
        # to that body without going through pool resolution.
        w = _gen(seed=42, starting_body="laythe")
        self.assertEqual(w.mission_builder.home, BodyName.LAYTHE)


class TestPoolSlotDataRoundTrip(unittest.TestCase):
    def test_slot_data_emits_resolved_body(self):
        # slot_data must carry the picked body, never the pool key,
        # so the C# client and UT regen both see a single body.
        w = _gen(seed=7, starting_body="planets", goal="complete_tech_tree")
        sd = w.fill_slot_data()
        self.assertEqual(sd["starting_body"], w.mission_builder.home.value)
        self.assertIn(BodyName(sd["starting_body"]), STARTING_BODY_POOLS["planets"])


class TestKscSiteSlotData(unittest.TestCase):
    """The ``ksc_site`` row carries the chosen body's landing coordinate so
    the C# client doesn't need a per-body table (the table lives only in
    ``ksc_sites.py`` now)."""

    def test_alien_start_emits_matching_site_row(self):
        from worlds.ksp1.ksc_sites import KSC_SITES
        w = _gen(seed=7, starting_body="duna")
        sd = w.fill_slot_data()
        self.assertIn("ksc_site", sd)
        lat, lon, alt, skip = KSC_SITES[BodyName.DUNA]
        self.assertEqual(
            sd["ksc_site"],
            {"lat": lat, "lon": lon, "terrain_alt": alt, "skip_decal": skip},
        )

    def test_kerbin_start_omits_site_row(self):
        # Kerbin uses the stock KSC — no row, so the client knows to leave
        # stock alone.
        w = _gen(seed=7, starting_body="kerbin")
        sd = w.fill_slot_data()
        self.assertNotIn("ksc_site", sd)

    def test_every_landable_body_has_a_site(self):
        # Any body the StartingBody option can resolve to (every landable
        # body except gas giants / the sun) must have a site row, or an
        # alien start would emit no ksc_site and the client would reject it.
        from worlds.ksp1.bodies import ALL_BODIES
        from worlds.ksp1.ksc_sites import KSC_SITES
        for b in ALL_BODIES:
            if b.can_land and b.name != BodyName.KERBIN:
                self.assertIn(
                    BodyName(b.name), KSC_SITES,
                    f"{b.name} is landable but has no KSC site row",
                )


if __name__ == "__main__":
    unittest.main()
