"""Plane-change rule for the mission graph.

A transfer pays a (difficulty-scaled) plane change to MATCH an inclined
target's orbit, with ONE exception: descending to the source body's own
direct parent is free — you keep whatever inclination you have and
aerobrake / prop-land from there (the final inclination never matters).

    free  iff  destination body IS the source body's DIRECT parent
    paid  otherwise   (ascending to encounter an inclined body, or any
                       interplanetary transfer — incl. returns home)

Note this is NOT "destination == home" (Moho -> Kerbin still pays: it is an
interplanetary encounter, Kerbin is not Moho's parent) and NOT gated on
atmosphere (that only changes the *descent* edge: ATMO_LANDING vs
VACUUM_LANDING).  The rule is purely the parent relationship of the two
endpoint bodies.

This is subtle and easily regressed in prose, so the invariant is pinned
here for every registered edge plus the specific cases from the design
table.  See ``bodies.MissionBuilder._add_home_return_paths``.
"""
import unittest

from worlds.ksp1.bodies import ALL_BODIES, BodyName, MissionBuilder

_NODE_SUFFIXES = ("_surface", "_low_orbit", "_soi", "_intercept")


def _body_of(node):
    """Map a graph node (``"minmus_low_orbit"``) back to its Body, else None."""
    for suffix in _NODE_SUFFIXES:
        if node.endswith(suffix):
            prefix = node[: -len(suffix)]
            for b in ALL_BODIES:
                if b.name.lower() == prefix:
                    return b
            return None
    return None


class PlaneChangeRuleTest(unittest.TestCase):

    @staticmethod
    def _all_edges(mb):
        for graph in (mb._outbound, mb._return):
            for _src, edges in graph.items():
                for _scheme, edge in edges:
                    yield edge

    @classmethod
    def _pc(cls, mb, src, dst):
        for edge in cls._all_edges(mb):
            if edge.source == src and edge.destination == dst:
                return edge.plane_change_dv
        return None

    def _assert_descents_free(self, home):
        """Every edge whose destination is the source's direct parent must
        carry zero plane change.  Returns the count checked (sanity > 0)."""
        mb = MissionBuilder(home=home)
        checked = 0
        for edge in self._all_edges(mb):
            src_b = _body_of(edge.source)
            dst_b = _body_of(edge.destination)
            if src_b is None or dst_b is None:
                continue
            if src_b.parent is not None and src_b.parent == dst_b.name:
                checked += 1
                self.assertEqual(
                    edge.plane_change_dv, 0.0,
                    msg=(f"{edge.source} -> {edge.destination} descends to its "
                         f"parent {dst_b.name}; plane change must be free, got "
                         f"{edge.plane_change_dv}"),
                )
        return checked

    def test_descent_to_parent_is_always_free(self):
        # Planet-home and moon-home exercise both branches of
        # _add_home_return_paths (home's-moons, foreign-moons, moon-home).
        self.assertGreater(self._assert_descents_free(BodyName.KERBIN), 0)
        self.assertGreater(self._assert_descents_free(BodyName.MINMUS), 0)

    def test_table_rows_kerbin_home(self):
        mb = MissionBuilder(home=BodyName.KERBIN)
        # PAID — ascending to encounter an inclined body / interplanetary.
        self.assertGreater(
            self._pc(mb, "kerbin_low_orbit", "minmus_intercept"), 0,
            "outbound LKO -> Minmus raises to encounter the inclined moon")
        self.assertGreater(
            self._pc(mb, "moho_low_orbit", "kerbin_intercept"), 0,
            "Moho -> Kerbin is an interplanetary encounter (Kerbin is not "
            "Moho's parent), so it pays")
        # FREE — descending to the source body's direct parent.
        self.assertEqual(
            self._pc(mb, "minmus_low_orbit", "kerbin_intercept"), 0.0,
            "Minmus -> Kerbin descends to its parent")
        self.assertEqual(
            self._pc(mb, "laythe_low_orbit", "jool_low_orbit"), 0.0,
            "Laythe -> Jool descends to its parent")

    def test_parent_to_home_moon_pays(self):
        # Moon-home: parent -> home-moon RAISES to encounter the inclined home
        # moon (its parent is Kerbin, not the moon), so it pays.
        mb = MissionBuilder(home=BodyName.MINMUS)
        self.assertGreater(
            self._pc(mb, "kerbin_low_orbit", "minmus_intercept"), 0,
            "Kerbin -> Minmus (Minmus home) raises to encounter Minmus; pays")


if __name__ == "__main__":
    unittest.main()
