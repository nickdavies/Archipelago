"""Unit tests for the unified capability-requirement currency (`requirements.py`).

These pin the `Signature` semantics the sphere ladder depends on: canonical
max-merge, the covering partial order (absent key = unavailable), monotonic
combination, and hashability for caching.
"""

import unittest

from ..ranks import RankAxisKey
from ..requirements import Counted, Rank, Signature

A = RankAxisKey.LAUNCH_ENGINE
B = RankAxisKey.VAC_ENGINE
RD = "Progressive R&D"
PAD = "Progressive Launch Pad"


class TestSignatureConstruction(unittest.TestCase):
    def test_empty_is_falsy_and_has_no_reqs(self):
        s = Signature.empty()
        self.assertFalse(s)
        self.assertEqual(s.reqs, ())

    def test_of_canonicalizes_and_max_merges_duplicate_keys(self):
        # Two Rank reqs on the same axis collapse to the max level.
        s = Signature.of([Rank(A, 2), Rank(A, 5), Rank(A, 1)])
        self.assertEqual(s.rank(A), 5)
        self.assertEqual(len(s.reqs), 1)

    def test_of_is_order_independent_and_hashable(self):
        s1 = Signature.of([Rank(A, 2), Counted(RD, 1)])
        s2 = Signature.of([Counted(RD, 1), Rank(A, 2)])
        self.assertEqual(s1, s2)
        self.assertEqual(hash(s1), hash(s2))
        # usable as a dict/set key (the bumper caches on it)
        self.assertEqual(len({s1, s2}), 1)

    def test_rank_and_counted_kept_distinct(self):
        s = Signature.of([Rank(A, 3), Counted(PAD, 2), Counted(RD, 1)])
        self.assertEqual(s.rank(A), 3)
        self.assertEqual(s.counted(PAD), 2)
        self.assertEqual(s.counted(RD), 1)
        self.assertEqual(len(s.rank_reqs), 1)
        self.assertEqual(len(s.counted_reqs), 2)


class TestSignatureQueries(unittest.TestCase):
    def test_absent_keys_read_as_zero(self):
        s = Signature.of([Rank(A, 2)])
        self.assertEqual(s.rank(B), 0)
        self.assertEqual(s.counted(RD), 0)


class TestSignatureCombination(unittest.TestCase):
    def test_with_rank_is_a_noop_when_already_satisfied(self):
        s = Signature.of([Rank(A, 3)])
        self.assertIs(s.with_rank(A, 2), s)  # lower demand → unchanged identity
        self.assertEqual(s.with_rank(A, 5).rank(A), 5)

    def test_with_counted_inserts_and_maxes(self):
        s = Signature.empty().with_counted(RD, 1).with_counted(RD, 3)
        self.assertEqual(s.counted(RD), 3)

    def test_merged_max_is_elementwise_max_over_union(self):
        s1 = Signature.of([Rank(A, 2), Counted(PAD, 1)])
        s2 = Signature.of([Rank(A, 4), Rank(B, 1), Counted(RD, 2)])
        m = s1.merged_max(s2)
        self.assertEqual(m.rank(A), 4)
        self.assertEqual(m.rank(B), 1)
        self.assertEqual(m.counted(PAD), 1)
        self.assertEqual(m.counted(RD), 2)

    def test_merged_max_with_empty_returns_other_side(self):
        s = Signature.of([Rank(A, 2)])
        self.assertEqual(Signature.empty().merged_max(s), s)
        self.assertEqual(s.merged_max(Signature.empty()), s)


class TestCovering(unittest.TestCase):
    """`covers` is the sphere-admission test: provisions cover a lesser need."""

    def test_provisions_cover_lesser_or_equal_need(self):
        prov = Signature.of([Rank(A, 3), Counted(RD, 2)])
        self.assertTrue(prov.covers(Signature.of([Rank(A, 2), Counted(RD, 1)])))
        self.assertTrue(prov.covers(Signature.of([Rank(A, 3), Counted(RD, 2)])))

    def test_empty_provisions_cover_only_empty_need(self):
        self.assertTrue(Signature.empty().covers(Signature.empty()))
        self.assertFalse(Signature.empty().covers(Signature.of([Rank(A, 1)])))

    def test_absent_provided_key_is_unavailable_not_unconstrained(self):
        # A sphere that provides nothing on axis B does NOT cover a B demand.
        prov = Signature.of([Rank(A, 5)])
        self.assertFalse(prov.covers(Signature.of([Rank(B, 1)])))
        self.assertFalse(prov.covers(Signature.of([Counted(RD, 1)])))

    def test_higher_demand_than_provided_fails(self):
        prov = Signature.of([Rank(A, 2), Counted(PAD, 1)])
        self.assertFalse(prov.covers(Signature.of([Rank(A, 3)])))
        self.assertFalse(prov.covers(Signature.of([Counted(PAD, 2)])))

    def test_covering_is_reflexive_and_monotone_along_merge(self):
        need = Signature.of([Rank(A, 2), Counted(RD, 1)])
        self.assertTrue(need.covers(need))  # reflexive
        # Anything that covers `need` still covers it after we strengthen
        # provisions by merging more in (monotonicity of the chain).
        prov = need.merged_max(Signature.of([Rank(B, 4)]))
        self.assertTrue(prov.covers(need))


if __name__ == "__main__":
    unittest.main()
