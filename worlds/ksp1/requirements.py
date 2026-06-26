"""Unified capability-requirement currency for the sphere system.

This module is the *low-level* half of the requirement model: the **threshold
requirements** that make up a location's capability **signature** and a sphere's
**provisions**. It deliberately depends on nothing but :mod:`ranks` so it can be
imported from anywhere (``sphere_ladder``, ``capability``, ``contracts``) without
an import cycle.

The model has two requirement flavors that live in two layers:

* **Threshold requirements** (here): :class:`Rank` and :class:`Counted`. Each is
  "key must reach >= level". They are what a :class:`Signature` is made of, and
  what the sphere **covering** test compares. They form a partial order under
  element-wise ``<=`` and grow monotonically along the ladder chain.
* **Part requirements** (authoring, in :mod:`contracts`): ``AnyOf`` / ``Part``.
  Those say "a part of this category / this exact part must be deliverable"; the
  sphere builder *resolves* them to a representative part whose ``rank_sig``
  contributes :class:`Rank` requirements. They never reach the covering test —
  they are lowered to :class:`Rank` first.

:class:`Signature` replaces the old split ``(MinimumRanks ranks, dict extras)``
currency: rank axes become :class:`Rank` requirements, counted progressives
(R&D / Pad / PSI / building levels) become :class:`Counted` requirements. One
representation, one covering test, one partial order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

from .ranks import RankAxisKey

# A requirement's canonical key: the dimension it constrains. Two requirements
# with the same key combine by taking the max level (the stronger demand).
ReqKey = tuple


@dataclass(frozen=True)
class Rank:
    """Need rank >= ``level`` on ``axis``.

    **Non-unique provider:** many parts share a rank, so no single part item is
    the unique way to satisfy this — fill placement uses the *alternates escape*
    (the chain selected reps; restrictive fill places them).
    """

    axis: RankAxisKey
    level: int

    unique_provider: bool = False

    @property
    def key(self) -> ReqKey:
        return ("rank", self.axis.value)


@dataclass(frozen=True)
class Counted:
    """Need >= ``level`` copies of counted progressive ``kind`` (R&D / Pad / PSI /
    a building level).

    **Unique provider:** only the Nth copy supplies level N, so placement uses the
    *tier window* — the copy granting level N may sit only where the location does
    not already require level N or higher (never circular).
    """

    kind: str
    level: int

    unique_provider: bool = True

    @property
    def key(self) -> ReqKey:
        return ("counted", self.kind)


@dataclass(frozen=True)
class Item:
    """Need ``has(name)`` — a specific NON-PHYSICS gate item (e.g. a contract
    award item, whose real rule is ``has(award) AND can_deliver``).

    **Unique provider:** only this exact item satisfies it.  **Access-only:** it
    gates a location's *reachability* but is NOT a rocket capability, so no sphere
    *provides* it — it is excluded from the physics covering test (see
    :meth:`Signature.covers`).  Lives in the same Signature as the physics
    thresholds so a single rule-deriver covers physics and non-physics gates
    uniformly (the two-axis model).
    """

    name: str

    unique_provider: bool = True
    level: int = 1  # binary gate; kept for the max-merge in Signature.of

    @property
    def key(self) -> ReqKey:
        return ("item", self.name)


# A threshold requirement: the elements a Signature is built from.
Threshold = Union[Rank, Counted, Item]


@dataclass(frozen=True)
class Signature:
    """A capability signature — the canonical set of threshold requirements a
    location *needs*, or a sphere *provides*.

    One (max) level per key; a key absent from the set means level 0 — i.e.
    **unavailable**, not unconstrained, on the provisions side, and **no demand**
    on the needs side. Stored as a tuple sorted by key so the value is hashable
    and canonical (it keys the bumper's caches, like the old ``MinimumRanks``).
    """

    reqs: tuple[Threshold, ...] = ()

    # ---- construction -----------------------------------------------------

    @classmethod
    def empty(cls) -> "Signature":
        return cls(())

    @classmethod
    def of(cls, reqs: Iterable[Threshold]) -> "Signature":
        """Canonicalize ``reqs`` into a Signature, max-merging duplicate keys."""
        best: dict[ReqKey, Threshold] = {}
        for r in reqs:
            cur = best.get(r.key)
            if cur is None or r.level > cur.level:
                best[r.key] = r
        return cls(tuple(sorted(best.values(), key=lambda r: r.key)))

    # ---- queries ----------------------------------------------------------

    def level_on(self, key: ReqKey) -> int:
        for r in self.reqs:
            if r.key == key:
                return r.level
        return 0

    def rank(self, axis: RankAxisKey) -> int:
        """Required/provided rank on ``axis`` (0 if absent)."""
        return self.level_on(("rank", axis.value))

    def counted(self, kind: str) -> int:
        """Required/provided level of counted progressive ``kind`` (0 if absent)."""
        return self.level_on(("counted", kind))

    @property
    def rank_reqs(self) -> tuple[Rank, ...]:
        return tuple(r for r in self.reqs if isinstance(r, Rank))

    @property
    def counted_reqs(self) -> tuple[Counted, ...]:
        return tuple(r for r in self.reqs if isinstance(r, Counted))

    @property
    def item_reqs(self) -> tuple[Item, ...]:
        return tuple(r for r in self.reqs if isinstance(r, Item))

    def __bool__(self) -> bool:
        return bool(self.reqs)

    def __iter__(self):
        return iter(self.reqs)

    # ---- combination ------------------------------------------------------

    def with_req(self, req: Threshold) -> "Signature":
        """Return a copy with ``req`` max-merged in (no-op if already satisfied)."""
        if self.level_on(req.key) >= req.level:
            return self
        return Signature.of((*self.reqs, req))

    def with_rank(self, axis: RankAxisKey, level: int) -> "Signature":
        return self.with_req(Rank(axis, level))

    def with_counted(self, kind: str, level: int) -> "Signature":
        return self.with_req(Counted(kind, level))

    def with_item(self, name: str) -> "Signature":
        return self.with_req(Item(name))

    def merged_max(self, other: "Signature") -> "Signature":
        """Element-wise max over the union of keys (the cumulative-chain merge)."""
        if not self.reqs:
            return other
        if not other.reqs:
            return self
        return Signature.of((*self.reqs, *other.reqs))

    # ---- the covering test ------------------------------------------------

    def covers(self, need: "Signature") -> bool:
        """True iff ``self`` (a sphere's provisions) satisfies every threshold in
        ``need`` (a location's requirements). Absent key in ``self`` => level 0,
        so any positive demand on an absent key fails the test.
        """
        for r in need.reqs:
            if isinstance(r, Item):
                continue  # access-only gate; no sphere provides it (non-physics)
            if self.level_on(r.key) < r.level:
                return False
        return True
