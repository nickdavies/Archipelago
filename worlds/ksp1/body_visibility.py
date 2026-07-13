"""
Hidden / unlockable celestial bodies — visibility resolution.

Pure derivation for the BodyVisibilityMode feature: resolve the option
(including its ``auto`` default) to a concrete mode, and from that derive which
bodies start hidden and which of those need a ``Discover`` gate + item.

These are functions of (mode, home, goal_spec, owning-set) only — no AP-region
or item knowledge — so ``world.generate_early`` composes them with the region
topology (regions.py) and item pool (items.py).  The home body and the Sun are
never hidden.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, home_system_bodies, science_budget,
)
from .options import BodyVisibilityMode
from .tech_tree import TECH_NODES, cumulative_tier_cost

if TYPE_CHECKING:
    from .rules import GoalSpec


def resolve_visibility_mode(
    raw_value: int, goal_spec: "GoalSpec", home: BodyName
) -> int:
    """Resolve BodyVisibilityMode to a concrete option value.

    ``auto`` (the default) picks by goal reach: a goal confined to the home
    system hides even the home system (``home_only``) so the player discovers
    their own neighbourhood progressively; any wider goal hides only what lies
    outside it (``home_system``).  Explicit modes pass through unchanged.
    """
    if raw_value != BodyVisibilityMode.option_auto:
        return raw_value
    if goal_spec.is_home_system_only(home):
        return BodyVisibilityMode.option_home_only
    return BodyVisibilityMode.option_home_system


def hidden_bodies(resolved_mode: int, home: BodyName) -> frozenset[BodyName]:
    """Bodies hidden at start under a resolved (concrete) visibility mode.

    ``all_visible`` (and an unresolved ``auto``, defensively) hides nothing.
    ``home_system`` keeps the home's whole local system visible; ``home_only``
    keeps only the home body.  The home body and the Sun are never hidden.
    """
    if resolved_mode == BodyVisibilityMode.option_home_system:
        visible = home_system_bodies(home)
    elif resolved_mode == BodyVisibilityMode.option_home_only:
        visible = frozenset({home})
    else:
        return frozenset()
    return frozenset(
        b.name for b in ALL_BODIES
        if b.name not in visible
        and b.name != home
        and b.name != BodyName.KERBOL
    )


def gated_hidden_bodies(
    hidden: frozenset[BodyName], owning: frozenset[BodyName]
) -> list[BodyName]:
    """Hidden bodies that need a ``Discover`` gate + item.

    A body qualifies if it owns a location, or if it (transitively) parents a
    hidden body that owns one — so a hidden planet whose only located children
    are hidden moons still gets its gate (the moons hang off its region).  A
    hidden body that gates nothing is pruned, keeping the item count down.
    Returned in stable ``ALL_BODIES`` order.
    """
    def qualifies(body: BodyName) -> bool:
        if body in owning:
            return True
        for descendant in hidden:
            if descendant not in owning:
                continue
            cur = BODY_BY_NAME[descendant].parent
            while cur is not None:
                if cur == body:
                    return True
                cur = BODY_BY_NAME[cur].parent
        return False

    return [b.name for b in ALL_BODIES if b.name in hidden and qualifies(b.name)]


def _full_kit_science(body_name: BodyName, home: BodyName) -> float:
    """Full-kit science a single body can bank (pure ``science_budget``, no
    capability needed)."""
    b = BODY_BY_NAME[body_name]
    return science_budget(
        b, has_thermometer=True, has_barometer=True, has_capsule=True,
        can_land_crewed=b.can_land, home=home, psi_tier=3,
        can_land_uncrewed=b.can_land)


def hidden_planets(gated_hidden: frozenset[BodyName]) -> list[BodyName]:
    """The gated hidden bodies that are planets (no parent) — the bodies whose
    ``Discover`` item is obtainable independently (a moon's needs its planet's,
    via region topology, so a moon Discover alone yields no bankable science)."""
    return [b for b in gated_hidden if BODY_BY_NAME[b].parent is None]


def tech_gate_reqs_by_tier(
    home: BodyName, safety: float, hidden: frozenset[BodyName],
    planet_order: list[BodyName],
) -> dict[int, int]:
    """How deep into ``planet_order`` each science-insufficient tier reaches.

    Returns ``{tier: K}``: for each tier whose cumulative science cost exceeds
    what the *visible* bodies (reachable without any Discover) can bank, ``K`` is
    the length of the shortest prefix of ``planet_order`` whose full-kit science
    covers the tier's gap.  regions.py gates that tier on ``has_all(order[:K])``
    — the tree's top is reachable only once those specific planets are found.

    ``planet_order`` is a per-seed random permutation of the hidden planets (the
    caller shuffles it), so requiring ``order[:K]`` picks a specific random
    subset — each seed forces a different tour (variance), while the un-required
    planets and every moon float free as bonus discoveries.  The gap is covered
    by the actual chosen planets (exact, since we know which are required).
    Empty when nothing is hidden or the visible set self-funds the whole tree.
    """
    if not planet_order:
        return {}
    visible_science = safety * sum(
        _full_kit_science(b.name, home)
        for b in ALL_BODIES
        if b.name not in hidden and b.name != BodyName.KERBOL
    )
    order_sci = [safety * _full_kit_science(p, home) for p in planet_order]
    reqs: dict[int, int] = {}
    for tier in {n.tier for n in TECH_NODES}:
        gap = cumulative_tier_cost(tier) - visible_science
        if gap <= 0:
            continue
        acc = 0.0
        k = 0
        for s in order_sci:
            if acc >= gap:
                break
            acc += s
            k += 1
        reqs[tier] = k
    return reqs
