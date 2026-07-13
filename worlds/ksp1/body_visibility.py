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

from .bodies import ALL_BODIES, BODY_BY_NAME, BodyName, home_system_bodies
from .options import BodyVisibilityMode

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
