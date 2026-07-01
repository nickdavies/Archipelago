"""Deep Space Network (DSN) comms model.

The capability system reasons about the *physical* Deep Space Network power
(``dsn_power``, watts) — a ground-station strength — never a building level.
The translation from Tracking-Station level to ``dsn_power`` lives in
``effects.DSN_POWER_BY_LEVEL``; only the sphere-ladder inverse
(``min_dsn_level_for``) speaks in building levels, because item placement is
inherently level-shaped.

The antenna-tier relay model in ``bodies.py`` (``min_relay_tier`` /
``_RELAY_TIER_AU_THRESHOLDS``) is derived at a *maxed* DSN.  KSP CommNet link
range is ``sqrt(P_antenna * P_dsn)``, so a below-max DSN shrinks reach: to reuse
the max-DSN thresholds we scale the heliocentric separation up by
``sqrt(P_dsn_max / P_dsn)`` before the tier lookup.  At max power the scale is
1.0 and this reduces to today's behaviour (a strict no-op when the option is off).

Layering: imports ``bodies`` (separation + threshold primitives) and ``effects``
(the per-level DSN power table).  ``capability`` and ``rules`` import from here;
nothing here imports them.
"""
import math

from .bodies import ALL_BODIES, BodyName, min_relay_tier
from .effects import DSN_POWER_BY_LEVEL

# Physical DSN power (watts) of a maxed ground station, and the highest
# Tracking-Station level (used only by the level-shaped inverse below).
DSN_POWER_MAX: float = DSN_POWER_BY_LEVEL[-1]
DSN_MAX_LEVEL: int = len(DSN_POWER_BY_LEVEL) - 1


def dsn_sep_scale(dsn_power: float) -> float:
    """Separation multiplier for a DSN power (>= 1.0, and exactly 1.0 at max).

    ``range = sqrt(P_antenna * P_dsn)``, so a ground station of power ``P``
    reaches ``sqrt(P/P_max)`` as far as a maxed one.  Equivalently a target at
    separation ``d`` looks ``sqrt(P_max/P) * d`` away when sizing the antenna
    tier — that factor is returned here.
    """
    p = min(max(dsn_power, DSN_POWER_BY_LEVEL[0]), DSN_POWER_MAX)
    return math.sqrt(DSN_POWER_MAX / p)


# Cache: (home, dsn_power) -> {body: required antenna tier at this DSN power}.
_DSN_REQUIRED_TABLES: dict[tuple[BodyName, float], dict[BodyName, int]] = {}


def dsn_required_relay_table(home: BodyName, dsn_power: float) -> dict[BodyName, int]:
    """``{body: min antenna tier to hold the link from home at dsn_power}``.

    At ``dsn_power == DSN_POWER_MAX`` this equals ``relay_tier_table_for(home)``
    (today's maxed-DSN requirement).  Cached per (home, power) — a handful of
    tables per home — so the hot comms gate stays a flat dict lookup.
    """
    key = (home, dsn_power)
    table = _DSN_REQUIRED_TABLES.get(key)
    if table is None:
        scale = dsn_sep_scale(dsn_power)
        # uncapped=True: a scaled separation past tier-4's reach returns tier 5
        # ("no antenna reaches at this power"), so the gate charges a TS upgrade.
        table = {b.name: min_relay_tier(b.name, home, scale, uncapped=True)
                 for b in ALL_BODIES}
        _DSN_REQUIRED_TABLES[key] = table
    return table


def min_dsn_level_for(body: BodyName, home: BodyName, antenna_tier: int) -> int:
    """Smallest Tracking-Station *level* at which an antenna of ``antenna_tier``
    holds the link to ``body`` from ``home``.

    The sphere-ladder inverse: item placement gates the Nth Tracking-Station copy
    at the sphere that first needs level N, so this speaks in levels (translating
    each to its ``DSN_POWER_BY_LEVEL`` power internally).  Returns 0 for
    home-system bodies and, more generally, when the antenna already reaches at
    the lowest power.  Never exceeds ``DSN_MAX_LEVEL``.
    """
    for level in range(DSN_MAX_LEVEL + 1):
        if dsn_required_relay_table(home, DSN_POWER_BY_LEVEL[level])[body] <= antenna_tier:
            return level
    return DSN_MAX_LEVEL
