"""Deep Space Network (DSN / Tracking Station) comms model.

The antenna-tier relay model in ``bodies.py`` (``min_relay_tier`` /
``_RELAY_TIER_AU_THRESHOLDS``) is derived at a *maxed* DSN ground station.  When
``buildings_in_logic`` gates the Tracking Station, the DSN starts weaker and the
player upgrades it.  KSP CommNet link range is ``sqrt(P_antenna * P_dsn)``, so a
below-max DSN shrinks reach: to reuse the max-DSN thresholds we scale the
heliocentric separation up by ``sqrt(P_dsn_max / P_dsn_level)`` before the tier
lookup.  At the max level the scale is 1.0 and this reduces to today's behavior
(a strict no-op when the option is off).

Layering: imports ``bodies`` (the separation + threshold primitives) and
``effects`` (the per-level DSN power table).  ``capability`` and ``rules`` import
from here; nothing here imports them.
"""
import math

from .bodies import ALL_BODIES, BodyName, min_relay_tier
from .effects import DSN_POWER_BY_LEVEL

# Highest DSN (Tracking Station) level; equals today's maxed ground station.
DSN_MAX_LEVEL: int = len(DSN_POWER_BY_LEVEL) - 1


def dsn_sep_scale(dsn_level: int) -> float:
    """Separation multiplier for a DSN level (>= 1.0, and exactly 1.0 at max).

    ``range = sqrt(P_antenna * P_dsn)``, so a level-``L`` ground station reaches
    ``sqrt(P[L]/P[max])`` as far as a maxed one.  Equivalently, a target at
    separation ``d`` looks ``sqrt(P[max]/P[L]) * d`` away when sizing the antenna
    tier — that factor is returned here.
    """
    lvl = max(0, min(dsn_level, DSN_MAX_LEVEL))
    return math.sqrt(DSN_POWER_BY_LEVEL[DSN_MAX_LEVEL] / DSN_POWER_BY_LEVEL[lvl])


# Cache: (home, dsn_level) -> {body: required antenna tier at this DSN level}.
_DSN_REQUIRED_TABLES: dict[tuple[BodyName, int], dict[BodyName, int]] = {}


def dsn_required_relay_table(home: BodyName, dsn_level: int) -> dict[BodyName, int]:
    """``{body: min antenna tier to hold the link from home at dsn_level}``.

    At ``dsn_level == DSN_MAX_LEVEL`` this equals ``relay_tier_table_for(home)``
    (today's maxed-DSN requirement).  Cached per (home, level) — three cheap
    tables per home — so the hot comms gate stays a flat dict lookup.
    """
    key = (home, dsn_level)
    table = _DSN_REQUIRED_TABLES.get(key)
    if table is None:
        scale = dsn_sep_scale(dsn_level)
        # uncapped=True: a scaled separation past tier-4's reach returns tier 5
        # ("no antenna reaches at this DSN"), so the gate charges a TS upgrade.
        table = {b.name: min_relay_tier(b.name, home, scale, uncapped=True)
                 for b in ALL_BODIES}
        _DSN_REQUIRED_TABLES[key] = table
    return table


def min_dsn_level_for(body: BodyName, home: BodyName, antenna_tier: int) -> int:
    """Smallest DSN (Tracking Station) level at which an antenna of
    ``antenna_tier`` holds the link to ``body`` from ``home``.

    Inverse of ``dsn_required_relay_table`` — the sphere-ladder uses it to gate
    the Nth Tracking-Station item at the sphere that first needs DSN level N.
    Returns 0 for home-system bodies (no relay needed) and, more generally, when
    the antenna already reaches at the lowest DSN.  Never exceeds
    ``DSN_MAX_LEVEL``: at max DSN the base antenna requirement always suffices
    for any body reachable at all.
    """
    for level in range(DSN_MAX_LEVEL + 1):
        if dsn_required_relay_table(home, level)[body] <= antenna_tier:
            return level
    return DSN_MAX_LEVEL
