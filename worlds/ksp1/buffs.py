"""
Buff item definitions for KSP1 Archipelago.

Buffs are filler-pool upside: permanent, additively-stacking stat boosts the
client mod applies to every part (engine efficiency, thrust, heat tolerance,
structural strength, control authority, power generation).
The AP item NAME is the entire wire signal — there is no slot_data payload;
the client recognizes each name and owns the effect entirely.

Buffs are deliberately invisible to the capability/physics model: no buff name
appears in ``DEFAULT_PART_MANAGER.parts``, therefore none is in
``CAPABILITY_ITEMS``, therefore ``world.collect_item`` returns ``None`` for one
and ``state.count("Buff: ...")`` is identically 0.  Logic never sees a buff, so
a buff can never make an out-of-logic mission "reachable" — it only makes an
in-logic mission easier to fly.  (Item *classification* alone would not be
enough: USEFUL-classified parts ARE capability-relevant.)

This module is the single source of truth for the buff universe: the
``BuffType`` / ``BuffTier`` option keys, tier magnitudes, the ``normal``
baseline tier counts, the player-visible item names, and their stable id
offsets.  Leaf module — imported by items.py and options.py, imports neither
(mirrors traps.py owning ``TrapType``).
"""
from collections.abc import Iterable
from enum import StrEnum


class BuffType(StrEnum):
    """Option keys for buff category selection (``buff_types``).

    Declaration order is load-bearing: ``build_buff_pool`` walks types in this
    order when it has to clamp a buff block into a short filler pool, so it is
    also the tie-break order for which type keeps the odd extra copy.
    """
    ISP = "isp"
    THRUST = "thrust"
    HEAT_TOLERANCE = "heat_tolerance"
    STRUCTURAL = "structural"
    CONTROL = "control"
    POWER = "power"


class BuffTier(StrEnum):
    """Magnitude ladder.  Effects stack additively across every copy held."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


#: Percentage-point boost a single copy of each tier grants.  The client mod
#: reads nothing from here — it maps the item name to the same numbers itself
#: and is the AUTHORITY on effect size — but the option docstrings and the
#: player doc quote these to state each density's ceiling, so they must not
#: drift from ``Buffs/BuffDefs.cs``.
BUFF_TIER_PERCENT: dict[BuffTier, int] = {
    BuffTier.SMALL: 1,
    BuffTier.MEDIUM: 3,
    BuffTier.LARGE: 5,
}

#: Per-type overrides of the default ladder.  Structural runs 5/15/25 because
#: the default 1/3/5 was near-worthless there: the fields it scales are small
#: in absolute terms, so a top-tier copy moved an LV-N's impact tolerance
#: 12.0 -> 12.6 m/s.  Percentage (not a flat +N m/s) is kept because
#: crashTolerance is only one of five fields in that buff — breakingForce,
#: breakingTorque, gTolerance and maxPressure are different units, and a flat
#: bonus would make one item grant a mixed-unit effect.
BUFF_TIER_PERCENT_OVERRIDE: dict[BuffType, dict[BuffTier, int]] = {
    BuffType.STRUCTURAL: {BuffTier.SMALL: 5, BuffTier.MEDIUM: 15, BuffTier.LARGE: 25},
}


def tier_percent(buff_type: BuffType, tier: BuffTier) -> int:
    """Percentage a single copy grants, honouring per-type overrides."""
    return BUFF_TIER_PERCENT_OVERRIDE.get(buff_type, BUFF_TIER_PERCENT)[tier]


def type_ceiling(buff_type: BuffType, counts: dict[BuffTier, int]) -> int:
    """Max total percent for a type at the given per-tier copy counts."""
    return sum(tier_percent(buff_type, tier) * n for tier, n in counts.items())

#: Copies of each tier per enabled buff type at the ``normal`` density — the
#: baseline the ``BuffDensity`` option scales around.  Ceiling per type:
#: 3*1 + 2*3 + 1*5 = +14%.  Single source of truth: options.py references this
#: rather than re-typing 3/2/1.
BUFF_TIER_COUNTS: dict[BuffTier, int] = {
    BuffTier.SMALL: 3,
    BuffTier.MEDIUM: 2,
    BuffTier.LARGE: 1,
}

# AP item ids are KSP1_BASE_ID + offset.  Offsets are hand-written (never
# enumerate()d) because id stability is a datapackage contract: inserting or
# reordering a member must never shift another buff's id.
#
# Block 12000-12999 is reserved for buffs.  This world enforces that item and
# location offsets are globally disjoint (test_parts_data.
# test_item_and_location_ranges_no_overlap), so a buff block has to dodge BOTH
# spaces: items hold 0-1999 (special/science/discover/traps, then parts) and
# 10000+ (contracts); locations hold 2000-4199 (legacy buckets), 19000-19099
# (goal thresholds) and 20000-29999 (contract completions).  The obvious-looking
# 2000-2999 collides head-on with the legacy location band, and 5000-9999 is
# only free by accident — the legacy band is already at 4153 of 4199, so the
# next batch of locations has to widen it.  12000+ sits clear of both and of
# any plausible growth in either.
#
# The stride of 10 per type leaves room for a fourth tier without renumbering.
# Names are player-visible in AP clients and freeze at first release.
BUFF_DEFS: dict[tuple[BuffType, BuffTier], tuple[str, int]] = {
    (BuffType.ISP, BuffTier.SMALL):             ("Buff: Engine Efficiency I", 12000),
    (BuffType.ISP, BuffTier.MEDIUM):            ("Buff: Engine Efficiency II", 12001),
    (BuffType.ISP, BuffTier.LARGE):             ("Buff: Engine Efficiency III", 12002),
    (BuffType.THRUST, BuffTier.SMALL):          ("Buff: Engine Thrust I", 12010),
    (BuffType.THRUST, BuffTier.MEDIUM):         ("Buff: Engine Thrust II", 12011),
    (BuffType.THRUST, BuffTier.LARGE):          ("Buff: Engine Thrust III", 12012),
    (BuffType.HEAT_TOLERANCE, BuffTier.SMALL):  ("Buff: Heat Tolerance I", 12020),
    (BuffType.HEAT_TOLERANCE, BuffTier.MEDIUM): ("Buff: Heat Tolerance II", 12021),
    (BuffType.HEAT_TOLERANCE, BuffTier.LARGE):  ("Buff: Heat Tolerance III", 12022),
    (BuffType.STRUCTURAL, BuffTier.SMALL):      ("Buff: Structural Integrity I", 12030),
    (BuffType.STRUCTURAL, BuffTier.MEDIUM):     ("Buff: Structural Integrity II", 12031),
    (BuffType.STRUCTURAL, BuffTier.LARGE):      ("Buff: Structural Integrity III", 12032),
    (BuffType.CONTROL, BuffTier.SMALL):         ("Buff: Control Authority I", 12040),
    (BuffType.CONTROL, BuffTier.MEDIUM):        ("Buff: Control Authority II", 12041),
    (BuffType.CONTROL, BuffTier.LARGE):         ("Buff: Control Authority III", 12042),
    (BuffType.POWER, BuffTier.SMALL):           ("Buff: Power Generation I", 12050),
    (BuffType.POWER, BuffTier.MEDIUM):          ("Buff: Power Generation II", 12051),
    (BuffType.POWER, BuffTier.LARGE):           ("Buff: Power Generation III", 12052),
}

#: Reverse map: item name -> (type, tier).  Used by the pool tests and by any
#: consumer that resolves a received item back to its category/magnitude.
BUFF_NAME_TO_DEF: dict[str, tuple[BuffType, BuffTier]] = {
    name: key for key, (name, _offset) in BUFF_DEFS.items()
}

_offsets = [offset for _name, offset in BUFF_DEFS.values()]
_names = [name for name, _offset in BUFF_DEFS.values()]
assert len(set(_offsets)) == len(_offsets), "buff id offsets must be unique"
assert all(12000 <= o <= 12999 for o in _offsets), "buff ids live in the 12000-12999 block"
assert len(set(_names)) == len(_names), "buff item names must be unique"
assert len(BUFF_DEFS) == len(BuffType) * len(BuffTier), \
    "every (BuffType, BuffTier) pair needs a BUFF_DEFS entry"
assert set(BUFF_TIER_PERCENT) == set(BuffTier), "every tier needs a percentage"
assert set(BUFF_TIER_COUNTS) == set(BuffTier), "every tier needs a baseline count"
del _offsets, _names


#: Tier emit order: strongest first.  Under a short filler budget the block is
#: truncated from the end, so large-first means the ceiling degrades smoothly
#: instead of a seed losing its only +5% copies to keep six +1% ones.
_TIER_ORDER: tuple[BuffTier, ...] = (BuffTier.LARGE, BuffTier.MEDIUM, BuffTier.SMALL)


def build_buff_pool(
    counts: dict[BuffTier, int],
    types: Iterable[BuffType],
    budget: int,
) -> list[str]:
    """Return the buff item names to add to the pool, clamped to ``budget``.

    Pure function of its arguments — deliberately not a world method — so the
    clamp can be unit-tested directly at any budget.  Testing it through a
    built world would tie the test to the seed's location count, which drifts
    every time a location is added.

    Emit order is tier-descending, then *copy-major*, then type.  Copy-major is
    what keeps truncation even: within a tier every enabled type gets its Nth
    copy before any type gets its (N+1)th, so two enabled types can differ by
    at most one copy no matter where the cut lands.  The odd copy goes to the
    earlier ``BuffType`` member, which is why that declaration order is
    load-bearing.

    ``budget`` is the number of filler slots available; a budget at or above
    the full block size returns the whole block, and a non-positive budget
    returns nothing (callers must tolerate the feature silently vanishing in a
    location-starved seed rather than failing generation).
    """
    if budget <= 0:
        return []
    enabled = [t for t in BuffType if t in set(types)]
    if not enabled:
        return []
    out: list[str] = []
    for tier in _TIER_ORDER:
        for _copy in range(counts.get(tier, 0)):
            for buff_type in enabled:
                out.append(BUFF_DEFS[(buff_type, tier)][0])
    return out[:budget]
