"""
Buff item definitions for KSP1 Archipelago.

Buffs are filler-pool upside, in two flavours:

* PERMANENT (``BuffType`` x ``BuffTier``) — additively-stacking stat boosts the
  client mod applies to every part (engine efficiency, thrust, heat tolerance,
  structural strength, control authority, power generation).  Received once,
  active forever.
* CONSUMABLE (``ConsumableType``) — a one-shot charge the player spends from
  the mod UI when they choose.  Each copy received is one charge; spending it
  burns it for good.

The AP item NAME is the entire wire signal — there is no slot_data payload;
the client recognizes each name and owns the effect entirely.

Buffs of both flavours are deliberately invisible to the capability/physics
model: no buff name appears in ``DEFAULT_PART_MANAGER.parts``, therefore none is in
``CAPABILITY_ITEMS``, therefore ``world.collect_item`` returns ``None`` for one
and ``state.count("Buff: ...")`` is identically 0.  Logic never sees a buff, so
a buff can never make an out-of-logic mission "reachable" — it only makes an
in-logic mission easier to fly.  (Item *classification* alone would not be
enough: USEFUL-classified parts ARE capability-relevant.)

This module is the single source of truth for the buff universe: the
``BuffType`` / ``BuffTier`` / ``ConsumableType`` option keys, tier magnitudes,
the per-density copy counts, the player-visible item names, and their stable id
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


class ConsumableType(StrEnum):
    """Option keys for one-shot buff selection.

    Consumables share the ``buff_types`` option set with ``BuffType`` — one
    roster of names the player enables or disables — so adding a member here
    widens that option automatically.  They have no tier ladder: a copy is a
    charge, and density alone sets how many charges a run holds.

    Declaration order is load-bearing for the same reason as ``BuffType``:
    ``build_consumable_pool`` walks types in this order when clamping into a
    short filler pool, so it is the tie-break order for the odd extra copy.
    """
    REFUEL = "refuel"


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

#: Charges of each enabled consumable type per ``BuffDensity`` rung.  Keyed by
#: the option's key NAME, not its int value: this module is a leaf and cannot
#: import options.py, and the names are the stable half of that contract (an
#: int value could be renumbered).  ``BuffDensity.consumable_count`` looks the
#: rung up via ``current_key``, so options.py never re-types these numbers; a
#: test pins the two key sets together so a new rung cannot be added on one
#: side only.
#:
#: A consumable copy is a whole charge with no tier ladder to soften it, so the
#: ladder is flatter than the permanent block's: heavy is 5 charges per type,
#: not 5+3+2 items.
CONSUMABLE_DENSITY_COUNTS: dict[str, int] = {
    "none": 0,
    "light": 1,
    "normal": 3,
    "heavy": 5,
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
#
# Inside that block the permanent ladder holds 12000-12099 and consumables hold
# 12100-12199 (asserted below, so a member landing in the wrong sub-band fails
# at import rather than silently interleaving).  If either outgrows its hundred
# the next sub-band opens at 12200 — ids never move to make room.
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

#: One-shot buffs.  Same id-stability contract and same "the name IS the wire
#: signal" contract as the permanent block: the client maps the name to the
#: effect, grants one spendable charge per copy received, and burns the charge
#: when the player activates it.  Offsets stride 10 like the permanent block so
#: a consumable that later grows a tier ladder needs no renumbering.
CONSUMABLE_DEFS: dict[ConsumableType, tuple[str, int]] = {
    ConsumableType.REFUEL: ("Buff: Mid-Air Refuel", 12100),
}

#: Reverse map: item name -> consumable type.
CONSUMABLE_NAME_TO_TYPE: dict[str, ConsumableType] = {
    name: key for key, (name, _offset) in CONSUMABLE_DEFS.items()
}

_offsets = [offset for _name, offset in BUFF_DEFS.values()]
_names = [name for name, _offset in BUFF_DEFS.values()]
_c_offsets = [offset for _name, offset in CONSUMABLE_DEFS.values()]
_c_names = [name for name, _offset in CONSUMABLE_DEFS.values()]
assert len(set(_offsets)) == len(_offsets), "buff id offsets must be unique"
assert all(12000 <= o <= 12099 for o in _offsets), "permanent buff ids live in 12000-12099"
assert len(set(_names)) == len(_names), "buff item names must be unique"
assert len(BUFF_DEFS) == len(BuffType) * len(BuffTier), \
    "every (BuffType, BuffTier) pair needs a BUFF_DEFS entry"
assert set(BUFF_TIER_PERCENT) == set(BuffTier), "every tier needs a percentage"
assert set(BUFF_TIER_COUNTS) == set(BuffTier), "every tier needs a baseline count"
assert len(set(_c_offsets)) == len(_c_offsets), "consumable id offsets must be unique"
assert all(12100 <= o <= 12199 for o in _c_offsets), "consumable ids live in 12100-12199"
assert len(set(_c_names)) == len(_c_names), "consumable item names must be unique"
assert len(CONSUMABLE_DEFS) == len(ConsumableType), \
    "every ConsumableType needs a CONSUMABLE_DEFS entry"
assert not set(_offsets) & set(_c_offsets), "permanent and consumable buff ids must not collide"
assert not set(_names) & set(_c_names), "permanent and consumable buff names must not collide"
del _offsets, _names, _c_offsets, _c_names


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


def build_consumable_pool(
    count_per_type: int,
    types: Iterable[ConsumableType],
    budget: int,
) -> list[str]:
    """Return the consumable item names to add to the pool, clamped to ``budget``.

    Same contract as ``build_buff_pool`` — pure, so the clamp is unit-testable
    at any budget without tying the test to a seed's location count — and the
    same copy-major emit order, so truncation stays even across enabled types
    (every type gets its Nth charge before any gets its (N+1)th, leaving at
    most one charge of spread wherever the cut lands).  The odd charge goes to
    the earlier ``ConsumableType`` member.

    There is no tier dimension to order by: every copy of a given type is the
    same charge, so the block is a flat ``count_per_type`` charges per enabled
    type.  A non-positive budget returns nothing — this runs on the REMAINING
    filler budget after the permanent block took its slice, so a
    location-starved seed must lose the feature silently, not fail generation.
    """
    if budget <= 0 or count_per_type <= 0:
        return []
    enabled = [t for t in ConsumableType if t in set(types)]
    if not enabled:
        return []
    out: list[str] = []
    for _copy in range(count_per_type):
        for consumable in enabled:
            out.append(CONSUMABLE_DEFS[consumable][0])
    return out[:budget]
