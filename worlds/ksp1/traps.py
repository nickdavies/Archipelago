"""
Trap item definitions for KSP1 Archipelago.

Traps are filler-pool spice: detrimental-but-quirky items the client mod
actuates in flight (random staging, gravity anomalies, forced EVA, ...).
The AP item NAME is the entire wire signal — there is no slot_data payload;
the client recognizes each name and owns the effect entirely.

This module is the single source of truth for the trap universe: the
``TrapType`` option keys, the player-visible item names, and their stable
id offsets.  Leaf module — imported by items.py and options.py, imports
neither (mirrors contracts.py owning ``ContractType``).
"""
from enum import StrEnum


class TrapType(StrEnum):
    """Option keys for trap weighting (``trap_type_weights``)."""
    STAGING = "staging"
    GRAVITY = "gravity"
    SPIN = "spin"
    COMMS_OUTAGE = "comms_outage"
    POWER_DRAIN = "power_drain"
    OVERHEAT = "overheat"
    PART_FAILURE = "part_failure"
    SURPRISE_EVA = "surprise_eva"
    TIMEWARP = "timewarp"
    THROTTLE = "throttle"
    DEPLOYABLES = "deployables"


# AP item ids are KSP1_BASE_ID + offset.  Offsets are hand-written (never
# enumerate()d) because id stability is a datapackage contract: inserting or
# reordering a member must never shift another trap's id.  Block 300-999 is
# reserved for traps (Discover items own 200-299, parts 1000-1999).  Names
# are player-visible in AP clients and freeze at first release.
TRAP_DEFS: dict[TrapType, tuple[str, int]] = {
    TrapType.STAGING:      ("Trap: Stage Fright", 300),
    TrapType.GRAVITY:      ("Trap: Gravity Storm", 301),
    TrapType.SPIN:         ("Trap: Spin Cycle", 302),
    TrapType.COMMS_OUTAGE: ("Trap: Radio Silence", 303),
    TrapType.POWER_DRAIN:  ("Trap: Short Circuit", 304),
    TrapType.OVERHEAT:     ("Trap: Thermal Runaway", 305),
    TrapType.PART_FAILURE: ("Trap: Loose Bolts", 306),
    TrapType.SURPRISE_EVA: ("Trap: Mandatory Spacewalk", 307),
    TrapType.TIMEWARP:     ("Trap: Time Slip", 308),
    TrapType.THROTTLE:     ("Trap: Sticky Throttle", 309),
    TrapType.DEPLOYABLES:  ("Trap: Minor Kraken Attack", 310),
}

#: Reverse map: item name -> trap type (weights lookup, client-facing docs).
TRAP_NAME_TO_TYPE: dict[str, TrapType] = {
    name: trap_type for trap_type, (name, _offset) in TRAP_DEFS.items()
}

_offsets = [offset for _name, offset in TRAP_DEFS.values()]
assert len(set(_offsets)) == len(_offsets), "trap id offsets must be unique"
assert all(300 <= o <= 999 for o in _offsets), "trap ids live in the 300-999 block"
assert len(TRAP_DEFS) == len(TrapType), "every TrapType needs a TRAP_DEFS entry"
del _offsets
