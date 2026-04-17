"""
Item power tier data for pacing rules.

Tiers are precomputed by scripts/compute_item_tiers.py and stored in
data/item_tiers.json.  Regenerate after any change to parts.json:

    python -m worlds.ksp1.scripts.compute_item_tiers

Tier meanings:
  0 (absent from dict): starter — sounding rockets, suborbital only
  1:                     orbital — Kerbin orbit, maybe Mun flyby
  2:                     interplanetary — Mun landing+, interplanetary capable
"""
from __future__ import annotations

import json
import pkgutil

from .parts import PROGRESSIVE_PART_TIERS


def _load_item_tiers() -> dict[str, int]:
    raw = pkgutil.get_data("worlds.ksp1", "data/item_tiers.json")
    assert raw is not None, "data/item_tiers.json not found in package"
    return json.loads(raw.decode("utf-8"))


# Non-part items that need tier assignments for pacing.
# Science Pack 1/5/10/25 are tier 0 (absent = tier 0).
# Progressive part items are tier 0 (always placeable in early locations)
# except Progressive Heat Shield which is tier 1 (not needed until returns).
_NON_PART_TIERS: dict[str, int] = {
    "Science Pack 50": 1,
    "Science Pack 100": 2,
    "Science Pack 250": 2,
    "Progressive R&D": 2,
    "Progressive Heat Shield": 1,
    # All other progressive part items are tier 0 (absent = tier 0)
}

# Progressive tier floors: non-representative absorbed parts get a minimum
# power tier based on their progressive tier to prevent early placement.
# T1 parts keep their physics tier; T2 → floor 1; T3+ → floor 2.
_PROGRESSIVE_TIER_FLOORS: dict[str, int] = {}
for _tiers in PROGRESSIVE_PART_TIERS.values():
    for _tier_num, _parts in _tiers.items():
        if _tier_num >= 3:
            _floor = 2
        elif _tier_num >= 2:
            _floor = 1
        else:
            continue  # T1 parts keep their physics tier
        for _ksp_name in _parts:
            _PROGRESSIVE_TIER_FLOORS[_ksp_name] = max(
                _PROGRESSIVE_TIER_FLOORS.get(_ksp_name, 0), _floor
            )

#: {item_name: tier} for items with tier >= 1.  Absent items are tier 0.
ITEM_TIERS: dict[str, int] = {**_load_item_tiers(), **_NON_PART_TIERS}
# Apply progressive tier floors (max of physics tier and progressive floor).
for _name, _floor in _PROGRESSIVE_TIER_FLOORS.items():
    ITEM_TIERS[_name] = max(ITEM_TIERS.get(_name, 0), _floor)
