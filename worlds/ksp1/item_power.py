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


def _load_item_tiers() -> dict[str, int]:
    raw = pkgutil.get_data("worlds.ksp1", "data/item_tiers.json")
    assert raw is not None, "data/item_tiers.json not found in package"
    return json.loads(raw.decode("utf-8"))


#: {item_name: tier} for items with tier >= 1.  Absent items are tier 0.
ITEM_TIERS: dict[str, int] = _load_item_tiers()
