#!/usr/bin/env python3
"""
Compute physics-based item power tiers from PART_DB and write data/item_tiers.json.

Run after extract_parts.py whenever parts.json changes:
    cd Archipelago && python -m worlds.ksp1.scripts.compute_item_tiers

Tier meanings:
  0 (starter):        sounding rockets, suborbital only
  1 (orbital):        Kerbin orbit, maybe Mun flyby
  2 (interplanetary): Mun landing+, interplanetary capable
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from worlds.ksp1.parts import (
    PART_DB,
    Engine, FuelTank, SolidBooster, HeatShield,
    Decoupler, MiscEquipment,
)

# ---------------------------------------------------------------------------
# Physical constants (inlined to avoid importing rocket_math)
# ---------------------------------------------------------------------------

G0: float = 9.80665  # standard gravity, m/s²

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

_PAYLOAD = 3.0        # reference payload mass (tonnes)
_KERBIN_G = 9.80665   # Kerbin surface gravity (m/s²)
_MIN_TWR = 1.5        # minimum TWR for surface launch feasibility
_MAX_TANKS = 8        # max tanks in a single stage (KSP symmetry limit)

# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

# Engine/SRB: surface-launch dv (atmospheric Isp, TWR >= 1.5 at Kerbin)
_ENGINE_TIER_1_DV = 1200.0   # below → tier 0
_ENGINE_TIER_2_DV = 3000.0   # above → tier 2

# Vacuum Isp threshold — engines above this are tier 2 regardless of
# surface capability (e.g. Nerv 800s, Wolfhound 380s, Ion 4200s).
_EXCEPTIONAL_VAC_ISP = 370.0

# Tank: fuel_mass / dry_mass ratio
_TANK_TIER_1_RATIO = 4.0
_TANK_TIER_2_RATIO = 7.0

# Equipment provide flags → fixed tier assignments
_EQUIPMENT_HIGH: frozenset[str] = frozenset({
    "docking_port",   # enables staging tier 3
    "fuel_line",      # enables asparagus staging
})

_EQUIPMENT_MEDIUM: frozenset[str] = frozenset({
    "launch_clamp",   # gates interplanetary missions
})

_DECOUPLER_TIER = 2   # multi-staging = high impact


# ---------------------------------------------------------------------------
# Engine scoring — surface-launch dv
# ---------------------------------------------------------------------------

def _score_engine_surface(engine: Engine, all_tanks: list[FuelTank]) -> float:
    """
    Best single-stage dv at Kerbin surface with TWR >= 1.5.

    High-thrust engines can support more tanks before TWR drops below
    the floor, so they naturally score higher. Vacuum-only engines
    (Terrier, Poodle, Nerv) score 0 here — they can't launch.
    """
    if engine.atm_isp <= 0 or engine.atm_thrust <= 0:
        return 0.0

    _log = math.log
    best = 0.0

    for tank in all_tanks:
        if tank.fuel_type != engine.fuel_type:
            continue
        if engine.size_class > tank.size_class:
            continue

        # Max tanks before TWR drops below floor:
        # thrust / ((payload + eng_mass + n*(dry+fuel)) * g) >= MIN_TWR
        # n <= (thrust/(MIN_TWR*g) - payload - eng_mass) / (dry + fuel)
        max_mass = engine.atm_thrust / (_MIN_TWR * _KERBIN_G)
        n_limit = (max_mass - _PAYLOAD - engine.mass) / (tank.dry_mass + tank.fuel_mass)
        if n_limit < 1:
            continue
        n = min(int(n_limit), _MAX_TANKS)

        m_dry = _PAYLOAD + engine.mass + tank.dry_mass * n
        m_wet = m_dry + tank.fuel_mass * n
        dv = engine.atm_isp * G0 * _log(m_wet / m_dry)
        if dv > best:
            best = dv

    return best


# ---------------------------------------------------------------------------
# SRB and tank scoring
# ---------------------------------------------------------------------------

def _score_srb(srb: SolidBooster) -> float:
    isp = srb.atm_isp
    if isp <= 0:
        return 0.0
    m_dry = _PAYLOAD + srb.dry_mass
    m_wet = m_dry + srb.fuel_mass
    if m_dry <= 0 or m_wet <= m_dry:
        return 0.0
    return isp * G0 * math.log(m_wet / m_dry)


def _score_tank(tank: FuelTank) -> float:
    if tank.dry_mass <= 0:
        return 0.0
    return tank.fuel_mass / tank.dry_mass


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _tier_from_engine(engine: Engine, surface_dv: float) -> int:
    if engine.vac_isp >= _EXCEPTIONAL_VAC_ISP:
        return 2
    if surface_dv >= _ENGINE_TIER_2_DV:
        return 2
    if surface_dv >= _ENGINE_TIER_1_DV:
        return 1
    return 0


def _tier_from_dv(dv: float) -> int:
    if dv >= _ENGINE_TIER_2_DV:
        return 2
    if dv >= _ENGINE_TIER_1_DV:
        return 1
    return 0


def _tier_from_tank_ratio(ratio: float) -> int:
    if ratio >= _TANK_TIER_2_RATIO:
        return 2
    if ratio >= _TANK_TIER_1_RATIO:
        return 1
    return 0


def _tier_equipment(equip: MiscEquipment) -> int:
    if equip.provides & _EQUIPMENT_HIGH:
        return 2
    if equip.provides & _EQUIPMENT_MEDIUM:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_item_tiers() -> dict[str, int]:
    """
    Return {item_name: tier} for every item in PART_DB with tier >= 1.

    Items absent from the dict are tier 0 (unrestricted).
    """
    all_tanks: list[FuelTank] = []
    for parts in PART_DB.values():
        for part in parts:
            if isinstance(part, FuelTank):
                all_tanks.append(part)

    tiers: dict[str, int] = {}

    for item_name, parts in PART_DB.items():
        best_tier = 0

        for part in parts:
            if isinstance(part, Engine):
                surface_dv = _score_engine_surface(part, all_tanks)
                tier = _tier_from_engine(part, surface_dv)
            elif isinstance(part, SolidBooster):
                tier = _tier_from_dv(_score_srb(part))
            elif isinstance(part, FuelTank):
                tier = _tier_from_tank_ratio(_score_tank(part))
            elif isinstance(part, Decoupler):
                tier = _DECOUPLER_TIER
            elif isinstance(part, HeatShield):
                tier = 1  # gates aero-capture/landing
            elif isinstance(part, MiscEquipment):
                tier = _tier_equipment(part)
            else:
                tier = 0  # Parachute, LandingLeg — basic function

            if tier > best_tier:
                best_tier = tier

        if best_tier > 0:
            tiers[item_name] = best_tier

    return tiers


def main() -> None:
    tiers = compute_item_tiers()

    out_path = Path(__file__).resolve().parent.parent / "data" / "item_tiers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tiers, f, indent=2, sort_keys=True)

    tier_counts = {0: len(PART_DB) - len(tiers), 1: 0, 2: 0}
    for v in tiers.values():
        tier_counts[v] += 1
    print(f"Wrote {out_path}: {tier_counts[0]} tier 0, {tier_counts[1]} tier 1, {tier_counts[2]} tier 2")


if __name__ == "__main__":
    main()
