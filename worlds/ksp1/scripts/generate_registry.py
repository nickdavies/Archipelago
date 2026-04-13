#!/usr/bin/env python3
"""Generate PART_REGISTRY entries from parts.json.

Run from Archipelago/:
    python worlds/ksp1/scripts/generate_registry.py

Outputs Python code for the PART_REGISTRY list to stdout.
Paste the output into parts.py, replacing the existing PART_REGISTRY.

Stable offset assignment:
  - First run: assigns offsets 1000, 1001, ... alphabetically by ap_item name.
  - Re-run: reads existing PART_REGISTRY from parts.py to preserve assigned
    offsets.  New parts get max(existing) + 1.
"""

import json
import re
import sys
from pathlib import Path


PARTS_RANGE_START = 1000

# ---------------------------------------------------------------------------
# Skip list: entries in parts.json that aren't placeable rocket parts
# ---------------------------------------------------------------------------

SKIP = {
    "_meta",
    # EVA kerbals (not buildable parts)
    "kerbalEVA", "kerbalEVAfemale", "kerbalEVASlimSuit", "kerbalEVASlimSuitFemale",
    # Empty title
    "flag",
}

# ---------------------------------------------------------------------------
# Force overrides: cfg_name → part_type string
# These override the auto-detection logic.
# ---------------------------------------------------------------------------

FORCE_TYPE: dict[str, str] = {
    # Twin-Boar: integrated engine+tank, classify as Engine (tank is bonus)
    "Size2LFB_v2": "Engine",
    # Heat shields that have has_module_decouple (would be misclassified)
    "HeatShield0": "HeatShield",
    "InflatableHeatShield": "HeatShield",
    # Jet engines: need IntakeAir, irrelevant for space missions
    "JetEngine": "MiscEquipment",
    "miniJetEngine": "MiscEquipment",
    "turboFanEngine": "MiscEquipment",
    "turboJet": "MiscEquipment",
    "turboFanSize2": "MiscEquipment",
    # RAPIER: JSON only captures air-breathing mode, not rocket mode
    "RAPIER": "MiscEquipment",
    # Landing gear (wheels, not landing legs)
    "GearFixed": "MiscEquipment",
    "GearFree": "MiscEquipment",
    "GearLarge": "MiscEquipment",
    "GearMedium": "MiscEquipment",
    "GearSmall": "MiscEquipment",
    "SmallGearBay": "MiscEquipment",
    # Hardpoints/pylons have decouple module but aren't functional decouplers
    "smallHardpoint": "MiscEquipment",
    "structuralPylon": "MiscEquipment",
    # EVA parachute: not a launchable parachute
    "evaChute": "MiscEquipment",
}

# ---------------------------------------------------------------------------
# Parts with resources that should NOT be classified as FuelTank.
# These have resources but their primary function is something else.
# ---------------------------------------------------------------------------

NOT_FUEL_TANK: set[str] = {
    # Cockpits/pods/cabins (primary function: crew command)
    "mk1pod_v2", "mk1-3pod", "Mark1Cockpit", "Mark2Cockpit",
    "mk2Cockpit_Standard", "mk2Cockpit_Inline", "mk3Cockpit_Shuttle",
    "landerCabinSmall", "mk2LanderCabin_v2", "cupola",
    # Drone core
    "mk2DroneCore",
    # Probe cores (ElectricCharge only)
    "HECS2_ProbeCore", "probeCoreCube", "probeCoreHex_v2",
    "probeCoreOcto2_v2", "probeCoreOcto_v2", "probeCoreSphere_v2",
    "probeStackLarge", "probeStackSmall", "roverBody_v2",
    # Batteries (ElectricCharge only)
    "batteryBank", "batteryBankLarge", "batteryBankMini",
    "batteryPack", "ksp_r_largeBatteryPack",
    # Fuel cells (ElectricCharge only)
    "FuelCell", "FuelCellArray",
    # Intakes (IntakeAir)
    "CircularIntake", "IntakeRadialLong", "MK1IntakeFuselage",
    "airScoop", "miniIntake", "ramAirIntake", "shockConeIntake",
    # Engine nacelles with IntakeAir
    "nacelleBody", "radialEngineBody",
    # Ore tanks
    "LargeTank", "SmallTank", "RadialOreTank",
    # Mixed-resource probes
    "MpoProbe", "MtmStage",
    # Docking port with monoprop (primary function: docking)
    "mk2DockingPort",
    # Wings with LF (primary function: wing)
    "airlinerMainWing", "wingShuttleDelta", "wingShuttleStrake",
    # EVA gear
    "evaCylinder", "evaJetpack",
    # Goliath turbofan (IntakeAir resource)
    "turboFanSize2",
    # Heat shields (Ablator resource, classified as HeatShield separately)
    "HeatShield0", "HeatShield1", "HeatShield2", "HeatShield3",
    "InflatableHeatShield",
    # SRBs (SolidFuel, classified as SolidBooster separately)
    "Clydesdale", "Mite", "Shrimp", "MassiveBooster", "Thoroughbred",
    "solidBooster1-1", "solidBooster_sm_v2", "solidBooster_v2",
    "LaunchEscapeSystem", "sepMotor1",
    # Twin-Boar (classified as Engine)
    "Size2LFB_v2",
}

# ---------------------------------------------------------------------------
# MiscEquipment provides flags
# Parts not in this map get provides=frozenset() (filler).
# ---------------------------------------------------------------------------

PROVIDES: dict[str, frozenset] = {
    # Probe cores (reaction_wheel for those with meaningful SAS torque)
    "probeCoreHex_v2": frozenset({"probe_core", "reaction_wheel"}),
    "probeCoreOcto2_v2": frozenset({"probe_core"}),
    "probeCoreOcto_v2": frozenset({"probe_core"}),
    "probeCoreCube": frozenset({"probe_core"}),
    "probeCoreSphere_v2": frozenset({"probe_core"}),
    "probeStackSmall": frozenset({"probe_core", "reaction_wheel"}),
    "probeStackLarge": frozenset({"probe_core", "reaction_wheel"}),
    "HECS2_ProbeCore": frozenset({"probe_core", "reaction_wheel"}),
    "roverBody_v2": frozenset({"probe_core"}),
    "mk2DroneCore": frozenset({"probe_core", "reaction_wheel"}),
    "MpoProbe": frozenset({"probe_core"}),
    "MtmStage": frozenset({"probe_core"}),
    # Capsules / cockpits
    "mk1pod_v2": frozenset({"capsule", "reaction_wheel"}),
    "mk1-3pod": frozenset({"capsule", "reaction_wheel"}),
    "Mark1Cockpit": frozenset({"capsule"}),
    "Mark2Cockpit": frozenset({"capsule"}),
    "mk2Cockpit_Standard": frozenset({"capsule"}),
    "mk2Cockpit_Inline": frozenset({"capsule"}),
    "mk3Cockpit_Shuttle": frozenset({"capsule", "reaction_wheel"}),
    "landerCabinSmall": frozenset({"capsule"}),
    "mk2LanderCabin_v2": frozenset({"capsule"}),
    "cupola": frozenset({"capsule"}),
    "seatExternalCmd": frozenset({"capsule"}),
    # Crew cabins
    "MK1CrewCabin": frozenset({"capsule"}),
    "crewCabin": frozenset({"capsule"}),
    "mk2CrewCabin": frozenset({"capsule"}),
    "mk3CrewCabin": frozenset({"capsule"}),
    "Large_Crewed_Lab": frozenset({"capsule"}),
    # Reaction wheels
    "advSasModule": frozenset({"reaction_wheel"}),
    "sasModule": frozenset({"reaction_wheel"}),
    "asasmodule1-2": frozenset({"reaction_wheel"}),
    # Solar panels (fixed)
    "solarPanels5": frozenset({"solar_fixed"}),
    "LgRadialSolarPanel": frozenset({"solar_fixed"}),
    "solarPanelOX10C": frozenset({"solar_fixed"}),
    "solarPanelSP10C": frozenset({"solar_fixed"}),
    # Solar panels (retractable)
    "solarPanels3": frozenset({"solar_retractable"}),
    "solarPanels4": frozenset({"solar_retractable"}),
    "solarPanels1": frozenset({"solar_retractable"}),
    "solarPanels2": frozenset({"solar_retractable"}),
    "solarPanelOX10L": frozenset({"solar_retractable"}),
    "solarPanelSP10L": frozenset({"solar_retractable"}),
    "largeSolarPanel": frozenset({"solar_retractable", "solar_array_large"}),
    # RTG
    "rtg": frozenset({"rtg"}),
    # Antennas (t1=local, t2=inner planets, t3=mid system, t4=outer system)
    "longAntenna": frozenset({"relay_t1"}),
    "SurfAntenna": frozenset({"relay_t1"}),
    "HighGainAntenna5_v2": frozenset({"relay_t1"}),
    "RelayAntenna5": frozenset({"relay_t2"}),
    "mediumDishAntenna": frozenset({"relay_t2"}),
    "HighGainAntenna": frozenset({"relay_t3"}),
    "RelayAntenna50": frozenset({"relay_t3"}),
    "commDish": frozenset({"relay_t4"}),
    "RelayAntenna100": frozenset({"relay_t4"}),
    # RCS (includes vernor engine which functions as RCS)
    "RCSBlock_v2": frozenset({"rcs"}),
    "RCSLinearSmall": frozenset({"rcs"}),
    "linearRcs": frozenset({"rcs"}),
    "RCSblock_01_small": frozenset({"rcs"}),
    "vernierEngine": frozenset({"rcs"}),
    # Batteries
    "batteryPack": frozenset({"battery_small"}),
    "batteryBankMini": frozenset({"battery_small"}),
    "ksp_r_largeBatteryPack": frozenset({"battery_small"}),
    "batteryBank": frozenset({"battery_large", "battery_small"}),
    "batteryBankLarge": frozenset({"battery_large", "battery_small"}),
    # Docking ports
    "dockingPort1": frozenset({"docking_port"}),
    "dockingPort2": frozenset({"docking_port"}),
    "dockingPort3": frozenset({"docking_port"}),
    "dockingPortLarge": frozenset({"docking_port"}),
    "dockingPortLateral": frozenset({"docking_port"}),
    "mk2DockingPort": frozenset({"docking_port"}),
    # Fuel lines
    "fuelLine": frozenset({"fuel_line"}),
    # Ladders
    "ladder1": frozenset({"ladder"}),
    "telescopicLadder": frozenset({"ladder"}),
    "telescopicLadderBay": frozenset({"ladder"}),
    # Launch clamp
    "launchClamp1": frozenset({"launch_clamp"}),
    # ISRU
    "ISRU": frozenset({"isru"}),
    "MiniISRU": frozenset({"isru"}),
    # Science instruments
    "sensorThermometer": frozenset({"science_instrument", "thermometer"}),
    "sensorBarometer": frozenset({"science_instrument", "barometer"}),
    "sensorAccelerometer": frozenset({"science_instrument"}),
    "sensorAtmosphere": frozenset({"science_instrument"}),
    "sensorGravimeter": frozenset({"science_instrument"}),
    "science_module": frozenset({"science_instrument"}),
    "GooExperiment": frozenset({"science_instrument"}),
}

# ---------------------------------------------------------------------------
# Landing leg tiers (by mass: lighter = smaller = lower tier)
# ---------------------------------------------------------------------------

LANDING_LEG_TIER: dict[str, int] = {
    "miniLandingLeg": 1,    # 0.015t
    "landingLeg1": 1,       # 0.05t
    "landingLeg1-2": 2,     # 0.1t
}

# Resources considered valid fuel for FuelTank classification
_FUEL_RESOURCES = {"LiquidFuel", "Oxidizer", "MonoPropellant", "XenonGas"}


def classify_part(name: str, cfg: dict) -> tuple[str, dict] | None:
    """Return (part_type_name, overrides) or None to skip."""
    title = cfg.get("title", "").strip()
    if not title:
        return None

    # Forced type overrides
    if name in FORCE_TYPE:
        ptype = FORCE_TYPE[name]
        overrides = {}
        if ptype == "MiscEquipment":
            overrides["provides"] = PROVIDES.get(name, frozenset())
        return ptype, overrides

    # Engines and SRBs
    if "engine" in cfg:
        eng = cfg["engine"]
        engine_type = eng.get("engine_type", "")
        if engine_type == "SolidBooster":
            return "SolidBooster", {}
        # Turbine engines handled by FORCE_TYPE above
        return "Engine", {}

    # Parachutes
    if "parachute" in cfg:
        is_drogue = "drogue" in name.lower() or "Drogue" in title
        return "Parachute", {"is_drogue": is_drogue}

    # Decouplers and heat shields (both have decouple modules)
    if cfg.get("has_module_decouple") or cfg.get("has_module_anchored_decouple"):
        if "HeatShield" in name or "heatshield" in name.lower():
            return "HeatShield", {}
        return "Decoupler", {}

    # Landing legs
    if name in LANDING_LEG_TIER:
        return "LandingLeg", {"tier": LANDING_LEG_TIER[name]}

    # Fuel tanks: parts whose resources are purely fuel (no IntakeAir/Ore/etc)
    if "resources" in cfg and name not in NOT_FUEL_TANK:
        resources = cfg["resources"]
        res_keys = set(resources.keys()) - {"ElectricCharge"}
        if res_keys and res_keys <= _FUEL_RESOURCES:
            return "FuelTank", {}

    # Everything else is MiscEquipment
    return "MiscEquipment", {"provides": PROVIDES.get(name, frozenset())}


def format_provides(provides: frozenset) -> str:
    """Format a frozenset for Python source code."""
    if not provides:
        return "frozenset()"
    items = sorted(provides)
    if len(items) == 1:
        return 'frozenset({"' + items[0] + '"})'
    inner = ", ".join(f'"{s}"' for s in items)
    return "frozenset({" + inner + "})"


def format_entry(name: str, part_type: str, ap_item: str, offset: int,
                 overrides: dict) -> str:
    """Format a single PartMapping entry."""
    # Escape quotes in AP item name for Python string
    escaped = ap_item.replace('"', '\\"')

    parts = [f'    PartMapping("{name}", {part_type}, "{escaped}", {offset}']

    if overrides:
        override_parts = []
        for k, v in sorted(overrides.items()):
            if isinstance(v, frozenset):
                override_parts.append(f'"{k}": {format_provides(v)}')
            elif isinstance(v, bool):
                override_parts.append(f'"{k}": {v}')
            elif isinstance(v, int):
                override_parts.append(f'"{k}": {v}')
            else:
                override_parts.append(f'"{k}": {v!r}')
        override_str = ", ".join(override_parts)
        parts.append(f",\n                {{{override_str}}})")
    else:
        parts.append(")")

    return "".join(parts)


# Type display order for grouping output
TYPE_ORDER = [
    "Engine", "SolidBooster", "FuelTank", "HeatShield",
    "Parachute", "LandingLeg", "Decoupler", "MiscEquipment",
]

# Section headers for each type
TYPE_HEADERS = {
    "Engine": "Engines",
    "SolidBooster": "Solid Rocket Boosters",
    "FuelTank": "Fuel Tanks",
    "HeatShield": "Heat Shields",
    "Parachute": "Parachutes",
    "LandingLeg": "Landing Legs",
    "Decoupler": "Decouplers",
    "MiscEquipment": "Misc Equipment",
}


# ---------------------------------------------------------------------------
# Offset persistence: read existing offsets from parts.py on re-runs
# ---------------------------------------------------------------------------

_OFFSET_RE = re.compile(
    r'PartMapping\(\s*"[^"]+"\s*,\s*\w+\s*,\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*,\s*(\d+)'
)


def read_existing_offsets() -> dict[str, int]:
    """Parse parts.py to extract ap_item → offset from current PART_REGISTRY."""
    parts_py = Path(__file__).resolve().parent.parent / "parts.py"
    if not parts_py.exists():
        return {}
    text = parts_py.read_text()
    offsets: dict[str, int] = {}
    for m in _OFFSET_RE.finditer(text):
        # Unescape the ap_item string
        ap_item = m.group(1).replace('\\"', '"')
        offset = int(m.group(2))
        offsets[ap_item] = offset
    return offsets


def assign_offsets(ap_items: list[str]) -> dict[str, int]:
    """Assign stable offsets: preserve existing, append new alphabetically."""
    existing = read_existing_offsets()
    result: dict[str, int] = {}

    # Preserve existing offsets for items still present
    for name in ap_items:
        if name in existing:
            result[name] = existing[name]

    # Assign new items starting after max existing offset
    if result:
        next_offset = max(result.values()) + 1
    else:
        next_offset = PARTS_RANGE_START

    for name in sorted(ap_items):
        if name not in result:
            result[name] = next_offset
            next_offset += 1

    return result


def main():
    parts_path = Path(__file__).resolve().parent.parent / "data" / "parts.json"
    with open(parts_path) as f:
        all_parts = json.load(f)

    # Classify all parts
    by_type: dict[str, list[tuple[str, str, str, dict]]] = {
        t: [] for t in TYPE_ORDER
    }

    for name in sorted(all_parts.keys()):
        if name in SKIP:
            continue
        cfg = all_parts[name]
        result = classify_part(name, cfg)
        if result is None:
            continue
        part_type, overrides = result
        title = cfg["title"].strip()
        ap_item = title
        by_type[part_type].append((name, part_type, ap_item, overrides))

    # Collect all ap_item names and assign stable offsets
    all_ap_items = [ap for entries in by_type.values() for _, _, ap, _ in entries]
    offsets = assign_offsets(all_ap_items)

    # Output
    print("PART_REGISTRY: list[PartMapping] = [")
    for ptype in TYPE_ORDER:
        entries = by_type[ptype]
        if not entries:
            continue
        print(f"    # --- {TYPE_HEADERS[ptype]} ({len(entries)}) ---")
        for name, pt, ap_item, overrides in entries:
            print(format_entry(name, pt, ap_item, offsets[ap_item], overrides) + ",")
    print("]")

    # Summary to stderr
    total = sum(len(v) for v in by_type.values())
    print(f"\n# Total: {total} parts", file=sys.stderr)
    for ptype in TYPE_ORDER:
        count = len(by_type[ptype])
        if count:
            print(f"#   {TYPE_HEADERS[ptype]}: {count}", file=sys.stderr)
    offset_vals = sorted(offsets.values())
    print(f"# Offset range: {offset_vals[0]}–{offset_vals[-1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
