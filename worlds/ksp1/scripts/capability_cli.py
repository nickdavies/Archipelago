"""
CLI tool for inspecting KSP1 Archipelago capability state.

Connects to a running AP server, fetches the player's received items,
constructs a local AP world with matching options, and evaluates the
actual access rules — no duplicated logic.

Usage (from the Archipelago/ directory):
    python -m worlds.ksp1.scripts.capability_cli in-logic \
        --host localhost:38281 --slot Player1

    python -m worlds.ksp1.scripts.capability_cli rocket "Mun Return 1" \
        --host localhost:38281 --slot Player1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from websockets.sync.client import connect as ws_connect

from BaseClasses import CollectionState, MultiWorld
from test.general import setup_multiworld
from worlds.AutoWorld import call_all

from worlds.ksp1.bodies import (
    ALL_BODIES, BODY_BY_NAME, DIFFICULTY_PROFILES,
    DifficultyProfile, EdgeType, MissionEdge,
)
from worlds.ksp1.capability import (
    EquipmentFlags, ProfileResult,
    compute_capability_from_items, evaluate_mission_detailed,
    get_capability,
    _required_chute_count,
    _support_equipment_mass,
)
from worlds.ksp1.locations import (
    event_location_names, get_body_events,
)
from worlds.ksp1.parts import PART_DB, PART_REGISTRY, MiscEquipment
from worlds.ksp1.world import KSP1World


# ---------------------------------------------------------------------------
# Event → (mission_type, crewed) — used only by rocket command to map
# check names to evaluate_mission_detailed params
# ---------------------------------------------------------------------------

EVENT_TO_MISSION: dict[str, tuple[str, bool]] = {
    "Flyby":         ("orbit", False),
    "SOI Leave":     ("orbit", False),
    "Orbit":         ("orbit", False),
    "Landing":       ("land", False),
    "Crewed Landing": ("land", True),
    "Flag Plant":    ("land", True),
    "Return":        ("return", False),
    "Sample Return": ("sample_return", True),
}


@dataclass
class CheckInfo:
    body_name: str
    event: str
    mission_type: str
    crewed: bool


def _build_check_map() -> dict[str, CheckInfo]:
    """Build mapping from location name → mission parameters."""
    result: dict[str, CheckInfo] = {}
    for body in ALL_BODIES:
        if body.name == "Kerbin":
            continue
        for event in get_body_events(body):
            mission_type, crewed = EVENT_TO_MISSION[event]
            for loc_name in event_location_names(body.name, event):
                result[loc_name] = CheckInfo(body.name, event, mission_type, crewed)

    # Kerbin uses hand-crafted location names, not the standard {body} {event} {slot}
    # pattern. EVA in Orbit is the only one that uses the mission profile system —
    # the rest are sounding rocket altitude checks, equipment gates, or trivial.
    result["Kerbin EVA in Orbit"] = CheckInfo("Kerbin", "Orbit", "orbit", True)

    return result


CHECK_MAP: dict[str, CheckInfo] = _build_check_map()


# ---------------------------------------------------------------------------
# AP server connection
# ---------------------------------------------------------------------------

@dataclass
class APState:
    """State fetched from an AP server for one player slot."""
    item_id_counts: dict[int, int]
    checked_locations: set[int]
    missing_locations: set[int]
    slot_data: dict
    item_id_to_name: dict[int, str]
    item_name_to_id: dict[str, int]
    location_id_to_name: dict[int, str]
    location_name_to_id: dict[str, int]


def fetch_ap_state(host: str, slot: str, password: str = "") -> APState:
    """Connect to an AP server, fetch items/locations, return APState."""
    uri = f"ws://{host}"
    with ws_connect(uri) as ws:
        room_info = json.loads(ws.recv())
        assert room_info[0]["cmd"] == "RoomInfo", f"Expected RoomInfo, got {room_info[0]['cmd']}"

        ws.send(json.dumps([{
            "cmd": "GetDataPackage",
            "games": ["Kerbal Space Program"],
        }]))
        dp_msg = json.loads(ws.recv())
        assert dp_msg[0]["cmd"] == "DataPackage", f"Expected DataPackage, got {dp_msg[0]['cmd']}"

        game_data = dp_msg[0]["data"]["games"]["Kerbal Space Program"]
        item_name_to_id: dict[str, int] = game_data["item_name_to_id"]
        location_name_to_id: dict[str, int] = game_data["location_name_to_id"]
        item_id_to_name = {v: k for k, v in item_name_to_id.items()}
        location_id_to_name = {v: k for k, v in location_name_to_id.items()}

        ws.send(json.dumps([{
            "cmd": "Connect",
            "game": "Kerbal Space Program",
            "name": slot,
            "uuid": "capability-cli",
            "version": {"major": 0, "minor": 5, "build": 1, "class": "Version"},
            "tags": ["AP", "Tracker"],
            "items_handling": 0b111,
            "slot_data": True,
            "password": password,
        }]))

        item_id_counts: dict[int, int] = defaultdict(int)
        slot_data = {}
        checked: set[int] = set()
        missing: set[int] = set()

        while True:
            raw = ws.recv()
            messages = json.loads(raw)
            done = False
            for msg in messages:
                cmd = msg["cmd"]
                if cmd == "Connected":
                    slot_data = msg.get("slot_data", {})
                    checked = set(msg.get("checked_locations", []))
                    missing = set(msg.get("missing_locations", []))
                elif cmd == "ReceivedItems":
                    for item in msg.get("items", []):
                        item_id_counts[item["item"]] += 1
                    if msg.get("index", 0) == 0:
                        done = True
                elif cmd == "ConnectionRefused":
                    errors = msg.get("errors", [])
                    print(f"Connection refused: {errors}", file=sys.stderr)
                    sys.exit(1)
            if done:
                break

    return APState(
        item_id_counts=dict(item_id_counts),
        checked_locations=checked,
        missing_locations=missing,
        slot_data=slot_data,
        item_id_to_name=item_id_to_name,
        item_name_to_id=item_name_to_id,
        location_id_to_name=location_id_to_name,
        location_name_to_id=location_name_to_id,
    )


# ---------------------------------------------------------------------------
# Build a local AP world + CollectionState from server data
# ---------------------------------------------------------------------------

def build_world_and_state(ap: APState) -> tuple[MultiWorld, int]:
    """
    Construct a KSP1World with matching options and populate a
    CollectionState with the received items.
    Returns (multiworld, player).
    """
    sd = ap.slot_data
    options = {
        "difficulty": sd.get("difficulty", 1),
        "start_with_launch_clamps": sd.get("start_with_launch_clamps", 1),
        "goal": sd.get("goal", 0),
    }

    multiworld = setup_multiworld(
        KSP1World,
        steps=("generate_early", "create_regions", "create_items", "set_rules"),
        options=options,
    )
    player = 1
    state = multiworld.state

    # Populate state with received items (by name, resolved from AP IDs)
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id)
        if name is not None:
            state.prog_items[player][name] += count

    return multiworld, player


# ---------------------------------------------------------------------------
# in-logic command
# ---------------------------------------------------------------------------

def cmd_in_logic(ap: APState, parts_list: bool = False) -> None:
    """Show all in-logic unchecked locations."""
    multiworld, player = build_world_and_state(ap)
    state = multiworld.state

    # Build location_name → location_id for filtering checked/missing
    loc_name_to_id = ap.location_name_to_id

    # Group results by body (or category)
    actionable: dict[str, list[str]] = defaultdict(list)

    for loc in multiworld.get_locations(player):
        if loc.address is None:
            continue  # skip event locations (Victory)
        loc_id = loc_name_to_id.get(loc.name)
        if loc_id is None or loc_id not in ap.missing_locations:
            continue
        if not loc.access_rule(state):
            continue

        # Group by body name (first word) or "Tech Tree" / "KSC"
        if loc.name.startswith("KSC "):
            actionable["KSC"].append(loc.name)
        elif " " in loc.name:
            group = _location_group(loc.name)
            actionable[group].append(loc.name)
        else:
            actionable["Other"].append(loc.name)

    total = sum(len(v) for v in actionable.values())
    print(f"\n=== In-Logic Unchecked Locations ({total}) ===\n")

    if not actionable:
        print("  (none)")
        return

    # Print in body order, then tech tree, then KSC
    body_names = [b.name for b in ALL_BODIES]
    ordered_groups = (
        [n for n in body_names if n in actionable]
        + sorted(k for k in actionable if k not in body_names and k != "KSC")
        + (["KSC"] if "KSC" in actionable else [])
    )

    for group_name in ordered_groups:
        locs = actionable[group_name]
        print(f"  {group_name}:")
        for loc in locs:
            print(f"    - {loc}")

    print()

    if parts_list:
        _print_parts_list(ap)


def _location_group(loc_name: str) -> str:
    """Extract the body/category name from a location name for grouping."""
    for body in ALL_BODIES:
        if loc_name.startswith(body.name + " "):
            return body.name
    # Tech tree nodes don't start with a body name
    return "Tech Tree"


# ---------------------------------------------------------------------------
# rocket command — per-stage equipment reconstruction
# ---------------------------------------------------------------------------

def _find_best_heat_shield(flags: EquipmentFlags) -> Optional[str]:
    """Return name of the best heat shield (largest size_class)."""
    if not flags.available_heat_shields:
        return None
    best = max(flags.available_heat_shields, key=lambda hs: hs.size_class)
    return best.name


def _find_lightest_aero_control(flags: EquipmentFlags) -> Optional[str]:
    """Return name of the lightest available aero control surface, or None."""
    if not flags.available_aero_controls:
        return None
    return min(flags.available_aero_controls, key=lambda p: p.mass).name


def _find_lightest_decoupler(flags: EquipmentFlags) -> Optional[tuple[str, float]]:
    """Return (name, mass) of the lightest stack decoupler, or None."""
    stack = [d for d in flags.available_decouplers if d.kind == "stack"]
    if not stack:
        return None
    best = min(stack, key=lambda d: d.mass)
    return best.name, best.mass


def _find_best_legs(flags: EquipmentFlags, required_tier: int) -> Optional[str]:
    """Return name of the lightest legs meeting the tier."""
    for leg in sorted(flags.available_landing_legs, key=lambda l: l.mass):
        if leg.tier >= required_tier:
            return leg.name
    if flags.available_landing_legs:
        return min(flags.available_landing_legs, key=lambda l: l.mass).name
    return None


def _find_command_module(flags: EquipmentFlags, crewed: bool) -> Optional[str]:
    """Find the command module matching what _evaluate_profile selects."""
    target_mass = flags.heaviest_capsule_mass if crewed else flags.lightest_probe_mass
    provides_flag = "capsule" if crewed else "probe_core"

    for item_name, parts in PART_DB.items():
        for part in parts:
            if isinstance(part, MiscEquipment):
                if provides_flag in part.provides and abs(part.mass - target_mass) < 1e-6:
                    return part.name
    return f"{'Capsule' if crewed else 'Probe Core'} ({target_mass:.3f}t)"


def _estimate_chute_count(
    landing_mass: float,
    body_name: str,
    flags: EquipmentFlags,
    diff: DifficultyProfile,
) -> tuple[Optional[str], int]:
    """Estimate parachute count for aero landing. Returns (chute_name, count)."""
    body = BODY_BY_NAME.get(body_name)
    if body is None or not body.has_atmosphere:
        return None, 0

    non_drogue = [p for p in flags.available_parachutes if not p.is_drogue]
    if not non_drogue:
        return None, 0

    chute = non_drogue[0]
    count = _required_chute_count(landing_mass, body, flags, diff)
    if count <= 0:
        return chute.name, max(1, count)
    return chute.name, count


def _edge_desc(edge: MissionEdge) -> str:
    """Human-readable edge description."""
    return f"{edge.source} -> {edge.destination} ({edge.base_dv:.0f} m/s)"


def cmd_rocket(ap: APState, check_name: str, verbose: bool = False) -> None:
    """Show detailed rocket design for a specific check."""
    multiworld, player = build_world_and_state(ap)
    state = multiworld.state

    # Check if this location is actually in-logic
    loc_obj = None
    for loc in multiworld.get_locations(player):
        if loc.name == check_name:
            loc_obj = loc
            break

    if loc_obj is None:
        print(f"Unknown check: '{check_name}'", file=sys.stderr)
        sys.exit(1)

    in_logic = loc_obj.access_rule(state)

    # Look up in the mission map for rocket details
    info = CHECK_MAP.get(check_name)
    if info is None:
        # Not a per-body mission check — just report in-logic status
        print(f"\n'{check_name}' — In logic: {'YES' if in_logic else 'NO'}")
        print("(Not a per-body mission; no rocket design to show.)")
        if verbose:
            _print_item_dump(ap)
        return

    difficulty_name = ["casual", "normal", "expert", "insane"][
        ap.slot_data.get("difficulty", 1)
    ]
    diff = DIFFICULTY_PROFILES[difficulty_name]

    cap = get_capability(state, player)
    _, flags = compute_capability_from_items(
        lambda name: state.count(name, player),
        difficulty_name,
        bool(ap.slot_data.get("start_with_launch_clamps", 1)),
    )

    body_name = info.body_name
    mission_type = info.mission_type
    crewed = info.crewed

    result = evaluate_mission_detailed(flags, diff, body_name, mission_type, crewed)

    # --- Print mission summary ---
    print(f"\n{'=' * 60}")
    print(f"  Mission: {check_name}")
    print(f"  Body: {body_name} | Type: {mission_type} | Crewed: {crewed}")
    print(f"  Difficulty: {difficulty_name}")
    print(f"  In logic: {'YES' if in_logic else 'NO'}")
    print(f"  Feasible: {'YES' if result.feasible else 'NO'}")
    if result.feasible:
        print(f"  Launch mass: {result.launch_mass:.2f} t")
    elif result.failure_reason:
        print(f"  Failure reason: {result.failure_reason}")
    print(f"{'=' * 60}")

    if not result.feasible:
        if verbose:
            _print_item_dump(ap)
        return

    # --- Per-stage breakdown (KSP convention: stage 0 = last to fire) ---
    num_stages = len(result.stage_results)
    asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)
    for i, stage in enumerate(result.stage_results):
        group = result.edge_groups[i] if i < len(result.edge_groups) else []
        is_terminal = (i == num_stages - 1)

        stage_body_name = group[0].body if group else body_name
        stage_body = BODY_BY_NAME.get(stage_body_name)

        edge_names = [f"{e.source} -> {e.destination}" for e in group]
        header = ", ".join(edge_names) if edge_names else "unknown"

        ksp_stage_num = num_stages - 1 - i
        asp_tag = " [ASPARAGUS]" if asparagus and not is_terminal else ""
        print(f"\n  Stage {ksp_stage_num} ({header}):{asp_tag}")
        print(f"    Parts:")

        # Command module + support equipment (terminal stage only)
        if is_terminal:
            cmd_name = _find_command_module(flags, crewed)
            if cmd_name:
                print(f"      1x {_titled(cmd_name)}")
            # Support equipment (antenna, power)
            all_edges = [e for g in result.edge_groups for e in g]
            _, support_parts = _support_equipment_mass(flags, all_edges)
            for sp in support_parts:
                print(f"      1x {_titled(sp)}")

        # Propulsion
        print(f"      {stage.engine_count}x {_titled(stage.engine_name)}")
        if stage.tank_count > 0:
            fill_pct = stage.fill_fraction * 100
            fill_str = f" ({fill_pct:.0f}% fill)" if fill_pct < 100 else ""
            print(f"      {stage.tank_count}x {_titled(stage.tank_name)}{fill_str}")

        # Heat shield
        if any(e.needs_heat_shield for e in group):
            hs_name = _find_best_heat_shield(flags)
            if hs_name:
                print(f"      1x {_titled(hs_name)}")

        # Landing legs
        if any(e.needs_landing_legs for e in group):
            leg_tier = stage_body.landing_leg_tier if stage_body else 1
            leg_name = _find_best_legs(flags, leg_tier)
            if leg_name:
                print(f"      4x {_titled(leg_name)}")

        # Aero control surfaces (atmospheric ascent without a gimbal engine
        # requires actuated fins/elevons; we always show them when available
        # on atmospheric-ascent stages so the player knows they're needed).
        if (flags.has_aero_control_surface
                and any(e.edge_type == EdgeType.ATMOSPHERIC_ASCENT for e in group)):
            fin_name = _find_lightest_aero_control(flags)
            if fin_name:
                print(f"      4x {_titled(fin_name)}")

        # Parachutes (aero landing edges)
        aero_edges = [e for e in group if e.edge_type == EdgeType.ATMO_LANDING_AERO]
        if aero_edges:
            aero_body_name = aero_edges[0].body
            chute_name, chute_count = _estimate_chute_count(
                stage.stage_mass_dry, aero_body_name, flags, diff,
            )
            if chute_name and chute_count > 0:
                print(f"      {chute_count}x {_titled(chute_name)}")

        # Ladder
        if any(e.needs_ladder for e in group):
            print(f"      1x Pegasus I Mobility Enhancer")

        # Decoupler (on the lower stage, separates it from the stage above)
        if not is_terminal and num_stages > 1:
            dec = _find_lightest_decoupler(flags)
            if dec:
                dec_name, dec_mass = dec
                print(f"      1x {_titled(dec_name)} ({dec_mass:.3f}t)")

        # Edges
        print(f"    Edges:")
        for edge in group:
            print(f"      {_edge_desc(edge)}")

        # Stats
        print(f"    Stats:")
        print(f"      dv: {stage.delta_v:.0f} m/s | TWR: {stage.twr_at_ignition:.2f} -> {stage.twr_at_burnout:.2f}")
        print(f"      Wet: {stage.stage_mass_wet:.2f}t | Dry: {stage.stage_mass_dry:.2f}t")

    # --- Edge → stage summary ---
    print(f"\n  Edge -> Stage Summary:")
    for i, group in enumerate(result.edge_groups):
        ksp_stage_num = num_stages - 1 - i
        for edge in group:
            print(f"    {edge.source} -> {edge.destination}: Stage {ksp_stage_num}")

    if verbose:
        _print_item_dump(ap)


def _print_item_dump(ap: APState) -> None:
    """Print all received items for bug reports."""
    print(f"\n  Received Items ({sum(ap.item_id_counts.values())} total):")
    items_by_name: list[tuple[str, int]] = []
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id, f"Unknown ({item_id})")
        items_by_name.append((name, count))
    items_by_name.sort()
    for name, count in items_by_name:
        print(f"    {count}x {name}")
    print()


# ksp_name → human-readable title from PART_REGISTRY
_ITEM_TITLES: dict[str, str] = {m.ksp_name: m.title for m in PART_REGISTRY}

# ksp_name → part type name (Engine, FuelTank, etc.)
_ITEM_TYPES: dict[str, str] = {m.ksp_name: m.part_type.__name__ for m in PART_REGISTRY}


def _titled(ksp_name: str) -> str:
    """Format a part name with its human-readable title, e.g. 'liquidEngine_v2 (LV-T30 "Reliant")'."""
    title = _ITEM_TITLES.get(ksp_name)
    if title:
        return f"{ksp_name} ({title})"
    return ksp_name


def _print_parts_list(ap: APState) -> None:
    """Print received items grouped by type, with human-readable names."""
    items_by_name: list[tuple[str, int]] = []
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id, f"Unknown ({item_id})")
        items_by_name.append((name, count))

    # Group by part type
    groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for name, count in sorted(items_by_name):
        part_type = _ITEM_TYPES.get(name, "Other")
        title = _ITEM_TITLES.get(name, name)
        groups[part_type].append((name, title, count))

    total = sum(ap.item_id_counts.values())
    print(f"\n=== Received Parts ({total} items) ===\n")

    # Print part types in a useful order, then "Other" last
    type_order = [
        "Engine", "SolidBooster", "FuelTank", "Decoupler",
        "HeatShield", "Parachute", "LandingLeg", "MiscEquipment", "Other",
    ]
    seen = set()
    for type_name in type_order:
        if type_name not in groups:
            continue
        seen.add(type_name)
        print(f"  {type_name}:")
        for name, title, count in groups[type_name]:
            print(f"    {count}x {name:<35s} {title}")
    # Any types not in the ordering
    for type_name in sorted(groups):
        if type_name in seen:
            continue
        print(f"  {type_name}:")
        for name, title, count in groups[type_name]:
            print(f"    {count}x {name:<35s} {title}")

    print()


# ---------------------------------------------------------------------------
# Bug report JSON output
# ---------------------------------------------------------------------------

import dataclasses

def _to_json_serializable(obj):
    """Convert dataclasses, sets, and other non-JSON types for serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_json_serializable(v)
                for k, v in dataclasses.asdict(obj).items()
                if not k.startswith("_")}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj) if all(isinstance(x, (str, int, float)) for x in obj) else list(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    return obj


def cmd_bug_report(ap: APState, check_name: Optional[str] = None) -> None:
    """Dump full state as JSON for bug reports."""
    difficulty_name = ["casual", "normal", "expert", "insane"][
        ap.slot_data.get("difficulty", 1)
    ]

    # Received items by name
    items_by_name: dict[str, int] = {}
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id, f"unknown_{item_id}")
        items_by_name[name] = count

    # Checked locations by name
    checked_names = sorted(
        ap.location_id_to_name.get(loc_id, f"unknown_{loc_id}")
        for loc_id in ap.checked_locations
    )

    report: dict = {
        "slot_data": ap.slot_data,
        "difficulty": difficulty_name,
        "received_items": items_by_name,
        "checked_locations": checked_names,
        "missing_location_count": len(ap.missing_locations),
    }

    # Equipment flags
    _, flags = compute_capability_from_items(
        lambda name: items_by_name.get(name, 0),
        difficulty_name,
        bool(ap.slot_data.get("start_with_launch_clamps", 1)),
    )
    report["equipment_flags"] = {
        "staging_tier": flags.staging_tier,
        "has_heat_shield": flags.has_heat_shield,
        "has_parachutes": flags.has_parachutes,
        "has_probe_core": flags.has_probe_core,
        "has_capsule": flags.has_capsule,
        "has_rcs": flags.has_rcs,
        "has_reaction_wheels": flags.has_reaction_wheels,
        "has_rtg": flags.has_rtg,
        "has_solar": flags.has_solar,
        "has_fuel_lines": flags.has_fuel_lines,
        "has_launch_clamp": flags.has_launch_clamp,
        "relay_tier": flags.relay_tier,
        "landing_leg_tier": flags.landing_leg_tier,
        "engines": [e.name for e in flags.available_engines],
        "srbs": [s.name for s in flags.available_srbs],
        "tanks": [t.name for t in flags.available_tanks],
        "decouplers": [d.name for d in flags.available_decouplers],
    }

    # Rocket evaluation for specific check
    if check_name:
        info = CHECK_MAP.get(check_name)
        if info:
            diff = DIFFICULTY_PROFILES[difficulty_name]
            result = evaluate_mission_detailed(
                flags, diff, info.body_name, info.mission_type, info.crewed,
            )
            report["rocket"] = {
                "check_name": check_name,
                "body": info.body_name,
                "mission_type": info.mission_type,
                "crewed": info.crewed,
                "feasible": result.feasible,
                "launch_mass": result.launch_mass,
                "failure_reason": result.failure_reason,
                "stages": _to_json_serializable(result.stage_results),
                "edge_groups": [
                    [_to_json_serializable({"source": e.source, "destination": e.destination,
                                            "base_dv": e.base_dv, "edge_type": e.edge_type.name})
                     for e in group]
                    for group in result.edge_groups
                ],
            }

    # In-logic locations
    multiworld, player = build_world_and_state(ap)
    state = multiworld.state
    in_logic_locs = []
    for loc in multiworld.get_locations(player):
        if loc.address is None:
            continue
        loc_id = ap.location_name_to_id.get(loc.name)
        if loc_id is None or loc_id not in ap.missing_locations:
            continue
        if loc.access_rule(state):
            in_logic_locs.append(loc.name)
    report["in_logic_locations"] = sorted(in_logic_locs)

    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KSP1 Archipelago capability inspector",
        prog="python -m worlds.ksp1.scripts.capability_cli",
    )
    parser.add_argument("command", choices=["in-logic", "rocket", "bug-report"],
                        help="Command to run")
    parser.add_argument("check_name", nargs="?", default=None,
                        help="Check name for 'rocket' / 'bug-report' command")
    parser.add_argument("--host", required=True,
                        help="AP server host:port (e.g. localhost:38281)")
    parser.add_argument("--slot", required=True,
                        help="Player slot name")
    parser.add_argument("--password", default="",
                        help="Server password (optional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show received items list")
    parser.add_argument("--parts-list", action="store_true",
                        help="Show received parts with human-readable names")

    args = parser.parse_args()

    if args.command == "rocket" and not args.check_name:
        parser.error("'rocket' command requires a check_name argument")

    print(f"Connecting to {args.host} as {args.slot}...", file=sys.stderr)
    ap = fetch_ap_state(args.host, args.slot, args.password)
    print(f"Connected. Received {sum(ap.item_id_counts.values())} items, "
          f"{len(ap.checked_locations)} checked / {len(ap.missing_locations)} missing locations.",
          file=sys.stderr)

    if args.command == "in-logic":
        cmd_in_logic(ap, parts_list=args.parts_list)
    elif args.command == "rocket":
        cmd_rocket(ap, args.check_name, verbose=args.verbose)
    elif args.command == "bug-report":
        cmd_bug_report(ap, args.check_name)


if __name__ == "__main__":
    main()
