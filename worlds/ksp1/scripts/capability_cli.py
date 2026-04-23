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
    ALL_BODIES, BODY_BY_NAME, DIFFICULTY_PROFILES, MissionEdge,
)
from worlds.ksp1.capability import (
    EquipmentFlags, ProfileResult,
    compute_capability_from_items, evaluate_mission_detailed,
    get_capability,
)
from worlds.ksp1.locations import (
    event_location_names, get_body_events,
)
from worlds.ksp1.parts import PART_REGISTRY
from worlds.ksp1.world import KSP1World


# ---------------------------------------------------------------------------
# Event → (mission_type, crewed) — used only by rocket command to map
# check names to evaluate_mission_detailed params
# ---------------------------------------------------------------------------

EVENT_TO_MISSION: dict[str, tuple[str, bool]] = {
    "Flyby":         ("orbit", False),
    "SOI Leave":     ("orbit", False),
    "Orbit":         ("orbit", False),
    "EVA in Orbit":  ("orbit", True),
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
        for event in get_body_events(body):
            mission_type, crewed = EVENT_TO_MISSION[event]
            for loc_name in event_location_names(body.name, event):
                result[loc_name] = CheckInfo(body.name, event, mission_type, crewed)
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
            "games": ["Kerbal Space Program 1"],
        }]))
        dp_msg = json.loads(ws.recv())
        assert dp_msg[0]["cmd"] == "DataPackage", f"Expected DataPackage, got {dp_msg[0]['cmd']}"

        game_data = dp_msg[0]["data"]["games"]["Kerbal Space Program 1"]
        item_name_to_id: dict[str, int] = game_data["item_name_to_id"]
        location_name_to_id: dict[str, int] = game_data["location_name_to_id"]
        item_id_to_name = {v: k for k, v in item_name_to_id.items()}
        location_id_to_name = {v: k for k, v in location_name_to_id.items()}

        ws.send(json.dumps([{
            "cmd": "Connect",
            "game": "Kerbal Space Program 1",
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

    # For custom goals, reconstruct body lists from goal_locations in slot_data.
    # Entries follow "{Body} {Event} 1" — e.g. "Mun Flag Plant 1", "Duna Return 1".
    if options["goal"] == 99:  # Goal.option_custom
        flag_bodies = set()
        return_bodies = set()
        sample_return_bodies = set()
        for loc in sd.get("goal_locations", []):
            if loc.endswith(" Flag Plant 1"):
                flag_bodies.add(loc.replace(" Flag Plant 1", ""))
            elif loc.endswith(" Sample Return 1"):
                sample_return_bodies.add(loc.replace(" Sample Return 1", ""))
            elif loc.endswith(" Return 1"):
                return_bodies.add(loc.replace(" Return 1", ""))
        options["flag_bodies"] = flag_bodies
        options["return_bodies"] = return_bodies
        options["sample_return_bodies"] = sample_return_bodies

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
        if not loc.can_reach(state):
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

    in_logic = loc_obj.can_reach(state)
    loc_id = ap.location_name_to_id.get(check_name)
    already_checked = loc_id is not None and loc_id in ap.checked_locations

    # Look up in the mission map for rocket details
    info = CHECK_MAP.get(check_name)
    if info is None:
        # Not a per-body mission check — just report in-logic status
        print(f"\n'{check_name}' — In logic: {'YES' if in_logic else 'NO'}")
        if already_checked:
            print("(Already checked — won't appear in 'in-logic' listing.)")
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
    logic_str = "YES" if in_logic else "NO"
    if already_checked:
        logic_str += " (already checked)"
    print(f"  In logic: {logic_str}")
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

    if not result.stage_results:
        # Trivial mission (e.g. Kerbin Flag Plant — no propulsion required)
        print("\n  (Trivial mission — no propulsion required.)")
        if verbose:
            _print_item_dump(ap)
        return

    # --- Per-stage breakdown (KSP convention: stage 0 = last to fire) ---
    num_stages = len(result.stage_results)
    asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)
    for i, stage in enumerate(result.stage_results):
        group = result.edge_groups[i] if i < len(result.edge_groups) else []
        is_terminal = (i == num_stages - 1)

        edge_names = [f"{e.source} -> {e.destination}" for e in group]
        header = ", ".join(edge_names) if edge_names else "unknown"

        ksp_stage_num = num_stages - 1 - i

        # Stage header with symmetry/asparagus tags
        tags: list[str] = []
        if asparagus and not is_terminal:
            tags.append("ASPARAGUS")
        if stage.engine_count > 1:
            tags.append(f"{stage.engine_count}-WAY")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"\n  Stage {ksp_stage_num} ({header}):{tag_str}")
        print(f"    Parts:")

        # Terminal parts (command module + support equipment)
        if is_terminal:
            for count, part_id in result.terminal_parts:
                print(f"      {count}x {_titled(part_id)}")

        # Propulsion
        if stage.engine_count > 0 and stage.engine_name != "none":
            print(f"      {stage.engine_count}x {_titled(stage.engine_name)}")
        if stage.tank_count > 0 and stage.tank_name != "none":
            fill_pct = stage.fill_fraction * 100
            fill_str = f" ({fill_pct:.0f}% fill)" if fill_pct < 100 else ""
            print(f"      {stage.tank_count}x {_titled(stage.tank_name)}{fill_str}")

        # Non-propulsion equipment (from capability manifest)
        for count, part_id in stage.equipment:
            print(f"      {count}x {_titled(part_id)}")

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
        if loc.can_reach(state):
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
