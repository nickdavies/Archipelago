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

from websockets.sync.client import connect as ws_connect

from BaseClasses import MultiWorld
from test.general import setup_multiworld

from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, DIFFICULTY_PROFILES, MissionBuilder, effective_dv,
    physics_profile_name_from_slot_data,
)
from worlds.ksp1.capability import (
    compute_capability_from_items, evaluate_mission_detailed,
    get_capability,
)
from worlds.ksp1.locations import EventName, KSC_LOCATION_PREFIX, MissionLocation
from worlds.ksp1.world import KSP1World

from worlds.ksp1.capability_format import (
    CHECK_MAP, CheckInfo,
    titled, edge_desc, location_group,
    format_rocket_output, format_parts_list,
    format_in_logic_locations, build_bug_report_dict,
    to_json_serializable, ITEM_TITLES, ITEM_TYPES,
)


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
        for loc_str in sd.get("goal_locations", []):
            parsed = MissionLocation.parse(loc_str)
            if parsed is None:
                continue
            if parsed.event == EventName.FLAG_PLANT:
                flag_bodies.add(parsed.body)
            elif parsed.event == EventName.SAMPLE_RETURN:
                sample_return_bodies.add(parsed.body)
            elif parsed.event == EventName.RETURN:
                return_bodies.add(parsed.body)
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

    loc_name_to_id = ap.location_name_to_id
    actionable: dict[str, list[str]] = defaultdict(list)

    for loc in multiworld.get_locations(player):
        if loc.address is None:
            continue
        loc_id = loc_name_to_id.get(loc.name)
        if loc_id is None or loc_id not in ap.missing_locations:
            continue
        if not loc.can_reach(state):
            continue

        if loc.name.startswith(KSC_LOCATION_PREFIX):
            actionable["KSC"].append(loc.name)
        elif " " in loc.name:
            group = location_group(loc.name)
            actionable[group].append(loc.name)
        else:
            actionable["Other"].append(loc.name)

    for line in format_in_logic_locations(actionable):
        print(line)
    print()

    if parts_list:
        _print_parts_list(ap)


def cmd_rocket(ap: APState, check_name: str, verbose: bool = False) -> None:
    """Show detailed rocket design for a specific check."""
    multiworld, player = build_world_and_state(ap)
    state = multiworld.state
    world: KSP1World = multiworld.worlds[player]

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

    info = CHECK_MAP.get(check_name)

    difficulty_name = physics_profile_name_from_slot_data(ap.slot_data)

    cap, flags = compute_capability_from_items(
        lambda name: state.count(name, player),
        difficulty_name,
        bool(ap.slot_data.get("start_with_launch_clamps", 1)),
        world.mission_builder,
    )

    result = None
    if info is not None:
        diff = DIFFICULTY_PROFILES[difficulty_name]
        result = evaluate_mission_detailed(
            flags, diff, info.body_name, info.mission_type, info.crewed,
            world.mission_builder,
            threshold_km=info.threshold_km,
        )

    lines = format_rocket_output(
        check_name, in_logic, already_checked, info, result, flags, difficulty_name,
        world.mission_builder,
        sounding_altitude_km=cap.sounding_altitude_km,
    )
    for line in lines:
        print(line)

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


def _print_parts_list(ap: APState) -> None:
    """Print received items grouped by type, with human-readable names."""
    items_by_name: dict[str, int] = {}
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id, f"Unknown ({item_id})")
        items_by_name[name] = count

    for line in format_parts_list(items_by_name):
        print(line)
    print()


# ---------------------------------------------------------------------------
# missions — static dump of the Kerbin-home mission graph sorted by delta-v
# ---------------------------------------------------------------------------

def cmd_missions(difficulty_name: str) -> None:
    """Dump every mission in the Kerbin-home mission graph sorted by delta-v
    ascending.

    No server connection required — pure static walk of the mission graph.
    For each (body, mission_type) we pick the cheapest profile alternative
    by raw base_dv sum, and also report the difficulty-adjusted budget.
    """
    diff = DIFFICULTY_PROFILES[difficulty_name]
    mission_builder = MissionBuilder(home=BodyName.KERBIN)

    rows: list[tuple[float, float, str, str, int, int]] = []
    for (body_name, mission_type), profiles in mission_builder.all_profiles().items():
        if not profiles:
            # Zero-dv missions (Kerbin flag_plant, Kerbin sample_return)
            rows.append((0.0, 0.0, str(body_name), str(mission_type), 0, 0))
            continue

        # Pick the cheapest alternative by raw base_dv sum
        best_base = None
        best_adj = None
        best_edges = 0
        for edges in profiles:
            base = sum(e.base_dv for e in edges)
            pc = sum(e.plane_change_dv for e in edges)
            adj = effective_dv(base, diff, plane_change_dv=pc)
            if best_base is None or base < best_base:
                best_base = base
                best_adj = adj
                best_edges = len(edges)
        rows.append((
            best_base, best_adj, str(body_name), str(mission_type),
            best_edges, len(profiles),
        ))

    rows.sort(key=lambda r: (r[0], r[2], r[3]))

    print(f"Mission profiles sorted by base delta-v (difficulty: {difficulty_name})")
    print(f"{'base_dv':>10}  {'adj_dv':>10}  {'edges':>5}  {'alts':>4}  body / mission")
    print("-" * 72)
    for base, adj, body, mission, n_edges, n_alts in rows:
        print(f"{base:>10.0f}  {adj:>10.0f}  {n_edges:>5}  {n_alts:>4}  {body} {mission}")


# ---------------------------------------------------------------------------
# Bug report
# ---------------------------------------------------------------------------

def cmd_bug_report(ap: APState, check_name: str | None = None) -> None:
    """Dump full state as JSON for bug reports."""
    items_by_name: dict[str, int] = {}
    for item_id, count in ap.item_id_counts.items():
        name = ap.item_id_to_name.get(item_id, f"unknown_{item_id}")
        items_by_name[name] = count

    checked_names = sorted(
        ap.location_id_to_name.get(loc_id, f"unknown_{loc_id}")
        for loc_id in ap.checked_locations
    )

    # Compute in-logic locations
    multiworld, player = build_world_and_state(ap)
    state = multiworld.state
    world: KSP1World = multiworld.worlds[player]
    in_logic_locs = []
    for loc in multiworld.get_locations(player):
        if loc.address is None:
            continue
        loc_id = ap.location_name_to_id.get(loc.name)
        if loc_id is None or loc_id not in ap.missing_locations:
            continue
        if loc.can_reach(state):
            in_logic_locs.append(loc.name)

    report = build_bug_report_dict(
        slot_data=ap.slot_data,
        items_by_name=items_by_name,
        checked_names=checked_names,
        missing_count=len(ap.missing_locations),
        in_logic_locs=in_logic_locs,
        mission_builder=world.mission_builder,
        check_name=check_name,
    )
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KSP1 Archipelago capability inspector",
        prog="python -m worlds.ksp1.scripts.capability_cli",
    )
    parser.add_argument("command",
                        choices=["in-logic", "rocket", "bug-report", "missions"],
                        help="Command to run")
    parser.add_argument("check_name", nargs="?", default=None,
                        help="Check name for 'rocket' / 'bug-report' command")
    parser.add_argument("--host", default=None,
                        help="AP server host:port (required for server commands)")
    parser.add_argument("--slot", default=None,
                        help="Player slot name (required for server commands)")
    parser.add_argument("--password", default="",
                        help="Server password (optional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show received items list")
    parser.add_argument("--parts-list", action="store_true",
                        help="Show received parts with human-readable names")
    parser.add_argument("--difficulty", default="comfortable",
                        choices=list(DIFFICULTY_PROFILES.keys()),
                        help="Physics difficulty for 'missions' adj_dv column "
                             "(default: comfortable)")

    args = parser.parse_args()

    if args.command == "rocket" and not args.check_name:
        parser.error("'rocket' command requires a check_name argument")

    if args.command == "missions":
        cmd_missions(args.difficulty)
        return

    if not args.host or not args.slot:
        parser.error(f"'{args.command}' requires --host and --slot")

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
