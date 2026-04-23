"""
KSP1 Tracker Client — an Archipelago CommonClient that auto-displays
in-logic locations and provides /rocket, /parts, and /bug_report commands.

Can be launched from the Archipelago Launcher or run directly:
    python -m worlds.ksp1.tracker_client --connect localhost:38281
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from typing import Optional

import CommonClient
from CommonClient import (
    CommonContext, get_base_parser, gui_enabled, server_loop,
)
from MultiServer import mark_raw

from BaseClasses import MultiWorld
from test.general import setup_multiworld

from worlds.ksp1.bodies import ALL_BODIES, DIFFICULTY_PROFILES
from worlds.ksp1.capability import (
    compute_capability_from_items, evaluate_mission_detailed,
)
from worlds.ksp1.world import KSP1World

from worlds.ksp1.scripts.capability_format import (
    CHECK_MAP, location_group,
    format_rocket_output, format_parts_list,
    format_in_logic_locations, build_bug_report_dict,
)

logger = logging.getLogger("Client")

GAME_NAME = "Kerbal Space Program 1"


# ---------------------------------------------------------------------------
# Build a local AP world + CollectionState from tracker state
# ---------------------------------------------------------------------------

def build_world_and_state(
    slot_data: dict,
    item_counts: dict[str, int],
) -> tuple[MultiWorld, int]:
    """Construct a KSP1World and populate state from received item counts."""
    sd = slot_data
    options = {
        "difficulty": sd.get("difficulty", 1),
        "start_with_launch_clamps": sd.get("start_with_launch_clamps", 1),
        "goal": sd.get("goal", 0),
    }

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

    for name, count in item_counts.items():
        state.prog_items[player][name] += count

    return multiworld, player


# ---------------------------------------------------------------------------
# Compute in-logic locations grouped by body/category
# ---------------------------------------------------------------------------

def compute_in_logic_groups(
    slot_data: dict,
    item_counts: dict[str, int],
    missing_locations: set[int],
    location_name_to_id: dict[str, int],
) -> tuple[dict[str, list[str]], MultiWorld, int]:
    """Return (grouped_locations, multiworld, player)."""
    multiworld, player = build_world_and_state(slot_data, item_counts)
    state = multiworld.state

    actionable: dict[str, list[str]] = defaultdict(list)
    for loc in multiworld.get_locations(player):
        if loc.address is None:
            continue
        loc_id = location_name_to_id.get(loc.name)
        if loc_id is None or loc_id not in missing_locations:
            continue
        if not loc.can_reach(state):
            continue

        if loc.name.startswith("KSC "):
            actionable["KSC"].append(loc.name)
        elif " " in loc.name:
            group = location_group(loc.name)
            actionable[group].append(loc.name)
        else:
            actionable["Other"].append(loc.name)

    return actionable, multiworld, player


# ---------------------------------------------------------------------------
# Command Processor
# ---------------------------------------------------------------------------

class KSP1CommandProcessor(CommonClient.ClientCommandProcessor):
    ctx: KSP1TrackerContext

    @mark_raw
    def _cmd_rocket(self, location_name: str = "") -> None:
        """Show detailed rocket design for a location. Usage: /rocket Mun Orbit 1"""
        if not location_name:
            self.output("Usage: /rocket <location name>")
            return
        if not self.ctx.slot_data:
            self.output("Not connected to a server yet.")
            return

        multiworld, player = self.ctx.rebuild_state()
        state = multiworld.state

        loc_obj = None
        for loc in multiworld.get_locations(player):
            if loc.name == location_name:
                loc_obj = loc
                break

        if loc_obj is None:
            self.output(f"Unknown location: '{location_name}'")
            return

        in_logic = loc_obj.can_reach(state)
        loc_id = self.ctx.location_name_to_id.get(location_name)
        already_checked = loc_id is not None and loc_id in self.ctx.checked_locations

        info = CHECK_MAP.get(location_name)
        difficulty_name = ["casual", "normal", "expert", "insane"][
            self.ctx.slot_data.get("difficulty", 1)
        ]

        result = None
        flags = None
        if info is not None:
            diff = DIFFICULTY_PROFILES[difficulty_name]
            _, flags = compute_capability_from_items(
                lambda name: state.count(name, player),
                difficulty_name,
                bool(self.ctx.slot_data.get("start_with_launch_clamps", 1)),
            )
            result = evaluate_mission_detailed(
                flags, diff, info.body_name, info.mission_type, info.crewed,
            )

        lines = format_rocket_output(
            location_name, in_logic, already_checked, info, result, flags,
            difficulty_name,
        )
        for line in lines:
            self.output(line)

    @mark_raw
    def _cmd_parts(self, filter_text: str = "") -> None:
        """Show received parts grouped by type. Usage: /parts [filter]"""
        if not self.ctx.slot_data:
            self.output("Not connected to a server yet.")
            return

        item_counts = self.ctx.get_item_counts_by_name()
        if filter_text:
            filter_lower = filter_text.lower()
            item_counts = {
                k: v for k, v in item_counts.items()
                if filter_lower in k.lower()
            }
        lines = format_parts_list(item_counts)
        for line in lines:
            self.output(line)

    @mark_raw
    def _cmd_bug_report(self, description: str = "") -> None:
        """Dump state to a JSON file for bug reports. Usage: /bug_report [description]"""
        if not self.ctx.slot_data:
            self.output("Not connected to a server yet.")
            return

        multiworld, player = self.ctx.rebuild_state()
        state = multiworld.state

        item_counts = self.ctx.get_item_counts_by_name()

        checked_names = sorted(
            self.ctx.location_names.lookup_in_game(loc_id)
            for loc_id in self.ctx.checked_locations
        )

        in_logic_locs = []
        for loc in multiworld.get_locations(player):
            if loc.address is None:
                continue
            loc_id = self.ctx.location_name_to_id.get(loc.name)
            if loc_id is None or loc_id not in self.ctx.missing_locations:
                continue
            if loc.can_reach(state):
                in_logic_locs.append(loc.name)

        report = build_bug_report_dict(
            slot_data=self.ctx.slot_data,
            items_by_name=item_counts,
            checked_names=checked_names,
            missing_count=len(self.ctx.missing_locations),
            in_logic_locs=in_logic_locs,
            user_description=description or None,
        )

        import Utils
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = Utils.user_path(f"ksp1_bug_report_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        self.output(f"Bug report saved to: {path}")


# ---------------------------------------------------------------------------
# Tracker Context
# ---------------------------------------------------------------------------

class KSP1TrackerContext(CommonContext):
    game = GAME_NAME
    tags = CommonContext.tags | {"Tracker"}
    items_handling = 0b111
    want_slot_data = True
    command_processor = KSP1CommandProcessor

    slot_data: dict
    location_name_to_id: dict[str, int]
    tracker_tab: Optional[object]  # set by GUI if running

    def __init__(self, server_address: Optional[str], password: Optional[str]):
        super().__init__(server_address, password)
        self.slot_data = {}
        self.location_name_to_id = {}
        self.tracker_tab = None

    def make_gui(self) -> type:
        from kvui import GameManager, UILog, SelectableLabel

        tracker_ctx = self

        from kivy.properties import StringProperty
        from kivymd.app import MDApp

        class TrackerLabel(SelectableLabel):
            """A label that runs /rocket on double-click for location rows."""
            location_name = StringProperty("")

            def on_touch_down(self, touch):
                if (self.collide_point(*touch.pos)
                        and touch.is_double_tap
                        and self.location_name):
                    app = MDApp.get_running_app()
                    # Run the rocket command — output goes to the log tab
                    cmd_proc = tracker_ctx.command_processor(tracker_ctx)
                    cmd_proc._cmd_rocket(self.location_name)
                    # Switch to the log tab
                    log_tab = app.screens.current_tab
                    for child in app.tabs.children:
                        if hasattr(child, "text") and child.text in ("Archipelago", "All"):
                            log_tab = child
                            break
                    app.screens.switch_screens(log_tab)
                    log_tab.active = True
                    return True
                return super().on_touch_down(touch)

        class TrackerLog(UILog):
            """A UILog variant that replaces its data wholesale and uses TrackerLabel."""
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.viewclass = "TrackerLabel"

            def set_locations(self, data: list[dict]):
                self.data = data

        # Register TrackerLabel with Kivy's Factory so the recycleview can
        # instantiate it by class name string.
        from kivy.factory import Factory
        if not hasattr(Factory, "TrackerLabel"):
            Factory.register("TrackerLabel", cls=TrackerLabel)

        class KSP1Manager(GameManager):
            base_title = "KSP1 Tracker"

            def build(self):
                container = super().build()
                self.tracker_log = TrackerLog()
                self.tracker_log.data = [{"text": "Waiting for connection..."}]
                self.add_client_tab("Tracker", self.tracker_log)
                tracker_ctx.tracker_tab = self.tracker_log
                return container

        return KSP1Manager

    async def server_auth(self, password_requested: bool = False):
        if password_requested and not self.password:
            await super().server_auth(password_requested)
        await self.get_username()
        await self.send_connect()

    def on_package(self, cmd: str, args: dict):
        if cmd == "Connected":
            self.slot_data = args.get("slot_data", {})
            # Build reverse lookup: name -> id
            game_lookup = self.location_names[self.game]
            self.location_name_to_id = {
                name: lid for lid, name in game_lookup.items()
                if not isinstance(name, str) or not name.startswith("Unknown")
            }
            # Don't display here — ReceivedItems follows immediately and
            # will trigger display with items already populated.

        elif cmd == "ReceivedItems":
            self.display_in_logic()

        elif cmd == "RoomUpdate":
            if "checked_locations" in args:
                self.display_in_logic()

    def get_item_counts_by_name(self) -> dict[str, int]:
        """Aggregate received items into {item_name: count}."""
        counts: dict[str, int] = defaultdict(int)
        game_lookup = self.item_names[self.game]
        for item in self.items_received:
            name = game_lookup[item.item]
            counts[name] += 1
        return dict(counts)

    def rebuild_state(self) -> tuple[MultiWorld, int]:
        """Build a MultiWorld + CollectionState from current received items."""
        item_counts = self.get_item_counts_by_name()
        return build_world_and_state(self.slot_data, item_counts)

    def display_in_logic(self) -> None:
        """Update both the text console and the tracker tab."""
        if not self.slot_data:
            return

        item_counts = self.get_item_counts_by_name()
        actionable, _, _ = compute_in_logic_groups(
            self.slot_data, item_counts,
            self.missing_locations, self.location_name_to_id,
        )

        # GUI tracker tab — build structured data with location_name for
        # double-click rocket lookup
        if self.tracker_tab is not None:
            tab_data = []
            total = sum(len(v) for v in actionable.values())
            tab_data.append({
                "text": f"In-Logic Unchecked Locations ({total})"
                        "  [double-click a location for rocket details]",
                "location_name": "",
            })
            if not actionable:
                tab_data.append({"text": "  (none)", "location_name": ""})
            else:
                body_names = [b.name for b in ALL_BODIES]
                ordered_groups = (
                    [n for n in body_names if n in actionable]
                    + sorted(k for k in actionable
                             if k not in body_names and k != "KSC")
                    + (["KSC"] if "KSC" in actionable else [])
                )
                for group_name in ordered_groups:
                    tab_data.append({
                        "text": f"\n  {group_name}:",
                        "location_name": "",
                    })
                    for loc in actionable[group_name]:
                        tab_data.append({
                            "text": f"    - {loc}",
                            "location_name": loc,
                        })
            self.tracker_tab.data = tab_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args):
    ctx = KSP1TrackerContext(args.connect, args.password)
    ctx.auth = args.name
    ctx.server_task = asyncio.create_task(server_loop(ctx), name="server loop")

    if gui_enabled:
        ctx.run_gui()
    ctx.run_cli()

    await ctx.exit_event.wait()
    await ctx.shutdown()


def launch(*args):
    import Utils
    Utils.init_logging("KSP1TrackerClient", exception_logger="Client")

    import colorama
    colorama.just_fix_windows_console()

    parser = get_base_parser(description="KSP1 Tracker Client")
    parser.add_argument("--name", default=None, help="Slot name")
    parser.add_argument("url", nargs="?", help="Archipelago connection URL")
    args = CommonClient.handle_url_arg(parser.parse_args(args))
    asyncio.run(main(args))
    colorama.deinit()


if __name__ == "__main__":
    launch(*sys.argv[1:])
