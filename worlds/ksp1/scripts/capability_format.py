"""
Shared formatting and data helpers for KSP1 capability inspection.

Used by both capability_cli.py (developer CLI) and tracker_client.py
(Archipelago Launcher tracker client).
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from worlds.ksp1.bodies import (
    ALL_BODIES, BODY_BY_NAME, DIFFICULTY_PROFILES, EdgeType,
    MISSION_PROFILES, MissionEdge,
)
from worlds.ksp1.capability import (
    EquipmentFlags, ProfileResult,
    compute_capability_from_items, evaluate_mission_detailed,
)
from worlds.ksp1.locations import event_location_names, get_body_events
from worlds.ksp1.parts import PART_REGISTRY


# ---------------------------------------------------------------------------
# Event → (mission_type, crewed) mapping
# ---------------------------------------------------------------------------

EVENT_TO_MISSION: dict[str, tuple[str, bool | None]] = {
    "Flyby":          ("orbit", None),
    "SOI Leave":      ("orbit", None),
    "Orbit":          ("orbit", None),
    "EVA in Orbit":   ("orbit", True),
    "Landing":        ("land", None),
    "Crewed Landing": ("land", True),
    "Flag Plant":     ("land", True),
    "Return":         ("return", None),
    "Sample Return":  ("sample_return", True),
}


@dataclass
class CheckInfo:
    body_name: str
    event: str
    mission_type: str
    crewed: bool | None  # True=crewed, False=unmanned, None=try both


def _build_check_map() -> dict[str, CheckInfo]:
    """Build mapping from location name -> mission parameters."""
    result: dict[str, CheckInfo] = {}
    for body in ALL_BODIES:
        for event in get_body_events(body):
            mission_type, crewed = EVENT_TO_MISSION[event]
            for loc_name in event_location_names(body.name, event):
                result[loc_name] = CheckInfo(body.name, event, mission_type, crewed)
    return result


CHECK_MAP: dict[str, CheckInfo] = _build_check_map()


# ---------------------------------------------------------------------------
# Part name formatting
# ---------------------------------------------------------------------------

ITEM_TITLES: dict[str, str] = {m.ksp_name: m.title for m in PART_REGISTRY}
ITEM_TYPES: dict[str, str] = {m.ksp_name: m.part_type.__name__ for m in PART_REGISTRY}


def titled(ksp_name: str) -> str:
    """Format a part name with its human-readable title."""
    title = ITEM_TITLES.get(ksp_name)
    if title:
        return f"{ksp_name} ({title})"
    return ksp_name


def edge_desc(edge: MissionEdge) -> str:
    """Human-readable edge description."""
    return f"{edge.source} -> {edge.destination} ({edge.base_dv:.0f} m/s)"


def location_group(loc_name: str) -> str:
    """Extract the body/category name from a location name for grouping."""
    for body in ALL_BODIES:
        if loc_name.startswith(body.name + " "):
            return body.name
    return "Tech Tree"


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def to_json_serializable(obj):
    """Convert dataclasses, sets, and other non-JSON types for serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_json_serializable(v)
                for k, v in dataclasses.asdict(obj).items()
                if not k.startswith("_")}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj) if all(isinstance(x, (str, int, float)) for x in obj) else list(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    return obj


# ---------------------------------------------------------------------------
# Profile pre-check summary
# ---------------------------------------------------------------------------

def _profile_prereqs(profile: list[MissionEdge]) -> list[str]:
    """Extract the equipment pre-checks a profile demands (no delta-v)."""
    reqs: list[str] = []

    if any(e.requires_attitude_control for e in profile):
        reqs.append("attitude_control")

    leg_bodies = [BODY_BY_NAME[e.body] for e in profile if e.needs_landing_legs]
    if leg_bodies:
        tier = max(b.landing_leg_tier for b in leg_bodies)
        reqs.append(f"landing_legs(tier>={tier})")

    if any(e.needs_ladder for e in profile):
        reqs.append("ladder")

    if any(e.needs_heat_shield for e in profile):
        reqs.append("heat_shield")

    if any(e.edge_type == EdgeType.ATMO_LANDING_AERO for e in profile):
        reqs.append("parachutes")

    power_bodies = sorted({e.body for e in profile})
    if power_bodies:
        reqs.append(f"power({','.join(power_bodies)})")

    relay_tiers = {BODY_BY_NAME[e.body].min_relay_tier for e in profile}
    max_relay = max(relay_tiers, default=0)
    if max_relay > 0:
        reqs.append(f"relay(tier>={max_relay})")

    prop_types: set[str] = set()
    for e in profile:
        et = e.edge_type
        if et in (EdgeType.ATMOSPHERIC_ASCENT, EdgeType.ATMO_LANDING_PROPULSIVE):
            prop_types.add("launch_engine+fuel")
        elif et in (EdgeType.VACUUM_ASCENT, EdgeType.PURE_VACUUM,
                    EdgeType.PLANET_TRANSFER, EdgeType.VACUUM_LANDING):
            prop_types.add("engine+fuel")
    if prop_types:
        reqs.append(f"propulsion({','.join(sorted(prop_types))})")

    return reqs


def _format_profile_summary(info: CheckInfo) -> list[str]:
    """Return lines summarising the mission profile pre-checks for a check."""
    lines: list[str] = []

    crewed = info.crewed
    if crewed is True:
        cmd = "capsule"
    elif crewed is False:
        cmd = "probe_core"
    else:
        cmd = "probe_core OR capsule"

    crewed_label = {True: "crewed", False: "unmanned", None: "either"}[crewed]
    lines.append(f"  Profile: [{info.mission_type}, {crewed_label}]")
    lines.append(f"  Command: {cmd}")

    profiles = MISSION_PROFILES.get((info.body_name, info.mission_type), [])
    if not profiles:
        lines.append("    (no profiles defined)")
        return lines

    for i, profile in enumerate(profiles):
        prereqs = _profile_prereqs(profile)
        edge_summary = " -> ".join(
            f"{e.source}->{e.destination}({e.edge_type.name},{e.base_dv:.0f}m/s)"
            for e in profile
        )
        tag = f"  alt {i+1}" if len(profiles) > 1 else "  route"
        lines.append(f"{tag}: {', '.join(prereqs) if prereqs else '(no equipment gates)'}")
        lines.append(f"    edges: {edge_summary}")

    return lines


# ---------------------------------------------------------------------------
# Rocket output formatting
# ---------------------------------------------------------------------------

def format_rocket_output(
    check_name: str,
    in_logic: bool,
    already_checked: bool,
    info: Optional[CheckInfo],
    result: Optional[ProfileResult],
    flags: Optional[EquipmentFlags],
    difficulty_name: str,
) -> list[str]:
    """Return lines of the rocket breakdown for a given check.

    If ``info`` is None the check isn't a per-body mission; returns a short
    status message.  If ``result`` is None, only the non-mission status is
    returned.
    """
    lines: list[str] = []

    if info is None:
        logic_str = "YES" if in_logic else "NO"
        lines.append(f"\n'{check_name}' -- In logic: {logic_str}")
        if already_checked:
            lines.append("(Already checked -- won't appear in 'in-logic' listing.)")
        lines.append("(Not a per-body mission; no rocket design to show.)")
        return lines

    assert result is not None and flags is not None

    body_name = info.body_name
    mission_type = info.mission_type
    crewed = info.crewed

    logic_str = "YES" if in_logic else "NO"
    if already_checked:
        logic_str += " (already checked)"

    lines.append(f"\n{'=' * 60}")
    lines.append(f"  Mission: {check_name}")
    lines.append(f"  Body: {body_name} | Difficulty: {difficulty_name}")
    lines.append(f"  In logic: {logic_str}")
    lines.extend(_format_profile_summary(info))
    lines.append(f"  Feasible: {'YES' if result.feasible else 'NO'}")
    if result.feasible:
        lines.append(f"  Launch mass: {result.launch_mass:.2f} t")
    elif result.failure_reasons:
        if len(result.failure_reasons) == 1:
            lines.append(f"  Failure reason: {result.failure_reasons[0]}")
        else:
            lines.append("  Failure reasons:")
            for r in result.failure_reasons:
                lines.append(f"    - {r}")
    lines.append(f"{'=' * 60}")

    if not result.feasible:
        return lines

    if not result.stage_results:
        lines.append("\n  (Trivial mission -- no propulsion required.)")
        return lines

    num_stages = len(result.stage_results)
    asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)
    for i, stage in enumerate(result.stage_results):
        group = result.edge_groups[i] if i < len(result.edge_groups) else []
        is_terminal = (i == num_stages - 1)

        edge_names = [f"{e.source} -> {e.destination}" for e in group]
        header = ", ".join(edge_names) if edge_names else "unknown"
        ksp_stage_num = num_stages - 1 - i

        tags: list[str] = []
        if asparagus and not is_terminal:
            tags.append("ASPARAGUS")
        if stage.engine_count > 1:
            tags.append(f"{stage.engine_count}-WAY")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(f"\n  Stage {ksp_stage_num} ({header}):{tag_str}")
        lines.append(f"    Parts:")

        if is_terminal:
            for count, part_id in result.terminal_parts:
                lines.append(f"      {count}x {titled(part_id)}")

        if stage.engine_count > 0 and stage.engine_name != "none":
            lines.append(f"      {stage.engine_count}x {titled(stage.engine_name)}")
        if stage.tank_count > 0 and stage.tank_name != "none":
            fill_pct = stage.fill_fraction * 100
            fill_str = f" ({fill_pct:.0f}% fill)" if fill_pct < 100 else ""
            lines.append(f"      {stage.tank_count}x {titled(stage.tank_name)}{fill_str}")

        for count, part_id in stage.equipment:
            lines.append(f"      {count}x {titled(part_id)}")

        lines.append(f"    Edges:")
        for edge in group:
            lines.append(f"      {edge_desc(edge)}")

        lines.append(f"    Stats:")
        lines.append(f"      dv: {stage.delta_v:.0f} m/s | TWR: {stage.twr_at_ignition:.2f} -> {stage.twr_at_burnout:.2f}")
        lines.append(f"      Wet: {stage.stage_mass_wet:.2f}t | Dry: {stage.stage_mass_dry:.2f}t")

    lines.append(f"\n  Edge -> Stage Summary:")
    for i, group in enumerate(result.edge_groups):
        ksp_stage_num = num_stages - 1 - i
        for edge in group:
            lines.append(f"    {edge.source} -> {edge.destination}: Stage {ksp_stage_num}")

    return lines


# ---------------------------------------------------------------------------
# Parts list formatting
# ---------------------------------------------------------------------------

TYPE_ORDER = [
    "Engine", "SolidBooster", "FuelTank", "Decoupler",
    "HeatShield", "Parachute", "LandingLeg", "MiscEquipment", "Other",
]


def format_parts_list(item_counts: dict[str, int]) -> list[str]:
    """Return lines of the parts list grouped by type."""
    groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for name, count in sorted(item_counts.items()):
        part_type = ITEM_TYPES.get(name, "Other")
        title = ITEM_TITLES.get(name, name)
        groups[part_type].append((name, title, count))

    total = sum(item_counts.values())
    lines: list[str] = [f"\n=== Received Parts ({total} items) ===\n"]

    seen: set[str] = set()
    for type_name in TYPE_ORDER:
        if type_name not in groups:
            continue
        seen.add(type_name)
        lines.append(f"  {type_name}:")
        for name, title, count in groups[type_name]:
            lines.append(f"    {count}x {name:<35s} {title}")
    for type_name in sorted(groups):
        if type_name in seen:
            continue
        lines.append(f"  {type_name}:")
        for name, title, count in groups[type_name]:
            lines.append(f"    {count}x {name:<35s} {title}")

    return lines


# ---------------------------------------------------------------------------
# In-logic location grouping
# ---------------------------------------------------------------------------

def format_in_logic_locations(actionable: dict[str, list[str]]) -> list[str]:
    """Return formatted lines for grouped in-logic locations."""
    total = sum(len(v) for v in actionable.values())
    lines: list[str] = [f"\n=== In-Logic Unchecked Locations ({total}) ===\n"]

    if not actionable:
        lines.append("  (none)")
        return lines

    body_names = [b.name for b in ALL_BODIES]
    ordered_groups = (
        [n for n in body_names if n in actionable]
        + sorted(k for k in actionable if k not in body_names and k != "KSC")
        + (["KSC"] if "KSC" in actionable else [])
    )

    for group_name in ordered_groups:
        locs = actionable[group_name]
        lines.append(f"  {group_name}:")
        for loc in locs:
            lines.append(f"    - {loc}")

    return lines


# ---------------------------------------------------------------------------
# Bug report builder
# ---------------------------------------------------------------------------

def build_bug_report_dict(
    slot_data: dict,
    items_by_name: dict[str, int],
    checked_names: list[str],
    missing_count: int,
    in_logic_locs: list[str],
    check_name: Optional[str] = None,
    user_description: Optional[str] = None,
) -> dict:
    """Build a JSON-serializable bug report dict."""
    difficulty_name = ["casual", "normal", "expert", "insane"][
        slot_data.get("difficulty", 1)
    ]

    report: dict = {
        "slot_data": slot_data,
        "difficulty": difficulty_name,
        "received_items": items_by_name,
        "checked_locations": checked_names,
        "missing_location_count": missing_count,
    }

    if user_description:
        report["user_description"] = user_description

    _, flags = compute_capability_from_items(
        lambda name: items_by_name.get(name, 0),
        difficulty_name,
        bool(slot_data.get("start_with_launch_clamps", 1)),
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
                "failure_reasons": result.failure_reasons,
                "stages": to_json_serializable(result.stage_results),
                "edge_groups": [
                    [to_json_serializable({"source": e.source, "destination": e.destination,
                                            "base_dv": e.base_dv, "edge_type": e.edge_type.name})
                     for e in group]
                    for group in result.edge_groups
                ],
            }

    report["in_logic_locations"] = sorted(in_logic_locs)
    return report
