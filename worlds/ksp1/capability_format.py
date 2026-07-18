"""
Shared formatting and data helpers for KSP1 capability inspection.

Used by both capability_cli.py (developer CLI) and world.py
(Universal Tracker explain_rule hook).
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from worlds.ksp1.bodies import (
    ALL_BODIES, BODY_BY_NAME, BodyName, MissionType, DIFFICULTY_PROFILES, EdgeType,
    ReboardMode,
    MissionBuilder, MissionEdge, min_relay_tier, physics_profile_name_from_slot_data,
)
from worlds.ksp1.capability import (
    EquipmentFlags, ProfileResult,
    compute_capability_from_items, evaluate_mission_detailed,
    event_mission_info,
)
from worlds.ksp1.locations import LocationBuilder, event_locations, get_body_events
from worlds.ksp1.parts import PART_REGISTRY


@dataclass
class CheckInfo:
    body_name: BodyName | None  # None = body-agnostic (e.g. Splashdown)
    event: str
    mission_type: MissionType
    crewed: bool | None  # True=crewed, False=unmanned, None=try both
    threshold_km: float | None = None  # sounding rocket target altitude


@dataclass(frozen=True)
class DiscoveryStatus:
    """Whether a location's body is reachable through the hidden-body Discover
    gates, for the ``/explain`` report.

    When ``discovered`` is False the body region can't be reached at all, so it
    is the real blocker regardless of rocket physics; ``blocking_body`` /
    ``discover_item`` name the first region you can't reach (the topmost
    undiscovered ancestor).  ``None`` is passed to the formatters when the
    hidden-bodies feature isn't gating anything, keeping their output unchanged.
    """
    discovered: bool
    blocking_body: Optional[str] = None
    discover_item: Optional[str] = None


def _discovery_lines(discovery: Optional[DiscoveryStatus]) -> list[str]:
    """Render the hidden-body discovery gate as a leading /explain section.

    Empty when nothing is gated (``discovery is None``).  When the body is
    undiscovered its region can't be reached at all, so this is the headline
    Failure reason — surfaced ahead of, and independent of, the rocket-physics
    verdict (which is discovery-blind)."""
    if discovery is None:
        return []
    if discovery.discovered:
        return ["  Body discovered: YES"]
    return [
        "  Body discovered: NO",
        f"  Failure reason: cannot reach {discovery.blocking_body} region "
        f"-- need '{discovery.discover_item}'",
        "",
    ]


def _build_check_map() -> dict[str, CheckInfo]:
    """Build mapping from location name -> mission parameters.

    Includes per-body mission events AND home-body specials for every
    landable body.  The CLI/tracker layer renders any of these locations
    even when the current seed isn't pinned to that home.
    """
    result: dict[str, CheckInfo] = {}
    # Per-body mission events
    for body in ALL_BODIES:
        for event in get_body_events(body):
            mission_type, crewed = event_mission_info(event)
            for loc in event_locations(body.name, event):
                result[str(loc)] = CheckInfo(body.name, event, mission_type, crewed)

    # Home-body specials (sounding, first_launch, …) across all landable bodies.
    for loc in LocationBuilder.all_home_locations().values():
        result[loc.name] = CheckInfo(
            loc.body, loc.name, loc.mission_type, None,
            threshold_km=loc.threshold_km,
        )

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

def _profile_prereqs(profile: list[MissionEdge],
                     home: BodyName) -> list[str]:
    """Extract the equipment pre-checks a profile demands (no delta-v).

    ``home`` is needed for the heliocentric-distance relay-tier formula.
    """
    reqs: list[str] = []

    if any(e.requires_attitude_control for e in profile):
        reqs.append("attitude_control")

    leg_bodies = [BODY_BY_NAME[e.body] for e in profile if e.needs_landing_legs]
    if leg_bodies:
        tier = max(b.landing_leg_tier for b in leg_bodies)
        reqs.append(f"landing_legs(tier>={tier})")

    if any(e.reboard is ReboardMode.LADDER_ONLY for e in profile):
        reqs.append("ladder")
    elif any(e.reboard is ReboardMode.LADDER_OR_JETPACK for e in profile):
        reqs.append("ladder|jetpack")

    if any(e.needs_heat_shield for e in profile):
        reqs.append("heat_shield")

    if any(e.edge_type == EdgeType.ATMO_LANDING for e in profile):
        # Staged descent: parachutes and/or a touchdown burn, chosen per kit.
        reqs.append("descent(parachutes|burn)")

    power_bodies = sorted({e.body for e in profile})
    if power_bodies:
        reqs.append(f"power({','.join(power_bodies)})")

    relay_tiers = {min_relay_tier(BODY_BY_NAME[e.body].name, home) for e in profile}
    max_relay = max(relay_tiers, default=0)
    if max_relay > 0:
        reqs.append(f"relay(tier>={max_relay})")

    prop_types: set[str] = set()
    for e in profile:
        et = e.edge_type
        if et == EdgeType.ATMOSPHERIC_ASCENT:
            prop_types.add("launch_engine+fuel")
        elif et in (EdgeType.VACUUM_ASCENT, EdgeType.PURE_VACUUM,
                    EdgeType.PLANET_TRANSFER, EdgeType.VACUUM_LANDING):
            prop_types.add("engine+fuel")
    if prop_types:
        reqs.append(f"propulsion({','.join(sorted(prop_types))})")

    return reqs


def _format_profile_summary(info: CheckInfo, mission_builder: MissionBuilder) -> list[str]:
    """Return lines summarising the mission profile pre-checks for a check."""
    lines: list[str] = []

    mt = info.mission_type

    # --- Kerbin-specific mission types ---
    if mt == MissionType.SOUNDING:
        lines.append(f"  Profile: [sounding rocket]")
        lines.append(f"  Command: probe_core OR capsule (with parachute + decoupler)")
        if info.threshold_km is not None:
            lines.append(f"  Target altitude: {info.threshold_km:.1f} km")
        lines.append(f"  Propulsion: SRB or engine + fuel tank")
        return lines

    if mt == MissionType.FIRST_LAUNCH:
        lines.append(f"  Profile: [first launch]")
        lines.append(f"  Command: any propulsion OR capsule (kerbal EVA)")
        return lines

    if mt == MissionType.FIRST_LANDING:
        lines.append(f"  Profile: [first safe landing]")
        lines.append(f"  Command: capsule (EVA) OR propulsion + safe descent")
        return lines

    if mt == MissionType.FIRST_STAGING:
        lines.append(f"  Profile: [first staging]")
        lines.append(f"  Requires: decoupler (staging_tier >= 1) + command part "
                     f"(capsule or probe core)")
        return lines

    if mt == MissionType.SPLASHDOWN:
        lines.append(f"  Profile: [splashdown]")
        lines.append(f"  Target altitude: >= {info.threshold_km or 1.0:.0f} km")
        lines.append(f"  Requires: safe descent (parachute or throttleable engine)")
        return lines

    # --- Standard body mission profiles ---
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

    profiles = mission_builder.profiles_for(info.body_name, info.mission_type)
    if not profiles:
        lines.append("    (no profiles defined)")
        return lines

    for i, profile in enumerate(profiles):
        prereqs = _profile_prereqs(profile, home=mission_builder.home)
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
    mission_builder: MissionBuilder,
) -> list[str]:
    """Return lines of the rocket breakdown for a given check.

    All locations with a ``CheckInfo`` (body missions AND Kerbin-specific
    locations) get the same header format.  Locations without info (KSC
    biomes, starting inventory) get a short status line.
    """
    lines: list[str] = []
    logic_str = "YES" if in_logic else "NO"
    if already_checked:
        logic_str += " (already checked)"

    if info is None:
        lines.append(f"\n'{check_name}' -- In logic: {logic_str}")
        lines.append("(Not a mission location; no rocket design to show.)")
        return lines

    assert result is not None and flags is not None

    body_name = info.body_name

    lines.append(f"\n{'=' * 60}")
    lines.append(f"  Mission: {check_name}")
    body_label = body_name if body_name is not None else "any ocean body"
    lines.append(f"  Body: {body_label} | Difficulty: {difficulty_name}")
    lines.append(f"  In logic: {logic_str}")
    lines.extend(_mission_summary_lines(result, info.mission_type))
    lines.extend(_format_profile_summary(info, mission_builder))
    if info.mission_type == MissionType.SOUNDING and info.threshold_km is not None:
        lines.append(f"  Target altitude: {info.threshold_km:.0f} km")
    lines.append(f"  Feasible: {'YES' if result.feasible else 'NO'}")
    if result.feasible:
        lines.extend(architecture_lines(result))
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

    lines.extend(format_stage_breakdown(result, flags, info.mission_type))
    return lines


# Mission types with no per-stage build (evaluated by flags/altitude, not a
# staged rocket).
_NON_PROFILE_TYPES = {
    MissionType.SOUNDING, MissionType.FIRST_LAUNCH, MissionType.FIRST_LANDING,
    MissionType.FIRST_STAGING, MissionType.SPLASHDOWN,
}


def architecture_lines(result: ProfileResult) -> list[str]:
    """Header lines naming the flight architecture the capability layer used.

    A single-launch build reports its ``Launch mass``.  The two multi-launch
    architectures the capability-ceiling retries can pick — the Apollo split
    and multi-launch orbital assembly — are otherwise invisible in the stage
    list, so name them here and relabel the mass (under assembly ``launch_mass``
    is the HEAVIEST single launch, not the stack total).
    """
    lines: list[str] = []
    via_assembly = bool(getattr(result, "via_assembly", False)
                        and result.assembly_launches)
    via_apollo = bool(getattr(result, "via_apollo", False))

    if via_assembly:
        n = len(result.assembly_launches)
        lines.append(f"  Architecture: multi-launch orbital assembly "
                     f"({n} launches docked in low orbit)")
        if result.launch_mass > 0:
            lines.append(f"  Heaviest launch: {result.launch_mass:.2f} t "
                         f"(the pad supports the largest single launch, not "
                         f"the stack total)")
    elif result.launch_mass > 0:
        lines.append(f"  Launch mass: {result.launch_mass:.2f} t")

    if via_apollo:
        lines.append("  Architecture: Apollo split -- the lander separates in "
                     "destination orbit,")
        lines.append("                descends and re-ascends with only the "
                     "pod, then rejoins the")
        lines.append("                parked return stage (rendezvous + "
                     "docking) for the trip home.")
    return lines


def _mission_summary_lines(
    result: ProfileResult,
    mission_type: MissionType,
) -> list[str]:
    """At-a-glance top-line totals: the craft's total delta-v budget and total
    launch mass.  Rendered as its own section near the top of a report so the
    overall figures are visible before the detailed profile / per-stage build.

    Only emitted for feasible, staged missions.  The non-profile Kerbin types
    (sounding / first-*) model no staged craft, and a trivial (no-propulsion)
    mission has nothing to total, so both return ``[]`` — the section is simply
    absent rather than showing a misleading zero.
    """
    if mission_type in _NON_PROFILE_TYPES:
        return []
    if not result.feasible or not result.stage_results:
        return []

    lines: list[str] = ["\n  --- Mission summary ---"]

    if getattr(result, "via_assembly", False) and result.assembly_launches:
        launches = result.assembly_launches
        n = len(launches)
        # Total tonnage the player must build across every separate launch (each
        # launch's pad mass is its heaviest/bottom stage).
        total_mass = sum(
            max((s.stage_mass_wet for s in L.stages), default=0.0)
            for L in launches
        )
        # Delta-v of the coherent vehicle that actually flies the mission after
        # docking (the group>0 assembled stack); the ascent delta-v is split
        # across the N launches detailed further down.
        stack_dv = sum(
            s.delta_v
            for s, gi in zip(result.stage_results, result.stage_group_indices)
            if gi != 0
        )
        lines.append(f"    Architecture:  {n}-launch orbital assembly")
        lines.append(f"    Total mass:    {total_mass:.2f} t "
                     f"({n} launches; heaviest {result.launch_mass:.2f} t)")
        lines.append(f"    Total delta-v: {stack_dv:.0f} m/s "
                     f"(assembled orbital stack, post-docking)")
    else:
        total_dv = sum(s.delta_v for s in result.stage_results)
        n = len(result.stage_results)
        lines.append(f"    Total delta-v: {total_dv:.0f} m/s "
                     f"({n} stage{'' if n == 1 else 's'})")
        lines.append(f"    Total mass:    {result.launch_mass:.2f} t")

    return lines


def _format_one_stage(
    stage,                          # rocket_math.StageResult
    ksp_stage_num: int,
    header: str,
    is_terminal: bool,
    terminal_parts: list[tuple[int, str]],
    asparagus: bool,
    edges: Optional[list[MissionEdge]] = None,
) -> list[str]:
    """Render a single stage's tag / parts / (optional) edges / stats block."""
    lines: list[str] = []

    tags: list[str] = []
    if stage.n_boosters > 0:
        # Real parallel build on this stage (not the old blanket flag).
        mode = "ASPARAGUS" if asparagus else "ONION"
        kind = "engine-boost" if stage.booster_engines > 0 else "drop-tank"
        tags.append(f"{mode}: {stage.n_boosters} {kind} boosters")
    elif stage.engine_count > 1:
        tags.append(f"{stage.engine_count}-WAY")
    tag_str = f"  [{', '.join(tags)}]" if tags else ""
    lines.append(f"\n  Stage {ksp_stage_num} ({header}):{tag_str}")
    lines.append(f"    Parts:")

    if is_terminal:
        for count, part_id in terminal_parts:
            lines.append(f"      {count}x {titled(part_id)}")

    if stage.engine_count > 0 and stage.engine_name != "none":
        lines.append(f"      {stage.engine_count}x {titled(stage.engine_name)}")
    for count, tank_name in stage.tank_manifest:
        if tank_name and tank_name != "none":
            lines.append(f"      {count}x {titled(tank_name)}")

    # Coalesce identical parts: the optimizer appends equipment from several
    # sites (fins, decouplers, attitude bundles) so the same part_id can
    # recur. Sum counts per part_id, preserving first-seen order.
    coalesced_equip: dict[str, int] = {}
    for count, part_id in stage.equipment:
        coalesced_equip[part_id] = coalesced_equip.get(part_id, 0) + count
    for part_id, count in coalesced_equip.items():
        lines.append(f"      {count}x {titled(part_id)}")

    # The heat shield the optimizer charged for this stage (it lives in
    # stage_mass but isn't an engine/tank/equipment entry) — without it the
    # reported build can't survive the reentry/aerocapture edge.
    if stage.heat_shield_name:
        lines.append(f"      1x {titled(stage.heat_shield_name)}")

    if edges is not None:
        lines.append(f"    Edges:")
        for edge in edges:
            lines.append(f"      {edge_desc(edge)}")

    lines.append(f"    Stats:")
    lines.append(f"      dv: {stage.delta_v:.0f} m/s | TWR: {stage.twr_at_ignition:.2f} -> {stage.twr_at_burnout:.2f}")
    lines.append(f"      Wet: {stage.stage_mass_wet:.2f}t | Dry: {stage.stage_mass_dry:.2f}t")
    return lines


def _format_stack(
    stages: list,                   # list[StageResult]
    group_indices: list[int],
    edge_groups: list[list[MissionEdge]],
    terminal_parts: list[tuple[int, str]],
    flags: EquipmentFlags,
    skip_empty_groups: bool = False,
) -> list[str]:
    """Render one staged rocket: its stages (KSP-numbered top=0) + an
    edge→stage summary.

    ``group_indices[i]`` is the ``edge_groups`` index the i-th stage performs
    (a multi-stage ascent maps several stages to one group).  ``skip_empty_groups``
    omits groups no stage in ``stages`` serves — used when rendering only the
    assembled orbital stack (the home-ascent group 0 is flown by the separate
    launches, not by any stage here).
    """
    lines: list[str] = []
    num_stages = len(stages)
    asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)

    def _group_idx(i: int) -> int:
        if i < len(group_indices):
            return group_indices[i]
        return i

    for i, stage in enumerate(stages):
        gi = _group_idx(i)
        group = edge_groups[gi] if 0 <= gi < len(edge_groups) else []
        edge_names = [f"{e.source} -> {e.destination}" for e in group]
        header = ", ".join(edge_names) if edge_names else "unknown"
        lines.extend(_format_one_stage(
            stage, num_stages - 1 - i, header,
            is_terminal=(i == num_stages - 1),
            terminal_parts=terminal_parts, asparagus=asparagus, edges=group))

    lines.append(f"\n  Edge -> Stage Summary:")
    # Invert the stage→group map: each group may be served by >1 stage (a
    # multi-stage ascent), so list every stage that performs the edge.
    group_to_stages: dict[int, list[int]] = defaultdict(list)
    for i in range(num_stages):
        group_to_stages[_group_idx(i)].append(num_stages - 1 - i)
    for gi, group in enumerate(edge_groups):
        stage_nums = sorted(set(group_to_stages.get(gi, [])), reverse=True)
        if not stage_nums and skip_empty_groups:
            continue
        label = ", ".join(f"Stage {n}" for n in stage_nums) if stage_nums else "Stage ?"
        for edge in group:
            lines.append(f"    {edge.source} -> {edge.destination}: {label}")

    return lines


def _format_assembly_breakdown(
    result: ProfileResult,
    flags: EquipmentFlags,
) -> list[str]:
    """Render a ``via_assembly`` result as N distinct launches followed by the
    assembled orbital stack that continues the mission.

    Each launch is its own lifter (a separate rocket that reaches the assembly
    orbit and docks its chunk); the delivered chunks' parts are NOT repeated
    under the launches — they appear once in the assembled stack (every
    ``group > 0`` stage of ``result.stage_results``).
    """
    lines: list[str] = []
    launches = result.assembly_launches
    n = len(launches)
    asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)
    total_chunk = sum(L.chunk_payload_mass for L in launches)

    ascent = result.edge_groups[0] if result.edge_groups else []
    ascent_desc = ", ".join(f"{e.source} -> {e.destination}"
                            for e in ascent) or "home ascent"

    lines.append(f"\n  ===== Multi-launch orbital assembly: {n} launches "
                 f"=====")
    lines.append("  Each launch below is a SEPARATE rocket that reaches the "
                 "assembly orbit and")
    lines.append("  docks its chunk; the combined stack (further down) then "
                 "flies the mission.")

    for idx, launch in enumerate(launches, 1):
        launch_wet = max((s.stage_mass_wet for s in launch.stages), default=0.0)
        lines.append(f"\n  --- Launch {idx} of {n}: pad mass {launch_wet:.2f} t "
                     f"-> delivers {launch.chunk_payload_mass:.2f} t chunk to "
                     f"orbit ---")
        lines.append(f"    flies: {ascent_desc} (+ rendezvous to assembly orbit)")
        nstg = len(launch.stages)
        for j, stage in enumerate(launch.stages):
            # Pure ascent lifters: no terminal command parts, and the ascent
            # edge is named once in the launch header above (edges=None here).
            lines.extend(_format_one_stage(
                stage, nstg - 1 - j, "ascent", is_terminal=False,
                terminal_parts=[], asparagus=asparagus, edges=None))

    lines.append(f"\n  ===== Assembled orbital stack ({total_chunk:.2f} t "
                 f"docked) -- continues the mission =====")
    # Mission-continuation stages: everything that reached orbit (group > 0);
    # the home-ascent group 0 is the launches above.
    m_stages: list = []
    m_groups: list[int] = []
    for stage, gi in zip(result.stage_results, result.stage_group_indices):
        if gi != 0:
            m_stages.append(stage)
            m_groups.append(gi)
    lines.extend(_format_stack(
        m_stages, m_groups, result.edge_groups, result.terminal_parts,
        flags, skip_empty_groups=True))
    return lines


def format_stage_breakdown(
    result: ProfileResult,
    flags: EquipmentFlags,
    mission_type: MissionType,
) -> list[str]:
    """Per-stage parts/edges/stats breakdown for a feasible mission ``result``.

    Shared by the mission view (``format_rocket_output``) and the contract view
    (``format_contract_output``) so they render stages identically. Returns ``[]``
    for non-profile mission types (sounding etc.) and a single note line for a
    trivial (no-propulsion) mission. Contract-required parts ride
    ``result.terminal_parts`` (the capability layer appends ``extra_payload_parts``
    there), so they show on the terminal stage with no special-casing here.

    A ``via_assembly`` result is rendered as its distinct launches plus the
    assembled orbital stack (see ``_format_assembly_breakdown``) instead of one
    flat stage list, which would otherwise show several independent launches as
    an impossible single stack.
    """
    if mission_type in _NON_PROFILE_TYPES:
        return []

    if not result.stage_results:
        return ["\n  (Trivial mission -- no propulsion required.)"]

    if getattr(result, "via_assembly", False) and result.assembly_launches:
        return _format_assembly_breakdown(result, flags)

    return _format_stack(
        result.stage_results, result.stage_group_indices, result.edge_groups,
        result.terminal_parts, flags)


# ---------------------------------------------------------------------------
# Contract output formatting
# ---------------------------------------------------------------------------

def format_contract_output(
    spec,                       # contracts.ContractSpec
    in_logic: bool,
    item_held: bool,
    flags: EquipmentFlags,
    diff,                       # bodies.DifficultyProfile
    difficulty_name: str,
    mission_builder: MissionBuilder,
    proxy: bool = False,
    part_manager=None,
) -> list[str]:
    """Render the ``/explain`` breakdown for a contract location.

    GENERIC over contract types: everything shown is derived from the
    ``ContractSpec`` / ``ContractTypeDef`` data (required categories, base
    mission, parameter tree) and the shared feasibility eval — adding a new
    contract type or parameter primitive needs NO change here. Shows the three
    access gates (item held / required parts / physics delivery), the
    client-facing parameter tree, and, when feasible, the delivery rocket with
    the contract parts on the terminal stage.

    ``proxy`` flags a goal contract whose access rule substitutes the
    all-progression-items proxy for the physics gate (a model-infeasible body).
    """
    # Local import keeps the capability_format <-> contracts edge one-directional.
    from worlds.ksp1.contracts import evaluate_contract, required_part_breakdown
    from worlds.ksp1.parts import DEFAULT_PART_MANAGER

    if part_manager is None:
        part_manager = DEFAULT_PART_MANAGER

    td = spec.type_def
    lines: list[str] = []

    lines.append(f"\n{'=' * 60}")
    lines.append(f"  {spec.display_name}")
    lines.append(f"  Type: {td.contract_type} | Body: {spec.body} | "
                 f"Difficulty: {difficulty_name} | Goal: {'yes' if spec.is_goal else 'no'}")
    lines.append(f"  In logic: {'YES' if in_logic else 'NO'}")
    lines.append(f"{'=' * 60}")

    # Physics feasibility of the delivery rocket, via the SAME evaluate_contract
    # the access rule's contract_access uses (so the verdict matches the rule
    # exactly).  Computed up front so the mission summary can lead the report,
    # matching the mission-report layout; Gate 3 and the delivery rocket below
    # reuse it.  None = a required category has no part (see Gate 2).
    result = evaluate_contract(spec, flags, diff, mission_builder)
    feasible = result is not None and result.feasible
    if feasible:
        lines.extend(_mission_summary_lines(result, td.base_mission_type))

    # Gate 1 — the AP item itself (a hard state.has gate).
    lines.append(f"  Gate 1 - contract item held: {'YES' if item_held else 'NO'}")

    # Gate 2 — required part categories (clean booleans, immune to skill).
    breakdown = required_part_breakdown(spec, flags)
    if not breakdown:
        lines.append("  Gate 2 - required parts: (none)")
    else:
        lines.append("  Gate 2 - required parts:")
        for cat, got in breakdown:
            if got is None:
                lines.append(f"      {cat}: MISSING")
            else:
                lines.append(f"      {cat}: HAVE  {', '.join(titled(p.name) for p in got)}")

    # Gate 3 — physics can deliver the contract kit (computed up front, above).
    lines.append(f"  Gate 3 - physics delivery: {'YES' if feasible else 'NO'}")
    if result is None:
        lines.append("      (a required part category is unavailable -- see Gate 2)")
    elif not result.feasible and result.failure_reasons:
        if len(result.failure_reasons) == 1:
            lines.append(f"      reason: {result.failure_reasons[0]}")
        else:
            for r in result.failure_reasons:
                lines.append(f"      - {r}")

    if proxy:
        lines.append("")
        lines.append("  NOTE: goal contract on a model-infeasible body -- the access rule")
        lines.append("        substitutes the all-progression-items proxy for Gate 3 (the")
        lines.append("        physics verdict above is shown for reference only).")

    # Client-facing parameter tree (what the dumb client actuates). Rendered
    # generically from each primitive's wire form so new primitives need no
    # change here -- every parameter dataclass has a to_json().
    lines.append("")
    lines.append("  Contract parameters (client builds a native KSP contract):")
    for p in td.build_parameters(spec.body, mission_builder,
                                 part_manager=part_manager):
        j = dict(p.to_json())
        kind = j.pop("kind", "?")
        detail = ", ".join(f"{k}={v}" for k, v in j.items())
        lines.append(f"      - {kind}" + (f": {detail}" if detail else ""))

    # Delivery rocket — only when feasible; the contract parts ride
    # result.terminal_parts onto the terminal stage.  architecture_lines names
    # the flight architecture and the (possibly per-launch) mass, so an
    # assembly-delivered contract isn't mislabelled with a single launch mass.
    if feasible:
        lines.append("\n  --- Delivery rocket ---")
        lines.extend(architecture_lines(result))
        lines.extend(format_stage_breakdown(result, flags, td.base_mission_type))

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
    mission_builder: MissionBuilder,
    check_name: Optional[str] = None,
    user_description: Optional[str] = None,
) -> dict:
    """Build a JSON-serializable bug report dict."""
    difficulty_name = physics_profile_name_from_slot_data(slot_data)

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
        mission_builder,
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
                mission_builder,
                threshold_km=info.threshold_km,
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
