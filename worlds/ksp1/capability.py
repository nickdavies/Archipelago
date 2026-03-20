"""
Rocket capability evaluation system for KSP1 Archipelago.

Entry point: get_capability(state, player) → RocketCapability

Pipeline (per design doc):
  1. Pre-pass  — collect equipment flags from unlocked items.
  2. Profile evaluation — for each (body, mission_type), try all profile
     alternatives via a forward pre-filter + backward stage optimizer.
  3. Per-body assessment — populate BodyAccessProfile from profile results.
  4. Assemble — return the completed RocketCapability.

See ksp_archipelago_design.md for the full specification.
Golden rule: err toward saying something is NOT achievable rather than IS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from BaseClasses import CollectionState

from .bodies import (
    BODY_BY_NAME, MISSION_PROFILES, ALL_BODIES,
    DifficultyProfile, DIFFICULTY_PROFILES,
    Body, MissionEdge, EdgeType, effective_dv, parent_chain,
)
from .parts import (
    PART_DB, Engine, FuelTank, SolidBooster, HeatShield,
    Parachute, LandingLeg, Decoupler, MiscEquipment,
)
from .rocket_math import (
    StageResult, find_optimal_stage, terminal_velocity,
    FILL_LEVELS, merge_edge_groups,
)

if TYPE_CHECKING:
    from .world import KSP1World

_CACHE_KEY = "ksp1_capability"

# Terminal velocity threshold for parachute adequacy (m/s)
_MAX_SAFE_LANDING_SPEED: float = 6.0

# Ship cross-section assumed for parachute calc: π*(1.25/2)² ≈ 1.23 m²
# (conservative: assume a 1.25m diameter capsule/probe)
_SHIP_CROSS_SECTION: float = math.pi * (1.25 / 2) ** 2

# Solar distance threshold for ION engines
_ION_MAX_SOLAR_AU: float = 1.0  # only consider Dawn closer than Kerbin

# Minimum jetpack TWR for ladder-free sample return
_MIN_EVA_JETPACK_TWR: float = 1.05

# Bodies that are purely orbital (cannot land regardless of equipment)
_ORBITAL_ONLY_BODIES: frozenset[str] = frozenset({"Jool", "Kerbol"})

# Interplanetary gate: any mission leaving Kerbin SOI requires launch clamps
_INTERPLANETARY_BODIES: frozenset[str] = frozenset({
    "Moho", "Eve", "Gilly", "Duna", "Ike", "Dres",
    "Jool", "Laythe", "Vall", "Tylo", "Bop", "Pol",
    "Eeloo", "Kerbol",
})


# ---------------------------------------------------------------------------
# EquipmentFlags — output of pre-pass
# ---------------------------------------------------------------------------

@dataclass
class EquipmentFlags:
    # Binary presence flags
    has_heat_shield: bool = False
    has_parachutes: bool = False
    has_reaction_wheels: bool = False
    has_rcs: bool = False
    has_probe_core: bool = False
    has_capsule: bool = False
    has_rtg: bool = False
    has_solar: bool = False         # any solar (fixed or retractable)
    has_solar_retractable: bool = False
    has_solar_array_large: bool = False
    has_battery_large: bool = False
    has_docking_port: bool = False
    has_fuel_lines: bool = False
    has_ladder: bool = False
    has_launch_clamp: bool = False
    has_isru: bool = False

    # Tiered values
    landing_leg_tier: int = 0       # 0 = no legs
    relay_tier: int = 0             # 0 = no relay
    staging_tier: int = 0           # 0=none, 1=stack, 2=radial, 3=docking

    # Derived values
    max_heat_shield_size: Optional[float] = None   # largest heat shield size_class
    best_heat_shield_mass: float = 0.0             # mass of that shield
    total_chute_drag_area: float = 0.0             # sum of non-drogue drag areas
    parachute_count: int = 0
    heaviest_capsule_mass: float = 0.0
    lightest_probe_mass: float = float("inf")

    # Solar distance for ION logic (set from the edge being evaluated)
    target_solar_au: float = 1.0

    # Available part lists (populated by pre-pass)
    available_engines: list[Engine] = field(default_factory=list)
    available_srbs: list[SolidBooster] = field(default_factory=list)
    available_tanks: list[FuelTank] = field(default_factory=list)
    available_heat_shields: list[HeatShield] = field(default_factory=list)
    available_parachutes: list[Parachute] = field(default_factory=list)
    available_landing_legs: list[LandingLeg] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level result dataclasses (public API)
# ---------------------------------------------------------------------------

@dataclass
class BodyAccessProfile:
    can_orbit_low: bool = False
    can_orbit_high: bool = False
    can_land_unmanned: bool = False
    can_land_crewed: bool = False
    can_return_to_kerbin: bool = False
    can_return_crewed: bool = False
    can_sample_return: bool = False
    blocking_reason: Optional[str] = None


@dataclass
class RocketCapability:
    # Equipment flags (mirrors EquipmentFlags for external access)
    has_heat_shield: bool = False
    has_parachutes: bool = False
    landing_leg_tier: int = 0
    has_reaction_wheels: bool = False
    has_rcs: bool = False
    has_probe_core: bool = False
    has_capsule: bool = False
    has_rtg: bool = False
    has_isru: bool = False
    has_docking_port: bool = False
    relay_tier: int = 0
    power_profile: str = "none"
    staging_tier: int = 0
    has_launch_clamp: bool = False

    # Per-body assessments
    bodies: dict[str, BodyAccessProfile] = field(default_factory=dict)

    # Stage detail list (last computed profile, for debugging)
    stage_results: list[StageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache accessor — only public entry point for rules
# ---------------------------------------------------------------------------

def get_capability(state: CollectionState, player: int) -> RocketCapability:
    """
    Return the cached RocketCapability for this state/player, computing it
    if cold.  This is the only function that access rules should call.
    """
    cache: dict[int, RocketCapability] = state.prog_items.setdefault(_CACHE_KEY, {})
    if player not in cache:
        cache[player] = _compute_capability(state, player)
    return cache[player]


# ---------------------------------------------------------------------------
# Step 1: Pre-pass
# ---------------------------------------------------------------------------

def _pre_pass(state: CollectionState, player: int,
              start_with_clamps: bool) -> EquipmentFlags:
    """
    Iterate every item the player has collected and build EquipmentFlags.
    """
    flags = EquipmentFlags()

    # Grant launch clamps if the option is enabled
    if start_with_clamps:
        flags.has_launch_clamp = True

    for item_name, parts in PART_DB.items():
        count = state.count(item_name, player)
        if count == 0:
            continue

        for part in parts:
            if isinstance(part, Engine):
                flags.available_engines.append(part)

            elif isinstance(part, SolidBooster):
                flags.available_srbs.append(part)

            elif isinstance(part, FuelTank):
                flags.available_tanks.append(part)

            elif isinstance(part, HeatShield):
                flags.has_heat_shield = True
                flags.available_heat_shields.append(part)
                if flags.max_heat_shield_size is None or \
                        part.size_class > flags.max_heat_shield_size:
                    flags.max_heat_shield_size = part.size_class
                    flags.best_heat_shield_mass = part.mass

            elif isinstance(part, Parachute):
                if not part.is_drogue:  # drogue chutes excluded from all logic
                    flags.has_parachutes = True
                    flags.total_chute_drag_area += part.drag_area * count
                    flags.parachute_count += count
                    flags.available_parachutes.extend([part] * count)

            elif isinstance(part, LandingLeg):
                if part.tier > flags.landing_leg_tier:
                    flags.landing_leg_tier = part.tier
                flags.available_landing_legs.append(part)

            elif isinstance(part, Decoupler):
                if part.kind == "stack" and flags.staging_tier < 1:
                    flags.staging_tier = 1
                elif part.kind == "radial" and flags.staging_tier < 2:
                    flags.staging_tier = 2

            elif isinstance(part, MiscEquipment):
                _apply_misc(flags, part, count)

    # Docking port upgrades staging tier to 3
    if flags.has_docking_port and flags.staging_tier < 3:
        flags.staging_tier = 3

    # Asparagus requires both radial decouplers AND fuel lines
    # (staging_tier=2 is only valid asparagus if has_fuel_lines)
    # The optimizer checks flags.staging_tier >= 2 AND flags.has_fuel_lines

    # Derive relay tier from available relays
    flags.relay_tier = _compute_relay_tier(flags)

    # Capsule/probe mass defaults if not found
    if flags.lightest_probe_mass == float("inf"):
        flags.lightest_probe_mass = 0.0  # no probe: will be blocked by gate

    return flags


def _apply_misc(flags: EquipmentFlags, part: MiscEquipment, count: int) -> None:
    for flag in part.provides:
        if flag == "probe_core":
            flags.has_probe_core = True
            if part.mass < flags.lightest_probe_mass:
                flags.lightest_probe_mass = part.mass
        elif flag == "capsule":
            flags.has_capsule = True
            if part.mass > flags.heaviest_capsule_mass:
                flags.heaviest_capsule_mass = part.mass
        elif flag == "reaction_wheel":
            flags.has_reaction_wheels = True
        elif flag == "rcs":
            flags.has_rcs = True
        elif flag in ("solar_fixed", "solar_retractable"):
            flags.has_solar = True
            if flag == "solar_retractable":
                flags.has_solar_retractable = True
        elif flag == "solar_array_large":
            flags.has_solar_array_large = True
            flags.has_solar = True
        elif flag == "rtg":
            flags.has_rtg = True
        elif flag == "battery_large":
            flags.has_battery_large = True
        elif flag == "docking_port":
            flags.has_docking_port = True
        elif flag == "fuel_line":
            flags.has_fuel_lines = True
        elif flag == "ladder":
            flags.has_ladder = True
        elif flag == "launch_clamp":
            flags.has_launch_clamp = True
        elif flag == "isru":
            flags.has_isru = True


def _compute_relay_tier(flags: EquipmentFlags) -> int:
    # relay_t1/t2/t3 flags set via _apply_misc on MiscEquipment provides
    # We check which were granted via the provides mechanism — the flags
    # object tracks this through has_relay_t* attributes, but we collapsed
    # them into relay_tier during _apply_misc.  To avoid over-engineering,
    # we track relay tier via the highest relay flag seen.
    # Re-scan available engines is unnecessary; tier is set during _apply_misc.
    # The tier is already computed progressively — return current value.
    return flags.relay_tier


def _apply_misc_relay(flags: EquipmentFlags, flag: str) -> None:
    """Set relay tier from a provides flag string."""
    if flag == "relay_t1" and flags.relay_tier < 1:
        flags.relay_tier = 1
    elif flag == "relay_t2" and flags.relay_tier < 2:
        flags.relay_tier = 2
    elif flag == "relay_t3" and flags.relay_tier < 3:
        flags.relay_tier = 3


# Patch _apply_misc to also handle relay flags properly
_original_apply_misc = _apply_misc


def _apply_misc(flags: EquipmentFlags, part: MiscEquipment, count: int) -> None:  # type: ignore[no-redef]
    for flag in part.provides:
        if flag == "probe_core":
            flags.has_probe_core = True
            if part.mass < flags.lightest_probe_mass:
                flags.lightest_probe_mass = part.mass
        elif flag == "capsule":
            flags.has_capsule = True
            if part.mass > flags.heaviest_capsule_mass:
                flags.heaviest_capsule_mass = part.mass
        elif flag == "reaction_wheel":
            flags.has_reaction_wheels = True
        elif flag == "rcs":
            flags.has_rcs = True
        elif flag in ("solar_fixed", "solar_retractable"):
            flags.has_solar = True
            if flag == "solar_retractable":
                flags.has_solar_retractable = True
        elif flag == "solar_array_large":
            flags.has_solar_array_large = True
            flags.has_solar = True
        elif flag == "rtg":
            flags.has_rtg = True
        elif flag == "battery_large":
            flags.has_battery_large = True
        elif flag == "docking_port":
            flags.has_docking_port = True
        elif flag == "fuel_line":
            flags.has_fuel_lines = True
        elif flag == "ladder":
            flags.has_ladder = True
        elif flag == "launch_clamp":
            flags.has_launch_clamp = True
        elif flag == "isru":
            flags.has_isru = True
        elif flag.startswith("relay_"):
            _apply_misc_relay(flags, flag)


# ---------------------------------------------------------------------------
# Step 2: Profile evaluation
# ---------------------------------------------------------------------------

@dataclass
class ProfileResult:
    feasible: bool
    launch_mass: float = 0.0          # total wet mass at kerbin_surface
    stage_results: list[StageResult] = field(default_factory=list)
    failure_reason: str = ""


def _has_attitude_control(flags: EquipmentFlags) -> bool:
    """Any combination of gimbal (engine), reaction wheels, or RCS."""
    return (flags.has_reaction_wheels or flags.has_rcs
            or any(e.has_gimbal for e in flags.available_engines)
            or any(s.has_gimbal for s in flags.available_srbs))


def _check_power_for_body(flags: EquipmentFlags, body: Body,
                           after_aero: bool = False) -> bool:
    """
    Return True if the player has adequate power for operations at *body*.

    after_aero: if True, fixed solar panels were destroyed during aerobrake
                and cannot count toward power.  Exception: Kerbin reentry
                (recovery, no ongoing power needed).
    """
    req = body.power_requirement
    if req == "solar":
        if after_aero:
            return flags.has_rtg or flags.has_solar_retractable
        return flags.has_solar or flags.has_rtg
    elif req == "solar_marginal":
        if after_aero:
            return flags.has_rtg or flags.has_solar_retractable
        return flags.has_solar or flags.has_rtg
    elif req == "rtg":
        return flags.has_rtg
    return True


def _filter_engines_for_ion(engines: list[Engine],
                             solar_au: float,
                             flags: EquipmentFlags) -> list[Engine]:
    """
    Remove ION (xenon) engines from the list unless the target body is
    close enough to Kerbol AND the player has adequate power for sustained burns.
    """
    result = []
    ion_ok = (
        solar_au <= _ION_MAX_SOLAR_AU
        and (flags.has_battery_large or flags.has_solar_array_large)
    )
    for engine in engines:
        if engine.fuel_type == "xenon" and not ion_ok:
            continue
        result.append(engine)
    return result


def _evaluate_profile(
    profile: list[MissionEdge],
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_type: str,              # "orbit" | "land" | "return" | "sample_return"
    is_crewed: bool,
) -> ProfileResult:
    """
    Run the two-pass evaluation on a single mission profile alternative.

    Forward pass:  check broad category gates; compute effective dv per edge.
    Backward pass: walk in reverse, run optimizer per stage group, propagate mass.
    """
    # ------------------------------------------------------------------
    # Forward pass — broad gate checks
    # ------------------------------------------------------------------

    has_aero_edge = any(e.needs_heat_shield for e in profile)
    has_atmo_ascent = any(e.edge_type == EdgeType.ATMOSPHERIC_ASCENT for e in profile)
    has_vacuum_land = any(e.edge_type == EdgeType.VACUUM_LANDING for e in profile)
    has_atmo_land_aero = any(e.edge_type == EdgeType.ATMO_LANDING_AERO for e in profile)
    has_land = has_vacuum_land or has_atmo_land_aero or \
               any(e.edge_type == EdgeType.ATMO_LANDING_PROPULSIVE for e in profile)
    needs_legs = any(e.needs_landing_legs for e in profile)
    needs_ladder = any(e.needs_ladder for e in profile)

    # Command check
    if is_crewed:
        if not flags.has_capsule:
            return ProfileResult(False, failure_reason="no capsule for crewed mission")
    else:
        if not flags.has_probe_core:
            return ProfileResult(False, failure_reason="no probe core for unmanned mission")

    # Attitude control (required by almost every edge via requires_attitude_control)
    if any(e.requires_attitude_control for e in profile):
        if not _has_attitude_control(flags):
            return ProfileResult(False, failure_reason="no attitude control")

    # Landing legs
    if needs_legs:
        leg_bodies = [
            BODY_BY_NAME[e.body] for e in profile if e.needs_landing_legs
        ]
        required_tier = max((b.landing_leg_tier for b in leg_bodies), default=0)
        if flags.landing_leg_tier < required_tier:
            return ProfileResult(False,
                failure_reason=f"need leg tier {required_tier}, have {flags.landing_leg_tier}")

    # Ladder
    if needs_ladder and not flags.has_ladder:
        return ProfileResult(False, failure_reason="need ladder for sample return")

    # Heat shield
    if has_aero_edge and not flags.has_heat_shield:
        return ProfileResult(False, failure_reason="no heat shield for aero edge")

    # Parachutes (broad check: any parachutes at all for aero landing)
    if has_atmo_land_aero and not flags.has_parachutes:
        return ProfileResult(False, failure_reason="no parachutes for aero landing")

    # Power: check each unique body in the profile
    # Detect if aero edges destroy fixed solar panels
    body_aero_destroyed: dict[str, bool] = {}  # body -> whether fixed solar destroyed
    post_aero = False
    for edge in profile:
        is_kerbin_reentry = (edge.destination == "kerbin_surface" and
                             edge.edge_type == EdgeType.ATMO_LANDING_AERO)
        if edge.needs_heat_shield and not is_kerbin_reentry:
            post_aero = True
        if edge.body not in body_aero_destroyed:
            body_aero_destroyed[edge.body] = post_aero
        elif post_aero:
            body_aero_destroyed[edge.body] = True

    for body_name, after_aero in body_aero_destroyed.items():
        body = BODY_BY_NAME[body_name]
        if not _check_power_for_body(flags, body, after_aero=after_aero):
            return ProfileResult(False,
                failure_reason=f"insufficient power at {body_name} "
                               f"(after_aero={after_aero})")

    # Relay tier
    for edge in profile:
        body = BODY_BY_NAME[edge.body]
        if flags.relay_tier < body.min_relay_tier:
            return ProfileResult(False,
                failure_reason=f"relay tier too low for {body.name}: "
                               f"need {body.min_relay_tier}, have {flags.relay_tier}")

    # ------------------------------------------------------------------
    # Stage grouping
    # ------------------------------------------------------------------
    # Build stage groups by splitting on staging opportunities.
    # Each group is a list of consecutive edges that share a stage.
    # Kerbin ascent is always its own stage.

    groups = _group_edges(profile, flags.staging_tier)

    # ------------------------------------------------------------------
    # Backward pass — compute masses from destination back to Kerbin
    # ------------------------------------------------------------------

    # Terminal payload mass
    if is_crewed:
        terminal_mass = max(flags.heaviest_capsule_mass, 0.08)  # min capsule
    else:
        terminal_mass = max(flags.lightest_probe_mass, 0.04)    # min probe

    # Add equipment mass for the terminal stage
    # (legs, ladder, heat shield on the last edge in the profile)
    terminal_equip = _terminal_equipment_mass(profile, flags)
    payload = terminal_mass + terminal_equip

    stage_results_list: list[StageResult] = []
    consumed_chutes = 0

    for group in reversed(groups):
        body = BODY_BY_NAME[group[0].body]
        solar_au = body.solar_distance_au

        # Filter ION engines for this body
        eligible_engines = _filter_engines_for_ion(
            flags.available_engines, solar_au, flags
        )

        # Compute effective dv with difficulty margins
        base_dv = sum(e.base_dv for e in group)
        pc_dv = sum(e.plane_change_dv for e in group)
        req_dv = effective_dv(base_dv, diff, plane_change_dv=pc_dv)

        # Group-level constraints (union = strictest)
        from .bodies import EdgeType as ET
        atmo_types = {
            ET.ATMOSPHERIC_ASCENT, ET.ATMO_LANDING_PROPULSIVE,
            ET.ATMO_LANDING_AERO, ET.AEROBRAKE_CAPTURE,
        }
        in_atmo = any(e.edge_type in atmo_types for e in group)
        min_twr = max(e.min_twr for e in group)
        # Apply difficulty-specific TWR floors
        if min_twr > 0:
            if in_atmo:
                min_twr = max(min_twr, diff.min_twr_atmo)
            else:
                min_twr = max(min_twr, diff.min_twr_vac)
        req_throttle = any(e.requires_throttleable for e in group)
        needs_hs = any(e.needs_heat_shield for e in group)
        needs_legs_g = any(e.needs_landing_legs for e in group)

        # Equipment mass for this stage
        equip_mass = 0.0
        if needs_hs:
            equip_mass += flags.best_heat_shield_mass
        if needs_legs_g:
            leg_body = BODY_BY_NAME[next(e.body for e in group if e.needs_landing_legs)]
            equip_mass += _leg_mass_for_tier(flags, leg_body.landing_leg_tier)

        # Parachute consumption check for aero landing edges in this group
        aero_land_edges = [e for e in group if e.edge_type == ET.ATMO_LANDING_AERO]
        if aero_land_edges:
            needed = _required_chute_count(
                payload + equip_mass, body, flags, diff
            )
            if needed < 0:
                return ProfileResult(False, failure_reason="parachute terminal velocity check failed")
            consumed_chutes += needed
            if consumed_chutes > flags.parachute_count:
                return ProfileResult(False,
                    failure_reason=f"need {consumed_chutes} chutes, have {flags.parachute_count}")

        asparagus = (flags.staging_tier >= 2 and flags.has_fuel_lines)

        result = find_optimal_stage(
            available_engines=eligible_engines,
            available_srbs=flags.available_srbs,
            available_tanks=flags.available_tanks,
            required_dv=req_dv,
            payload_mass=payload,
            gravity=body.surface_gravity,
            min_twr=min_twr,
            requires_throttleable=req_throttle,
            needs_heat_shield=needs_hs,
            max_heat_shield_size=flags.max_heat_shield_size,
            heat_shield_mass=equip_mass if needs_hs else 0.0,
            in_atmosphere=in_atmo,
            asparagus=asparagus,
            srb_needs_rcs=diff.srb_needs_rcs,
            player_has_rcs=flags.has_rcs,
        )

        if result is None:
            return ProfileResult(False,
                failure_reason=f"no viable stage for group dv={req_dv:.0f} m/s at {body.name}")

        stage_results_list.append(result)
        # The stage's wet mass becomes the payload for the next stage back
        payload = result.stage_mass_wet

    return ProfileResult(
        feasible=True,
        launch_mass=payload,
        stage_results=list(reversed(stage_results_list)),
    )


def _group_edges(profile: list[MissionEdge], staging_tier: int) -> list[list[MissionEdge]]:
    """
    Group consecutive edges into stages based on staging_tier.

    staging_tier=0: everything is one stage (no decouplers)
    staging_tier>=1: split at natural staging points

    Natural stage boundaries:
      - After Kerbin ascent (always own stage if staging_tier >= 1)
      - Before and after landing (always own stage if staging_tier >= 1)
      - Before any vacuum-to-atmo transition

    When the number of available stage breaks > staging_tier allows, merge
    the smallest adjacent groups first (conservative: bigger combined dv = harder).
    """
    if staging_tier == 0:
        return [profile]

    # Mark preferred split indices (after edge i, before edge i+1)
    from .bodies import EdgeType as ET
    splits: list[int] = []

    for i in range(len(profile) - 1):
        cur = profile[i]
        nxt = profile[i + 1]

        # Always split after Kerbin atmospheric ascent
        if cur.edge_type == ET.ATMOSPHERIC_ASCENT and cur.body == "Kerbin":
            splits.append(i)
            continue

        # Split before any landing edge
        if nxt.edge_type in (ET.VACUUM_LANDING, ET.ATMO_LANDING_PROPULSIVE,
                              ET.ATMO_LANDING_AERO):
            splits.append(i)
            continue

        # Split after any landing edge (ascent from surface is separate)
        if cur.edge_type in (ET.VACUUM_LANDING, ET.ATMO_LANDING_PROPULSIVE,
                              ET.ATMO_LANDING_AERO):
            splits.append(i)
            continue

    # Remove duplicates and sort
    splits = sorted(set(splits))

    # Build groups
    groups: list[list[MissionEdge]] = []
    start = 0
    for split_idx in splits:
        groups.append(profile[start:split_idx + 1])
        start = split_idx + 1
    groups.append(profile[start:])

    # Remove empty groups
    groups = [g for g in groups if g]

    # If we have more groups than staging_tier allows, merge smallest pairs
    # Max stages = staging_tier + 1 (e.g. tier 1 = 2 stages: booster + payload)
    max_stages = staging_tier + 1
    while len(groups) > max_stages:
        # Find the pair with the smallest combined dv to merge
        best_pair = 0
        best_dv = float("inf")
        for i in range(len(groups) - 1):
            combined = sum(e.base_dv for e in groups[i]) + sum(e.base_dv for e in groups[i + 1])
            if combined < best_dv:
                best_dv = combined
                best_pair = i
        merged = groups[best_pair] + groups[best_pair + 1]
        groups = groups[:best_pair] + [merged] + groups[best_pair + 2:]

    return groups


def _terminal_equipment_mass(profile: list[MissionEdge],
                              flags: EquipmentFlags) -> float:
    """
    Equipment mass carried all the way to the terminal destination.
    This is the mass of legs / ladder / heat shield on the final landing.
    """
    # Find the last landing edge (if any) — its equipment rides with the terminal payload
    from .bodies import EdgeType as ET
    mass = 0.0
    for edge in reversed(profile):
        if edge.needs_landing_legs:
            body = BODY_BY_NAME.get(edge.body)
            if body:
                mass += _leg_mass_for_tier(flags, body.landing_leg_tier)
            break
    # Ladder
    if any(e.needs_ladder for e in profile):
        mass += 0.005  # Pegasus ladder mass
    return mass


def _leg_mass_for_tier(flags: EquipmentFlags, required_tier: int) -> float:
    """Return the mass of the lightest available legs that meet the tier."""
    for leg in sorted(flags.available_landing_legs, key=lambda l: l.mass):
        if leg.tier >= required_tier:
            return leg.mass * 4  # conservative: 4 legs per landing
    # Fall back: if we have any legs at all, use them
    if flags.available_landing_legs:
        return min(l.mass for l in flags.available_landing_legs) * 4
    return 0.0


def _required_chute_count(
    landing_mass: float,
    body: Body,
    flags: EquipmentFlags,
    diff: DifficultyProfile,
) -> int:
    """
    Return the number of Mk16 parachutes needed to achieve terminal velocity
    <= _MAX_SAFE_LANDING_SPEED.

    Uses pessimistic mass estimate (landing_mass + all chutes) to avoid
    under-counting (golden rule).

    Returns -1 if no configuration of available chutes achieves the threshold.
    """
    if not body.has_atmosphere or body.atm_density_kg_m3 <= 0:
        return 0  # vacuum body — no chutes needed

    # Pessimistic upper bound: add mass of all available chutes
    if not flags.available_parachutes:
        return -1

    non_drogue = [p for p in flags.available_parachutes if not p.is_drogue]
    if not non_drogue:
        return -1

    chute = non_drogue[0]  # use the first available (all Mk16s in dummy DB)
    max_chutes = flags.parachute_count

    for n in range(1, max_chutes + 1):
        total_mass = landing_mass + chute.mass * n
        drag_area = chute.drag_area * n
        v_term = terminal_velocity(
            total_mass, body.surface_gravity,
            body.atm_density_kg_m3,
            diff.ship_cd, _SHIP_CROSS_SECTION, drag_area,
        )
        if v_term <= _MAX_SAFE_LANDING_SPEED:
            return n

    return -1  # even all chutes aren't enough


# ---------------------------------------------------------------------------
# Step 3: Per-body assessment
# ---------------------------------------------------------------------------

def _assess_bodies(
    flags: EquipmentFlags,
    diff: DifficultyProfile,
) -> dict[str, BodyAccessProfile]:
    """
    Populate a BodyAccessProfile for every body in ALL_BODIES.

    Evaluation order respects the parent-gating rule: if a planet orbit is
    unreachable, all its moons are immediately marked False.
    """
    results: dict[str, BodyAccessProfile] = {}

    # We need to evaluate planets before moons.
    # ALL_BODIES is ordered: Kerbin first, then moons, then outer bodies.
    # Process in dependency order: planets first, then moons.
    planets = [b for b in ALL_BODIES if b.parent is None]
    moons = [b for b in ALL_BODIES if b.parent is not None]

    for body in planets + moons:
        results[body.name] = _assess_one_body(body, flags, diff, results)

    return results


def _assess_one_body(
    body: Body,
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    computed: dict[str, BodyAccessProfile],
) -> BodyAccessProfile:
    prof = BodyAccessProfile()

    # --- Parent gating ---
    if body.parent is not None:
        parent_prof = computed.get(body.parent)
        # For Kerbin moons: parent orbit must be reachable (Kerbin orbit always is)
        # For other moons: parent planet must have can_orbit_low
        if body.parent != "Kerbin" and parent_prof is not None:
            if not parent_prof.can_orbit_low:
                prof.blocking_reason = f"parent {body.parent} orbit unreachable"
                return prof

    # --- Kerbin is always accessible (starting body) ---
    if body.name == "Kerbin":
        prof.can_orbit_low = True
        prof.can_orbit_high = True
        prof.can_land_unmanned = True
        prof.can_land_crewed = flags.has_capsule
        prof.can_return_to_kerbin = True
        prof.can_return_crewed = flags.has_capsule
        prof.can_sample_return = flags.has_capsule
        return prof

    # --- Launch clamp gate for interplanetary ---
    if body.name in _INTERPLANETARY_BODIES and not flags.has_launch_clamp:
        prof.blocking_reason = "no launch clamp for interplanetary mission"
        return prof

    # --- Orbit ---
    orbit_profiles = MISSION_PROFILES.get((body.name, "orbit"), [])
    orbit_ok = _try_profiles(orbit_profiles, flags, diff, "orbit", crewed=False)
    prof.can_orbit_low = orbit_ok
    prof.can_orbit_high = orbit_ok  # trivially extends from LKO

    if not orbit_ok:
        prof.blocking_reason = "orbit not achievable"
        return prof

    # --- Landing (unmanned) ---
    if body.can_land and body.name not in _ORBITAL_ONLY_BODIES:
        land_profiles = MISSION_PROFILES.get((body.name, "land"), [])
        prof.can_land_unmanned = _try_profiles(
            land_profiles, flags, diff, "land", crewed=False
        )

        # Landing (crewed)
        if flags.has_capsule:
            prof.can_land_crewed = _try_profiles(
                land_profiles, flags, diff, "land", crewed=True
            )

    # --- Return (unmanned) ---
    return_profiles = MISSION_PROFILES.get((body.name, "return"), [])
    if return_profiles:
        prof.can_return_to_kerbin = _try_profiles(
            return_profiles, flags, diff, "return", crewed=False
        )

        # Return (crewed)
        if flags.has_capsule:
            prof.can_return_crewed = _try_profiles(
                return_profiles, flags, diff, "return", crewed=True
            )

    # --- Sample return (crewed + ladder check) ---
    if body.can_land and body.name not in _ORBITAL_ONLY_BODIES:
        sr_profiles = MISSION_PROFILES.get((body.name, "sample_return"), [])
        if sr_profiles and flags.has_capsule:
            # Inject ladder requirement if EVA jetpack can't lift off
            if body.eva_jetpack_twr < _MIN_EVA_JETPACK_TWR:
                sr_profiles_with_ladder = _inject_ladder(sr_profiles)
            else:
                sr_profiles_with_ladder = sr_profiles
            prof.can_sample_return = _try_profiles(
                sr_profiles_with_ladder, flags, diff, "sample_return", crewed=True
            )

    return prof


def _inject_ladder(profiles: list[list[MissionEdge]]) -> list[list[MissionEdge]]:
    """
    Return copies of the profiles with needs_ladder=True on any landing edge.
    Used when the body's EVA jetpack TWR is too low for unassisted reentry.
    """
    import dataclasses
    result = []
    for profile in profiles:
        new_profile = []
        for edge in profile:
            if edge.needs_landing_legs:
                edge = dataclasses.replace(edge, needs_ladder=True)
            new_profile.append(edge)
        result.append(new_profile)
    return result


def _try_profiles(
    profiles: list[list[MissionEdge]],
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_type: str,
    crewed: bool,
) -> bool:
    """Return True if any profile alternative is feasible."""
    for profile in profiles:
        result = _evaluate_profile(profile, flags, diff, mission_type, is_crewed=crewed)
        if result.feasible:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 4: Assemble RocketCapability
# ---------------------------------------------------------------------------

def _compute_capability(state: CollectionState, player: int) -> RocketCapability:
    """Full capability computation from the current collection state."""
    # Retrieve world options
    world = state.multiworld.worlds[player]
    options = world.options

    difficulty_name = ["casual", "normal", "expert", "insane"][options.difficulty.value]
    diff = DIFFICULTY_PROFILES[difficulty_name]
    start_with_clamps = bool(options.start_with_launch_clamps.value)

    flags = _pre_pass(state, player, start_with_clamps)
    body_profiles = _assess_bodies(flags, diff)

    # Determine power profile string for the RocketCapability summary
    if flags.has_rtg:
        power_str = "rtg"
    elif flags.has_solar_retractable:
        power_str = "solar_retractable"
    elif flags.has_solar:
        power_str = "solar"
    else:
        power_str = "none"

    cap = RocketCapability(
        has_heat_shield=flags.has_heat_shield,
        has_parachutes=flags.has_parachutes,
        landing_leg_tier=flags.landing_leg_tier,
        has_reaction_wheels=flags.has_reaction_wheels,
        has_rcs=flags.has_rcs,
        has_probe_core=flags.has_probe_core,
        has_capsule=flags.has_capsule,
        has_rtg=flags.has_rtg,
        has_isru=flags.has_isru,
        has_docking_port=flags.has_docking_port,
        relay_tier=flags.relay_tier,
        power_profile=power_str,
        staging_tier=flags.staging_tier,
        has_launch_clamp=flags.has_launch_clamp,
        bodies=body_profiles,
    )

    return cap
