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
from typing import TYPE_CHECKING, Callable, Optional

from BaseClasses import CollectionState

from .bodies import (
    BODY_BY_NAME, ALL_BODIES,
    BodyName, MissionType, DifficultyProfile, DIFFICULTY_PROFILES,
    Body, MissionEdge, MissionBuilder, EdgeType,
    effective_dv, home_system_bodies, parent_chain,
)
from .parts import (
    PART_DB, CapabilityFlag, Engine, FuelTank, SolidBooster, HeatShield,
    Parachute, LandingLeg, Decoupler, MiscEquipment,
    MultiMount, MULTI_MOUNT_TABLE,
    PART_TO_CONTRACT_CATEGORIES,
    PROGRESSIVE_PART_TIERS, PROGRESSIVE_PART_NAMES, PROGRESSIVE_PART_COUNTS,
    usable_fuel_mass,
)
from .capability_reasons import BlockingInfo, BlockingReason
from .locations import (
    ALL_EVENTS, EVENT_BY_NAME, EventName,
    get_body_events,
)
from .rocket_math import (
    StageResult, find_optimal_stage, find_optimal_multistage_ascent,
    terminal_velocity,
    FILL_LEVELS, merge_edge_groups,
)

if TYPE_CHECKING:
    from .world import KSP1World

# NOTE: cross-validation that every body/event has a profile now lives inside
# ``MissionBuilder._validate`` (see bodies.py).  It runs once at the first
# ``MissionBuilder(BodyName.KERBIN)`` construction (world.generate_early /
# test-file module-level builders), so registration gaps surface eagerly there.

# Item names that affect capability computation. Built once at module load
# from PART_DB: any item containing an Engine, FuelTank, SolidBooster,
# HeatShield, Parachute, LandingLeg, Decoupler, or MiscEquipment with
# non-empty `provides`.  Includes parts in progressive chains (they're useful
# items in the pool and individually receivable).
_CAPABILITY_PART_TYPES = (
    Engine, FuelTank, SolidBooster, HeatShield,
    Parachute, LandingLeg, Decoupler, MiscEquipment,
)
CAPABILITY_ITEMS: frozenset[str] = frozenset(
    name for name, parts in PART_DB.items()
    if any(
        isinstance(p, _CAPABILITY_PART_TYPES) and (
            not isinstance(p, MiscEquipment) or p.provides
        )
        for p in parts
    )
) | {
    # Counted progressives that still affect capability — counts must
    # appear in the L2 fingerprint to distinguish e.g. Pad=0 vs Pad=3.
    "Progressive Launch Pad",
    "Progressive R&D",
    "Progressive Science Instrument",
}

# Terminal velocity threshold for parachute adequacy (m/s)
_MAX_SAFE_LANDING_SPEED: float = 6.0

# Ship cross-section assumed for parachute calc: π*(1.25/2)² ≈ 1.23 m²
# (conservative: assume a 1.25m diameter capsule/probe)
_SHIP_CROSS_SECTION: float = math.pi * (1.25 / 2) ** 2

# Precomputed once: the part name + mass that provides FUEL_LINE (the
# asparagus crossfeed enabler).  The real parallel-stage builder needs the
# fuel line's mass/name to charge an asparagus crossfeed build.
_FUEL_LINE_PART: Optional[str] = None
_FUEL_LINE_MASS: float = 0.0
for _nm, _parts in PART_DB.items():
    _fl = next((_p for _p in _parts if isinstance(_p, MiscEquipment)
                and CapabilityFlag.FUEL_LINE in _p.provides), None)
    if _fl is not None:
        _FUEL_LINE_PART = _nm
        _FUEL_LINE_MASS = _fl.mass
        break
del _nm, _parts

# Minimum jetpack TWR for ladder-free sample return
_MIN_EVA_JETPACK_TWR: float = 1.05


# Sounding rocket parameters
_SOUNDING_MIN_TWR: float = 1.1   # minimum sea-level TWR to count as a viable rocket



# ---------------------------------------------------------------------------
# Capsule eligibility is data-driven: ``parts.py:_capsule_spec_from_cfg``
# returns ``None`` for srf-only crew positions (chair-style), and that
# strips the ``capsule`` provides flag at MiscEquipment construction.
# No hand-curated exclusion list needed here.
# ---------------------------------------------------------------------------


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
    has_thermometer: bool = False
    has_barometer: bool = False
    has_wheel: bool = False
    has_throttleable_engine: bool = False
    has_aero_control_surface: bool = False
    _has_inline_chute: bool = False     # internal: cap inline chutes at 1

    # Progressive-item binary gates (set by _pre_pass from progressive counts)
    has_launch_engine: bool = False     # Progressive Launch Engine ≥1
    has_vacuum_engine: bool = False     # Progressive Vacuum Engine ≥1
    has_lfo_fuel: bool = False          # Progressive LFO Tank ≥1
    has_srb_fuel: bool = False          # Progressive SRB ≥1

    # Tiered values
    landing_leg_tier: int = 0       # 0 = no legs
    relay_tier: int = 0             # 0 = no relay
    staging_tier: int = 0           # 0=none, 1=stack, 2=radial

    # Part references — None means not available.
    # Each stores the selected part so both .mass and .name are accessible.
    best_heat_shield: Optional[HeatShield] = None
    total_chute_drag_area: float = 0.0             # sum of non-drogue drag areas
    parachute_count: int = 0
    lightest_capsule: Optional[MiscEquipment] = None
    lightest_probe: Optional[MiscEquipment] = None

    # Support equipment — part references per category
    lightest_relay: dict[int, MiscEquipment] = field(default_factory=dict)  # tier → part
    lightest_solar: Optional[MiscEquipment] = None
    lightest_solar_retractable: Optional[MiscEquipment] = None
    lightest_rtg: Optional[MiscEquipment] = None
    lightest_aero_control: Optional[MiscEquipment] = None
    lightest_ladder: Optional[MiscEquipment] = None
    # Per-stage attitude control parts. Used when a stage needs attitude
    # control (`requires_attitude_control` edge) AND the terminal payload
    # has no built-in reaction wheels AND the chosen propulsion lacks gimbal.
    # `lightest_reaction_wheel` and `lightest_rcs_thruster` are standalone
    # parts (their `provides` excludes probe_core/capsule). `lightest_monoprop_tank`
    # is the lightest unlocked FuelTank with fuel_type="monoprop" (skipped when
    # the terminal command part already supplies monopropellant).
    lightest_reaction_wheel: Optional[MiscEquipment] = None
    lightest_rcs_thruster: Optional[MiscEquipment] = None
    lightest_monoprop_tank: Optional[FuelTank] = None

    # Lightest available part per contract part-category (e.g. "drill",
    # "ore_tank"). Populated generically from PART_TO_CONTRACT_CATEGORIES — no
    # hardcoded capability flag per part. Contracts read these to size the
    # required-equipment payload and to gate on part presence.
    category_lightest: dict[str, MiscEquipment] = field(default_factory=dict)
    # All available crew_cabin parts (real pressurized pods/cabins — excludes
    # the external command seat). The station contract needs the cheapest
    # combination reaching N seats, so it needs the full list, not just the
    # lightest.
    available_crew_parts: list[MiscEquipment] = field(default_factory=list)

    # Solar distance for ION logic (set from the edge being evaluated)
    target_solar_au: float = 1.0

    # Launch-pad mass cap (tonnes). Default is unlimited; set by progressive
    # launch-pad tier when the option is enabled. Missions whose computed
    # launch mass exceeds this are infeasible.
    launch_pad_mass_cap: float = float("inf")

    # Available part lists (populated by pre-pass)
    available_engines: list[Engine] = field(default_factory=list)
    available_srbs: list[SolidBooster] = field(default_factory=list)
    available_tanks: list[FuelTank] = field(default_factory=list)
    available_heat_shields: list[HeatShield] = field(default_factory=list)
    available_parachutes: list[Parachute] = field(default_factory=list)
    # Asymptote-best non-drogue parachute (lowest mass-per-drag-area)
    # picked once at ``_pre_pass`` time so ``_required_chute_count``
    # doesn't ``min(...)`` per call.  ``None`` when the player has no
    # non-drogue chutes.
    best_chute: Optional[Parachute] = None
    available_landing_legs: list[LandingLeg] = field(default_factory=list)

    # Multi-mount adapters/plates available to the player
    available_multi_mounts: list[MultiMount] = field(default_factory=list)

    # Aero control surfaces available (elevon/fin/winglet)
    available_aero_controls: list[MiscEquipment] = field(default_factory=list)

    # Decouplers available to the player (for CLI display)
    available_decouplers: list[Decoupler] = field(default_factory=list)

    # Pre-indexed tanks by fuel type (built once after pre-pass)
    tanks_by_fuel_type: Optional[dict[str, list[FuelTank]]] = None


# ---------------------------------------------------------------------------
# Top-level result dataclasses (public API)
# ---------------------------------------------------------------------------

@dataclass
class BodyAccessProfile:
    access: dict[EventName, bool] = field(
        default_factory=lambda: {ev.name: False for ev in ALL_EVENTS}
    )
    # Structured blocking info; the first entry's __str__ is what
    # `blocking_reason` reports. Mutation goes through ``set_blocking``
    # (or direct ``blocking.append``); the legacy string attribute is
    # derived.
    blocking: list[BlockingInfo] = field(default_factory=list)

    @property
    def blocking_reason(self) -> Optional[str]:
        """Back-compat string view of the first blocking entry."""
        return str(self.blocking[0]) if self.blocking else None

    def set_blocking(self, info: BlockingInfo) -> None:
        """Replace any prior reasons with a single structured one."""
        self.blocking = [info]


def event_mission_info(event_name: str) -> tuple[str, bool | None]:
    """Return (mission_type, crewed) for an event. Used by CLI scripts."""
    ev = EVENT_BY_NAME[event_name]
    return (ev.mission_type, ev.crewed)


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
    has_thermometer: bool = False
    has_barometer: bool = False
    has_wheel: bool = False
    has_throttleable_engine: bool = False
    has_aero_control_surface: bool = False

    # Sounding rocket: best achievable altitude (km) with a single stage
    sounding_altitude_km: float = 0.0

    # Per-body assessments
    bodies: dict[BodyName, BodyAccessProfile] = field(default_factory=dict)

    # Contract feasibility, computed once per state (keyed by ContractSpec.contract_id):
    # True iff the player has the required parts AND can deliver the contract's
    # equipment payload to the body for its base mission. Access rules read this
    # as an O(1) dict lookup. Empty when the world has no contracts.
    contract_access: dict[str, bool] = field(default_factory=dict)

    # Stage detail list (last computed profile, for debugging)
    stage_results: list[StageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache accessor — only public entry point for rules
# ---------------------------------------------------------------------------

def _capability_fingerprint(state: CollectionState, player: int) -> frozenset[tuple[str, int]]:
    """Return the set of (name, count) for capability-affecting items.

    Phase 2: counts matter for the surviving counted progressives
    (Pad / R&D / PSI); every other item is a binary presence/absence.
    """
    from .items import (
        PROGRESSIVE_LAUNCH_PAD_NAME, PROGRESSIVE_LAUNCH_PAD_COUNT,
        PROGRESSIVE_RD_NAME, PROGRESSIVE_RD_COUNT,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME, PROGRESSIVE_PSI_COUNT,
    )
    _COUNTED = {
        PROGRESSIVE_LAUNCH_PAD_NAME: PROGRESSIVE_LAUNCH_PAD_COUNT,
        PROGRESSIVE_RD_NAME: PROGRESSIVE_RD_COUNT,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: PROGRESSIVE_PSI_COUNT,
    }
    result: list[tuple[str, int]] = []
    for name in CAPABILITY_ITEMS:
        c = state.count(name, player)
        if c > 0:
            cap = _COUNTED.get(name)
            result.append((name, min(c, cap) if cap is not None else 1))
    return frozenset(result)


def get_capability(state: CollectionState, player: int) -> RocketCapability:
    """
    Return the cached RocketCapability for this state/player, computing it
    if the cache is cold or stale.

    Two-level cache:
      L1 — stale flag on CollectionState (via LogicMixin). Within a single
            sweep step, multiple rule checks reuse the same result for free.
      L2 — equipment fingerprint on World object. When stale, compute which
            capability-affecting items the player has. If another state already
            computed the same set, reuse that result without running the full
            capability pipeline.
    """
    # L1: not stale → return state-local cached result
    if not state.ksp1_cap_stale[player]:
        return state.ksp1_cap_result[player]

    # L2: check fingerprint cache on the World object
    fingerprint = _capability_fingerprint(state, player)
    world: KSP1World = state.multiworld.worlds[player]

    result = world.capability_cache.get(fingerprint)
    if result is None:
        result = _compute_capability(state, player)
        world.capability_cache[fingerprint] = result

    state.ksp1_cap_result[player] = result
    state.ksp1_cap_stale[player] = False
    return result


def explain_body_unreachable(state: CollectionState, player: int, body_name: str) -> str:
    """
    Return a human-readable explanation of why *body_name* is not fully
    accessible given the current collection state.  Useful for debugging
    test failures and in-game diagnostics.
    """
    cap = get_capability(state, player)
    bp = cap.bodies.get(body_name)
    if bp is None:
        return f"{body_name}: not evaluated (not in mission graph)"
    if bp.blocking_reason:
        return f"{body_name}: {bp.blocking_reason}"
    lines = []
    for ev in ALL_EVENTS:
        if not bp.access[ev.name]:
            lines.append(f"cannot {ev.name}")
    return f"{body_name}: " + ("; ".join(lines) if lines else "fully accessible")


# ---------------------------------------------------------------------------
# Step 1: Pre-pass
# ---------------------------------------------------------------------------

def _pre_pass(item_count_fn: Callable[[str], int],
              start_with_clamps: bool,
              progressive_launch_pad: bool = False,
              launch_pad_caps: tuple[float, ...] | None = None) -> EquipmentFlags:
    """
    Iterate every PART_DB item the player has and build EquipmentFlags.

    Phase 2: parts are individual AP items — there is no progressive-tier
    expansion.  ``item_count_fn(item_name)`` returns 1 if the player has
    received that part and 0 otherwise.  The ``Progressive Launch Pad``
    counted progressive is the only non-binary count consulted (drives
    ``launch_pad_mass_cap``).
    """
    flags = EquipmentFlags()

    if start_with_clamps:
        flags.has_launch_clamp = True

    # Launch-pad mass cap: index by collected count of "Progressive Launch Pad".
    if progressive_launch_pad:
        from .items import PROGRESSIVE_LAUNCH_PAD_NAME, PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN
        caps = launch_pad_caps or PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN
        pad_count = item_count_fn(PROGRESSIVE_LAUNCH_PAD_NAME)
        idx = min(pad_count, len(caps) - 1)
        flags.launch_pad_mass_cap = caps[idx]

    # Process every PART_DB item.
    for item_name, parts in PART_DB.items():
        count = item_count_fn(item_name)
        if count == 0:
            continue
        for part in parts:
            _add_part_to_flags(flags, part, count)

    # Binary gate flags are derived from concrete parts (no more progressive
    # binary checks).
    flags.has_launch_engine = any(e.atm_thrust > 0 for e in flags.available_engines)
    flags.has_vacuum_engine = any(e.vac_thrust > 0 for e in flags.available_engines)
    flags.has_lfo_fuel = any(t.fuel_type == "lfo" for t in flags.available_tanks)
    flags.has_srb_fuel = bool(flags.available_srbs)

    # Docking ports don't affect staging_tier — they can't be used for
    # practical stage separation (can't attach below engines, no automatic
    # staging). They set has_docking_port for future orbital assembly support.

    # Parallel staging mode (determined per-stage in _evaluate_profile):
    #   staging_tier >= 2 + fuel lines → asparagus (real radial crossfeed build)
    #   staging_tier >= 2, no fuel lines → onion (radial ring drop)
    #   staging_tier < 2 → none (no parallel staging)

    # Derive relay tier from available relays
    flags.relay_tier = _compute_relay_tier(flags)

    # Pick the asymptote-best parachute once.  Lowest mass-per-drag-area
    # wins (see ``_required_chute_count`` doc).  Drogues are included —
    # the physics-accurate landing check (``_required_chute_count``)
    # already handles "drag insufficient even with all chutes" by
    # returning -1; the boolean ``has_parachutes`` was a redundant
    # shortcut that surprised the bumper into picking a non-drogue rep
    # at higher ranks when a drogue-only kit could land with enough copies.
    if flags.available_parachutes:
        flags.best_chute = min(flags.available_parachutes,
                                key=lambda p: p.mass / max(p.drag_area, 1e-3))

    # (lightest_probe=None when no probe found — gate blocks before use)

    # Sort engines by Isp desc, mass asc — high-Isp lightweight engines
    # produce lighter stages, helping the optimizer's upper-bound pruning.
    flags.available_engines.sort(key=lambda e: (-e.vac_isp, e.mass))

    # Throttleable engine flag (for powered landing)
    flags.has_throttleable_engine = any(e.throttleable for e in flags.available_engines)

    # Build fuel-type index for tanks, keyed by engine fuel_type (consumer).
    # Each tank appears under every engine fuel_type it can fuel. For tanks
    # carrying propellants the engine doesn't need (e.g. LFO tank fueling a
    # NERV that only consumes LiquidFuel), we synthesize a view tank with
    # reduced fuel_mass — same dry mass, oxidizer drained. MonoPropellant
    # cannot be drained, so monoprop-bearing tanks only fuel monoprop engines.
    #
    # Deduplicate by (engine_fuel_type, dry_mass, effective_fuel_mass, size_class).
    # Sort each bucket by ratio (fuel/dry) desc then fuel_mass asc — the
    # optimizer's best_wet upper-bound pruning finds good solutions fastest
    # when high-ratio small tanks are tried first.
    engine_propellants_by_ft: dict[str, frozenset[str]] = {}
    for engine in flags.available_engines:
        engine_propellants_by_ft.setdefault(
            engine.fuel_type, frozenset(engine.propellants)
        )
    tank_index: dict[str, list[FuelTank]] = {}
    seen_tank_stats: set[tuple[str, float, float, float]] = set()
    for engine_ft, engine_props in engine_propellants_by_ft.items():
        bucket: list[FuelTank] = []
        for tank in flags.available_tanks:
            eff_mass = usable_fuel_mass(tank, engine_props)
            if eff_mass <= 0:
                continue
            key = (engine_ft, tank.dry_mass, eff_mass, tank.size_class)
            if key in seen_tank_stats:
                continue
            seen_tank_stats.add(key)
            if eff_mass == tank.fuel_mass:
                bucket.append(tank)
            else:
                # Synthetic view: same physical tank, reduced fuel mass.
                bucket.append(FuelTank(
                    name=tank.name,
                    dry_mass=tank.dry_mass,
                    fuel_mass=eff_mass,
                    fuel_type=engine_ft,
                    size_class=tank.size_class,
                    max_count=tank.max_count,
                    fuel_masses=tank.fuel_masses,
                ))
        if bucket:
            tank_index[engine_ft] = bucket
    for fuel_type in tank_index:
        tank_index[fuel_type].sort(
            key=lambda t: (
                -(t.fuel_mass / t.dry_mass if t.dry_mass > 0 else 0),
                t.fuel_mass,
            )
        )
    flags.tanks_by_fuel_type = tank_index

    # Select the lightest monoprop tank (loaded mass = dry + fuel). Used as the
    # second half of an RCS attitude bundle when the terminal command part
    # doesn't supply its own monopropellant.
    for tank in flags.available_tanks:
        if tank.fuel_type != "monoprop":
            continue
        loaded = tank.dry_mass + tank.fuel_mass
        if (flags.lightest_monoprop_tank is None
                or (flags.lightest_monoprop_tank.dry_mass
                    + flags.lightest_monoprop_tank.fuel_mass) > loaded):
            flags.lightest_monoprop_tank = tank

    return flags


def _add_part_to_flags(flags: EquipmentFlags, part, count: int) -> None:
    """Add a single part object to EquipmentFlags."""
    if isinstance(part, Engine):
        flags.available_engines.append(part)

    elif isinstance(part, SolidBooster):
        flags.available_srbs.append(part)

    elif isinstance(part, FuelTank):
        flags.available_tanks.append(part)

    elif isinstance(part, HeatShield):
        flags.has_heat_shield = True
        flags.available_heat_shields.append(part)
        if flags.best_heat_shield is None or part.size_class > flags.best_heat_shield.size_class:
            flags.best_heat_shield = part

    elif isinstance(part, Parachute):
        # Drogues now count toward parachute capability — the
        # ``_required_chute_count`` physics check decides if drag area
        # is sufficient (drogues need many copies to land a craft, but
        # the math handles that).  Treating drogues as no-op was a
        # gameplay-incorrect shortcut.
        if part.is_radial:
            flags.has_parachutes = True
            flags.total_chute_drag_area += part.drag_area * count
            flags.parachute_count += count
            flags.available_parachutes.extend([part] * count)
        elif not flags._has_inline_chute:
            flags._has_inline_chute = True
            flags.has_parachutes = True
            flags.total_chute_drag_area += part.drag_area
            flags.parachute_count += 1
            flags.available_parachutes.append(part)

    elif isinstance(part, LandingLeg):
        if part.tier > flags.landing_leg_tier:
            flags.landing_leg_tier = part.tier
        flags.available_landing_legs.append(part)

    elif isinstance(part, Decoupler):
        if part.kind == "stack" and flags.staging_tier < 1:
            flags.staging_tier = 1
        elif part.kind == "radial" and flags.staging_tier < 2:
            flags.staging_tier = 2
        flags.available_decouplers.append(part)

    elif isinstance(part, MiscEquipment):
        _apply_misc(flags, part, count)


def _apply_misc_relay(flags: EquipmentFlags, flag: CapabilityFlag, part: MiscEquipment) -> None:
    """Set relay tier and track lightest relay per tier."""
    CF = CapabilityFlag
    tier = {CF.RELAY_T1: 1, CF.RELAY_T2: 2, CF.RELAY_T3: 3, CF.RELAY_T4: 4}.get(flag, 0)
    if tier == 0:
        return
    if tier > flags.relay_tier:
        flags.relay_tier = tier
    existing = flags.lightest_relay.get(tier)
    if existing is None or part.mass < existing.mass:
        flags.lightest_relay[tier] = part


def _compute_relay_tier(flags: EquipmentFlags) -> int:
    """Return the relay tier already computed by _apply_misc."""
    return flags.relay_tier


def _apply_misc(flags: EquipmentFlags, part: MiscEquipment, count: int) -> None:
    CF = CapabilityFlag
    # Generic contract-category membership — tracked by part name, independent
    # of the provides flags (drills/ore tanks carry no provides). Keeps the
    # lightest available part per category for contract payload sizing.
    for cat_key in PART_TO_CONTRACT_CATEGORIES.get(part.name, ()):
        cur = flags.category_lightest.get(cat_key)
        if cur is None or part.mass < cur.mass:
            flags.category_lightest[cat_key] = part
    # A station's crew rides real pressurized cabins, not exposed external
    # seats: gate on crew_cabin category membership (which excludes
    # seatExternalCmd), matching category_lightest above. A raw crew_capacity
    # test admits a "station" of 5 lawn chairs (~0.25t), wrecking the mass model.
    if "crew_cabin" in PART_TO_CONTRACT_CATEGORIES.get(part.name, ()):
        flags.available_crew_parts.append(part)
    for flag in part.provides:
        if flag == CF.PROBE_CORE:
            flags.has_probe_core = True
            if flags.lightest_probe is None or part.mass < flags.lightest_probe.mass:
                flags.lightest_probe = part
        elif flag == CF.CAPSULE:
            flags.has_capsule = True
            if flags.lightest_capsule is None or part.mass < flags.lightest_capsule.mass:
                flags.lightest_capsule = part
        elif flag == CF.REACTION_WHEEL:
            flags.has_reaction_wheels = True
            # Standalone wheel module = provides reaction_wheel without also
            # providing probe_core or capsule. (Probes/capsules that include
            # a wheel are handled by `_terminal_has_built_in_wheels`; they
            # cover every stage automatically.)
            if (CF.PROBE_CORE not in part.provides
                    and CF.CAPSULE not in part.provides):
                if (flags.lightest_reaction_wheel is None
                        or part.mass < flags.lightest_reaction_wheel.mass):
                    flags.lightest_reaction_wheel = part
        elif flag == CF.RCS:
            flags.has_rcs = True
            if (flags.lightest_rcs_thruster is None
                    or part.mass < flags.lightest_rcs_thruster.mass):
                flags.lightest_rcs_thruster = part
        elif flag in (CF.SOLAR_FIXED, CF.SOLAR_RETRACTABLE):
            flags.has_solar = True
            if flags.lightest_solar is None or part.mass < flags.lightest_solar.mass:
                flags.lightest_solar = part
            if flag == CF.SOLAR_RETRACTABLE:
                flags.has_solar_retractable = True
                if flags.lightest_solar_retractable is None or part.mass < flags.lightest_solar_retractable.mass:
                    flags.lightest_solar_retractable = part
        elif flag == CF.SOLAR_ARRAY_LARGE:
            flags.has_solar_array_large = True
            flags.has_solar = True
            flags.has_solar_retractable = True
            if flags.lightest_solar is None or part.mass < flags.lightest_solar.mass:
                flags.lightest_solar = part
            if flags.lightest_solar_retractable is None or part.mass < flags.lightest_solar_retractable.mass:
                flags.lightest_solar_retractable = part
        elif flag == CF.RTG:
            flags.has_rtg = True
            if flags.lightest_rtg is None or part.mass < flags.lightest_rtg.mass:
                flags.lightest_rtg = part
        elif flag == CF.BATTERY_LARGE:
            flags.has_battery_large = True
        elif flag == CF.DOCKING_PORT:
            flags.has_docking_port = True
        elif flag == CF.FUEL_LINE:
            flags.has_fuel_lines = True
        elif flag == CF.LADDER:
            flags.has_ladder = True
            if flags.lightest_ladder is None or part.mass < flags.lightest_ladder.mass:
                flags.lightest_ladder = part
        elif flag == CF.LAUNCH_CLAMP:
            flags.has_launch_clamp = True
        elif flag == CF.ISRU:
            flags.has_isru = True
        elif flag == CF.MULTI_MOUNT:
            mount = MULTI_MOUNT_TABLE.get(part.name)
            if mount is not None:
                flags.available_multi_mounts.append(mount)
        elif flag == CF.THERMOMETER:
            flags.has_thermometer = True
        elif flag == CF.BAROMETER:
            flags.has_barometer = True
        elif flag == CF.WHEEL:
            flags.has_wheel = True
        elif flag == CF.AERO_CONTROL:
            flags.has_aero_control_surface = True
            flags.available_aero_controls.append(part)
            if flags.lightest_aero_control is None or part.mass < flags.lightest_aero_control.mass:
                flags.lightest_aero_control = part
        elif flag.startswith("relay_"):
            _apply_misc_relay(flags, flag, part)


# ---------------------------------------------------------------------------
# Step 2: Profile evaluation
# ---------------------------------------------------------------------------

@dataclass
class KitUsed:
    """Complete set of parts the capability evaluator depended on to
    reach feasibility.  Populated only on ``ProfileResult.feasible``.

    Distinguishes three categories of part:

    1. **Explicit picks**: the engines, tanks, and per-stage equipment
       the optimizer selected (recorded per stage).
    2. **Implicit picks**: the ``lightest_*`` / ``best_*`` parts
       capability pulled in for terminal payload (capsule, probe),
       reentry (heat shield, parachute), landing (legs), support
       (relay antennas, power source), and attitude control
       (reaction wheel, RCS thruster + tank).
    3. **Presence-only representatives**: parts whose mere availability
       enabled a boolean / tier flag the optimizer relied on but didn't
       directly consume — e.g. a radial decoupler enables the radial
       asparagus/onion build, a
       ``fuelLine`` enables ``has_fuel_lines`` (asparagus mode), an SRB
       enables ``has_srb_fuel`` even when the optimal stage was
       liquid-only.  These flags affect the search bounds; without
       them in the rep set, the re-evaluation can't reproduce the same
       optimal kit.

    ``all_parts()`` returns the union — feed that into
    ``_pre_pass_for_ranks(..., reps_only=...)`` and the resulting flags
    object will be functionally equivalent to the one this kit was
    extracted from, so the feasibility claim is reproducible.
    """
    # 1. Explicit
    stage_engines: list[str] = field(default_factory=list)
    stage_tanks: list[str] = field(default_factory=list)
    stage_equipment: list[str] = field(default_factory=list)
    # 2. Implicit
    capsule: Optional[str] = None
    probe_core: Optional[str] = None
    parachute: Optional[str] = None
    heat_shields: list[str] = field(default_factory=list)
    landing_legs: list[str] = field(default_factory=list)
    relays: list[str] = field(default_factory=list)
    rtg: Optional[str] = None
    solar: Optional[str] = None
    solar_retractable: Optional[str] = None
    monoprop_tank: Optional[str] = None
    rcs_thruster: Optional[str] = None
    reaction_wheel: Optional[str] = None
    aero_control: Optional[str] = None
    ladder: Optional[str] = None
    # 3. Presence-only representatives
    stack_decoupler: Optional[str] = None
    radial_decoupler: Optional[str] = None
    fuel_line: Optional[str] = None
    srb: Optional[str] = None
    # Large-power enabler for ION (xenon) engines — see
    # _filter_engines_for_ion.  Without it in the rep set, a re-eval
    # filters the ion engine out (NO_VIABLE_STAGE).
    ion_power: Optional[str] = None

    # Per-role viable substitutes for the chosen pick.  Keyed by the
    # KitUsed field name (e.g. "capsule", "radial_decoupler"); values are
    # the set of part names that satisfy the same physical role at a
    # similar quality level.  Derived from the rank model (same axis-rank)
    # for ranked parts; from capability-flag membership for presence-only
    # roles.  Bumper / chain code can ``rng.choice`` among the union of
    # ``{chosen} ∪ alternates`` to vary per-seed picks while staying
    # within feasibility (verification is the caller's responsibility —
    # rank-equivalence is necessary but not sufficient for cascading
    # mission profiles).
    alternates: dict[str, frozenset[str]] = field(default_factory=dict)

    # Per-stage propulsion alternates: list aligned with stage_engines /
    # stage_tanks.  Each element is the set of viable substitutes for
    # that stage's pick.
    stage_engine_alternates: list[frozenset[str]] = field(default_factory=list)
    stage_tank_alternates: list[frozenset[str]] = field(default_factory=list)

    def all_parts(self) -> frozenset[str]:
        out: set[str] = set()
        out.update(self.stage_engines)
        out.update(self.stage_tanks)
        out.update(self.stage_equipment)
        out.update(self.landing_legs)
        out.update(self.heat_shields)
        out.update(self.relays)
        for v in (self.capsule, self.probe_core, self.parachute,
                  self.rtg, self.solar,
                  self.solar_retractable, self.monoprop_tank,
                  self.rcs_thruster, self.reaction_wheel,
                  self.aero_control, self.ladder,
                  self.stack_decoupler, self.radial_decoupler,
                  self.fuel_line, self.srb, self.ion_power):
            if v:
                out.add(v)
        return frozenset(out)


@dataclass
class ProfileResult:
    feasible: bool
    launch_mass: float = 0.0          # total wet mass at kerbin_surface
    stage_results: list[StageResult] = field(default_factory=list)
    edge_groups: list[list[MissionEdge]] = field(default_factory=list)
    # Index into ``edge_groups`` for each entry of ``stage_results`` (same
    # order).  Usually 1:1, but a multi-stage ascent group expands into K
    # stages that all map back to the single ascent group — so the formatter
    # must use this rather than zipping ``stage_results``/``edge_groups`` by
    # position (which silently misaligns every stage above the ascent).
    stage_group_indices: list[int] = field(default_factory=list)
    # Structured blocking info; ``failure_reasons`` is the legacy string
    # view derived from ``blocking``. Producers populate ``blocking``;
    # downstream consumers can read either.
    blocking: list[BlockingInfo] = field(default_factory=list)
    # Command module + support equipment for the terminal stage: [(count, part_id), ...]
    terminal_parts: list[tuple[int, str]] = field(default_factory=list)
    # Complete structured kit (populated on feasible results, see KitUsed).
    kit_used: Optional[KitUsed] = None
    # Best-effort partial rocket captured when the result is INFEASIBLE:
    # the stages that did build (terminal -> as far up the ascent as the
    # optimizer got) before the binding stage failed.  Lets the bumper /
    # analysis layer examine the near-miss architecture (e.g. a heavy
    # terminal stage driving a mass cascade) instead of only seeing the
    # single failing-stage diagnostic.  Empty on feasible results.
    partial_stages: list[StageResult] = field(default_factory=list)

    @property
    def failure_reasons(self) -> list[str]:
        """Back-compat string view of ``blocking``."""
        return [str(b) for b in self.blocking]


# Precomputed once: the part name that provides FUEL_LINE (asparagus
# enabler).  Scanning PART_DB inside _build_kit_used per call was a hot-
# loop regression (called for every feasible eval).
_FUEL_LINE_PART: Optional[str] = None
_FUEL_LINE_MASS: float = 0.0
for _nm, _parts in PART_DB.items():
    _fl = next((_p for _p in _parts if isinstance(_p, MiscEquipment)
                and CapabilityFlag.FUEL_LINE in _p.provides), None)
    if _fl is not None:
        _FUEL_LINE_PART = _nm
        _FUEL_LINE_MASS = _fl.mass
        break
del _nm, _parts


def _lightest_part_providing(flag: "CapabilityFlag") -> Optional[str]:
    """Lightest PART_DB item that provides *flag*, by part mass.  Used to
    capture presence-only enablers (large battery / solar array) in the
    kit so a re-eval can reproduce the flag state."""
    best: Optional[tuple[str, float]] = None
    for nm, parts in PART_DB.items():
        for p in parts:
            if flag in getattr(p, "provides", ()):  # type: ignore[arg-type]
                if best is None or p.mass < best[1]:
                    best = (nm, p.mass)
                break
    return best[0] if best else None


# Precomputed once: lightest parts that enable ION (xenon) engines via the
# large-power gate in ``_filter_engines_for_ion``.  Battery is preferred —
# it carries no rank axis, so adding it to a kit doesn't inflate any rank
# ceiling (the large solar panel sits at SOLAR rank 3).
_BATTERY_LARGE_PART: Optional[str] = _lightest_part_providing(CapabilityFlag.BATTERY_LARGE)
_SOLAR_LARGE_PART: Optional[str] = _lightest_part_providing(CapabilityFlag.SOLAR_ARRAY_LARGE)


def _build_kit_used(flags: EquipmentFlags,
                    stage_results: list[StageResult],
                    terminal_parts: list[tuple[int, str]]) -> KitUsed:
    """Assemble the full structured kit the optimizer relied on.

    See ``KitUsed`` for the categorization.  Called only at the two
    sites that consume a kit — the per-mission ceiling computation
    (sphere_ladder loop 1) and the capability-guided rescue — NOT on
    every feasible eval.  Building it for the bumper's ~60k feasibility
    probes per seed was pure overhead.
    """
    kit = KitUsed()
    # 1. Explicit per-stage parts
    for sr in stage_results:
        if sr.engine_name and sr.engine_name != "none":
            kit.stage_engines.append(sr.engine_name)
        for _, tname in sr.tank_manifest:
            if tname and tname != "none":
                kit.stage_tanks.append(tname)
        for _, p in sr.equipment:
            if p:
                kit.stage_equipment.append(p)
    # 2. Implicit lightest_/best_ picks
    if flags.lightest_capsule:
        kit.capsule = flags.lightest_capsule.name
    if flags.lightest_probe:
        kit.probe_core = flags.lightest_probe.name
    if flags.best_chute:
        kit.parachute = flags.best_chute.name
    # Heat shields: capture the exact shield each stage charged (lightest
    # covering that stage's engine) — NOT the global biggest — so a re-eval
    # has the same shield options and reproduces the optimizer's choice.
    _seen_hs: set[str] = set()
    for sr in stage_results:
        if sr.heat_shield_name and sr.heat_shield_name not in _seen_hs:
            _seen_hs.add(sr.heat_shield_name)
            kit.heat_shields.append(sr.heat_shield_name)
    if flags.lightest_rtg:
        kit.rtg = flags.lightest_rtg.name
    if flags.lightest_solar:
        kit.solar = flags.lightest_solar.name
    if flags.lightest_solar_retractable:
        kit.solar_retractable = flags.lightest_solar_retractable.name
    if flags.lightest_monoprop_tank:
        kit.monoprop_tank = flags.lightest_monoprop_tank.name
    if flags.lightest_rcs_thruster:
        kit.rcs_thruster = flags.lightest_rcs_thruster.name
    if flags.lightest_reaction_wheel:
        kit.reaction_wheel = flags.lightest_reaction_wheel.name
    if flags.lightest_aero_control:
        kit.aero_control = flags.lightest_aero_control.name
    if flags.lightest_ladder:
        kit.ladder = flags.lightest_ladder.name
    # Relays: keep every tier the mission needed (terminal_parts captured
    # only the lightest, but we want each tier represented so the chain
    # can attribute them correctly on the relay rank axis).
    for tier, part in flags.lightest_relay.items():
        if part:
            kit.relays.append(part.name)
    # All admitted landing legs — the optimizer picks per body, and the
    # bumper's rank axis stretches across the tier ladder, so include
    # every option capability had.
    for leg in flags.available_landing_legs:
        kit.landing_legs.append(leg.name)
    # 3. Presence-only representatives.  Pick the lightest matching part
    # for each True flag that affects stage optimization — without these
    # in the rep set, re-eval can't reproduce the same flag-state and
    # the optimizer's choices fall apart (e.g. dropping the radial decoupler
    # + fuel line would disable the asparagus build the optimizer chose).
    if flags.staging_tier >= 1:
        stack = [d for d in flags.available_decouplers if d.kind == "stack"]
        if stack:
            kit.stack_decoupler = min(stack, key=lambda d: d.mass).name
    if flags.staging_tier >= 2:
        radial = [d for d in flags.available_decouplers if d.kind == "radial"]
        if radial:
            kit.radial_decoupler = min(radial, key=lambda d: d.mass).name
    if flags.has_fuel_lines and _FUEL_LINE_PART is not None:
        kit.fuel_line = _FUEL_LINE_PART
    if flags.has_srb_fuel and flags.available_srbs:
        kit.srb = min(flags.available_srbs,
                      key=lambda s: s.dry_mass + s.fuel_mass).name
    # ION power gate: if the optimizer used a xenon engine in any stage,
    # the kit must carry the large-power enabler it relied on, else a
    # re-eval filters the ion engine out (NO_VIABLE_STAGE).
    _xenon_names = {e.name for e in flags.available_engines
                    if e.fuel_type == "xenon"}
    if _xenon_names and any(se in _xenon_names for se in kit.stage_engines):
        if flags.has_battery_large and _BATTERY_LARGE_PART is not None:
            kit.ion_power = _BATTERY_LARGE_PART
        elif flags.has_solar_array_large and _SOLAR_LARGE_PART is not None:
            kit.ion_power = _SOLAR_LARGE_PART
    return kit


def build_kit_for_result(flags: EquipmentFlags,
                         result: "ProfileResult") -> Optional[KitUsed]:
    """Build the structured KitUsed from a feasible ProfileResult + the
    flags it was evaluated against.  Returns ``None`` if the result is
    infeasible.  This is the explicit entry point for the two consumers
    (per-mission ceiling, rescue) now that ``_evaluate_profile`` no
    longer builds the kit eagerly."""
    if not result.feasible:
        return None
    return _build_kit_used(flags, result.stage_results, result.terminal_parts)


def _has_attitude_control(flags: EquipmentFlags) -> bool:
    """Any combination of gimbal (engine), reaction wheels, or RCS."""
    return (flags.has_reaction_wheels or flags.has_rcs
            or any(e.has_gimbal for e in flags.available_engines)
            or any(s.has_gimbal for s in flags.available_srbs))


# Command pods / probes that ship MonoPropellant in their own tankage (stock
# KSP). When the terminal payload is one of these, an RCS attitude bundle
# doesn't need a separate monoprop tank — the pod supplies the fuel itself.
# Derived from data/parts.json (resources containing MonoPropellant on parts
# that aren't pure tanks). Keep in sync if KSP adds new monoprop-carrying pods.
_TERMINAL_PARTS_WITH_INTERNAL_MONOPROP: frozenset[str] = frozenset({
    "MEMLander",
    "Mark1Cockpit",
    "Mark2Cockpit",
    "cupola",
    "landerCabinSmall",
    "mk1-3pod",
    "mk1pod.v2",
    "mk2Cockpit.Inline",
    "mk2Cockpit.Standard",
    "mk2LanderCabin.v2",
    "mk3Cockpit.Shuttle",
})


def _terminal_part(flags: EquipmentFlags, is_crewed: bool) -> Optional[MiscEquipment]:
    return flags.lightest_capsule if is_crewed else flags.lightest_probe


def _terminal_has_built_in_wheels(flags: EquipmentFlags, is_crewed: bool) -> bool:
    part = _terminal_part(flags, is_crewed)
    if part is None:
        return False
    return CapabilityFlag.REACTION_WHEEL in part.provides


def _terminal_has_built_in_monoprop(flags: EquipmentFlags, is_crewed: bool) -> bool:
    part = _terminal_part(flags, is_crewed)
    if part is None:
        return False
    return part.name in _TERMINAL_PARTS_WITH_INTERNAL_MONOPROP


@dataclass(frozen=True)
class AttitudeBundle:
    """Concrete on-stage attitude-control parts (and their summed mass).

    Whatever mass we charge to the optimizer here MUST equal the sum of the
    real part masses listed in `parts`. Both fields are consumed together —
    `mass` goes to `find_optimal_stage(attitude_module_mass=...)` and `parts`
    is appended to the stage's equipment manifest when a non-gimballed
    propulsion choice triggers the charge.
    """
    mass: float
    parts: tuple[tuple[int, str], ...]   # ((count, internal_name), ...)


# Module-level kill switch for the per-stage attitude bundle. Used by the
# regression harness to A/B against pre-fix behaviour without editing code.
# Production code must leave this True.
_PER_STAGE_ATTITUDE_ENABLED: bool = True


def _attitude_bundle_for_stage(
    flags: EquipmentFlags, is_crewed: bool,
) -> Optional[AttitudeBundle]:
    """Pick the lightest concrete on-stage attitude bundle from real PART_DB
    parts. Returns None if no on-stage source is available.

    Two candidate bundles, real masses only:
      - **Wheel**: 1 × lightest standalone reaction-wheel module.
      - **RCS**:   4 × lightest standalone RCS thruster
                   + (1 × lightest monoprop tank, unless the terminal payload
                      already supplies MonoPropellant internally).

    The lighter of the two wins. The chosen parts (with counts) appear
    verbatim in the stage manifest so manifest mass == charged mass.
    """
    candidates: list[AttitudeBundle] = []
    if flags.lightest_reaction_wheel is not None:
        w = flags.lightest_reaction_wheel
        candidates.append(AttitudeBundle(
            mass=w.mass,
            parts=((1, w.name),),
        ))
    if flags.lightest_rcs_thruster is not None:
        t = flags.lightest_rcs_thruster
        rcs_parts: list[tuple[int, str]] = [(4, t.name)]
        rcs_mass = 4 * t.mass
        rcs_skip = False
        if not _terminal_has_built_in_monoprop(flags, is_crewed):
            tank = flags.lightest_monoprop_tank
            if tank is None:
                # Can't fly an RCS bundle without monopropellant — skip.
                rcs_skip = True
            else:
                rcs_parts.append((1, tank.name))
                rcs_mass += tank.dry_mass + tank.fuel_mass
        if not rcs_skip:
            candidates.append(AttitudeBundle(
                mass=rcs_mass,
                parts=tuple(rcs_parts),
            ))
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.mass)


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
    Remove ION (xenon) engines from the logic entirely.

    Ion's ~4200s Isp is a ~12x outlier: mass-minimisation always crowns it
    for any dv-bound mission, so it becomes the de-facto required engine and
    flattens per-seed variance.  It's also never *needed* — every mission is
    reachable with chemical/nuclear, just heavier (and the launch-pad ladder
    is sized for those non-ion masses).  So ion is out of logic: capability
    never relies on it.  It stays in the item pool as an out-of-logic bonus
    the player can fly if they collect it (an acceptable false-negative under
    the conservative golden rule).  ``solar_au`` is retained for signature
    stability but no longer gates anything.
    """
    return [e for e in engines if e.fuel_type != "xenon"]


def _evaluate_profile(
    profile: list[MissionEdge],
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_type: MissionType,
    is_crewed: bool,
    home: BodyName,
    extra_payload_parts: tuple[MiscEquipment, ...] = (),
    run_parallel: bool = True,
) -> ProfileResult:
    """
    Run the two-pass evaluation on a single mission profile alternative.

    Forward pass:  check broad category gates; compute effective dv per edge.
    Backward pass: walk in reverse, run optimizer per stage group, propagate mass.

    ``home`` is the player's starting body; used by the relay-tier gate
    (heliocentric distance to each edge body).  Test callers can rely on
    the Kerbin default.

    ``extra_payload_parts`` is contract-required equipment (e.g. a drill + ore
    tank) that must be *delivered* to the destination. Their summed mass is
    added to the terminal payload — so every stage below carries it — and the
    parts are listed on the terminal manifest. Default empty = ordinary mission.
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

    # Collect all pre-check failures before attempting stage optimization.
    blocking: list[BlockingInfo] = []

    # Command check
    if is_crewed:
        if not flags.has_capsule:
            blocking.append(BlockingInfo(reason=BlockingReason.NO_CAPSULE))
    else:
        if not flags.has_probe_core:
            blocking.append(BlockingInfo(reason=BlockingReason.NO_PROBE_CORE))

    # Attitude control (required by almost every edge via requires_attitude_control)
    if any(e.requires_attitude_control for e in profile):
        if not _has_attitude_control(flags):
            blocking.append(BlockingInfo(reason=BlockingReason.NO_ATTITUDE_CONTROL))

    # Landing legs
    if needs_legs:
        leg_bodies = [
            BODY_BY_NAME[e.body] for e in profile if e.needs_landing_legs
        ]
        required_tier = max((b.landing_leg_tier for b in leg_bodies), default=0)
        if flags.landing_leg_tier < required_tier:
            blocking.append(BlockingInfo(
                reason=BlockingReason.LANDING_LEGS_MISSING,
                leg_tier_needed=required_tier,
                leg_tier_available=flags.landing_leg_tier,
            ))

    # Ladder
    if needs_ladder and not flags.has_ladder:
        blocking.append(BlockingInfo(reason=BlockingReason.NO_LADDER))

    # Heat shield
    if has_aero_edge and not flags.has_heat_shield:
        blocking.append(BlockingInfo(reason=BlockingReason.NO_HEAT_SHIELD))

    # Parachutes (broad check: any parachutes at all for aero landing)
    if has_atmo_land_aero and not flags.has_parachutes:
        blocking.append(BlockingInfo(reason=BlockingReason.NO_PARACHUTE))

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
            # "solar_helps" iff body.power_requirement is solar-based
            # AND we're not in a post-aero state where only retractable/rtg
            # would work.  This drives sphere-ladder bump priority (solar
            # vs rtg).
            solar_helps = body.power_requirement in ("solar", "solar_marginal") \
                          and not after_aero
            blocking.append(BlockingInfo(
                reason=(BlockingReason.INSUFFICIENT_POWER_SOLAR_OK
                        if solar_helps
                        else BlockingReason.INSUFFICIENT_POWER_NEEDS_RTG),
                body=body_name,
                after_aero=after_aero,
                solar_helps=solar_helps,
            ))

    # Relay tier — baked into ``MissionEdge.relay_tier`` at
    # MissionBuilder construction (the tier is a function of edge body
    # and the world's home, both fixed at that point).  The hot loop
    # reads a struct field — no dict lookup, no function call.
    #
    # Crewed missions skip the gate: a pilot in a manned capsule provides
    # control authority directly with no radio link to home.
    if not is_crewed:
        for edge in profile:
            required = edge.relay_tier
            if flags.relay_tier < required:
                blocking.append(BlockingInfo(
                    reason=BlockingReason.RELAY_TIER_TOO_LOW,
                    body=edge.body,
                    relay_needed=required,
                    relay_available=flags.relay_tier,
                ))
                break  # one relay failure is sufficient

    # Propulsion gate: bail if the player has no engines or fuel at all.
    # Progressive flags are set by _pre_pass; also check actual part lists
    # (tests may construct EquipmentFlags directly with parts).
    has_any_engine = (flags.has_launch_engine or flags.has_vacuum_engine
                      or bool(flags.available_engines))
    has_launch_engine = (flags.has_launch_engine
                         or any(e.atm_isp > 200 for e in flags.available_engines))
    has_any_fuel = (flags.has_lfo_fuel or flags.has_srb_fuel
                    or bool(flags.available_tanks) or bool(flags.available_srbs))

    from .bodies import EdgeType as _ET
    # Dedup propulsion failures by (reason, edge_type) so two ascent
    # edges don't double-report.
    propulsion_seen: set[tuple[BlockingReason, str]] = set()
    propulsion: list[BlockingInfo] = []
    def _add_prop(r: BlockingReason, et_name: str) -> None:
        key = (r, et_name)
        if key in propulsion_seen:
            return
        propulsion_seen.add(key)
        propulsion.append(BlockingInfo(reason=r, edge_type=et_name))
    for edge in profile:
        et = edge.edge_type
        if et == _ET.ATMOSPHERIC_ASCENT or et == _ET.ATMO_LANDING_PROPULSIVE:
            if not has_launch_engine:
                _add_prop(BlockingReason.NO_LAUNCH_ENGINE, et.name)
            if not has_any_fuel:
                _add_prop(BlockingReason.NO_FUEL, et.name)
        elif et in (_ET.VACUUM_ASCENT, _ET.PURE_VACUUM,
                    _ET.PLANET_TRANSFER, _ET.VACUUM_LANDING):
            if not has_any_engine:
                _add_prop(BlockingReason.NO_ENGINE, et.name)
            if not has_any_fuel:
                _add_prop(BlockingReason.NO_FUEL, et.name)
    # Sort for deterministic output matching the legacy `sorted(set)` path.
    propulsion.sort(key=lambda b: str(b))
    blocking.extend(propulsion)

    # ------------------------------------------------------------------
    # Stage grouping
    # ------------------------------------------------------------------
    # Build stage groups by splitting on staging opportunities.
    # Each group is a list of consecutive edges that share a stage.
    # Kerbin ascent is always its own stage.

    groups = _group_edges(profile, flags.staging_tier)

    # Lightest stack decoupler from the player's kit — enables multi-stage
    # atmospheric ascent (F4) when present. K-1 interstages are needed for
    # K-stage; absence forces K=1 (single-stage ascent).
    _stack_decouplers = [d for d in flags.available_decouplers
                         if d.kind == "stack"]
    stack_decoupler_for_ascent = (
        min(_stack_decouplers, key=lambda d: d.mass)
        if _stack_decouplers else None
    )

    # Staging feasibility: if merging couldn't reduce groups to what the
    # player's decouplers allow, the profile is physically impossible.
    max_stages = 1 if flags.staging_tier == 0 else len(groups)
    if len(groups) > max_stages:
        blocking.append(BlockingInfo(
            reason=BlockingReason.STAGING_TIER_INSUFFICIENT,
            stages_needed=len(groups),
            stages_available=flags.staging_tier,
        ))

    # Return all collected pre-check failures before attempting optimization.
    if blocking:
        return ProfileResult(False, blocking=blocking)

    # ------------------------------------------------------------------
    # Backward pass — compute masses from destination back to Kerbin
    # ------------------------------------------------------------------

    # Terminal payload mass
    if is_crewed:
        capsule_mass = flags.lightest_capsule.mass if flags.lightest_capsule else 0.0
        terminal_mass = max(capsule_mass, 0.08)  # min capsule
    else:
        probe_mass = flags.lightest_probe.mass if flags.lightest_probe else 0.0
        terminal_mass = max(probe_mass, 0.04)    # min probe

    # Add equipment mass for the terminal stage
    # (legs, ladder, heat shield on the last edge in the profile)
    terminal_equip = _terminal_equipment_mass(profile, flags, home=home)
    # Contract-required equipment delivered to the destination (drill, ore tank,
    # …). Added to the terminal payload so all stages below carry it.
    extra_payload_mass = sum(p.mass for p in extra_payload_parts)
    payload = terminal_mass + terminal_equip + extra_payload_mass

    # Global attitude strategy. One reaction wheel or RCS bundle covers
    # every stage that flies under it: place it on the *last* (highest
    # flight-order index) stage whose edges require attitude control. That
    # stage's wet mass propagates downward as payload, so earlier stages
    # automatically carry the wheel. Stages *above* that point (e.g. a
    # passive aero-capture reentry) don't pay for it.
    attitude_group_indices = [
        i for i, g in enumerate(groups)
        if any(e.requires_attitude_control for e in g)
    ]
    global_attitude_bundle: Optional[AttitudeBundle] = None
    global_attitude_stage_idx: int = -1   # flight-order index of placement
    global_attitude_force_gimbal: bool = False
    if _PER_STAGE_ATTITUDE_ENABLED and attitude_group_indices:
        if not _terminal_has_built_in_wheels(flags, is_crewed):
            global_attitude_bundle = _attitude_bundle_for_stage(flags, is_crewed)
            if global_attitude_bundle is not None:
                global_attitude_stage_idx = attitude_group_indices[-1]
            else:
                # No wheel/RCS source anywhere — every attitude-requiring stage
                # must pick a gimballed engine/SRB to self-provide control.
                global_attitude_force_gimbal = True

    stage_results_list: list[StageResult] = []
    # Edge-group index for each appended stage (parallel to
    # ``stage_results_list``).  A multi-stage ascent appends K stages for one
    # group, so this is the only reliable stage→group map for the formatter.
    stage_group_list: list[int] = []

    # ``reversed(groups)`` iterates terminal → ascent; track the matching
    # flight-order index so we can hook stage-specific behaviour.
    for rev_idx, group in enumerate(reversed(groups)):
        flight_idx = len(groups) - 1 - rev_idx
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
        # Only propulsive burns force atmospheric ISP and higher TWR floors.
        # Aerocapture (passive drag) and aero landings (parachutes) are not
        # engine-powered, so they must not contaminate the ISP selection for
        # the rest of the group (e.g. vacuum interplanetary burns).
        atmo_types = {
            ET.ATMOSPHERIC_ASCENT,
            ET.ATMO_LANDING_PROPULSIVE,
        }
        in_atmo = any(e.edge_type in atmo_types for e in group)

        # Per-edge TWR → acceleration conversion to handle merged groups that
        # span bodies with very different gravities.  A Gilly VL min_twr=1.2
        # at g=0.049 m/s² must not be evaluated at Kerbin g=9.81 m/s².
        _min_accel = 0.0
        for _e in group:
            if _e.min_twr > 0:
                _g_e = BODY_BY_NAME[_e.body].surface_gravity
                _floor = diff.min_twr_atmo if _e.edge_type in atmo_types else diff.min_twr_vac
                _min_accel = max(_min_accel, max(_e.min_twr, _floor) * _g_e)
        min_twr = _min_accel / body.surface_gravity if body.surface_gravity > 0 else 0.0

        req_throttle = any(e.requires_throttleable for e in group)
        needs_hs = any(e.needs_heat_shield for e in group)
        needs_legs_g = any(e.needs_landing_legs for e in group)

        # Stage equipment.  Landing legs are a fixed, size-independent mass
        # folded into the stage payload.  The heat shield is NOT a fixed mass:
        # the optimizer charges the lightest shield that covers each candidate
        # engine (``heat_shields_arg`` below), so a heavier shield is never
        # forced — keeping the model monotonic in shield count.
        equip_mass = 0.0          # non-shield fixed equipment (landing legs)
        stage_equipment: list[tuple[int, str]] = []
        if needs_legs_g:
            leg_body = BODY_BY_NAME[next(e.body for e in group if e.needs_landing_legs)]
            leg_mass, leg_id = _leg_mass_for_tier(flags, leg_body.landing_leg_tier)
            equip_mass += leg_mass
            if leg_id:
                stage_equipment.append((_LANDING_LEG_COUNT, leg_id))
        # Heat-shield options the optimizer may charge (per-engine lightest
        # covering shield).  Empty when this stage needs no shield.
        heat_shields_arg: tuple[tuple[float, float, str], ...] = ()
        if needs_hs and flags.available_heat_shields:
            heat_shields_arg = tuple(sorted(
                (hs.size_class, hs.mass, hs.name) for hs in flags.available_heat_shields
            ))

        # Heat-shield options the optimizer may charge (per-engine lightest
        # covering shield).  Empty when this stage needs no shield.
        heat_shields_arg: tuple[tuple[float, float, str], ...] = ()
        if needs_hs and flags.available_heat_shields:
            heat_shields_arg = tuple(sorted(
                (hs.size_class, hs.mass, hs.name) for hs in flags.available_heat_shields
            ))

        # Atmospheric-ascent gate: steering a gravity turn in atmosphere
        # requires either a gimballed engine or actuated aero surfaces.
        # Reaction wheels/RCS are not enough.  When no aero surface is
        # available we force the optimizer to pick a gimbal engine.
        # When aero surfaces ARE available, include 4x the lightest
        # surface's mass (min. needed for control on all axes) in the
        # stage payload so the optimizer accounts for it.
        has_atmo_ascent_in_group = any(
            e.edge_type == ET.ATMOSPHERIC_ASCENT for e in group
        )
        needs_gimbal_engine = (
            has_atmo_ascent_in_group and not flags.has_aero_control_surface
        )
        # Landing legs (equip_mass) are charged whenever a stage needs them,
        # independent of the heat shield.  The pre-refactor code routed
        # equip_mass through ``heat_shield_mass`` and zeroed it when the stage
        # needed no shield, silently dropping leg mass on powered (no-shield)
        # vacuum-body landings — an anti-conservative under-charge.
        stage_payload = payload + equip_mass
        if has_atmo_ascent_in_group and flags.lightest_aero_control:
            stage_payload += 4.0 * flags.lightest_aero_control.mass
            stage_equipment.append((4, flags.lightest_aero_control.name))
        # Place the attitude bundle on the highest-flight-index stage that
        # needs attitude. Its wet mass cascades down to earlier stages, so
        # every prior stage carries it for free.
        if (global_attitude_bundle is not None
                and flight_idx == global_attitude_stage_idx):
            stage_payload += global_attitude_bundle.mass
            stage_equipment.extend(global_attitude_bundle.parts)

        # Attitude control. The terminal-stage bundle (if any) is already
        # priced into the payload, so this stage already carries it. If no
        # bundle is available anywhere on the rocket and the group needs
        # attitude control, the stage must self-provide via a gimballed
        # engine/SRB. Atmospheric ascent stages have their own gimbal-or-aero
        # gate above (separate concern from attitude bundling).
        group_needs_attitude = any(e.requires_attitude_control for e in group)
        if (global_attitude_force_gimbal
                and group_needs_attitude
                and not needs_gimbal_engine):
            needs_gimbal_engine = True

        # Parachute consumption check for aero landing edges in this group.
        # Use the landing edge's actual body (not the group's first body) since
        # groups may be merged across bodies.  The heat shield is jettisoned
        # during aero-braking before parachutes deploy, so it is excluded from
        # the chute landing-mass estimate.
        aero_land_edges = [e for e in group if e.edge_type == ET.ATMO_LANDING_AERO]
        if aero_land_edges:
            for _aero_e in aero_land_edges:
                _aero_body = BODY_BY_NAME[_aero_e.body]
                needed = _required_chute_count(payload, _aero_body, flags, diff)
                if needed < 0:
                    # Surface partial-mass-attempt for the bumper scorer.
                    return ProfileResult(False, launch_mass=payload, blocking=[BlockingInfo(
                        reason=BlockingReason.PARACHUTE_TERMINAL_VELOCITY,
                        body=_aero_body.name,
                    )])

        # Aero-landing groups are passive — heat shield + parachutes do all
        # the work.  Skip the engine optimizer entirely.
        if all(e.edge_type == ET.ATMO_LANDING_AERO for e in group):
            # Add chutes to the manifest
            for _aero_e in aero_land_edges:
                _aero_body = BODY_BY_NAME[_aero_e.body]
                chute_count, chute_id = _best_chute_for_body(
                    payload, _aero_body, flags, diff,
                )
                if chute_id and chute_count > 0:
                    stage_equipment.append((chute_count, chute_id))
            # Size the reentry shield to the widest part it protects — the
            # command pod (capsule/probe).  Lightest shield that COVERS the pod
            # diameter; if none is big enough, the largest available.  (Legs are
            # already folded into ``stage_payload`` via ``equip_mass``.)
            passive_shield = None
            if needs_hs and heat_shields_arg:
                _term = _terminal_part(flags, is_crewed)
                _term_dia = _term.size_class if _term else 0.0
                _covering = [hs for hs in heat_shields_arg if hs[0] >= _term_dia]
                passive_shield = (min(_covering, key=lambda x: x[1])
                                  if _covering
                                  else max(heat_shields_arg, key=lambda x: x[0]))
            passive_shield_mass = passive_shield[1] if passive_shield else 0.0
            passive_mass = stage_payload + passive_shield_mass
            stage_results_list.append(StageResult(
                delta_v=0.0,
                twr_at_ignition=0.0,
                twr_at_burnout=0.0,
                engine_is_throttleable=False,
                engine_has_gimbal=False,
                stage_mass_wet=passive_mass,
                stage_mass_dry=passive_mass,
                engine_count=0,
                fill_fraction=0.0,
                engine_name="none",
                tank_manifest=(),
                equipment=stage_equipment,
                heat_shield_name=passive_shield[2] if passive_shield else None,
            ))
            stage_group_list.append(flight_idx)
            payload = passive_mass
            continue

        if flags.staging_tier >= 2 and flags.has_fuel_lines:
            parallel_mode = "asparagus"
        elif flags.staging_tier >= 2:
            parallel_mode = "onion"
        else:
            parallel_mode = "none"

        # Parts the real parallel builder needs: a radial decoupler to shed
        # boosters, and the fuel line for asparagus crossfeed (onion has none).
        _radial_decs = [d for d in flags.available_decouplers if d.kind == "radial"]
        _rdec = min(_radial_decs, key=lambda d: d.mass) if _radial_decs else None
        rdec_mass = _rdec.mass if _rdec else 0.0
        rdec_name = _rdec.name if _rdec else ""
        fl_mass = _FUEL_LINE_MASS if flags.has_fuel_lines else 0.0
        fl_name = _FUEL_LINE_PART if (flags.has_fuel_lines and _FUEL_LINE_PART) else ""

        diagnostic_out: list = []
        stage_kwargs = dict(
            run_parallel=run_parallel,
            radial_decoupler_mass=rdec_mass,
            radial_decoupler_name=rdec_name,
            fuel_line_mass=fl_mass,
            fuel_line_name=fl_name,
            available_engines=eligible_engines,
            available_srbs=flags.available_srbs,
            available_tanks=flags.available_tanks,
            required_dv=req_dv,
            payload_mass=stage_payload,
            gravity=body.surface_gravity,
            min_twr=min_twr,
            requires_throttleable=req_throttle,
            needs_heat_shield=needs_hs,
            max_heat_shield_size=flags.best_heat_shield.size_class if flags.best_heat_shield else None,
            heat_shields=heat_shields_arg,
            in_atmosphere=in_atmo,
            srb_needs_rcs=diff.srb_needs_rcs,
            player_has_rcs=flags.has_rcs,
            tanks_by_fuel_type=flags.tanks_by_fuel_type,
            available_multi_mounts=flags.available_multi_mounts,
            require_gimbal=needs_gimbal_engine,
            diagnostic_out=diagnostic_out,
            body_name=body.name,
            launch_pad_mass_cap=flags.launch_pad_mass_cap,
            atm_scale_height_m=body.atm_scale_height_m,
            atm_top_m=body.safe_altitude_km * 1000.0 if body.has_atmosphere else 0.0,
        )

        # F4 multi-stage ascent: detect any ascent group (atmospheric or
        # vacuum-body). Tsiolkovsky benefit applies to both — vacuum-body
        # ascents like Moho/Mun still benefit from multi-stage even with
        # no atm Isp transition.
        is_ascent_group = any(
            e.edge_type in (ET.ATMOSPHERIC_ASCENT, ET.VACUUM_ASCENT)
            for e in group
        )
        if is_ascent_group:
            ms_diag_out: list = []
            ms_partial_out: list = []
            multistage = find_optimal_multistage_ascent(
                required_dv=req_dv,
                payload_mass=stage_payload,
                gravity=body.surface_gravity,
                in_atmosphere=in_atmo,
                min_twr_liftoff=min_twr,
                available_engines=eligible_engines,
                available_tanks=flags.available_tanks,
                available_srbs=flags.available_srbs,
                tanks_by_fuel_type=flags.tanks_by_fuel_type,
                available_multi_mounts=flags.available_multi_mounts,
                stack_decoupler=stack_decoupler_for_ascent,
                staging_tier=flags.staging_tier,
                needs_heat_shield=needs_hs,
                max_heat_shield_size=flags.best_heat_shield.size_class if flags.best_heat_shield else None,
                heat_shields=heat_shields_arg,
                requires_throttleable=req_throttle,
                require_gimbal=needs_gimbal_engine,
                srb_needs_rcs=diff.srb_needs_rcs,
                player_has_rcs=flags.has_rcs,
                attitude_module_mass=(
                    global_attitude_bundle.mass
                    if (global_attitude_bundle is not None
                        and flight_idx == global_attitude_stage_idx)
                    else 0.0
                ),
                body_name=body.name,
                launch_pad_mass_cap=flags.launch_pad_mass_cap,
                atm_scale_height_m=body.atm_scale_height_m,
                atm_top_m=body.safe_altitude_km * 1000.0 if body.has_atmosphere else 0.0,
                parallel_mode=parallel_mode,
                radial_decoupler_mass=rdec_mass,
                radial_decoupler_name=rdec_name,
                fuel_line_mass=fl_mass,
                fuel_line_name=fl_name,
                run_parallel=run_parallel,
            )
            if multistage is None:
                stage_diag = ms_diag_out[0] if ms_diag_out else None
                # Whole near-miss rocket: stages already built downstream
                # (terminal -> this group) + the partial ascent that got
                # furthest before the binding stage failed.
                partial = list(stage_results_list) + ms_partial_out
                return ProfileResult(False, launch_mass=payload, blocking=[BlockingInfo(
                    reason=BlockingReason.NO_VIABLE_STAGE,
                    body=body.name,
                    dv_needed=req_dv,
                    stage_diag=stage_diag,
                )], partial_stages=partial)
            # Bottom stage carries the group-level equipment (aero surfaces,
            # ladder, etc.) for the multi-stage ascent.
            multistage[0].equipment = stage_equipment + multistage[0].equipment
            # Append top-to-bottom so the outer loop's reverse-chronological
            # ordering produces bottom-first launch-to-orbit after final
            # reversal at the ProfileResult assembly.
            for sr in reversed(multistage):
                stage_results_list.append(sr)
                stage_group_list.append(flight_idx)
            # The bottom stage's wet mass is the launch mass (running total
            # for the outer loop's next-back-up iteration).
            payload = multistage[0].stage_mass_wet
            continue

        result = find_optimal_stage(parallel_mode=parallel_mode, **stage_kwargs)

        if result is None:
            stage_diag = diagnostic_out[0] if diagnostic_out else None
            # Surface the partial-mass-attempt to the bumper.  ``payload``
            # is the running wet mass of every stage already computed
            # (terminal → upstream), so it represents how heavy the
            # launch vehicle would be if this stage could fly.  Lower is
            # closer to feasible; the bumper's scorer uses this to rank
            # infeasible candidates by mass-reduction progress.
            return ProfileResult(False, launch_mass=payload, blocking=[BlockingInfo(
                reason=BlockingReason.NO_VIABLE_STAGE,
                body=body.name,
                dv_needed=req_dv,
                stage_diag=stage_diag,
            )], partial_stages=list(stage_results_list))

        # Chutes for aero-landing edges in mixed groups
        if aero_land_edges:
            for _aero_e in aero_land_edges:
                _aero_body = BODY_BY_NAME[_aero_e.body]
                chute_count, chute_id = _best_chute_for_body(
                    payload, _aero_body, flags, diff,
                )
                if chute_id and chute_count > 0:
                    stage_equipment.append((chute_count, chute_id))

        # Ladder
        if any(e.needs_ladder for e in group) and flags.lightest_ladder:
            stage_equipment.append((1, flags.lightest_ladder.name))

        # Prepend group equipment; KEEP what the optimizer already attached
        # (a parallel build's radial decouplers + fuel lines), else they're
        # lost from both the displayed build and the kit/gating.
        result.equipment = stage_equipment + result.equipment
        stage_results_list.append(result)
        stage_group_list.append(flight_idx)
        # The stage's wet mass becomes the payload for the next stage back
        payload = result.stage_mass_wet

    # Add decouplers to non-terminal stages (not in mass budget, just for build guide)
    num_stages = len(stage_results_list)
    if num_stages > 1:
        stack_decs = [d for d in flags.available_decouplers if d.kind == "stack"]
        if stack_decs:
            best_dec = min(stack_decs, key=lambda d: d.mass)
            for sr in stage_results_list[:-1]:
                sr.equipment.append((1, best_dec.name))

    # Build terminal parts list (command module + support equipment)
    terminal_parts: list[tuple[int, str]] = []
    if is_crewed and flags.lightest_capsule:
        terminal_parts.append((1, flags.lightest_capsule.name))
    elif not is_crewed and flags.lightest_probe:
        terminal_parts.append((1, flags.lightest_probe.name))
    support_mass, support_parts = _support_equipment_mass(flags, profile, home=home)
    terminal_parts.extend(support_parts)
    # Contract equipment is part of the delivered terminal payload — list it on
    # the manifest so /explain shows the real parts whose mass was charged.
    terminal_parts.extend((1, p.name) for p in extra_payload_parts)

    if payload > flags.launch_pad_mass_cap:
        return ProfileResult(
            feasible=False,
            launch_mass=payload,
            blocking=[BlockingInfo(
                reason=BlockingReason.LAUNCH_MASS_EXCEEDED,
                mass_actual=payload,
                mass_cap=flags.launch_pad_mass_cap,
            )],
        )
    reversed_stages = list(reversed(stage_results_list))
    return ProfileResult(
        feasible=True,
        launch_mass=payload,
        stage_results=reversed_stages,
        edge_groups=groups,
        stage_group_indices=list(reversed(stage_group_list)),
        terminal_parts=terminal_parts,
        # kit_used is NOT built here — it's expensive and only two call
        # sites consume it.  They call ``build_kit_for_result`` explicitly.
    )


def _group_edges(profile: list[MissionEdge], staging_tier: int) -> list[list[MissionEdge]]:
    """
    Group consecutive edges into stages based on staging_tier.

    staging_tier=0: everything is one stage (no decouplers).
    staging_tier=1-2: limited staging (staging_tier + 1 stages). Excess
      natural groups are merged (smallest-dv pairs first).

    Natural stage boundaries:
      - After Kerbin ascent (always own stage if staging_tier >= 1)
      - Before and after landing (always own stage if staging_tier >= 1)
      - Before any vacuum-to-atmo transition

    When forced to merge (tier 0), constraint-aware merging skips incompatible
    pairs (e.g. atmospheric + vacuum, TWR-constrained + unconstrained) to avoid
    creating physically impossible combined stages.
    """
    # Mark preferred split indices (after edge i, before edge i+1)
    from .bodies import EdgeType as ET
    splits: list[int] = []

    for i in range(len(profile) - 1):
        cur = profile[i]
        nxt = profile[i + 1]

        # Always split after any atmospheric or vacuum ascent.
        # Surface ascents (AT/VA) are always their own stage — they have
        # different ISP, TWR requirements, and gravity from the burns that
        # follow once the vehicle is in orbit.
        if cur.edge_type in (ET.ATMOSPHERIC_ASCENT, ET.VACUUM_ASCENT):
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

    # Tier 0 = no decouplers = single stage.
    # Tier 1+ = any decoupler allows all natural stage groups (stack
    # decouplers are cheap and stackable in KSP — no practical limit).
    if staging_tier == 0:
        max_stages = 1
    else:
        max_stages = len(groups)

    # Constraint-aware merging: skip incompatible pairs when forced to merge.
    atmo_types = {ET.ATMOSPHERIC_ASCENT, ET.ATMO_LANDING_PROPULSIVE}

    def _can_merge(a: list[MissionEdge], b: list[MissionEdge]) -> bool:
        """Groups are compatible for merging if they share physics regime."""
        a_atmo = any(e.edge_type in atmo_types for e in a)
        b_atmo = any(e.edge_type in atmo_types for e in b)
        if a_atmo != b_atmo:
            return False
        a_twr = any(e.min_twr > 0 for e in a)
        b_twr = any(e.min_twr > 0 for e in b)
        if a_twr != b_twr:
            return False
        return True

    while len(groups) > max_stages:
        best_pair = -1
        best_dv = float("inf")
        for i in range(len(groups) - 1):
            # At tier 0 (no decouplers), skip incompatible merges to avoid
            # creating physically impossible single stages. At tier 1-2,
            # merge unconditionally (decouplers allow staging; the optimizer
            # handles mixed groups by picking the most constrained engine).
            if staging_tier == 0 and not _can_merge(groups[i], groups[i + 1]):
                continue
            combined = sum(e.base_dv for e in groups[i]) + sum(e.base_dv for e in groups[i + 1])
            if combined < best_dv:
                best_dv = combined
                best_pair = i
        if best_pair < 0:
            break  # no compatible merge possible
        merged = groups[best_pair] + groups[best_pair + 1]
        groups = groups[:best_pair] + [merged] + groups[best_pair + 2:]

    return groups


def _support_equipment_mass(
    flags: EquipmentFlags, profile: list[MissionEdge],
    home: BodyName,
) -> tuple[float, list[tuple[int, str]]]:
    """
    Return (mass, parts) for required support equipment (antenna, power)
    based on the most demanding body in the mission profile.
    Each part entry is (count, part_id).
    """
    mass = 0.0
    parts: list[tuple[int, str]] = []

    # Find the most demanding relay tier and power requirement across all edges
    max_relay = 0
    power_req = "none"
    needs_retractable = False
    for edge in profile:
        body = BODY_BY_NAME.get(edge.body)
        if body is None:
            continue
        if edge.relay_tier > max_relay:
            max_relay = edge.relay_tier
        # Power: rtg > solar_marginal > solar > none
        prio = {"none": 0, "solar": 1, "solar_marginal": 2, "rtg": 3}
        if prio.get(body.power_requirement, 0) > prio.get(power_req, 0):
            power_req = body.power_requirement
        # If any edge involves aerobraking, we need retractable solar
        if edge.needs_heat_shield:
            needs_retractable = True

    # Relay: find lightest antenna meeting the required tier
    if max_relay > 0:
        best_relay: Optional[MiscEquipment] = None
        for tier in range(max_relay, 4):
            candidate = flags.lightest_relay.get(tier)
            if candidate and (best_relay is None or candidate.mass < best_relay.mass):
                best_relay = candidate
        if best_relay:
            mass += best_relay.mass
            parts.append((1, best_relay.name))

    # Power: find lightest power source meeting requirement
    if power_req == "rtg":
        if flags.lightest_rtg:
            mass += flags.lightest_rtg.mass
            parts.append((1, flags.lightest_rtg.name))
    elif power_req in ("solar", "solar_marginal"):
        if needs_retractable:
            if flags.lightest_solar_retractable:
                mass += flags.lightest_solar_retractable.mass
                parts.append((1, flags.lightest_solar_retractable.name))
            elif flags.lightest_rtg:
                mass += flags.lightest_rtg.mass
                parts.append((1, flags.lightest_rtg.name))
        else:
            if flags.lightest_solar:
                mass += flags.lightest_solar.mass
                parts.append((1, flags.lightest_solar.name))

    return mass, parts


def _terminal_equipment_mass(profile: list[MissionEdge],
                              flags: EquipmentFlags,
                              home: BodyName) -> float:
    """
    Equipment mass carried all the way to the terminal destination.

    Includes landing legs, ladder, and support equipment (antenna, power).
    Legs are only included if the LAST edge in the profile needs landing legs.
    On return missions the last edge is Kerbin reentry (no legs); the legs
    used at intermediate bodies are left on the surface, not returned.
    """
    mass = 0.0
    if profile:
        last_edge = profile[-1]
        if last_edge.needs_landing_legs:
            body = BODY_BY_NAME.get(last_edge.body)
            if body:
                leg_mass, _ = _leg_mass_for_tier(flags, body.landing_leg_tier)
                mass += leg_mass
    # Ladder — only if last edge needs one (same logic: left at surface otherwise)
    if profile and profile[-1].needs_ladder and flags.lightest_ladder:
        mass += flags.lightest_ladder.mass
    # Support equipment (antenna + power source)
    support_mass, _ = _support_equipment_mass(flags, profile, home=home)
    mass += support_mass
    return mass


_LANDING_LEG_COUNT = 4  # conservative: 4 legs per landing


def _leg_mass_for_tier(
    flags: EquipmentFlags, required_tier: int,
) -> tuple[float, str]:
    """Return (total_mass, part_id) of the lightest available legs meeting the tier."""
    for leg in sorted(flags.available_landing_legs, key=lambda l: l.mass):
        if leg.tier >= required_tier:
            return leg.mass * _LANDING_LEG_COUNT, leg.name
    # Fall back: if we have any legs at all, use them
    if flags.available_landing_legs:
        best = min(flags.available_landing_legs, key=lambda l: l.mass)
        return best.mass * _LANDING_LEG_COUNT, best.name
    return 0.0, ""


def _required_chute_count(
    landing_mass: float,
    body: Body,
    flags: EquipmentFlags,
    diff: DifficultyProfile,
) -> int:
    """Return the number of parachutes (of ``flags.best_chute``) needed
    to achieve terminal velocity ≤ ``_MAX_SAFE_LANDING_SPEED``.

    ``flags.best_chute`` is precomputed in ``_pre_pass`` to be the
    asymptote-best non-drogue chute (lowest mass-per-drag-area).  Two
    chutes with identical drag but different masses
    (``parachuteLarge``=0.3t vs ``parachuteRadial``=0.1t, both 500 m²)
    give very different terminal-velocity asymptotes: the heavier
    chute carries too much of its own weight and stalls above 6 m/s
    on Duna no matter how many you stack, so the lighter one is the
    only chute that can ever beat the safety threshold there.

    Workaround for the structural limitation that aero landings are
    modelled as parachute-only OR engine-only — the real fix is a
    mixed-strategy landing edge.  See
    ``bugs/084-no-mixed-parachute-and-engine-landing.md``.

    Uses pessimistic mass estimate (landing_mass + all chutes) to
    avoid under-counting (golden rule).  Returns ``-1`` if no count
    of the chosen chute beats the threshold.
    """
    if not body.has_atmosphere or body.atm_density_kg_m3 <= 0:
        return 0  # vacuum body — no chutes needed

    # Pessimistic upper bound: add mass of all available chutes
    if not flags.available_parachutes:
        return -1

    chute = flags.best_chute
    if chute is None:
        return -1
    # The AP item represents the parachute *type* being unlocked, not a single
    # physical part.  Once unlocked, the player can attach as many as needed.
    # Use a generous per-mission budget (50) so the physics check can succeed.
    max_chutes = flags.parachute_count * 50

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


def _best_chute_for_body(
    landing_mass: float, body: Body, flags: EquipmentFlags,
    diff: DifficultyProfile,
) -> tuple[int, str]:
    """Return (count, part_id) for the chute used in aero landing, or (0, "")."""
    chute = flags.best_chute
    if chute is None:
        return 0, ""
    count = _required_chute_count(landing_mass, body, flags, diff)
    return (max(1, count), chute.name) if count != 0 else (0, "")


# ---------------------------------------------------------------------------
# Step 3: Per-body assessment
# ---------------------------------------------------------------------------

def _assess_bodies(
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_builder: MissionBuilder,
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
        results[body.name] = _assess_one_body(body, flags, diff, results, mission_builder)

    return results


class _LazyBodyProfiles:
    """Lazy ``body_name -> BodyAccessProfile`` mapping.

    Assesses a body only on first access and memoizes the result.
    Computing a moon first ensures its parent is assessed (the parent-
    gating check in ``_assess_one_body`` reads ``computed[parent]``).

    Why lazy: AP's fill sweep recomputes capability on every item add
    (~225 distinct capability states per SSR seed), and each full
    ``_assess_bodies`` evaluated ALL 17 bodies.  But a location's access
    rule only queries ``.bodies[its_own_body]`` — most states touch a
    handful of bodies, not all 17.  Deferring per-body assessment turns
    225×17 body evals into 225×(few).  Iteration / ``values`` / ``items``
    still materialize everything (used by display + multi-body goal
    rules like flag_every_body), so those paths are unchanged.
    """
    __slots__ = ("_flags", "_diff", "_mb", "_cache")

    def __init__(self, flags: EquipmentFlags, diff: DifficultyProfile,
                 mission_builder: MissionBuilder):
        self._flags = flags
        self._diff = diff
        self._mb = mission_builder
        self._cache: dict[str, BodyAccessProfile] = {}

    def _get(self, body_name: str) -> BodyAccessProfile:
        prof = self._cache.get(body_name)
        if prof is not None:
            return prof
        body = BODY_BY_NAME.get(body_name)
        if body is None:
            raise KeyError(body_name)
        # Parent-gating: ensure the parent is assessed first so
        # ``_assess_one_body``'s ``computed[parent]`` lookup succeeds.
        if body.parent is not None and body.parent not in self._cache:
            self._get(body.parent)
        prof = _assess_one_body(body, self._flags, self._diff,
                                self._cache, self._mb)
        self._cache[body_name] = prof
        return prof

    def __getitem__(self, body_name: str) -> BodyAccessProfile:
        return self._get(body_name)

    def get(self, body_name: str, default=None):
        try:
            return self._get(body_name)
        except KeyError:
            return default

    def __contains__(self, body_name: str) -> bool:
        return body_name in BODY_BY_NAME

    def _materialize(self) -> dict[str, BodyAccessProfile]:
        for b in ALL_BODIES:
            self._get(b.name)
        return self._cache

    def __iter__(self):
        return iter(self._materialize())

    def keys(self):
        return self._materialize().keys()

    def values(self):
        return self._materialize().values()

    def items(self):
        return self._materialize().items()


def _assess_one_body(
    body: Body,
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    computed: dict[str, BodyAccessProfile],
    mission_builder: MissionBuilder,
) -> BodyAccessProfile:
    prof = BodyAccessProfile()
    home_system = home_system_bodies(mission_builder.home)

    # --- Parent gating ---
    # Moons whose parent is in the home system are trivially reachable
    # (the parent is the home, or shares the home's local neighbourhood).
    # Moons whose parent is interplanetary need the parent orbitally
    # reachable before the moon can be considered.
    if body.parent is not None and body.parent not in home_system:
        parent_prof = computed[body.parent]
        if not parent_prof.access[EventName.ORBIT]:
                prof.set_blocking(BlockingInfo(
                    reason=BlockingReason.PARENT_BODY_UNREACHABLE,
                    body=body.name,
                    parent_body=body.parent,
                ))
                return prof

    # --- Evaluate the events that locations.py exposes for this body ---
    # Iterating get_body_events instead of ALL_EVENTS means "events that don't
    # exist for this body" naturally stay at their default False access value.
    # No special-case branches per body needed.
    body_events = set(get_body_events(body))
    for event in ALL_EVENTS:
        if event.name not in body_events:
            continue
        if event.crewed is True and not flags.has_capsule:
            prof.access[event.name] = False
            continue

        profiles = mission_builder.profiles_for(body.name, event.mission_type)

        # Empty profile = always achievable (e.g. Kerbin launchpad EVA).
        if not profiles:
            prof.access[event.name] = True
            continue

        # High-gravity sample return requires ladder for EVA re-boarding
        if event.mission_type == MissionType.SAMPLE_RETURN and body.eva_jetpack_twr < _MIN_EVA_JETPACK_TWR:
            profiles = _inject_ladder(profiles)

        ok, sub_blocking = _try_profiles_reason(
            profiles, flags, diff, event.mission_type,
            crewed=event.crewed, home=mission_builder.home,
        )
        prof.access[event.name] = ok
        if not ok and not prof.blocking:
            prof.set_blocking(BlockingInfo(
                reason=BlockingReason.EVENT_COMPOUND,
                body=body.name,
                mission_type=str(event.mission_type),
                detail="; ".join(str(b) for b in sub_blocking),
            ))

    if not prof.access[EventName.ORBIT] and not prof.blocking:
        prof.set_blocking(BlockingInfo(
            reason=BlockingReason.ORBIT_NOT_ACHIEVABLE,
            body=body.name,
        ))

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


def _crewed_options(crewed: bool | None, flags: EquipmentFlags) -> list[bool]:
    """Return the list of is_crewed values to attempt for a mission.

    crewed=True/False → single attempt.
    crewed=None → try unmanned first, then crewed if the player has a capsule.
    """
    if crewed is not None:
        return [crewed]
    opts = [False]
    if flags.has_capsule:
        opts.append(True)
    return opts


def _try_profiles(
    profiles: list[list[MissionEdge]],
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_type: MissionType,
    crewed: bool | None,
    home: BodyName,
    extra_payload_parts: tuple[MiscEquipment, ...] = (),
) -> bool:
    """Return True if any profile alternative is feasible.

    ``home`` is needed by the relay-tier gate (heliocentric-distance
    based); test callers can rely on the Kerbin default.
    """
    # Serial-first: gating only needs feasibility, and the exact asparagus
    # build is ~2x the serial cost.  Asparagus only makes a build LIGHTER, so
    # serial-feasible ⟹ parallel-feasible — trying serial first and only
    # falling back to the parallel build when NO profile closes serially is
    # feasibility-identical to always-parallel, just far cheaper in the common
    # (serial-feasible) case.
    for run_par in (False, True):
        for is_crewed in _crewed_options(crewed, flags):
            for profile in profiles:
                result = _evaluate_profile(profile, flags, diff, mission_type,
                                           is_crewed=is_crewed, home=home,
                                           extra_payload_parts=extra_payload_parts,
                                           run_parallel=run_par)
                if result.feasible:
                    return True
    return False


def _try_profiles_reason(
    profiles: list[list[MissionEdge]],
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    mission_type: MissionType,
    crewed: bool | None,
    home: BodyName,
    extra_payload_parts: tuple[MiscEquipment, ...] = (),
) -> tuple[bool, list[BlockingInfo]]:
    """
    Like _try_profiles but also returns deduplicated blocking entries
    collected across all profile attempts on failure.
    """
    all_blocking: list[BlockingInfo] = []
    seen: set[str] = set()
    if not profiles:
        return False, [BlockingInfo(reason=BlockingReason.NO_PROFILES,
                                     mission_type=str(mission_type))]
    # Serial-first (see _try_profiles): asparagus only makes builds lighter, so
    # serial-feasible ⟹ parallel-feasible.  Try the cheap serial mass first;
    # only if no profile closes serially do we pay for the exact parallel build.
    # Blocking reasons come from the parallel pass (the real, lightest-build
    # failure).
    for run_par in (False, True):
        all_blocking = []
        seen = set()
        for is_crewed in _crewed_options(crewed, flags):
            for profile in profiles:
                result = _evaluate_profile(profile, flags, diff, mission_type,
                                           is_crewed=is_crewed, home=home,
                                           extra_payload_parts=extra_payload_parts,
                                           run_parallel=run_par)
                if result.feasible:
                    return True, []
                for b in result.blocking:
                    key = str(b)
                    if key not in seen:
                        seen.add(key)
                        all_blocking.append(b)
    return False, all_blocking


def evaluate_mission_detailed(
    flags: EquipmentFlags,
    diff: DifficultyProfile,
    body_name: str,
    mission_type: MissionType,
    crewed: bool | None,
    mission_builder: MissionBuilder,
    threshold_km: float | None = None,
    extra_payload_parts: tuple[MiscEquipment, ...] = (),
    mission_transform: Optional[Callable[[list], list]] = None,
) -> ProfileResult:
    """
    Evaluate a specific mission and return the winning ProfileResult
    with full stage details + edge groups. Returns a non-feasible
    ProfileResult if no profile alternative succeeds.

    crewed: True = crewed only, False = unmanned only, None = try both.

    For ``mission_type="sounding"``, evaluates sounding rocket altitude
    against ``threshold_km``.  Other Kerbin-specific types (first_launch,
    first_landing, first_staging, splashdown) evaluate the relevant
    capability flags.
    """
    home = mission_builder.home_body

    # --- Sounding rocket (altitude milestones, first crash) ---
    if mission_type == MissionType.SOUNDING:
        return _evaluate_sounding(flags, threshold_km or 0.0, home)

    # --- Other home-body-specific mission types ---
    if mission_type == MissionType.FIRST_LAUNCH:
        # KSP fires the FirstLaunch event for any kerbal EVA off the pad as
        # well as a real rocket launch. We model only the rocket path here
        # as a conservative subset — false negatives are acceptable, and
        # accepting "kerbal walks off pad" caused minimal_rocket_for to
        # always pick Progressive Capsule for sphere 0 (Capsule alone makes
        # the location feasible), which forced every starting inventory to
        # contain a capsule and suppressed probe-only starts.
        sounding = _compute_sounding_altitude(flags, home)
        if sounding > 0:
            return ProfileResult(True)
        # Delegate to the sounding-rocket evaluator (with threshold=0.1 to
        # force a "needs altitude" failure) so the structured reasons name
        # the specific missing parts (payload, propulsion).
        sub = _evaluate_sounding(flags, 0.1, home)
        return ProfileResult(False, blocking=list(sub.blocking))

    if mission_type == MissionType.FIRST_LANDING:
        sounding = _compute_sounding_altitude(flags, home)
        # Capsule-only path (kerbal EVA)
        if flags.has_capsule:
            return ProfileResult(True)
        # Engine path: sounding + safe descent
        if sounding > 0 and (flags.has_parachutes or flags.has_throttleable_engine):
            return ProfileResult(True)
        blocking_list = []
        if not flags.has_capsule:
            blocking_list.append(BlockingInfo(
                reason=BlockingReason.NO_CAPSULE, detail="EVA path"))
        if sounding <= 0:
            blocking_list.append(BlockingInfo(
                reason=BlockingReason.NO_SOUNDING_ALTITUDE))
        elif not flags.has_parachutes and not flags.has_throttleable_engine:
            blocking_list.append(BlockingInfo(
                reason=BlockingReason.NO_SAFE_DESCENT))
        return ProfileResult(False, blocking=blocking_list)

    if mission_type == MissionType.FIRST_STAGING:
        if flags.staging_tier >= 1:
            return ProfileResult(True)
        return ProfileResult(False, blocking=[BlockingInfo(
            reason=BlockingReason.STAGING_TIER_INSUFFICIENT,
            stages_needed=0,
            stages_available=0,
        )])

    if mission_type == MissionType.SPLASHDOWN:
        threshold = threshold_km or 1.0
        home_body_obj = BODY_BY_NAME[home]
        blocking_list: list[BlockingInfo] = []

        # Path 1: home has an ocean → sounding rocket + safe descent.
        if home_body_obj.has_ocean:
            sounding = _compute_sounding_altitude(flags, home)
            descent_ok = flags.has_parachutes or flags.has_throttleable_engine
            if sounding >= threshold and descent_ok:
                return ProfileResult(True)
            if sounding < threshold:
                blocking_list.append(BlockingInfo(
                    reason=BlockingReason.SOUNDING_ALTITUDE_TOO_LOW,
                    altitude_km=sounding,
                    threshold_km=threshold,
                ))
            if not descent_ok:
                blocking_list.append(BlockingInfo(
                    reason=BlockingReason.NO_SAFE_DESCENT))

        # Path 2: any other ocean body's LAND profile succeeds.
        for ocean in ALL_BODIES:
            if not ocean.has_ocean or ocean.name == home:
                continue
            profiles = mission_builder.profiles_for(ocean.name, MissionType.LAND)
            if not profiles:
                continue
            ok, sub_blocking = _try_profiles_reason(
                profiles, flags, diff, MissionType.LAND,
                crewed=None, home=home,
            )
            if ok:
                return ProfileResult(True)
            for b in sub_blocking:
                blocking_list.append(b)

        return ProfileResult(False, blocking=blocking_list)

    # --- Standard body mission profiles ---
    profiles = mission_builder.profiles_for(body_name, mission_type)
    if not profiles:
        return ProfileResult(False, blocking=[BlockingInfo(
            reason=BlockingReason.NO_PROFILES,
            body=body_name,
            mission_type=str(mission_type),
        )])

    if mission_type == MissionType.SAMPLE_RETURN:
        body = BODY_BY_NAME[body_name]
        if body.eva_jetpack_twr < _MIN_EVA_JETPACK_TWR:
            profiles = _inject_ladder(profiles)

    # Contract-supplied mission modifier: rewrite each profile's edge list
    # (insert/append/modify maneuvers) before sizing. Used by orbit-variant
    # contracts (polar ascent penalty, stationary raise edge). Identity for
    # ordinary missions. See ContractTypeDef.transform_mission.
    if mission_transform is not None:
        profiles = [mission_transform(p) for p in profiles]

    all_blocking: list[BlockingInfo] = []
    seen: set[str] = set()
    for is_crewed in _crewed_options(crewed, flags):
        for profile in profiles:
            result = _evaluate_profile(profile, flags, diff, mission_type,
                                       is_crewed=is_crewed,
                                       home=mission_builder.home,
                                       extra_payload_parts=extra_payload_parts)
            if result.feasible:
                return result
            for b in result.blocking:
                key = str(b)
                if key not in seen:
                    seen.add(key)
                    all_blocking.append(b)

    return ProfileResult(False, blocking=all_blocking)


def _evaluate_sounding(flags: EquipmentFlags, threshold_km: float, home: Body) -> ProfileResult:
    """Evaluate sounding rocket capability against a target altitude."""
    sounding_km = _compute_sounding_altitude(flags, home)
    if sounding_km >= threshold_km:
        return ProfileResult(True, launch_mass=0.0)

    blocking_list: list[BlockingInfo] = []
    if sounding_km > 0:
        blocking_list.append(BlockingInfo(
            reason=BlockingReason.SOUNDING_ALTITUDE_TOO_LOW,
            altitude_km=sounding_km,
            threshold_km=threshold_km,
            detail="need bigger SRB/engine",
        ))
    else:
        if not flags.has_probe_core and not flags.has_capsule:
            blocking_list.append(BlockingInfo(
                reason=BlockingReason.NO_COMMAND_MODULE))
        elif not flags.has_probe_core and flags.has_capsule:
            missing = []
            if not flags.has_parachutes:
                missing.append("parachute")
            if flags.staging_tier < 1:
                missing.append("decoupler")
            if missing:
                blocking_list.append(BlockingInfo(
                    reason=BlockingReason.CAPSULE_SOUNDING_INCOMPLETE,
                    detail=", ".join(missing),
                ))
        if not flags.available_srbs and not flags.available_engines:
            blocking_list.append(BlockingInfo(
                reason=BlockingReason.NO_PROPULSION))
        elif flags.available_engines and not flags.available_tanks:
            blocking_list.append(BlockingInfo(reason=BlockingReason.NO_FUEL))
    return ProfileResult(False, blocking=blocking_list)


# ---------------------------------------------------------------------------
# Step 4: Assemble RocketCapability
# ---------------------------------------------------------------------------

def _compute_sounding_altitude(flags: EquipmentFlags, home: Body) -> float:
    """
    Estimate the maximum altitude (km) achievable with a single-stage sounding
    rocket built from the player's current parts, launched from ``home``.

    Back-computation goes through ``home.max_suborbital_altitude_km(dv, twr)``
    (no atm drag; simple gravity-drag).  ``home.surface_gravity`` is used for
    TWR; ``9.81`` (the Isp reference constant ``g0``) is used for the
    rocket equation.  Engine thrust uses sea-level ``atm_thrust`` on atmo
    bodies and ``vac_thrust`` on vacuum bodies — the engine actually
    performs at vac in vacuum, so atm_thrust would understate it.

    Vacuum Isp is used throughout: drag is ignored, so the dv that does
    real work is the high-altitude regime where atm Isp loss is small.

    Payload options:
      • Probe core (unmanned) — no survival constraint.
      • Capsule (crewed) — requires decoupler + parachute so the pod can
        separate from the rocket body and land safely.
    """
    G0 = 9.81  # standard Isp reference (physical constant, not body-dependent)
    g = home.surface_gravity
    use_atm_thrust = home.has_atmosphere
    best_km = 0.0

    payloads: list[float] = []
    if flags.lightest_probe:
        payloads.append(flags.lightest_probe.mass)
    if flags.lightest_capsule and flags.lightest_capsule.mass > 0:
        # Crewed: survivable iff (decoupler + at least one parachute)
        if flags.has_parachutes and flags.staging_tier >= 1:
            payloads.append(flags.lightest_capsule.mass)

    if not payloads:
        return 0.0

    for payload_mass in payloads:
        # Liquid / LF engines
        for engine in flags.available_engines:
            if engine.fuel_type not in ("lfo", "lf"):
                continue  # ion/xenon have negligible atm thrust
            thrust = engine.atm_thrust if use_atm_thrust else engine.vac_thrust
            m_base = engine.mass + payload_mass
            max_total = thrust / (_SOUNDING_MIN_TWR * g)
            if m_base >= max_total:
                continue
            max_prop = max_total - m_base
            for tank in flags.available_tanks:
                if tank.fuel_type != engine.fuel_type:
                    continue
                tank_full = tank.dry_mass + tank.fuel_mass
                n = int(max_prop / tank_full)
                if n < 1:
                    continue
                m0 = m_base + n * tank_full
                m_dry = m_base + n * tank.dry_mass
                if m_dry <= 0 or m0 <= m_dry:
                    continue
                dv = engine.vac_isp * G0 * math.log(m0 / m_dry)
                twr = thrust / (g * m0)
                if twr < _SOUNDING_MIN_TWR:
                    continue
                h_km = home.max_suborbital_altitude_km(dv, twr)
                best_km = max(best_km, h_km)

        # Solid rocket boosters
        for srb in flags.available_srbs:
            m0 = srb.dry_mass + srb.fuel_mass + payload_mass
            m_dry = srb.dry_mass + payload_mass
            if m_dry <= 0 or m0 <= m_dry:
                continue
            thrust = srb.atm_thrust if use_atm_thrust else srb.vac_thrust
            twr = thrust / (g * m0)
            if twr < _SOUNDING_MIN_TWR:
                continue
            dv = srb.vac_isp * G0 * math.log(m0 / m_dry)
            h_km = home.max_suborbital_altitude_km(dv, twr)
            best_km = max(best_km, h_km)

    return best_km


def compute_capability_from_items(
    item_count_fn: Callable[[str], int],
    difficulty_name: str,
    start_with_clamps: bool,
    mission_builder: MissionBuilder,
    progressive_launch_pad: bool = False,
    contract_specs: tuple = (),
) -> tuple[RocketCapability, EquipmentFlags]:
    """Compute capability without a CollectionState. For CLI/external tools.

    ``contract_specs`` (a tuple of ContractSpec) makes this also compute
    per-contract feasibility into ``cap.contract_access``. Empty = no contracts.
    """
    diff = DIFFICULTY_PROFILES[difficulty_name]
    flags = _pre_pass(item_count_fn, start_with_clamps,
                      progressive_launch_pad,
                      launch_pad_caps=mission_builder.launch_pad_caps)
    # Lazy: bodies are assessed on first query (AP fill rules touch only
    # a few bodies per state; eager _assess_bodies evaluated all 17).
    body_profiles = _LazyBodyProfiles(flags, diff, mission_builder)
    sounding_km = _compute_sounding_altitude(flags, mission_builder.home_body)

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
        sounding_altitude_km=sounding_km,
        relay_tier=flags.relay_tier,
        power_profile=power_str,
        staging_tier=flags.staging_tier,
        has_launch_clamp=flags.has_launch_clamp,
        has_thermometer=flags.has_thermometer,
        has_barometer=flags.has_barometer,
        has_wheel=flags.has_wheel,
        has_throttleable_engine=flags.has_throttleable_engine,
        has_aero_control_surface=flags.has_aero_control_surface,
        bodies=body_profiles,
    )

    if contract_specs:
        # Lazy import breaks the capability <-> contracts cycle (contracts.py
        # imports this module for the physics primitive).
        from .contracts import compute_contract_access
        cap.contract_access = compute_contract_access(
            contract_specs, flags, diff, mission_builder)

    return cap, flags


def _compute_capability(state: CollectionState, player: int) -> RocketCapability:
    """Full capability computation from the current collection state."""
    world = state.multiworld.worlds[player]
    options = world.options
    difficulty_name = ["casual", "normal", "expert", "insane"][options.difficulty.value]
    start_with_clamps = bool(options.start_with_launch_clamps.value)
    cap, _ = compute_capability_from_items(
        lambda name: state.count(name, player),
        difficulty_name, start_with_clamps, world.mission_builder,
        progressive_launch_pad=bool(options.progressive_launch_pad.value),
        contract_specs=(*getattr(world, "contract_specs", ()),
                        *getattr(world, "goal_contract_specs", ())),
    )
    return cap


# ---------------------------------------------------------------------------
# Import-time assertions — pytest catches violations automatically
# ---------------------------------------------------------------------------

# (a) Every MiscEquipment with multi_mount flag must be in MULTI_MOUNT_TABLE
for _item_name, _parts in PART_DB.items():
    for _part in _parts:
        if isinstance(_part, MiscEquipment) and CapabilityFlag.MULTI_MOUNT in _part.provides:
            assert _part.name in MULTI_MOUNT_TABLE, (
                f"{_part.name} has multi_mount flag but not in MULTI_MOUNT_TABLE"
            )

# (b) Every MiscEquipment flag in parts.py must be a recognized CapabilityFlag.
_KNOWN_FLAGS: frozenset[str] = frozenset(CapabilityFlag)

for _item_name, _parts in PART_DB.items():
    for _part in _parts:
        if isinstance(_part, MiscEquipment):
            _unknown = _part.provides - _KNOWN_FLAGS
            assert not _unknown, (
                f"Part {_part.name} has unrecognized flags: {_unknown}"
            )

del _item_name, _parts, _part, _unknown, _KNOWN_FLAGS

# Mission-profile coverage assertion now lives in MissionBuilder._validate
# (bodies.py); it runs at builder construction.
