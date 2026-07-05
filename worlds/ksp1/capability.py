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
    GameplayDifficulty, CONSERVATIVE_GAMEPLAY,
    Body, MissionEdge, MissionBuilder, EdgeType,
    effective_dv, effective_physics_profile_name, home_system_bodies, parent_chain,
)
from .comms import DSN_POWER_MAX, dsn_required_relay_table
from .parts import (
    DEFAULT_PART_MANAGER, CapabilityFlag, Engine, FuelTank, SolidBooster,
    HeatShield, Parachute, LandingLeg, Decoupler, MiscEquipment,
    MultiMount, MULTI_MOUNT_TABLE,
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
    ESCALATED_MAX_ASCENT_STAGES, ESCALATED_BOOSTER_COUNTS,
    ESCALATED_MAX_ENG_PER_COL,
)
from .rocket_math import aero
from .data.feasibility import (
    ESCALATED_ASCENT_EDGES, ESCALATED_HOME_ASCENT_EDGES,
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
    name for name, parts in DEFAULT_PART_MANAGER.parts.items()
    if any(
        isinstance(p, _CAPABILITY_PART_TYPES) and (
            not isinstance(p, MiscEquipment) or p.provides
        )
        for p in parts
    )
) | {
    # Contract required-category parts (drill / ore_tank / science_lab /
    # eva_jetpack, ...).  ``contract_access`` depends on their presence via
    # ``required_part_manifest`` (category_lightest), but several are payload
    # parts with no ``provides``, so the part-type filter above misses them.
    # They MUST be fingerprinted or a state without the drill and a state with
    # it share a fingerprint and the cached (drill-less → mine_ore infeasible)
    # contract_access is reused after the drill is collected -- the same
    # collision the Mission Control note below describes, for parts.
    name
    for members in DEFAULT_PART_MANAGER.category_members.values()
    for name in members
} | {
    # Counted progressives that still affect capability — counts must
    # appear in the L2 fingerprint to distinguish e.g. Pad=0 vs Pad=3.
    "Progressive Launch Pad",
    "Progressive R&D",
    "Progressive Science Instrument",
    # Curated-building progressives (buildings_in_logic).  When the option is
    # OFF these items are never in the pool, so their count is always 0 and
    # they never enter the fingerprint — a no-op.  When ON, their counts must
    # be tracked so the cache distinguishes e.g. VAB=0 vs VAB=2.  EVERY
    # building with a capability effect must be here: Mission Control was
    # missing after the navigation gate gave it one (maneuver nodes), so two
    # states differing only in MC shared a fingerprint and the capability
    # cached at MC=0 (navigation blocked) was reused after MC was collected —
    # stranding every interplanetary mission in the strict post_fill sweep.
    "Progressive VAB",
    "Progressive Tracking Station",
    "Progressive Astronaut Complex",
    "Progressive Mission Control",
}

# Terminal velocity threshold for parachute adequacy (m/s)
_MAX_SAFE_LANDING_SPEED: float = 6.0

# Ship cross-section assumed for parachute calc: π*(1.25/2)² ≈ 1.23 m²
# (conservative: assume a 1.25m diameter capsule/probe)
_SHIP_CROSS_SECTION: float = math.pi * (1.25 / 2) ** 2

# How many of each chute kind the physics check may assume on one craft.
# Radial chutes surface-mount around the body in symmetry groups; a group's
# effective drag area scales SUPER-linearly (group^1.5, a KSP quirk — see
# ``_radial_drag_multiplier``), so a realistic count lands a heavy craft.  The
# cap is 7 neat groups of ``_RADIAL_SYMMETRY_GROUP`` (7×8=56) — a generous
# attach-geometry bound.  An inline (stack) chute occupies the craft's top node:
# a returning craft reliably has exactly ONE, and the exceptions (multi-stack-top
# clusters, radial-booster nose mounts, a strut-cube adapter) are geometry we
# deliberately don't model and can't detect — so the conservative assumption
# (Golden Rule) is one inline chute, scaling linearly.
_RADIAL_SYMMETRY_GROUP: int = 8   # KSP's max radial symmetry; bigger = more rings
_MAX_RADIAL_CHUTES: int = 56      # 7 groups of 8
_MAX_INLINE_CHUTES: int = 1

# The part name + mass that provides FUEL_LINE (the asparagus crossfeed
# enabler).  The real parallel-stage builder needs the fuel line's mass/name to
# charge an asparagus crossfeed build.  From the full installed universe — the
# per-seed flag that gates its use is set count-gated in ``_pre_pass``.
_FUEL_LINE_PART: Optional[str] = DEFAULT_PART_MANAGER.fuel_line_part
_FUEL_LINE_MASS: float = DEFAULT_PART_MANAGER.fuel_line_mass

# Minimum jetpack TWR for ladder-free sample return
_MIN_EVA_JETPACK_TWR: float = 1.05

# Mission types that inherently require a Kerbal EVA (walk out of the craft):
# planting a flag, taking a surface sample, and rescuing a stranded kerbal all
# need a kerbal outside the craft.  This drives the curated Astronaut-Complex
# ``can_eva`` gate (buildings_in_logic), for both mission locations and the
# contracts that share these base mission types (flag / sample / rescue).
# EVA-in-orbit shares the ORBIT mission_type, so it can't be inferred from the
# type alone — its caller passes ``requires_eva=True`` explicitly.
MISSION_TYPES_REQUIRING_EVA: frozenset[MissionType] = frozenset({
    MissionType.FLAG_PLANT,
    MissionType.SAMPLE_RETURN,
    MissionType.RESCUE,
})

# Mission types that require a rendezvous — matching orbits with another vessel.
# Rescuing a stranded Kerbal is the sole type today; other callers (e.g. the
# Apollo-split return retry) pass ``requires_rendezvous=True`` explicitly.
# Rendezvous needs patched conics + maneuver nodes (Tracking Station + Mission
# Control), the ``can_rendezvous`` gate — this is the NAVIGATION axis, orthogonal
# to the precise-pointing (attitude-hardware) gate above.
MISSION_TYPES_REQUIRING_RENDEZVOUS: frozenset[MissionType] = frozenset({
    MissionType.RESCUE,
})

# Mission types that take a surface sample (a Kerbal collecting surface material)
# — sample returns.  In stock KSP this needs the R&D facility at index 1 (KSP
# "level 2"), even on the home body, so it gates on ``can_collect_samples``
# regardless of travel — ON TOP OF the EVA gate (which is home-surface-exempt for
# samples: plain home EVA is free, so a home sample needs only R&D).
MISSION_TYPES_REQUIRING_SAMPLES: frozenset[MissionType] = frozenset({
    MissionType.SAMPLE_RETURN,
})


# Home-system bodies that are NOT the home itself (its moons for a planet home,
# or the parent + siblings for a moon home) — the "local" navigation targets.
# Cached per home; consulted only when the local-nav gate is active.
_HOME_SYSTEM_MOONS: dict[BodyName, frozenset] = {}


def _home_system_moons(home: BodyName) -> frozenset:
    moons = _HOME_SYSTEM_MOONS.get(home)
    if moons is None:
        moons = frozenset(home_system_bodies(home)) - {home}
        _HOME_SYSTEM_MOONS[home] = moons
    return moons


@dataclass(frozen=True)
class MissionLogicNeeds:
    """A mission's non-physics logic requirements, as player *capabilities*
    (never buildings/items) plus the comms DSN Tracking-Station level.

    This is the single semantic source for "what does this mission need beyond
    raw dv/rank physics".  It mirrors the capability gate stack in
    ``_evaluate_profile`` clause-for-clause; the sphere-ladder translates it into
    building items at one point (``sphere_ladder._needs_to_counted`` via
    ``effects.buildings_for_capability``) so mission locations and contracts
    cannot drift from the real evaluator or from each other.

    Deliberately EXCLUDES precise pointing (an attitude-*part* gate, satisfied by
    the bracket's reps) and the launch pad (a kit-dependent physics ceiling, kept
    bracket-derived) — both live off the capability/building axis.
    """
    capabilities: "frozenset[Capability]"
    min_ts_dsn_level: int = 0


def mission_logic_needs(
    body: BodyName, mission_type: MissionType, crewed: Optional[bool],
    requires_eva: Optional[bool], mission_builder: MissionBuilder,
) -> MissionLogicNeeds:
    """Capability + comms requirements a mission imposes beyond dv/rank physics.

    Shared by mission locations (via ``sphere_ladder._mission_building_reqs``)
    and contracts (via ``contracts.contract_logic_needs``).  Each clause below
    mirrors a named gate in ``_evaluate_profile``:

    * EVA (``CAN_EVA``) — flag/sample/rescue; home-surface SAMPLE_RETURN rides the
      free home EVA (the ``home_surface_sample`` exemption), so it is excluded.
    * Samples (``CAN_COLLECT_SAMPLES``) — SAMPLE_RETURN, home and off-home.
    * Navigation — interplanetary (target outside the home system) always needs
      ``CAN_NAVIGATE_INTERPLANETARY``; a home-system moon transfer needs
      ``CAN_NAVIGATE_LOCAL`` (scaled by the resolved HomeSystem* options in the
      translation layer).  Rendezvous (RESCUE) always needs ``CAN_RENDEZVOUS``.
    * Comms/DSN — uncrewed missions leaving the home system need the Tracking
      Station at ``min_dsn_level_for`` the minimal antenna the target requires.

    Navigation keys on home-system *membership* (not the profile's edge types) to
    keep mission signatures byte-identical; the equivalence to the real gate's
    ``PLANET_TRANSFER``-edge test is pinned by a unit tripwire.
    """
    from .effects import Capability
    from .comms import min_dsn_level_for
    from .bodies import min_relay_tier

    home = mission_builder.home
    caps: set[Capability] = set()
    needs_travel = any(
        bool(p) for p in mission_builder.profiles_for(body, mission_type))

    eva_required = (requires_eva if requires_eva is not None
                    else mission_type in MISSION_TYPES_REQUIRING_EVA)
    if eva_required:
        home_exempt = (mission_type in MISSION_TYPES_REQUIRING_SAMPLES
                       and body == home)
        if not home_exempt:
            caps.add(Capability.CAN_EVA)

    if mission_type in MISSION_TYPES_REQUIRING_SAMPLES:
        caps.add(Capability.CAN_COLLECT_SAMPLES)

    if needs_travel:
        if body not in home_system_bodies(home):
            caps.add(Capability.CAN_NAVIGATE_INTERPLANETARY)
        elif body != home:
            caps.add(Capability.CAN_NAVIGATE_LOCAL)
    if mission_type in MISSION_TYPES_REQUIRING_RENDEZVOUS:
        caps.add(Capability.CAN_RENDEZVOUS)

    min_ts_dsn_level = 0
    if needs_travel and crewed is not True:
        min_ts_dsn_level = min_dsn_level_for(
            body, home, min_relay_tier(body, home))

    return MissionLogicNeeds(frozenset(caps), min_ts_dsn_level)


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
    lightest_capsule: Optional[MiscEquipment] = None
    lightest_probe: Optional[MiscEquipment] = None
    # Full pod inventories: passive aero-entry profiles pick the pod TOGETHER
    # with its covering heat shield (cheapest pair), which needs the whole set
    # — the lightest pod may be wider than every owned shield while a heavier
    # narrow pod flies fine.
    available_capsules: list[MiscEquipment] = field(default_factory=list)
    available_probes: list[MiscEquipment] = field(default_factory=list)

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
    # Lightest docking port — the Apollo-split rejoin interface (one per
    # docked side, real part mass charged to each stack's manifest).
    lightest_docking_port: Optional[MiscEquipment] = None

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

    # --- Curated-building abilities (buildings_in_logic) -------------------
    # Each defaults to the MAXED value so that when buildings are NOT in logic
    # ``_evaluate_profile`` behaves exactly as it did before this feature (every
    # ability present, ``dsn_power=max`` reduces the comms gate to antenna-only).
    # When buildings_in_logic is on, ``_pre_pass`` overrides these from the
    # collected building-progressive counts via ``effects.player_capabilities``.
    # The system names player *abilities*, never a building or a building level.
    # (VAB/SPH buildable limits are wired but not gated this release — the
    # facilities ship at max; see effects.py for the forward seam.)
    can_eva: bool = True
    can_collect_samples: bool = True
    can_rendezvous: bool = True
    can_navigate_local: bool = True
    can_navigate_interplanetary: bool = True
    dsn_power: float = DSN_POWER_MAX

    # Available part lists (populated by pre-pass)
    available_engines: list[Engine] = field(default_factory=list)
    available_srbs: list[SolidBooster] = field(default_factory=list)
    available_tanks: list[FuelTank] = field(default_factory=list)
    available_heat_shields: list[HeatShield] = field(default_factory=list)
    available_parachutes: list[Parachute] = field(default_factory=list)
    # Best (lowest mass-per-drag-area) chute, picked once at ``_pre_pass`` time
    # so the staged-descent mix evaluator never ``min(...)``s in the hot path.
    # ``best_chute`` is the overall best (display/kit).  The four role slots are
    # main (touchdown) vs drogue (high-q bridge) × radial (scales in symmetry)
    # vs inline (one attach point) — the mix builder combines them.  ``None``
    # when no chute of that role is held.
    best_chute: Optional[Parachute] = None
    best_radial_main: Optional[Parachute] = None
    best_inline_main: Optional[Parachute] = None
    best_radial_drogue: Optional[Parachute] = None
    best_inline_drogue: Optional[Parachute] = None
    # Heaviest-drag shield (the inflatable when owned) — the bleed enabler for
    # thin atmospheres.  Distinct from ``best_heat_shield`` (largest size_class,
    # the coverage pick).  Only used as its OWN candidate mix, so owning it
    # never displaces a lighter mains-only landing.
    best_drag_shield: Optional[HeatShield] = None
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
    # DSN ground-station power (watts), buildings_in_logic. Defaults to max so
    # the transmit-science comms check is antenna-only when the option is off.
    dsn_power: float = DSN_POWER_MAX
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
        PROGRESSIVE_TRACKING_STATION_NAME, PROGRESSIVE_TRACKING_STATION_COUNT,
        PROGRESSIVE_ASTRONAUT_COMPLEX_NAME, PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT,
        PROGRESSIVE_MISSION_CONTROL_NAME, PROGRESSIVE_MISSION_CONTROL_COUNT,
    )
    # VAB/SPH are wired but not capability-gated this release, so they don't
    # appear here (their count would always be 0 and never affect capability).
    _COUNTED = {
        PROGRESSIVE_LAUNCH_PAD_NAME: PROGRESSIVE_LAUNCH_PAD_COUNT,
        PROGRESSIVE_RD_NAME: PROGRESSIVE_RD_COUNT,
        PROGRESSIVE_SCIENCE_INSTRUMENT_NAME: PROGRESSIVE_PSI_COUNT,
        PROGRESSIVE_TRACKING_STATION_NAME: PROGRESSIVE_TRACKING_STATION_COUNT,
        PROGRESSIVE_ASTRONAUT_COMPLEX_NAME: PROGRESSIVE_ASTRONAUT_COMPLEX_COUNT,
        PROGRESSIVE_MISSION_CONTROL_NAME: PROGRESSIVE_MISSION_CONTROL_COUNT,
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
              launch_pad_caps: tuple[float, ...] | None = None,
              buildings_in_logic: bool = False,
              home: BodyName | None = None,
              local_needs_conics: bool = True,
              local_needs_nodes: bool = True) -> EquipmentFlags:
    """
    Iterate every PART_DB item the player has and build EquipmentFlags.

    Phase 2: parts are individual AP items — there is no progressive-tier
    expansion.  ``item_count_fn(item_name)`` returns 1 if the player has
    received that part and 0 otherwise.  The ``Progressive Launch Pad``
    counted progressive is the only non-binary count consulted (drives
    ``launch_pad_mass_cap``).

    ``buildings_in_logic`` (default off) gates curated facility effects on the
    collected building-progressive counts.  When OFF (the default), the
    building effect fields keep their maxed ``EquipmentFlags`` defaults
    (``can_eva=True``, ``dsn_power=max``), so evaluation is identical
    to before this feature existed — a strict no-op.  ``home`` only matters
    when the option is on (none of the curated building effects are home-scaled
    today, but the translation layer signature requires it).
    """
    flags = EquipmentFlags()

    if start_with_clamps:
        flags.has_launch_clamp = True

    # Launch-pad mass cap: index by collected count of "Progressive Launch Pad".
    # Routed through the effects translation layer (the pad cap is the only
    # capability effect consumed today); ``pad_mass_limit_from_caps`` produces
    # the identical value the old inline derivation did.
    if progressive_launch_pad:
        from .items import PROGRESSIVE_LAUNCH_PAD_NAME, PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN
        from .effects import pad_mass_limit_from_caps
        caps = launch_pad_caps or PROGRESSIVE_LAUNCH_PAD_CAPS_KERBIN
        pad_count = item_count_fn(PROGRESSIVE_LAUNCH_PAD_NAME)
        flags.launch_pad_mass_cap = pad_mass_limit_from_caps(caps, pad_count)

    # Curated-building abilities (buildings_in_logic). OFF -> the maxed defaults
    # stand untouched (strict no-op). ON -> translate the collected building
    # counts into player abilities via ``effects.player_capabilities``; the
    # capability system reads only the ability booleans, never the levels.
    if buildings_in_logic:
        from .items import (
            PROGRESSIVE_ASTRONAUT_COMPLEX_NAME, PROGRESSIVE_TRACKING_STATION_NAME,
            PROGRESSIVE_MISSION_CONTROL_NAME, PROGRESSIVE_RD_NAME,
        )
        from .effects import (
            Building, Capability, Effect, building_effects, player_capabilities,
        )
        eva_home = home or BodyName.KERBIN
        ts_level = item_count_fn(PROGRESSIVE_TRACKING_STATION_NAME)
        caps = player_capabilities(
            {
                Building.ASTRONAUT_COMPLEX:
                    item_count_fn(PROGRESSIVE_ASTRONAUT_COMPLEX_NAME),
                Building.TRACKING_STATION: ts_level,
                Building.MISSION_CONTROL:
                    item_count_fn(PROGRESSIVE_MISSION_CONTROL_NAME),
                # R&D facility rides the Progressive R&D count (samples gate).
                Building.RESEARCH_AND_DEVELOPMENT:
                    item_count_fn(PROGRESSIVE_RD_NAME),
            },
            local_needs_conics=local_needs_conics,
            local_needs_nodes=local_needs_nodes,
        )
        flags.can_eva = caps[Capability.CAN_EVA]
        flags.can_collect_samples = caps[Capability.CAN_COLLECT_SAMPLES]
        flags.can_rendezvous = caps[Capability.CAN_RENDEZVOUS]
        flags.can_navigate_local = caps[Capability.CAN_NAVIGATE_LOCAL]
        flags.can_navigate_interplanetary = caps[Capability.CAN_NAVIGATE_INTERPLANETARY]
        # Comms/DSN is the one quantitative ability: translate the Tracking
        # Station level to a physical ground-station power (see comms.py); the
        # comms gate scales the required antenna tier by it.
        flags.dsn_power = building_effects(
            Building.TRACKING_STATION, ts_level, home=eva_home)[Effect.DSN_POWER]

    # Process every part in the installed universe; ``item_count_fn`` gates to
    # what the player actually has (so a disabled pack's parts are excluded).
    for item_name, parts in DEFAULT_PART_MANAGER.parts.items():
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

    # Pick the best parachute per role once.  Lowest mass-per-drag-area wins;
    # the staged-descent mix evaluator (``_solve_atmo_landing``) combines mains
    # (touchdown braking) and drogues (high-q bridge) and finishes any residual
    # propulsively, so a drogue-only or chute-light kit is no longer a hard
    # fail — the burn covers the gap.
    if flags.available_parachutes:
        _chute_key = lambda p: p.mass / max(p.drag_area, 1e-3)
        flags.best_chute = min(flags.available_parachutes, key=_chute_key)
        # Role-aware best-of-kind, picked once here so the staged-descent mix
        # evaluator reads slots instead of scanning.  Split main vs drogue
        # (drogues bridge the high-speed gap; mains do the low-speed braking)
        # and radial vs inline (attach-geometry scaling differs).
        def _best(pred):
            cands = [c for c in flags.available_parachutes if pred(c)]
            return min(cands, key=_chute_key) if cands else None
        flags.best_radial_main = _best(lambda c: c.is_radial and not c.is_drogue)
        flags.best_inline_main = _best(lambda c: not c.is_radial and not c.is_drogue)
        flags.best_radial_drogue = _best(lambda c: c.is_radial and c.is_drogue)
        flags.best_inline_drogue = _best(lambda c: not c.is_radial and c.is_drogue)

    # Heaviest-drag shield for the bleed-enabler mix (the inflatable when owned).
    if flags.available_heat_shields:
        flags.best_drag_shield = max(flags.available_heat_shields,
                                     key=lambda hs: hs.drag_area)

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
        # Record every collected chute; ``_pre_pass`` picks the best part per
        # role (main/drogue × radial/inline) and the staged-descent evaluator
        # (``_solve_atmo_landing``) applies the attach-point caps and combines
        # roles.  Count is irrelevant to selection (best-of-kind), but kept for
        # parity with other part axes.
        flags.has_parachutes = True
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
    for cat_key in DEFAULT_PART_MANAGER.part_to_categories.get(part.name, ()):
        cur = flags.category_lightest.get(cat_key)
        if cur is None or part.mass < cur.mass:
            flags.category_lightest[cat_key] = part
    # A station's crew rides real pressurized cabins, not exposed external
    # seats: gate on crew_cabin category membership (which excludes
    # seatExternalCmd), matching category_lightest above. A raw crew_capacity
    # test admits a "station" of 5 lawn chairs (~0.25t), wrecking the mass model.
    if "crew_cabin" in DEFAULT_PART_MANAGER.part_to_categories.get(part.name, ()):
        flags.available_crew_parts.append(part)
    for flag in part.provides:
        if flag == CF.PROBE_CORE:
            flags.has_probe_core = True
            flags.available_probes.append(part)
            if flags.lightest_probe is None or part.mass < flags.lightest_probe.mass:
                flags.lightest_probe = part
        elif flag == CF.CAPSULE:
            flags.has_capsule = True
            flags.available_capsules.append(part)
            if flags.lightest_capsule is None or part.mass < flags.lightest_capsule.mass:
                flags.lightest_capsule = part
        elif flag == CF.REACTION_WHEEL:
            flags.has_reaction_wheels = True
            # Standalone wheel module = provides reaction_wheel without also
            # providing probe_core or capsule. (Probes/capsules that include
            # a wheel are handled by `_pod_has_built_in_wheels`; they
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
            if (flags.lightest_docking_port is None
                    or part.mass < flags.lightest_docking_port.mass):
                flags.lightest_docking_port = part
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
    # True when feasibility came from the Apollo-split retry, which imposes
    # requires_rendezvous=True beyond what ``mission_logic_needs`` derives from
    # the mission type alone.  Bracket-side consumers must union the rendezvous
    # buildings (conics + nodes) into the location's gate — for interplanetary
    # targets that's a no-op (CAN_NAVIGATE_INTERPLANETARY maps to the same
    # buildings), but a home-SYSTEM heavy-moon return closed only by Apollo
    # (e.g. Laythe-home Tylo Return) would otherwise under-gate to
    # CAN_NAVIGATE_LOCAL's weaker building set.
    via_apollo: bool = False
    # Command module + support equipment for the terminal stage: [(count, part_id), ...]
    terminal_parts: list[tuple[int, str]] = field(default_factory=list)
    # The flown pod.  On passive aero-entry profiles this is pair-picked with
    # its covering shield and may differ from the global lightest pick; the
    # kit must record it or a re-eval of the kit can't reproduce feasibility.
    terminal_pod_name: str = ""
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


# Lightest parts that enable ION (xenon) engines via the large-power gate in
# ``_filter_engines_for_ion``.  Battery is preferred — it carries no rank axis,
# so adding it to a kit doesn't inflate any rank ceiling (the large solar panel
# sits at SOLAR rank 3).  From the full installed universe (see
# ``PartManager.lightest_providing``); the per-seed flags that gate their use
# are set count-gated in ``_pre_pass``.
_BATTERY_LARGE_PART: Optional[str] = DEFAULT_PART_MANAGER.lightest_providing(
    CapabilityFlag.BATTERY_LARGE)
_SOLAR_LARGE_PART: Optional[str] = DEFAULT_PART_MANAGER.lightest_providing(
    CapabilityFlag.SOLAR_ARRAY_LARGE)


def _build_kit_used(flags: EquipmentFlags,
                    stage_results: list[StageResult],
                    terminal_parts: list[tuple[int, str]],
                    terminal_pod_name: str = "") -> KitUsed:
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
    # The FLOWN pod overrides the lightest pick: on passive aero-entry
    # profiles it is pair-picked with its covering shield and may be a
    # heavier-but-narrower pod.  Recording the lightest instead would leave
    # the kit unable to reproduce the feasibility claim on re-eval.
    if terminal_pod_name:
        if any(p.name == terminal_pod_name for p in flags.available_capsules):
            kit.capsule = terminal_pod_name
        elif any(p.name == terminal_pod_name for p in flags.available_probes):
            kit.probe_core = terminal_pod_name
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
    return _build_kit_used(flags, result.stage_results, result.terminal_parts,
                           result.terminal_pod_name)


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


def _pod_has_built_in_wheels(pod: Optional[MiscEquipment]) -> bool:
    return pod is not None and CapabilityFlag.REACTION_WHEEL in pod.provides


def _pod_has_built_in_monoprop(pod: Optional[MiscEquipment]) -> bool:
    return pod is not None and pod.name in _TERMINAL_PARTS_WITH_INTERNAL_MONOPROP


def _passive_entry_pod_and_shield(
    pods: list[MiscEquipment],
    heat_shields: list[HeatShield],
    n_entries: int,
    extra_pod_cost=None,
) -> Optional[tuple[MiscEquipment, HeatShield]]:
    """Cheapest (pod, covering shield) pair for a profile with ``n_entries``
    passive aero entries, or None when no owned shield covers any owned pod.

    The reentry shield must COVER the command module it protects; flying a
    smaller shield is not modelled (the pod burns).  Picking the pod and the
    shield *together* — minimising pod.mass + n_entries * shield.mass over the
    owned set — keeps the model monotone: the part DB has capsules that are
    lighter but WIDER than others (cupola: 0.94t/2.5m), so a fixed
    lightest-pod pick would let acquiring one flip a covered pod to an
    uncoverable one and lose the mission (the bug-092 shape).  A pure min over
    a growing candidate set can only improve.  Both masses are real flown
    parts, so the charge stays conservative.

    ``extra_pod_cost`` extends the same pair-pick to architectures where the
    pod carries an additional per-pod gear consequence (the Apollo docking
    approach: ``_docking_approach_gear`` mass).  It returns that mass for a
    pod, or None when the pod can't fly the architecture at all — those pods
    are skipped, exactly like an uncoverable one.
    """
    best: Optional[tuple[MiscEquipment, HeatShield]] = None
    best_key: tuple[float, float, str] = (float("inf"), float("inf"), "")
    for pod in pods:
        extra = 0.0
        if extra_pod_cost is not None:
            e = extra_pod_cost(pod)
            if e is None:
                continue
            extra = e
        shield = None
        for hs in heat_shields:
            if hs.size_class >= pod.size_class and (
                    shield is None or hs.mass < shield.mass):
                shield = hs
        if shield is None:
            continue
        key = (pod.mass + extra + n_entries * shield.mass, pod.mass, pod.name)
        if key < best_key:
            best_key, best = key, (pod, shield)
    return best


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


def _rcs_bundle(
    flags: EquipmentFlags, terminal_pod: Optional[MiscEquipment],
) -> Optional[AttitudeBundle]:
    """Concrete RCS translation/attitude kit from real PART_DB parts:
    4 × lightest standalone RCS thruster + (1 × lightest monoprop tank,
    unless the terminal payload already supplies MonoPropellant internally).

    Returns None when the kit can't fly RCS at all (no thruster, or no
    monopropellant source).  Shared by the attitude bundle (RCS-vs-wheel
    trade) and the Apollo docking gear (the docking approach mandates RCS
    below expert gameplay — ``docking_needs_rcs``; a wheel is not a
    substitute for translation, nor RCS for the always-required wheels).
    """
    t = flags.lightest_rcs_thruster
    if t is None:
        return None
    parts: list[tuple[int, str]] = [(4, t.name)]
    mass = 4 * t.mass
    if not _pod_has_built_in_monoprop(terminal_pod):
        tank = flags.lightest_monoprop_tank
        if tank is None:
            # Can't fly an RCS bundle without monopropellant.
            return None
        parts.append((1, tank.name))
        mass += tank.dry_mass + tank.fuel_mass
    return AttitudeBundle(mass=mass, parts=tuple(parts))


def _docking_approach_gear(
    flags: EquipmentFlags, pod: Optional[MiscEquipment],
    gameplay: GameplayDifficulty,
) -> Optional[AttitudeBundle]:
    """Docking attitude gear the active vehicle flies with ``pod``: torque
    ALWAYS (built-in pod wheels or a charged standalone module) and, below
    expert gameplay, the RCS translation kit (``_rcs_bundle`` — monoprop tank
    included unless the pod carries internal monoprop).  None when this pod
    cannot dock with the current kit.

    Single source of truth for BOTH the Apollo pod pick (pod.mass +
    gear.mass, the same pair-pick shape as ``_passive_entry_pod_and_shield``)
    and the approach-gear charge in ``_apollo_split_for`` — pick basis must
    equal charge, or a lighter pod whose gear consequence is huge (kv3Pod
    forcing the kit's only monoprop tank, a Mk3 fuselage, onto the lander)
    wins the pick and a BIGGER kit loses the mission (bug-092 shape; the
    docking-gear pod-pick facet of bugs/105).
    """
    parts: list[tuple[int, str]] = []
    mass = 0.0
    if not _pod_has_built_in_wheels(pod):
        wheel = flags.lightest_reaction_wheel
        if wheel is None:
            return None
        parts.append((1, wheel.name))
        mass += wheel.mass
    if gameplay.docking_needs_rcs:
        rcs = _rcs_bundle(flags, pod)
        if rcs is None:
            return None
        parts.extend(rcs.parts)
        mass += rcs.mass
    return AttitudeBundle(mass=mass, parts=tuple(parts))


def _attitude_bundle_for_stage(
    flags: EquipmentFlags, terminal_pod: Optional[MiscEquipment],
) -> Optional[AttitudeBundle]:
    """Pick the lightest concrete on-stage attitude bundle from real PART_DB
    parts. Returns None if no on-stage source is available.

    Two candidate bundles, real masses only:
      - **Wheel**: 1 × lightest standalone reaction-wheel module.
      - **RCS**:   the shared :func:`_rcs_bundle`.

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
    rcs = _rcs_bundle(flags, terminal_pod)
    if rcs is not None:
        candidates.append(rcs)
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
    requires_eva: bool | None = None,
    requires_rendezvous: bool | None = None,
    requires_samples: bool | None = None,
    requires_precise_pointing: bool = False,
    gameplay: GameplayDifficulty = CONSERVATIVE_GAMEPLAY,
    apollo_split: bool = False,
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

    ``apollo_split`` evaluates the Apollo architecture instead of the
    whole-stack cascade: the return stack parks in destination orbit while
    the lander flies descent + ascent carrying only the pod (+ delivered
    payload + docking gear), then rejoins for the trip home.  Gated on a
    docking port + docking attitude gear — wheels always, +RCS below expert
    gameplay (see ``_apollo_split_for``); callers try the standard
    architecture first and only retry with this on failure.
    """
    # EVA / rendezvous / surface-sample requirements.  An explicit override wins
    # (EVA-in-orbit forces requires_eva=True; docking/station contracts force
    # requires_rendezvous=True); otherwise derive from the mission type.  Kept in
    # one place so every evaluation entry point gates identically — the per-body
    # ``_try_profiles*`` path used to omit samples/rendezvous, silently skipping
    # those gates.
    if requires_eva is None:
        requires_eva = mission_type in MISSION_TYPES_REQUIRING_EVA
    if requires_rendezvous is None:
        requires_rendezvous = mission_type in MISSION_TYPES_REQUIRING_RENDEZVOUS
    if requires_samples is None:
        requires_samples = mission_type in MISSION_TYPES_REQUIRING_SAMPLES

    # ------------------------------------------------------------------
    # Forward pass — broad gate checks
    # ------------------------------------------------------------------

    has_aero_edge = any(e.needs_heat_shield for e in profile)
    has_atmo_ascent = any(e.edge_type == EdgeType.ATMOSPHERIC_ASCENT for e in profile)
    has_vacuum_land = any(e.edge_type == EdgeType.VACUUM_LANDING for e in profile)
    has_atmo_land = any(e.edge_type == EdgeType.ATMO_LANDING for e in profile)
    has_land = has_vacuum_land or has_atmo_land
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

    # Precise pointing: a mission that must hold a fixed attitude with the engine
    # off — a specific target orbit (equatorial/polar/stationary/random), a space
    # station, or a kerbal rescue's fine approach. Engine gimbal only steers under
    # thrust, so on casual/normal physics this needs a reaction wheel or RCS (a
    # wheel-bearing pod counts). Expert (small/zero) is trusted to fly these on
    # gimbal alone, so the profile leaves it False. Physically joining two craft
    # (docking — RCS + a docking port) is a separate concern from holding a fixed
    # attitude and is modelled elsewhere, not by this gate.
    if (requires_precise_pointing
            and gameplay.precise_pointing_needs_reaction_control
            and not (flags.has_reaction_wheels or flags.has_rcs)):
        blocking.append(BlockingInfo(reason=BlockingReason.NO_PRECISE_ATTITUDE))

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

    # EVA (Astronaut Complex, buildings_in_logic).  ``flags.can_eva`` defaults
    # True, so when buildings aren't in logic this never fires.  Stock AC level 0
    # permits ONLY plain surface EVA on the home body; flag planting and any
    # orbital EVA need the AC upgrade even at home, and all EVA off-home needs it
    # too (verified against in-game truth tables, Kerbin and alien homes alike).
    # Surface samples ride that free home-surface EVA, so a home-surface
    # SAMPLE_RETURN is exempt here — its separate R&D gate below still applies.
    if requires_eva and not flags.can_eva:
        home_surface_sample = (
            mission_type == MissionType.SAMPLE_RETURN
            and not any(edge.body != home for edge in profile))
        if not home_surface_sample:
            blocking.append(BlockingInfo(reason=BlockingReason.CANNOT_EVA))

    # Surface samples (R&D facility, buildings_in_logic).  Needs the facility
    # upgraded even on the home body (stock), so unlike EVA there is no home
    # exemption — the gate fires regardless of travel.  ``can_collect_samples``
    # defaults True, so it never fires when buildings aren't in logic.
    if requires_samples and not flags.can_collect_samples:
        blocking.append(BlockingInfo(reason=BlockingReason.CANNOT_COLLECT_SAMPLES))

    # Navigation + rendezvous (Tracking Station patched conics + Mission Control
    # maneuver nodes, buildings_in_logic).  All three abilities default True, so
    # when buildings aren't in logic none of this fires (the guard is skipped).
    # These gate crewed AND uncrewed alike — a pilot doesn't remove the need to
    # plan a transfer.  Interplanetary (a PLANET_TRANSFER edge) and rendezvous
    # always need conics+nodes; a home-system (moon) transfer scales with the
    # HomeSystem* options (resolved into ``can_navigate_local``).
    if not (flags.can_navigate_interplanetary and flags.can_navigate_local
            and flags.can_rendezvous):
        interplanetary = any(e.edge_type == EdgeType.PLANET_TRANSFER
                             for e in profile)
        if interplanetary:
            if not flags.can_navigate_interplanetary:
                blocking.append(BlockingInfo(
                    reason=BlockingReason.CANNOT_NAVIGATE_INTERPLANETARY,
                    body=next(e.body for e in profile
                              if e.edge_type == EdgeType.PLANET_TRANSFER)))
        elif not flags.can_navigate_local:
            moon = next((e.body for e in profile
                         if e.body in _home_system_moons(home)), None)
            if moon is not None:
                blocking.append(BlockingInfo(
                    reason=BlockingReason.CANNOT_NAVIGATE_LOCAL, body=moon))
        if (not flags.can_rendezvous
                and (requires_rendezvous
                     or mission_type in MISSION_TYPES_REQUIRING_RENDEZVOUS)):
            blocking.append(BlockingInfo(reason=BlockingReason.CANNOT_RENDEZVOUS))

    # Heat shield — required for any aero edge (reentry heating).  A staged
    # atmospheric landing always carries one; the exotic fully-propulsive
    # descent that could skip it is a deliberate conservative omission (see
    # _solve_atmo_landing).  No separate parachute gate: chutes are optional now
    # — a shield + throttleable-engine landing finishes the descent propulsively
    # (the mix evaluator decides, emitting ATMO_DESCENT_INFEASIBLE if neither
    # drag nor a burn can land the craft).
    if has_aero_edge and not flags.has_heat_shield:
        blocking.append(BlockingInfo(reason=BlockingReason.NO_HEAT_SHIELD))

    # Power: check each unique body in the profile
    # Detect if aero edges destroy fixed solar panels
    body_aero_destroyed: dict[str, bool] = {}  # body -> whether fixed solar destroyed
    post_aero = False
    for edge in profile:
        # The home-recovery descent (is_recovery) is the final leg; solar loss
        # there needs no further power.  Any OTHER aero edge leaves the craft
        # potentially panel-less for the rest of the mission.
        if edge.needs_heat_shield and not edge.is_recovery:
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
    #
    # DSN (Tracking Station, buildings_in_logic) rides on top: below max power
    # the ground station is weaker, so the antenna needs a higher tier to hold
    # the same link.  ``dsn_power`` defaults to max (option off) -> ``dsn_table``
    # is None -> antenna-only check, byte-identical to before.  The shortfall is
    # always charged to the Tracking Station (never the antenna), because a
    # max-power + ``edge.relay_tier`` antenna reaches by construction.
    if not is_crewed:
        dsn_table = (dsn_required_relay_table(home, flags.dsn_power)
                     if flags.dsn_power < DSN_POWER_MAX else None)
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
            if dsn_table is not None and flags.relay_tier < dsn_table[edge.body]:
                blocking.append(BlockingInfo(
                    reason=BlockingReason.DSN_POWER_INSUFFICIENT,
                    body=edge.body,
                    relay_needed=dsn_table[edge.body],
                    relay_available=flags.relay_tier,
                ))
                break  # one comms failure is sufficient

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
        # ATMO_LANDING is NOT statically propulsive: a passive chute descent
        # needs no engine at all.  Whether a landing burn (and thus an engine)
        # is required is decided by _solve_atmo_landing per kit, which raises
        # ATMO_DESCENT_INFEASIBLE if a burn is needed but unavailable — so it is
        # deliberately absent from both propulsion branches here.
        if et == _ET.ATMOSPHERIC_ASCENT:
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

    # Passive aero entries (chute + shield do all the work) must fly a shield
    # that COVERS the pod — see ``_passive_entry_pod_and_shield``.  Pick the
    # (pod, shield) pair here so the terminal payload below carries the chosen
    # pod's mass through every stage; when no owned shield covers any owned
    # pod the profile is infeasible outright.
    passive_pod: Optional[MiscEquipment] = None
    passive_pod_shield: Optional[HeatShield] = None
    passive_entry_groups = [
        g for g in groups
        if all(e.edge_type == EdgeType.ATMO_LANDING for e in g)
        and any(e.needs_heat_shield for e in g)
    ]
    if passive_entry_groups:
        pods = flags.available_capsules if is_crewed else flags.available_probes
        # Empty pods / shields are already blocked by the NO_CAPSULE /
        # NO_PROBE_CORE / NO_HEAT_SHIELD gates above — only report the more
        # specific "too small" when both exist and no pair covers.
        if pods and flags.available_heat_shields:
            pair = _passive_entry_pod_and_shield(
                pods, flags.available_heat_shields, len(passive_entry_groups),
                # On the Apollo retry the pod also pays its docking-gear
                # consequence (bugs/106) — score it into the pair.
                extra_pod_cost=(
                    (lambda p: getattr(_docking_approach_gear(
                        flags, p, gameplay), "mass", None))
                    if apollo_split else None))
            if pair is None:
                blocking.append(BlockingInfo(
                    reason=BlockingReason.HEAT_SHIELD_TOO_SMALL,
                    body=passive_entry_groups[0][0].body,
                    size_needed=min(p.size_class for p in pods),
                    size_available=max(
                        hs.size_class for hs in flags.available_heat_shields),
                ))
            else:
                passive_pod, passive_pod_shield = pair

    # Return all collected pre-check failures before attempting optimization.
    if blocking:
        return ProfileResult(False, blocking=blocking)

    # ------------------------------------------------------------------
    # Backward pass — compute masses from destination back to Kerbin
    # ------------------------------------------------------------------

    # Terminal payload mass — the flown pod.  On passive aero-entry profiles
    # this is the pod pair-picked with its covering shield above; otherwise
    # the lightest pod of the required kind.
    terminal_pod = passive_pod if passive_pod is not None else (
        flags.lightest_capsule if is_crewed else flags.lightest_probe)
    # Apollo pod pick (bugs/106): on the docking architecture the pod's true
    # cost includes its approach-gear consequence — a lighter pod without
    # internal monoprop can force the kit's only (huge) monoprop tank onto
    # the lander, blow the pad cap, and a BIGGER kit loses the mission
    # (bug-092 shape; seed receipt: eeloo/SSR, +kv3Pod 155.8t→infeasible).
    # Pair-pick pod + gear with the same helper _apollo_split_for charges
    # with; a pure min over a growing candidate set can only improve.
    if apollo_split and passive_pod is None:
        _cands = (flags.available_capsules if is_crewed
                  else flags.available_probes)
        _best = None
        _best_total = float("inf")
        for _p in _cands:
            _g = _docking_approach_gear(flags, _p, gameplay)
            if _g is not None and _p.mass + _g.mass < _best_total:
                _best_total = _p.mass + _g.mass
                _best = _p
        if _best is not None:
            terminal_pod = _best
    pod_mass = terminal_pod.mass if terminal_pod else 0.0
    terminal_mass = max(pod_mass, 0.08 if is_crewed else 0.04)

    # Add equipment mass for the terminal stage
    # (legs, ladder, heat shield on the last edge in the profile)
    terminal_equip = _terminal_equipment_mass(
        profile, flags, home=home, is_crewed=is_crewed)
    # Contract-required equipment delivered to the destination (drill, ore tank,
    # …). Added to the terminal payload so all stages below carry it.
    extra_payload_mass = sum(p.mass for p in extra_payload_parts)
    payload = terminal_mass + terminal_equip + extra_payload_mass

    # Apollo split: resolve the lander boundary + docking gear now (needs the
    # terminal pod for the RCS monoprop decision).  Not applicable → nothing
    # new to report: the caller already ran the standard architecture, so the
    # honest failure reasons are that attempt's.
    apollo: Optional[_ApolloSplit] = None
    if apollo_split:
        apollo = _apollo_split_for(groups, home, flags, terminal_pod,
                                   is_crewed, gameplay)
        if apollo is None:
            return ProfileResult(False, launch_mass=payload)
    # Wet mass of the parked return stack, stashed when the reverse walk
    # crosses from the parked groups into the lander ascent group.
    apollo_parked_wet = 0.0

    # Global attitude strategy.  Every stage whose edges require attitude
    # control must have an on-stage source: a gimballed engine/SRB (free) or
    # the lightest wheel/RCS bundle (its real mass).  The bundle is an
    # optimizer OPTION, never a mandate: ``attitude_module_mass`` charges it
    # per candidate inside ``find_optimal_stage`` (gimballed candidates fly
    # free, ungimballed carry the bundle), so owning a wheel can only widen
    # the candidate set.  The old shape — flat-charging the bundle onto the
    # last attitude stage whenever one existed in the kit — made the model
    # non-monotone (bug 092 family): granting advSasModule added mandatory
    # mass that a gimballed build never needed, flipping missions infeasible
    # at the launch-pad mass cap.  A terminal pod with built-in wheels covers
    # the whole flight (it rides at the top of every stage's stack).
    attitude_has_any = any(
        any(e.requires_attitude_control for e in g) for g in groups
    )
    global_attitude_bundle: Optional[AttitudeBundle] = None
    global_attitude_force_gimbal: bool = False
    attitude_covered_by_terminal = _pod_has_built_in_wheels(terminal_pod)
    if _PER_STAGE_ATTITUDE_ENABLED and attitude_has_any:
        if not attitude_covered_by_terminal:
            global_attitude_bundle = _attitude_bundle_for_stage(flags, terminal_pod)
            if global_attitude_bundle is None:
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

        if apollo is not None:
            if flight_idx == apollo.ascent_gidx:
                # Park everything above (return transfer + home entry) in
                # destination orbit; the lander flies down/up with only the
                # pod + delivered payload (docking gear is charged as this
                # group's equipment below).  The pod is deliberately counted
                # in BOTH stacks on the outbound legs — it physically rides
                # the lander, while the parked stack stays sized as if
                # already carrying it home — a conservative double-count.
                apollo_parked_wet = payload
                payload = terminal_mass + terminal_equip + extra_payload_mass
            elif flight_idx == apollo.land_gidx - 1:
                # Rejoin for the outbound legs: everything below the landing
                # hauls the full lander stack AND the parked return stack.
                payload += apollo_parked_wet

        body = BODY_BY_NAME[group[0].body]
        solar_au = body.solar_distance_au

        # Filter ION engines for this body
        eligible_engines = _filter_engines_for_ion(
            flags.available_engines, solar_au, flags
        )

        from .bodies import EdgeType as ET

        # Staged atmospheric-landing mix for this group (if any).  Decided from
        # the running ``payload`` (the delivered surface mass — everything above
        # this stage), so it must be computed before req_dv / in_atmo / min_twr.
        # A passive mix skips the optimizer (synthetic stage below); a burn mix
        # folds its dv into req_dv and imposes atmo ISP + TWR floor + throttle.
        atmo_land_edges = [e for e in group if e.edge_type == ET.ATMO_LANDING]
        landing_mix: Optional[LandingMix] = None
        landing_needs_burn = False
        landing_burn_dv = 0.0
        _landing_twr_floor = max(1.3, diff.min_twr_atmo)
        if atmo_land_edges:
            _land_edge = atmo_land_edges[0]
            _land_body = BODY_BY_NAME[_land_edge.body]
            # Coverage shield + pod come from the pod/shield pair-pick above
            # (part-packs' covering rule: no undersized fallback — a profile
            # with no covering shield was already blocked HEAT_SHIELD_TOO_SMALL).
            # Landing-site elevation: the home pad and highlands-in-logic
            # bodies (Eve) touch down at the elevated site — thinner air,
            # higher terminal velocity, shorter braking column.  Mirrors the
            # ascent-dv site rule (bodies.py trunk-ascent: home pad or
            # assume_highlands_landing pays the pad figure).
            _land_ground = (
                _land_body.pad_altitude_m
                if (_land_body.assume_highlands_landing
                    or _land_body.name == home) else 0.0)
            landing_mix = _solve_atmo_landing(
                payload, _land_body, flags, diff,
                twr_floor=_landing_twr_floor,
                v_entry=_land_edge.entry_speed,
                dvGL_cap=_land_body.dv.dvGL or 0.0,
                coverage_shield=passive_pod_shield,
                pod_size=terminal_pod.size_class if terminal_pod else 0.0,
                ground_altitude_m=_land_ground,
            )
            if not landing_mix.feasible:
                return ProfileResult(False, launch_mass=payload, blocking=[BlockingInfo(
                    reason=BlockingReason.ATMO_DESCENT_INFEASIBLE,
                    body=_land_body.name,
                    dv_needed=landing_mix.burn_dv,
                    residual_speed=landing_mix.residual_speed,
                )])
            landing_needs_burn = landing_mix.needs_burn
            if landing_needs_burn:
                # A burn landing needs active attitude control (the old
                # propulsive-landing edge demanded it); a passive chute descent
                # does NOT, so this gate is conditional on the mix, not the edge.
                if not _has_attitude_control(flags):
                    return ProfileResult(False, launch_mass=payload, blocking=[
                        BlockingInfo(reason=BlockingReason.NO_ATTITUDE_CONTROL)])
                landing_burn_dv = landing_mix.burn_dv

        # Compute effective dv with difficulty margins
        base_dv = sum(e.base_dv for e in group)
        pc_dv = sum(e.plane_change_dv for e in group)
        # Apollo rejoin: the lander's ascent ends in a rendezvous + docking
        # with the parked stack — charge the phasing/matching burn here so
        # the margins below apply to it like any other burn.
        if apollo is not None and flight_idx == apollo.ascent_gidx:
            base_dv += _APOLLO_RENDEZVOUS_DV
        # Landing edges carry base_dv=0; a pure landing group must not pick up
        # the spurious fixed_margin floor that effective_dv(0) would add.
        if atmo_land_edges and base_dv <= 1e-9:
            req_dv = 0.0
        else:
            req_dv = effective_dv(base_dv, diff, plane_change_dv=pc_dv)
        # The landing burn gets percent_margin only (no fixed_margin): the burn
        # factors already embed the loss margins, and a fixed 100-200 m/s adder
        # would swamp a ~30 m/s finish burn (design decision).
        req_dv += landing_burn_dv * (1.0 + diff.percent_margin)

        # Group-level constraints (union = strictest).  Only propulsive burns
        # force atmospheric ISP and higher TWR floors — a passive aero landing
        # or aerocapture is not engine-powered, so it must not contaminate the
        # ISP selection for the rest of the group (e.g. vacuum burns).  A burn
        # LANDING, however, IS atmospheric propulsion.
        atmo_types = {
            ET.ATMOSPHERIC_ASCENT,
        }
        in_atmo = any(e.edge_type in atmo_types for e in group) or landing_needs_burn

        # Per-edge TWR → acceleration conversion to handle merged groups that
        # span bodies with very different gravities.  A Gilly VL min_twr=1.2
        # at g=0.049 m/s² must not be evaluated at Kerbin g=9.81 m/s².
        _min_accel = 0.0
        for _e in group:
            if _e.min_twr > 0:
                _g_e = BODY_BY_NAME[_e.body].surface_gravity
                _floor = diff.min_twr_atmo if _e.edge_type in atmo_types else diff.min_twr_vac
                _min_accel = max(_min_accel, max(_e.min_twr, _floor) * _g_e)
        if landing_needs_burn:
            _lg = BODY_BY_NAME[atmo_land_edges[0].body].surface_gravity
            _min_accel = max(_min_accel, _landing_twr_floor * _lg)
        min_twr = _min_accel / body.surface_gravity if body.surface_gravity > 0 else 0.0

        req_throttle = (any(e.requires_throttleable for e in group)
                        or landing_needs_burn
                        # The docking approach needs fine thrust control.
                        or (apollo is not None
                            and flight_idx == apollo.ascent_gidx))
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
        # Apollo docking gear — real part masses, manifest == charge.  The
        # lander (active vehicle) carries its port + the docking attitude
        # gear (wheels always; +RCS below expert gameplay) down and back
        # up; the parked stack's port AND its control gear
        # (probe core / attitude / power — see _ApolloSplit.parked_gear)
        # ride the bottom of the first parked group (the docking interface
        # and the stack's brain stay with the transfer stage — they are not
        # carried through the home entry above it).
        if apollo is not None:
            if flight_idx == apollo.ascent_gidx:
                equip_mass += apollo.port.mass + apollo.approach_gear.mass
                stage_equipment.append((1, apollo.port.name))
                stage_equipment.extend(apollo.approach_gear.parts)
            elif flight_idx == apollo.ascent_gidx + 1:
                equip_mass += apollo.port.mass + apollo.parked_gear_mass
                stage_equipment.append((1, apollo.port.name))
                stage_equipment.extend(apollo.parked_gear_parts)
        # Staged atmospheric landing: the whole descent kit (coverage shield +
        # chutes) is fixed PAYLOAD mass on this stage.  The shield protects the
        # pod during the aero bleed and is jettisoned before any touchdown burn,
        # so it is NOT an optimizer heat-shield the landing engine must fit
        # under (the engine fires post-entry) — the mix already sized it to the
        # pod, so drop needs_hs here.  Ascent / aerocapture stages keep the real
        # per-engine shield model.
        landing_shield_name: Optional[str] = None
        if landing_mix is not None:
            needs_hs = False
            equip_mass += landing_mix.hardware_mass
            stage_equipment.extend(landing_mix.equipment)
            if landing_mix.shield is not None:
                landing_shield_name = landing_mix.shield[2]
                # A burn-landing's stage comes from the optimizer (no shield of
                # its own), so put the shield(s) on the manifest here; the
                # passive branch reports it via heat_shield_name instead.
                if landing_mix.needs_burn:
                    stage_equipment.append(
                        (landing_mix.shield_count, landing_shield_name))
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
        # available we force the optimizer to pick a gimbal engine.  When
        # aero surfaces ARE available, 4x the lightest surface (min. needed
        # for control on all axes) is charged per candidate to UNGIMBALLED
        # propulsion only (``aero_steering_mass``) — a gimballed build never
        # pays for fins it doesn't need, so owning fins can't make a mission
        # infeasible (bug 092 family).
        has_atmo_ascent_in_group = any(
            e.edge_type == ET.ATMOSPHERIC_ASCENT for e in group
        )
        needs_gimbal_engine = (
            has_atmo_ascent_in_group and not flags.has_aero_control_surface
        )
        stage_aero_mass = 0.0
        if has_atmo_ascent_in_group and flags.lightest_aero_control:
            stage_aero_mass = 4.0 * flags.lightest_aero_control.mass
        # Landing legs (equip_mass) are charged whenever a stage needs them,
        # independent of the heat shield.  The pre-refactor code routed
        # equip_mass through ``heat_shield_mass`` and zeroed it when the stage
        # needed no shield, silently dropping leg mass on powered (no-shield)
        # vacuum-body landings — an anti-conservative under-charge.
        stage_payload = payload + equip_mass
        # Attitude control for this group.  If a wheel/RCS bundle exists, the
        # optimizer trades "gimballed alone" against "ungimballed + bundle"
        # per candidate (``attitude_module_mass``); the bundle parts land on
        # the manifest below only when an ungimballed choice actually won.
        # If no bundle is available anywhere on the rocket, the stage must
        # self-provide via a gimballed engine/SRB.  Atmospheric ascent stages
        # have their own gimbal-or-aero gate above (separate concern).
        group_needs_attitude = (any(e.requires_attitude_control for e in group)
                                and not attitude_covered_by_terminal)
        stage_attitude_mass = (
            global_attitude_bundle.mass
            if (group_needs_attitude and global_attitude_bundle is not None)
            else 0.0
        )
        if (global_attitude_force_gimbal
                and group_needs_attitude
                and not needs_gimbal_engine):
            needs_gimbal_engine = True

        if landing_mix is not None and not landing_mix.needs_burn:
            # Passive descent: drag alone reaches a safe touchdown, so this is a
            # synthetic zero-dv stage (no engine optimizer).  The whole descent
            # kit (shield + chutes) is already folded into stage_payload via
            # equip_mass and onto stage_equipment above.
            passive_mass = stage_payload
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
                heat_shield_name=landing_shield_name,
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
            srb_needs_rcs=gameplay.srb_needs_rcs,
            player_has_rcs=flags.has_rcs,
            tanks_by_fuel_type=flags.tanks_by_fuel_type,
            available_multi_mounts=flags.available_multi_mounts,
            require_gimbal=needs_gimbal_engine,
            attitude_module_mass=stage_attitude_mass,
            aero_steering_mass=stage_aero_mass,
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
            # Escalated build caps, two offline-probed eligibility channels
            # (the standard architecture everywhere else keeps the default
            # caps — the hot path never searches the escalated space,
            # bug 094):
            #   * Apollo lander ascents on ESCALATED_ASCENT_EDGES (e.g.
            #     Eve's ~9 km/s lander ascent needs K=3 + wide asparagus);
            #   * the world's OWN home ascent on the allowlist-bounded
            #     ESCALATED_HOME_ASCENT_EDGES (Eve-home mesa launch) — an
            #     operator-approved hot-path exception, applied on the
            #     PRIMARY evaluation because every mission from that home
            #     traverses it.
            _asc_tuples = [(e.body, e.edge_type) for e in group
                           if e.edge_type in (ET.ATMOSPHERIC_ASCENT,
                                              ET.VACUUM_ASCENT)]
            _escalate = (
                (apollo is not None and flight_idx == apollo.ascent_gidx
                 and any(t in _escalated_edges() for t in _asc_tuples))
                or any(t[0] == home and t in _escalated_home_edges()
                       for t in _asc_tuples)
            )
            _esc_kwargs: dict = {}
            if _escalate:
                _esc_kwargs = dict(
                    max_ascent_stages=ESCALATED_MAX_ASCENT_STAGES,
                    booster_counts=ESCALATED_BOOSTER_COUNTS,
                    max_eng_per_col=ESCALATED_MAX_ENG_PER_COL,
                )
            ms_diag_out: list = []
            ms_partial_out: list = []
            multistage = find_optimal_multistage_ascent(
                **_esc_kwargs,
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
                srb_needs_rcs=gameplay.srb_needs_rcs,
                player_has_rcs=flags.has_rcs,
                attitude_module_mass=stage_attitude_mass,
                aero_steering_mass=stage_aero_mass,
                body_name=body.name,
                launch_pad_mass_cap=flags.launch_pad_mass_cap,
                atm_scale_height_m=body.atm_scale_height_m,
                atm_top_m=body.safe_altitude_km * 1000.0 if body.has_atmosphere else 0.0,
                pad_altitude_m=body.pad_altitude_m,
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
            # Bottom stage carries the group-level equipment (ladder etc.)
            # for the multi-stage ascent.
            multistage[0].equipment = stage_equipment + multistage[0].equipment
            # Control surcharges were applied per candidate inside the
            # optimizer; the parts land on exactly the sub-stages whose
            # winning propulsion paid for them (manifest mass == charged
            # mass).
            for sr in multistage:
                if sr.carries_attitude_module:
                    sr.equipment = (sr.equipment
                                    + list(global_attitude_bundle.parts))
                if sr.carries_aero_steering:
                    sr.equipment = (sr.equipment
                                    + [(4, flags.lightest_aero_control.name)])
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

        # (Landing-mix chutes were added to stage_equipment before the passive
        # branch above; a burn-landing group falls through to here with them
        # already on the manifest and the burn folded into req_dv.)

        # Ladder
        if any(e.needs_ladder for e in group) and flags.lightest_ladder:
            stage_equipment.append((1, flags.lightest_ladder.name))

        # Prepend group equipment; KEEP what the optimizer already attached
        # (a parallel build's radial decouplers + fuel lines), else they're
        # lost from both the displayed build and the kit/gating.
        result.equipment = stage_equipment + result.equipment
        # Control surcharges: on the manifest only when the winning propulsion
        # paid for them — the optimizer charged exactly those candidates.
        if result.carries_attitude_module:
            result.equipment = result.equipment + list(global_attitude_bundle.parts)
        if result.carries_aero_steering:
            result.equipment = result.equipment + [(4, flags.lightest_aero_control.name)]
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
    if terminal_pod is not None:
        terminal_parts.append((1, terminal_pod.name))
    support_mass, support_parts = _support_equipment_mass(
        flags, profile, home=home, is_crewed=is_crewed)
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
        terminal_pod_name=terminal_pod.name if terminal_pod else "",
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
        if nxt.edge_type in (ET.VACUUM_LANDING, ET.ATMO_LANDING):
            splits.append(i)
            continue

        # Split after any landing edge (ascent from surface is separate)
        if cur.edge_type in (ET.VACUUM_LANDING, ET.ATMO_LANDING):
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
    # ATMO_LANDING is treated as atmospheric here (a landing that needs a burn
    # is atmospheric propulsion) so a forced merge never puts it with a vacuum
    # burn and contaminates ISP; landing is normally its own group anyway.
    atmo_types = {ET.ATMOSPHERIC_ASCENT, ET.ATMO_LANDING}

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


# ---------------------------------------------------------------------------
# Apollo split — leave the return stack parked in destination orbit
# ---------------------------------------------------------------------------

# Rendezvous budget for the lander's post-ascent rejoin with the parked
# return stack: the same low-orbit phasing + matching burn the rescue
# contract charges.  Added to the ascent group's base_dv, so the difficulty
# margins apply via effective_dv like any other burn.
_APOLLO_RENDEZVOUS_DV: float = MissionBuilder._RESCUE_RENDEZVOUS_DV

# Offline override for the escalated-edge set — generate_feasibility.py sets
# this while probing per-edge eligibility (the checked-in set is that probe's
# OUTPUT, so the probe can't read it).  Production code never touches it.
_ESCALATION_OVERRIDE: Optional[frozenset] = None
# Same contract for the HOME-ascent escalation set (see
# _escalated_home_edges).
_HOME_ESCALATION_OVERRIDE: Optional[frozenset] = None


def _escalated_edges() -> frozenset:
    """The (body, EdgeType) ascent edges eligible for escalated build caps."""
    return (_ESCALATION_OVERRIDE if _ESCALATION_OVERRIDE is not None
            else ESCALATED_ASCENT_EDGES)


def _escalated_home_edges() -> frozenset:
    """HOME-ascent edges eligible for escalated build caps.

    Operator-approved exceptions to the home-hot-path ban (bug 094): a
    listed edge escalates the PRIMARY evaluation whenever it is the
    world's home ascent — every mission from that home pays the bigger
    search, so entries are allowlist-bounded in the generator, never free
    probe output.  Today: Eve only, whose recalibrated ~9,000 m/s pad
    ascent exceeds the standard caps at the table's probe bar."""
    return (_HOME_ESCALATION_OVERRIDE if _HOME_ESCALATION_OVERRIDE is not None
            else ESCALATED_HOME_ASCENT_EDGES)


@dataclass(frozen=True)
class _ApolloSplit:
    """Resolved lander boundary + concrete docking gear for one profile.

    ``land_gidx``/``ascent_gidx`` are flight-order indices into the stage
    groups: the destination landing group and the surface-ascent group that
    follows it.  Everything after ``ascent_gidx`` is the parked return
    stack; everything before ``land_gidx`` hauls lander + parked stack
    outbound.  ``port`` is charged once per docked side; ``approach_gear``
    flies on the lander (the active vehicle in the docking approach): its
    torque source (standalone wheel unless the pod has built-in wheels)
    plus, below expert gameplay, the RCS translation kit.

    ``parked_gear`` is the parked stack's own control hardware: while the
    pod is away the parked stack is a pilotless craft, so it carries a
    probe core (command), an attitude source (wheel unless the core has
    one, else RCS), and a power source — real parts, charged on the first
    parked group's manifest and hauled outbound like the rest of the stack.
    """
    land_gidx: int
    ascent_gidx: int
    port: MiscEquipment
    approach_gear: AttitudeBundle
    parked_gear_mass: float
    parked_gear_parts: tuple[tuple[int, str], ...]


def _apollo_split_for(
    groups: list[list[MissionEdge]],
    home: BodyName,
    flags: EquipmentFlags,
    terminal_pod: Optional[MiscEquipment],
    is_crewed: bool,
    gameplay: GameplayDifficulty = CONSERVATIVE_GAMEPLAY,
) -> Optional[_ApolloSplit]:
    """Locate the Apollo lander boundary in ``groups``, or None when the
    profile isn't a parkable round trip or the docking gear is missing.

    Applicable iff the profile lands at a non-home body and ascends from it
    again with at least one post-ascent leg to park (the return transfer /
    home entry).  The gear gates are concrete parts: a docking port plus the
    docking attitude gear — torque authority ALWAYS (built-in pod wheels or a
    standalone reaction wheel), and below expert gameplay an RCS translation
    kit on top (``docking_needs_rcs``; an expert player can dock on
    main-engine translation, but wheels are never substitutable by RCS nor
    RCS by wheels).  Without the gear, the standard whole-stack evaluation
    (already attempted by the caller) is the only architecture.
    """
    if flags.lightest_docking_port is None:
        return None
    land_types = (EdgeType.VACUUM_LANDING, EdgeType.ATMO_LANDING)
    ascent_types = (EdgeType.ATMOSPHERIC_ASCENT, EdgeType.VACUUM_ASCENT)
    pair: Optional[tuple[int, int]] = None
    for i in range(len(groups) - 1):
        land_bodies = {e.body for e in groups[i]
                       if e.edge_type in land_types and e.body != home}
        if not land_bodies:
            continue
        if any(e.edge_type in ascent_types and e.body in land_bodies
               for e in groups[i + 1]):
            pair = (i, i + 1)  # keep the LAST qualifying pair
    if pair is None:
        return None
    land_gidx, ascent_gidx = pair
    # Need an outbound side to rejoin from and a return stack to park.
    if land_gidx == 0 or ascent_gidx + 1 >= len(groups):
        return None
    # Docking attitude gear on the lander (the active vehicle): wheels
    # always — the pod's built-in torque or a standalone module, charged —
    # and the RCS approach kit below expert gameplay.  MUST stay the same
    # helper the pod pick scores with (see _docking_approach_gear).
    approach_gear = _docking_approach_gear(flags, terminal_pod, gameplay)
    if approach_gear is None:
        return None
    # Parked-stack control gear.  While the pod is away the parked stack
    # needs a command source, an attitude source to hold orientation as the
    # dock target (wheel unless the command part provides one, else an RCS
    # kit), and its own power.  UNCREWED missions need a probe core — an
    # empty capsule is not commandable.  CREWED missions may instead leave
    # a pilot aboard a second capsule instance (the Apollo CM pattern), so
    # the command part is the lightest suitable one the kit has unlocked.
    # All concrete parts; without a command source (or any attitude source)
    # the split is not flyable and the standard architecture is the only
    # one.
    cmd_candidates = [p for p in (
        flags.lightest_probe,
        flags.lightest_capsule if is_crewed else None,
    ) if p is not None]
    if not cmd_candidates:
        return None
    command = min(cmd_candidates, key=lambda p: p.mass)
    parked_parts: list[tuple[int, str]] = [(1, command.name)]
    parked_mass = command.mass
    if not _pod_has_built_in_wheels(command):
        wheel = flags.lightest_reaction_wheel
        if wheel is not None:
            parked_parts.append((1, wheel.name))
            parked_mass += wheel.mass
        else:
            park_rcs = _rcs_bundle(flags, command)
            if park_rcs is None:
                return None
            parked_parts.extend(park_rcs.parts)
            parked_mass += park_rcs.mass
    # Lightest power source adequate for the profile's strictest per-leg
    # requirement (same selection the terminal support gear uses) — the
    # parked stack rides through the same aerobrake/solar-distance regime.
    power = _required_power_source(
        flags, [e for g in groups for e in g], home)
    if power is not None:
        parked_parts.append((1, power.name))
        parked_mass += power.mass
    return _ApolloSplit(land_gidx, ascent_gidx,
                        flags.lightest_docking_port, approach_gear,
                        parked_mass, tuple(parked_parts))


def _apollo_candidate(flags: EquipmentFlags, mission_type: MissionType) -> bool:
    """Cheap pre-gate for the Apollo retry: only round-trip mission types,
    and only kits that own a docking port (the common early-ladder kit has
    none, so the retry costs nothing there)."""
    return (flags.has_docking_port
            and mission_type in (MissionType.RETURN,
                                 MissionType.SAMPLE_RETURN))


def _required_power_source(
    flags: EquipmentFlags, profile: list[MissionEdge], home: BodyName,
) -> Optional[MiscEquipment]:
    """Lightest power source adequate for the strictest per-leg power requirement
    of ``profile``, or ``None`` when no leg needs power.

    The per-leg ``after_aero`` walk (fixed solar is destroyed by a non-recovery
    aero edge; the home parachute recovery is exempt because no leg flies after
    it) is the SAME determination the forward power gate
    (``_check_power_for_body`` via ``body_aero_destroyed``) makes — so the source
    whose mass is charged is exactly the one the feasibility verdict required.
    The gate runs first and blocks the profile when no adequate source exists,
    so an adequate source is guaranteed here whenever a requirement is present.

    Returning the *lightest* adequate source makes the charge MONOTONE in the
    kit: owning extra equipment (e.g. an RTG on top of fixed solar) can only
    lower the chosen mass, never raise it.  The previous code charged ``rtg``
    whenever ``needs_retractable`` was set and one was owned, so acquiring an RTG
    inflated the terminal payload and could push it past the parachute limit —
    a strictly larger kit losing a mission (the bug-092 non-monotonicity).
    """
    post_aero = False
    needs_rtg = needs_retractable = needs_solar = False
    for edge in profile:
        # Mirror the forward gate: a heat-shield edge that is NOT the final home
        # recovery destroys fixed solar for every later leg.
        if edge.needs_heat_shield and not edge.is_recovery:
            post_aero = True
        body = BODY_BY_NAME.get(edge.body)
        if body is None:
            continue
        req = body.power_requirement
        if req == "rtg":
            needs_rtg = True
        elif req in ("solar", "solar_marginal"):
            if post_aero:
                needs_retractable = True
            else:
                needs_solar = True
    # Strictest first: rtg-only ⊂ retractable-or-rtg ⊂ any-solar-or-rtg.
    if needs_rtg:
        return flags.lightest_rtg
    if needs_retractable:
        cands = [c for c in (flags.lightest_solar_retractable, flags.lightest_rtg)
                 if c is not None]
        return min(cands, key=lambda p: p.mass) if cands else None
    if needs_solar:
        cands = [c for c in (flags.lightest_solar, flags.lightest_rtg)
                 if c is not None]
        return min(cands, key=lambda p: p.mass) if cands else None
    return None


def _support_equipment_mass(
    flags: EquipmentFlags, profile: list[MissionEdge],
    home: BodyName, is_crewed: bool,
) -> tuple[float, list[tuple[int, str]]]:
    """
    Return (mass, parts) for required support equipment (antenna, power).
    Each part entry is (count, part_id).

    Power is the lightest source adequate for the strictest per-leg requirement
    (:func:`_required_power_source`) — the SAME requirement the forward power
    gate enforces, charged as the lightest adequate part so the charge can't
    exceed what a smaller kit pays (monotone) and can't diverge from the
    feasibility verdict.  Relay is the lightest antenna meeting the strictest
    tier across all edges — charged ONLY when the forward relay gate enforces
    it (uncrewed; a pilot needs no radio link, so crewed profiles carry no
    antenna).  Charging what the gate doesn't require broke monotonicity the
    same way the old power charge did: only the kit that OWNS the higher-tier
    antenna paid its mass, so acquiring one pushed the launch past the pad cap
    (a strictly larger kit losing a mission, bug-092 class; the gate blocks
    RELAY_TIER_TOO_LOW for uncrewed kits below tier, so the charge here is
    exactly the part the verdict required).
    """
    mass = 0.0
    parts: list[tuple[int, str]] = []

    # Relay: lightest antenna meeting the strictest tier across all edges,
    # mirroring the forward gate's crewed exemption.  Scan every OWNED tier
    # ≥ required (not a hardcoded range: the old ``range(max_relay, 4)``
    # excluded tier 4 — the highest real antenna — so a kit whose only
    # adequate antenna was the tier-4 dish charged NOTHING while a kit that
    # also owned a mid-tier antenna paid its mass: an under-charge on the
    # poorer kit AND a bigger-kit-pays-more non-monotonicity, bugs/103).
    max_relay = (max((edge.relay_tier for edge in profile), default=0)
                 if not is_crewed else 0)
    if max_relay > 0:
        best_relay: Optional[MiscEquipment] = None
        for tier, candidate in flags.lightest_relay.items():
            if tier < max_relay:
                continue
            if candidate and (best_relay is None or candidate.mass < best_relay.mass):
                best_relay = candidate
        if best_relay:
            mass += best_relay.mass
            parts.append((1, best_relay.name))

    # Power: lightest source adequate for the strictest per-leg requirement.
    src = _required_power_source(flags, profile, home)
    if src is not None:
        mass += src.mass
        parts.append((1, src.name))

    return mass, parts


def _terminal_equipment_mass(profile: list[MissionEdge],
                              flags: EquipmentFlags,
                              home: BodyName, is_crewed: bool) -> float:
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
    support_mass, _ = _support_equipment_mass(
        flags, profile, home=home, is_crewed=is_crewed)
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


def _radial_drag_multiplier(n: int) -> float:
    """Effective drag-area multiplier (in single-chute units) for ``n`` radial
    chutes.  KSP scales a radial chute group placed IN SYMMETRY super-linearly
    (``group^1.5`` — symmetric chutes are more efficient than independent ones),
    but symmetry tops out at ``_RADIAL_SYMMETRY_GROUP``; past that you add more
    rings, which stack linearly.  Closed form (no search): every full ring of
    ``g`` contributes ``g^1.5``, plus one partial ring of the remainder.

    This super-linear scaling is a non-obvious KSP engine quirk: the terminal
    velocity of a craft under ``n`` symmetric radial chutes is
    ``v = sqrt(m·B / n^α)`` with ``α = 1.5`` for radial-in-symmetry (``α = 1``
    for stack chutes or radial chutes placed independently).  Without it the
    model over-estimates radial chutes ~4×, spuriously banning real missions.
    Derivation + measured per-chute constants:
    https://forum.kerbalspaceprogram.com/topic/156287-boring-maths-on-parachutes-in-12/
    """
    g = _RADIAL_SYMMETRY_GROUP
    full, rem = divmod(n, g)
    return full * (g ** 1.5) + rem ** 1.5


# Chute-count search ceilings per role (attach-geometry bounds, see
# _MAX_RADIAL_CHUTES / _MAX_INLINE_CHUTES).  Drogues share the same geometry
# limits as mains of their kind.
_CHUTE_COUNT_CAPS: dict[bool, int] = {True: _MAX_RADIAL_CHUTES, False: _MAX_INLINE_CHUTES}

# Max upward probe steps when the closed-form terminal seed needs a burn: the
# seed can under-shoot the true passive boundary by a chute or two (piecewise
# radial multiplier + settle floor), so probe a bounded distance above it before
# concluding the craft is deploy-limited (no passive count exists).
_PASSIVE_PROBE_STEPS: int = 4

# Representative chemical Isp (s) for the landing-burn fuel proxy used to RANK
# burn mixes (the optimizer sizes the real stage).  Landing burns are
# TWR-limited chemical maneuvers; a fixed chemical value stops a high-Isp
# nuclear engine in the kit from making a full propulsive descent look cheap
# and out-ranking a chute-heavy low-burn mix.
_LANDING_PROXY_ISP: float = 300.0

# Effective drag coefficient for the command part's own body during descent,
# credited to the entry-bleed area alongside the (occluded) heat shield.  The
# shield occludes the pod during hypersonic entry, but by the subsonic
# chute-deploy regime the pod's blunt body adds real drag — without it a light
# capsule + small shield arrives too fast for its own low-q chutes to open, and
# the most basic Kerbin pod-on-one-Mk16 return would spuriously demand a burn.
# 0.5 is below any real pod cube (Mk1 ≈ 0.7); the 0.8 factor is the KSP
# drag-cube globals (Physics.cfg), matching the shield's effective-area units.
_POD_BLEED_CD: float = 0.5

# Max drag-device (inflatable) shields one descent stack may mount.  Heavy
# stacks radially mount several 10m inflatables — the operator's Eve
# calibration flight (2026-07-05) flew FIVE (one per core, 4+1) on a 729 t
# lander and its recorded speeds sit ~2.2x below what this model predicts
# even when all five are credited at their raw cube (the craft's uncredited
# core/tank body drag stays a conservative margin).  Each extra shield
# charges its real part mass, so the search stays monotone and honest.
_MAX_DRAG_SHIELDS: int = 5


@dataclass
class LandingMix:
    """Chosen staged-descent mix for one ATMO_LANDING edge.

    ``burn_dv`` is the propulsive shortfall (bridge + finish), already capped at
    the body's full propulsive-descent figure; the caller folds it into the
    stage ``req_dv`` (so difficulty ``percent_margin`` applies) and, when
    ``needs_burn``, runs the stage optimizer.  ``equipment`` are the chute parts
    to add to the manifest; ``shield`` is the (size, mass, name) coverage shield.
    """
    feasible: bool
    needs_burn: bool
    burn_dv: float
    equipment: list[tuple[int, str]]
    shield: Optional[tuple[float, float, str]]
    hardware_mass: float          # shields + chutes (passive-mix comparison key)
    residual_speed: float = 0.0   # touchdown m/s left unbraked when infeasible
    shield_count: int = 1         # copies of ``shield`` mounted (drag devices)


def _chute_role_stages(chute: Parachute, n: int, body: Body, label: str,
                       ground_altitude_m: float = 0.0):
    """aero.DragStage tuple for ``n`` copies of ``chute`` on ``body`` (radial
    groups scale super-linearly via ``_radial_drag_multiplier``; inline linear),
    or None if the chute can't open on this body.  Also returns the set mass."""
    mult = _radial_drag_multiplier(n) if chute.is_radial else float(n)
    stages = aero.chute_stages(
        full_area=chute.drag_area * mult,
        semi_area=chute.semi_drag_area * mult,
        q_safe_kpa=chute.q_safe_kpa,
        deploy_altitude_m=chute.deploy_altitude_m,
        min_pressure_atm=chute.min_pressure_atm,
        p0_kpa=body.atm_pressure_kpa,
        scale_height_m=body.atm_scale_height_m,
        label=label,
        ground_altitude_m=ground_altitude_m,
    )
    return stages, chute.mass * n


def _solve_atmo_landing(
    payload: float, body: Body, flags: EquipmentFlags, diff: DifficultyProfile,
    twr_floor: float, v_entry: float, dvGL_cap: float,
    coverage_shield: Optional[HeatShield], pod_size: float,
    ground_altitude_m: float = 0.0,
) -> LandingMix:
    """Pick the min-mass staged-descent mix for a single atmospheric landing.

    Enumerates {coverage shield, drag shield × count} × {mains, mains+drogues}
    × chute count, evaluates each with the closed-form ``aero.staged_descent``
    (entry bleed → chute ladder → touchdown), and picks:

    * the lightest PASSIVE mix (drag alone reaches ≤ safe touchdown) if any —
      no optimizer, matches the old passive aero path; else
    * the burn mix with the smallest (hardware + fuel-proxy) mass; the caller
      folds ``burn_dv`` into ``req_dv`` and runs the stage optimizer.

    ``coverage_shield`` is the pod-pair-picked shield from the pre-check — it is
    guaranteed to COVER the pod (a profile whose owned shields can't cover was
    already blocked ``HEAT_SHIELD_TOO_SMALL``, no undersized-fallback), so this
    never re-derives coverage.  ``ground_altitude_m`` is the landing-site
    elevation (highlands sites land in thinner air with less braking column).
    Returns ``feasible=False`` when no drag reaches safe touchdown AND no
    propulsive finish is available.
    """
    g = body.surface_gravity
    rho0 = body.atm_density_kg_m3
    H = body.atm_scale_height_m
    rho_site = aero.local_density(rho0, H, ground_altitude_m)

    coverage = coverage_shield
    if coverage is None:
        # No covering shield — the NO_HEAT_SHIELD / HEAT_SHIELD_TOO_SMALL gates
        # already fired; refuse defensively.
        return LandingMix(False, False, 0.0, [], None, 0.0, v_entry)
    # The command part's own subsonic drag, credited to every mix's bleed area.
    pod_bleed = 0.8 * _POD_BLEED_CD * math.pi * (max(pod_size, 1.25) / 2.0) ** 2

    # Shield candidates for the BLEED phase: (shield, count, bleed_area,
    # jettisoned, chute-phase shield mass).
    #
    # * The coverage shield rides in a pod stack, so its credit is the
    #   occluded cube (SHIELD_BLEED_OCCLUSION — calibrated on an in-game
    #   shield+pod CdA measurement); rigid and staged off before the chutes.
    # * The drag shield (the inflatable, when it out-drags coverage) is a NOSE
    #   device — the stack hides behind its 10m disk, nothing occludes it — so
    #   it credits its RAW cube, stays mounted through touchdown, and may be
    #   mounted up to ``_MAX_DRAG_SHIELDS`` times (each charging real part
    #   mass).  Operator's Eve calibration flight (5 shields, 729 t, passive
    #   landing) shows raw-cube crediting is still ~2x conservative.
    #
    # Owning the drag shield only ADDS candidates; it never displaces the
    # lighter coverage-only mix, so a heavy shield can't make a stage worse.
    shield_cands: list[tuple[HeatShield, int, float, bool]] = [
        (coverage, 1,
         aero.SHIELD_BLEED_OCCLUSION * coverage.drag_area + pod_bleed, True)]
    if (flags.best_drag_shield is not None
            and flags.best_drag_shield.drag_area > coverage.drag_area):
        drag_sh = flags.best_drag_shield
        for n_sh in range(1, _MAX_DRAG_SHIELDS + 1):
            shield_cands.append(
                (drag_sh, n_sh, drag_sh.drag_area * n_sh + pod_bleed, False))

    # Chute roles available (best of each kind, from _pre_pass).
    mains = [c for c in (flags.best_radial_main, flags.best_inline_main) if c]
    drogues = [c for c in (flags.best_radial_drogue, flags.best_inline_drogue) if c]

    # Representative exhaust velocity for the burn-mix fuel proxy (ranking ONLY;
    # the optimizer sizes the real stage).  A landing burn is a TWR-limited
    # maneuver done on a chemical engine — NOT the high-Isp nuclear/ion the kit
    # may also own — so a fixed chemical proxy keeps the ranking honest: it
    # correctly makes a large burn expensive (favouring chute-heavy, low-burn
    # mixes), which is both the physical mass-optimum and the feature's intent.
    ve = _LANDING_PROXY_ISP * 9.80665
    has_burn_capacity = (flags.has_throttleable_engine
                         and bool(flags.available_tanks or flags.available_srbs))

    best_passive: Optional[LandingMix] = None
    best_burn: Optional[LandingMix] = None
    best_burn_key = math.inf
    min_residual = v_entry  # track closest-to-feasible for the block reason

    def _consider(sh_cand, main, main_n, drogue, drogue_n) -> float:
        """Evaluate one mix, record it into best_passive/best_burn, and return
        its total landing burn (0.0 passive, capped burn, or inf infeasible) so
        the count search can binary-search on it."""
        nonlocal best_passive, best_burn, best_burn_key, min_residual
        shield, n_sh, bleed_area, jettisoned = sh_cand
        equip: list[tuple[int, str]] = []
        stages: list = []
        chute_mass = 0.0
        for chute, n in ((drogue, drogue_n), (main, main_n)):
            if chute is None or n <= 0:
                continue
            built, mass = _chute_role_stages(chute, n, body,
                                             "drogue" if chute.is_drogue else "main",
                                             ground_altitude_m)
            if built is None:
                return math.inf  # chute can't open on this body / above this site
            stages.extend(built)
            chute_mass += mass
            equip.append((n, chute.name))
        shield_mass = shield.mass * n_sh
        entry_mass = payload + shield_mass + chute_mass
        # A rigid ablative shield is jettisoned before the chutes deploy, so it
        # weighs down the bleed but NOT the terminal-velocity / touchdown calc
        # (matches the pre-rework model, which kept early Kerbin pod returns
        # passive).  The inflatable used as the bleed device stays on, so it is
        # not jettisoned.
        plan = aero.staged_descent(
            v_entry=v_entry, mass_t=entry_mass, bleed_area=bleed_area,
            stages=stages, rho0=rho0, scale_height_m=H, gravity=g,
            twr=twr_floor, max_safe_touchdown=_MAX_SAFE_LANDING_SPEED,
            jettison_mass_t=shield_mass if jettisoned else 0.0,
            ground_altitude_m=ground_altitude_m,
        )
        min_residual = min(min_residual, plan.touchdown_speed)
        shield_tuple = (shield.size_class, shield.mass, shield.name)
        hardware = shield_mass + chute_mass
        if not plan.requires_burn:
            mix = LandingMix(True, False, 0.0, equip, shield_tuple, hardware,
                             shield_count=n_sh)
            if best_passive is None or hardware < best_passive.hardware_mass:
                best_passive = mix
            return 0.0
        # Burn mix: needs a throttleable engine + fuel; infeasible otherwise.
        if not has_burn_capacity or math.isinf(plan.total_burn_dv):
            return math.inf
        burn = min(plan.total_burn_dv, dvGL_cap)
        fuel_proxy = entry_mass * (math.exp(burn / ve) - 1.0) if ve > 0 else math.inf
        key = hardware + fuel_proxy
        if key < best_burn_key:
            best_burn_key = key
            best_burn = LandingMix(True, True, burn, equip, shield_tuple,
                                   hardware, shield_count=n_sh)
        return burn

    def _seed_count(sh_cand, main, cap) -> int:
        """Closed-form lightest-passive main-count estimate: solve terminal
        velocity == safe for the continuous count (aero.terminal_limited_count),
        then round up.  Only a SEED — the caller confirms on the real staged
        model at the integer neighbours (the piecewise radial multiplier and the
        settle floor shift the true boundary by a chute or two)."""
        shield, n_sh, bleed, jettisoned = sh_cand
        # A rigid shield is jettisoned before the chutes carry the craft; the
        # inflatable-as-bleed-device stays on, so its mass rides the chute phase.
        m0 = payload + (0.0 if jettisoned else shield.mass * n_sh)
        n = aero.terminal_limited_count(
            payload_t=m0, chute_drag=main.drag_area, chute_mass_t=main.mass,
            bleed_area=bleed, rho0=rho_site, gravity=g,
            v_safe=_MAX_SAFE_LANDING_SPEED, is_radial=main.is_radial)
        if math.isinf(n):
            return cap
        return max(1, min(cap, math.ceil(n)))

    def _find_passive(sh_cand, main, drogue, drogue_n, cap) -> bool:
        """Find the lightest passive main-count by seeding from the closed-form
        terminal boundary and confirming on the real staged model.

        Passive is a MIDDLE interval (U-shaped burn: past the sweet spot the
        chutes' own mass slows the bleed until the craft can't deploy), and the
        lightest passive is its LOW edge.  So: evaluate at the analytic seed; if
        it's passive, step DOWN to the true minimum; if not, step UP a bounded
        amount (the seed can under-shoot by a chute or two).  ~3-5 evals, no full
        sweep.  Returns True if a passive count was found; every evaluated count
        is recorded into best_passive/best_burn for the burn ranker too."""
        seed = _seed_count(sh_cand, main, cap)
        if _consider(sh_cand, main, seed, drogue, drogue_n) <= 0.0:
            # Passive at the seed — walk down to the lightest still-passive count.
            n = seed
            while n > 1 and _consider(sh_cand, main, n - 1, drogue, drogue_n) <= 0.0:
                n -= 1
            return True
        # Seed needs a burn: either it under-shot the terminal boundary (step up
        # for more drag) or it is deploy-limited (more chutes only add mass —
        # stop).  Bounded upward probe; a plateau/worsening burn means no passive.
        prev = math.inf
        n = seed
        for _ in range(_PASSIVE_PROBE_STEPS):
            n += max(1, seed // 4)
            if n > cap:
                break
            burn = _consider(sh_cand, main, n, drogue, drogue_n)
            if burn <= 0.0:
                # Found passive above the seed; tighten down to the min.
                while n > 1 and _consider(sh_cand, main, n - 1, drogue, drogue_n) <= 0.0:
                    n -= 1
                return True
            if burn >= prev:
                break  # burn no longer improving with more chutes — deploy-limited
            prev = burn
        return False

    for sh_cand in shield_cands:
        for main in mains:
            cap = _CHUTE_COUNT_CAPS[main.is_radial]
            if not _find_passive(sh_cand, main, None, 0, cap):
                # Drogues bridge the high-speed gap so mains can land a thin-atmo
                # or heavy craft passively (or with a smaller burn).
                for drogue in drogues:
                    dcap = _CHUTE_COUNT_CAPS[drogue.is_radial]
                    for dn in (min(4, dcap), dcap):
                        _find_passive(sh_cand, main, drogue, dn, cap)
        # Drogues WITHOUT mains + a propulsive finish: the drogue-braked burn
        # landing (operator's Eve receipt: drogues alone reach ~200 m/s and
        # the engine finishes).  Matters when mains can't open at the site
        # (highlands above their gate) or their deploy-q needs a bigger
        # bridge than the drogue-only finish.
        for drogue in drogues:
            dcap = _CHUTE_COUNT_CAPS[drogue.is_radial]
            for dn in (min(4, dcap), dcap):
                _consider(sh_cand, None, 0, drogue, dn)
        # Shield + burn, no chutes (bleed + full propulsive finish).
        _consider(sh_cand, None, 0, None, 0)

    if best_passive is not None:
        return best_passive
    if best_burn is not None:
        return best_burn
    return LandingMix(False, False, 0.0, [], None, 0.0, min_residual)


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
    gameplay = mission_builder.gameplay  # world-carried skill/equipment gates
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
        # Single chokepoint for banned/unachievable missions (curated edge bans
        # ∪ dv-infeasible).  Routing it through capability means every
        # reachability consumer — location access rules, contract feasibility,
        # goal completion — inherits the ban without its own check.  Must be
        # BEFORE the empty-profiles branch (which means "trivially achievable").
        if not mission_builder.is_achievable(body.name, event.mission_type):
            prof.access[event.name] = False
            continue
        if event.crewed is True and not flags.has_capsule:
            prof.access[event.name] = False
            continue

        profiles = mission_builder.profiles_for(body.name, event.mission_type)

        # No profile alternatives at all (defensive: an achievable mission always
        # registers at least one, even the empty-edge home profile ``[[]]``).
        if not profiles:
            prof.access[event.name] = True
            continue

        # High-gravity sample return requires ladder for EVA re-boarding
        if event.mission_type == MissionType.SAMPLE_RETURN and body.eva_jetpack_twr < _MIN_EVA_JETPACK_TWR:
            profiles = _inject_ladder(profiles)

        ok, sub_blocking = _try_profiles_reason(
            profiles, flags, diff, event.mission_type,
            crewed=event.crewed, home=mission_builder.home,
            requires_eva=event.requires_eva, gameplay=gameplay,
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
    requires_eva: bool = False,
    gameplay: GameplayDifficulty = CONSERVATIVE_GAMEPLAY,
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
                                           run_parallel=run_par,
                                           requires_eva=requires_eva,
                                           gameplay=gameplay)
                if result.feasible:
                    return True
    # Apollo retry — failure path only, and only for round-trip missions on
    # kits that own a docking port (see _apollo_candidate).  The rejoin is a
    # rendezvous, so the buildings gate applies (requires_rendezvous).
    if _apollo_candidate(flags, mission_type):
        for run_par in (False, True):
            for is_crewed in _crewed_options(crewed, flags):
                for profile in profiles:
                    result = _evaluate_profile(profile, flags, diff, mission_type,
                                               is_crewed=is_crewed, home=home,
                                               extra_payload_parts=extra_payload_parts,
                                               run_parallel=run_par,
                                               requires_eva=requires_eva,
                                               requires_rendezvous=True,
                                               gameplay=gameplay,
                                               apollo_split=True)
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
    requires_eva: bool = False,
    gameplay: GameplayDifficulty = CONSERVATIVE_GAMEPLAY,
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
                                           run_parallel=run_par,
                                           requires_eva=requires_eva,
                                           gameplay=gameplay)
                if result.feasible:
                    return True, []
                for b in result.blocking:
                    key = str(b)
                    if key not in seen:
                        seen.add(key)
                        all_blocking.append(b)
    # Apollo retry — failure path only (see _try_profiles).  Blocking stays
    # the standard architecture's: those reasons drive the bumper's guidance
    # axes, and an Apollo near-miss adds no rankable signal beyond them.
    if _apollo_candidate(flags, mission_type):
        for run_par in (False, True):
            for is_crewed in _crewed_options(crewed, flags):
                for profile in profiles:
                    result = _evaluate_profile(profile, flags, diff, mission_type,
                                               is_crewed=is_crewed, home=home,
                                               extra_payload_parts=extra_payload_parts,
                                               run_parallel=run_par,
                                               requires_eva=requires_eva,
                                               requires_rendezvous=True,
                                               gameplay=gameplay,
                                               apollo_split=True)
                    if result.feasible:
                        return True, []
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
    requires_eva: bool | None = None,
    requires_rendezvous: bool | None = None,
    requires_samples: bool | None = None,
    requires_precise_pointing: bool = False,
    run_parallel: bool = True,
) -> ProfileResult:
    """
    Evaluate a specific mission and return the winning ProfileResult
    with full stage details + edge groups. Returns a non-feasible
    ProfileResult if no profile alternative succeeds.

    crewed: True = crewed only, False = unmanned only, None = try both.

    ``requires_eva`` overrides the EVA requirement (buildings_in_logic gate).
    ``None`` (the default) derives it from ``mission_type`` via
    ``MISSION_TYPES_REQUIRING_EVA`` — FLAG_PLANT/SAMPLE_RETURN always need EVA.
    EVA-in-orbit shares the ORBIT type, so its caller passes ``True`` here.
    When ``flags.can_eva`` is True (the default / option off) this has no
    effect.

    For ``mission_type="sounding"``, evaluates sounding rocket altitude
    against ``threshold_km``.  Other Kerbin-specific types (first_launch,
    first_landing, first_staging, splashdown) evaluate the relevant
    capability flags.
    """
    home = mission_builder.home_body
    gameplay = mission_builder.gameplay  # world-carried skill/equipment gates

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
                crewed=None, home=home, gameplay=gameplay,
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

    # requires_eva / requires_rendezvous / requires_samples pass straight through
    # to _evaluate_profile, which derives them from the mission type when None (an
    # override — e.g. EVA-in-orbit, docking contracts — wins).

    # ``run_parallel`` controls whether the exact asparagus (parallel-staged)
    # build is searched.  The bumper's GUIDANCE trials pass run_parallel=False:
    # serial mass is a fine ranking proxy (asparagus only makes a build lighter,
    # so the serial dv/mass ordering tracks the parallel one) and avoids the
    # 42-config parallel-stage search on the bumper's many infeasible trials —
    # the dominant cost.  The main-loop feasibility decision and the rescue keep
    # run_parallel=True so the committed kit (and asparagus-only missions) are
    # judged exactly.
    all_blocking: list[BlockingInfo] = []
    seen: set[str] = set()
    for is_crewed in _crewed_options(crewed, flags):
        for profile in profiles:
            result = _evaluate_profile(profile, flags, diff, mission_type,
                                       is_crewed=is_crewed,
                                       home=mission_builder.home,
                                       extra_payload_parts=extra_payload_parts,
                                       run_parallel=run_parallel,
                                       requires_eva=requires_eva,
                                       requires_rendezvous=requires_rendezvous,
                                       requires_samples=requires_samples,
                                       requires_precise_pointing=requires_precise_pointing,
                                       gameplay=gameplay)
            if result.feasible:
                return result
            for b in result.blocking:
                key = str(b)
                if key not in seen:
                    seen.add(key)
                    all_blocking.append(b)

    # Apollo retry — same failure-path-only order as the gating layer
    # (_try_profiles), so a mission gated feasible-via-Apollo reproduces
    # here with its real stage list (spoiler / post_fill cross-check).
    if _apollo_candidate(flags, mission_type):
        for is_crewed in _crewed_options(crewed, flags):
            for profile in profiles:
                result = _evaluate_profile(profile, flags, diff, mission_type,
                                           is_crewed=is_crewed,
                                           home=mission_builder.home,
                                           extra_payload_parts=extra_payload_parts,
                                           run_parallel=run_parallel,
                                           requires_eva=requires_eva,
                                           requires_rendezvous=True,
                                           requires_samples=requires_samples,
                                           requires_precise_pointing=(
                                               requires_precise_pointing),
                                           gameplay=gameplay,
                                           apollo_split=True)
                if result.feasible:
                    result.via_apollo = True
                    return result

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
    buildings_in_logic: bool = False,
    local_needs_conics: bool = True,
    local_needs_nodes: bool = True,
) -> tuple[RocketCapability, EquipmentFlags]:
    """Compute capability without a CollectionState. For CLI/external tools.

    ``contract_specs`` (a tuple of ContractSpec) makes this also compute
    per-contract feasibility into ``cap.contract_access``. Empty = no contracts.

    ``buildings_in_logic`` (default off) gates curated facility effects; OFF is
    a strict no-op (see ``_pre_pass``).
    """
    diff = DIFFICULTY_PROFILES[difficulty_name]
    flags = _pre_pass(item_count_fn, start_with_clamps,
                      progressive_launch_pad,
                      launch_pad_caps=mission_builder.launch_pad_caps,
                      buildings_in_logic=buildings_in_logic,
                      home=mission_builder.home,
                      local_needs_conics=local_needs_conics,
                      local_needs_nodes=local_needs_nodes)
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
        dsn_power=flags.dsn_power,
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
    difficulty_name = effective_physics_profile_name(options)
    start_with_clamps = bool(options.start_with_launch_clamps.value)
    cap, _ = compute_capability_from_items(
        lambda name: state.count(name, player),
        difficulty_name, start_with_clamps, world.mission_builder,
        progressive_launch_pad=bool(options.progressive_launch_pad.value),
        contract_specs=(*getattr(world, "contract_specs", ()),
                        *getattr(world, "goal_contract_specs", ())),
        buildings_in_logic=bool(options.buildings_in_logic.value),
        local_needs_conics=getattr(world, "local_needs_conics", True),
        local_needs_nodes=getattr(world, "local_needs_nodes", True),
    )
    return cap


def cheap_flags(state: CollectionState, player: int) -> EquipmentFlags:
    """The cheap half of capability: the equipment flags from the pre-pass,
    WITHOUT the per-body mission evaluation (the rocket optimizer).

    ``compute_capability_from_items`` assesses bodies lazily, so building the
    flags is cheap — only touching ``cap.bodies[...].access`` runs the
    optimizer.  Rules that read ONLY instrument / relay / capsule / power flags
    (KSC science, the science-budget instrument inputs) use this to stay off the
    expensive ``get_capability`` path during fill.
    """
    world = state.multiworld.worlds[player]
    options = world.options
    return _pre_pass(
        lambda name: state.count(name, player),
        bool(options.start_with_launch_clamps.value),
        bool(options.progressive_launch_pad.value),
        launch_pad_caps=world.mission_builder.launch_pad_caps,
        buildings_in_logic=bool(options.buildings_in_logic.value),
        home=world.mission_builder.home,
        local_needs_conics=getattr(world, "local_needs_conics", True),
        local_needs_nodes=getattr(world, "local_needs_nodes", True),
    )


# ---------------------------------------------------------------------------
# Import-time assertions — pytest catches violations automatically
# ---------------------------------------------------------------------------

# (a) Every MiscEquipment with multi_mount flag must be in MULTI_MOUNT_TABLE
for _item_name, _parts in DEFAULT_PART_MANAGER.parts.items():
    for _part in _parts:
        if isinstance(_part, MiscEquipment) and CapabilityFlag.MULTI_MOUNT in _part.provides:
            assert _part.name in MULTI_MOUNT_TABLE, (
                f"{_part.name} has multi_mount flag but not in MULTI_MOUNT_TABLE"
            )

# (b) Every MiscEquipment flag in parts.py must be a recognized CapabilityFlag.
_KNOWN_FLAGS: frozenset[str] = frozenset(CapabilityFlag)

for _item_name, _parts in DEFAULT_PART_MANAGER.parts.items():
    for _part in _parts:
        if isinstance(_part, MiscEquipment):
            _unknown = _part.provides - _KNOWN_FLAGS
            assert not _unknown, (
                f"Part {_part.name} has unrecognized flags: {_unknown}"
            )

del _item_name, _parts, _part, _unknown, _KNOWN_FLAGS

# Mission-profile coverage assertion now lives in MissionBuilder._validate
# (bodies.py); it runs at builder construction.
