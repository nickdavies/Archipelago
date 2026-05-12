"""
Structured reasons explaining why a mission profile is infeasible.

Capability evaluation produces a list of ``BlockingInfo`` objects when a
profile fails.  ``ProfileResult.failure_reasons`` and
``BodyAccessProfile.blocking_reason`` are now back-compat shims that
render these objects to strings via ``BlockingInfo.__str__``.

Downstream consumers (sphere ladder pre-fill in particular) match on
``BlockingReason`` enum values directly instead of parsing strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockingReason(str, Enum):
    """Distinct categories of why a profile evaluation failed."""

    # --- Propulsion / staging -------------------------------------------
    NO_VIABLE_STAGE = "no_viable_stage"                # find_optimal_stage returned None
    NO_ENGINE = "no_engine"                            # no engine available for an edge
    NO_LAUNCH_ENGINE = "no_launch_engine"              # no atmo-ascent engine
    NO_FUEL = "no_fuel"                                # no tanks available
    NO_PROPULSION = "no_propulsion"                    # combined sounding-rocket case
    STAGING_TIER_INSUFFICIENT = "staging_tier_insufficient"

    # --- Command --------------------------------------------------------
    NO_CAPSULE = "no_capsule"
    NO_PROBE_CORE = "no_probe_core"
    NO_COMMAND_MODULE = "no_command_module"            # either would do

    # --- Control --------------------------------------------------------
    NO_ATTITUDE_CONTROL = "no_attitude_control"

    # --- Landing / safe descent -----------------------------------------
    LANDING_LEGS_MISSING = "landing_legs_missing"
    NO_LADDER = "no_ladder"
    NO_HEAT_SHIELD = "no_heat_shield"
    NO_PARACHUTE = "no_parachute"
    PARACHUTE_TERMINAL_VELOCITY = "parachute_terminal_velocity"
    NO_SAFE_DESCENT = "no_safe_descent"                # chute OR throttleable
    CAPSULE_SOUNDING_INCOMPLETE = "capsule_sounding_incomplete"  # missing chute or decoupler

    # --- Power / comms --------------------------------------------------
    INSUFFICIENT_POWER_SOLAR_OK = "insufficient_power_solar_ok"
    INSUFFICIENT_POWER_NEEDS_RTG = "insufficient_power_needs_rtg"
    RELAY_TIER_TOO_LOW = "relay_tier_too_low"

    # --- Mass / pad -----------------------------------------------------
    LAUNCH_MASS_EXCEEDED = "launch_mass_exceeded"

    # --- Sounding / altitude --------------------------------------------
    SOUNDING_ALTITUDE_TOO_LOW = "sounding_altitude_too_low"
    NO_SOUNDING_ALTITUDE = "no_sounding_altitude"

    # --- Mission graph / infrastructure ---------------------------------
    PARENT_BODY_UNREACHABLE = "parent_body_unreachable"
    NO_LAUNCH_CLAMP = "no_launch_clamp"
    NO_PROFILES = "no_profiles"
    ORBIT_NOT_ACHIEVABLE = "orbit_not_achievable"

    # --- Composite event-level reason (kept for back-compat output) -----
    EVENT_COMPOUND = "event_compound"


@dataclass(frozen=True)
class BlockingInfo:
    """Structured detail of a single profile-evaluation failure.

    Producers fill the fields relevant to their ``reason``; readers
    match on ``reason`` first and inspect the typed fields.  The
    ``__str__`` reproduces the human-readable string the codebase
    used to produce directly, so existing log output and test asserts
    remain stable.
    """

    reason: BlockingReason
    body: str = ""
    parent_body: str = ""        # for PARENT_BODY_UNREACHABLE
    edge_type: str = ""          # name of EdgeType enum value
    mission_type: str = ""       # for EVENT_COMPOUND
    dv_needed: float = 0.0
    dv_available: float = 0.0
    altitude_km: float = 0.0
    threshold_km: float = 0.0
    relay_needed: int = 0
    relay_available: int = 0
    leg_tier_needed: int = 0
    leg_tier_available: int = 0
    after_aero: bool = False
    mass_actual: float = 0.0     # tonnes
    mass_cap: float = 0.0        # tonnes
    stages_needed: int = 0
    stages_available: int = 0
    solar_helps: bool = True     # only meaningful for INSUFFICIENT_POWER_*
    # Sub-strings or sub-reasons; used by composite reasons to preserve
    # exact human-readable output without forcing every variant into the
    # enum.  Free-form; consumers should prefer typed fields.
    detail: str = ""

    def __str__(self) -> str:
        r = self.reason
        # Propulsion ----------------------------------------------------
        if r == BlockingReason.NO_VIABLE_STAGE:
            return (f"no viable stage for group dv={self.dv_needed:.0f} m/s "
                    f"at {self.body}")
        if r == BlockingReason.NO_ENGINE:
            return f"no engine for {self.edge_type}"
        if r == BlockingReason.NO_LAUNCH_ENGINE:
            return f"no launch engine for {self.edge_type}"
        if r == BlockingReason.NO_FUEL:
            if self.edge_type:
                return f"no fuel for {self.edge_type}"
            return "engines but no fuel tanks"
        if r == BlockingReason.NO_PROPULSION:
            if self.detail:
                return f"no sounding altitude ({self.detail})"
            return "no propulsion (need SRB or engine + fuel)"
        if r == BlockingReason.STAGING_TIER_INSUFFICIENT:
            if self.stages_needed > 0:
                return (f"need {self.stages_needed} stages but "
                        f"staging_tier={self.stages_available} only allows "
                        f"{max(1, self.stages_available)}")
            return "no stack decoupler (staging_tier < 1)"
        # Command -------------------------------------------------------
        if r == BlockingReason.NO_CAPSULE:
            if self.detail:
                return f"no capsule ({self.detail})"
            return "no capsule for crewed mission"
        if r == BlockingReason.NO_PROBE_CORE:
            return "no probe core for unmanned mission"
        if r == BlockingReason.NO_COMMAND_MODULE:
            return "no command module (need probe core or capsule)"
        # Control / landing --------------------------------------------
        if r == BlockingReason.NO_ATTITUDE_CONTROL:
            return "no attitude control"
        if r == BlockingReason.LANDING_LEGS_MISSING:
            return (f"need leg tier {self.leg_tier_needed}, "
                    f"have {self.leg_tier_available}")
        if r == BlockingReason.NO_LADDER:
            return "need ladder for sample return"
        if r == BlockingReason.NO_HEAT_SHIELD:
            return "no heat shield for aero edge"
        if r == BlockingReason.NO_PARACHUTE:
            return "no parachutes for aero landing"
        if r == BlockingReason.PARACHUTE_TERMINAL_VELOCITY:
            return f"parachute terminal velocity check failed at {self.body}"
        if r == BlockingReason.NO_SAFE_DESCENT:
            return "no safe descent (need parachute or throttleable engine)"
        if r == BlockingReason.CAPSULE_SOUNDING_INCOMPLETE:
            return f"capsule-only sounding needs: {self.detail}"
        # Power / comms ------------------------------------------------
        if r in (BlockingReason.INSUFFICIENT_POWER_SOLAR_OK,
                 BlockingReason.INSUFFICIENT_POWER_NEEDS_RTG):
            return (f"insufficient power at {self.body} "
                    f"(after_aero={self.after_aero})")
        if r == BlockingReason.RELAY_TIER_TOO_LOW:
            return (f"relay tier too low for {self.body}: "
                    f"need {self.relay_needed}, have {self.relay_available}")
        # Mass ---------------------------------------------------------
        if r == BlockingReason.LAUNCH_MASS_EXCEEDED:
            return (f"launch mass {self.mass_actual:.0f}t exceeds launch pad "
                    f"cap {self.mass_cap:.0f}t")
        # Sounding -----------------------------------------------------
        if r == BlockingReason.SOUNDING_ALTITUDE_TOO_LOW:
            base = (f"sounding altitude {self.altitude_km:.1f} km < "
                    f"{self.threshold_km:.0f} km")
            if self.detail:
                return f"{base} ({self.detail})"
            return base
        if r == BlockingReason.NO_SOUNDING_ALTITUDE:
            return "no sounding altitude"
        # Mission graph ------------------------------------------------
        if r == BlockingReason.PARENT_BODY_UNREACHABLE:
            return f"parent {self.parent_body} orbit unreachable"
        if r == BlockingReason.NO_LAUNCH_CLAMP:
            return "no launch clamp for interplanetary mission"
        if r == BlockingReason.NO_PROFILES:
            return f"no profiles for ({self.body}, {self.mission_type})"
        if r == BlockingReason.ORBIT_NOT_ACHIEVABLE:
            return "orbit not achievable"
        # Composite ----------------------------------------------------
        if r == BlockingReason.EVENT_COMPOUND:
            return f"{self.mission_type}: {self.detail}"
        return r.value  # fallback


__all__ = ["BlockingReason", "BlockingInfo"]
