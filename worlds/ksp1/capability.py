"""
Rocket capability evaluation system.

This module owns the RocketCapability dataclass (cached on CollectionState),
the backpropagation algorithm that computes it, and the delta-V graph data.

See ksp_archipelago_design.md for the full design specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from BaseClasses import CollectionState

if TYPE_CHECKING:
    from .world import KSP1World

# Key used to store the capability cache on a CollectionState.
_CACHE_KEY = "ksp1_capability"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    delta_v: float
    twr_at_ignition: float
    twr_at_burnout: float
    engine_is_throttleable: bool
    engine_has_gimbal: bool


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
    # Equipment flags
    has_heat_shield: bool = False
    has_parachutes: bool = False
    landing_leg_tier: int = 0
    has_reaction_wheels: bool = False
    has_rcs: bool = False
    has_gimbal: bool = False
    has_throttle_control: bool = False
    has_probe_core: bool = False
    has_capsule: bool = False
    has_rtg: bool = False
    has_isru: bool = False
    has_docking_port: bool = False
    relay_tier: int = 0
    power_profile: str = "solar"
    staging_tier: int = 0

    # Delta-V summary
    max_dv_total: float = 0.0
    max_payload_to_lko: float = 0.0

    # Per-body assessments
    bodies: dict[str, BodyAccessProfile] = field(default_factory=dict)

    # Stage detail (for debugging)
    stage_results: list[StageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache accessor
# ---------------------------------------------------------------------------

def get_capability(state: CollectionState, player: int) -> RocketCapability:
    """
    Return the cached RocketCapability for this state, computing it if cold.

    This is the only entry point that access rules should call.  All expensive
    work happens once per state copy inside _compute_capability.
    """
    cache: dict[int, RocketCapability] = state.prog_items.setdefault(_CACHE_KEY, {})
    if player not in cache:
        cache[player] = _compute_capability(state, player)
    return cache[player]


# ---------------------------------------------------------------------------
# Computation pipeline
# ---------------------------------------------------------------------------

def _compute_capability(state: CollectionState, player: int) -> RocketCapability:
    """
    Full capability computation from the current collection state.

    Pipeline (per design doc):
      1. Pre-pass  — collect equipment flags from unlocked items.
      2. Backprop  — compute staged delta-V for all mission profiles.
      3. Per-body  — assess each body given the computed stages + flags.
      4. Assemble  — populate and return RocketCapability.
    """
    cap = RocketCapability()

    # TODO: Step 1 — pre-pass over state items to populate equipment flags.

    # TODO: Step 2 — backpropagation algorithm.

    # TODO: Step 3 — per-body assessment using dv_graph data.

    return cap
