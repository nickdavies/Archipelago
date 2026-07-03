"""Capability effects + the building->effects translation layer.

The capability system must NOT know about KSP facility buildings — it knows
*effects*.  A building level maps to a set of capability effects through the
translation in this module (``building_effects``).  The sphere layer inverts
the relation: a mission's required effect -> the minimal building level that
supplies it (``min_building_level_for``) -> a ``Counted(building, level)``
requirement.

Layering
--------
This is a low-level module.  It imports only ``.bodies`` (for the per-home
launch-pad cap table) and the standard library.  It must never import
``.capability``, ``.rules``, ``.world`` etc. — that would create a cycle.

Behavior-preservation
---------------------
Today the *only* effect consumed by the capability system is
``Effect.PAD_MASS_LIMIT`` (the launch-pad mass cap).  The launch-pad branch of
``building_effects`` reuses the existing per-home cap table verbatim, so the
value it produces is numerically identical to the old inline derivation in
``capability._pre_pass``.

All other effects/buildings are the *forward-design seam* for the curated
buildings work (buildings_in_logic).  They are defined and documented here but
are NOT yet consumed anywhere in the capability system.  Do not add dead
consumption of them — they are wired when ``buildings_in_logic`` lands.
"""

from enum import StrEnum

from .bodies import BodyName, progressive_launch_pad_caps_for
from .tech_tree import MAX_RD_BAND, max_reachable_node_cost


class Effect(StrEnum):
    """The curated capability-effect vocabulary.

    An *effect* is the only thing the capability system understands about a
    building.  The translation layer (``building_effects``) maps a concrete
    facility level onto a set of these.
    """

    # The only effect consumed by the capability system today.  Launch-pad
    # mass cap, in tonnes.  Produced from the per-home pad cap table.
    PAD_MASS_LIMIT = "pad_mass_limit"

    # --- Forward-design seam (NOT yet consumed by capability) --------------
    # These are wired when buildings_in_logic lands (curated-buildings task).
    # They exist so producers/consumers share a curated vocabulary rather than
    # ad-hoc strings; do NOT add dead consumption of them.

    # Astronaut Complex: whether EVA is permitted (bool).  Stock level 1
    # disallows EVA beyond a landed/low-Kerbin context; higher levels allow it.
    CAN_EVA = "can_eva"

    # Tracking Station: Deep Space Network comm/relay strength (numeric, in
    # the same "power" units KSP uses — see DSN_POWER_BY_LEVEL).
    DSN_POWER = "dsn_power"

    # VAB / SPH: buildable vessel mass limit, in tonnes.
    VESSEL_MASS_LIMIT = "vessel_mass_limit"

    # VAB / SPH: buildable vessel part-count limit (int).
    VESSEL_PART_LIMIT = "vessel_part_limit"

    # Astronaut Complex: maximum hireable crew rank / experience level (int).
    CREW_RANK = "crew_rank"


class Building(StrEnum):
    """The curated set of KSP facilities the translation layer covers."""

    LAUNCH_PAD = "launch_pad"
    VAB = "vab"
    SPH = "sph"
    TRACKING_STATION = "tracking_station"
    ASTRONAUT_COMPLEX = "astronaut_complex"
    MISSION_CONTROL = "mission_control"
    RESEARCH_AND_DEVELOPMENT = "research_and_development"


class Capability(StrEnum):
    """Player *abilities* the capability/rules system gates on.

    The capability system never names a building or a building level — it asks
    "can the player do X".  ``player_capabilities`` translates a set of facility
    levels (+ difficulty-resolved options) into these booleans; the sphere-ladder
    inverts via ``buildings_for_capability`` to decide which building items a
    mission requires.  (Comms/DSN is the one *quantitative* ability and stays a
    physical ``dsn_power``, not a boolean — see ``comms.py``.)
    """

    CAN_EVA = "can_eva"                                    # Astronaut Complex
    CAN_COLLECT_SAMPLES = "can_collect_samples"            # R&D facility (samples)
    CAN_RENDEZVOUS = "can_rendezvous"                      # conics + nodes
    CAN_NAVIGATE_LOCAL = "can_navigate_local"              # home-system transfers
    CAN_NAVIGATE_INTERPLANETARY = "can_navigate_interplanetary"


# Facility level (0-indexed = stock level - 1) at which each ability turns on.
# Stock: EVA-anywhere needs Astronaut Complex L2; patched conics needs Tracking
# Station L2; maneuver nodes need Mission Control L2 *and* conics.  Nodes without
# conics is impossible, so navigation/rendezvous need both TS and MC.
_EVA_AC_LEVEL = 1
_CONICS_TS_LEVEL = 1
_NODES_MC_LEVEL = 1

# R&D facility level gates the most expensive tech node you are PERMITTED to buy
# (stock GameVariables.GetScienceCostLimit: 100 at L0, 500 at L1, unlimited at L2 —
# independent of how much science you've banked).  The facility has no dedicated AP
# item; its level rides the Progressive R&D count (which also unlocks tech bands).
# We pick the Building->Progressive-R&D thresholds DELIBERATELY so the building cap
# is never the binding gate — the Progressive R&D band always is.  Building L1 by
# count 2 (tier-5 nodes exceed L0's 100 cap), L2 by count 4 (tier-7 nodes exceed
# L1's 500 cap).  ``RESEARCH_AND_DEVELOPMENT``'s "level" in a levels dict is the
# raw Progressive R&D count.
RD_SCIENCE_COST_LIMIT_BY_LEVEL: tuple[float, ...] = (100.0, 500.0, float("inf"))
RD_FACILITY_THRESHOLDS: tuple[int, ...] = (2, 4)

# Surface samples additionally need the R&D facility at ``_RD_SAMPLES_LEVEL`` (KSP
# "level 2", index 1), on TOP of the Astronaut-Complex EVA gate.  Samples unlock
# when the building reaches that level, so the count rides the schedule above.
_RD_SAMPLES_LEVEL = 1
RD_SAMPLES_COUNT: int = RD_FACILITY_THRESHOLDS[_RD_SAMPLES_LEVEL - 1]

# Guardrail (check only — never consulted on the hot path): the deliberate
# thresholds must keep the building cap ahead of every reachable node so the
# Progressive R&D band, not the building, is always the binding gate.  Fails loudly
# at import if a future tier/cost/band change breaks that invariant.
for _c in range(MAX_RD_BAND + 1):
    _lvl = sum(1 for _t in RD_FACILITY_THRESHOLDS if _c >= _t)
    assert RD_SCIENCE_COST_LIMIT_BY_LEVEL[_lvl] >= max_reachable_node_cost(_c), (
        f"R&D building cap binds before the band at Progressive R&D count {_c}; "
        f"raise RD_FACILITY_THRESHOLDS"
    )


# ---------------------------------------------------------------------------
# Stock-KSP facility-level constants (forward-design seam, task #7)
# ---------------------------------------------------------------------------
#
# These mirror stock KSP career-mode facility upgrade levels.  Each tuple is
# indexed by upgrade level (0 = starting facility, last = fully upgraded).
# They are consumed by the curated-buildings work, NOT by capability today.
#
# Sources: KSP wiki "Tracking Station", "Vehicle Assembly Building",
# "Astronaut Complex", and the stock GameVariables facility curves.  Values
# are the documented stock defaults.

# Tracking Station: Deep Space Network antenna power per level, in the same
# units as KSP antenna power.  Stock: 2 G, 50 G, 250 G.
DSN_POWER_BY_LEVEL: tuple[float, ...] = (2.0e9, 50.0e9, 250.0e9)

# VAB / SPH: buildable vessel mass limit (tonnes) per level.  Stock:
# 30 t, 140 t, unlimited.
VESSEL_MASS_LIMIT_BY_LEVEL: tuple[float, ...] = (30.0, 140.0, float("inf"))

# VAB / SPH: buildable vessel part-count limit per level.  Stock:
# 30, 255, unlimited.
VESSEL_PART_LIMIT_BY_LEVEL: tuple[int, ...] = (30, 255, 2_147_483_647)

# Astronaut Complex: whether EVA is permitted per level.  Stock: level 0
# allows EVA only when landed on / in low orbit of the home body; level 1
# allows EVA anywhere; level 2 allows EVA + construction.  We model the
# coarse "can EVA freely" gate.
CAN_EVA_BY_LEVEL: tuple[bool, ...] = (False, True, True)

# Astronaut Complex: maximum hireable crew experience rank per level.  Stock:
# level 0 caps recruits at rank/star 0 effectively, level 1 lifts it, level 2
# removes the cap (5).
CREW_RANK_BY_LEVEL: tuple[int, ...] = (0, 2, 5)


def _clamp_index(level: int, table) -> int:
    """Index ``table`` by ``level``, clamping into range (level is a count)."""
    if level < 0:
        level = 0
    return min(level, len(table) - 1)


def pad_mass_limit_from_caps(caps: tuple[float, ...], level: int) -> float:
    """Launch-pad mass cap for ``level`` given an already-resolved cap table.

    Thin helper so a caller that already holds the per-home cap table (e.g.
    ``capability._pre_pass`` via ``mission_builder.launch_pad_caps``) can route
    through the effects layer without re-threading ``home``.  It produces the
    *identical* value to ``building_effects(LAUNCH_PAD, level, home)[PAD_MASS_LIMIT]``
    when ``caps == progressive_launch_pad_caps_for(home)`` — this is the
    behavior-preservation contract for the existing pad-cap derivation.
    """
    return caps[_clamp_index(level, caps)]


def building_effects(building: Building, level: int, *, home: BodyName) -> dict[Effect, object]:
    """Translate a (building, level) into the set of capability effects it provides.

    ``home`` is the player's starting body; it scales the launch-pad cap table
    per-home.  ``level`` is a 0-based upgrade count (clamped into range).

    Only the ``LAUNCH_PAD`` branch is consumed by the capability system today;
    it reuses ``progressive_launch_pad_caps_for(home)`` so the produced
    ``PAD_MASS_LIMIT`` is identical to the old inline derivation.
    """
    if building is Building.LAUNCH_PAD:
        caps = progressive_launch_pad_caps_for(home)
        return {Effect.PAD_MASS_LIMIT: pad_mass_limit_from_caps(caps, level)}

    if building in (Building.VAB, Building.SPH):
        return {
            Effect.VESSEL_MASS_LIMIT: VESSEL_MASS_LIMIT_BY_LEVEL[_clamp_index(level, VESSEL_MASS_LIMIT_BY_LEVEL)],
            Effect.VESSEL_PART_LIMIT: VESSEL_PART_LIMIT_BY_LEVEL[_clamp_index(level, VESSEL_PART_LIMIT_BY_LEVEL)],
        }

    if building is Building.TRACKING_STATION:
        return {Effect.DSN_POWER: DSN_POWER_BY_LEVEL[_clamp_index(level, DSN_POWER_BY_LEVEL)]}

    if building is Building.ASTRONAUT_COMPLEX:
        return {
            Effect.CAN_EVA: CAN_EVA_BY_LEVEL[_clamp_index(level, CAN_EVA_BY_LEVEL)],
            Effect.CREW_RANK: CREW_RANK_BY_LEVEL[_clamp_index(level, CREW_RANK_BY_LEVEL)],
        }

    raise ValueError(f"unknown building: {building!r}")


# Map each building to the cap-table length that defines its max upgrade level.
# Launch-pad length is per-home so it's resolved inside ``max_effects``.
_NON_PAD_MAX_LEVEL: dict[Building, int] = {
    Building.VAB: len(VESSEL_MASS_LIMIT_BY_LEVEL) - 1,
    Building.SPH: len(VESSEL_MASS_LIMIT_BY_LEVEL) - 1,
    Building.TRACKING_STATION: len(DSN_POWER_BY_LEVEL) - 1,
    Building.ASTRONAUT_COMPLEX: len(CAN_EVA_BY_LEVEL) - 1,
}


def max_effects(home: BodyName) -> dict[Effect, object]:
    """Every curated building fully upgraded — the "career hack" defaults.

    This is the effect set used when buildings are NOT in logic: all facilities
    are maxed, so no effect gates anything.  ``PAD_MASS_LIMIT`` is the top
    (infinite) pad cap; the other effects are each building's top level.
    """
    caps = progressive_launch_pad_caps_for(home)
    effects: dict[Effect, object] = dict(
        building_effects(Building.LAUNCH_PAD, len(caps) - 1, home=home)
    )
    for building, max_level in _NON_PAD_MAX_LEVEL.items():
        effects.update(building_effects(building, max_level, home=home))
    return effects


# Which building supplies each effect, and the per-level table used to invert
# the threshold.  Higher level == higher value for every modelled effect.
_EFFECT_PROVIDER: dict[Effect, tuple[Building, object]] = {
    Effect.VESSEL_MASS_LIMIT: (Building.VAB, VESSEL_MASS_LIMIT_BY_LEVEL),
    Effect.VESSEL_PART_LIMIT: (Building.VAB, VESSEL_PART_LIMIT_BY_LEVEL),
    Effect.DSN_POWER: (Building.TRACKING_STATION, DSN_POWER_BY_LEVEL),
    Effect.CAN_EVA: (Building.ASTRONAUT_COMPLEX, CAN_EVA_BY_LEVEL),
    Effect.CREW_RANK: (Building.ASTRONAUT_COMPLEX, CREW_RANK_BY_LEVEL),
}


def min_building_level_for(effect: Effect, value, *, home: BodyName) -> tuple[Building, int]:
    """Inverse seam: cheapest (building, level) that provides ``effect`` >= ``value``.

    Given a required effect threshold, return the lowest-cost building and
    upgrade level whose effect value meets or exceeds ``value``.  If no level
    reaches the threshold, the maximum level is returned (the building can't do
    better — the caller decides whether that's feasible).

    For ``CAN_EVA`` (a bool) the threshold is "truthiness": a falsy ``value``
    is satisfied by level 0; a truthy ``value`` needs the first level that
    enables EVA.
    """
    if effect is Effect.PAD_MASS_LIMIT:
        caps = progressive_launch_pad_caps_for(home)
        for level, cap in enumerate(caps):
            if cap >= value:
                return (Building.LAUNCH_PAD, level)
        return (Building.LAUNCH_PAD, len(caps) - 1)

    provider = _EFFECT_PROVIDER.get(effect)
    if provider is None:
        raise ValueError(f"no building provides effect {effect!r}")
    building, table = provider
    for level, level_value in enumerate(table):
        if level_value >= value:
            return (building, level)
    return (building, len(table) - 1)


# ---------------------------------------------------------------------------
# Capability layer: facility levels (+ resolved options) -> player abilities
# ---------------------------------------------------------------------------

def player_capabilities(
    levels: dict[Building, int], *,
    local_needs_conics: bool, local_needs_nodes: bool,
) -> dict[Capability, bool]:
    """Translate a set of facility levels into the ability booleans the
    capability system gates on.

    Buildings absent from ``levels`` default to their turn-on threshold (i.e.
    ungated — the ability is present), so this is a strict no-op for facilities
    that ship maxed.  ``local_needs_conics`` / ``local_needs_nodes`` are the
    difficulty-and-option-resolved requirements for home-system (moon) transfers;
    rendezvous and interplanetary transfers always need both conics and nodes.
    """
    ac = levels.get(Building.ASTRONAUT_COMPLEX, _EVA_AC_LEVEL)
    ts = levels.get(Building.TRACKING_STATION, _CONICS_TS_LEVEL)
    mc = levels.get(Building.MISSION_CONTROL, _NODES_MC_LEVEL)
    rd = levels.get(Building.RESEARCH_AND_DEVELOPMENT, RD_SAMPLES_COUNT)
    conics = ts >= _CONICS_TS_LEVEL
    nodes = conics and mc >= _NODES_MC_LEVEL
    can_local = ((not local_needs_conics or conics)
                 and (not local_needs_nodes or nodes))
    return {
        Capability.CAN_EVA: CAN_EVA_BY_LEVEL[_clamp_index(ac, CAN_EVA_BY_LEVEL)],
        Capability.CAN_COLLECT_SAMPLES: rd >= RD_SAMPLES_COUNT,
        Capability.CAN_RENDEZVOUS: nodes,
        Capability.CAN_NAVIGATE_INTERPLANETARY: nodes,
        Capability.CAN_NAVIGATE_LOCAL: can_local,
    }


def buildings_for_capability(
    cap: Capability, *, local_needs_conics: bool, local_needs_nodes: bool,
) -> tuple[tuple[Building, int], ...]:
    """Inverse of ``player_capabilities`` for one ability: the minimal
    ``(building, level)`` requirements that provide it.  Multi-building abilities
    (rendezvous / navigation need conics AND nodes) return several; a
    ``CAN_NAVIGATE_LOCAL`` that the options leave ungated returns ``()``.
    """
    conics_req = (Building.TRACKING_STATION, _CONICS_TS_LEVEL)
    nodes_req = (Building.MISSION_CONTROL, _NODES_MC_LEVEL)
    if cap is Capability.CAN_EVA:
        return ((Building.ASTRONAUT_COMPLEX, _EVA_AC_LEVEL),)
    if cap is Capability.CAN_COLLECT_SAMPLES:
        # The "level" is the Progressive R&D count threshold (no separate item).
        return ((Building.RESEARCH_AND_DEVELOPMENT, RD_SAMPLES_COUNT),)
    if cap in (Capability.CAN_RENDEZVOUS, Capability.CAN_NAVIGATE_INTERPLANETARY):
        return (conics_req, nodes_req)          # nodes imply conics
    if cap is Capability.CAN_NAVIGATE_LOCAL:
        reqs: list[tuple[Building, int]] = []
        if local_needs_nodes:
            reqs += [conics_req, nodes_req]     # nodes imply conics
        elif local_needs_conics:
            reqs.append(conics_req)
        return tuple(reqs)
    raise ValueError(f"no building mapping for capability {cap!r}")
