#!/usr/bin/env python3
"""Generate the static FEASIBILITY tables consumed by the world rules.

One table is produced PER DIFFICULTY (feasibility depends on the dv
margin, which differs per difficulty), each at a uniform rep-selection
overhead.  For every body that can serve as a starting body, the script:

1. Builds an "everything maxed" item-count function — one of every
   individual part in ``PART_DB`` plus every progressive item at its
   max tier — and adds a small ``percent_margin`` overhead on top as a
   rep-selection safety buffer (a real seed gets one rep per category,
   possibly worse than the best part).
2. Runs ``compute_capability_from_items`` against a ``MissionBuilder``
   rooted at that home, at each difficulty profile.
3. Records EVERY non-home body event reported as unreachable (False
   access) — all events ``locations.py`` exposes for the body, not just
   returns.  A body harder than the maximal kit's ceiling (even for
   plain orbit) therefore excludes gracefully instead of leaving
   progression-eligible locations behind a permanently-false rule.
   These locations get the "all-parts collected" proxy rule at
   goal-evaluation time instead of the dv-based capability check.

The output is checked in at ``worlds/ksp1/data/feasibility.py`` and
read by ``world.py`` at generation time as a pure lookup — no per-seed
recomputation.  That keeps the banned-location set stable per commit
hash: if the dv model or graph changes, this script must be re-run and
the table re-committed.

Usage
-----

Dump the regenerated table to stdout::

    cd Archipelago
    ../.venv/bin/python -m worlds.ksp1.scripts.generate_feasibility

Write the regenerated table to the checked-in path::

    ../.venv/bin/python -m worlds.ksp1.scripts.generate_feasibility --write

Verify the checked-in table matches what this script would produce
(non-zero exit on drift — wire into CI)::

    ../.venv/bin/python -m worlds.ksp1.scripts.generate_feasibility --check
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
from pathlib import Path

from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, DIFFICULTY_PROFILES,
    EdgeType, MissionBuilder, MissionType,
)
import worlds.ksp1.capability as capability
from worlds.ksp1.capability import (
    compute_capability_from_items, evaluate_mission_detailed,
)
from worlds.ksp1.locations import (
    EVENT_BY_NAME, get_body_events,
)
from worlds.ksp1.parts import ALL_PACKS, part_manager_for
from worlds.ksp1.parts.packs import (
    STOCK, OPTIONAL_PACKS, DEFAULT_ENABLED_OPTIONAL_PACKS,
)


# One table is baked per PHYSICS difficulty.  Feasibility genuinely depends on
# the dv margin profile (generous demands the most cushion, zero none at all),
# so a mission can be flyable at 'small' yet infeasible at 'generous'.  A single
# margin-agnostic table can't express that, so the world reads the table
# matching the seed's resolved physics profile
# (``effective_physics_profile_name``).  Keys are ``DIFFICULTY_PROFILES`` keys.
DIFFICULTIES: tuple[str, ...] = ("generous", "comfortable", "small", "zero")

# Extra ``percent_margin`` added on top of EACH difficulty profile.  This is
# a REP-SELECTION safety buffer, not a difficulty knob: the probe runs with
# the max kit (one of every part), but a real seed gets ONE representative
# per category that may be heavier or weaker, so a mission that *just barely*
# closes with the best parts could fail with the reps actually granted.  The
# buffer keeps such marginal missions classified infeasible.  Uniform across
# difficulties because rep variance is difficulty-independent.
#
# Lowered 0.25 -> 0.10 (2026-07-06, operator decision): the 9km/s-class Eve
# ascents sit just past a Tsiolkovsky cliff at +25% (the escalated Eve-home
# mesa lifter's payload ceiling collapses from full stacks at 1.25x effective
# dv to 1.8t at 1.30x), so the old buffer was excluding whole mission classes
# (Eve-home at small: 169 location names) that close comfortably at the real
# seed margins.  At 0.10 nothing is newly excluded anywhere and the measured
# solve rate holds the zero-reject bar: expert+AllowEve {SSR, flag, ctt} x
# {kerbin, laythe, tylo} 9/9 configs x 40/40 strict CLEAN, casual tylo SSR
# 40/40 CLEAN, Tier 2 rows byte-identical (see solve_rate_log 2026-07-06).
# The historic fill-famine tail the 25% figure was sized against (expert deep
# SSR) predates the location-signature and HomeContractFloor fixes that now
# carry that load.
DEFAULT_OVERHEAD: float = 0.10

# Default physics difficulty for the single-home helper (the per-home probe
# still takes an explicit difficulty; this only covers callers that omit it).
DEFAULT_DIFFICULTY: str = "comfortable"


def _max_kit_counts(enabled_packs: frozenset[str]) -> dict[str, int]:
    """One of every individual part available under ``enabled_packs``.

    With parts de-progressivized, every concrete part is its own item, so the
    most permissive kit is one of each — strictly more permissive than any
    single seed's selection under the same packs, making the bodies this flags
    infeasible the floor for that pack configuration.
    """
    return {name: 1 for name in part_manager_for(enabled_packs).parts}


def _profile_with_overhead(base_name: str, overhead: float) -> str:
    """Register (and return the name of) a ``DifficultyProfile`` that's
    ``base_name`` with an extra ``overhead`` added to ``percent_margin``.

    Returns ``base_name`` itself when ``overhead == 0`` so we don't
    pollute the registry with no-op clones.
    """
    if overhead == 0.0:
        return base_name
    base = DIFFICULTY_PROFILES[base_name]
    name = f"_probe_{base_name}_p{int(round(overhead * 100))}"
    # replace() copies every field and overrides only the margin, so adding a
    # field to DifficultyProfile can't silently drop it from the probe profile.
    DIFFICULTY_PROFILES[name] = dataclasses.replace(
        base, percent_margin=base.percent_margin + overhead)
    return name


def compute_model_infeasible_for_home(
    home: BodyName,
    difficulty_name: str = DEFAULT_DIFFICULTY,
    overhead: float = DEFAULT_OVERHEAD,
    enabled_packs: frozenset[str] = ALL_PACKS,
) -> frozenset[tuple[BodyName, MissionType]]:
    """``(target_body, mission_type)`` pairs whose mission the dv model can't
    verify from ``home``, even given a maxed-out parts kit.  Used to drive the
    all-parts proxy rule for those missions' completion checks.

    Canonical as ``(body, mission_type)`` — the exact shape the world's
    ``unachievable_missions`` set carries — so the world reads the table with no
    per-name parsing.  A single pair covers every AP slot of every event sharing
    that mission type; the world expands it back over ``MISSION_LOCATIONS``.

    ``overhead`` adds extra ``percent_margin`` on top of the chosen
    difficulty profile — tightens the "what counts as feasible" bar
    when you need missions that *just barely* pass with maxed reps
    to still be classified as model-infeasible.

    Probing is per-event, but ``world.py`` canonicalises the table's
    location names back to ``(body, mission_type)`` tuples — and several
    events SHARE a mission_type (Landing/Crewed Landing → LAND,
    Orbit/EVA in Orbit → ORBIT, Flyby/SOI Leave → ESCAPE).  One
    infeasible event therefore excludes its mission_type siblings on
    that body too.  That collapse is deliberate and must stay in the
    conservative direction: the alternative (excluding only the failing
    sibling) would leave the feasible sibling's locations progression-
    eligible while the infeasible one strands.
    """
    counts = _max_kit_counts(enabled_packs)
    mission_builder = MissionBuilder(home=home)
    probe_difficulty = _profile_with_overhead(difficulty_name, overhead)
    cap, _flags = compute_capability_from_items(
        lambda name: counts.get(name, 0),
        difficulty_name=probe_difficulty,
        start_with_clamps=True,
        mission_builder=mission_builder,
    )
    infeasible: set[tuple[BodyName, MissionType]] = set()
    for body in ALL_BODIES:
        if body.name == home:
            continue
        # Probe exactly the events locations.py exposes for this body —
        # Jool contributes its orbital events, Kerbol contributes none.
        events = get_body_events(body)
        if not events:
            continue
        body_cap = cap.bodies[body.name]
        for event in events:
            if body_cap.access.get(event, False):
                continue
            infeasible.add((body.name, EVENT_BY_NAME[event].mission_type))
    return frozenset(infeasible)


def _candidate_homes() -> list[BodyName]:
    """Bodies that can serve as a starting body.  Must have a surface
    (``can_land``) and be inside the patched-conic graph (Kerbol is
    excluded — it's the root star, not a launch site).
    """
    return [b.name for b in ALL_BODIES
            if b.can_land and b.name != BodyName.KERBOL]


# ---------------------------------------------------------------------------
# Escalated-build eligibility — which ascent edges may use the ESCALATED_*
# caps (rocket_math), applied by capability ONLY inside Apollo-split
# evaluations.  Probed offline HERE so generation never pays a
# try-fail-retry: the checked-in set is the single source the escalation
# predicate reads.  Home ascents are NEVER candidates (operator constraint:
# a home ascent is that seed's hottest search path).
# ---------------------------------------------------------------------------

_ASCENT_EDGE_TYPES = (EdgeType.ATMOSPHERIC_ASCENT, EdgeType.VACUUM_ASCENT)


def compute_escalated_edges(
    overhead: float = DEFAULT_OVERHEAD,
    enabled_packs: frozenset[str] = ALL_PACKS,
) -> frozenset[tuple[BodyName, EdgeType]]:
    """(body, EdgeType) ascent edges whose escalation flips some max-kit
    mission from infeasible to feasible, at any (difficulty, home).

    Two passes per (difficulty, home), both with the escalation override
    pinned (so the checked-in set — this function's own previous output —
    can never influence the result):

    * baseline (override = ∅): collect the infeasible missions and the
      non-home ascent edges their profiles traverse (the candidates);
    * per candidate edge (override = {edge}): re-evaluate just those
      missions; any flip marks the edge eligible.

    Eligibility is a permission, not a feasibility claim — the table pass
    afterwards decides per config what actually closes.
    """
    counts = _max_kit_counts(enabled_packs)
    eligible: set[tuple[BodyName, EdgeType]] = set()
    try:
        for difficulty in DIFFICULTIES:
            probe_difficulty = _profile_with_overhead(difficulty, overhead)
            for home in _candidate_homes():
                capability._ESCALATION_OVERRIDE = frozenset()
                mission_builder = MissionBuilder(home=home)
                cap, flags = compute_capability_from_items(
                    lambda name: counts.get(name, 0),
                    difficulty_name=probe_difficulty,
                    start_with_clamps=True,
                    mission_builder=mission_builder,
                )
                # Baseline-infeasible missions and their candidate edges.
                probes: list[tuple[BodyName, object, tuple]] = []
                for body in ALL_BODIES:
                    if body.name == home:
                        continue
                    events = get_body_events(body)
                    if not events:
                        continue
                    body_cap = cap.bodies[body.name]
                    for event in events:
                        if body_cap.access.get(event, False):
                            continue
                        ev = EVENT_BY_NAME[event]
                        edges = {
                            (e.body, e.edge_type)
                            for profile in mission_builder.profiles_for(
                                body.name, ev.mission_type)
                            for e in profile
                            if e.edge_type in _ASCENT_EDGE_TYPES
                            and e.body != home
                        }
                        if edges:
                            probes.append((body.name, ev, tuple(sorted(
                                edges, key=lambda t: (t[0].name, t[1].name)))))
                for body_name, ev, edges in probes:
                    for edge in edges:
                        if edge in eligible:
                            continue  # union semantics — already proven
                        capability._ESCALATION_OVERRIDE = frozenset({edge})
                        res = evaluate_mission_detailed(
                            flags, DIFFICULTY_PROFILES[probe_difficulty],
                            body_name, ev.mission_type, crewed=ev.crewed,
                            mission_builder=mission_builder,
                        )
                        if res.feasible:
                            eligible.add(edge)
    finally:
        capability._ESCALATION_OVERRIDE = None
    return frozenset(eligible)


# HOME-ascent escalation is a hot-path exception (every mission from that
# home pays the bigger search), so eligibility is OPERATOR-ALLOWLISTED here —
# the probe below only verifies an allowlisted edge still flips feasibility
# (dropping it if it stops mattering); it can never add a new home to the
# escalated set on its own.  Approved 2026-07-03: Eve (recalibrated ~9,000
# m/s mesa-pad ascent exceeds standard caps at the probe bar).
_HOME_ESCALATION_ALLOWLIST: frozenset[tuple[BodyName, EdgeType]] = frozenset({
    (BodyName.EVE, EdgeType.ATMOSPHERIC_ASCENT),
})


def compute_escalated_home_edges(
    apollo_escalated: frozenset[tuple[BodyName, EdgeType]],
    overhead: float = DEFAULT_OVERHEAD,
    enabled_packs: frozenset[str] = ALL_PACKS,
) -> frozenset[tuple[BodyName, EdgeType]]:
    """Allowlisted HOME-ascent edges whose escalation flips some max-kit
    mission feasible from that home, at any difficulty.

    Runs with the Apollo escalation set pinned (production behaviour) and
    the home override pinned per candidate, mirroring
    :func:`compute_escalated_edges`'s override discipline.
    """
    counts = _max_kit_counts(enabled_packs)
    eligible: set[tuple[BodyName, EdgeType]] = set()
    try:
        capability._ESCALATION_OVERRIDE = apollo_escalated
        for edge_body, edge_type in sorted(
                _HOME_ESCALATION_ALLOWLIST,
                key=lambda t: (t[0].name, t[1].name)):
            home = edge_body
            mission_builder = MissionBuilder(home=home)
            for difficulty in DIFFICULTIES:
                if (edge_body, edge_type) in eligible:
                    break
                probe_difficulty = _profile_with_overhead(difficulty, overhead)
                capability._HOME_ESCALATION_OVERRIDE = frozenset()
                cap, flags = compute_capability_from_items(
                    lambda name: counts.get(name, 0),
                    difficulty_name=probe_difficulty,
                    start_with_clamps=True,
                    mission_builder=mission_builder,
                )
                # Collect the baseline-infeasible missions BEFORE flipping
                # the override: ``cap.bodies`` is a LAZY mapping (assessed
                # on first read), so reads made after the flip would be the
                # escalated verdicts, not the baseline.
                baseline_infeasible: list = []
                for body in ALL_BODIES:
                    if body.name == home:
                        continue
                    events = get_body_events(body)
                    if not events:
                        continue
                    body_cap = cap.bodies[body.name]
                    for event in events:
                        if not body_cap.access.get(event, False):
                            baseline_infeasible.append(
                                (body.name, EVENT_BY_NAME[event]))
                capability._HOME_ESCALATION_OVERRIDE = frozenset(
                    {(edge_body, edge_type)})
                for body_name, ev in baseline_infeasible:
                    res = evaluate_mission_detailed(
                        flags, DIFFICULTY_PROFILES[probe_difficulty],
                        body_name, ev.mission_type, crewed=ev.crewed,
                        mission_builder=mission_builder,
                    )
                    if res.feasible:
                        eligible.add((edge_body, edge_type))
                        break
    finally:
        capability._ESCALATION_OVERRIDE = None
        capability._HOME_ESCALATION_OVERRIDE = None
    return frozenset(eligible)


def compute_assembly_missions(
    apollo_escalated: frozenset[tuple[BodyName, EdgeType]],
    home_escalated: frozenset[tuple[BodyName, EdgeType]],
    overhead: float = DEFAULT_OVERHEAD,
    enabled_packs: frozenset[str] = ALL_PACKS,
) -> frozenset[tuple[BodyName, BodyName, MissionType]]:
    """(home, destination, MissionType) triples the multi-launch assembly
    retry flips from infeasible to feasible at max kit, at any difficulty.

    Runs with both escalation sets pinned (production behaviour) and the
    assembly override pinned per candidate, mirroring
    :func:`compute_escalated_edges`'s override discipline.  A candidate that
    doesn't fail on the home launch costs one evaluation — the retry's
    partition enumerator returns nothing when an UPPER stage failed, so
    assembly is only ever priced where it can actually help.
    """
    counts = _max_kit_counts(enabled_packs)
    eligible: set[tuple[BodyName, BodyName, MissionType]] = set()
    try:
        capability._ESCALATION_OVERRIDE = apollo_escalated
        capability._HOME_ESCALATION_OVERRIDE = home_escalated
        for difficulty in DIFFICULTIES:
            probe_difficulty = _profile_with_overhead(difficulty, overhead)
            for home in _candidate_homes():
                capability._ASSEMBLY_OVERRIDE = frozenset()
                mission_builder = MissionBuilder(home=home)
                cap, flags = compute_capability_from_items(
                    lambda name: counts.get(name, 0),
                    difficulty_name=probe_difficulty,
                    start_with_clamps=True,
                    mission_builder=mission_builder,
                )
                # Collect the baseline-infeasible missions BEFORE flipping
                # the override (``cap.bodies`` is lazy — see
                # compute_escalated_home_edges).
                baseline_infeasible: list = []
                for body in ALL_BODIES:
                    if body.name == home:
                        continue
                    events = get_body_events(body)
                    if not events:
                        continue
                    body_cap = cap.bodies[body.name]
                    for event in events:
                        if not body_cap.access.get(event, False):
                            baseline_infeasible.append(
                                (body.name, EVENT_BY_NAME[event]))
                for body_name, ev in baseline_infeasible:
                    key = (home, body_name, ev.mission_type)
                    if key in eligible:
                        continue  # union semantics — already proven
                    capability._ASSEMBLY_OVERRIDE = frozenset({key})
                    res = evaluate_mission_detailed(
                        flags, DIFFICULTY_PROFILES[probe_difficulty],
                        body_name, ev.mission_type, crewed=ev.crewed,
                        mission_builder=mission_builder,
                    )
                    if res.feasible:
                        eligible.add(key)
    finally:
        capability._ASSEMBLY_OVERRIDE = None
        capability._ESCALATION_OVERRIDE = None
        capability._HOME_ESCALATION_OVERRIDE = None
    return frozenset(eligible)


def build_all_tables(
    overhead: float = DEFAULT_OVERHEAD,
    enabled_packs: frozenset[str] = ALL_PACKS,
) -> dict[str, dict[BodyName, frozenset[tuple[BodyName, MissionType]]]]:
    """One per-home table per difficulty (see ``DIFFICULTIES``) for a single
    pack configuration."""
    return {
        difficulty: {
            home: compute_model_infeasible_for_home(
                home, difficulty_name=difficulty, overhead=overhead,
                enabled_packs=enabled_packs)
            for home in _candidate_homes()
        }
        for difficulty in DIFFICULTIES
    }


# ---------------------------------------------------------------------------
# Pack configurations — the table is keyed by which capability-relevant packs
# are enabled.  A pack that contributes no propulsion/descent part can't change
# feasibility, so it is NOT an axis; only capability-relevant packs are probed.
# The BASE table is the default config; every other config is stored as a
# signed delta vs base, and only non-empty deltas are kept.
# ---------------------------------------------------------------------------

def _capability_relevant_optional_packs() -> list[str]:
    """Optional packs that contribute a capability-relevant part (sorted)."""
    relevant = part_manager_for(ALL_PACKS).capability_relevant_packs()
    return sorted(p for p in OPTIONAL_PACKS if p in relevant)


def _config_key(enabled_optional: frozenset[str]) -> tuple[str, ...]:
    """The capability-relevant pack set a config presents (Stock + the relevant
    optional packs it enables), sorted — the key the world looks up at runtime
    via ``PartManager.capability_relevant_packs``."""
    pm = part_manager_for(frozenset({STOCK}) | enabled_optional)
    return tuple(sorted(pm.capability_relevant_packs()))


def _base_optional() -> frozenset[str]:
    """The default config's enabled capability-relevant optional packs."""
    relevant = set(_capability_relevant_optional_packs())
    return frozenset(DEFAULT_ENABLED_OPTIONAL_PACKS) & relevant


# A per-(difficulty, home) signed delta: (added, removed) (body, mission_type)
# pairs.  Computed from the final collapsed per-config sets, so the world can
# apply ``(base | added) - removed`` as exact set algebra on tuples.
DeltaCell = tuple[frozenset[tuple[BodyName, MissionType]],
                  frozenset[tuple[BodyName, MissionType]]]


def build_base_and_deltas(
    overhead: float = DEFAULT_OVERHEAD,
) -> tuple[tuple[str, ...],
           dict[str, dict[BodyName, frozenset[tuple[BodyName, MissionType]]]],
           dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]],
           frozenset[tuple[BodyName, EdgeType]],
           frozenset[tuple[BodyName, EdgeType]],
           frozenset[tuple[BodyName, BodyName, MissionType]]]:
    """Compute escalation eligibility (Apollo + allowlisted home) and the
    assembly-mission eligibility, then the BASE table (default config) and
    signed deltas vs base for every other capability-relevant pack subset —
    all table passes run WITH every eligibility set active, exactly as
    generation will.  Returns ``(base_key, base_tables, deltas,
    escalated_edges, escalated_home_edges, assembly_missions)``; ``deltas``
    keeps only configs that differ and only their non-empty (difficulty,
    home) cells."""
    opt = _capability_relevant_optional_packs()
    base_optional = _base_optional()
    base_key = _config_key(base_optional)
    escalated = compute_escalated_edges(
        overhead=overhead, enabled_packs=frozenset({STOCK}) | base_optional)
    escalated_home = compute_escalated_home_edges(
        escalated, overhead=overhead,
        enabled_packs=frozenset({STOCK}) | base_optional)
    assembly = compute_assembly_missions(
        escalated, escalated_home, overhead=overhead,
        enabled_packs=frozenset({STOCK}) | base_optional)
    capability._ESCALATION_OVERRIDE = escalated
    capability._HOME_ESCALATION_OVERRIDE = escalated_home
    capability._ASSEMBLY_OVERRIDE = assembly
    try:
        base = build_all_tables(
            overhead=overhead, enabled_packs=frozenset({STOCK}) | base_optional)

        deltas: dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]] = {}
        for r in range(len(opt) + 1):
            for combo in itertools.combinations(opt, r):
                enabled_optional = frozenset(combo)
                if enabled_optional == base_optional:
                    continue
                cfg = build_all_tables(
                    overhead=overhead,
                    enabled_packs=frozenset({STOCK}) | enabled_optional)
                cells: dict[str, dict[BodyName, DeltaCell]] = {}
                for difficulty in DIFFICULTIES:
                    for home in _candidate_homes():
                        b = base[difficulty][home]
                        c = cfg[difficulty][home]
                        added, removed = c - b, b - c
                        if added or removed:
                            cells.setdefault(difficulty, {})[home] = (added, removed)
                if cells:
                    deltas[_config_key(enabled_optional)] = cells
    finally:
        capability._ESCALATION_OVERRIDE = None
        capability._HOME_ESCALATION_OVERRIDE = None
        capability._ASSEMBLY_OVERRIDE = None
    return base_key, base, deltas, escalated, escalated_home, assembly


def _pair_sort_key(pair: tuple[BodyName, MissionType]) -> tuple[str, str]:
    return (pair[0].name, pair[1].name)


def _fmt_pair(pair: tuple[BodyName, MissionType]) -> str:
    """Render a ``(body, mission_type)`` pair as valid, stable source."""
    return f"(BodyName.{pair[0].name}, MissionType.{pair[1].name})"


def _fmt_pair_set(pairs: frozenset[tuple[BodyName, MissionType]]) -> str:
    if not pairs:
        return "frozenset()"
    body = ", ".join(_fmt_pair(p) for p in sorted(pairs, key=_pair_sort_key))
    return f"frozenset({{{body}}})"


def _format_base(
    base: dict[str, dict[BodyName, frozenset[tuple[BodyName, MissionType]]]],
) -> list[str]:
    lines = ['MODEL_INFEASIBLE_BASE: dict[str, dict[BodyName, '
             'frozenset[tuple[BodyName, MissionType]]]] = {']
    for difficulty in DIFFICULTIES:
        table = base[difficulty]
        lines.append(f"    {difficulty!r}: {{")
        for home in sorted(table.keys()):
            lines.append(
                f"        BodyName.{home.name}: {_fmt_pair_set(table[home])},")
        lines.append("    },")
    lines.append("}")
    return lines


def _format_deltas(
    deltas: dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]],
) -> list[str]:
    lines = [
        '# Signed (added, removed) diffs vs MODEL_INFEASIBLE_BASE, keyed by the',
        "# seed's sorted capability-relevant pack tuple.  Empty == every pack",
        '# config matches base (those packs do not change representable',
        '# capability).',
        'MODEL_INFEASIBLE_DELTAS: dict[',
        '    tuple[str, ...],',
        '    dict[str, dict[BodyName, tuple['
        'frozenset[tuple[BodyName, MissionType]], '
        'frozenset[tuple[BodyName, MissionType]]]]],',
        '] = {',
    ]
    for key in sorted(deltas):
        lines.append(f"    {key!r}: {{")
        cells = deltas[key]
        for difficulty in DIFFICULTIES:
            if difficulty not in cells:
                continue
            lines.append(f"        {difficulty!r}: {{")
            for home in sorted(cells[difficulty], key=lambda b: b.name):
                added, removed = cells[difficulty][home]
                lines.append(
                    f"            BodyName.{home.name}: "
                    f"({_fmt_pair_set(added)}, {_fmt_pair_set(removed)}),")
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return lines


def _format_escalated(
    escalated: frozenset[tuple[BodyName, EdgeType]],
    escalated_home: frozenset[tuple[BodyName, EdgeType]],
) -> list[str]:
    lines = [
        '# Ascent edges eligible for the ESCALATED build caps (rocket_math',
        '# ESCALATED_*), applied by capability ONLY inside Apollo-split',
        '# evaluations.  Probed offline by compute_escalated_edges — an edge is',
        '# listed iff escalating it flips some max-kit mission feasible.  Home',
        '# ascents are never listed here (hot-path constraint) — the separate',
        '# allowlisted ESCALATED_HOME_ASCENT_EDGES below carries the approved',
        '# exceptions.',
        'ESCALATED_ASCENT_EDGES: frozenset[tuple[BodyName, EdgeType]] = frozenset({',
    ]
    for b, et in sorted(escalated, key=lambda t: (t[0].name, t[1].name)):
        lines.append(f"    (BodyName.{b.name}, EdgeType.{et.name}),")
    lines.append("})")
    lines += [
        '',
        '# HOME-ascent edges eligible for the ESCALATED caps on the PRIMARY',
        '# evaluation (every mission from that home traverses its ascent, so',
        '# this is a deliberate hot-path exception).  Operator-allowlisted in',
        '# generate_feasibility._HOME_ESCALATION_ALLOWLIST and probe-verified',
        '# to still flip feasibility; the probe can never add a home on its',
        '# own.',
        'ESCALATED_HOME_ASCENT_EDGES: '
        'frozenset[tuple[BodyName, EdgeType]] = frozenset({',
    ]
    for b, et in sorted(escalated_home, key=lambda t: (t[0].name, t[1].name)):
        lines.append(f"    (BodyName.{b.name}, EdgeType.{et.name}),")
    lines.append("})")
    return lines


def _format_assembly(
    assembly: frozenset[tuple[BodyName, BodyName, MissionType]],
) -> list[str]:
    lines = [
        '# (home, destination, mission_type) triples eligible for the multi-launch',
        '# orbital-assembly retry: missions the probe verified a single launch can',
        '# NEVER close at max kit (unlimited pad) but ≤3 docked launches can.',
        '# Probed offline by compute_assembly_missions; consulted by capability\'s',
        '# _assembly_candidate on the failure path only.',
        'ASSEMBLY_ELIGIBLE_MISSIONS: frozenset[',
        '    tuple[BodyName, BodyName, MissionType]] = frozenset({',
    ]
    for h, b, mt in sorted(assembly,
                           key=lambda t: (t[0].name, t[1].name, t[2].name)):
        lines.append(
            f"    (BodyName.{h.name}, BodyName.{b.name}, MissionType.{mt.name}),")
    lines.append("})")
    return lines


def _format(
    base_key: tuple[str, ...],
    base: dict[str, dict[BodyName, frozenset[str]]],
    deltas: dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]],
    escalated: frozenset[tuple[BodyName, EdgeType]],
    escalated_home: frozenset[tuple[BodyName, EdgeType]],
    assembly: frozenset[tuple[BodyName, BodyName, MissionType]],
) -> str:
    """Serialise base + deltas as a Python source file.  Stable ordering keeps
    the diff minimal across regenerations."""
    header = [
        '"""Static model-infeasible-missions table — checked in,',
        'regenerated by ``worlds/ksp1/scripts/generate_feasibility.py``.',
        '',
        'For each (difficulty, home), lists the ``(target_body, mission_type)``',
        'pairs whose mission the dv model cannot verify even given a maxed-out',
        'parts kit; those missions fall back to the "all-parts collected" proxy',
        'at goal time.  The world reads the pairs directly into',
        '``unachievable_missions`` — no per-name parsing — and expands them back',
        'over ``MISSION_LOCATIONS`` for the name-keyed consumers.',
        '',
        'Keyed by which capability-relevant part packs are enabled.',
        '``MODEL_INFEASIBLE_BASE`` is the default config (``BASE_RELEVANT_PACKS``);',
        '``MODEL_INFEASIBLE_DELTAS`` holds signed (added, removed) diffs vs base',
        'for any other capability-relevant pack set.  An empty deltas dict means',
        'no pack changes what capability can represent.  The world resolves',
        'base ⊕ delta for the seed\'s enabled capability-relevant packs.',
        '',
        'This file is generated; do not hand-edit.  Re-run the script whenever',
        'the dv model, mission graph, or part database changes.',
        '"""',
        'from __future__ import annotations',
        '',
        'from worlds.ksp1.bodies import BodyName, EdgeType, MissionType',
        '',
        '',
        f'BASE_RELEVANT_PACKS: tuple[str, ...] = {base_key!r}',
        '',
        '',
    ]
    lines = (header + _format_base(base) + ['', ''] + _format_deltas(deltas)
             + ['', ''] + _format_escalated(escalated, escalated_home)
             + ['', ''] + _format_assembly(assembly))
    lines.append("")
    return "\n".join(lines)


# Stable path of the checked-in table, relative to the repo root.
_TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "feasibility.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Overwrite the checked-in table at "
                             f"{_TABLE_PATH.relative_to(Path.cwd()) if _TABLE_PATH.is_relative_to(Path.cwd()) else _TABLE_PATH}.")
    parser.add_argument("--check", action="store_true",
                        help="Verify the checked-in table matches what "
                             "the script would produce; non-zero exit on drift.")
    parser.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                        help="Extra fractional percent_margin added on top of every "
                             "difficulty profile as a rep-selection safety buffer "
                             f"(default: {DEFAULT_OVERHEAD}).")
    args = parser.parse_args(argv)

    (base_key, base, deltas, escalated, escalated_home,
     assembly) = build_base_and_deltas(overhead=args.overhead)
    rendered = _format(base_key, base, deltas, escalated, escalated_home,
                       assembly)

    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    if args.write:
        _TABLE_PATH.write_text(rendered)
        print(f"Wrote {_TABLE_PATH}")
        return 0

    if args.check:
        if not _TABLE_PATH.exists():
            print(f"FAIL: {_TABLE_PATH} does not exist; run --write first.",
                  file=sys.stderr)
            return 1
        on_disk = _TABLE_PATH.read_text()
        if on_disk == rendered:
            print(f"OK: {_TABLE_PATH} matches generator output.")
            return 0
        print(f"FAIL: {_TABLE_PATH} does not match generator output.",
              file=sys.stderr)
        print("Re-run with --write and commit the result.", file=sys.stderr)
        return 1

    # Default: dump to stdout
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
