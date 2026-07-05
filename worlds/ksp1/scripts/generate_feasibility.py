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
3. Records which non-home landable bodies have ``RETURN`` or
   ``SAMPLE_RETURN`` reported as unreachable (False access).  These
   bodies get the "all-parts collected" proxy rule at goal-evaluation
   time instead of the dv-based capability check.

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
    MissionBuilder, MissionType,
)
from worlds.ksp1.capability import compute_capability_from_items
from worlds.ksp1.locations import EVENT_BY_NAME, EventName
from worlds.ksp1.parts import ALL_PACKS, part_manager_for
from worlds.ksp1.parts.packs import (
    STOCK, OPTIONAL_PACKS, DEFAULT_ENABLED_OPTIONAL_PACKS,
)


# Mission events whose feasibility the script probes.  Each (body, event)
# pair that fails the capability check is recorded as its canonical
# ``(target_body, mission_type)`` pair — the same shape the world's
# ``unachievable_missions`` carries; the world expands it back over every
# matching AP location (all slots of every event sharing that mission type).
# The probed events map to DISTINCT mission types (RETURN vs SAMPLE_RETURN), so
# the pair keeps them separable.
_PROBED_EVENTS: tuple[EventName, ...] = (EventName.RETURN, EventName.SAMPLE_RETURN)


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
# difficulties because rep variance is difficulty-independent.  At 25% the
# marginal deep sample-returns that widen the fill-famine tail (e.g. expert SSR
# requiring Tylo+Laythe) drop back to the proxy, restoring the zero-reject bar.
DEFAULT_OVERHEAD: float = 0.25

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
        if body.name == home or not body.can_land:
            continue
        body_cap = cap.bodies[body.name]
        for event in _PROBED_EVENTS:
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
           dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]]]:
    """Compute the BASE table (default config) and signed deltas vs base for
    every other capability-relevant pack subset.  Returns
    ``(base_key, base_tables, deltas)``; ``deltas`` keeps only configs that
    differ and only their non-empty (difficulty, home) cells."""
    opt = _capability_relevant_optional_packs()
    base_optional = _base_optional()
    base_key = _config_key(base_optional)
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
    return base_key, base, deltas


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


def _format(
    base_key: tuple[str, ...],
    base: dict[str, dict[BodyName, frozenset[str]]],
    deltas: dict[tuple[str, ...], dict[str, dict[BodyName, DeltaCell]]],
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
        'from worlds.ksp1.bodies import BodyName, MissionType',
        '',
        '',
        f'BASE_RELEVANT_PACKS: tuple[str, ...] = {base_key!r}',
        '',
        '',
    ]
    lines = header + _format_base(base) + ['', ''] + _format_deltas(deltas)
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

    base_key, base, deltas = build_base_and_deltas(overhead=args.overhead)
    rendered = _format(base_key, base, deltas)

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
