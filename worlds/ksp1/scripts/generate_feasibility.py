#!/usr/bin/env python3
"""Generate the static FEASIBILITY table consumed by the world rules.

For every body that can serve as a starting body, this script:

1. Builds an "everything maxed" item-count function — one of every
   individual part in ``PART_DB`` plus every progressive item at its
   max tier.  This is strictly more permissive than any single seed's
   rep selection, so the set of bodies it flags as infeasible is the
   floor: anything banned here can't be reached even by the best-case
   parts kit.
2. Runs ``compute_capability_from_items`` against a ``MissionBuilder``
   rooted at that home.
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
import sys
from pathlib import Path

from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, DIFFICULTY_PROFILES, DifficultyProfile,
    MissionBuilder,
)
from worlds.ksp1.capability import compute_capability_from_items
from worlds.ksp1.locations import EVENT_BY_NAME, EventName, MissionLocation
from worlds.ksp1.parts import PART_DB, PROGRESSIVE_PART_COUNTS


# Mission events whose feasibility the script probes.  Each (body, event)
# pair that fails the capability check expands to every AP location name
# in that event's scale (e.g. RETURN scale=3 → "Body Return 1..3"),
# giving the per-location granularity the rules layer can filter on.
_PROBED_EVENTS: tuple[EventName, ...] = (EventName.RETURN, EventName.SAMPLE_RETURN)


# Difficulty used when probing feasibility.  ``casual`` is intentional —
# its 30% percent-margin and 200 m/s fixed-margin add enough headroom on
# top of the raw capability check that a body has to be solidly feasible
# with maxed reps before we declare it non-proxy.  Marginal missions
# (Tylo return at ~4 km/s in a single Kerbin stage) that *just barely*
# clear the normal-difficulty check fail under casual margins and end
# up routed to the all-parts proxy, where they belong.
DEFAULT_DIFFICULTY: str = "casual"

# Extra ``percent_margin`` on top of the chosen difficulty profile.
# ``casual + 25%`` matches the post-F4 banned set against the historical
# hand-tuned list with only one residual difference: Tylo→Kerbin Sample
# Return is now feasible (F4's multi-stage + LF Tank + ion guarantee
# makes the Tylo→Kerbin→Tylo round trip genuinely buildable).  Lower
# overheads under-ban Laythe Return from many homes; higher overheads
# (≥30%) over-ban Tylo→Laythe Return as collateral.  Was 10% pre-F4
# when single-stage modelling left more missions naturally infeasible.
DEFAULT_OVERHEAD: float = 0.25


def _max_kit_counts() -> dict[str, int]:
    """One of every individual part, every progressive at max tier.

    The progressive-tier overrides come second so they win when an item
    name appears in both PART_DB and PROGRESSIVE_PART_COUNTS.
    """
    counts: dict[str, int] = {name: 1 for name in PART_DB}
    counts.update(PROGRESSIVE_PART_COUNTS)
    return counts


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
    DIFFICULTY_PROFILES[name] = DifficultyProfile(
        fixed_margin=base.fixed_margin,
        percent_margin=base.percent_margin + overhead,
        plane_change_fraction=base.plane_change_fraction,
        min_twr_atmo=base.min_twr_atmo,
        min_twr_vac=base.min_twr_vac,
        ship_cd=base.ship_cd,
        srb_needs_rcs=base.srb_needs_rcs,
    )
    return name


def compute_model_infeasible_for_home(
    home: BodyName,
    difficulty_name: str = DEFAULT_DIFFICULTY,
    overhead: float = DEFAULT_OVERHEAD,
) -> frozenset[str]:
    """AP location names whose mission the dv model can't verify from
    ``home``, even given a maxed-out parts kit.  Used to drive the
    all-parts proxy rule for those locations' completion checks.

    Locations are listed individually (one entry per AP slot — e.g.
    ``"Eve Return 1"``, ``"Eve Return 2"``, ``"Eve Return 3"``) so
    callers can filter per-mission-type (Return vs Sample Return vs
    eventual Flag Plant) if behaviour needs to diverge between them.

    ``overhead`` adds extra ``percent_margin`` on top of the chosen
    difficulty profile — tightens the "what counts as feasible" bar
    when you need locations that *just barely* pass with maxed reps
    to still be classified as model-infeasible.
    """
    counts = _max_kit_counts()
    mission_builder = MissionBuilder(home=home)
    probe_difficulty = _profile_with_overhead(difficulty_name, overhead)
    cap, _flags = compute_capability_from_items(
        lambda name: counts.get(name, 0),
        difficulty_name=probe_difficulty,
        start_with_clamps=True,
        mission_builder=mission_builder,
    )
    infeasible: set[str] = set()
    for body in ALL_BODIES:
        if body.name == home or not body.can_land:
            continue
        body_cap = cap.bodies[body.name]
        for event in _PROBED_EVENTS:
            if body_cap.access.get(event, False):
                continue
            scale = EVENT_BY_NAME[event].scale
            for slot in range(1, scale + 1):
                infeasible.add(str(MissionLocation(body.name, event, slot)))
    return frozenset(infeasible)


def _candidate_homes() -> list[BodyName]:
    """Bodies that can serve as a starting body.  Must have a surface
    (``can_land``) and be inside the patched-conic graph (Kerbol is
    excluded — it's the root star, not a launch site).
    """
    return [b.name for b in ALL_BODIES
            if b.can_land and b.name != BodyName.KERBOL]


def build_table(
    difficulty_name: str = DEFAULT_DIFFICULTY,
    overhead: float = DEFAULT_OVERHEAD,
) -> dict[BodyName, frozenset[str]]:
    """Compute the full per-home model-infeasible-locations table."""
    table: dict[BodyName, frozenset[str]] = {}
    for home in _candidate_homes():
        table[home] = compute_model_infeasible_for_home(
            home, difficulty_name=difficulty_name, overhead=overhead,
        )
    return table


def _format_table(table: dict[BodyName, frozenset[str]]) -> str:
    """Serialise the table as a Python source file.  Stable ordering so
    the diff is minimal across regenerations.
    """
    lines = [
        '"""Static model-infeasible-locations table — checked in,',
        'regenerated by ``worlds/ksp1/scripts/generate_feasibility.py``.',
        '',
        'For each body that can serve as a starting body, lists the AP',
        'location names whose mission the dv model cannot verify even',
        'when the player has every progressive item at max tier and one',
        'of every part in ``PART_DB``.  Goal rules that include any of',
        'these locations fall back to the "all-parts collected" proxy',
        'at completion-check time.',
        '',
        'Entries are full AP location names (one per slot — e.g. three',
        '``Eve Return 1..3`` entries for the three RETURN slots) so the',
        'rules layer can filter Return vs Sample Return vs other mission',
        'types independently.',
        '',
        'This file is generated; do not hand-edit.  Re-run the script',
        'whenever the dv model, mission graph, or part database changes.',
        '"""',
        'from __future__ import annotations',
        '',
        'from worlds.ksp1.bodies import BodyName',
        '',
        '',
        'MODEL_INFEASIBLE_LOCATIONS: dict[BodyName, frozenset[str]] = {',
    ]
    for home in sorted(table.keys()):
        entries = sorted(table[home])
        if not entries:
            lines.append(f"    BodyName.{home.name}: frozenset(),")
        else:
            quoted = ", ".join(f"{e!r}" for e in entries)
            lines.append(f"    BodyName.{home.name}: frozenset({{{quoted}}}),")
    lines.append("}")
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
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY,
                        choices=sorted(DIFFICULTY_PROFILES.keys()),
                        help=f"DifficultyProfile name to use for the feasibility "
                             f"probe (default: {DEFAULT_DIFFICULTY!r} — its larger "
                             f"margins keep marginal missions in the proxy set).")
    parser.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                        help="Extra fractional percent_margin added on top of the "
                             "chosen difficulty profile (e.g. 0.10 = require an "
                             "extra 10%% dv headroom for a body to count as non-"
                             "proxy).  Default 0.")
    args = parser.parse_args(argv)

    table = build_table(difficulty_name=args.difficulty, overhead=args.overhead)
    rendered = _format_table(table)

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
