"""Generate the Universal Tracker map tab data: ``tracker/maps.json`` and
``tracker/locations.json``.

Inputs are the Kerbonaut map pack's marker coordinates — a checkout of
https://github.com/1Kerbonaut/KSP-Style-Flags (``UT Maps/Marker_Coords/<Home>_Home.json``)
— and the location catalog, placed on nodes by ``ut_map.location_node`` /
``ut_map.location_map``.  A node whose sections are the same on every map is
one PopTracker location with a ``map_locations`` entry per map; map-specific
nodes (home specials) get one per map.  Every marker must have locations and
every mapped location must have a marker, so drift on either side fails loudly.

Regenerate after changing locations or updating the pack:

    cd Archipelago && ../.venv/bin/python -m worlds.ksp1.scripts.generate_ut_map \\
        --source <KSP-Style-Flags checkout> --write

``--check`` verifies the checked-in files match (non-zero exit on drift).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worlds.ksp1.bodies import ALL_BODIES, BodyName
from worlds.ksp1.locations.catalog import LocationCatalog
from worlds.ksp1.locations.defs import MissionLocation
from worlds.ksp1.ut_map import UT_ENTRANCES, UT_MAP_HOMES, location_map, location_node

TRACKER_DIR = Path(__file__).resolve().parent.parent / "tracker"
MAPS_PATH = TRACKER_DIR / "maps.json"
LOCATIONS_PATH = TRACKER_DIR / "locations.json"


def _marker_nodes(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    locs = blob["UT Locations"]
    return [
        *blob["UT Entrances"]["nodes"], *locs["nodes"], *locs["sidebar"]["nodes"],
        *locs["altitude"]["nodes"], *locs["science"]["nodes"],
    ]


def _sections_by_node() -> dict[str, list[tuple[str, BodyName | None]]]:
    """Every node label → (location / entrance name, the one map it belongs
    to or ``None`` for every map) listed on it."""
    out: dict[str, list[tuple[str, BodyName | None]]] = {}
    for defn in LocationCatalog.all():
        node = location_node(defn)
        if node is not None:
            out.setdefault(node, []).append((str(defn), location_map(defn)))
    missions = set(LocationCatalog.missions())
    for body in ALL_BODIES:
        for entrance in UT_ENTRANCES:
            if MissionLocation(body.name, entrance.event, 1) in missions:
                out.setdefault(entrance.node(body.name), []).append(
                    (entrance.name(body.name), None))
    return out


def build(source: Path) -> tuple[list[dict], list[dict]]:
    """UT draws a location's sections on every map in its ``map_locations``,
    filtered only by the connected slot (not by map).  So a node is shared
    across maps only where its section list is identical; map-specific
    sections (home specials) give that node one entry per map."""
    marker_dir = source / "UT Maps" / "Marker_Coords"
    sections = _sections_by_node()
    entries: dict[tuple[str, tuple[str, ...]], dict] = {}
    for home in UT_MAP_HOMES:
        path = marker_dir / f"{home}_Home.json"
        for node in _marker_nodes(path):
            label = node["label"]
            names = tuple(name for name, only_on in sections.get(label, ())
                          if only_on in (None, home))
            if not names:
                raise SystemExit(f"{path.name}: marker {label!r} has no locations; "
                                 "update ut_map.location_node / UT_ENTRANCES")
            entry = entries.setdefault((label, names), {
                "name": label,
                "sections": [{"name": s} for s in names],
                "map_locations": [],
            })
            entry["map_locations"].append({"map": str(home), "x": node["x"], "y": node["y"]})
    unplaced = sorted(set(sections) - {label for label, _names in entries})
    if unplaced:
        raise SystemExit(f"nodes with locations but no marker on any map: {unplaced}")
    maps = [{"name": str(home), "img": f"{home}_Home.png"} for home in UT_MAP_HOMES]
    return maps, list(entries.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Checkout of the KSP-Style-Flags repo.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="Overwrite the checked-in tracker/*.json.")
    mode.add_argument("--check", action="store_true",
                      help="Verify tracker/*.json match; non-zero exit on drift.")
    args = parser.parse_args(argv)

    maps, locations = build(args.source)
    rendered = {MAPS_PATH: json.dumps(maps, indent=2),
                LOCATIONS_PATH: json.dumps(locations, indent=2)}
    if args.write:
        for path, text in rendered.items():
            path.write_text(text)
            print(f"Wrote {path}")
        return 0
    stale = [p for p, text in rendered.items() if not p.exists() or p.read_text() != text]
    for path in stale:
        print(f"FAIL: {path} does not match generator output.", file=sys.stderr)
    if stale:
        print("Re-run with --write and commit the result.", file=sys.stderr)
        return 1
    print("OK: tracker/*.json match generator output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
