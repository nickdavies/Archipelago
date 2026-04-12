#!/usr/bin/env python3
"""
Extract tech tree data from KSP's TechTree.cfg into a JSON database.

Standalone script — no Archipelago imports. Usage:
    python extract_tech_tree.py /path/to/TechTree.cfg

Outputs data/tech_tree.json in the same directory as this script's parent.
Prints a C# TechDisplayNames dictionary snippet to stderr (paste convenience).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from cfg_parser import parse_cfg

# Map science cost → tier number. Derived from stock TechTree.cfg.
COST_TO_TIER: dict[int, int] = {
    5: 1,
    15: 2, 18: 2, 20: 2,
    45: 3,
    90: 4,
    160: 5,
    300: 6,
    550: 7,
    1000: 8,
}


def _read_titles_from_raw(cfg_path: Path) -> dict[str, str]:
    """
    Second pass: read titles from raw file text (before // stripping)
    to recover the autoLOC inline translations.

    Lines look like:
        title = #autoLOC_501022 //#autoLOC_501022 = Basic Rocketry
    """
    titles: dict[str, str] = {}
    try:
        text = cfg_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return titles

    current_id: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        id_match = re.match(r"id\s*=\s*(\S+)", stripped)
        if id_match:
            current_id = id_match.group(1)
        title_match = re.match(r"title\s*=\s*(.+)", stripped)
        if title_match and current_id:
            raw = title_match.group(1).strip()
            # Extract English name from: #autoLOC_NNN //#autoLOC_NNN = English Title
            eng_match = re.search(r"//\s*#autoLOC_\d+\s*=\s*(.+)", raw)
            if eng_match:
                titles[current_id] = eng_match.group(1).strip()
            elif not raw.startswith("#autoLOC_"):
                titles[current_id] = raw.strip()
            current_id = None
    return titles


def extract_tech_tree(cfg_path: Path) -> list[dict]:
    """Parse TechTree.cfg and return a list of node dicts (excluding 'start')."""
    text = cfg_path.read_text(encoding="utf-8-sig", errors="replace")
    top_blocks = parse_cfg(text)

    # Find the TechTree block
    tech_tree_content = None
    for block_name, content in top_blocks:
        if block_name == "TechTree":
            tech_tree_content = content
            break

    if tech_tree_content is None:
        print("Error: No TechTree block found in cfg file", file=sys.stderr)
        sys.exit(1)

    rd_nodes = tech_tree_content.get("RDNode", [])
    if not isinstance(rd_nodes, list):
        rd_nodes = [rd_nodes]

    # Raw-line title recovery (// stripping loses the inline translations)
    raw_titles = _read_titles_from_raw(cfg_path)

    nodes: list[dict] = []
    for node_dict in rd_nodes:
        node_id = node_dict.get("id", "")
        cost = int(node_dict.get("cost", "0"))

        # Skip the free start node
        if cost == 0:
            continue

        tier = COST_TO_TIER.get(cost)
        if tier is None:
            print(f"Warning: Unknown cost {cost} for node '{node_id}', skipping",
                  file=sys.stderr)
            continue

        # Title: prefer raw-line recovery, fall back to parsed value
        title = raw_titles.get(node_id, node_dict.get("title", node_id))

        # Parents
        parent_blocks = node_dict.get("Parent", [])
        if not isinstance(parent_blocks, list):
            parent_blocks = [parent_blocks]
        parents = [
            pb["parentID"]
            for pb in parent_blocks
            if isinstance(pb, dict) and "parentID" in pb
        ]

        any_to_unlock = node_dict.get("anyToUnlock", "False").strip().lower() == "true"

        nodes.append({
            "id": node_id,
            "title": title,
            "cost": cost,
            "tier": tier,
            "parents": parents,
            "any_to_unlock": any_to_unlock,
        })

    # Sort by tier, then by id
    nodes.sort(key=lambda n: (n["tier"], n["id"]))
    return nodes


def _print_csharp_snippet(nodes: list[dict]) -> None:
    """Print a C# dictionary snippet to stderr for paste convenience."""
    print("\n// C# TechDisplayNames (paste into MissionTracker.cs):", file=sys.stderr)
    print("internal static readonly Dictionary<string, string> TechDisplayNames = new Dictionary<string, string>", file=sys.stderr)
    print("{", file=sys.stderr)
    for node in nodes:
        print(f'    {{ "{node["id"]}",{" " * max(1, 32 - len(node["id"]))}"{node["title"]}" }},',
              file=sys.stderr)
    print("};", file=sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/TechTree.cfg>", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    if not cfg_path.is_file():
        print(f"Error: {cfg_path} is not a file", file=sys.stderr)
        sys.exit(1)

    nodes = extract_tech_tree(cfg_path)

    output = {
        "_meta": {
            "source": str(cfg_path.resolve()),
            "generated": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
        },
        "nodes": nodes,
    }

    # Write to data/tech_tree.json relative to this script
    out_path = Path(__file__).resolve().parent.parent / "data" / "tech_tree.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(nodes)} nodes to {out_path}")

    # Print tier summary
    tier_counts: dict[int, int] = {}
    for n in nodes:
        tier_counts[n["tier"]] = tier_counts.get(n["tier"], 0) + 1
    for tier in sorted(tier_counts):
        print(f"  Tier {tier}: {tier_counts[tier]} nodes")

    _print_csharp_snippet(nodes)


if __name__ == "__main__":
    main()
