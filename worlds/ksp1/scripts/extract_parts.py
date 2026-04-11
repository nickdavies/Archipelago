#!/usr/bin/env python3
"""
Extract part data from KSP .cfg files into a JSON database.

Standalone script — no Archipelago imports. Usage:
    python extract_parts.py /path/to/GameData/Squad/Parts

Outputs data/parts.json in the same directory as this script's parent.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _strip_comment(line: str) -> str:
    """Remove inline // comments."""
    idx = line.find("//")
    if idx >= 0:
        return line[:idx]
    return line


def parse_cfg(text: str) -> list[tuple[str, dict]]:
    """
    Parse KSP cfg text into a list of (block_name, content_dict) tuples.

    Content dicts have:
    - String values for key=value pairs (last wins)
    - Lists of sub-dicts for repeated sub-blocks (MODULE, PROPELLANT, etc.)
    - '_keys' list for atmosphereCurve-style 'key = ...' entries
    """
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    tokens = _tokenize(lines)
    result, _ = _parse_top_level(tokens, 0)
    return result


def _tokenize(lines: list[str]) -> list[str]:
    """Flatten lines into meaningful tokens: block names, '{', '}', 'key=value'."""
    tokens = []
    for raw_line in lines:
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        # Could have inline braces like "PART {" on one line
        # Split on braces while preserving them as tokens
        parts = re.split(r'(\{|\})', line)
        for part in parts:
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


def _parse_top_level(tokens: list[str], pos: int) -> tuple[list[tuple[str, dict]], int]:
    """Parse top-level: sequence of NAME { content }."""
    results = []
    while pos < len(tokens):
        tok = tokens[pos]
        if tok == "}":
            return results, pos + 1
        if tok == "{":
            # Unexpected open brace — skip
            pos += 1
            continue
        if "=" in tok:
            # Stray key=value at top level — skip
            pos += 1
            continue
        # This should be a block name; next token should be '{'
        block_name = tok
        pos += 1
        if pos < len(tokens) and tokens[pos] == "{":
            pos += 1
            content, pos = _parse_content(tokens, pos)
            results.append((block_name, content))
        # else: block name without { — just continue
    return results, pos


def _parse_content(tokens: list[str], pos: int) -> tuple[dict, int]:
    """Parse content between { and }. Returns (dict, new_pos)."""
    result: dict = {}
    pending_name: str | None = None

    while pos < len(tokens):
        tok = tokens[pos]

        if tok == "}":
            return result, pos + 1

        if tok == "{":
            # Open brace after a pending block name
            pos += 1
            if pending_name is not None:
                sub_content, pos = _parse_content(tokens, pos)
                result.setdefault(pending_name, [])
                result[pending_name].append(sub_content)
                pending_name = None
            else:
                # Unexpected { — parse and discard
                _, pos = _parse_content(tokens, pos)
            continue

        if "=" in tok:
            # key = value line; flush pending_name first
            pending_name = None
            key, _, val = tok.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "key":
                result.setdefault("_keys", [])
                result["_keys"].append(val)
            else:
                result[key] = val
            pos += 1
            continue

        # Plain identifier — this is a sub-block name
        # Flush any previous pending_name (would be a stray name without {)
        pending_name = tok
        pos += 1

    return result, pos


def _extract_title(raw_title: str) -> str:
    """
    Parse title from cfg format.
    '#autoLOC_500439 //#autoLOC_500439 = LV-T30 "Reliant" Liquid Fuel Engine'
    → 'LV-T30 "Reliant" Liquid Fuel Engine'
    """
    # The title field itself may have been truncated by // stripping.
    # Re-parse from original: the format is usually:
    #   title = #autoLOC_500439 //#autoLOC_500439 = Display Name
    # But by the time we see it, // is stripped, so we just get: #autoLOC_500439
    # We need the full line. Let's handle both cases:
    match = re.search(r"//\s*#autoLOC_\d+\s*=\s*(.+)", raw_title)
    if match:
        return match.group(1).strip()
    if raw_title.startswith("#autoLOC_"):
        return raw_title
    return raw_title.strip()


def _parse_atm_curve(keys: list[str]) -> dict[str, float]:
    """Parse atmosphereCurve keys into {pressure: isp} dict."""
    result = {}
    for entry in keys:
        parts = entry.split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return result


def _parse_propellants(module_dict: dict) -> dict[str, float]:
    """Extract propellant names and ratios from a MODULE dict."""
    props = {}
    prop_blocks = module_dict.get("PROPELLANT", [])
    if not isinstance(prop_blocks, list):
        prop_blocks = [prop_blocks]
    for pb in prop_blocks:
        if isinstance(pb, dict):
            name = pb.get("name", "")
            try:
                props[name] = float(pb.get("ratio", "0"))
            except ValueError:
                props[name] = 0.0
    return props


def _parse_resources(part_dict: dict) -> dict[str, float]:
    """Extract top-level RESOURCE blocks: {name: maxAmount}."""
    resources = {}
    res_blocks = part_dict.get("RESOURCE", [])
    if not isinstance(res_blocks, list):
        res_blocks = [res_blocks]
    for rb in res_blocks:
        if isinstance(rb, dict):
            name = rb.get("name", "")
            try:
                resources[name] = float(rb.get("maxAmount", "0"))
            except ValueError:
                resources[name] = 0.0
    return resources


def _find_module(part_dict: dict, module_name: str) -> dict | None:
    """Find first MODULE block with the given name."""
    modules = part_dict.get("MODULE", [])
    if not isinstance(modules, list):
        modules = [modules]
    for mod in modules:
        if isinstance(mod, dict) and mod.get("name") == module_name:
            return mod
    return None


def _has_module(part_dict: dict, module_name: str) -> bool:
    return _find_module(part_dict, module_name) is not None


def extract_part(part_dict: dict) -> dict | None:
    """Extract relevant fields from a parsed PART block. Returns None if filtered."""
    name = part_dict.get("name", "")
    if not name:
        return None

    # Filter: TechHidden = True
    if part_dict.get("TechHidden", "").strip().lower() == "true":
        return None

    # Filter: category = none
    if part_dict.get("category", "").strip().lower() == "none":
        return None

    result: dict = {"name": name}

    # Title — // comments were stripped by tokenizer, so we lose the inline translation.
    # Re-read from the original if needed; for now store what we have.
    result["title"] = _extract_title(part_dict.get("title", ""))

    # Mass
    try:
        result["mass"] = float(part_dict.get("mass", "0"))
    except ValueError:
        result["mass"] = 0.0

    # Bulkhead profiles
    raw_profiles = part_dict.get("bulkheadProfiles", "")
    result["bulkhead_profiles"] = [p.strip() for p in raw_profiles.split(",") if p.strip()]

    # Tech required
    result["tech_required"] = part_dict.get("TechRequired", "")

    # Engine data
    engine_mod = _find_module(part_dict, "ModuleEngines") or _find_module(part_dict, "ModuleEnginesFX")
    if engine_mod:
        engine: dict = {}
        try:
            engine["max_thrust"] = float(engine_mod.get("maxThrust", "0"))
        except ValueError:
            engine["max_thrust"] = 0.0

        # ISP from atmosphereCurve
        atm_blocks = engine_mod.get("atmosphereCurve", [])
        curve_dict = atm_blocks[0] if isinstance(atm_blocks, list) and atm_blocks else {}
        isp_map = _parse_atm_curve(curve_dict.get("_keys", []))
        engine["isp_vac"] = isp_map.get("0", 0.0)
        engine["isp_atm"] = isp_map.get("1", 0.0)

        engine["engine_type"] = engine_mod.get("EngineType", "")
        engine["throttle_locked"] = engine_mod.get("throttleLocked", "False").strip().lower() == "true"
        engine["propellants"] = _parse_propellants(engine_mod)
        result["engine"] = engine

    # Gimbal
    result["has_gimbal"] = _has_module(part_dict, "ModuleGimbal")

    # Parachute
    para_mod = _find_module(part_dict, "ModuleParachute")
    if para_mod:
        try:
            result["parachute"] = {
                "fully_deployed_drag": float(para_mod.get("fullyDeployedDrag", "0")),
            }
        except ValueError:
            result["parachute"] = {"fully_deployed_drag": 0.0}

    # Decoupler flags
    result["has_module_decouple"] = _has_module(part_dict, "ModuleDecouple")
    result["has_module_anchored_decouple"] = _has_module(part_dict, "ModuleAnchoredDecoupler")

    # Resources
    resources = _parse_resources(part_dict)
    if resources:
        result["resources"] = resources

    return result


def walk_and_extract(parts_dir: str) -> dict[str, dict]:
    """Recursively find all .cfg files, parse PART blocks, extract data."""
    parts_path = Path(parts_dir)
    all_parts: dict[str, dict] = {}

    for cfg_file in sorted(parts_path.rglob("*.cfg")):
        try:
            text = cfg_file.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue

        top_blocks = parse_cfg(text)
        for block_name, content in top_blocks:
            if block_name != "PART":
                continue
            part = extract_part(content)
            if part is not None:
                all_parts[part["name"]] = part

    return all_parts


def _read_title_from_raw(cfg_path: Path) -> dict[str, str]:
    """
    Second pass: read titles from raw file text (before // stripping)
    to recover the autoLOC inline translations.
    """
    titles = {}
    try:
        text = cfg_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return titles

    current_name = None
    for line in text.splitlines():
        stripped = line.strip()
        # Track part name
        name_match = re.match(r"name\s*=\s*(\S+)", stripped)
        if name_match and current_name is None:
            current_name = name_match.group(1)
        # Track title WITH the // comment intact
        title_match = re.match(r"title\s*=\s*(.+)", stripped)
        if title_match and current_name:
            raw_title = title_match.group(1).strip()
            parsed = _extract_title(raw_title)
            titles[current_name] = parsed
            current_name = None
    return titles


def walk_and_extract_with_titles(parts_dir: str) -> dict[str, dict]:
    """Extract parts and fix up titles from raw file text."""
    parts_path = Path(parts_dir)
    all_parts = walk_and_extract(parts_dir)

    # Second pass for titles (// stripping loses autoLOC translations)
    for cfg_file in sorted(parts_path.rglob("*.cfg")):
        title_map = _read_title_from_raw(cfg_file)
        for part_name, title in title_map.items():
            if part_name in all_parts and not title.startswith("#autoLOC"):
                all_parts[part_name]["title"] = title

    return all_parts


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/Parts> [<path/to/Parts> ...]",
              file=sys.stderr)
        print("  e.g.: extract_parts.py GameData/Squad/Parts GameData/SquadExpansion/MakingHistory/Parts",
              file=sys.stderr)
        sys.exit(1)

    parts_dirs = sys.argv[1:]
    for d in parts_dirs:
        if not os.path.isdir(d):
            print(f"Error: {d} is not a directory", file=sys.stderr)
            sys.exit(1)

    # Merge parts from all directories (later dirs override earlier on conflict)
    parts: dict[str, dict] = {}
    for parts_dir in parts_dirs:
        parts.update(walk_and_extract_with_titles(parts_dir))

    # Sort alphabetically and add metadata
    sorted_parts = dict(sorted(parts.items()))
    output = {
        "_meta": {
            "sources": [os.path.abspath(d) for d in parts_dirs],
            "generated": datetime.now(timezone.utc).isoformat(),
            "part_count": len(sorted_parts),
        },
        **sorted_parts,
    }

    # Write to data/parts.json relative to this script
    out_path = Path(__file__).resolve().parent.parent / "data" / "parts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(sorted_parts)} parts to {out_path}")


if __name__ == "__main__":
    main()
