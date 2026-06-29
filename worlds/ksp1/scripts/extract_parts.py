#!/usr/bin/env python3
"""
Extract part data from KSP .cfg files into a JSON database.

Point it at a GameData directory (ideally one with every pack installed). It
discovers every part-bearing pack under GameData, labels each from
``parts/packs.py`` (``KNOWN_PACK_ROOTS``), and stamps a per-part ``pack`` field.
An unlabeled part-bearing pack is a hard error — add a mapping or ``--exclude``
it — so a pack is never silently dropped. Usage:

    python extract_parts.py /path/to/GameData [--exclude PACK ...]

Loads only ``parts/packs.py`` (a dependency-free leaf) by file path, so it does
not import the Archipelago part database. Outputs data/parts.json under the
script's parent directory.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from cfg_parser import parse_cfg

# Load the pack-identity model by file path so this script stays standalone
# (importing worlds.ksp1.parts would build the whole part DB). packs.py is a
# dependency-free leaf, so loading it in isolation is safe.
import importlib.util as _ilu

_packs_path = Path(__file__).resolve().parent.parent / "parts" / "packs.py"
_spec = _ilu.spec_from_file_location("ksp1_part_packs", _packs_path)
assert _spec is not None and _spec.loader is not None
packs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(packs)


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


def _parse_vec(val: str) -> list[float] | None:
    """Parse a comma-separated float vector ('x, y, z, ...') into a list of
    floats, or None if any field is non-numeric.  Used for attach nodes and
    CoMOffset."""
    out: list[float] = []
    for tok in val.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out or None


def _parse_node(val: str) -> dict | None:
    """Parse a KSP attach-node line into ``{pos:[x,y,z], dir:[x,y,z], size}``.

    KSP node format is ``x, y, z, orient_x, orient_y, orient_z[, size]`` where
    ``y`` is the vertical/stack axis and (x,z) the radial plane.  ``size`` is the
    node diameter class (int) and may be omitted.  Returns None if fewer than 3
    numeric fields parse.  Emitted raw so consumers can derive structural roles
    (spine-stackable / radial / splitter) from the geometry rather than guessing
    from bulkhead profiles."""
    nums = _parse_vec(val)
    if nums is None or len(nums) < 3:
        return None
    return {
        "pos": nums[0:3],
        "dir": nums[3:6] if len(nums) >= 6 else [0.0, 0.0, 0.0],
        "size": int(nums[6]) if len(nums) >= 7 else None,
    }


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

    # Crew capacity (seats) — drives crewed-station contracts. Only emitted when
    # >0 so non-crew parts stay lean (mirrors the resources/engine pattern).
    try:
        crew = int(float(part_dict.get("CrewCapacity", "0")))
    except ValueError:
        crew = 0
    if crew > 0:
        result["crew_capacity"] = crew

    # Bulkhead profiles
    raw_profiles = part_dict.get("bulkheadProfiles", "")
    result["bulkhead_profiles"] = [p.strip() for p in raw_profiles.split(",") if p.strip()]

    # Tech required
    result["tech_required"] = part_dict.get("TechRequired", "")

    # Attach geometry (generic facts; consumers derive structural roles —
    # spine-stackable / radial / splitter — from this rather than from bulkhead
    # profiles, which can't tell a straight adapter from a slanted one).
    stack_nodes = []
    for k in sorted(part_dict):
        if k.startswith("node_stack"):
            node = _parse_node(part_dict[k])
            if node is not None:
                node["id"] = k[len("node_stack_"):] or "stack"
                stack_nodes.append(node)
    if stack_nodes:
        result["stack_nodes"] = stack_nodes
    if "node_attach" in part_dict:
        an = _parse_node(part_dict["node_attach"])
        if an is not None:
            result["attach_node"] = an
    if "CoMOffset" in part_dict:
        com = _parse_vec(part_dict["CoMOffset"])
        if com is not None and len(com) >= 3:
            result["com_offset"] = com[0:3]
    if "attachRules" in part_dict:
        rules = _parse_vec(part_dict["attachRules"])
        if rules is not None:
            result["attach_rules"] = [int(r) for r in rules]

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

    # Crew capacity (top-level field, present on command pods / crew cabins)
    try:
        crew = int(part_dict.get("CrewCapacity", "0"))
    except ValueError:
        crew = 0
    if crew > 0:
        result["crew_capacity"] = crew

    # Solar panel (fixed and deployable both use ModuleDeployableSolarPanel)
    solar_mod = _find_module(part_dict, "ModuleDeployableSolarPanel")
    if solar_mod:
        try:
            charge = float(solar_mod.get("chargeRate", "0"))
        except ValueError:
            charge = 0.0
        if charge > 0:
            # `isTracking = false` is set explicitly on fixed panels (OX-STAT,
            # OX-STAT-XL).  Absent or "true" means deployable/tracking.
            tracking = solar_mod.get("isTracking", "true").strip().lower() != "false"
            result["solar"] = {
                "charge_rate": charge,
                "tracking": tracking,
            }

    # Antenna (ModuleDataTransmitter)
    antenna_mod = _find_module(part_dict, "ModuleDataTransmitter")
    if antenna_mod:
        try:
            power = float(antenna_mod.get("antennaPower", "0"))
        except ValueError:
            power = 0.0
        if power > 0:
            combinable = antenna_mod.get("antennaCombinable", "False").strip().lower() == "true"
            atype = antenna_mod.get("antennaType", "").strip()
            result["antenna"] = {
                "power": power,
                "combinable": combinable,
                "type": atype,
            }

    # SAS service level (ModuleSAS — appears on probe cores AND command pods AND
    # standalone reaction wheel modules).  Standalone reaction wheels generally
    # don't carry SAS in stock; probe cores and pods do.  Stored unconditionally
    # so the rank scorer can use it directly.
    sas_mod = _find_module(part_dict, "ModuleSAS")
    if sas_mod:
        try:
            lvl = int(sas_mod.get("SASServiceLevel", "0"))
        except ValueError:
            lvl = 0
        result["sas_level"] = lvl

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


def _tag_pack(part: dict, pack: str) -> dict:
    """Return ``part`` with a ``pack`` field inserted right after ``name`` (so
    JSON key order stays stable: name, pack, then the rest)."""
    return {"name": part.get("name", ""), "pack": pack,
            **{k: v for k, v in part.items() if k not in ("name", "pack")}}


def _has_part_block(root: Path) -> bool:
    """True if any .cfg under ``root`` declares a top-level PART block. Used to
    decide whether a GameData directory is a part-bearing pack (and therefore
    must be explicitly labeled or excluded)."""
    for cfg_file in root.rglob("*.cfg"):
        try:
            text = cfg_file.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for block_name, _ in parse_cfg(text):
            if block_name == "PART":
                return True
    return False


def _candidate_pack_roots(gamedata: Path) -> list[Path]:
    """Pack roots under a GameData directory: every top-level mod folder, with
    ``SquadExpansion`` expanded one level (its children are the DLC packs)."""
    roots: list[Path] = []
    for child in sorted(gamedata.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "SquadExpansion":
            roots.extend(sorted(s for s in child.iterdir() if s.is_dir()))
        else:
            roots.append(child)
    return roots


def main() -> None:
    argv = sys.argv[1:]
    excluded: set[str] = set()
    positional: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--exclude" and i + 1 < len(argv):
            excluded.add(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1

    if len(positional) != 1:
        print(f"Usage: {sys.argv[0]} <path/to/GameData> [--exclude PACK ...]",
              file=sys.stderr)
        print("  Discovers every part-bearing pack under GameData and labels it"
              " from packs.KNOWN_PACK_ROOTS.", file=sys.stderr)
        print("  An unlabeled pack is a hard error (add a mapping or --exclude"
              " it) so a pack is never silently dropped.", file=sys.stderr)
        sys.exit(1)

    gamedata = Path(positional[0])
    if not gamedata.is_dir():
        print(f"Error: {gamedata} is not a directory", file=sys.stderr)
        sys.exit(1)

    parts: dict[str, dict] = {}
    sources: list[str] = []
    packs_present: set[str] = set()

    for root in _candidate_pack_roots(gamedata):
        if not _has_part_block(root):
            continue
        rel = root.relative_to(gamedata).as_posix()
        pack = packs.pack_for_gamedata_dir(rel)
        if pack is None:
            print(
                f"Error: part-bearing pack root {rel!r} has no entry in "
                f"packs.KNOWN_PACK_ROOTS.\n"
                f"  Add a mapping for it, or pass --exclude <pack> to drop it "
                f"on purpose. Refusing to silently drop a pack.",
                file=sys.stderr,
            )
            sys.exit(2)
        if pack in excluded:
            print(f"Skipping excluded pack {pack!r} ({rel})", file=sys.stderr)
            continue
        # Later packs override earlier on cfg-name conflict (matches old merge).
        for name, part in walk_and_extract_with_titles(str(root)).items():
            parts[name] = _tag_pack(part, pack)
        sources.append(os.path.abspath(str(root)))
        packs_present.add(pack)

    sorted_parts = dict(sorted(parts.items()))
    output = {
        "_meta": {
            "sources": sources,
            "packs": sorted(packs_present),
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

    print(f"Wrote {len(sorted_parts)} parts from packs "
          f"{sorted(packs_present)} to {out_path}")


if __name__ == "__main__":
    main()
