"""
Shared parser for KSP .cfg files.

Handles the brace-delimited block structure, inline // comments,
and repeated sub-blocks (MODULE, PROPELLANT, Parent, etc.).
"""
from __future__ import annotations

import re


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
