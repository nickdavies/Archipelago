"""Structural role classification from generic attach-node geometry.

Pure module — no AP imports, like ``rocket_math``.  It consumes the raw geometry
``scripts/extract_parts.py`` emits (``stack_nodes`` / ``attach_node`` /
``com_offset``) and derives the structural **roles** a part can play in a rocket.

The point is a single, shared definition of *what a part can build*.  The old
``is_radial`` flag (bulkhead == srf-only) is too coarse in both directions: it
treats single-node and slanted tanks as stackable spines (a Golden-Rule false
positive — claiming buildable when it isn't), and it ignores that a genuinely
radial tank is a perfectly good droppable booster.  Capability and the
rank/ladder side both read these roles, so the rep pool can never offer a tank
capability would refuse to fly.

Roles (a part may carry several):

* ``SPINE`` — can be a stage's central, load-bearing column: at least two stack
  nodes, all colinear on a common vertical axis, with the centre of mass on that
  axis.  Normal cylindrical tanks and *straight* adapters.  Also valid as a
  radially-decoupled booster column.
* ``RADIAL_MOUNT`` — surface-attachable to the side of a spine (drop tank /
  radial booster): has a surface attach node and is **not** a ``SPINE``.  This is
  the role that lets a single radial tank serve as a droppable booster.
* ``SPLITTER`` — one stack node one way, two-or-more the other: a coupler that
  fans a stack out to several mounts (multi-engine mounting).  Never a spine.

KSP geometry convention: a node is ``pos (x,y,z)`` + ``dir (x,y,z)``; ``y`` is the
vertical/stack axis and ``(x,z)`` the radial plane.  A part stacks cleanly only
when its stack nodes line up on one vertical axis.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional


class PartRole(str, Enum):
    SPINE = "spine"
    RADIAL_MOUNT = "radial_mount"
    SPLITTER = "splitter"


class SpineReject(str, Enum):
    """Why a part is *not* a ``SPINE`` — for diagnostics / validation only."""
    OK = "ok"                       # is a spine
    TOO_FEW_NODES = "too_few_nodes"  # < 2 stack nodes (B): can't series-stack
    NON_COLINEAR = "non_colinear"    # stack nodes not on one vertical axis (C)
    COM_OFFSET = "com_offset"        # centre of mass off the stack axis (A)


# Tolerances in metres.  KSP node positions are exact, small, hand-authored
# numbers: a true colinear stack shares (x,z) to within float noise, while every
# real slant/branch offsets by >= ~0.1 m (the slant adapter is 0.625, the
# bicoupler 0.625).  0.05 m cleanly separates "noise" from "intentional offset".
_AXIS_TOL = 0.05    # max spread of stack-node (x,z) positions for "one axis"
_DIR_TOL = 0.10     # max horizontal component of a stack-node dir (≈ vertical)
_COM_TOL = 0.05     # max CoM radial offset sqrt(x^2 + z^2) for "aligned"


def _radial(vec) -> float:
    """Magnitude of a vector's component in the radial (x,z) plane."""
    return math.hypot(vec[0], vec[2])


def _stack_nodes(geom: dict) -> list[dict]:
    return geom.get("stack_nodes", []) or []


def _is_vertical(node: dict) -> bool:
    """True if the node's direction points along ±y (a vertical stack node)."""
    d = node.get("dir") or [0.0, 0.0, 0.0]
    return _radial(d) <= _DIR_TOL and abs(d[1]) > _DIR_TOL


def spine_reject(geom: dict) -> SpineReject:
    """Classify why ``geom`` is or isn't a central-spine-capable part."""
    nodes = _stack_nodes(geom)
    if len(nodes) < 2:
        return SpineReject.TOO_FEW_NODES
    # All stack nodes must be vertical and share one (x,z) axis.
    xs = [n["pos"][0] for n in nodes]
    zs = [n["pos"][2] for n in nodes]
    if (max(xs) - min(xs) > _AXIS_TOL) or (max(zs) - min(zs) > _AXIS_TOL):
        return SpineReject.NON_COLINEAR
    if not all(_is_vertical(n) for n in nodes):
        return SpineReject.NON_COLINEAR
    com = geom.get("com_offset")
    if com is not None and _radial(com) > _COM_TOL:
        return SpineReject.COM_OFFSET
    return SpineReject.OK


def is_splitter(geom: dict) -> bool:
    """One stack node one way, two-or-more the other (bi/tri/quad coupler)."""
    nodes = _stack_nodes(geom)
    up = sum(1 for n in nodes if (n.get("dir") or [0, 0, 0])[1] > _DIR_TOL)
    down = sum(1 for n in nodes if (n.get("dir") or [0, 0, 0])[1] < -_DIR_TOL)
    return min(up, down) >= 1 and max(up, down) >= 2 and (up + down) >= 3


def has_surface_mount(geom: dict) -> bool:
    """Has a surface (radial) attach node, and the part allows surface attach.

    ``attach_rules`` is KSP's ``[stack, srfAttach, allowStack, allowSrfAttach,
    allowCollision]``; index 1 is whether the part itself surface-attaches."""
    if "attach_node" not in geom:
        return False
    rules = geom.get("attach_rules")
    if rules is not None and len(rules) >= 2 and rules[1] == 0:
        return False
    return True


def derive_roles(geom: dict) -> frozenset[PartRole]:
    """The structural roles a part can fill, from its geometry alone."""
    roles: set[PartRole] = set()
    reject = spine_reject(geom)
    if reject is SpineReject.OK:
        roles.add(PartRole.SPINE)
    if is_splitter(geom):
        roles.add(PartRole.SPLITTER)
    # Radial-mount is the side-booster role: surface-attachable and not a spine.
    if reject is not SpineReject.OK and has_surface_mount(geom):
        roles.add(PartRole.RADIAL_MOUNT)
    return frozenset(roles)
