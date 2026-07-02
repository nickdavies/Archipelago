"""Atmospheric-descent physics: entry bleed, the chute drag-stage ladder, and
touchdown-speed bounds.

Pure closed-form math — no AP imports, no part/body imports; callers pass
scalars from ``Body`` and ``Parachute``/``HeatShield`` data.  Shared by the
unified atmospheric landing model and (later) aerobrake-capture.

Every formula errs toward retaining MORE speed / charging MORE burn (the
Golden Rule).  Conservatism is sim-validated by
``scratchpad/sim_staged_landing.py`` (RK4 entry + staged-descent reference);
the constants below are the pessimistic anchors that validation pinned.

The model, in descent order:

1. **Entry bleed** (Allen-Eggers + settle floor): ballistic entry at a fixed
   steep flight-path angle bleeds speed by ``exp(-K)`` with
   ``K = rho·H·A / (2·m·sin(gamma))``; the craft never ends slower than
   ``SETTLE_FACTOR`` times its local terminal velocity (it is still settling
   when it arrives — sim shows up to ~1.25x).
2. **Drag-stage ladder**: each chute contributes a semi-deployed stage
   (engages at its ``minAirPressureToOpen`` pressure altitude) and a
   fully-deployed stage (engages at ``deployAltitude``).  A stage may engage
   only at or below its max safe dynamic pressure, evaluated at the gate's
   local density; arriving faster charges a bridge burn down to the limit.
3. **Touchdown bound**: between gates and to the ground, speed follows the
   exact 1D drag+gravity decay at the (thinnest) upper-gate density — an
   over-prediction of the true arrival speed.
4. **Finish burn**: any touchdown residual above the safe speed is charged as
   a vertical constant-thrust burn including gravity loss at the TWR floor,
   with no credit for chute drag during the burn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# Pessimistic (steep) ballistic entry angle: sin(30°).  Real entries from low
# orbit are far shallower and bleed far more; relaxing this requires flight
# receipts (see plan).  THE golden-rule-sensitive constant.
SIN_ENTRY_GAMMA: float = 0.5

# A craft reaching the surface is still settling toward terminal velocity from
# above; sim grid shows arrival at up to ~1.25x local terminal (shielded
# loadings <= 1.18x).
SETTLE_FACTOR: float = 1.3

# Fraction of a heat shield's drag-cube area credited to entry bleed.  The
# raw cube (e.g. 2.5m shield: 3.79) over-states a real stack — KSP occludes
# the mated faces — and the design doc's in-game measurement of a shield+pod
# stack was CdA ≈ 2.5.  0.65 reproduces that receipt; the pod behind the
# shield gets no separate credit (it is occluded).
SHIELD_BLEED_OCCLUSION: float = 0.65

# Semi-deployed chutes tolerate far more dynamic pressure than full deployment:
# the reefed canopy has ~1/100th the area (semiDeployedDrag≈1 vs 500) so the
# aero load is tiny and the limit is thermal.  Stock mains routinely semi-deploy
# at ~250-300 m/s during reentry.  x5 puts the sea-level semi limit at ~270 m/s
# — enough for a craft sitting at its shield-only terminal to open its mains
# without a spurious bridge burn, still below the shred point.  Like
# SIN_ENTRY_GAMMA this is a calibration constant: a shred-test flight receipt
# can tighten or loosen it.
SEMI_DEPLOY_Q_MULT: float = 5.0

# Candidate engagement altitudes, in scale heights: a stage may engage at any
# altitude at or below its gate, so the walk scans a small ladder of
# candidates and engages at the highest one whose q-limit is satisfied
# (engaging early maximises the remaining braking room).
_ENGAGE_CANDIDATE_SCALE_HEIGHTS: tuple[float, ...] = (3.0, 2.0, 1.5, 1.0, 0.5, 0.25)

# Slab thickness (in scale heights) for the piecewise-constant-density decay
# bound: each slab uses its upper density, so thinner slabs only tighten the
# (still conservative) bound.
_DECAY_SLAB_SCALE_HEIGHTS: float = 0.5

ATM_SEA_LEVEL_KPA: float = 101.325


def local_density(rho0: float, scale_height_m: float, altitude_m: float) -> float:
    """Exponential-atmosphere density at altitude (sea-level value at <=0)."""
    if rho0 <= 0.0 or scale_height_m <= 0.0:
        return 0.0
    if altitude_m <= 0.0:
        return rho0
    return rho0 * math.exp(-altitude_m / scale_height_m)


def pressure_altitude(p0_kpa: float, scale_height_m: float,
                      p_min_atm: float) -> Optional[float]:
    """Altitude where pressure falls to ``p_min_atm`` (a chute's semi-deploy
    gate), or None when the body's sea-level pressure is already below it —
    the chute cannot open anywhere on that body."""
    if p0_kpa <= 0.0 or scale_height_m <= 0.0 or p_min_atm <= 0.0:
        return None
    p0_atm = p0_kpa / ATM_SEA_LEVEL_KPA
    if p0_atm < p_min_atm:
        return None
    return scale_height_m * math.log(p0_atm / p_min_atm)


def local_terminal_velocity(mass_t: float, gravity: float, rho: float,
                            drag_area: float) -> float:
    """Terminal velocity at a given local density (same effective-area
    convention as ``rocket_math.terminal_velocity``)."""
    if rho <= 0.0 or drag_area <= 0.0:
        return math.inf
    return math.sqrt(2.0 * mass_t * 1000.0 * gravity / (rho * drag_area))


def terminal_limited_count(payload_t: float, chute_drag: float,
                           chute_mass_t: float, bleed_area: float, rho0: float,
                           gravity: float, v_safe: float,
                           is_radial: bool) -> float:
    """Continuous chute count at which sea-level terminal velocity equals
    ``v_safe`` — the closed-form seed for the lightest passive count.

    Solves ``2·(payload + μn)·g = v_safe²·ρ₀·(bleed + chute_drag·s(n))`` where
    the per-count drag scaling ``s(n)`` is ``n`` (inline, α=1) or ``n^1.5``
    (radial in symmetry, α=1.5 — the continuous approximation of
    ``_radial_drag_multiplier``; the caller confirms on the real piecewise
    multiplier at the integer neighbours).  Returns the smallest positive real
    ``n`` that reaches ``v_safe``, ``0.0`` if the payload alone is already safe,
    or ``inf`` if no finite count can (the linear inline case where chute drag
    can't overcome the added mass).  Terminal ∝ n^(−α/2) is monotone
    decreasing, so this crossing is unique."""
    p, mu, b = payload_t * 1000.0, chute_mass_t * 1000.0, bleed_area
    k = v_safe * v_safe * rho0
    # Passive already with zero chutes?
    if k * b >= 2.0 * p * gravity:
        return 0.0
    if not is_radial:
        # Linear: 2(p+μn)g = k(b + drag·n) → n·(kd − 2μg) = 2pg − kb
        denom = k * chute_drag - 2.0 * mu * gravity
        num = 2.0 * p * gravity - k * b
        if denom <= 0.0:
            return math.inf  # added mass outpaces added drag — never safe
        return num / denom
    # Radial: k·drag·x³ − 2μg·x² + (kb − 2pg) = 0, x = √n.  Solve for the
    # largest positive root by Newton from a drag-dominated start.
    c3 = k * chute_drag
    c2 = 2.0 * mu * gravity
    c0 = k * b - 2.0 * p * gravity  # < 0 here (not passive at n=0)
    if c3 <= 0.0:
        return math.inf
    x = max((-c0 / c3) ** (1.0 / 3.0), (c2 / c3) ** 0.5, 1e-3)
    for _ in range(40):
        f = c3 * x ** 3 - c2 * x * x + c0
        fp = 3.0 * c3 * x * x - 2.0 * c2 * x
        if fp == 0.0:
            break
        step = f / fp
        x -= step
        if x <= 0.0:
            x = 1e-3
        if abs(step) < 1e-6:
            break
    return x * x


def bleed_speed(v_entry: float, mass_t: float, drag_area: float, rho: float,
                scale_height_m: float, gravity: float,
                sin_gamma: float = SIN_ENTRY_GAMMA) -> float:
    """Speed retained by ballistic entry once the craft is down where the
    local density is ``rho`` (Allen-Eggers), floored at the settle factor
    times local terminal velocity.  With no drag area there is no credit and
    no floor: the entry speed is retained outright."""
    if drag_area <= 0.0 or rho <= 0.0:
        return v_entry
    m_kg = mass_t * 1000.0
    k = rho * scale_height_m * drag_area / (2.0 * m_kg * sin_gamma)
    decayed = v_entry * math.exp(-k)
    floor = SETTLE_FACTOR * local_terminal_velocity(mass_t, gravity, rho, drag_area)
    return max(decayed, floor)


def drag_decayed_speed(v0: float, mass_t: float, drag_area: float,
                       rho_upper: float, dist_m: float, gravity: float) -> float:
    """Speed after descending ``dist_m`` under drag+gravity, using the exact
    constant-density solution at the UPPER point's density (the thinnest air
    on the way down, so the result only over-predicts):

        w = v² − vt²  decays as  exp(−dist · rho · A / m)

    Handles v0 below terminal too (w < 0 relaxes toward terminal from below,
    which at the thin upper density is again an over-prediction)."""
    if drag_area <= 0.0 or rho_upper <= 0.0:
        # No drag: energy gain only.
        return math.sqrt(v0 * v0 + 2.0 * gravity * max(dist_m, 0.0))
    m_kg = mass_t * 1000.0
    vt2 = 2.0 * m_kg * gravity / (rho_upper * drag_area)
    w0 = v0 * v0 - vt2
    w = w0 * math.exp(-max(dist_m, 0.0) * rho_upper * drag_area / m_kg)
    return math.sqrt(max(vt2 + w, 0.0))


def max_engage_speed(q_safe_kpa: float, rho: float) -> float:
    """Fastest a drag stage may engage at local density ``rho`` without
    exceeding its dynamic-pressure limit."""
    if rho <= 0.0:
        return math.inf
    return math.sqrt(2.0 * q_safe_kpa * 1000.0 / rho)


def decayed_speed_over(v0: float, mass_t: float, drag_area: float,
                       rho0: float, scale_height_m: float,
                       h_from: float, h_to: float, gravity: float) -> float:
    """Speed after descending from ``h_from`` to ``h_to`` under drag+gravity,
    chaining the exact constant-density solution over slabs of
    ``_DECAY_SLAB_SCALE_HEIGHTS`` scale heights, each at its upper (thinnest)
    density.  A single-span bound is uselessly loose across multiple scale
    heights; slabbing keeps it conservative AND tight (the craft tracks local
    terminal velocity down the density gradient)."""
    if h_from <= h_to:
        return v0
    slab = max(scale_height_m * _DECAY_SLAB_SCALE_HEIGHTS, 1.0)
    v = v0
    h = h_from
    while h > h_to:
        h_next = max(h - slab, h_to)
        v = drag_decayed_speed(v, mass_t, drag_area,
                               local_density(rho0, scale_height_m, h),
                               h - h_next, gravity)
        h = h_next
    return v


def burn_dv_for(delta_speed: float, twr: float, sin_gamma: float) -> float:
    """Delta-v charged to shed ``delta_speed`` by a constant-thrust retro burn
    on a path inclined at ``sin_gamma``, including gravity loss:

        dv = delta_speed / (1 − sin_gamma / twr)

    Exact for constant thrust and angle; vertical (sin_gamma=1) reduces to the
    familiar suicide-burn ``twr/(twr−1)`` factor.  A TWR at or below
    ``sin_gamma`` cannot brake at all — the burn is infeasible (inf)."""
    if delta_speed <= 0.0:
        return 0.0
    if twr <= sin_gamma:
        return math.inf
    return delta_speed / (1.0 - sin_gamma / twr)


@dataclass(frozen=True)
class DragStage:
    """One deployable drag increment in the descent ladder.

    ``gate_altitude_m`` is the HIGHEST altitude the stage may engage at (a
    chute's pressure gate / deploy altitude); ``floor_altitude_m`` the lowest
    useful one (a semi stage engaged below its own full-deploy altitude is
    pointless).  The walk picks the best engagement altitude in between.
    """
    label: str
    added_drag_area: float   # effective area ADDED when this stage engages
    gate_altitude_m: float
    q_safe_kpa: float        # max dynamic pressure at engagement
    floor_altitude_m: float = 0.0


def chute_stages(full_area: float, semi_area: float, q_safe_kpa: float,
                 deploy_altitude_m: float, min_pressure_atm: float,
                 p0_kpa: float, scale_height_m: float,
                 label: str = "chute") -> Optional[tuple[DragStage, ...]]:
    """Build the ladder stages one chute set contributes: a semi-deployed
    stage engageable anywhere between its pressure gate and its full-deploy
    altitude (with the higher semi q envelope), and the fully-deployed stage
    at ``deployAltitude``.  Areas are the TOTALS for the whole set (caller
    applies chute count / radial-symmetry scaling).  Returns None when the
    body's surface pressure cannot open the chute at all."""
    h_semi = pressure_altitude(p0_kpa, scale_height_m, min_pressure_atm)
    if h_semi is None:
        return None
    if h_semi > deploy_altitude_m and semi_area > 0.0:
        # The q-limit governs the RISKY initial opening — that's the semi stage
        # (reefed, higher tolerance ×SEMI_DEPLOY_Q_MULT).  Reefing out to full
        # at ``deployAltitude`` is a controlled transition on an already-open
        # canopy that has slowed the craft, so the full stage is NOT re-gated
        # (q_safe=inf): no spurious bridge burn between semi and full.
        return (
            DragStage(f"{label}:semi", semi_area, h_semi,
                      q_safe_kpa * SEMI_DEPLOY_Q_MULT,
                      floor_altitude_m=deploy_altitude_m),
            DragStage(f"{label}:full", full_area - semi_area,
                      deploy_altitude_m, math.inf),
        )
    # Pressure gate sits at/below the full-deploy altitude (thin atmospheres):
    # the chute goes straight to full deployment when it opens — this single
    # stage IS the initial opening, so it keeps the real q-limit.
    return (DragStage(f"{label}:full", full_area,
                      min(h_semi, deploy_altitude_m), q_safe_kpa),)


@dataclass(frozen=True)
class Engagement:
    """One drag stage's resolved engagement, for tracing/validation/display."""
    label: str
    altitude_m: float
    arrival_speed: float   # conservative arrival estimate at the engagement
    speed_after: float     # min(arrival, q-limit) — post-bridge-burn speed


@dataclass(frozen=True)
class DescentPlan:
    """Outcome of a staged-descent evaluation.  Delta-v values are UNCAPPED —
    the caller caps the total at the body's full propulsive-descent figure and
    applies difficulty margins."""
    bridge_dv: float        # burns spent reaching stage engagement speeds
    finish_dv: float        # final touchdown burn (0 when drag alone lands safely)
    touchdown_speed: float  # residual speed at the ground before any finish burn
    engagements: tuple[Engagement, ...] = ()

    @property
    def total_burn_dv(self) -> float:
        return self.bridge_dv + self.finish_dv

    @property
    def requires_burn(self) -> bool:
        return self.total_burn_dv > 0.0


def _engage_candidates(stage: DragStage, scale_height_m: float,
                       h_below: float) -> list[float]:
    """Altitudes at which a stage could engage: its gate, plus a ladder of
    scale-height multiples below it — a player deploys wherever the q-limit
    allows, not blindly at the gate.  ``h_below`` additionally caps candidates
    to at/below the previous engagement (stages engage in descent order).

    The lower bound is the stage's own ``floor_altitude_m`` (a semi stage's
    floor is its full-deploy altitude; a full/single stage's is the ground).
    A floor of 0 is NOT added as a candidate — engaging with no braking room
    left is a slam the greedy gap objective would otherwise pick; the
    scale-height rungs keep a natural minimum altitude above it."""
    top = min(max(stage.gate_altitude_m, 0.0), h_below)
    floor = max(stage.floor_altitude_m, 0.0)
    cands = {top}
    for k in _ENGAGE_CANDIDATE_SCALE_HEIGHTS:
        h = k * scale_height_m
        if floor <= h <= top:
            cands.add(h)
    if 0.0 < floor < top:
        cands.add(floor)
    return sorted(cands, reverse=True)


def staged_descent(v_entry: float, mass_t: float, bleed_area: float,
                   stages: Sequence[DragStage], rho0: float,
                   scale_height_m: float, gravity: float, twr: float,
                   max_safe_touchdown: float,
                   sin_gamma: float = SIN_ENTRY_GAMMA,
                   jettison_mass_t: float = 0.0) -> DescentPlan:
    """Walk the descent ladder and price the propulsive shortfall.

    ``bleed_area`` is the pre-chute drag area (occluded shield credit — see
    ``SHIELD_BLEED_OCCLUSION``); ``stages`` are the deployable increments,
    walked in descending gate order.  Each stage engages at the highest
    candidate altitude whose q-limit its arrival speed satisfies; if none
    qualifies, the cheapest bridge burn (smallest gap over the candidates) is
    charged.  ``twr`` is the local thrust-to-weight the landing stage is
    required to have; burns before any stage engages are charged at the entry
    angle, burns after at vertical (chute descent).

    ``jettison_mass_t`` is dropped the instant the first chute engages — a rigid
    ablative heat shield is staged off before the parachutes deploy, so it must
    NOT weigh down the terminal-velocity / touchdown calc (its drag still
    counted the whole bleed).  Pass 0 for a shield that stays on (the inflatable
    used as the bleed device).

    The result's dv is uncapped and may be ``inf`` when the TWR cannot brake the
    required gap (caller treats that mix as infeasible)."""
    ordered = sorted(stages, key=lambda s: s.gate_altitude_m, reverse=True)
    bridge_dv = 0.0
    area = bleed_area
    engaged = False
    v_prev = 0.0
    h_prev = math.inf
    engagements: list[Engagement] = []
    # Mass carried once chutes are out (shield jettisoned); the bleed uses the
    # full entry mass, everything after uses the lighter post-jettison mass.
    chute_mass_t = max(mass_t - jettison_mass_t, 1e-6)

    def arrival(h: float) -> float:
        rho_h = local_density(rho0, scale_height_m, h)
        if not engaged:
            return bleed_speed(v_entry, mass_t, area, rho_h,
                               scale_height_m, gravity, sin_gamma)
        return decayed_speed_over(v_prev, chute_mass_t, area, rho0,
                                  scale_height_m, h_prev, h, gravity)

    def burn_gamma(v_arrive: float) -> float:
        """Path angle for a bridge burn: near entry speed the trajectory is
        still at the entry angle; as speed dies the descent goes vertical.
        Interpolating on retained speed is monotone and errs vertical
        (bigger gravity loss) exactly when speeds are low."""
        frac = min(v_arrive / v_entry, 1.0) if v_entry > 0.0 else 0.0
        return 1.0 - (1.0 - sin_gamma) * frac

    for stage in ordered:
        best: Optional[tuple[float, float, float]] = None  # (gap, -h, v_capped)
        for h in _engage_candidates(stage, scale_height_m, h_prev):
            v_arrive = arrival(h)
            v_max = max_engage_speed(stage.q_safe_kpa,
                                     local_density(rho0, scale_height_m, h))
            gap = max(v_arrive - v_max, 0.0)
            key = (gap, -h)
            if best is None or key < best[:2]:
                best = (gap, -h, min(v_arrive, v_max))
            if gap == 0.0:
                break  # candidates are scanned top-down: first free one wins
        assert best is not None
        gap, neg_h, v_engage = best
        if gap > 0.0:
            bridge_dv += burn_dv_for(gap, twr, burn_gamma(v_engage + gap))
        engagements.append(Engagement(stage.label, -neg_h,
                                      v_engage + gap, v_engage))
        area += stage.added_drag_area
        engaged = True
        v_prev, h_prev = v_engage, -neg_h

    if engaged:
        # Touchdown speed is measured AT the ground, so it settles toward the
        # sea-level terminal velocity (densest air), not the thinner air at the
        # last deploy gate.  Evaluate the final descent from the last gate to
        # the ground at sea-level density: accurate at touchdown (converges to
        # ground terminal when there's settling room), still > terminal when the
        # gate is low and fast (thin atmo).  Using the gate-altitude density
        # here instead would leave a light craft a fraction above the true
        # ground terminal — enough to spuriously fail the safe-speed check.
        touchdown = drag_decayed_speed(v_prev, chute_mass_t, area, rho0,
                                       h_prev, gravity)
    else:
        # No chute engaged — shield stays on, full mass bleeds to the ground.
        touchdown = bleed_speed(v_entry, mass_t, area, rho0, scale_height_m,
                                gravity, sin_gamma)

    finish_dv = 0.0
    if touchdown > max_safe_touchdown:
        finish_dv = burn_dv_for(touchdown - max_safe_touchdown, twr, 1.0)

    return DescentPlan(bridge_dv=bridge_dv, finish_dv=finish_dv,
                       touchdown_speed=touchdown,
                       engagements=tuple(engagements))
