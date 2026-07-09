"""Generate worlds/ksp1/data/lifter_chains/ — offline lifter chains + bindings.

Chains are ordered part-adds grown per ascent-regime CLUSTER against the
cluster's hardest member (highest effective home-ascent dv), using the exact
production build path: ``_pre_pass`` part-set -> EquipmentFlags,
``_ascent_stage_kwargs`` (shared with ``_evaluate_profile``), and
``find_optimal_multistage_ascent`` with ESCALATED bounds everywhere (offline
cost is free — "optimal the first time"; the runtime never re-searches).

Bindings are per HOME BODY: for each (pack set, physics profile, required_dv
variant), walk the cluster's chains against a payload-threshold ladder and
record (threshold, prefix_len, launch_mass, stage manifests).  Runtime serves
these rows via worlds/ksp1/lifter_binding.py without any optimizer calls.

Usage (from Archipelago/):
    python -m worlds.ksp1.scripts.generate_lifter_chains            # stdout summary
    python -m worlds.ksp1.scripts.generate_lifter_chains --write    # regenerate data
    python -m worlds.ksp1.scripts.generate_lifter_chains --check    # drift gate (structural + digest + sampled re-bind)
    python -m worlds.ksp1.scripts.generate_lifter_chains --check --full   # full regen compare (release/nightly)

Options: --homes kerbin,eve  --n-chains 100  --release (grow 5xN, keep the N
most mutually different)  --jobs N
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    __package__ = "worlds.ksp1.scripts"  # noqa: A001

from .. import capability as cap
from ..bodies import (
    ALL_BODIES, BodyName, DIFFICULTY_PROFILES, EdgeType, MissionBuilder,
    progressive_launch_pad_caps_for,
)
from ..lifter_binding import (
    AscentConstraints, NAME_TO_OFFSET, home_ascent_dv_variants,
)
from ..parts import CapabilityFlag, Decoupler, Engine, FuelTank, SolidBooster
from ..parts.manager import part_manager_for
from ..parts import packs as packs_mod
from ..rocket_math import (
    ESCALATED_BOOSTER_COUNTS, ESCALATED_MAX_ASCENT_STAGES,
    ESCALATED_MAX_ENG_PER_COL, _F4_DV_SPLITS, find_optimal_multistage_ascent,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lifter_chains")

# ---------------------------------------------------------------------------
# Clusters — hand-curated from the cross-body reuse experiment (2026-07-08):
# chains transfer within a regime (identical prefixes for the vacuum
# clusters, Kerbin<->Laythe mutually interchangeable) and fail across regime
# boundaries (Duna->Kerbin, Mun->Tylo).  Membership is a deliberate design
# constant; ``_assert_cluster_envelopes`` keeps it honest against body/dv
# drift.  The rep (growth target) is always the max-effective-dv member.
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, tuple[BodyName, ...]] = {
    "vac_micro": (BodyName.GILLY, BodyName.POL, BodyName.MINMUS, BodyName.BOP),
    "vac_low": (BodyName.IKE, BodyName.DRES, BodyName.MUN, BodyName.EELOO,
                BodyName.VALL, BodyName.MOHO),
    "tylo": (BodyName.TYLO,),
    "atmo_thin": (BodyName.DUNA,),
    "atmo_thick": (BodyName.LAYTHE, BodyName.KERBIN),
    "eve": (BodyName.EVE,),
}
HOME_CLUSTER: dict[BodyName, str] = {
    b: c for c, members in CLUSTERS.items() for b in members
}

# Physics profiles bound per home.  "zero" (retired insane) is deliberately
# absent; Eve-as-home is expert-only (world.py gate), so only "small" exists.
_ALL_PHYS = ("generous", "comfortable", "small")
_EVE_PHYS = ("small",)

# Growth runs under the strictest profile a cluster's homes can be played at
# (highest TWR floors + margins) so chain ordering is useful everywhere.
_GROWTH_PHYS_DEFAULT = "generous"
_GROWTH_PHYS = {"eve": "small"}

PACK_KEYS: tuple[tuple[str, ...], ...] = (
    (packs_mod.STOCK,),
    tuple(sorted((packs_mod.STOCK,) + packs_mod.OPTIONAL_PACKS)),
)

# Payload-threshold ladder: log-spaced at ~1.7x from 0.6t up to the ceiling
# (micro-body ceilings reach hundreds of kt — the ladder must extend all the
# way or mid-range payloads get served by absurdly oversized ceiling builds),
# augmented with the home's finite pad caps (keeps the served launch-mass
# overestimate tight exactly where the pad-cap check bites).
_LADDER_BASE_T = 0.6
_LADDER_RATIO = 2.0

# Chains per cluster: full menu where homes are common or the regime is hard
# (variance matters most there), a leaner one for the rarely-picked vacuum
# clusters — their chains are near-interchangeable anyway (reuse experiment:
# identical prefixes across the vacuum clusters).
_N_CHAINS_LEAN = 32
_LEAN_CLUSTERS = frozenset({"vac_micro", "vac_low"})

_BODY = {b.name: b for b in ALL_BODIES}

_ESC_KWARGS = dict(
    max_ascent_stages=ESCALATED_MAX_ASCENT_STAGES,
    booster_counts=ESCALATED_BOOSTER_COUNTS,
    max_eng_per_col=ESCALATED_MAX_ENG_PER_COL,
    parallel_substages=True,
)

_POOL_MISC_FLAGS = frozenset({
    CapabilityFlag.FUEL_LINE, CapabilityFlag.RCS,
    CapabilityFlag.REACTION_WHEEL, CapabilityFlag.AERO_CONTROL,
    CapabilityFlag.MULTI_MOUNT,
})


def lifter_pool(pack_key: tuple[str, ...]) -> list[str]:
    """AP-grantable parts a lifter build can consume, in registry order:
    propulsion/staging by type, plus the MiscEquipment whose capability flags
    the ascent kwargs read (fuel line, RCS, wheels, fins, multi-mounts) and
    monoprop tanks (the RCS attitude bundle needs a monopropellant source)."""
    pm = part_manager_for(frozenset(pack_key))
    pool: list[str] = []
    for name, parts in pm.parts.items():
        for p in parts:
            if isinstance(p, (Engine, FuelTank, SolidBooster, Decoupler)):
                pool.append(name)
                break
            provides = getattr(p, "provides", None)
            if provides and provides & _POOL_MISC_FLAGS:
                pool.append(name)
                break
    return pool


class AscentBuilder:
    """Bind-time build context for one (home, phys).  Reproduces the group-0
    locals of ``_evaluate_profile`` for a bare home-ascent group and calls the
    shared ``_ascent_stage_kwargs`` — the same parameter set the live path
    hands ``find_optimal_multistage_ascent``."""

    def __init__(self, home: BodyName, phys: str):
        self.body = _BODY[home]
        self.diff = DIFFICULTY_PROFILES[phys]
        mb = MissionBuilder(home=home)
        self.dv_variants = home_ascent_dv_variants(mb, self.diff)
        edge = self._home_ascent_edge(mb, home)
        self.edge = edge
        self.in_atmo = edge.edge_type == EdgeType.ATMOSPHERIC_ASCENT
        floor = (self.diff.min_twr_atmo if self.in_atmo
                 else self.diff.min_twr_vac)
        self.min_twr = max(edge.min_twr, floor)
        self.req_throttle = edge.requires_throttleable
        self.requires_attitude = edge.requires_attitude_control
        # Bind under the SAME search bounds the runtime uses for this home:
        # escalated only for the allowlisted home-ascent edges (Eve), standard
        # elsewhere.  Binding wider than the runtime would mint architectures
        # the live rebuild and the post_fill cross-check can't reproduce
        # (e.g. 6 engines/column when standard allows 4) -> served-but-
        # unbuildable rungs.  The offline budget buys more SEARCH TIME, not
        # wider bounds than runtime.
        from ..capability import _escalated_home_edges
        self.escalated = (edge.body, edge.edge_type) in _escalated_home_edges()
        self._esc_kwargs = dict(_ESC_KWARGS) if self.escalated else {}
        self.n_builds = 0

    @staticmethod
    def _home_ascent_edge(mb: MissionBuilder, home: BodyName):
        from ..bodies import MissionType
        for mt in (MissionType.ORBIT, MissionType.RETURN):
            for profile in mb.profiles_for(home, mt) or ():
                for e in profile:
                    if (e.body == home and e.edge_type in
                            (EdgeType.ATMOSPHERIC_ASCENT,
                             EdgeType.VACUUM_ASCENT)):
                        return e
        raise ValueError(f"no home ascent edge for {home}")

    def build(self, kit: frozenset, required_dv: float,
              payload_t: float, guide=None) -> Optional[list]:
        """One real multistage build (this home's bounds; optionally pinned to
        a guide architecture); returns bottom->top stages with the control
        surcharges attached (prune_chain reads the attached names so control
        parts survive into the stored chain) or None.

        Delegates to ``capability._guided_ascent_build`` — the SAME producer the
        runtime SERVED rebuild calls, so a bound rung is byte-identical to what
        the live path reconstructs (no bind/serve drift possible)."""
        self.n_builds += 1
        return cap._guided_ascent_build(
            kit, self.body,
            in_atmo=self.in_atmo, min_twr=self.min_twr,
            req_throttle=self.req_throttle,
            requires_attitude=self.requires_attitude,
            srb_needs_rcs=True, run_parallel=True,
            required_dv=required_dv, payload_mass=payload_t,
            guide=guide, esc_kwargs=self._esc_kwargs)

    def canonical_hint(self, kit: frozenset, required_dv: float,
                       payload_t: float, stages: list) -> tuple[tuple, float]:
        """Canonicalize a full-search winner into (hint_tuple, launch_mass):
        re-run the guided build per candidate dv-split and keep the argmin.
        Stored mass and hint are self-consistent with the runtime rebuild by
        construction (same guided computation, same prefix kit)."""
        per_stage_names = tuple(
            (s.n_boosters, s.booster_engines, s.engine_name) for s in stages)
        best_split, best_mass = None, float("inf")
        for sp in _F4_DV_SPLITS[len(stages)]:
            r = self.build(kit, required_dv, payload_t,
                           guide=(sp, per_stage_names))
            if r is not None and r[0].stage_mass_wet < best_mass:
                best_mass = r[0].stage_mass_wet
                best_split = sp
        if best_split is None:
            return None, 0.0
        hint = (best_split,
                tuple((nb, be, NAME_TO_OFFSET[nm])
                      for nb, be, nm in per_stage_names))
        return hint, best_mass

    def feasible(self, kit: frozenset, required_dv: float,
                 payload_t: float) -> bool:
        return self.build(kit, required_dv, payload_t) is not None

    def max_lift(self, kit: frozenset, required_dv: float,
                 lo: float = 0.05, iters: int = 12) -> float:
        if not self.feasible(kit, required_dv, lo):
            return 0.0
        hi = lo
        while hi < 1e6 and self.feasible(kit, required_dv, hi * 4):
            hi *= 4
        lo2, hi2 = hi, hi * 4
        for _ in range(iters):
            mid = (lo2 * hi2) ** 0.5
            if self.feasible(kit, required_dv, mid):
                lo2 = mid
            else:
                hi2 = mid
        return lo2

    def constraints(self) -> AscentConstraints:
        return AscentConstraints(
            in_atmosphere=self.in_atmo,
            min_twr_liftoff=round(self.min_twr, 4),
            requires_throttleable=self.req_throttle,
            srb_needs_rcs=True,
            needs_heat_shield=False,
            pad_altitude_m=self.body.pad_altitude_m,
        )


# ---------------------------------------------------------------------------
# Chain growth (per cluster, against the rep body's hardest dv variant)
# ---------------------------------------------------------------------------

def cluster_rep(cluster: str) -> BodyName:
    phys = _GROWTH_PHYS.get(cluster, _GROWTH_PHYS_DEFAULT)
    diff = DIFFICULTY_PROFILES[phys]

    def eff(b: BodyName) -> float:
        from ..bodies import effective_dv
        return effective_dv(_BODY[b].home_pad_ascent_dv(), diff)
    return max(CLUSTERS[cluster], key=eff)


def assert_cluster_envelopes() -> list[str]:
    """The cluster map is a hand-curated design constant (reuse experiment
    2026-07-08); these invariants catch body/dv-model drift that would
    invalidate it: regime uniformity (atmo flag), and every member's ascent
    regime within the measured-transferable envelope of its rep (the reuse
    experiment showed harder-regime chains bind easier regimes, so the rep
    must dominate on effective dv)."""
    errors: list[str] = []
    for cluster, members in CLUSTERS.items():
        atmos = {_BODY[b].has_atmosphere for b in members}
        if len(atmos) != 1:
            errors.append(f"{cluster}: mixed atmosphere regimes")
        rep = cluster_rep(cluster)
        phys = _GROWTH_PHYS.get(cluster, _GROWTH_PHYS_DEFAULT)
        diff = DIFFICULTY_PROFILES[phys]
        from ..bodies import effective_dv
        rep_dv = effective_dv(_BODY[rep].home_pad_ascent_dv(), diff)
        for b in members:
            if effective_dv(_BODY[b].home_pad_ascent_dv(), diff) > rep_dv:
                errors.append(f"{cluster}: {b.value} exceeds rep {rep.value}")
    return errors


def grow_chain(builder: AscentBuilder, pool: list[str], pack_key: tuple,
               seed_key: str, thresholds: tuple[float, ...], grow_dv: float,
               step_budget: int = 160) -> Optional[list[str]]:
    """Guided-random growth: at each infeasible threshold, trial up to 6
    random candidate parts; prefer one that closes the threshold, else the
    one whose kit gets closest (probe at T/2, T/4, T/8).  Weighted toward
    thrust/fuel like the bumper's tier-1 axes.  Returns None if the budget
    runs out (caller regrows with a bumped sub-seed)."""
    pm_weight = {Engine: 3, FuelTank: 3, SolidBooster: 2}
    pmgr = part_manager_for(frozenset(pack_key))
    weighted: list[str] = []
    for name in pool:
        w = 1
        for p in pmgr.parts.get(name, ()):
            w = max(w, pm_weight.get(type(p), 1))
        weighted.extend([name] * w)
    rng = random.Random(seed_key)
    kit: frozenset = frozenset()
    chain: list[str] = []
    steps = 0
    for T in thresholds:
        while not builder.feasible(kit, grow_dv, T):
            steps += 1
            remaining = set(pool) - set(chain)
            if steps > step_budget or not remaining:
                return None
            cands: list[str] = []
            tries = 0
            while len(cands) < 6 and tries < 400:
                tries += 1
                c = rng.choice(weighted)
                if c in remaining and c not in cands:
                    cands.append(c)
            if not cands:
                cands = rng.sample(sorted(remaining),
                                   min(6, len(remaining)))
            feas = [c for c in cands
                    if builder.feasible(kit | {c}, grow_dv, T)]
            if feas:
                pick = rng.choice(feas)
            else:
                best_score, best = -1.0, []
                for c in cands:
                    score = 0.0
                    for frac in (0.5, 0.25, 0.125):
                        if builder.feasible(kit | {c}, grow_dv, T * frac):
                            score = frac
                            break
                    if score > best_score:
                        best_score, best = score, [c]
                    elif score == best_score:
                        best.append(c)
                pick = rng.choice(best)
            kit = kit | {pick}
            chain.append(pick)
    return chain


def prune_chain(builder: AscentBuilder, chain: list[str],
                thresholds: tuple[float, ...], grow_dv: float) -> list[str]:
    """Drop parts that no rung's build manifest uses, then verify the pruned
    chain still binds every threshold at a prefix no longer than before;
    revert wholesale on any regression.  Kills the pathological no-op tails
    the guided-random growth leaves near the ceiling."""
    used: set[str] = set()
    marks: dict[float, int] = {}
    kit: frozenset = frozenset()
    idx = 0
    for T in thresholds:
        while not builder.feasible(kit, grow_dv, T) and idx < len(chain):
            kit = kit | {chain[idx]}
            idx += 1
        if not builder.feasible(kit, grow_dv, T):
            return chain
        marks[T] = idx
        for sr in builder.build(kit, grow_dv, T) or ():
            if sr.engine_name and sr.engine_name != "none":
                used.add(sr.engine_name)
            used.update(nm for _, nm in sr.tank_manifest)
            used.update(nm for _, nm in sr.equipment)
            if sr.heat_shield_name:
                used.add(sr.heat_shield_name)
    pruned = [p for p in chain if p in used]
    if pruned == chain:
        return chain
    kit = frozenset()
    idx = 0
    for T in thresholds:
        while not builder.feasible(kit, grow_dv, T) and idx < len(pruned):
            kit = kit | {pruned[idx]}
            idx += 1
        if not builder.feasible(kit, grow_dv, T) or idx > marks[T]:
            return chain
    return pruned


def select_diverse(chains: list[list[str]], n: int,
                   seed_key: str) -> list[list[str]]:
    """Greedy max-min Jaccard-distance selection (release mode grows 5xN and
    keeps the N most mutually different)."""
    if len(chains) <= n:
        return chains
    rng = random.Random(seed_key)
    sets = [frozenset(c) for c in chains]
    picked = [rng.randrange(len(chains))]
    while len(picked) < n:
        best_i, best_d = -1, -1.0
        for i in range(len(chains)):
            if i in picked:
                continue
            d = min(1.0 - len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                    for j in picked)
            if d > best_d:
                best_d, best_i = d, i
        picked.append(best_i)
    return [chains[i] for i in sorted(picked)]


# ---------------------------------------------------------------------------
# Binding (per home body)
# ---------------------------------------------------------------------------

def ladder_for(home: BodyName, ceiling: float) -> tuple[float, ...]:
    rungs = set()
    t = _LADDER_BASE_T
    while t <= ceiling:
        rungs.add(round(t, 1))
        t *= _LADDER_RATIO
    # Pad caps bound LAUNCH mass, not payload; a payload rung near the cap
    # still helps keep the overestimate tight below it.
    rungs.update(c for c in progressive_launch_pad_caps_for(home)
                 if math.isfinite(c) and c <= ceiling)
    rungs.add(round(ceiling, 3))
    return tuple(sorted(rungs))


# ---------------------------------------------------------------------------
# Digest — model inputs the tables derive from; --check recomputes cheaply.
# ---------------------------------------------------------------------------

def _canon(obj):
    """Hash-stable canonical form: set/frozenset iteration order and part
    dataclass reprs are PYTHONHASHSEED-dependent, so everything is sorted /
    field-ordered before hashing."""
    import dataclasses
    if isinstance(obj, (frozenset, set)):
        return tuple(sorted(repr(_canon(x)) for x in obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return (type(obj).__name__,
                tuple((f.name, _canon(getattr(obj, f.name)))
                      for f in dataclasses.fields(obj)))
    if isinstance(obj, dict):
        return tuple(sorted((repr(k), repr(_canon(v)))
                            for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_canon(x) for x in obj)
    return obj


def _binding_source_fingerprint() -> list[str]:
    """Source text of everything the binding physics depends on.  Folding this
    into the digest makes it a COMPLETE currency fingerprint: a digest match
    proves the checked-in data is what the current code+model would generate,
    so ``--check`` needs no physics at all (Option A).  Conservative by design
    — any edit to these flips the digest and asks for a regen, even if the
    physics happens to be unchanged (safe: false positives, never a stale
    table passing).  ``--check --full`` re-binds for real as the deep gate."""
    import inspect
    from .. import capability, lifter_binding
    from .. import rocket_math
    srcs = [
        inspect.getsource(rocket_math),      # optimizer (find_optimal_*, splits, floors)
        inspect.getsource(sys.modules[__name__]),  # this generator
        # From lifter_binding, only the dv-variant enumerator affects the
        # GENERATED data.  consult / reconstruct_stage / BoundLifterTable are
        # runtime consumers — hashing the whole module would flag a "data
        # stale" regen for a runtime-only edit (e.g. a consult tweak) that
        # can't change a single stored row.
        inspect.getsource(lifter_binding.home_ascent_dv_variants),
    ]
    # capability.py churns for unrelated reasons, so hash only the exact
    # bind-path helpers the generator calls.  ``_guided_ascent_build`` /
    # ``_ascent_kwargs_for_kit`` / ``_kit_ascent_flags`` ARE the bind path now
    # (AscentBuilder.build delegates to them, shared with the runtime rebuild),
    # so they must be in the fingerprint or a physics edit there could slip a
    # stale table past ``--check``.
    for fn in (capability._guided_ascent_build,
               capability._ascent_kwargs_for_kit,
               capability._kit_ascent_flags.__wrapped__,
               capability._ascent_stage_kwargs,
               capability._parallel_staging_inputs,
               capability._pre_pass,
               capability._filter_engines_for_ion,
               capability._attitude_bundle_for_stage,
               capability._rcs_bundle,
               capability._own_stage):
        srcs.append(inspect.getsource(fn))
    return srcs


def model_digest() -> str:
    h = hashlib.sha256()

    def feed(*items):
        for it in items:
            h.update(repr(_canon(it)).encode())
    from ..capability import _escalated_home_edges
    feed(sorted(CLUSTERS.items()), _LADDER_BASE_T, _LADDER_RATIO, PACK_KEYS,
         _ESC_KWARGS, _ALL_PHYS, _N_CHAINS_LEAN, sorted(_LEAN_CLUSTERS),
         sorted((b.value, e.value) for b, e in _escalated_home_edges()))
    for name, prof in sorted(DIFFICULTY_PROFILES.items()):
        feed(name, prof)
    for b in ALL_BODIES:
        if not getattr(b, "can_land", False):
            continue
        feed(b.name, b.surface_gravity, b.has_atmosphere,
             b.atm_scale_height_m, b.safe_altitude_km, b.pad_altitude_m,
             round(b.home_pad_ascent_dv(), 3),
             progressive_launch_pad_caps_for(b.name))
    pm = part_manager_for(frozenset(PACK_KEYS[-1]))
    for name in lifter_pool(PACK_KEYS[-1]):
        for p in pm.parts[name]:
            feed(name, p)
    # Source of the binding code — the other half of the currency fingerprint.
    for src in _binding_source_fingerprint():
        h.update(src.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

_HEADER = '"""Generated by scripts/generate_lifter_chains.py; do not hand-edit."""\n'


def _emit_cluster(cluster: str, chains_by_pack: dict) -> str:
    lines = [_HEADER]
    lines.append("# CHAINS[pack_key] = tuple of chains; a chain is an ordered")
    lines.append("# tuple of registry part offsets (see lifter_binding).")
    lines.append("CHAINS = {")
    for pack_key, chains in sorted(chains_by_pack.items()):
        lines.append(f"    {pack_key!r}: (")
        for chain in chains:
            offs = tuple(NAME_TO_OFFSET[p] for p in chain)
            lines.append(f"        {offs!r},")
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_home(home: BodyName, cluster: str, data_by_pack: dict) -> str:
    lines = [_HEADER]
    lines.append(f"CLUSTER = {cluster!r}")
    lines.append("# CONSTRAINTS[pack_key][phys] = AscentConstraints tuple")
    lines.append("# HINTS[pack_key] = interned architecture hints "
                 "(dv_split, ((n_boosters, booster_engines, engine_offset), ...))")
    lines.append("# LADDER_META[pack_key][phys] = ((dv, ceiling_t, "
                 "thresholds), ...) per dv variant")
    lines.append("# LADDER_ROWS[pack_key][phys][profile_id][dv_idx] = ")
    lines.append("#   ((threshold_idx, prefix_len, launch_mass, "
                 "hint_idx), ...)")
    lines.append("HINTS = {")
    for pack_key, (_c, hints, _m, _r) in sorted(data_by_pack.items()):
        lines.append(f"    {pack_key!r}: (")
        for t in hints:
            lines.append(f"        {t!r},")
        lines.append("    ),")
    lines.append("}")
    for name, idx in (("CONSTRAINTS", 0), ("LADDER_META", 2),
                      ("LADDER_ROWS", 3)):
        lines.append(f"{name} = {{")
        for pack_key, data in sorted(data_by_pack.items()):
            by_phys = data[idx]
            lines.append(f"    {pack_key!r}: {{")
            for phys in sorted(by_phys):
                lines.append(f"        {phys!r}: {by_phys[phys]!r},")
            lines.append("    },")
        lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_index(digest: str, homes: list[BodyName]) -> str:
    lines = [_HEADER]
    lines.append(f"DIGEST = {digest!r}")
    lines.append("HOME_CLUSTER = {")
    for b in homes:
        lines.append(f"    {b.value!r}: {HOME_CLUSTER[b]!r},")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parallel workers.  Every unit (one candidate chain grow, one chain bind) is
# independent, so the generator saturates the box exactly like the solve-check
# rig: a flat task list through one shared Pool, imap_unordered, results keyed
# (not order-dependent) so output is deterministic regardless of completion
# order.  AscentBuilder construction is expensive (MissionBuilder + flag
# caches), so each worker process memoizes builders per (body, phys).
# ---------------------------------------------------------------------------

_BUILDER_CACHE: dict = {}


def _get_builder(body: BodyName, phys: str) -> "AscentBuilder":
    key = (body, phys)
    b = _BUILDER_CACHE.get(key)
    if b is None:
        b = AscentBuilder(body, phys)
        _BUILDER_CACHE[key] = b
    return b


def _grow_param_task(args):
    """(cluster, pack_key) -> grow dv + threshold ladder for that cluster's
    rep body (computed in the pool because the Eve rep ceiling is a real
    escalated max-lift)."""
    cluster, pack_key = args
    rep = cluster_rep(cluster)
    phys = _GROWTH_PHYS.get(cluster, _GROWTH_PHYS_DEFAULT)
    builder = _get_builder(rep, phys)
    pool = lifter_pool(pack_key)
    grow_dv = max(builder.dv_variants)
    ceiling = builder.max_lift(frozenset(pool), grow_dv)
    thresholds = tuple(
        t for t in ladder_for(rep, ceiling) if t <= ceiling * 0.9
    ) or (round(_LADDER_BASE_T, 1),)
    return (cluster, pack_key, grow_dv, thresholds, ceiling)


def _grow_task(args):
    """(cluster, pack_key, attempt, grow_dv, thresholds) -> a grown+pruned
    chain (or None).  Deterministic per attempt index."""
    cluster, pack_key, attempt, grow_dv, thresholds = args
    rep = cluster_rep(cluster)
    phys = _GROWTH_PHYS.get(cluster, _GROWTH_PHYS_DEFAULT)
    builder = _get_builder(rep, phys)
    pool = lifter_pool(pack_key)
    key = f"lifter|{cluster}|{','.join(pack_key)}|{attempt}"
    chain = grow_chain(builder, pool, pack_key, key, thresholds, grow_dv)
    if chain is not None:
        chain = prune_chain(builder, chain, thresholds, grow_dv)
    return (cluster, pack_key, attempt, chain)


def _bind_meta_task(args):
    """(home, pack_key, phys) -> constraints + per-dv (dv, ceiling, thresholds)
    for binding.  The home ceiling max-lifts live here (Eve escalated)."""
    home, pack_key, phys = args
    builder = _get_builder(home, phys)
    full = frozenset(lifter_pool(pack_key))
    meta = []
    for dv in builder.dv_variants:
        ceiling = builder.max_lift(full, dv)
        if ceiling <= 0.05:
            continue
        # dv is stored and BUILT WITH at full precision: rounding it (even to
        # 0.1 m/s) shifts an escalated Eve build's mass by ~1 kg, and any path
        # that rebuilds with a different rounding (the drift check, the runtime
        # consult) then disagrees.  It equals capability's runtime required_dv
        # exactly (same effective_dv inputs), so full precision loses nothing.
        meta.append((dv, round(ceiling, 3), ladder_for(home, ceiling)))
    return (home, pack_key, phys,
            _constraints_tuple(builder.constraints()), tuple(meta))


def _bind_chain_task(args):
    """(home, pack_key, phys, chain_idx, chain, per_dv_meta) -> per-dv rungs
    with RAW hint tuples (main-process interns them).  Binds ONE chain."""
    home, pack_key, phys, chain_idx, chain, per_dv_meta = args
    builder = _get_builder(home, phys)
    out = []
    for dv, ceiling, thresholds in per_dv_meta:
        kit: frozenset = frozenset()
        idx = 0
        rungs = []
        for t_idx, T in enumerate(thresholds):
            stages = builder.build(kit, dv, T)
            while stages is None and idx < len(chain):
                kit = kit | {chain[idx]}
                idx += 1
                stages = builder.build(kit, dv, T)
            if stages is None:
                continue
            hint, mass = builder.canonical_hint(kit, dv, T, stages)
            if hint is None:
                continue
            rungs.append((t_idx, idx, math.ceil(mass * 1000) / 1000, hint))
        out.append(tuple(rungs))
    return (home, pack_key, phys, chain_idx, tuple(out))


def _pool_map(pool, fn, tasks):
    if pool is None:
        return [fn(t) for t in tasks]
    return list(pool.imap_unordered(fn, tasks, chunksize=1))


def generate(homes: list[BodyName], n_chains: int, release: bool,
             workers: int = 1, verbose: bool = True):
    from multiprocessing import Pool
    import time as _time
    clusters = sorted({HOME_CLUSTER[h] for h in homes})
    pool = Pool(processes=workers) if workers > 1 else None
    try:
        # --- Phase 0: grow params (rep ceilings/thresholds) ---
        param_tasks = [(c, pk) for c in clusters for pk in PACK_KEYS]
        grow_params = {(c, pk): (dv, th, ceil)
                       for c, pk, dv, th, ceil
                       in _pool_map(pool, _grow_param_task, param_tasks)}

        # --- Phase 1: grow all candidate chains, one saturated pool ---
        t0 = _time.perf_counter()
        grow_tasks = []
        n_by_cp: dict = {}
        for cluster in clusters:
            n_cluster = (min(n_chains, _N_CHAINS_LEAN)
                         if cluster in _LEAN_CLUSTERS else n_chains)
            for pk in PACK_KEYS:
                n_by_cp[(cluster, pk)] = n_cluster
                grow_dv, thresholds, _ceil = grow_params[(cluster, pk)]
                n_attempts = n_cluster * (5 if release else 3)
                for a in range(n_attempts):
                    grow_tasks.append((cluster, pk, a, grow_dv, thresholds))
        grown_by_cp: dict = {(c, pk): [] for c in clusters for pk in PACK_KEYS}
        for cluster, pk, attempt, chain in _pool_map(pool, _grow_task,
                                                     grow_tasks):
            if chain is not None:
                grown_by_cp[(cluster, pk)].append((attempt, chain))
        # Select diverse from the deterministic candidate set (sort by attempt
        # so the pre-select order is completion-independent).
        chains_by_cluster: dict = {c: {} for c in clusters}
        for (cluster, pk), grown in grown_by_cp.items():
            cands = [ch for _a, ch in sorted(grown)]
            chosen = select_diverse(
                cands, n_by_cp[(cluster, pk)],
                f"select|{cluster}|{','.join(pk)}")
            chains_by_cluster[cluster][pk] = chosen
        if verbose:
            print(f"[grow] {len(grow_tasks)} candidates -> "
                  f"{sum(len(v) for c in chains_by_cluster.values() for v in c.values())} "
                  f"chains, {_time.perf_counter()-t0:.1f}s "
                  f"({workers}w)", flush=True)

        # --- Phase 2a: bind metas (home ceilings/thresholds) ---
        meta_tasks = []
        for home in homes:
            phys_list = _EVE_PHYS if home == BodyName.EVE else _ALL_PHYS
            for pk in PACK_KEYS:
                for phys in phys_list:
                    meta_tasks.append((home, pk, phys))
        bind_meta: dict = {}
        constr_by: dict = {}
        for home, pk, phys, ctuple, meta in _pool_map(pool, _bind_meta_task,
                                                      meta_tasks):
            bind_meta[(home, pk, phys)] = meta
            constr_by[(home, pk, phys)] = ctuple

        # --- Phase 2b: bind every chain, one saturated pool ---
        t0 = _time.perf_counter()
        bind_tasks = []
        for home in homes:
            cluster = HOME_CLUSTER[home]
            phys_list = _EVE_PHYS if home == BodyName.EVE else _ALL_PHYS
            for pk in PACK_KEYS:
                chains = chains_by_cluster[cluster][pk]
                for phys in phys_list:
                    meta = bind_meta[(home, pk, phys)]
                    for ci, chain in enumerate(chains):
                        bind_tasks.append((home, pk, phys, ci, chain, meta))
        # (home, pk, phys) -> {chain_idx: per_dv_rows}
        rows_raw: dict = {}
        for home, pk, phys, ci, rows in _pool_map(pool, _bind_chain_task,
                                                  bind_tasks):
            rows_raw.setdefault((home, pk, phys), {})[ci] = rows
        if verbose:
            print(f"[bind] {len(bind_tasks)} chain-binds, "
                  f"{_time.perf_counter()-t0:.1f}s ({workers}w)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # --- Assemble + intern (serial, deterministic order) ---
    home_data: dict = {}
    for home in homes:
        cluster = HOME_CLUSTER[home]
        phys_list = _EVE_PHYS if home == BodyName.EVE else _ALL_PHYS
        home_data[home] = {}
        for pk in PACK_KEYS:
            chains = chains_by_cluster[cluster][pk]
            intern: dict = {}
            constr: dict = {}
            meta_out: dict = {}
            rows_by_phys: dict = {}
            for phys in phys_list:
                meta = bind_meta[(home, pk, phys)]
                constr[phys] = constr_by[(home, pk, phys)]
                meta_out[phys] = tuple(meta)
                per_chain = rows_raw.get((home, pk, phys), {})
                prof_rows = []
                for ci in range(len(chains)):
                    dv_rows = per_chain.get(ci, tuple(() for _ in meta))
                    interned = []
                    for rungs in dv_rows:
                        interned.append(tuple(
                            (ti, p, m, intern.setdefault(h, len(intern)))
                            for ti, p, m, h in rungs))
                    prof_rows.append(tuple(interned))
                rows_by_phys[phys] = tuple(prof_rows)
            pool_tuples = [None] * len(intern)
            for h, i in intern.items():
                pool_tuples[i] = h
            home_data[home][pk] = (
                constr, tuple(pool_tuples), meta_out, rows_by_phys)
    return chains_by_cluster, home_data


def _constraints_tuple(c: AscentConstraints) -> tuple:
    return (int(c.in_atmosphere), c.min_twr_liftoff,
            int(c.requires_throttleable), int(c.srb_needs_rcs),
            int(c.needs_heat_shield), c.pad_altitude_m)


def write_data(chains_by_cluster, home_data, homes) -> dict[str, int]:
    os.makedirs(DATA_DIR, exist_ok=True)
    sizes: dict[str, int] = {}

    def put(fname: str, content: str):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        sizes[fname] = len(content.encode())
    for cluster, by_pack in chains_by_cluster.items():
        put(f"cluster_{cluster}.py", _emit_cluster(cluster, by_pack))
    for home, by_pack in home_data.items():
        put(f"{home.value.lower()}.py",
            _emit_home(home, HOME_CLUSTER[home], by_pack))
    put("_index.py", _emit_index(model_digest(), list(home_data)))
    return sizes


def structural_check() -> list[str]:
    """Invariants over every stored row — no physics, cheap enough for the
    PR gate."""
    errors: list[str] = []
    from ..data.lifter_chains import _index
    from ..lifter_binding import OFFSET_TO_NAME
    import importlib
    for home_name, cluster in _index.HOME_CLUSTER.items():
        cmod = importlib.import_module(
            f"..data.lifter_chains.cluster_{cluster}", __package__)
        hmod = importlib.import_module(
            f"..data.lifter_chains.{home_name.lower()}", __package__)
        for pack_key, chains in cmod.CHAINS.items():
            for ci, chain in enumerate(chains):
                bad = [o for o in chain if o not in OFFSET_TO_NAME]
                if bad:
                    errors.append(f"{cluster}/{pack_key}[{ci}]: unknown offsets {bad}")
                if len(set(chain)) != len(chain):
                    errors.append(f"{cluster}/{pack_key}[{ci}]: duplicate parts")
        for pack_key, by_phys in hmod.LADDER_ROWS.items():
            chains = cmod.CHAINS[pack_key]
            n_hints = len(hmod.HINTS[pack_key])
            for phys, prof_rows in by_phys.items():
                if len(prof_rows) != len(chains):
                    errors.append(
                        f"{home_name}/{phys}: {len(prof_rows)} profiles "
                        f"!= {len(chains)} chains")
                meta = hmod.LADDER_META[pack_key][phys]
                for dv, ceiling, thresholds in meta:
                    if ceiling <= 0:
                        errors.append(f"{home_name}/{phys}: ceiling {ceiling}")
                    if tuple(sorted(thresholds)) != tuple(thresholds):
                        errors.append(f"{home_name}/{phys}: unsorted ladder")
                for pid, rows in enumerate(prof_rows):
                    if len(rows) != len(meta):
                        errors.append(
                            f"{home_name}/{phys}[{pid}]: dv-variant count")
                        continue
                    for (dv, ceiling, thresholds), rungs in zip(meta, rows):
                        # Threshold indices strictly ascend and prefix length
                        # never shrinks (the walk only advances the chain).
                        # Launch mass is deliberately NOT asserted monotone: a
                        # longer prefix can unlock asparagus and build LIGHTER
                        # (consult serves the lightest admitted rung).
                        last_t = last_p = -1
                        for (ti, p, m, hidx) in rungs:
                            if ti >= len(thresholds):
                                errors.append(
                                    f"{home_name}/{phys}[{pid}]: t_idx OOB")
                                continue
                            if not (ti > last_t and p >= last_p):
                                errors.append(
                                    f"{home_name}/{phys}[{pid}] dv={dv}: "
                                    f"non-monotone rung idx {ti}")
                            if p > len(chains[pid]):
                                errors.append(
                                    f"{home_name}/{phys}[{pid}]: prefix {p} "
                                    f"> chain len")
                            if hidx >= n_hints:
                                errors.append(
                                    f"{home_name}/{phys}[{pid}]: hint "
                                    f"idx OOB")
                            last_t, last_p = ti, p
    return errors


def sampled_rebind_check(full: bool = False) -> list[str]:
    """Re-bind sampled (home, phys, chain) cells with the EXACT production
    functions (``_bind_meta_task`` + ``_bind_chain_task`` — the same code the
    generator ran) and compare row-exact against the stored tables.  Sharing
    one binding path is the point: a re-implementation is what let the two
    drift (a dv rounding lived in one copy only).  Hint indices are compared
    by resolved tuple (stored/fresh intern orders differ).  ``full`` re-binds
    every chain (release/nightly gate); the default samples first/last chain.
    """
    import importlib
    errors: list[str] = []
    from ..data.lifter_chains import _index
    from ..lifter_binding import OFFSET_TO_NAME
    pack_key = PACK_KEYS[-1]
    for home_name, cluster in sorted(_index.HOME_CLUSTER.items()):
        home = next(b for b in BodyName if b.value == home_name)
        cmod = importlib.import_module(
            f"..data.lifter_chains.cluster_{cluster}", __package__)
        hmod = importlib.import_module(
            f"..data.lifter_chains.{home_name.lower()}", __package__)
        chains_off = cmod.CHAINS.get(pack_key, ())
        if not chains_off:
            continue
        phys_list = sorted(hmod.LADDER_ROWS[pack_key])
        phys_sample = phys_list if full else phys_list[:1]
        pids = (list(range(len(chains_off))) if full
                else sorted({0, len(chains_off) - 1}))
        stored_hints = hmod.HINTS[pack_key]
        for phys in phys_sample:
            _h, _p, _ph, _ct, meta = _bind_meta_task((home, pack_key, phys))
            if tuple(meta) != tuple(hmod.LADDER_META[pack_key][phys]):
                errors.append(f"{home_name}/{phys}: ladder meta drift")
                continue
            for pid in pids:
                chain = [OFFSET_TO_NAME[o] for o in chains_off[pid]]
                *_ignore, rows = _bind_chain_task(
                    (home, pack_key, phys, pid, chain, meta))
                stored = hmod.LADDER_ROWS[pack_key][phys][pid]
                # stored rungs carry hint INDICES; fresh carry raw hint tuples.
                norm_s = tuple(tuple((t, p, m, stored_hints[hx])
                                     for t, p, m, hx in rungs)
                               for rungs in stored)
                norm_f = tuple(tuple((t, p, m, h)
                                     for t, p, m, h in rungs)
                               for rungs in rows)
                if norm_s != norm_f:
                    errors.append(
                        f"{home_name}/{phys}[{pid}]: re-bind drift")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--homes", default="")
    ap.add_argument("--n-chains", type=int, default=100)
    ap.add_argument("--release", action="store_true",
                    help="grow 5xN candidates, keep the N most different")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 1) - 1),
                    help="parallel worker processes (default cpu_count-1)")
    args = ap.parse_args()

    if args.homes:
        homes = [next(b for b in BodyName if b.value.lower() == h.strip().lower())
                 for h in args.homes.split(",")]
    else:
        homes = sorted(HOME_CLUSTER, key=lambda b: b.value)

    if args.check:
        # Default (PR) gate — Option A: structural invariants + a COMPLETE
        # digest (model data + binding source).  No physics: a digest match
        # proves the data is current with the code+model that would generate
        # it.  ``--full`` adds the deep gate (re-bind every chain and compare)
        # for a local / manual run before shipping.
        errors = structural_check()
        errors += assert_cluster_envelopes()
        from ..data.lifter_chains import _index
        if _index.DIGEST != model_digest():
            errors.append("model/source digest drift — regenerate with --write")
        if args.full and not errors:
            errors += sampled_rebind_check(full=True)
        for e in errors:
            print(f"CHECK FAIL: {e}", file=sys.stderr)
        print(f"lifter-chains check: "
              f"{'OK' if not errors else 'FAILED'}"
              f"{' (digest-only; --full to re-bind)' if not args.full else ''}")
        return 1 if errors else 0

    chains, home_data = generate(homes, args.n_chains, args.release,
                                 workers=args.workers)
    if args.write:
        sizes = write_data(chains, home_data, homes)
        total = sum(sizes.values())
        for f, s in sorted(sizes.items(), key=lambda kv: -kv[1]):
            print(f"  {s/1024:8.1f} KB  {f}")
        print(f"wrote {len(sizes)} files, {total/1024:.1f} KB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
