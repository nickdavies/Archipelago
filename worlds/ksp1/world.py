import math
import os
import random
from typing import Any

from BaseClasses import CollectionState, Item, MultiWorld, Tutorial
from Options import OptionError
from worlds.AutoWorld import LogicMixin, WebWorld, World

from . import contracts, items, locations, regions, rules
from .ksc_sites import ksc_site_slot_data
from .rules import GoalSpec, resolve_goal_spec, goal_spec_location_names
from .capability import CAPABILITY_ITEMS, RocketCapability
from .data.feasibility import MODEL_INFEASIBLE_LOCATIONS
from .bodies import (
    ALL_BODIES, BodyName, MissionBuilder, MissionType, RandomOrbitParams,
    generate_random_orbit_params, home_relative_science_values,
)
from .items import (
    ITEM_NAME_TO_ID, PROGRESSIVE_LAUNCH_PAD_CAPS, _FILLER_ITEMS,
    PROGRESSIVE_RD_NAME, PROGRESSIVE_RD_COUNT,
)
from .locations import (
    ALL_EVENTS, KSC_BIOMES, KSC_LOCATION_PREFIX,
    LOCATION_NAME_TO_ID, LocationBuilder, MAX_TECH_SLOTS,
    THRESHOLD_LOCATION_NAMES, TechTreeLocation,
    effective_starting_inv_count, effective_tech_slots_per_node,
)
from .options import Goal, GoalContractMode, KSP1Options, STARTING_BODY_POOLS, StartingBody
from .tech_tree import MAX_TIER, NODES_BY_TIER, TECH_NODES, TIER_TO_BAND


# Diagnostic flag: keep the strict_ladder post_fill physics cross-check but
# DISABLE the re-fill fallback — raise instead of rescuing.  For perf testing
# (the fallback is a whole-seed capability re-fill that otherwise dominates
# slow-goal wall time) and correctness testing (a cheap-rule-vs-capability
# divergence fails loudly here instead of being silently repaired).  Default
# off = normal rescue behaviour.
_NO_STRICT_LADDER_FALLBACK = os.environ.get("KSP_NO_STRICT_LADDER_FALLBACK") == "1"


class KSP1State(LogicMixin):
    """Inject per-player stale flag and cached result onto CollectionState."""
    ksp1_cap_stale: dict[int, bool]
    ksp1_cap_result: dict[int, RocketCapability]

    def init_mixin(self, multiworld: MultiWorld) -> None:
        self.ksp1_cap_stale = {p: True for p in multiworld.get_game_players("Kerbal Space Program 1")}
        self.ksp1_cap_result = {}


class KSP1WebWorld(WebWorld):
    theme = "ocean"
    tutorials = [
        Tutorial(
            "Setup Guide",
            "A guide to setting up the Kerbal Space Program Archipelago client",
            "English",
            "setup_en.md",
            "setup/en",
            ["nickdavies"],
        )
    ]


def _validate_goal_spec_has_targets(
    spec: GoalSpec, options: KSP1Options, home: BodyName,
) -> None:
    """Raise ``OptionError`` if the resolved goal has nothing to achieve.

    Happens when every body in the chosen preset is the player's home —
    e.g. ``goal=mun_flag`` with ``starting_body=mun``, or
    ``goal=duna_return`` with ``starting_body=duna``.  ``resolve_goal_spec``
    strips home from every body list, so a preset whose only target was
    home becomes empty.  We fail loudly at gen time instead of letting
    fill produce a trivially-winnable seed.
    """
    has_targets = (
        spec.complete_tech_tree
        or spec.flag_bodies
        or spec.return_bodies
        or spec.sample_return_bodies
        or spec.orbit_bodies
        or spec.flyby_bodies
    )
    if has_targets:
        return
    raise OptionError(
        f"KSP1: starting_body={home.value!r} is incompatible with "
        f"goal={options.goal.current_key!r} — every body in the preset is "
        "the home body, so the goal would be trivially complete at launch. "
        "Pick a different starting body, a different goal, or use a "
        "custom goal that targets a non-home body."
    )


def _validate_goal_not_excluded(
    spec: GoalSpec, exclude_locations: frozenset[str],
) -> None:
    """Raise ``OptionError`` if any mission-event goal location is in
    ``exclude_locations``.

    Excluding a mission-event goal location (e.g. ``Tylo Return 1`` for a
    Laythe-home ``jool_moons_return`` seed) is always a user mistake: the
    location holds no progression, but the goal still requires reaching
    it, and the sphere ladder treats excluded goals as unreachable and
    blows up at fill time.

    Tech-tree goal locations (Complete Tech Tree preset) are
    *intentionally* compatible with ``exclude_late_tech_tree`` — the
    player "completes" them by spending science, not by AP placing
    progression there.  Skipped here.
    """
    from .rules import goal_spec_location_names
    goal_names = set(goal_spec_location_names(spec))
    # Tech-tree slot names contain " - " (e.g. ``General Rocketry 1``)
    # vs mission-event names (``Tylo Return 1``).  Strip tech-tree goals
    # by name prefix membership instead — every node display name is in
    # ``TECH_NODES``.
    from .tech_tree import TECH_NODES
    tech_prefixes = {n.display_name + " " for n in TECH_NODES}
    mission_goal_names = {
        n for n in goal_names
        if not any(n.startswith(p) for p in tech_prefixes)
    }
    conflict = mission_goal_names & exclude_locations
    if not conflict:
        return
    sorted_conflict = sorted(conflict)
    raise OptionError(
        f"KSP1: exclude_locations covers goal location(s) "
        f"{sorted_conflict!r} for goal {spec.display_name!r}.  Remove "
        "these names from exclude_locations (or pick a different goal). "
        "Default exclude_locations is empty as of v0.4 — if these came "
        "from an older yaml, delete the stale entries."
    )


def _validate_goal_contracts_registrable(goal_contract_specs) -> None:
    """Raise ``OptionError`` if any goal contract lacks a registered location.

    Goal contracts are emitted for every goal achievement unconditionally — no
    ``body_compatible`` filter (see ``_goal_contract_specs``) — so a goal body
    that can't host its mission type would produce a ``ContractSpec`` whose
    location was never registered, and the lookup would ``KeyError`` deep in
    ``create_regions``.  The orbit/flyby body lists are option-validated against
    ``ORBITABLE_BODY_NAMES`` (the star has no missions), so this is defense in
    depth: fail fast with a clear message if any path ever slips one through.
    """
    from .locations import CONTRACT_LOCATION_NAME_SET
    bad = sorted(
        s.location_name for s in goal_contract_specs
        if s.location_name not in CONTRACT_LOCATION_NAME_SET
    )
    if not bad:
        return
    raise OptionError(
        f"KSP1: goal targets a body that can't host its mission type — no "
        f"registered location for {bad!r}.  The star (Sun) has no "
        "orbit/flyby/landing missions; remove it from your goal body lists."
    )


# KSP upgradeable facility ids (must match the client's CareerUpgradesManager).
# All are forced to max in the hacked career; per-building levels are emitted so
# real facility progression can be reintroduced one building at a time later.
_FACILITY_IDS: tuple[str, ...] = (
    "SpaceCenter/VehicleAssemblyBuilding",
    "SpaceCenter/SpaceplaneHangar",
    "SpaceCenter/LaunchPad",
    "SpaceCenter/Runway",
    "SpaceCenter/TrackingStation",
    "SpaceCenter/MissionControl",
    "SpaceCenter/AstronautComplex",
    "SpaceCenter/ResearchAndDevelopment",
    "SpaceCenter/Administration",
)
_MAX_FACILITY_LEVEL = 2  # stock 0/1/2 (level-3 buildings)

# Facilities the buildings_in_logic option gates as AP progression.  When the
# option is on these START at level 0 (the player upgrades them by collecting
# the curated building progressives); every other facility stays maxed.  The
# Launch Pad is gated separately via progressive_launch_pad (its tonnage caps
# ride their own slot_data key), so it is NOT listed here.  Tracking Station is
# omitted: its DSN effect is a deferred seam (relay_tier already gates comms),
# so it must not change today's maxed behavior.
_GATED_FACILITY_IDS: tuple[str, ...] = (
    "SpaceCenter/VehicleAssemblyBuilding",
    "SpaceCenter/SpaceplaneHangar",
    "SpaceCenter/AstronautComplex",
)


class KSP1World(World):
    """
    Kerbal Space Program is a space flight simulation game where you design and
    fly rockets to explore the Kerbol system.  Parts are randomized across the
    multiworld, and mission completions are the location checks.
    """

    game = "Kerbal Space Program 1"
    web = KSP1WebWorld()
    ut_can_gen_without_yaml = True

    # Tech tree entrance rules use can_reach_region() for parent dependencies.
    # The auto version retries blocked connections when new regions are reached;
    # the explicit version requires manual indirect condition registration.
    explicit_indirect_conditions = False

    options_dataclass = KSP1Options
    options: KSP1Options

    item_name_to_id = ITEM_NAME_TO_ID
    location_name_to_id = LOCATION_NAME_TO_ID

    # Fingerprint → RocketCapability, shared across all CollectionState copies
    capability_cache: dict[frozenset[tuple[str, int]], RocketCapability]

    # Resolved goal specification (preset or custom).
    goal_spec: GoalSpec

    # Mission graph builder for this world.  Pinned to Kerbin here; owned
    # by the world so its data lifetime matches the rest of the per-world
    # state.
    mission_builder: MissionBuilder

    # Per-world home-body location set (11 home specials: first launch /
    # landing / crash, altitude milestones, first staging — plus the single
    # body-agnostic "Splashdown" entry).  Owned alongside ``mission_builder``
    # so the two stay in sync on the same home.
    location_builder: LocationBuilder

    # AP location names whose mission the dv model can't verify from this
    # world's home, even given a full progressive kit + every part.
    # Looked up at world-init time from the checked-in
    # ``MODEL_INFEASIBLE_LOCATIONS`` table (regenerated offline by
    # ``scripts/generate_feasibility.py``).  Completion-condition rules
    # for these locations fall back to the "all-parts collected" proxy
    # because the dv model can't model their ascents (Eve's 8 km/s,
    # Laythe's atmospheric Jool-system return, etc.).
    model_infeasible_locations: frozenset[str]

    # Contracts generated for this seed (paced into the run as items). Non-goal
    # vs goal-achievement contracts; both are ContractSpec. Set in generate_early.
    contract_specs: list
    goal_contract_specs: list
    # Goal-mode (count / progressive_unlock) state. Resolved in generate_early.
    # ``contracts_required`` is X (completed non-goal contracts needed for the
    # goal). ``contract_threshold_defs`` is the list of
    # ``(threshold_location_name, required_count, locked_item_name)`` triples —
    # each threshold is a pre-filled location the client reports once the
    # completed-contract count reaches ``required_count``, releasing the locked
    # goal contract item (or Progressive R&D copy for the tech-tree goal). Empty
    # in findable / starting modes.
    contracts_required: int
    contract_threshold_defs: list
    # Part ksp_names this seed's contracts require — promoted to progression in
    # items.create_item so AP guarantees them reachable before the contract.
    contract_required_part_names: frozenset[str]
    # contract_ids whose access rule routes through the all-parts proxy (vs the
    # physics gate). Recorded by rules._set_contract_rules at rule-set time; read
    # by /explain so the reported gate is the one actually set, never re-derived.
    _proxy_contract_ids: set[str]

    def generate_early(self) -> None:
        """Resolve goal spec and apply ExcludeLateTechTree."""
        self.capability_cache = {}
        # Pool keys (atmospheric/standard/planets/all) resolve to a
        # concrete body via the seed RNG, then overwrite the option so
        # downstream code (and slot_data) sees a single body just like
        # if the player had typed it explicitly.  sorted() before
        # choice() keeps the pick deterministic for a given seed.
        key = self.options.starting_body.current_key
        if key in STARTING_BODY_POOLS:
            picked = self.random.choice(sorted(STARTING_BODY_POOLS[key]))
            self.options.starting_body.value = getattr(
                StartingBody, f"option_{picked.value.lower()}"
            )
        # ``starting_body`` option keys are lowercase BodyName values
        # (``option_mun`` → key ``"mun"`` → ``BodyName.MUN``).  The
        # title-case round-trip rebuilds the canonical ``StrEnum`` value.
        home = BodyName(self.options.starting_body.current_key.title())
        self.mission_builder = MissionBuilder(home=home)
        self.location_builder = LocationBuilder(home=home)
        # Per-world RankContext for sphere-ladder + item.rank_sig.
        # ``home_has_atmosphere`` drives the SRB axis scorer; the rest
        # of the rank table is body-agnostic.
        from .ranks import RankContext
        _atmo_homes = {BodyName.KERBIN, BodyName.EVE, BodyName.DUNA, BodyName.LAYTHE}
        self._rank_context = RankContext(home_has_atmosphere=(home in _atmo_homes))

        # UT regen: restore options from original generation's slot_data.
        passthrough = getattr(self.multiworld, "re_gen_passthrough", {})
        if isinstance(passthrough, dict) and self.game in passthrough:
            self._apply_slot_data(passthrough[self.game])

        # Model-infeasible-locations set is a checked-in static lookup
        # keyed by home body — generated offline by
        # ``scripts/generate_feasibility.py`` so the banned-location set
        # is deterministic per commit hash and never drifts between
        # seeds.  An empty fallback covers homes not yet in the table
        # (unreachable today; defensive).
        self.model_infeasible_locations = MODEL_INFEASIBLE_LOCATIONS.get(
            self.mission_builder.home, frozenset(),
        )
        self.goal_spec = resolve_goal_spec(
            self.options, self.mission_builder.home,
            self.model_infeasible_locations,
        )
        _validate_goal_spec_has_targets(
            self.goal_spec, self.options, self.mission_builder.home,
        )

        # Seeded target orbits for RANDOM_ORBIT contracts — must exist before
        # generate_contracts (the feasibility + harder-than-goal cap read the
        # real orbit cost via mission_builder.transform_mission). A derived RNG
        # keeps the draw count off the main sequence; UT regen restores the exact
        # orbits from slot_data instead of re-rolling.
        ut_orbits = getattr(self, "_ut_random_orbit_params", None)
        if ut_orbits is not None:
            self.mission_builder.random_orbit_params = ut_orbits
        else:
            self.mission_builder.random_orbit_params = generate_random_orbit_params(
                random.Random(self.random.getrandbits(64)), ALL_BODIES)

        # Generate this seed's contracts (deterministic from the world seed).
        # UT regen restores the exact set from slot_data instead of re-rolling.
        ut_contracts = getattr(self, "_ut_contract_specs", None)
        if ut_contracts is not None:
            self.contract_specs = [s for s in ut_contracts if not s.is_goal]
            self.goal_contract_specs = [s for s in ut_contracts if s.is_goal]
        else:
            self.contract_specs, self.goal_contract_specs = (
                contracts.generate_contracts(self))
        self.contract_required_part_names = contracts.required_part_names_for(
            (*self.contract_specs, *self.goal_contract_specs))
        # Reward slots each non-goal contract yields this seed: base 2 plus the
        # Contract Repeats option. Resolved once here so every per-seed consumer
        # (region registration, access rules, /explain, slot_data) reads one
        # value instead of re-deriving from options. 0 repeats == exactly 2.
        self.non_goal_slot_count = contracts.non_goal_slot_count(self.options)
        _validate_goal_contracts_registrable(self.goal_contract_specs)

        # Goal contract mode: validate + resolve X and the threshold locations.
        self._resolve_goal_contract_mode()

        if self.options.exclude_late_tech_tree:
            late_tier_locs: set[str] = {
                str(TechTreeLocation(node.display_name, slot))
                for node in NODES_BY_TIER.get(MAX_TIER, [])
                for slot in range(1, MAX_TECH_SLOTS + 1)
            }
            self.options.exclude_locations.value |= late_tier_locs

        # Final exclude_locations is resolved (yaml + late-tech-tree).  Now
        # verify no goal location ended up excluded — that combination is
        # always unsolvable, so fail at gen time with a clear message
        # rather than letting fill produce an opaque error hours later.
        _validate_goal_not_excluded(
            self.goal_spec, frozenset(self.options.exclude_locations.value),
        )

    def _resolve_goal_contract_mode(self) -> None:
        """Validate the goal-contract-mode configuration and build the threshold
        locations (count / progressive_unlock). Fail fast with ``OptionError`` on
        any unsolvable combination — predictable structural violations belong at
        generate_early, not as an opaque FillError later.

        Sets ``self.contracts_required`` (X) and ``self.contract_threshold_defs``
        (empty in findable / starting). UT regen restores X from slot_data; the
        threshold defs recompute deterministically from the restored contracts."""
        mode = self.options.goal_contract_mode.value
        is_random = self.options.goal.value == Goal.option_random_contracts
        is_tech = self.goal_spec.complete_tech_tree
        needs_thresholds = mode in (GoalContractMode.option_count,
                                    GoalContractMode.option_progressive_unlock)

        # random_contracts is a contracts-driven goal: only count / progressive
        # give it a win condition. findable / starting would be degenerate.
        if is_random and not needs_thresholds:
            raise OptionError(
                "KSP1: goal 'random_contracts' requires goal_contract_mode "
                "'count' or 'progressive_unlock' (it has no destination goal to "
                "find or start with)."
            )

        if not needs_thresholds:
            self.contracts_required = 0
            self.contract_threshold_defs = []
            return

        # count / progressive need contracts to count toward.
        n_contracts = len(self.contract_specs)
        if n_contracts == 0:
            raise OptionError(
                "KSP1: goal_contract_mode 'count'/'progressive_unlock' needs "
                "contracts to complete, but none were generated. Raise "
                "contracts_available (or enable more contract types)."
            )

        # Resolve X (auto = 80% of generated contracts, rounded up).
        ut_x = getattr(self, "_ut_contracts_required", None)
        if ut_x is not None:
            x = int(ut_x)
        else:
            raw_x = self.options.contracts_required_for_goal.value
            raw_y = self.options.contracts_available.value
            if raw_x >= 0 and raw_y >= 0 and raw_x > raw_y:
                raise OptionError(
                    f"KSP1: contracts_required_for_goal ({raw_x}) exceeds "
                    f"contracts_available ({raw_y})."
                )
            x = raw_x if raw_x >= 0 else math.ceil(0.8 * n_contracts)

        # Clamp to the contracts actually generated (candidate shortfall is not a
        # user error — warn and continue rather than abort).
        if x > n_contracts:
            import logging
            logging.warning(
                "KSP1: contracts_required_for_goal %d exceeds the %d contracts "
                "generated this seed; clamping to %d.", x, n_contracts, n_contracts)
            x = n_contracts
        if x < 1:
            raise OptionError(
                "KSP1: goal_contract_mode 'count'/'progressive_unlock' needs at "
                "least 1 contract required for the goal (got "
                f"{x}); pick a positive contracts_required_for_goal."
            )
        self.contracts_required = x

        # Which items the thresholds award, in unlock order:
        #   tech-tree goal -> Progressive R&D copies (count: just the final copy;
        #                     progressive: all PROGRESSIVE_RD_COUNT copies)
        #   body goal      -> goal contract items, easiest mission first
        if is_tech:
            n_copies = (1 if mode == GoalContractMode.option_count
                        else PROGRESSIVE_RD_COUNT)
            threshold_items = [PROGRESSIVE_RD_NAME] * n_copies
        else:
            threshold_items = [
                s.item_name for s in contracts.goal_contracts_easiest_first(self)
            ]

        k = len(threshold_items)
        defs: list = []
        for i, item_name in enumerate(threshold_items, start=1):
            # count: every goal item unlocks together at X. progressive: staggered
            # so the k-th unlocks exactly at X.
            count = x if mode == GoalContractMode.option_count else math.ceil(i * x / k)
            defs.append((THRESHOLD_LOCATION_NAMES[i - 1], count, item_name))
        self.contract_threshold_defs = defs

    def create_regions(self) -> None:
        regions.create_all_regions(self)
        locations.create_all_locations(self)
        rules.create_victory_location(self)
        rules.create_threshold_locations(self)

    def create_items(self) -> None:
        items.create_all_items(self)

    def set_rules(self) -> None:
        rules.set_all_rules(self)
        rules.set_completion_condition(self, self.goal_spec)

    def pre_fill(self) -> None:
        from .sphere_ladder import apply_sphere_ladder
        apply_sphere_ladder(self)

    def fill_hook(self, progitempool, usefulitempool, filleritempool,
                  fill_locations) -> None:
        """Order non-progression items MOST-restricted first for AP's fill.

        ``remaining_fill`` pops items from the END of the pool and places each
        at the first valid location.  A high-rank non-progression part is valid
        only at high spheres (its lower-bound placement rule), so if less-
        restricted items are placed first they can take the scarce high-sphere
        spots and wedge the few high-rank parts at the end (observed: SSR alien
        FILL_ERR on a couple of tank/adapter parts).  Sorting OUR non-progression
        items by ladder position so the hardest sit at the END (popped first)
        makes fill go most-restricted → least-restricted — a generic remedy for
        ordering-induced (not capacity) fill failures.

        Progression is deliberately left untouched: its restrictive fill already
        succeeds, and biasing the progression order is the known-negative lever
        (it exposes counted-progressive self-locking — see the fill-failure
        post-mortem).  Other players' items keep their order.
        """
        from .sphere_ladder import _item_min_sphere
        ladder = getattr(self, "_sphere_ladder", None)
        if ladder is None:
            return
        spheres = ladder.spheres
        for pool in (usefulitempool, filleritempool):
            mine = [it for it in pool if it.player == self.player]
            if not mine:
                continue
            # ascending min_sphere → hardest (highest) last → popped first
            mine.sort(key=lambda it: _item_min_sphere(it, spheres))
            it = iter(mine)
            for i, item in enumerate(pool):
                if item.player == self.player:
                    pool[i] = next(it)

    def post_fill(self) -> None:
        # strict_ladder cross-check: the cheap sphere-bracket access rules
        # were used during fill.  Swap the saved capability access rules
        # back in and confirm the placement is winnable under real physics.
        saved = getattr(self, "_strict_ladder_saved_rules", None)
        if not saved:
            return
        # Snapshot the cheap bracket rules the fill used, then swap the saved
        # capability rules in for the cross-check.  AP's can_beat_game reads
        # loc.access_rule, so the physics check must temporarily install the real
        # rules — but it is ONE sweep (~1.3s).  Leaving them installed would make
        # the spoiler playthrough's prune pass (many can_beat_game sweeps) re-pay
        # the full get_capability cost (~25s), so restore the cheap rules after a
        # passing check — the spoiler then describes the seed with the same
        # (conservative) rules the fill actually used.
        cheap_rules = {}
        for loc in self.multiworld.get_locations(self.player):
            orig = saved.get(loc.name)
            if orig is not None:
                cheap_rules[loc.name] = loc.access_rule
                loc.access_rule = orig
        if self.multiworld.can_beat_game():
            for loc in self.multiworld.get_locations(self.player):
                if loc.name in cheap_rules:
                    loc.access_rule = cheap_rules[loc.name]
            return  # cheap-rule fill is winnable under capability — done

        # The cheap-rule fill produced a placement capability can't solve — a
        # cheap-rule-vs-capability divergence (now rare, ~0.5%, mostly Laythe
        # deep-interplanetary after the contract-rule unification).
        self._strict_ladder_fell_back = True
        summary = self._strict_ladder_divergence_summary()

        if _NO_STRICT_LADDER_FALLBACK:
            # Diagnostic mode: keep the strict physics cross-check but skip the
            # rescue — surface the divergence as a hard failure (perf +
            # correctness testing).  solve-check classifies this as UNSOLVABLE.
            raise OptionError(
                "strict_ladder cross-check failed and the fallback is disabled "
                f"(KSP_NO_STRICT_LADDER_FALLBACK): home={self.mission_builder.home} "
                f"goal={self.options.goal.current_key} — {summary}"
            )

        # FALLBACK.  Log the divergence (the punch-list for the round-trip fix)
        # and RE-FILL with the capability rules now active — equivalent to
        # strict_validation for this one seed.  Rare, so the slow fill is only
        # paid where the cheap path is unsound.
        import logging
        from Fill import distribute_items_restrictive
        logging.warning(
            "KSP1 strict_ladder fallback (re-fill with capability rules): "
            "home=%s goal=%s — %s",
            self.mission_builder.home, self.options.goal.current_key, summary,
        )
        cleared = []
        for loc in self.multiworld.get_locations(self.player):
            if loc.address is not None and loc.item is not None and not loc.locked:
                it = loc.item
                loc.item = None
                it.location = None
                cleared.append(it)
        self.multiworld.itempool = cleared
        distribute_items_restrictive(self.multiworld)
        if not self.multiworld.can_beat_game():
            raise OptionError(
                "strict_ladder fallback FAILED: a capability-rule re-fill is "
                "still not winnable — genuine unsolvable seed, not a "
                "bracketing bug."
            )

    def _strict_ladder_divergence_summary(self) -> str:
        """Short description of what capability can't reach under the cheap
        fill — logged on fallback to build the round-trip (3) punch-list."""
        from BaseClasses import CollectionState, ItemClassification
        st = CollectionState(self.multiworld)
        st.sweep_for_advancements()
        unreached = [
            (l.name, l.item.name)
            for l in self.multiworld.get_locations(self.player)
            if l.item and (l.item.classification & ItemClassification.progression)
            and not l.can_reach(st)
        ]
        sample = ", ".join(f"{n}<-{it}" for n, it in unreached[:5])
        return f"{len(unreached)} unreachable progression; e.g. {sample}"

    def create_item(self, name: str) -> items.KSP1Item:
        return items.create_item(self, name)

    def get_filler_item_name(self) -> str:
        return items.get_filler_item_name(self)

    def fill_slot_data(self) -> dict[str, Any]:
        d = self.options.as_dict("goal", "difficulty", "start_with_launch_clamps", "item_pacing")
        # Home body — used by the client mod to drive every per-body
        # comparison (KSC biome prefixes, altitude polling guard, splashdown
        # detection, first-launch / first-landing / first-crash events).
        d["starting_body"] = self.mission_builder.home.value
        # KSC site row for an alien starting body: the landing coordinate
        # (lat/lon/terrain alt) + map-decal flag where the cloned KSC cluster
        # is placed.  The client materialises the alien KSC from this instead
        # of carrying a per-body table.  Absent for a Kerbin start (stock KSC).
        ksc_site = ksc_site_slot_data(self.mission_builder.home)
        if ksc_site is not None:
            d["ksc_site"] = ksc_site
        d["tech_slots_per_node"] = effective_tech_slots_per_node(
            self.options, self.options.difficulty.value
        )
        d["node_bands"] = {n.node_id: TIER_TO_BAND[n.tier] for n in TECH_NODES}
        d["goal_locations"] = goal_spec_location_names(self.goal_spec)
        d["goal_display_name"] = self.goal_spec.display_name
        # Authoritative data for C# client — eliminates hardcoded dicts.
        d["event_scales"] = {e.name: e.scale for e in ALL_EVENTS}
        d["tech_display_names"] = {n.node_id: n.display_name for n in TECH_NODES}
        # Per-home biome_key -> AP location name map.  The client populates
        # its detection table directly from this; it has no hardcoded copy.
        # ``location_builder.ksc_biomes`` filters out the ``KSC`` catchall
        # entry on non-Kerbin homes (the surrounding terrain doesn't
        # report as that biome off Kerbin).
        d["ksc_biome_locations"] = {
            key: KSC_LOCATION_PREFIX + name
            for key, name in self.location_builder.ksc_biomes
        }
        home_altitude_thresholds = [
            int(loc.threshold_km * 1000)
            for loc in self.location_builder.locations
            if loc.mission_type == MissionType.SOUNDING and loc.threshold_km is not None
            and loc.threshold_km >= 1.0  # exclude "First Crash" (0.1 km)
        ]
        d["home_altitude_thresholds"] = home_altitude_thresholds
        # Backwards-compat alias for client v0.3.x.  The renamed
        # ``home_altitude_thresholds`` key is the canonical form going
        # forward; the legacy ``kerbin_altitude_thresholds`` key will be
        # retired in a future breaking release.
        d["kerbin_altitude_thresholds"] = home_altitude_thresholds
        d["science_packs"] = {
            name: int(name.split()[-1])
            for name in _FILLER_ITEMS
        }
        # Home-relative science scaling.  Server-side ``science_budget``
        # (rules + sphere-ladder) and the client both consume the SAME
        # ``science_scalar(body, home) * stock_mult`` math; the values
        # below are the absolute CelestialBody.scienceValues the client
        # writes.  Key omitted for Kerbin home; client treats absent
        # key as "feature off, leave stock alone".  When present, the
        # dict contains every body and every situation — the client
        # hard-fails on a missing entry (no silent defaults).
        sci_values = home_relative_science_values(self.mission_builder.home)
        if sci_values:
            d["science_values"] = sci_values
        d["starting_inv_count"] = effective_starting_inv_count(
            self.options, self.options.difficulty.value
        )
        # Mass-cap progression: only set when option is enabled. Caps are in
        # tonnes, indexed by collected count of "Progressive Launch Pad"
        # (0..N). The sentinel -1.0 marks "unlimited" so JSON can carry it.
        if self.options.progressive_launch_pad:
            d["progressive_launch_pad_caps"] = [
                cap if cap != float("inf") else -1.0
                for cap in self.mission_builder.launch_pad_caps
            ]

        # Hacked-career directives — server→client, always emitted, actuated
        # verbatim by the dumb client. Career replaces the prior game mode; the
        # client rejects non-Career saves. Per-building start levels let real
        # facility progression be reintroduced piecemeal later.
        #
        # buildings_in_logic OFF (default): every facility maxed — today's
        # behavior, byte-for-byte.  ON: the curated-gated facilities START at
        # level 0 so the player upgrades them via the AP building progressives;
        # ungated facilities stay maxed.  (Client actuation of the start level
        # is fast-follow; this just emits the server-authoritative value.)
        building_levels = {b: _MAX_FACILITY_LEVEL for b in _FACILITY_IDS}
        if self.options.buildings_in_logic:
            for b in _GATED_FACILITY_IDS:
                building_levels[b] = 0
        d["career"] = {
            "building_levels": building_levels,
            "infinite_funds": True,
            "infinite_reputation": True,
            "unlimited_contracts": True,
        }
        # Contract manifest: each entry is self-describing; the client builds a
        # native KSP contract from `parameters` and reports `location` on
        # completion. Goal contracts ride the same array.
        d["contracts"] = [
            spec.to_slot_dict(self.mission_builder, self.non_goal_slot_count)
            for spec in (*self.contract_specs, *self.goal_contract_specs)
        ]
        # Seeded RANDOM_ORBIT target orbits, per body — carried so UT regen
        # restores the exact orbits (the client also gets them via each
        # contract's specific_orbit parameter; this is the server-side record).
        d["random_orbit_params"] = {
            str(body): {
                "inclination": p.inclination_deg,
                "sma": p.sma_m,
                "eccentricity": p.eccentricity,
            }
            for body, p in self.mission_builder.random_orbit_params.items()
        }
        # Goal contract mode. ``contract_thresholds`` is the client's watcher map
        # {completed-contract-count -> [threshold locations to report]}: when the
        # player's completed non-goal-contract count reaches a key, the client
        # reports those locations, releasing the goal contract item(s). Empty in
        # findable / starting. ``contracts_required``/``contracts_available`` are
        # carried for UT regen fidelity.
        d["goal_contract_mode"] = self.options.goal_contract_mode.value
        d["contracts_required"] = self.contracts_required
        d["contracts_available"] = self.options.contracts_available.value
        # Reward-slot repeats: carried so UT regen recomputes the same
        # non_goal_slot_count from the option (like every other option), rather
        # than inferring it from the contracts array length.
        d["contract_repeats"] = self.options.contract_repeats.value
        thresholds_map: dict[str, list[str]] = {}
        for loc_name, count, _item in self.contract_threshold_defs:
            thresholds_map.setdefault(str(count), []).append(loc_name)
        d["contract_thresholds"] = thresholds_map
        return d

    # ------------------------------------------------------------------
    # Universal Tracker hooks
    # ------------------------------------------------------------------

    @staticmethod
    def interpret_slot_data(slot_data: dict[str, Any] | None) -> dict[str, Any] | None:
        """UT hook: return slot_data to trigger regen with re_gen_passthrough."""
        return slot_data

    def _apply_slot_data(self, slot_data: dict[str, Any]) -> None:
        """Restore options from slot_data during UT regen."""
        self.options.goal.value = slot_data["goal"]
        self.options.difficulty.value = slot_data["difficulty"]
        self.options.start_with_launch_clamps.value = slot_data["start_with_launch_clamps"]
        # Restore the chosen home body.  ``starting_body`` in slot_data
        # is the canonical ``BodyName`` string (``"Kerbin"`` / ``"Mun"``
        # / ...) — the same value the client mod reads.  Reverse-map to
        # the option integer so any downstream consumer that reads
        # ``self.options.starting_body`` sees a consistent value, then
        # rebuild the MissionBuilder if the home actually changed.
        starting_body = slot_data.get("starting_body")
        if starting_body:
            from .options import StartingBody as _SB
            self.options.starting_body.value = getattr(
                _SB, f"option_{starting_body.lower()}"
            )
            if starting_body != self.mission_builder.home.value:
                new_home = BodyName(starting_body)
                self.mission_builder = MissionBuilder(home=new_home)
                self.location_builder = LocationBuilder(home=new_home)

        # Stash contracts so generate_early reconstructs the exact set rather
        # than re-randomizing (the contract pick is seed-RNG-derived).
        self._ut_contract_specs = [
            contracts.ContractSpec.from_slot_dict(entry)
            for entry in slot_data.get("contracts", [])
        ]

        # Goal contract mode: restore the options and the resolved X. The
        # threshold defs themselves recompute deterministically in
        # generate_early from the restored contracts + X (evaluate_contract is
        # pure), so only X needs carrying to avoid any auto-derivation drift.
        if "goal_contract_mode" in slot_data:
            self.options.goal_contract_mode.value = slot_data["goal_contract_mode"]
        if "contracts_available" in slot_data:
            self.options.contracts_available.value = slot_data["contracts_available"]
        if "contract_repeats" in slot_data:
            self.options.contract_repeats.value = slot_data["contract_repeats"]
        self._ut_contracts_required = slot_data.get("contracts_required")

        # Restore the exact RANDOM_ORBIT target orbits (re-rolling would diverge).
        rop = slot_data.get("random_orbit_params")
        if rop:
            self._ut_random_orbit_params = {
                BodyName(body): RandomOrbitParams(
                    inclination_deg=entry["inclination"],
                    sma_m=entry["sma"],
                    eccentricity=entry["eccentricity"],
                )
                for body, entry in rop.items()
            }

        # A custom goal isn't a single enum value — its body lists ARE the goal,
        # and resolve_goal_spec rebuilds the spec from those option values during
        # regen. Recover them from the goal *contracts* (the victory sentinels),
        # which carry (contract_type, body) structurally; goal_locations holds
        # contract display names that don't round-trip back to a body list.
        if slot_data["goal"] == Goal.option_custom:
            for attr, bodies in contracts.goal_body_lists_from_specs(
                    self._ut_contract_specs).items():
                getattr(self.options, attr).value = bodies

    def explain_rule(self, target_name: str, state: CollectionState) -> list[dict] | None:
        """UT hook: /explain <location> shows rocket design, /explain parts [filter] shows inventory."""
        from .bodies import DIFFICULTY_PROFILES
        from .capability import compute_capability_from_items, evaluate_mission_detailed
        from .capability_format import (
            CHECK_MAP, format_rocket_output, format_parts_list,
            format_contract_output,
        )

        # Sub-command: /explain parts [filter]
        if target_name.startswith("parts"):
            filter_text = target_name[5:].strip()
            item_counts: dict[str, int] = {}
            for name in self.item_name_to_id:
                count = state.count(name, self.player)
                if count > 0:
                    item_counts[name] = count
            if filter_text:
                filter_lower = filter_text.lower()
                item_counts = {k: v for k, v in item_counts.items() if filter_lower in k.lower()}
            lines = format_parts_list(item_counts)
            return [{"type": "text", "text": "\n".join(lines)}]

        # Default: look up location, compute capability, show rocket design.
        loc_obj = None
        for loc in self.multiworld.get_locations(self.player):
            if loc.name == target_name:
                loc_obj = loc
                break
        if loc_obj is None:
            return None  # fall back to UT default

        in_logic = loc_obj.can_reach(state)
        info = CHECK_MAP.get(target_name)
        difficulty_name = ["casual", "normal", "expert", "insane"][
            self.options.difficulty.value
        ]

        cap, flags = compute_capability_from_items(
            lambda name: state.count(name, self.player),
            difficulty_name,
            bool(self.options.start_with_launch_clamps.value),
            self.mission_builder,
        )

        # Contract locations aren't in CHECK_MAP — their feasibility needs the
        # extra-payload + mission-transform eval, not the bare milestone path — so
        # render them generically from the spec. Works for any contract type,
        # current or future, with no per-type handling here.
        contract_spec = self._contract_spec_for_name(target_name)
        if contract_spec is not None:
            lines = format_contract_output(
                contract_spec, in_logic,
                state.has(contract_spec.item_name, self.player),
                flags, DIFFICULTY_PROFILES[difficulty_name], difficulty_name,
                self.mission_builder, proxy=self._contract_uses_proxy(contract_spec),
            )
            return [{"type": "text", "text": "\n".join(lines)}]

        result = None
        if info is not None:
            diff = DIFFICULTY_PROFILES[difficulty_name]
            result = evaluate_mission_detailed(
                flags, diff, info.body_name, info.mission_type, info.crewed,
                self.mission_builder,
                threshold_km=info.threshold_km,
            )

        lines = format_rocket_output(
            target_name, in_logic, False, info, result, flags,
            difficulty_name, self.mission_builder,
            sounding_altitude_km=cap.sounding_altitude_km,
        )
        return [{"type": "text", "text": "\n".join(lines)}]

    def _contract_spec_for_name(self, name: str):
        """The ContractSpec whose location matches ``name``, or None. Looks up the
        world's own specs (the source of truth) rather than parsing the display
        name, so /explain covers every contract type without per-type handling."""
        for spec in (*self.contract_specs, *self.goal_contract_specs):
            if name in spec.location_names(self.non_goal_slot_count):
                return spec
        return None

    def _contract_uses_proxy(self, spec) -> bool:
        """True if this contract's access rule routes through the all-parts proxy
        instead of the physics gate (a goal contract on a model-infeasible body).
        Reads the set rules._set_contract_rules records when it sets the rule —
        the single source of truth — so /explain can't drift from the real gate."""
        return spec.contract_id in self._proxy_contract_ids

    def custom_ut_sort(self, region_label: str, location_label: str) -> str:
        """UT hook: sort by body order (ALL_BODIES), then tech tree, then KSC."""
        body_order = {b.name: f"A_{i:02d}" for i, b in enumerate(ALL_BODIES)}
        for prefix, sort_key in body_order.items():
            if location_label.startswith(prefix + " "):
                return f"{sort_key}_{location_label}"
        if region_label.startswith("Tech "):
            return f"B_{location_label}"
        if location_label.startswith(KSC_LOCATION_PREFIX) or location_label.startswith("Starting "):
            return f"Z_{location_label}"
        return f"C_{location_label}"

    def collect_item(self, state: CollectionState, item: Item, remove: bool = False) -> str | None:
        if item.advancement or item.name in CAPABILITY_ITEMS:
            return item.name
        return None

    def collect(self, state: CollectionState, item: Item) -> bool:
        change = super().collect(state, item)
        if change:
            state.ksp1_cap_stale[self.player] = True
        return change

    def remove(self, state: CollectionState, item: Item) -> bool:
        change = super().remove(state, item)
        if change:
            state.ksp1_cap_stale[self.player] = True
        return change
