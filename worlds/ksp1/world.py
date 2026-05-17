from typing import Any

from BaseClasses import CollectionState, Item, MultiWorld, Tutorial
from Options import OptionError
from worlds.AutoWorld import LogicMixin, WebWorld, World

from . import items, locations, regions, rules
from .rules import GoalSpec, resolve_goal_spec, goal_spec_location_names
from .capability import CAPABILITY_ITEMS, RocketCapability
from .data.feasibility import MODEL_INFEASIBLE_LOCATIONS
from .bodies import ALL_BODIES, BodyName, MissionBuilder, MissionType
from .items import ITEM_NAME_TO_ID, PROGRESSIVE_LAUNCH_PAD_CAPS, _FILLER_ITEMS
from .parts import PROGRESSIVE_PART_TIERS
from .locations import (
    ALL_EVENTS, EventName, KSC_BIOMES, KSC_LOCATION_PREFIX,
    LOCATION_NAME_TO_ID, LocationBuilder, MAX_TECH_SLOTS, MissionLocation,
    STARTING_INV_COUNTS, TECH_SLOTS_BY_DIFFICULTY, TechTreeLocation,
    effective_starting_inv_count,
)
from .options import KSP1Options
from .tech_tree import MAX_TIER, NODES_BY_TIER, TECH_NODES, TIER_TO_BAND


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

    # Per progressive tier, the randomly-selected representative part name.
    # Set during create_items(); included in slot_data for the client.
    progressive_representatives: dict[str, dict[int, str]]

    # Resolved goal specification (preset or custom).
    goal_spec: GoalSpec

    # Mission graph builder for this world.  Pinned to Kerbin here; owned
    # by the world so its data lifetime matches the rest of the per-world
    # state.
    mission_builder: MissionBuilder

    # Per-world home-body location set (12 specials: first launch / landing /
    # crash, altitude milestones, splashdown, first staging).  Owned alongside
    # ``mission_builder`` so the two stay in sync on the same home.
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

    def generate_early(self) -> None:
        """Resolve goal spec and apply ExcludeLateTechTree."""
        self.capability_cache = {}
        # ``starting_body`` option keys are lowercase BodyName values
        # (``option_mun`` → key ``"mun"`` → ``BodyName.MUN``).  The
        # title-case round-trip rebuilds the canonical ``StrEnum`` value.
        home = BodyName(self.options.starting_body.current_key.title())
        self.mission_builder = MissionBuilder(home=home)
        self.location_builder = LocationBuilder(home=home)

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
        if self.options.exclude_late_tech_tree:
            late_tier_locs: set[str] = {
                str(TechTreeLocation(node.display_name, slot))
                for node in NODES_BY_TIER.get(MAX_TIER, [])
                for slot in range(1, MAX_TECH_SLOTS + 1)
            }
            self.options.exclude_locations.value |= late_tier_locs

    def create_regions(self) -> None:
        regions.create_all_regions(self)
        locations.create_all_locations(self)
        rules.create_victory_location(self)

    def create_items(self) -> None:
        items.create_all_items(self)

    def set_rules(self) -> None:
        rules.set_all_rules(self)
        rules.set_completion_condition(self, self.goal_spec)

    def pre_fill(self) -> None:
        from .sphere_ladder import apply_sphere_ladder
        apply_sphere_ladder(self)

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
        d["tech_slots_per_node"] = TECH_SLOTS_BY_DIFFICULTY[self.options.difficulty.value]
        d["node_bands"] = {n.node_id: TIER_TO_BAND[n.tier] for n in TECH_NODES}
        d["goal_locations"] = goal_spec_location_names(self.goal_spec)
        d["goal_display_name"] = self.goal_spec.display_name
        # Progressive tier data for the client mod
        d["progressive_tiers"] = {
            name: {str(t): parts for t, parts in tiers.items()}
            for name, tiers in PROGRESSIVE_PART_TIERS.items()
        }
        # Server-selected representative per progressive tier
        d["progressive_representatives"] = {
            name: {str(t): rep for t, rep in reps.items()}
            for name, reps in self.progressive_representatives.items()
        }
        # Authoritative data for C# client — eliminates hardcoded dicts.
        d["event_scales"] = {e.name: e.scale for e in ALL_EVENTS}
        d["tech_display_names"] = {n.node_id: n.display_name for n in TECH_NODES}
        # Full biome_key -> AP location name map.  The client populates its
        # detection table directly from this; it has no hardcoded copy.
        d["ksc_biome_locations"] = {
            key: KSC_LOCATION_PREFIX + name for key, name in KSC_BIOMES
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

        if slot_data["goal"] == 99:  # Goal.option_custom
            flag_bodies: set[str] = set()
            return_bodies: set[str] = set()
            sample_return_bodies: set[str] = set()
            for loc_str in slot_data.get("goal_locations", []):
                parsed = MissionLocation.parse(loc_str)
                if parsed is None:
                    continue
                if parsed.event == EventName.FLAG_PLANT:
                    flag_bodies.add(parsed.body)
                elif parsed.event == EventName.SAMPLE_RETURN:
                    sample_return_bodies.add(parsed.body)
                elif parsed.event == EventName.RETURN:
                    return_bodies.add(parsed.body)
            self.options.flag_bodies.value = flag_bodies
            self.options.return_bodies.value = return_bodies
            self.options.sample_return_bodies.value = sample_return_bodies

        # Stash progressive reps so create_items() uses them instead of re-randomizing.
        self._ut_progressive_representatives = {
            name: {int(t): rep for t, rep in reps.items()}
            for name, reps in slot_data.get("progressive_representatives", {}).items()
        }

    def explain_rule(self, target_name: str, state: CollectionState) -> list[dict] | None:
        """UT hook: /explain <location> shows rocket design, /explain parts [filter] shows inventory."""
        from .bodies import DIFFICULTY_PROFILES
        from .capability import compute_capability_from_items, evaluate_mission_detailed
        from .capability_format import (
            CHECK_MAP, format_rocket_output, format_parts_list,
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

        rep_names = frozenset(
            rep
            for tiers in self.progressive_representatives.values()
            for rep in tiers.values()
        )
        cap, flags = compute_capability_from_items(
            lambda name: state.count(name, self.player),
            difficulty_name,
            bool(self.options.start_with_launch_clamps.value),
            self.mission_builder,
            rep_names=rep_names,
        )

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
