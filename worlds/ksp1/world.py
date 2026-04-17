from typing import Any

from BaseClasses import CollectionState, Item, MultiWorld
from worlds.AutoWorld import LogicMixin, WebWorld, World

from . import items, locations, regions, rules
from .capability import RocketCapability
from .items import ITEM_NAME_TO_ID
from .parts import PROGRESSIVE_PART_TIERS
from .locations import LOCATION_NAME_TO_ID, MAX_TECH_SLOTS, TECH_SLOTS_BY_DIFFICULTY
from .options import KSP1Options
from .tech_tree import MAX_TIER, NODES_BY_TIER, TECH_NODES, TIER_TO_BAND


class KSP1State(LogicMixin):
    """Inject per-player stale flag and cached result onto CollectionState."""
    ksp1_cap_stale: dict[int, bool]
    ksp1_cap_result: dict[int, RocketCapability]

    def init_mixin(self, multiworld: MultiWorld) -> None:
        self.ksp1_cap_stale = {p: True for p in multiworld.get_game_players("Kerbal Space Program")}
        self.ksp1_cap_result = {}


class KSP1WebWorld(WebWorld):
    theme = "ocean"
    # TODO: Add tutorial entries once setup docs are written.


class KSP1World(World):
    """
    Kerbal Space Program is a space flight simulation game where you design and
    fly rockets to explore the Kerbol system.  Parts are randomized across the
    multiworld, and mission completions are the location checks.
    """

    game = "Kerbal Space Program"
    web = KSP1WebWorld()

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

    def generate_early(self) -> None:
        """Apply ExcludeLateTechTree to the exclude_locations option set."""
        self.capability_cache = {}
        if self.options.exclude_late_tech_tree:
            late_tier_locs: set[str] = {
                f"{node.display_name} {slot}"
                for node in NODES_BY_TIER.get(MAX_TIER, [])
                for slot in range(1, MAX_TECH_SLOTS + 1)
            }
            self.options.exclude_locations.value |= late_tier_locs

    def create_regions(self) -> None:
        regions.create_all_regions(self)
        locations.create_all_locations(self)

    def create_items(self) -> None:
        items.create_all_items(self)

    def set_rules(self) -> None:
        rules.set_all_rules(self)
        rules.set_completion_condition(self)

    def create_item(self, name: str) -> items.KSP1Item:
        return items.create_item(self, name)

    def get_filler_item_name(self) -> str:
        return items.get_filler_item_name(self)

    def fill_slot_data(self) -> dict[str, Any]:
        d = self.options.as_dict("goal", "difficulty", "start_with_launch_clamps", "item_pacing")
        d["tech_slots_per_node"] = TECH_SLOTS_BY_DIFFICULTY[self.options.difficulty.value]
        d["node_bands"] = {n.node_id: TIER_TO_BAND[n.tier] for n in TECH_NODES}
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
        return d

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
