from dataclasses import dataclass

from Options import Choice, ExcludeLocations, OptionSet, PerGameCommonOptions, Toggle

from .bodies import ALL_BODIES

# All landable body names, derived from bodies.py (single source of truth).
LANDABLE_BODY_NAMES: frozenset[str] = frozenset(
    b.name for b in ALL_BODIES if b.can_land
)


class Goal(Choice):
    """
    The victory condition for this run.

    duna_return            -- Return a vessel (or crew) from Duna.
    eeloo_return           -- Return a vessel (or crew) from Eeloo.
    flag_every_body        -- Plant a flag on all 15 landable bodies (crewed).
    standard_returns       -- Return from 11 bodies (excl. Eve, Tylo, Laythe).
    standard_sample_returns -- Crewed sample return from the same 11 bodies.
    complete_tech_tree     -- Purchase all 62 tech tree nodes with science.
    eve_return             -- Return a vessel (or crew) from Eve (challenge).
    mun_flag               -- Plant a flag on the Mun.
    mun_sample_return      -- Crewed sample return from the Mun.
    custom                 -- Build a goal from the body-list options below.
    """
    display_name = "Goal"

    option_duna_return = 0
    option_eeloo_return = 1
    option_flag_every_body = 2
    option_standard_returns = 3
    option_standard_sample_returns = 4
    option_complete_tech_tree = 5
    option_eve_return = 6
    option_mun_flag = 7
    option_mun_sample_return = 8
    option_custom = 99

    default = option_duna_return


class FlagBodies(OptionSet):
    """Bodies to plant flags on (custom goal). Leave empty for preset goals."""
    display_name = "Flag Bodies"
    valid_keys = LANDABLE_BODY_NAMES


class ReturnBodies(OptionSet):
    """Bodies to return from (custom goal). Leave empty for preset goals."""
    display_name = "Return Bodies"
    valid_keys = LANDABLE_BODY_NAMES


class SampleReturnBodies(OptionSet):
    """Bodies to sample-return from (custom goal). Leave empty for preset goals."""
    display_name = "Sample Return Bodies"
    valid_keys = LANDABLE_BODY_NAMES


class Difficulty(Choice):
    """
    Controls delta-V margins and hardware requirement strictness.
    Also controls KSC starting slots and tech tree slots per node.

    casual  -- Generous margins; 20 KSC starts, 4 tech slots/node.
    normal  -- Default margins; 15 KSC starts, 4 tech slots/node.
    expert  -- Tight margins;   10 KSC starts, 3 tech slots/node.
    insane  -- Exact delta-V;    5 KSC starts, 2 tech slots/node.
    """
    display_name = "Difficulty"

    option_casual = 0
    option_normal = 1
    option_expert = 2
    option_insane = 3

    default = option_normal


class StartWithLaunchClamps(Toggle):
    """
    Start the run with Launch Clamps already collected.

    Launch Clamps are required to leave Kerbin's SOI.  Enabling this removes
    that gate, allowing interplanetary missions from the very first item check.
    Disable if you want the clamp to be a meaningful progression unlock.
    """
    display_name = "Start With Launch Clamps"
    default = 1


class KSP1ExcludeLocations(ExcludeLocations):
    """
    Locations that are excluded from containing progression items by default.

    The default set excludes return missions from Eve, Tylo, and Laythe.
    Eve returns require all progression parts (surface ascent is extremely demanding).
    Tylo and Laythe returns are gated normally through the capability system.
    """
    default = frozenset({
        # Eve returns (3 checks each); requires all progression parts
        "Eve Return 1", "Eve Return 2", "Eve Return 3",
        "Eve Sample Return 1", "Eve Sample Return 2", "Eve Sample Return 3",
        # Tylo returns (3 checks each)
        "Tylo Return 1", "Tylo Return 2", "Tylo Return 3",
        "Tylo Sample Return 1", "Tylo Sample Return 2", "Tylo Sample Return 3",
        # Laythe returns (3 checks each)
        "Laythe Return 1", "Laythe Return 2", "Laythe Return 3",
        "Laythe Sample Return 1", "Laythe Sample Return 2", "Laythe Sample Return 3",
    })


class ItemPacing(Choice):
    """
    Controls whether high-impact items are restricted from early locations.

    off     -- No restrictions. Any item can appear anywhere.
    gentle  -- Tier 2 items (big engines, decouplers, large tanks) excluded
               from Starting Inventory and KSC biome locations.
    strict  -- Additionally restricts tier 2 from early tech tree (tiers 1-3).
    """
    display_name = "Item Pacing"

    option_off = 0
    option_gentle = 1
    option_strict = 2

    default = option_gentle


class ExcludeLateTechTree(Toggle):
    """
    Exclude tier-8 tech tree locations from containing progression items.

    Tier-8 nodes require massive amounts of science to unlock.  Enabling this
    prevents late-game science grind from being required to complete the seed.
    Disable for Complete Tech Tree goal or challenge runs.
    """
    display_name = "Exclude Late Tech Tree"
    default = 1


@dataclass
class KSP1Options(PerGameCommonOptions):
    goal: Goal
    difficulty: Difficulty
    start_with_launch_clamps: StartWithLaunchClamps
    item_pacing: ItemPacing
    exclude_locations: KSP1ExcludeLocations
    exclude_late_tech_tree: ExcludeLateTechTree
    flag_bodies: FlagBodies
    return_bodies: ReturnBodies
    sample_return_bodies: SampleReturnBodies
