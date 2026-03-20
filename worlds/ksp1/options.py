from dataclasses import dataclass

from Options import Choice, ExcludeLocations, PerGameCommonOptions, Toggle


class Goal(Choice):
    """
    The victory condition for this run.

    duna_return            -- Return a vessel (or crew) from Duna.
    eeloo_return           -- Return a vessel (or crew) from Eeloo.
    flag_every_body        -- Plant a flag on all 15 landable bodies (crewed).
    standard_returns       -- Return from 11 bodies (excl. Eve, Tylo, Laythe).
    standard_sample_returns -- Crewed sample return from the same 11 bodies.
    complete_tech_tree     -- Purchase all 43 tech tree nodes with science.
    eve_return             -- Return a vessel (or crew) from Eve (challenge).
    """
    display_name = "Goal"

    option_duna_return = 0
    option_eeloo_return = 1
    option_flag_every_body = 2
    option_standard_returns = 3
    option_standard_sample_returns = 4
    option_complete_tech_tree = 5
    option_eve_return = 6

    default = option_duna_return


class Difficulty(Choice):
    """
    Controls delta-V margins and hardware requirement strictness.
    Also controls how many KSC starting location slots are created (more
    slots = easier early game).

    casual  -- Generous margins; 20 KSC starting slots.
    normal  -- Default margins; 15 KSC starting slots.
    expert  -- Tight margins;   10 KSC starting slots.
    insane  -- Exact delta-V;    5 KSC starting slots.
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

    The default set excludes return missions from Eve, Tylo, and Laythe —
    the three hardest bodies to return from.  Players can remove these
    exclusions to make those locations part of progression.
    """
    default = frozenset({
        # Eve returns (scale 2 = 2 checks each)
        "Eve Return 1", "Eve Return 2",
        "Eve Sample Return 1", "Eve Sample Return 2",
        # Tylo returns (scale 3 = 3 checks each)
        "Tylo Return 1", "Tylo Return 2", "Tylo Return 3",
        "Tylo Sample Return 1", "Tylo Sample Return 2", "Tylo Sample Return 3",
        # Laythe returns (scale 3 = 3 checks each)
        "Laythe Return 1", "Laythe Return 2", "Laythe Return 3",
        "Laythe Sample Return 1", "Laythe Sample Return 2", "Laythe Sample Return 3",
    })


class ExcludeLateTechTree(Toggle):
    """
    Exclude tier-9 tech tree locations from containing progression items.

    Tier-9 nodes require massive amounts of science to unlock.  Enabling this
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
    exclude_locations: KSP1ExcludeLocations
    exclude_late_tech_tree: ExcludeLateTechTree
