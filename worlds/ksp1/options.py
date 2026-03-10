from dataclasses import dataclass

from Options import Choice, PerGameCommonOptions


class Goal(Choice):
    """
    The victory condition for this run.
    """
    display_name = "Goal"

    option_mun_landing = 0
    option_mun_and_minmus = 1
    option_inner_system = 2
    option_jool_system = 3
    option_all_bodies = 4

    default = option_mun_landing


class Difficulty(Choice):
    """
    Controls delta-V margins and hardware requirement strictness.
    Casual: generous margins, all hardware gates enforced.
    Normal: default margins and gates.
    Expert: tight margins, some hardware gates relaxed.
    Insane: exact delta-V values, no margins.
    """
    display_name = "Difficulty"

    option_casual = 0
    option_normal = 1
    option_expert = 2
    option_insane = 3

    default = option_normal


@dataclass
class KSP1Options(PerGameCommonOptions):
    goal: Goal
    difficulty: Difficulty
