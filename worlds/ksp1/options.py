from dataclasses import dataclass

from Options import Choice, ExcludeLocations, ItemsAccessibility, OptionSet, PerGameCommonOptions, Range, Toggle

from .bodies import ALL_BODIES

# All landable body names, derived from bodies.py (single source of truth).
LANDABLE_BODY_NAMES: frozenset[str] = frozenset(
    b.name for b in ALL_BODIES if b.can_land
)

ALL_BODY_NAMES: frozenset[str] = frozenset(b.name for b in ALL_BODIES)


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
    jool_moons_return      -- Return from each Jool moon (Laythe, Vall, Tylo,
                              Bop, Pol).  Home is filtered out, so a Laythe
                              start gives a tight 4-target Jool-system goal.
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
    option_jool_moons_return = 9
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


class OrbitBodies(OptionSet):
    """Bodies to reach orbit around (custom goal). Leave empty for preset goals."""
    display_name = "Orbit Bodies"
    valid_keys = ALL_BODY_NAMES


class FlybyBodies(OptionSet):
    """Bodies to perform a flyby of (custom goal). Leave empty for preset goals."""
    display_name = "Flyby Bodies"
    valid_keys = ALL_BODY_NAMES


class StartingBody(Choice):
    """
    The celestial body whose surface the player launches from.  Mun /
    Minmus / Laythe / etc. are landable bodies that the Selector mod
    can spawn KSC at.  Jool and Kerbol are excluded — gas giant and
    star, no surface.

    Most existing goals still work from non-Kerbin homes (returns,
    flag plants, sample returns are filtered for the new home).
    Goals whose only target *is* the home body become unwinnable and
    generation aborts with OptionError — e.g. ``mun_flag`` with
    ``home = mun`` is rejected at gen time.

    Default ``kerbin`` preserves the existing single-home behaviour.
    """
    display_name = "Starting Body"

    # Integer values are stable and alphabetised by body name so adding
    # a body later (a mod, an outer-planets pack) doesn't shift the
    # ones already in player yaml files.
    option_bop     = 0
    option_dres    = 1
    option_duna    = 2
    option_eeloo   = 3
    option_eve     = 4
    option_gilly   = 5
    option_ike     = 6
    option_kerbin  = 7
    option_laythe  = 8
    option_minmus  = 9
    option_moho    = 10
    option_mun     = 11
    option_pol     = 12
    option_tylo    = 13
    option_vall    = 14

    default = option_kerbin


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

    Empty by default: missions the dv model can't verify from the active
    home (Eve returns from any home, plus Tylo/Laythe returns from a
    Kerbin home, etc.) are gated via the "all progression items
    collected" proxy rule (see ``MODEL_INFEASIBLE_LOCATIONS`` in
    ``data/feasibility.py``).  That mechanism already prevents fill
    failures without taking the locations out of the progression pool.

    The previous Kerbin-shaped hardcoded default (Eve / Tylo / Laythe
    returns) made non-Kerbin home configs trip over their own goal — a
    Laythe-home ``jool_moons_return`` seed needs Tylo Return as a goal
    target, but the default excluded it.  Failing early at gen time on
    that mismatch (see ``_validate_goal_not_excluded`` in ``world.py``)
    catches user errors more clearly than the previous fill-failure
    symptom.
    """
    default = frozenset()


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


class KSP1Accessibility(ItemsAccessibility):
    """
    Set rules for reachability of locations.

    KSP1 defaults to **minimal**: only locations on the path to your goal need
    to be reachable. Useful items (parts) may end up at locations the player
    can't reach — they're not required to win, just collectibles. This avoids
    `inaccessible_location_rules` filler-only marking that caused fill failures
    when only ~91 of ~530 locations were initially reachable from sphere-0.

    See bug 074 for the full rationale.
    """
    default = ItemsAccessibility.option_minimal
    __doc__ = ItemsAccessibility.__doc__


class ExcludeLateTechTree(Toggle):
    """
    Exclude tier-8 tech tree locations from containing progression items.

    Tier-8 nodes require massive amounts of science to unlock.  Enabling this
    prevents late-game science grind from being required to complete the seed.
    Disable for Complete Tech Tree goal or challenge runs.
    """
    display_name = "Exclude Late Tech Tree"
    default = 1


class ProgressiveLaunchPad(Toggle):
    """
    Gate buildable rocket launch mass through Progressive Launch Pad items.

    Mirrors KSP career mode launch pad upgrades. The pool gets 3 copies and
    collecting them raises the launch-mass cap through tiers
    (100 → 200 → 500 → unlimited tonnes by default — see
    PROGRESSIVE_LAUNCH_PAD_CAPS).

    Creates a real progression gate on broad goals like Standard Sample
    Returns — you can have all the parts but still need a bigger launch
    pad before tackling Vall / Duna / heavy missions.

    Disabling this option can result in more wide-open seeds where you can
    sometimes reach all bodies at once after a small bootstrap kit. Leave
    enabled if you want a sequential progression journey; disable if you
    find the mass cap annoying.
    """
    display_name = "Progressive Launch Pad"
    default = 1


@dataclass
class KSP1Options(PerGameCommonOptions):
    goal: Goal
    starting_body: StartingBody
    difficulty: Difficulty
    start_with_launch_clamps: StartWithLaunchClamps
    item_pacing: ItemPacing
    accessibility: KSP1Accessibility
    exclude_locations: KSP1ExcludeLocations
    exclude_late_tech_tree: ExcludeLateTechTree
    progressive_launch_pad: ProgressiveLaunchPad
    flag_bodies: FlagBodies
    return_bodies: ReturnBodies
    sample_return_bodies: SampleReturnBodies
    orbit_bodies: OrbitBodies
    flyby_bodies: FlybyBodies
