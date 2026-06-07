from dataclasses import dataclass

from Options import Choice, ExcludeLocations, ItemsAccessibility, NamedRange, OptionDict, OptionSet, PerGameCommonOptions, Range, Toggle

from .bodies import ALL_BODIES, BodyName
from .contracts import ContractType, NON_GOAL_TYPES

# All landable body names, derived from bodies.py (single source of truth).
LANDABLE_BODY_NAMES: frozenset[str] = frozenset(
    b.name for b in ALL_BODIES if b.can_land
)

ALL_BODY_NAMES: frozenset[str] = frozenset(b.name for b in ALL_BODIES)

# Pre-canned random pools for StartingBody.  Each key is the lowercase
# option_<name> stem; the value is the pool the option resolves to.
# Resolution happens once in world.generate_early via world.random.choice
# (deterministic from seed) and overwrites the option with the picked
# concrete body, so all downstream code sees a normal single-body value.
#
# AP's standard YAML weighted-random over Choice options also works on
# any of these keys (and on the concrete body keys), so users can write
# e.g. ``kerbin: 40, duna: 20, laythe: 40`` directly with no help from
# us — pools just expose curated subsets as quick picks.
STARTING_BODY_POOLS: dict[str, frozenset[BodyName]] = {
    "atmospheric": frozenset({BodyName.KERBIN, BodyName.DUNA, BodyName.LAYTHE}),
    "standard": frozenset({
        BodyName.KERBIN, BodyName.DUNA, BodyName.LAYTHE,
        BodyName.MOHO, BodyName.EELOO,
    }),
    "planets": frozenset({
        BodyName.MOHO, BodyName.KERBIN, BodyName.DUNA,
        BodyName.DRES, BodyName.EELOO,
    }),
    "all": frozenset(
        BodyName(b.name) for b in ALL_BODIES
        if b.can_land and b.name != BodyName.EVE
    ),
}


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

    Pool keys (resolved to a concrete body at generation time using the
    seed RNG) are quick picks for randomized starts:

    atmospheric -- Kerbin, Duna, Laythe.
    standard    -- Kerbin, Duna, Laythe, Moho, Eeloo.
    planets     -- Moho, Kerbin, Duna, Dres, Eeloo (planets only).
    all         -- Every landable body except Eve.  Includes Tylo and
                   Laythe; expect punishing seeds.

    For custom weights, use the standard AP weighted-random YAML form
    over the concrete body keys, e.g. ``kerbin: 40, duna: 20,
    laythe: 40``.  Pool keys can be weighted the same way.
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

    # Pool options — keep IDs well above the concrete-body range so a
    # future body addition can slot in without colliding.  Keys must
    # match STARTING_BODY_POOLS above.
    option_atmospheric = 100
    option_standard    = 101
    option_planets     = 102
    option_all         = 103

    default = option_kerbin


class Difficulty(Choice):
    """
    Controls delta-V margins and hardware requirement strictness.

    Also sets defaults for Tech Slots Per Node, Starting Inventory Count,
    and Science Safety Factor — each of which can be overridden independently.

    casual  -- Generous margins; 20 starts, 4 tech slots/node, 50% science.
    normal  -- Default margins;  15 starts, 4 tech slots/node, 70% science.
    expert  -- Tight margins;    10 starts, 3 tech slots/node, 85% science.
    insane  -- Exact delta-V;     5 starts, 2 tech slots/node, 100% science.
    """
    display_name = "Difficulty"

    option_casual = 0
    option_normal = 1
    option_expert = 2
    option_insane = 3

    default = option_normal


class TechSlotsPerNode(NamedRange):
    """
    Number of AP location slots created per tech tree node (62 nodes total).

    More slots = larger location pool, looser fill.  Fewer slots = tighter
    item pool tension and faster progression pacing.

    WARNING: Lowering this shrinks the location pool and can cause fill
    failures on long goals (e.g. standard_sample_returns, flag_every_body).
    If generation fails: reroll with a new seed, or raise this value, or
    raise Starting Inventory Count to give the fill algorithm more room.
    Short goals (mun_flag, duna_return) are unaffected.

    auto -- Derived from Difficulty (casual/normal=4, expert=3, insane=2).
    1..4 -- Explicit override.
    """
    display_name = "Tech Slots Per Node"
    range_start = 1
    range_end = 4
    default = -1
    special_range_names = {"auto": -1}


class StartingInventoryCount(NamedRange):
    """
    Number of zero-requirement bootstrap locations available at run start.

    These auto-check on connect — the AP fill algorithm uses them to seed
    your initial parts kit.  Higher = easier bootstrap, more wide-open
    early game.  Lower = scarcer starts, tighter pacing.

    The Progressive Launch Pad bonus (+3, capped at 20) still applies on
    top of this when enabled.

    WARNING: Lowering this shrinks the location pool and can cause fill
    failures on long goals (e.g. standard_sample_returns, flag_every_body).
    If generation fails: reroll with a new seed, or raise this value, or
    raise Tech Slots Per Node to give the fill algorithm more room.
    Short goals (mun_flag, duna_return) are unaffected.

    auto  -- Derived from Difficulty (20/15/10/5).
    0..20 -- Explicit override.
    """
    display_name = "Starting Inventory Count"
    range_start = 0
    range_end = 20
    default = -1
    special_range_names = {"auto": -1}


class ScienceSafetyFactor(NamedRange):
    """
    Percentage of estimated accessible science counted toward tech tree gates.

    The capability system estimates how much science the player could earn
    from reachable bodies; this factor scales that estimate.  Lower = stricter
    (tech tiers unlock later, more bodies must be reachable first).  Higher =
    more permissive.  This is the single most impactful knob for tech tree
    progression pacing.

    Does NOT affect fill success — only tech tree gating.  Safe to tune.

    auto    -- Derived from Difficulty (casual=50, normal=70, expert=85, insane=100).
    0..100  -- Explicit percentage override.
    """
    display_name = "Science Safety Factor"
    range_start = 0
    range_end = 100
    default = -1
    special_range_names = {"auto": -1}


class StartWithLaunchClamps(Toggle):
    """
    Start the run with Launch Clamps already collected.

    Launch Clamps stabilize tall rockets on the pad.  They are not required
    for any mission — the capability system never gates on them — but
    enabling this gives the player access to one from the start of the run
    instead of waiting for the part item.
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


class ContractTypeWeights(OptionDict):
    """
    Relative weight of each contract mission type in the non-goal contract pool.

    Contracts are paced into the run as items; completing one (a native KSP
    contract injected by the client) checks an AP location. A weight of 0
    disables that type entirely. Weights are relative — {"mine_ore": 2} alone
    behaves the same as {"mine_ore": 1}; mixing types biases the random pick.

    Only ever-achievable (type, body) combinations are placed.
    """
    display_name = "Contract Type Weights"
    valid_keys = frozenset(str(ct) for ct in NON_GOAL_TYPES)
    default = {str(ct): 1 for ct in NON_GOAL_TYPES}


class NonGoalContractCount(NamedRange):
    """
    Total number of non-goal contracts placed into the seed.

    auto  -- Derived from Difficulty (12/10/8/6).
    0..40 -- Explicit override. Capped at the number of ever-achievable
             (enabled-type, body) combinations available in the seed.
    """
    display_name = "Non-Goal Contract Count"
    range_start = 0
    range_end = 40
    default = -1
    special_range_names = {"auto": -1}


class AllowMissionsHarderThanGoal(Toggle):
    """
    Allow contracts whose mission is harder than your goal mission.

    On (default): the full ever-achievable contract range is eligible, so a
    contract can be significantly harder than the goal and (once goals are
    contracts) can even end up gating your goal. Off: contracts are capped at
    the goal's difficulty — e.g. a Duna-return goal won't hand you a Tylo mining
    contract. Difficulty is compared by intrinsic mission delta-v.

    Independent of the home-system-local invariant: a home-local goal (e.g.
    Mun flag) never gets out-of-system contracts regardless of this setting.
    """
    display_name = "Allow Missions Harder Than Goal"
    default = 1


@dataclass
class KSP1Options(PerGameCommonOptions):
    goal: Goal
    starting_body: StartingBody
    difficulty: Difficulty
    tech_slots_per_node: TechSlotsPerNode
    starting_inventory_count: StartingInventoryCount
    science_safety_factor: ScienceSafetyFactor
    start_with_launch_clamps: StartWithLaunchClamps
    item_pacing: ItemPacing
    accessibility: KSP1Accessibility
    exclude_locations: KSP1ExcludeLocations
    exclude_late_tech_tree: ExcludeLateTechTree
    progressive_launch_pad: ProgressiveLaunchPad
    contract_type_weights: ContractTypeWeights
    non_goal_contract_count: NonGoalContractCount
    allow_missions_harder_than_goal: AllowMissionsHarderThanGoal
    flag_bodies: FlagBodies
    return_bodies: ReturnBodies
    sample_return_bodies: SampleReturnBodies
    orbit_bodies: OrbitBodies
    flyby_bodies: FlybyBodies
