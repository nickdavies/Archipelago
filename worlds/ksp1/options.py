from dataclasses import dataclass

from Options import Choice, DefaultOnToggle, ExcludeLocations, ItemsAccessibility, NamedRange, OptionDict, OptionSet, PerGameCommonOptions, Range, Toggle, Visibility

from .bodies import ALL_BODIES, BodyName
from .contracts import ContractType, NON_GOAL_TYPES
from .parts.packs import OPTIONAL_PACKS, DEFAULT_ENABLED_OPTIONAL_PACKS

# All landable body names, derived from bodies.py (single source of truth).
LANDABLE_BODY_NAMES: frozenset[str] = frozenset(
    b.name for b in ALL_BODIES if b.can_land
)

# Bodies that can be orbited / flown by: every body except the star (Kerbol).
# The star is not a mission destination, so orbit/flyby goal lists must exclude
# it — a goal contract there has no registered location (KeyError at gen).
ORBITABLE_BODY_NAMES: frozenset[str] = frozenset(
    b.name for b in ALL_BODIES if b.is_orbitable
)

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
    mun_flag               -- Plant a flag on the Mun.
    mun_sample_return      -- Crewed sample return from the Mun.
    jool_moons_return      -- Return from each Jool moon (Laythe, Vall, Tylo,
                              Bop, Pol).  Home is filtered out, so a Laythe
                              start gives a tight 4-target Jool-system goal.
    random_contracts       -- No destination goal: complete X of your available
                              contracts, then plant a flag at home to win. Only
                              valid with goal_contract_mode = count or
                              progressive_unlock.
    custom                 -- Build a goal from the body-list options below.
    """
    display_name = "Goal"

    option_duna_return = 0
    option_eeloo_return = 1
    option_flag_every_body = 2
    option_standard_returns = 3
    option_standard_sample_returns = 4
    option_complete_tech_tree = 5
    # eve_return retired: Eve ascent (~9315 m/s) is model-infeasible — the
    # capability solver can't verify a winnable rocket, so it was never a sound
    # goal.  Renumbered (this release is not backward compatible).
    option_mun_flag = 6
    option_mun_sample_return = 7
    option_jool_moons_return = 8
    option_random_contracts = 9
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
    valid_keys = ORBITABLE_BODY_NAMES


class FlybyBodies(OptionSet):
    """Bodies to perform a flyby of (custom goal). Leave empty for preset goals."""
    display_name = "Flyby Bodies"
    valid_keys = ORBITABLE_BODY_NAMES


class EnabledPartPacks(OptionSet):
    """Optional part packs to include in the seed. Stock parts are always
    available; enable the DLC/expansion packs you own here (disable a pack to
    leave its parts out of the run). Defaults to every shipped optional pack."""
    display_name = "Enabled Part Packs"
    valid_keys = frozenset(OPTIONAL_PACKS)
    default = frozenset(DEFAULT_ENABLED_OPTIONAL_PACKS)


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

    ``eve`` launches from the 6,140 m mesa pad (~8,996 m/s ascent);
    heavy departures are lifted across up to three launches and docked
    in Eve orbit (multi-launch assembly).  The deepest casual-margin
    Return/Sample Return targets stay out of logic and route through the
    proxy; comfortable/expert are fully in logic.  Expect the hardest
    seeds in the game.

    Default ``kerbin`` preserves the existing single-home behaviour.

    Pool keys (resolved to a concrete body at generation time using the
    seed RNG) are quick picks for randomized starts:

    atmospheric -- Kerbin, Duna, Laythe.
    standard    -- Kerbin, Duna, Laythe, Moho, Eeloo.
    planets     -- Moho, Kerbin, Duna, Dres, Eeloo (planets only).
    all         -- Every landable body except Eve (Eve is opt-in only via
                   the explicit ``eve`` key).  Includes Tylo and Laythe;
                   expect punishing seeds.

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
    Sets pacing defaults: Tech Slots Per Node, Starting Inventory Count,
    Science Safety Factor, and contract pacing — each overridable independently.

    Delta-V margins and hardware strictness are NOT set here — those are the
    Physics Difficulty option (which defaults to follow this).

    casual  -- 20 starts, 4 tech slots/node, 50% science; physics 'generous'.
    normal  -- 15 starts, 4 tech slots/node, 70% science; physics 'comfortable'.
    expert  -- 10 starts, 3 tech slots/node, 85% science; physics 'small'.
    """
    display_name = "Difficulty"

    option_casual = 0
    option_normal = 1
    option_expert = 2

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

    auto -- Derived from Difficulty (casual/normal=4, expert=3).
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

    auto  -- Derived from Difficulty (20/15/10).
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

    auto    -- Derived from Difficulty (casual=50, normal=70, expert=85).
    0..100  -- Explicit percentage override.
    """
    display_name = "Science Safety Factor"
    range_start = 0
    range_end = 100
    default = -1
    special_range_names = {"auto": -1}


class PhysicsDifficulty(Choice):
    """
    Delta-V margins and hardware strictness used by the capability model.

    This is the physics axis only — it does NOT touch tech-slot, starting-
    inventory, science, or contract pacing (those follow Difficulty).  The
    levels are hand-curated profiles, not a numeric dial: margins shrink while
    other knobs (e.g. drag credit) move the other way, so there's no meaningful
    value "between" two levels.

    auto        -- Follow Difficulty (casual->generous, normal->comfortable,
                   expert->small).  The default.
    generous    -- Largest margins; most forgiving (casual physics).
    comfortable -- Default margins (normal physics).
    small       -- Tight margins (expert physics).
    zero        -- No margin at all: every dv budget must close exactly, no
                   plane-change cushion, lowest TWR floors.  "Fly it perfectly."
                   Maps to no Difficulty — opt in deliberately.
    """
    display_name = "Physics Difficulty"

    option_auto = 0
    option_generous = 1
    option_comfortable = 2
    option_small = 3
    option_zero = 4

    default = option_auto


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


class GuaranteeScienceRover(DefaultOnToggle):
    """
    Internal starting-convenience floor (not a player-facing option).

    Precollects a complete science-capable kit — a control source (a probe or
    a capsule, rolled per seed), a wheel, a power source, and a science
    instrument — so the KSC-biome science and splashdown checks are doable from
    the moment you connect, regardless of what your ascent parts rolled or where
    the multiworld scattered them.  The location rules stay real physics and
    generation is solvable without it; it exists so tests can build a genuinely
    empty starting state to verify those rules in isolation.
    """
    display_name = "Guarantee Science Rover"
    visibility = Visibility.none


class KSP1ExcludeLocations(ExcludeLocations):
    """
    Locations that are excluded from containing progression items by default.

    Empty by default: missions the dv model can't verify from the active
    home at the seed's difficulty (e.g. Eve surface returns) are gated via
    the "all progression items collected" proxy rule (see
    ``MODEL_INFEASIBLE_BASE`` in ``data/feasibility.py``).
    That mechanism already prevents fill failures without taking the
    locations out of the progression pool.

    The previous Kerbin-shaped hardcoded default (Eve / Tylo / Laythe
    returns) made non-Kerbin home configs trip over their own goal — a
    Laythe-home ``jool_moons_return`` seed needs Tylo Return as a goal
    target, but the default excluded it.  Failing early at gen time on
    that mismatch (see ``_validate_goal_not_excluded`` in ``world.py``)
    catches user errors more clearly than the previous fill-failure
    symptom.
    """
    default = frozenset()


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


class BuildingsInLogic(Toggle):
    """
    Gate capability-driving KSP facilities as in-logic progression.

    When on (default), the curated facilities start at level 0 and the player
    upgrades them by collecting building progressives.  When off, all facilities
    are maxed similar to science sandbox, no facility gates anything:

      - **Astronaut Complex** gates EVA.  Until upgraded, planting a flag and EVA
        in orbit are out of logic on EVERY body incl. home, as are rescues and
        surface samples away from home; only plain home-surface EVA works at
        level 0.
      - **Research & Development** gates surface samples (needs the R&D facility
        even on the home body) *and* the science-cost cap — how expensive a tech
        node you may buy.  Its level rides the Progressive R&D count on a
        ``(2, 4)`` schedule (samples + mid cap at the 2nd copy, max at the 4th),
        placed so the cap never binds before the tech band does.
      - **Tracking Station** gates the Deep Space Network comms range (a weaker
        DSN needs a stronger antenna, so far uncrewed missions and far science
        transmission require upgrading it) *and* patched conics (L2), a
        prerequisite for navigation.
      - **Mission Control** gates maneuver nodes (L2) — with conics, this is what
        lets you plan rendezvous and transfers.  Rendezvous (rescue, docking) and
        interplanetary transfers need both; home-system (moon) transfers scale
        with difficulty (see ``HomeSystemConics`` / ``HomeSystemNodes``).

    The VAB/SPH buildable limits are wired end-to-end but ship maxed this
    release; their part-count gate is a follow-up.
    """
    display_name = "Buildings In Logic"
    default = 1


class HomeSystemConics(Choice):
    """Whether home-system (moon) transfers require patched conics.

    Patched conics (Tracking Station L2) let you see an encounter before you
    commit the burn.  Interplanetary transfers and rendezvous always need them;
    this option only covers transfers to the home body's own moons, which a
    skilled pilot can eyeball.  Only matters when ``buildings_in_logic`` is on.

    auto         -- follow Difficulty (casual: required, normal: required,
                    expert: not required).  The default.
    required     -- home-system transfers always need conics.
    not_required -- home-system transfers never need conics.
    """
    display_name = "Home System Conics"
    option_auto = 0
    option_required = 1
    option_not_required = 2
    default = option_auto


class HomeSystemNodes(Choice):
    """Whether home-system (moon) transfers require maneuver nodes.

    Maneuver nodes (Mission Control L2, plus conics) let you plan a precise burn.
    Interplanetary transfers and rendezvous always need them; this option only
    covers transfers to the home body's own moons.  Only matters when
    ``buildings_in_logic`` is on.

    auto         -- follow Difficulty (casual: required, normal: not required,
                    expert: not required).  The default.
    required     -- home-system transfers always need nodes (and conics).
    not_required -- home-system transfers never need nodes.
    """
    display_name = "Home System Nodes"
    option_auto = 0
    option_required = 1
    option_not_required = 2
    default = option_auto


# Difficulty (0=casual, 1=normal, 2=expert) -> whether a home-system transfer
# needs conics / nodes when the option is left on ``auto``.
_HOME_CONICS_AUTO: tuple[bool, ...] = (True, True, False)
_HOME_NODES_AUTO: tuple[bool, ...] = (True, False, False)


def resolve_home_system_conics(option_value: int, difficulty: int) -> bool:
    """Resolve ``HomeSystemConics`` (+ Difficulty) to a single requirement bool."""
    if option_value == HomeSystemConics.option_required:
        return True
    if option_value == HomeSystemConics.option_not_required:
        return False
    return _HOME_CONICS_AUTO[difficulty]


def resolve_home_system_nodes(option_value: int, difficulty: int) -> bool:
    """Resolve ``HomeSystemNodes`` (+ Difficulty) to a single requirement bool."""
    if option_value == HomeSystemNodes.option_required:
        return True
    if option_value == HomeSystemNodes.option_not_required:
        return False
    return _HOME_NODES_AUTO[difficulty]


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


class ContractsAvailable(Range):
    """
    Total number of ordinary (non-goal) contracts placed into the seed.
    Independent of the goal; goal contracts are separate.

    Capped at the number of ever-achievable (enabled-type, body) combinations
    available in the seed, so a high value may yield fewer in practice.
    """
    display_name = "Contracts Available"
    range_start = 0
    range_end = 40
    default = 10


class ContractsRequiredForGoal(NamedRange):
    """
    Completed non-goal contracts (X) required before the goal contract item(s)
    are awarded. Read only by goal_contract_mode = count / progressive_unlock;
    ignored by findable / starting.

    auto  -- ceil(0.8 * contracts actually generated).
    0..40 -- Explicit. Must not exceed Contracts Available; clamped down to the
             number of contracts actually generated this seed.
    """
    display_name = "Contracts Required For Goal"
    range_start = 0
    range_end = 40
    default = -1
    special_range_names = {"auto": -1}


class GoalContractMode(Choice):
    """
    How the goal contract item(s) reach the player.

    findable           -- goal contract item is in the multiworld item pool,
                          found like any other item.
    starting           -- goal contract item(s) are precollected as EXTRA
                          starting items; you are limited only by physics, parts,
                          and buildings.
    count              -- (default) complete X of your Y available contracts; on
                          hitting X all goal contract items are awarded at once.
    progressive_unlock -- complete contracts to unlock the goal contract items
                          one at a time (easiest goal mission first), the last
                          at X.
    """
    display_name = "Goal Contract Mode"
    option_findable = 0
    option_starting = 1
    option_count = 2
    option_progressive_unlock = 3
    default = option_count


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
    default = 0


class AllowEveOnExpert(Toggle):
    """
    Allow Eve surface return / sample-return missions as goals and contracts.

    Off (default): Eve return and sample-return are excluded everywhere as a
    deliberate curation choice — they're physically achievable but tedious to
    fly, so they never become a goal target or a contract. On: they become
    available, but ONLY when base Difficulty is expert — independent of Physics
    Difficulty (so a 'zero' physics run on a casual/normal base never surfaces
    Eve).
    """
    display_name = "Allow Eve On Expert"
    default = 0


class HomeContractFloor(Range):
    """
    Minimum number of home-body contracts guaranteed in the seed, regardless of
    the ordinary contract selection.

    Home-body contracts are the earliest-reachable locations in a run, so this
    floor guarantees the item fill always has enough early slots to assemble a
    deep goal's kit. Without it, far-home / broad-goal seeds can rarely run out
    of reachable early slots and strand a progression item (an unsolvable seed).
    The floor draws from whatever home-safe contract types are available — so it
    stays generic as new contract types are added — and picks them randomly each
    seed, so it adds slack without making starts samey.

    0 = off (no guarantee). Counts toward the contract pool; does not raise the
    per-contract reward-slot count.
    """
    display_name = "Home Contract Floor"
    range_start = 0
    range_end = 20
    default = 5


class BodyVisibilityMode(Choice):
    """
    Which celestial bodies start HIDDEN, revealed only as you receive
    "Discover <Body>" items (ResearchBodies-style).

    auto         -- (default) pick a sane mode from the goal: a goal confined to
                    your home system hides EVEN the home system (home_only, so
                    you discover your own neighbourhood progressively); any wider
                    goal hides everything OUTSIDE the home system (home_system).
    all_visible  -- feature off: every body visible from the start, exactly as
                    before. No Discover items are added.
    home_system  -- your home body and the rest of its local system are visible;
                    every other body is hidden until discovered.
    home_only    -- only your home body is visible; everything else — including
                    the rest of your home system — is hidden until discovered.

    A hidden body's missions/contracts require its Discover item (logic-gated in
    both AllowUndiscoveredBodies settings). Parent planets are discovered before
    their moons. Your home body and the Sun are never hidden.
    """
    display_name = "Body Visibility Mode"
    option_auto = 0
    option_all_visible = 1
    option_home_system = 2
    option_home_only = 3
    default = option_auto


class AllowUndiscoveredBodies(Toggle):
    """
    Whether an undiscovered body's sphere still physically exists in-game.

    On (default): a hidden body renders but is uninteractable — no map label,
    not clickable, not Tab-selectable. You can still fly a probe there out of
    logic; arriving reveals it locally. Off: a hidden body is fully invisible
    (its sphere is removed); flying into its SOI destroys the craft, so you can
    only reach what you have discovered.

    Only meaningful when Body Visibility Mode is not all_visible; the generation
    logic requires the Discover item either way.
    """
    display_name = "Allow Undiscovered Bodies"
    default = 1


@dataclass
class KSP1Options(PerGameCommonOptions):
    goal: Goal
    starting_body: StartingBody
    difficulty: Difficulty
    tech_slots_per_node: TechSlotsPerNode
    starting_inventory_count: StartingInventoryCount
    science_safety_factor: ScienceSafetyFactor
    physics_difficulty: PhysicsDifficulty
    start_with_launch_clamps: StartWithLaunchClamps
    guarantee_science_rover: GuaranteeScienceRover
    accessibility: KSP1Accessibility
    exclude_locations: KSP1ExcludeLocations
    exclude_late_tech_tree: ExcludeLateTechTree
    progressive_launch_pad: ProgressiveLaunchPad
    buildings_in_logic: BuildingsInLogic
    home_system_conics: HomeSystemConics
    home_system_nodes: HomeSystemNodes
    contract_type_weights: ContractTypeWeights
    contracts_available: ContractsAvailable
    contracts_required_for_goal: ContractsRequiredForGoal
    goal_contract_mode: GoalContractMode
    allow_missions_harder_than_goal: AllowMissionsHarderThanGoal
    allow_eve_on_expert: AllowEveOnExpert
    home_contract_floor: HomeContractFloor
    body_visibility_mode: BodyVisibilityMode
    allow_undiscovered_bodies: AllowUndiscoveredBodies
    flag_bodies: FlagBodies
    return_bodies: ReturnBodies
    sample_return_bodies: SampleReturnBodies
    orbit_bodies: OrbitBodies
    flyby_bodies: FlybyBodies
    enabled_part_packs: EnabledPartPacks
