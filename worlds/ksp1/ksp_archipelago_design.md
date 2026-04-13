# KSP Archipelago World Design

## Overview

This document is the North Star design reference for a Kerbal Space Program Archipelago (AP) multiworld plugin. It covers the item model, location model, reachability logic, capability evaluation system, staging model, difficulty scaling, and AP integration constraints. It is intended to be fed to implementation agents alongside relevant codebases.

---

## Archipelago Integration Constraints

Understanding how AP's fill algorithm works is critical because it shapes every design decision.

### How AP Fill Works

AP fills the world by classifying items as `advancement` (progression), `useful`, or `filler`, then placing progression items into reachable locations first. It repeatedly calls `can_reach(location, state)` to determine what is accessible given the current collection state. This function is called inside tight loops — potentially tens of thousands of times during generation for a world with ~150 locations.

**Critical implication:** Access rules must be microseconds, not milliseconds. All expensive computation must be cached on the `CollectionState` object.

### CollectionState Caching Pattern

AP's `CollectionState` is copied during sweep operations. Caching derived data on the state object is both safe and expected — when AP copies a state and adds a new item, the new state is a fresh copy and the cache naturally invalidates on next access. There is no need for manual cache invalidation.

The pattern is:
1. A cheap access rule checks a cached `RocketCapability` profile on the state
2. If the cache is cold, it triggers `_compute_capability`, which does all expensive work once
3. All subsequent location checks within that sweep read from the warm cache

### Item and Location Count Constraint

AP generation fails if `items > locations`. The item pool must be sized to fit within the location pool. This is the primary design tension for KSP, which has ~470 parts (base + DLC) but a naturally smaller set of meaningful mission locations. KSP is consistently **location-short, not item-short** — the challenge is generating enough interesting locations to absorb the part pool, not padding the item pool. Solutions are discussed in the Items and Locations sections.

### Item Classification

- **Advancement:** Items that open new reachable locations. Engines, tanks, decouplers, heat shields, parachutes, relay antennas, RTGs, probe cores, capsules. Almost all physical rocket parts qualify.
- **Useful:** Items that help but rarely gate locations on their own. RCS systems, additional reaction wheels, fairings, docking ports (except where rendezvous is a location requirement).
- **Filler:** Cosmetic items, trap items (eg "mystery part" that turns out to be a structural panel), and genuinely redundant parts that provide no new capability.

Classifying too many items as `advancement` makes fill slow and generation brittle. Pre-classify conservatively. Note that because KSP is location-short, the filler category should be kept small — excess items over locations are handled by reducing the item pool via part group bundling rather than by padding with duplicates.

---

## Item Model

### Predefined Part Groups

Parts are organised into predefined groups. Each group is one AP item. The exact group definitions are provided as a constant by the world author and are not generated dynamically. This gives authorial control over progression shape while keeping the item count manageable.

The grouping strategy is still under active design. The axes below represent one reasonable approach but the final groupings will be determined by working through the actual part list. The key constraint to preserve is that parts which are always needed together (eg fuel lines and struts for asparagus staging) should be in the same group, and parts that gate completely unrelated capabilities should not be forced together.

Likely grouping axes:

- **Engines** — by tier and type (chemical atmospheric, chemical vacuum, nuclear, ion, solid)
- **Tanks** — by size class and fuel type (LFO, monoprop, xenon)
- **Staging connectors** — decouplers and separators (see Staging section)
- **Control systems** — reaction wheels, RCS thrusters and tanks, probe cores, capsules
- **Landing systems** — landing legs by tier, parachutes, heat shields
- **Power systems** — solar panels, RTGs, batteries
- **Communication** — relay antennas by tier
- **Utility** — docking ports, ISRU equipment, science instruments
- **Cosmetic starter pack** — flags, lights, ladders, decorative panels, paint variants. Given as precollected at game start.

### Same-Tier Parts as Separate Items

Parts within the same tier are kept as separate items rather than bundled. This preserves the "silly rocket" constraint — a square probe on a round rocket is a valid and fun choice. It also means the item pool naturally includes many items of similar capability tier, which is desirable given that KSP is location-short: the pool needs to be large to fill available locations.

### Item Count Management

Because KSP is location-short, the primary tool for balancing the ratio is **reducing the item pool** rather than expanding it. For runs with a smaller goal (and therefore fewer locations), parts are bundled by KSP tech tree node — each node becomes one item. This is opt-in. The default experience preserves individual parts and assumes a full location set.

### Quality of Life Items

Not all items are required for mission capability. Some items improve the experience without gating any location:

- Additional science instruments beyond the first of each type
- MechJeb or autopilot computer (reduces precision skill requirement)
- Enhanced navball or manoeuvre node planner
- Extra crew capacity beyond the minimum required

These are classified as `useful` rather than `advancement`. Finding them feels rewarding but missing them is never a blocker.

---

## Staging Model

### Staging Connector Tier

Staging requires physical connectors. This is modelled as a tiered item:

```
staging_tier: int
  0 = no staging (single stage only)
  1 = stack decouplers (serial staging only)
  2 = radial decouplers (parallel and asparagus staging unlocked)
  3 = docking ports (orbital assembly, multi-launch missions)
```

The backpropagation limits the number of modelled stages based on this tier. A player with `staging_tier = 0` gets a single-stage calculation regardless of how many engine and tank groups they hold. This is a meaningful early gate and creates a clear "aha" moment when decouplers arrive.

### Backpropagation Algorithm

Rocket capability is computed by working **backwards** from the mission endpoint to the launch pad. The mission is a path through the delta-V graph. At each edge, the algorithm asks: "what stage is needed to deliver the current payload across this burn?"

Steps:
1. Define the terminal payload mass for the mission type (probe, crewed, sample return)
2. Walk the mission path in reverse
3. For each edge, compute the minimum wet mass of a stage using the rocket equation: `wet_mass = payload * e^(dv / (isp * g))`
4. Select the engine that minimises total stage mass (highest ISP in vacuum, highest TWR at launch)
5. The output of each stage calculation becomes the payload mass input for the next stage back
6. Record a `StageResult` for each edge capturing delta-V, TWR at ignition and burnout, and engine properties

The algorithm respects `staging_tier` by capping how many separate stages can be modelled.

### Fuel Carryover Between Edges

Not every edge boundary requires a stage separation. Adjacent edges with low combined delta-V cost are merged before applying the rocket equation. The heuristic is: merge consecutive edges if their combined delta-V is below a threshold (suggested ~500 m/s) and the same engine type is optimal for both. Kerbin ascent is always its own stage. Small orbital corrections are typically merged with adjacent burns.

### Asparagus and Onion Staging (Future)

Parallel staging (asparagus, onion) is deferred to a later version. It requires `staging_tier >= 2`. When implemented, the algorithm models fuel crossfeed across parallel stacks before separation. For AP logic purposes, asparagus staging significantly increases effective delta-V for the same parts — its absence is a conservative (safe) approximation.

### ISRU (Future)

ISRU breaks the return trip calculation by allowing propellant to be sourced at the destination. When ISRU is modelled, return journey edges are removed from the initial launch delta-V requirement for bodies where ISRU is possible. This makes ISRU a very powerful late-game `advancement` item.

---

## Capability Profile

The capability evaluator outputs a `RocketCapability` struct that is cached on the `CollectionState`. All location access rules read from this cached profile — they never trigger recomputation themselves.

### Computation Pipeline

1. **Pre-pass** — iterate all unlocked parts, collect binary equipment flags. O(n parts), trivially fast.
2. **Backpropagation** — use equipment flags to select effective delta-V per edge (eg aerobraking reduces edge cost if heat shield is present), compute staged delta-V for all mission profiles.
3. **Per-body assessment** — for each celestial body, evaluate whether each mission type is achievable given the computed stages and equipment flags.
4. **Assemble profile** — populate `RocketCapability` with all results.

### RocketCapability Struct

```python
@dataclass
class StageResult:
    delta_v: float
    twr_at_ignition: float        # full tanks
    twr_at_burnout: float         # empty tanks
    engine_is_throttleable: bool
    engine_has_gimbal: bool

@dataclass
class BodyAccessProfile:
    can_orbit_low: bool
    can_orbit_high: bool
    can_land_unmanned: bool
    can_land_crewed: bool
    can_return_to_kerbin: bool    # from orbit
    can_return_crewed: bool       # from surface
    can_sample_return: bool       # crewed surface + sample + return
    blocking_reason: Optional[str]  # for debugging failed generations

@dataclass
class RocketCapability:
    # Equipment flags (from pre-pass)
    has_heat_shield: bool
    has_parachutes: bool
    landing_leg_tier: int         # 0=none, 1=small, 2=medium, 3=heavy
    has_reaction_wheels: bool
    has_rcs: bool
    has_gimbal: bool
    has_throttle_control: bool
    has_probe_core: bool
    has_capsule: bool
    has_rtg: bool
    has_isru: bool
    has_docking_port: bool
    relay_tier: int               # 0=none, 1=local, 2=inner planets, 3=mid system, 4=outer system
    power_profile: str            # "solar" | "solar_marginal" | "rtg"
    staging_tier: int

    # Delta-V summary
    max_dv_total: float
    max_payload_to_lko: float     # tonnes

    # Per-body assessments
    bodies: Dict[str, BodyAccessProfile]

    # Stage detail for debugging
    stage_results: List[StageResult]
```

The `blocking_reason` field is not used by AP but is invaluable when debugging bad generations or unexpected progressions.

---

## Delta-V Graph

The mission graph is a directed graph of celestial body states connected by transfer edges. Each edge has a propulsive delta-V cost and optional aerobraking cost.

### Edge Structure

```python
@dataclass
class MissionEdge:
    dv_propulsive: float
    dv_post_aero: float           # residual cost after aerobraking
    aerobrake_available: bool
    requires_heat_shield: bool
    requires_chute: bool
    min_twr: float                # 0.0 for transfer burns, >0 for landing/launch
    requires_throttleable: bool
    requires_gimbal: bool

    def effective_dv(self, cap: RocketCapability) -> float:
        if self.aerobrake_available and cap.has_heat_shield:
            return self.dv_post_aero
        return self.dv_propulsive
```

### Mission Profiles

For bodies where aerobraking is available, multiple mission profiles exist representing propulsive vs aerobrake approaches. The capability check tries all valid profiles and returns `True` if any is achievable. This avoids penalising players who lack heat shields — they simply use the more expensive propulsive profile if they have sufficient delta-V.

Each profile is a named sequence of edges. The aerobrake profile has a cheaper capture edge but requires a heat shield. The propulsive profile has no equipment requirement but costs significantly more delta-V:

```python
DUNA_LANDING_PROFILES = [
    MissionProfile(
        name="propulsive",
        edges=[
            "kerbin_surface → lko",
            "lko → kerbin_escape",
            "kerbin_escape → duna_transfer",
            "duna_transfer → duna_orbit_propulsive",  # ~1450 m/s capture burn
            "duna_orbit → duna_surface",
        ],
        requirements=[]
    ),
    MissionProfile(
        name="aerobrake",
        edges=[
            "kerbin_surface → lko",
            "lko → kerbin_escape",
            "kerbin_escape → duna_transfer",
            "duna_transfer → duna_orbit_aerobrake",   # ~150 m/s residual after aerobrake
            "duna_orbit → duna_surface",
        ],
        requirements=["heat_shield"]
    ),
]
```

Aerobraking also meaningfully changes the staging calculation, not just the delta-V total. Without aerobraking, the Duna capture burn may require a dedicated stage that must be carried from Kerbin. With aerobraking, that stage disappears and its dry mass is no longer payload for earlier stages. The backpropagation runs separately for each profile and takes the best achievable result.

### Body Hierarchy for Pruning

The body access tree mirrors the game's sphere-of-influence structure:

```
kerbin_surface
└── kerbin_orbit_low
    └── kerbin_orbit_high
        ├── mun_orbit_low → mun_orbit_high → mun_surface
        ├── minmus_orbit_low → minmus_orbit_high → minmus_surface
        └── kerbin_escape
            ├── duna_orbit_low → duna_orbit_high → duna_surface
            │   └── ike_orbit → ike_surface
            ├── eve_orbit_low → eve_orbit_high → eve_surface (gilly branch)
            ├── jool_orbit_low → jool_orbit_high
            │   ├── laythe_orbit → laythe_surface
            │   ├── vall_orbit → vall_surface
            │   ├── tylo_orbit → tylo_surface
            │   ├── bop_orbit → bop_surface
            │   └── pol_orbit → pol_surface
            ├── dres_orbit → dres_surface
            └── eeloo_orbit → eeloo_surface
```

Access rules chain through this hierarchy. If a parent node is unreachable, all children return `False` immediately without computing their own requirements. This is baked into `BodyAccessProfile` construction — computing Laythe's profile begins by checking whether Jool orbit is achievable.

---

## Location Model

### Mission Location Types Per Body

The following location types are defined per body. Landing and return have both unmanned and crewed variants. Flag plant and sample return are crewed-only by definition — a probe cannot plant a flag or return a sample.

| Location Type | Detection Event | Crewed Required | Notes |
|---|---|---|---|
| SOI Enter | Vessel crosses SOI boundary inbound | No | First contact |
| SOI Leave | Vessel crosses SOI boundary outbound | No | Confirms intentional visit |
| Low Orbit | Situation = orbiting, altitude < threshold, one full period elapsed | No | Stable orbit confirmation |
| High Orbit | Situation = orbiting, altitude > threshold, one full period elapsed | No | |
| Landing (Unmanned) | Situation = landed, no crew | No | Requires probe core |
| Landing (Crewed) | Situation = landed, crew present | Yes | |
| Return (Unmanned) | Unmanned vessel recovered on Kerbin after surface landing elsewhere | No | |
| Return (Crewed) | Crewed vessel recovered on Kerbin after surface landing elsewhere | Yes | |
| Flag Plant | Kerbal plants flag on surface | Yes | Implies crewed landing |
| Sample Return | Crewed vessel recovered on Kerbin with surface sample | Yes | Implies crewed landing + return |

"One full period elapsed" for orbit confirmation prevents an accidental periapsis dip from counting as a stable orbit.

### Kerbin Altitude Milestones

Seven early locations providing a progression ramp before orbit is possible. Detection is a simple altitude threshold cross with vessel in flight situation.

| Location | Altitude |
|---|---|
| Reached 5km | 5,000m |
| Reached 10km | 10,000m |
| Reached 20km | 20,000m |
| Reached 30km | 30,000m |
| Reached 40km | 40,000m |
| Reached 50km | 50,000m |
| Reached Space (70km) | 70,000m |

These require only TWR > 1 and sufficient delta-V. No staging required. No speed modelling required — aerodynamic simulation is explicitly out of scope for v1.

### Kerbin Special Locations

| Location | Detection Event |
|---|---|
| Splashdown | Vessel situation = splashed on Kerbin |
| Kerbal EVA in Kerbin Orbit | Kerbal on EVA while situation = orbiting Kerbin |
| Kerbal EVA Beyond Kerbin SOI | Kerbal on EVA while outside Kerbin SOI |
| First Staging Event | Decoupler firing event detected |
| Orbital Rendezvous | Two vessels within 100m while orbiting same body |
| Rescue Mission | Crew count increase on vessel near stranded kerbal location |

### Science as Non-Blocking Locations

Science instrument locations (temperature reading from Mun, mystery goo at Duna, etc.) are included only for Kerbin, Mun, and Minmus where revisiting is low-cost. These are always classified as `useful` or `filler`, never `advancement`. A player who visits without the required instrument is never blocked — missing science locations is never on the critical path.

For all other bodies, science gates are excluded to prevent mandatory return trips to distant planets.

### Multi-Check Events

A single in-game event can trigger multiple AP location checks simultaneously. This is a well-established AP pattern — a Dark Souls 3 boss kill fires checks for the soul, the armour set pieces, and any other associated drops all at once. The fill algorithm handles simultaneous checks without issue.

For KSP this is the primary mechanism for expanding the location pool without adding artificial intermediate states. The client mod fires all associated checks on a single event trigger. A Mun sample return recovery, for example, would fire every check associated with that mission in one go.

The number of checks per event is scaled by two factors:

**Mission complexity** — harder missions that require more hardware, planning, and capability unlock more checks. A simple SOI entry is one check; a crewed sample return from a distant body is several.

**Mission duration** — longer missions represent more real player time and delayed gratification. A Jool sample return might take an hour of in-game execution across multiple sessions. Rewarding that with a burst of checks on recovery mirrors the effort and makes the payoff feel proportionate. This is a separate consideration from complexity — Minmus is easy but quick, Eeloo is both hard and very long.

The suggested default check multipliers per event type are:

| Event Type | Checks | Rationale |
|---|---|---|
| SOI enter | 1 | Trivial once you have the delta-V |
| SOI leave | 1 | Confirmation event, low effort |
| Low orbit | 1 | Straightforward once in SOI |
| High orbit | 1 | Minor extension of low orbit |
| Unmanned landing | 2 | Requires landing hardware and precision |
| Crewed landing | 2 | Adds life support and crew risk considerations |
| Unmanned return | 3 | Round trip planning, significant mission length |
| Crewed return | 3 | Round trip with crew recovery |
| Flag plant | 2 | Requires crewed landing, EVA execution |
| Sample return | 4 | Maximum complexity and duration — hardest standard mission |

These multipliers are a starting point and should be tuned against the final part count once groupings are decided. Bodies that are intrinsically harder or more distant (Eeloo, Jool moons, Eve) could apply an additional +1 multiplier on top of the event base, reflecting both mission length and the skill required to execute them. This is worth evaluating once real part counts are known.

### Approximate Location Count

With multi-check multipliers applied across ~14 bodies and the Kerbin-specific locations:

| Category | Base Events | Avg Checks | Locations |
|---|---|---|---|
| Kerbin altitude milestones | 7 | 1 | 7 |
| Kerbin special locations | 6 | 1 | 6 |
| SOI enter + leave (~14 bodies) | 28 | 1 | 28 |
| Low orbit + high orbit (~14 bodies) | 28 | 1 | 28 |
| Unmanned landing (~10 bodies) | 10 | 2 | 20 |
| Crewed landing (~10 bodies) | 10 | 2 | 20 |
| Unmanned return (~8 bodies) | 8 | 3 | 24 |
| Crewed return (~8 bodies) | 8 | 3 | 24 |
| Flag plant (~10 bodies) | 10 | 2 | 20 |
| Sample return (~8 bodies) | 8 | 4 | 32 |
| **Total** | **~123 events** | | **~209 locations** |

With extended goals this expands further. ~209 locations is much more competitive against a granular individual-parts item pool, and reduces or eliminates the need for predefined part group bundling in standard runs. The multi-check approach also means individual parts can be items without requiring synthetic filler — each part is a genuinely distinct item, and the location pool is large enough to absorb them.

---

## Difficulty and Margin System

### Delta-V Margin

Required delta-V for each mission edge is inflated by a combined margin to account for player imprecision:

```
required_dv = (base_dv + fixed_margin_ms) * (1 + percent_margin)
```

The fixed margin is inside the percentage multiplication so the percentage also covers the fixed buffer. This combined model handles two distinct failure modes:
- **Fixed margin** covers accidental short burns, staging timing errors, and imprecise throttle management — errors that are roughly constant in magnitude regardless of the burn size
- **Percentage margin** covers transfer window imprecision, suboptimal gravity turns, and trajectory inefficiency — errors that scale with the complexity of the manoeuvre

### Difficulty Profiles

Difficulty is exposed as named presets in the AP YAML options rather than raw numbers:

| Preset | Fixed Margin | Percent Margin | Notes |
|---|---|---|---|
| Casual | 200 m/s | 30% | Generous, tolerates significant imprecision |
| Normal | 100 m/s | 15% | Default |
| Expert | 50 m/s | 5% | Near-optimal play assumed |
| Insane | 0 m/s | 0% | Exact delta-V values, no margin |

### Human Skill Flags

Some location requirements depend on assumed player skill. These flags, when **active**, add hardware requirements to locations that would otherwise be reachable on pure delta-V alone. They are controlled by difficulty settings and stored as a set of active human flags at world generation time.

```
requires_gravity_turn        — needs gimballed engine or aerodynamic fins for ascent
requires_suicide_burn        — needs throttleable engine + sufficient TWR ceiling for airless landings  
requires_manual_rendezvous   — needs docking port + RCS without autopilot assistance
requires_transfer_window     — interplanetary precision timing assumed
```

At **casual** difficulty, all human flags are **active** — the capability check enforces hardware requirements for all skill-dependent manoeuvres. This means locations stay locked until the player actually has the right equipment, preventing situations where a player is expected to land without landing legs or perform a precision burn with only SRBs. This is the **default and safer** behaviour.

At **expert** difficulty, human flags are **relaxed** — the capability check trusts that the player can execute difficult manoeuvres with suboptimal hardware. An expert player can land on a structural plate, perform an unassisted gravity turn, or manually time an interplanetary window without the game requiring specific hardware to confirm capability.

This direction is intentional: flags being on (restrictive) is the conservative, generation-safe default. Turning flags off is an opt-in for skilled players who want fewer gates.

### Controllability Checks

Beyond raw delta-V, some hardware combinations create controllability soft-locks:

**Solid Rocket Motor (SRB) constraints:** SRBs are all-or-nothing (not throttleable) and have no gimbal. They are modelled with `throttleable: false` and `gimbal: false`. Destinations requiring precision burns (Gilly landing, any rendezvous) have `requires_throttleable: true`. If the player's only engines are SRBs, these destinations are unreachable regardless of delta-V.

**TWR ceiling for low-gravity bodies:** An engine that is too powerful for a low-gravity landing creates an uncontrollable situation. Each body defines a `max_controllable_twr` in addition to `min_twr`. Landing assessment checks both:
- `twr_at_burnout > body.min_twr` (can decelerate against gravity)
- `twr_at_ignition < body.max_controllable_twr` (can throttle down enough to land)

**Attitude control:** If no gimbal, reaction wheels, or RCS are available, orbital manoeuvres are impossible. The capability check requires at least one attitude control source for any mission beyond Kerbin ascent.

**Power availability:** Solar panel output drops with distance from Kerbol (proportional to inverse square of distance). Bodies beyond a threshold require RTG or very large solar arrays for unmanned missions. Modelled as three tiers in `power_profile`:
- `solar` — adequate for Kerbin SOI and inner planets
- `solar_marginal` — adequate for Duna/Dres with large panels
- `rtg` — required for Jool system and beyond

**Communication range:** Missions beyond Kerbin SOI require sufficient antenna power. Modelled as `relay_tier` (4 tiers) derived from CommNet range formula `sqrt(antenna_power * DSN_power)`. Tier 1 = local (Mun/Minmus), tier 2 = inner planets (Moho), tier 3 = mid system (Eve, Duna, Dres), tier 4 = outer system (Jool, Eeloo). See `rocket_math.py` for the underlying power/range data.

### Landing Leg Tier

Landing leg requirements vary by terrain and are gated by a human skill flag (`requires_landing_legs`). When the flag is active (default), the body's minimum leg tier is enforced. When the flag is relaxed (expert), a player can attempt a structural plate landing on any body.

```
landing_leg_tier: int
  0 = none (structural plate landing — only valid when requires_landing_legs flag is relaxed)
  1 = small legs (adequate for Minmus flat terrain, Gilly, low-gravity smooth bodies)
  2 = medium legs (standard Mun, Duna, most bodies — default minimum)
  3 = heavy legs (rocky terrain variants, enabled by a separate difficulty option)
```

Body terrain requirements are static constants. The heavy leg requirement is off by default and enabled by a difficulty option for players who want the extra constraint. The structural plate option is off by default and enabled by relaxing the `requires_landing_legs` flag at expert difficulty.

---

## Victory Conditions

Victory conditions are configured in the AP YAML. Each condition determines which locations must be checked and thus influences the item/location ratio.

| Goal | Description | Approx Locations Needed |
|---|---|---|
| Mun Landing | Crewed Mun landing | ~30 |
| Mun and Minmus | Crewed landing on both | ~40 |
| Inner System | Land on Duna and Ike | ~60 |
| Jool System | Visit all Jool moons | ~85 |
| All Bodies | Land on every landable body | ~115 |
| Specific Hard Body | Eve surface return or Eeloo sample return | ~90 |

For smaller goals, item count is reduced by collapsing part groups (tech tree node bundling option) or by capping the item pool to a configured size.

---

## Client Mod Responsibilities

The KSP client mod is responsible for detecting game events and reporting checked locations to the AP server. Detectable events:

- Vessel SOI change (inbound and outbound)
- Vessel situation change (flying, orbiting, landed, splashed, on EVA)
- Altitude threshold crossed (for Kerbin milestones)
- Crew count change on vessel (rescue detection)
- Flag planted event
- Decoupler firing event (staging detection)
- Vessel proximity check for rendezvous (two vessels within threshold distance)
- Vessel recovery with crew and/or science payload

Events that are explicitly out of scope for v1:
- Aerodynamic speed modelling (no speed records)
- Specific part usage verification (ISRU activation, instrument firing at specific biomes)
- Transfer window precision detection

The client mod also receives items from AP and grants the corresponding parts to the player's available inventory. Duplicate parts are silently ignored.

---

## Open Design Questions

- **Apollo-style and docking missions:** Orbital rendezvous and multi-launch assembly are a natural extension of the docking port item. Apollo-style missions (separate lander and command module, rendezvous in orbit) could add a distinct mission profile type requiring `staging_tier >= 3` and docking port. Worth designing as a future extension once the base mission model is stable.
- **Asparagus/onion staging delta-V bonus:** How much more capable does `staging_tier >= 2` make the modelled rocket? This affects when outer planet locations become reachable and needs careful tuning.
- **ISRU modelling depth:** When ISRU is implemented, which bodies support it and how aggressively does it reduce launch requirements?
- **Relay network as locations:** Should placing a relay satellite in a specific orbit itself be a checkable location? This would add ~8 locations and give relay items a dedicated unlock moment.
- **KSP2 compatibility:** KSP2 development has ended so this is low priority, but the design is intentionally agnostic to KSP version. If KSP2 support were pursued, only the part group constants would need separate definitions — the capability model and AP logic are version-agnostic.
- **Difficulty interaction with random packs:** If part groups are ever randomised in a future version, the generation validator must confirm that the minimum progression path (Kerbin orbit → Mun landing) is achievable with items the fill algorithm can plausibly place early.
