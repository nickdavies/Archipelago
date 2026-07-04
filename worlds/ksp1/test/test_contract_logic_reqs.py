"""Consistency of the shared capability-needs derivation.

The cheap fill-time gate for a contract (and a goal event) is
``contract_logic_needs`` / ``mission_logic_needs`` translated to building
thresholds by ``sphere_ladder._needs_to_counted``.  It MUST agree with what the
real evaluator (``can_complete_contract`` → ``_evaluate_profile``) enforces:

* a MISSED requirement (cheap allows, real forbids) ships an unsolvable seed —
  the exact bug this system fixes;
* an OVER-gate (cheap forbids, real allows) makes dead content.

These tests pin both directions with a behavioural iff, plus the home-membership
⇔ ``PLANET_TRANSFER``-edge tripwire the nav derivation relies on, and the
buildings-off no-op.
"""
import unittest
import random as _random

from worlds.ksp1 import contracts as C
from worlds.ksp1.contracts import contract_logic_needs, can_complete_contract
from worlds.ksp1.capability import _pre_pass, mission_logic_needs
from worlds.ksp1.sphere_ladder import _needs_to_counted
from worlds.ksp1.bodies import (
    ALL_BODIES, BodyName, MissionBuilder, DIFFICULTY_PROFILES, MissionType, EdgeType,
    home_system_bodies, generate_random_orbit_params, generate_rescue_orbit_params,
)
from worlds.ksp1.items import (
    PROGRESSIVE_ASTRONAUT_COMPLEX_NAME as AC,
    PROGRESSIVE_TRACKING_STATION_NAME as TS,
    PROGRESSIVE_MISSION_CONTROL_NAME as MC,
    PROGRESSIVE_RD_NAME as RD,
)
from worlds.ksp1.effects import RD_SAMPLES_COUNT

# "Withhold fully" level per building: 0 removes the ability; R&D rides a count
# threshold, so drop it to one below the samples threshold.
_WITHHOLD = {AC: 0, TS: 0, MC: 0, RD: RD_SAMPLES_COUNT - 1}

# Three home archetypes: a planet with moons (Kerbin), a moonless planet where
# EVERY off-home target is interplanetary (Eeloo), and a moon home whose whole
# Jool system is "local" (Laythe).
HOMES = [BodyName.KERBIN, BodyName.EELOO, BodyName.LAYTHE]
DIFF = DIFFICULTY_PROFILES["comfortable"]     # normal physics
LNC, LNN = True, False                         # normal home-system nav resolution
BUILDINGS = [AC, TS, MC, RD]


def _mb(home):
    mb = MissionBuilder(home=home)
    mb.random_orbit_params = generate_random_orbit_params(_random.Random(0), ALL_BODIES)
    mb.rescue_orbit_params = generate_rescue_orbit_params(_random.Random(0), ALL_BODIES)
    return mb


def _flags(home, withhold=None):
    """buildings_in_logic flags with every item maxed except ``withhold`` (a
    dict item_name -> count)."""
    withhold = withhold or {}
    return _pre_pass(lambda n: withhold.get(n, 99), start_with_clamps=True,
                     progressive_launch_pad=False,
                     launch_pad_caps=_mb(home).launch_pad_caps,
                     buildings_in_logic=True, home=home,
                     local_needs_conics=LNC, local_needs_nodes=LNN)


def _counted(spec, mb):
    return dict(_needs_to_counted(
        contract_logic_needs(spec, mb), buildings_in_logic=True,
        local_needs_conics=LNC, local_needs_nodes=LNN))


class TestContractLogicConsistency(unittest.TestCase):
    def test_presence_iff_per_building(self):
        """Safety-critical: for every feasible (home, type, body) contract,
        the derived counted gate requires a building IFF the real evaluator
        genuinely needs it (withholding it fully makes the contract infeasible).

        This catches a MISSED requirement (real needs it, derived doesn't — the
        shipped-unsolvable-seed bug) and a PRESENCE over-gate (derived requires a
        building real never needs — dead content).  It does NOT assert level
        exactness: the DSN Tracking-Station level is a deliberate conservative
        UPPER bound (it assumes the minimal antenna), which over-estimates TS —
        Golden-Rule-safe (the cheap rule is stricter than reality, never looser).
        """
        checked = 0
        for home in HOMES:
            mb = _mb(home)
            full = _flags(home)
            for ct in C.ContractType:
                td = C.CONTRACT_TYPE_DEFS[ct]
                for body in ALL_BODIES:
                    if not td.body_compatible(body):
                        continue
                    spec = C.ContractSpec(ct, body.name)
                    if not can_complete_contract(spec, full, DIFF, mb):
                        continue  # infeasible even at max kit — never generated
                    needs = _counted(spec, mb)
                    for b in BUILDINGS:
                        derived_requires = needs.get(b, 0) > 0
                        withheld = _flags(home, {b: _WITHHOLD[b]})
                        real_needs = not can_complete_contract(spec, withheld, DIFF, mb)
                        checked += 1
                        with self.subTest(home=home.name, type=ct.name,
                                          body=body.name.name, building=b):
                            # UNDER-gate is the safety-critical direction (Golden
                            # Rule): if real needs a building, the cheap gate MUST
                            # require it, else the seed ships unsolvable.
                            if real_needs and not derived_requires:
                                self.fail(
                                    f"{spec.contract_id}: real needs {b} but the "
                                    f"derived counted omits it (UNDER-gate — would "
                                    f"ship an unsolvable seed)")
                            # Over-gate (dead content) is asserted only for the
                            # exact-by-construction buildings.  TS is EXCLUDED: it
                            # carries deliberate conservative over-gates (the DSN
                            # level assumes the minimal antenna; a descent to the
                            # home's PARENT is nav-free in the real model but
                            # CAN_NAVIGATE_LOCAL still gates conics) — both
                            # Golden-Rule-safe and shared with the mission path.
                            if b != TS and derived_requires and not real_needs:
                                self.fail(
                                    f"{spec.contract_id}: derived requires {b} but "
                                    f"real never needs it (over-gate — dead content)")
        self.assertGreater(checked, 200, "matrix suspiciously small")

    def test_derived_level_is_sufficient(self):
        """No level UNDER-gate: at exactly the derived level of each required
        building, the contract is feasible (the cheap rule never demands MORE than
        the real evaluator can satisfy — the other half of Golden-Rule safety)."""
        for home in HOMES:
            mb = _mb(home)
            full = _flags(home)
            for ct in C.ContractType:
                td = C.CONTRACT_TYPE_DEFS[ct]
                for body in ALL_BODIES:
                    if not td.body_compatible(body):
                        continue
                    spec = C.ContractSpec(ct, body.name)
                    if not can_complete_contract(spec, full, DIFF, mb):
                        continue
                    needs = _counted(spec, mb)
                    withhold = {b: lvl for b, lvl in needs.items() if b in _WITHHOLD}
                    if not withhold:
                        continue
                    at_derived = can_complete_contract(
                        spec, _flags(home, withhold), DIFF, mb)
                    with self.subTest(home=home.name, type=ct.name, body=body.name.name):
                        self.assertTrue(
                            at_derived,
                            f"{spec.contract_id}: infeasible at the derived "
                            f"building levels {withhold} — cheap rule over-demands")


class TestNavHomeMembershipTripwire(unittest.TestCase):
    """The nav derivation keys on home-system MEMBERSHIP (byte-identical
    signatures) but must stay equivalent to the real gate's PLANET_TRANSFER-edge
    test.  If this ever fails, switch ``mission_logic_needs`` to the profile
    edge test."""
    def test_interplanetary_membership_equals_planet_transfer_edge(self):
        for home in HOMES:
            mb = _mb(home)
            hs = home_system_bodies(home)
            for body in ALL_BODIES:
                for mt in MissionType:
                    profiles = mb.profiles_for(body.name, mt)
                    nonempty = [p for p in profiles if p]
                    if not nonempty:
                        continue
                    has_transfer = all(
                        any(e.edge_type == EdgeType.PLANET_TRANSFER for e in p)
                        for p in nonempty)
                    any_transfer = any(
                        any(e.edge_type == EdgeType.PLANET_TRANSFER for e in p)
                        for p in nonempty)
                    # uniformity across alternatives (the derivation assumes it)
                    with self.subTest(home=home.name, body=body.name.name, mt=mt.name):
                        self.assertEqual(
                            has_transfer, any_transfer,
                            "PLANET_TRANSFER not uniform across profile alternatives")
                        self.assertEqual(
                            body.name not in hs, has_transfer,
                            f"home-membership vs PLANET_TRANSFER edge disagree for "
                            f"{body.name.name}/{mt.name} from {home.name}")


class TestBuildingsOffNoOp(unittest.TestCase):
    def test_needs_to_counted_empty_when_off(self):
        mb = _mb(BodyName.EELOO)
        for ct in C.ContractType:
            td = C.CONTRACT_TYPE_DEFS[ct]
            body = next((b for b in ALL_BODIES if td.body_compatible(b)), None)
            if body is None:
                continue
            spec = C.ContractSpec(ct, body.name)
            counted = _needs_to_counted(
                contract_logic_needs(spec, mb), buildings_in_logic=False,
                local_needs_conics=LNC, local_needs_nodes=LNN)
            with self.subTest(type=ct.name):
                self.assertEqual(counted, ())


class TestEelooOffHomeNeedsNav(unittest.TestCase):
    def test_every_off_home_contract_requires_mc_and_ts(self):
        """Eeloo has no moons, so every off-home contract is interplanetary and
        must derive both Mission Control and Tracking Station (the shipped-bug
        regression)."""
        mb = _mb(BodyName.EELOO)
        full = _flags(BodyName.EELOO)
        seen = 0
        for ct in C.ContractType:
            td = C.CONTRACT_TYPE_DEFS[ct]
            for body in ALL_BODIES:
                if body.name == BodyName.EELOO or not td.body_compatible(body):
                    continue
                spec = C.ContractSpec(ct, body.name)
                if not can_complete_contract(spec, full, DIFF, mb):
                    continue
                counted = _counted(spec, mb)
                seen += 1
                with self.subTest(type=ct.name, body=body.name.name):
                    self.assertIn(MC, counted, f"{spec.contract_id} missing MC")
                    self.assertIn(TS, counted, f"{spec.contract_id} missing TS")
        self.assertGreater(seen, 0)


class TestCapabilityFingerprintCoversRequiredParts(unittest.TestCase):
    """The L2 capability cache is keyed by a fingerprint over ``CAPABILITY_ITEMS``.
    ``contract_access`` depends on the contract required-category parts (drill /
    ore_tank / science_lab / eva_jetpack) via ``required_part_manifest``, so those
    parts MUST be in the fingerprint — otherwise a state without the drill and a
    state with it share a fingerprint and the cached (drill-less → infeasible)
    ``contract_access`` is reused after the drill is collected (bug 106)."""

    def test_every_category_member_is_fingerprinted(self):
        from worlds.ksp1.capability import CAPABILITY_ITEMS
        from worlds.ksp1.parts import DEFAULT_PART_MANAGER
        missing = {}
        for cat, members in DEFAULT_PART_MANAGER.category_members.items():
            gap = [m for m in members if m not in CAPABILITY_ITEMS]
            if gap:
                missing[cat] = gap
        self.assertEqual(
            missing, {},
            "category-tracked parts missing from the capability fingerprint: "
            f"{missing}")


if __name__ == "__main__":
    unittest.main()
