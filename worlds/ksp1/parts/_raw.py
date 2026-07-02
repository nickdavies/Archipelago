"""PRIVATE loader: reads data/parts.json and builds the raw part database.

The raw DB is the full extracted universe. Consumers outside this package must
go through ``PartManager`` so part availability honors the enabled-pack set.
"""
from __future__ import annotations

import json
import pkgutil
from typing import Optional

from ..part_geometry import derive_roles
from .types import (
    AntennaSpec, AnyPart, CapabilityFlag, CapsuleSpec, Decoupler, Engine,
    FuelTank, HeatShield, LandingLeg, MiscEquipment, Parachute,
    ProbeCoreSpec, SolarSpec, SolidBooster,
)
from .registry import PART_REGISTRY
from .packs import STOCK


# ---------------------------------------------------------------------------
# Lookup tables for building parts from cfg data
# ---------------------------------------------------------------------------

_RESOURCE_DENSITY: dict[str, float] = {
    "LiquidFuel": 0.005,
    "Oxidizer": 0.005,
    "SolidFuel": 0.0075,
    "MonoPropellant": 0.004,
    "XenonGas": 0.0001,
}

_BULKHEAD_DIAMETER: dict[str, float] = {
    "size0": 0.625,
    "size1": 1.25,
    "size1p5": 1.875,
    "size2": 2.5,
    "size3": 3.75,
    "size4": 5.0,
    "mk2": 2.5,
    "mk3": 3.75,
}



# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

# Propellants a tank cannot drain to become compatible with an engine that
# doesn't consume them. MonoPropellant is RCS fuel — distinct supply chain,
# not interchangeable with main propellant feed.
UNDRAINABLE_PROPELLANTS: frozenset[str] = frozenset({"MonoPropellant"})


def usable_fuel_mass(
    tank: "FuelTank", engine_propellants: frozenset[str]
) -> float:
    """Return tonnes of usable propellant for an engine that consumes the
    given propellant set, or 0.0 if the tank cannot fuel the engine.

    All propellants the engine needs must be present in the tank with
    positive mass. Propellants the tank carries that the engine does not
    need are drained (0% fill), except those in UNDRAINABLE_PROPELLANTS.
    """
    tank_resources = dict(tank.fuel_masses)
    have = frozenset(tank_resources)
    if not engine_propellants <= have:
        return 0.0
    if (have - engine_propellants) & UNDRAINABLE_PROPELLANTS:
        return 0.0
    return sum(tank_resources[p] for p in engine_propellants)


def _fuel_type_from_propellants(propellants: dict[str, float]) -> str:
    """Derive fuel_type from an engine's propellant dict."""
    keys = set(propellants.keys()) - {"ElectricCharge"}
    if keys == {"LiquidFuel", "Oxidizer"}:
        return "lfo"
    if keys == {"LiquidFuel"}:
        return "lf"
    if "XenonGas" in keys:
        return "xenon"
    if keys == {"MonoPropellant"}:
        return "monoprop"
    raise ValueError(f"Unknown propellant combination: {propellants}")


def _fuel_type_from_resources(resources: dict[str, float]) -> str:
    """Derive fuel_type from a tank's resource dict."""
    keys = set(resources.keys()) - {"ElectricCharge"}
    if "LiquidFuel" in keys and "Oxidizer" in keys:
        return "lfo"
    if keys == {"LiquidFuel"}:
        return "lf"
    if "XenonGas" in keys:
        return "xenon"
    if "MonoPropellant" in keys:
        return "monoprop"
    raise ValueError(f"Unknown resource combination: {resources}")


def _best_size_class(profiles: list[str]) -> float:
    """Pick the largest non-srf bulkhead diameter."""
    best = 0.0
    for p in profiles:
        if p == "srf":
            continue
        d = _BULKHEAD_DIAMETER.get(p, 0.0)
        if d > best:
            best = d
    # srf-only parts (radial mounts) default to smallest size
    return best if best > 0.0 else 0.625


def _fuel_mass_from_resources(resources: dict[str, float]) -> float:
    """Compute total fuel mass (tonnes) from a resource dict."""
    total = 0.0
    for name, amount in resources.items():
        if name == "ElectricCharge":
            continue
        density = _RESOURCE_DENSITY.get(name)
        if density is None:
            continue
        total += amount * density
    return total


# Resources that capsule pods carry as drainable propellant.  ElectricCharge
# isn't a propellant; Ablator is structural (removing it removes reentry heat
# shielding).  All others (MonoPropellant, LiquidFuel, Oxidizer, XenonGas) are
# drainable pre-launch and reduce the capsule's effective dry mass for the
# CAPSULE rank ordering.
_DRAINABLE_RESOURCES: frozenset[str] = frozenset({
    "MonoPropellant", "LiquidFuel", "Oxidizer", "XenonGas",
})


def _solar_spec_from_cfg(cfg: dict) -> Optional["SolarSpec"]:
    s = cfg.get("solar")
    if not s:
        return None
    return SolarSpec(charge_rate=float(s["charge_rate"]), tracking=bool(s["tracking"]))


def _antenna_spec_from_cfg(cfg: dict) -> Optional["AntennaSpec"]:
    a = cfg.get("antenna")
    if not a:
        return None
    return AntennaSpec(
        power=float(a["power"]),
        combinable=bool(a["combinable"]),
        antenna_type=str(a.get("type", "")),
    )


def _capsule_spec_from_cfg(cfg: dict) -> Optional["CapsuleSpec"]:
    crew = int(cfg.get("crew_capacity", 0))
    if crew <= 0:
        return None
    # Data-driven exclusion of non-sealed crew positions (e.g. the
    # External Command Seat).  A "real" capsule has at least one stack
    # bulkhead so it can serve as a terminal payload mounted on top of
    # a rocket; an srf-only part is a chair clipped to the hull and
    # cannot survive reentry or pressurization missions.  Replaces the
    # legacy hand-curated ``_CAPSULE_EXCLUSIONS`` list.
    bulkheads = cfg.get("bulkhead_profiles", [])
    if all(b == "srf" for b in bulkheads):
        return None
    resources = cfg.get("resources", {})
    drainable = 0.0
    for res_name, amount in resources.items():
        if res_name not in _DRAINABLE_RESOURCES:
            continue
        density = _RESOURCE_DENSITY.get(res_name)
        if density is None:
            continue
        drainable += float(amount) * density
    return CapsuleSpec(crew_capacity=crew, drainable_mass=drainable)


def _probe_core_spec_from_cfg(cfg: dict, provides: frozenset) -> Optional["ProbeCoreSpec"]:
    if CapabilityFlag.PROBE_CORE not in provides:
        return None
    # Stayputnik has no ModuleSAS in cfg — SAS level defaults to 0.
    sas = int(cfg.get("sas_level", 0))
    return ProbeCoreSpec(sas_level=sas)


# --- Descent-model derivations (heat-shield drag, chute deploy envelope) ----

# Deployed-state preference for the entry-facing drag cube: the inflatable
# bakes its inflated shape as "A"; rigid shields use "Clean" (post fairing
# jettison — smaller than "Fairing", conservative) / "Default". No cube data
# -> 0.0, and the descent model credits no aero bleed (conservative).
_SHIELD_CUBE_STATES: tuple[str, ...] = ("A", "Clean", "Default")
# dragMultiplier(8) * dragCubeMultiplier(0.1) from KSP Physics.cfg: converts a
# cube's cd*area into the effective drag area the terminal-velocity model uses.
_CUBE_DRAG_GLOBALS: float = 0.8


def _shield_drag_area(cfg: dict) -> float:
    cubes = cfg.get("drag_cubes") or {}
    for state in _SHIELD_CUBE_STATES:
        cube = cubes.get(state)
        if cube:
            return _CUBE_DRAG_GLOBALS * float(cube["area_y"]) * float(cube["cd_y"])
    return 0.0


# Parachute deployment-envelope calibration. KSP's real gate is thermal
# (chuteMaxTemp vs shock heating scaled by machHeatMultBase); we map it onto a
# max safe dynamic pressure per chute, anchored at the stock main-chute limit
# and scaled by the parsed heat tolerance. Q_SAFE_MAIN is calibrated so a
# Kerbin main full-deploys at ~120 m/s at sea-level density
# (old_plans/staged_atmospheric_landing.md). Drogues derive a higher
# multiplier from their heat fields — Mk25 hits the x8 cap, matching its
# in-game ~1200 m/s Duna deploys — capped so the model always UNDER-estimates
# the safe deploy speed (golden rule). Chutes with a raised chuteMaxTemp but
# no mach field fall back to a flat x3.
_Q_SAFE_MAIN_KPA: float = 8.8
_MAIN_CHUTE_MAX_TEMP: float = 650.0   # Mk16 baseline chuteMaxTemp
_DROGUE_Q_MULT_CAP: float = 8.0
_DROGUE_Q_MULT_FALLBACK: float = 3.0


def _chute_q_safe_kpa(chute: dict) -> float:
    max_temp = chute.get("chute_max_temp")
    mach_mult = chute.get("mach_heat_mult_base")
    if max_temp is None or max_temp <= _MAIN_CHUTE_MAX_TEMP:
        return _Q_SAFE_MAIN_KPA
    if mach_mult is None or mach_mult <= 0:
        return _Q_SAFE_MAIN_KPA * _DROGUE_Q_MULT_FALLBACK
    mult = (max_temp / _MAIN_CHUTE_MAX_TEMP) / mach_mult
    return _Q_SAFE_MAIN_KPA * min(max(mult, 1.0), _DROGUE_Q_MULT_CAP)


def _build_part(part_type: type, cfg: dict, overrides: dict, name: str) -> AnyPart:
    """Construct a frozen part dataclass from cfg JSON data + manual overrides."""
    cfg_name = name
    mass = cfg["mass"]
    size = _best_size_class(cfg.get("bulkhead_profiles", []))

    bulkheads = cfg.get("bulkhead_profiles", [])
    is_radial = "srf" in bulkheads  # can be surface-attached
    # srf-ONLY parts have no stack node.  For tanks, that means they can't be a
    # stage's central spine (every externalTank* radial side tank is one).  For
    # engines, srf-only is the signal of a purpose-built radial engine (Spider,
    # Thud, Twitch, Puff): the nozzle bends 90deg off the mount, so it's only
    # useful side-mounted.  An engine with a stack node + srf (Ant, Vector,
    # Aerospike) is a stack engine whose nozzle points along the mount, so
    # radial-mounting it is pointless — those are stack-only here.
    is_radial_only = bool(bulkheads) and all(b == "srf" for b in bulkheads)

    if part_type is Engine:
        eng = cfg["engine"]
        isp_vac = eng["isp_vac"]
        isp_atm = eng["isp_atm"]
        vac_thrust = eng["max_thrust"]
        atm_thrust = vac_thrust * (isp_atm / isp_vac) if isp_vac > 0 else 0.0
        propellants = tuple(sorted(
            p for p in eng["propellants"].keys() if p != "ElectricCharge"
        ))
        return Engine(
            name=cfg_name,
            vac_isp=isp_vac,
            atm_isp=isp_atm,
            vac_thrust=vac_thrust,
            atm_thrust=atm_thrust,
            mass=mass,
            throttleable=not eng.get("throttle_locked", False),
            has_gimbal=cfg.get("has_gimbal", False),
            size_class=size,
            fuel_type=_fuel_type_from_propellants(eng["propellants"]),
            radial_mountable=is_radial_only,
            propellants=propellants,
        )

    if part_type is FuelTank:
        resources = cfg.get("resources", {})
        fuel_masses = tuple(sorted(
            (name, amount * _RESOURCE_DENSITY[name])
            for name, amount in resources.items()
            if name != "ElectricCharge" and name in _RESOURCE_DENSITY
        ))
        return FuelTank(
            name=cfg_name,
            dry_mass=mass,
            fuel_mass=_fuel_mass_from_resources(resources),
            fuel_type=_fuel_type_from_resources(resources),
            size_class=size,
            max_count=overrides.get("max_count", 0),
            roles=derive_roles(cfg),
            fuel_masses=fuel_masses,
        )

    if part_type is SolidBooster:
        eng = cfg["engine"]
        resources = cfg.get("resources", {})
        isp_vac = eng["isp_vac"]
        isp_atm = eng["isp_atm"]
        vac_thrust = eng["max_thrust"]
        atm_thrust = vac_thrust * (isp_atm / isp_vac) if isp_vac > 0 else 0.0
        return SolidBooster(
            name=cfg_name,
            vac_isp=isp_vac,
            atm_isp=isp_atm,
            vac_thrust=vac_thrust,
            atm_thrust=atm_thrust,
            dry_mass=mass,
            fuel_mass=_fuel_mass_from_resources(resources),
            has_gimbal=cfg.get("has_gimbal", False),
            size_class=size,
            radial_mountable=is_radial,
        )

    if part_type is HeatShield:
        return HeatShield(
            name=cfg_name,
            mass=mass,
            size_class=overrides.get("size_class", size),
            drag_area=_shield_drag_area(cfg),
        )

    if part_type is Parachute:
        chute = cfg.get("parachute", {})
        return Parachute(
            name=cfg_name,
            mass=mass,
            drag_area=chute.get("fully_deployed_drag", 0.0),
            is_drogue=overrides.get("is_drogue", False),
            is_radial=overrides.get("is_radial", False),
            semi_drag_area=chute.get("semi_deployed_drag", 0.0),
            q_safe_kpa=_chute_q_safe_kpa(chute),
            deploy_altitude_m=chute.get("deploy_altitude", 0.0),
            min_pressure_atm=chute.get("min_air_pressure_to_open", 0.0),
        )

    if part_type is LandingLeg:
        return LandingLeg(
            name=cfg_name,
            mass=mass,
            tier=overrides["tier"],
        )

    if part_type is Decoupler:
        kind = ("radial" if cfg.get("has_module_anchored_decouple")
                else "stack")
        return Decoupler(name=cfg_name, mass=mass, kind=kind, size_class=size)

    if part_type is MiscEquipment:
        provides = overrides.get("provides", frozenset())
        capsule_spec = _capsule_spec_from_cfg(cfg)
        # Honor the data-driven capsule rejection (e.g. srf-only chair):
        # if the part-name's PartMapping says it provides ``capsule`` but
        # the cfg says it isn't a sealed pod, strip ``capsule`` from
        # ``provides`` so capability code (has_capsule, available_capsules)
        # never sees it.
        if "capsule" in provides and capsule_spec is None:
            provides = provides - frozenset({"capsule"})
        return MiscEquipment(
            name=cfg_name,
            mass=mass,
            provides=provides,
            crew_capacity=int(cfg.get("crew_capacity", 0) or 0),
            size_class=size,
            solar=_solar_spec_from_cfg(cfg),
            antenna=_antenna_spec_from_cfg(cfg),
            capsule=capsule_spec,
            probe_core=_probe_core_spec_from_cfg(cfg, provides),
        )

    raise TypeError(f"Unknown part type: {part_type}")


# ---------------------------------------------------------------------------
# Loader: reads data/parts.json and builds PART_DB at import time
# ---------------------------------------------------------------------------

# Items that are registered as one type but also provide a secondary role.
# Each entry adds a MiscEquipment with the given provides to the same item.
_DUAL_PURPOSE: dict[str, frozenset[str]] = {
    "Size4_EngineAdapter_01": frozenset({"multi_mount"}),
    "mk2_1m_Bicoupler": frozenset({"multi_mount"}),
}


def _load_parts_json() -> dict[str, dict]:
    raw = pkgutil.get_data("worlds.ksp1", "data/parts.json")
    assert raw is not None, "data/parts.json not found in package"
    return json.loads(raw.decode("utf-8"))


# Raw parts.json, keyed by cfg_name. Loaded once; reused by the part-DB
# builder and the PartCategory resolver (which needs metadata fields like
# ``resources`` that the dataclasses don't retain).
_PARTS_JSON: dict[str, dict] = _load_parts_json()


def _load_part_db() -> dict[str, list[AnyPart]]:
    all_parts = _PARTS_JSON

    db: dict[str, list[AnyPart]] = {}
    for mapping in PART_REGISTRY:
        cfg = all_parts.get(mapping.cfg_name)
        if cfg is None:
            raise ValueError(
                f"PART_REGISTRY references cfg name {mapping.cfg_name!r} "
                f"which does not exist in parts.json"
            )
        part = _build_part(mapping.part_type, cfg, mapping.overrides,
                           name=mapping.ksp_name)
        db.setdefault(mapping.ksp_name, []).append(part)

        # Dual-purpose parts: also add a MiscEquipment to the same item
        provides = _DUAL_PURPOSE.get(mapping.cfg_name)
        if provides is not None:
            db[mapping.ksp_name].append(
                MiscEquipment(name=mapping.ksp_name, mass=cfg["mass"],
                              provides=provides)
            )
    return db


_RAW_PART_DB: dict[str, list[AnyPart]] = _load_part_db()


def _load_part_pack() -> dict[str, str]:
    """Map each item (ksp_name) to its originating pack id, read from the
    per-part ``pack`` field in parts.json (produced by the pack-aware
    extractor). Missing field defaults to ``STOCK`` so an un-migrated
    parts.json still loads — the result is just "everything is stock"."""
    return {
        mapping.ksp_name: _PARTS_JSON[mapping.cfg_name].get("pack", STOCK)
        for mapping in PART_REGISTRY
        if mapping.cfg_name in _PARTS_JSON
    }


# Item (ksp_name) -> pack id, parallel to PART_DB. Consumed only by PartManager.
_RAW_PART_PACK: dict[str, str] = _load_part_pack()
