"""The final emitted location set: settings-derived locations plus what this
seed's generation produced (contract rewards and goal-mode thresholds).

Built once in ``generate_early``, after contracts are resolved.  The seed RNG
(and the UT-regen restore from slot_data) live upstream, in contract
generation; this layer only turns the resolved contract specs and thresholds
into locations.  Code that runs before contracts exist takes
``ConfiguredLocations`` and so cannot see them.

Archipelago-free (see ``locations/__init__.py``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from .configured import ConfiguredLocations
from .defs import ContractLocation, KSP1LocationDef, ThresholdLocation

if TYPE_CHECKING:
    from ..contracts import ContractSpec


@dataclass(frozen=True)
class ContractThreshold:
    """A goal-mode threshold location and what it releases: once
    ``required_count`` contracts are complete the client reports ``location``,
    which holds the locked ``item_name`` (a goal contract item, or a
    Progressive R&D copy for the tech-tree goal)."""
    location: ThresholdLocation
    required_count: int
    item_name: str


@dataclass(frozen=True)
class SeedLocations:
    """Everything this world emits.  Build with ``build``."""
    configured: ConfiguredLocations
    contracts: tuple[ContractLocation, ...]
    thresholds: tuple[ContractThreshold, ...]

    @classmethod
    def build(
        cls, configured: ConfiguredLocations,
        contract_specs: Iterable[ContractSpec], per_contract: int,
        thresholds: Iterable[ContractThreshold],
    ) -> SeedLocations:
        """``contract_specs`` covers goal and non-goal contracts alike; each
        non-goal contract emits ``per_contract`` reward slots."""
        return cls(
            configured=configured,
            contracts=tuple(
                loc for spec in contract_specs
                for loc in ContractLocation.for_spec(spec, per_contract)),
            thresholds=tuple(thresholds),
        )

    def emitted(self) -> tuple[KSP1LocationDef, ...]:
        return (
            *self.configured.emitted(), *self.contracts,
            *(t.location for t in self.thresholds),
        )

    def by_name(self, name: str) -> KSP1LocationDef | None:
        """The emitted def behind an AP location name, or ``None``.  For the
        name-keyed boundaries (AP registry, UT hooks) only."""
        return self._by_name.get(name)

    @cached_property
    def _by_name(self) -> Mapping[str, KSP1LocationDef]:
        return {str(d): d for d in self.emitted()}
