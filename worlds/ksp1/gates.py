"""Accumulation gates — the second universal constraint kind.

Most locations are gated on **capability** (a :class:`~.requirements.Signature`
of rank/counted thresholds). A few are gated instead on **accumulating** enough
of a fungible resource: the player must be able to gather ``>= amount`` of the
resource from sources reachable below the gate. This module is the single home
for that concept, unifying two cases that were previously bespoke:

* ``SCIENCE`` — earned from reachable bodies/experiments; funds tech tiers
  (``amount`` = cumulative tier cost).
* ``CONTRACT_COMPLETION`` — one unit per completable non-goal contract; funds
  the count / progressive_unlock goal thresholds (``amount`` = X contracts).

**The shared invariant (worst-case-safe).** A gate ``(R, N)`` must be
satisfiable from sources reachable *at or below* its sphere even if the player
spends/branches the supply as wastefully as possible:

* SCIENCE is worst-case-safe automatically — ``amount`` is the *cumulative*
  tier cost, so affording every node through the tier ⊇ affording any subset.
* CONTRACT_COMPLETION is worst-case-safe because the ``N`` easiest contracts are
  each independently reachable at/below the gate (the sphere ladder threads them
  via ``S_contract`` anchors and keeps their reps PROGRESSION in the demote
  pass), so *any* ``N`` of the available contracts can be completed.

The gate's **access face** (:meth:`AccumulationGate.runtime_rule`) is the
``CollectionState`` predicate the fill / beatability sweep evaluates: "does the
player's current state supply ``>= amount``?". The measurement is injected so
the gate stays resource-agnostic and free of import cycles with ``rules`` /
``capability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from BaseClasses import CollectionState


class Resource(Enum):
    """A fungible resource an accumulation gate can demand."""

    SCIENCE = "science"
    CONTRACT_COMPLETION = "contract_completion"


@dataclass(frozen=True)
class AccumulationGate:
    """Need ``>= amount`` of ``resource`` accumulated from sources reachable
    below the gate. See the module docstring for the worst-case-safe invariant.
    """

    resource: Resource
    amount: float

    def runtime_rule(
        self, measure: Callable[["CollectionState"], float]
    ) -> Callable[["CollectionState"], bool]:
        """Wrap a state-supply ``measure`` into the gate's access predicate.

        ``measure(state)`` returns how much of ``resource`` the player can
        currently supply; the rule passes when that meets ``amount``.
        """
        amount = self.amount

        def rule(state: "CollectionState", _m=measure, _a=amount) -> bool:
            return _m(state) >= _a

        return rule
