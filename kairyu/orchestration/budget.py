"""Inference-time compute budget for orchestration (design doc D4).

Exceeding a budget is a normal, queryable state — the Conductor returns the
best result so far instead of raising, matching the Fugu framing of recursion
depth as an inference-time compute axis.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    max_steps: int = 16
    max_refine_depth: int = 2
    max_cost_usd: float | None = None


@dataclass(frozen=True)
class BudgetState:
    budget: Budget
    steps_used: int = 0
    cost_used: float = 0.0
    steps_reserved: int = 0
    unknown_cost_reserved: bool = False

    def charge(self, steps: int = 1, cost: float = 0.0) -> BudgetState:
        return dataclasses.replace(
            self, steps_used=self.steps_used + steps, cost_used=self.cost_used + cost
        )

    def try_reserve(
        self, steps: int = 1, *, unknown_cost: bool = False
    ) -> BudgetState | None:
        if steps < 0:
            raise ValueError("reservation steps must be non-negative")
        if self.steps_used + self.steps_reserved + steps > self.budget.max_steps:
            return None

        cost_cap = self.budget.max_cost_usd
        if cost_cap is not None and self.cost_used >= cost_cap:
            return None
        reserves_cost_slot = unknown_cost and cost_cap is not None
        if reserves_cost_slot and self.unknown_cost_reserved:
            return None
        return dataclasses.replace(
            self,
            steps_reserved=self.steps_reserved + steps,
            unknown_cost_reserved=(
                self.unknown_cost_reserved or reserves_cost_slot
            ),
        )

    def release(
        self, steps: int = 1, *, unknown_cost: bool = False
    ) -> BudgetState:
        if steps < 0:
            raise ValueError("released steps must be non-negative")
        if steps > self.steps_reserved:
            raise ValueError("released steps exceed reserved steps")

        releases_cost_slot = (
            unknown_cost and self.budget.max_cost_usd is not None
        )
        if releases_cost_slot and not self.unknown_cost_reserved:
            raise ValueError("unknown-cost slot is not reserved")
        return dataclasses.replace(
            self,
            steps_reserved=self.steps_reserved - steps,
            unknown_cost_reserved=(
                False if releases_cost_slot else self.unknown_cost_reserved
            ),
        )

    def reconcile_success(
        self,
        steps: int = 1,
        cost: float = 0.0,
        *,
        unknown_cost: bool = False,
    ) -> BudgetState:
        if cost < 0:
            raise ValueError("actual cost must be non-negative")
        released = self.release(steps=steps, unknown_cost=unknown_cost)
        return dataclasses.replace(
            released,
            steps_used=released.steps_used + steps,
            cost_used=released.cost_used + cost,
        )

    @property
    def is_exhausted(self) -> bool:
        if self.steps_used + self.steps_reserved >= self.budget.max_steps:
            return True
        if self.budget.max_cost_usd is not None and (
            self.cost_used >= self.budget.max_cost_usd
            or self.unknown_cost_reserved
        ):
            return True
        return False

    def can_refine(self, depth: int) -> bool:
        return not self.is_exhausted and depth < self.budget.max_refine_depth
