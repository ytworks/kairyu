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

    def charge(self, steps: int = 1, cost: float = 0.0) -> BudgetState:
        return dataclasses.replace(
            self, steps_used=self.steps_used + steps, cost_used=self.cost_used + cost
        )

    @property
    def is_exhausted(self) -> bool:
        if self.steps_used >= self.budget.max_steps:
            return True
        if self.budget.max_cost_usd is not None and self.cost_used > self.budget.max_cost_usd:
            return True
        return False

    def can_refine(self, depth: int) -> bool:
        return not self.is_exhausted and depth < self.budget.max_refine_depth
