"""Pydantic schema shared by the YAML loader and the decorator front-end (D7)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    backend: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict = Field(default_factory=dict)


class RoleNodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    worker: str
    prompt: str
    role_type: str = "worker"
    depends_on: tuple[str, ...] = ()
    verifies: str | None = None


class BudgetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=16, ge=1)
    max_refine_depth: int = Field(default=2, ge=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class OrchestratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: tuple[WorkerSpec, ...]
    roles: tuple[RoleNodeSpec, ...] = ()
    budget: BudgetSpec = BudgetSpec()
    shared_prefix: str = ""

    @model_validator(mode="after")
    def _roles_reference_known_workers(self) -> OrchestratorSpec:
        known = {worker.name for worker in self.workers}
        for role in self.roles:
            if role.worker not in known:
                raise ValueError(
                    f"role {role.name!r} references unknown worker {role.worker!r}; "
                    f"known workers: {sorted(known)}"
                )
        return self
