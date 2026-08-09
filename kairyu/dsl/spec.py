"""Pydantic schema shared by the YAML loader and the decorator front-end (D7)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairyu.engine.openai_capabilities import resolve_openai_capabilities


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    backend: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_openai_capabilities(self) -> WorkerSpec:
        if self.backend == "openai":
            resolve_openai_capabilities(
                self.options.get("upstream", "generic"),
                self.options.get("capabilities"),
            )
        return self


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
    cost_per_1k_chars_usd: float | None = Field(default=None, gt=0)


class RouterSpec(BaseModel):
    """Immutable routing policy loaded from a calibrated artifact."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["rules", "calibrated"] = "rules"
    artifact: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_mode: Literal["auto", "auto-max"] = "auto"

    @model_validator(mode="after")
    def _calibrated_artifact_is_pinned(self) -> RouterSpec:
        if self.kind == "calibrated" and (self.artifact is None or self.sha256 is None):
            raise ValueError("calibrated router requires artifact and sha256")
        if self.kind == "rules" and (self.artifact is not None or self.sha256 is not None):
            raise ValueError("rules router cannot specify artifact or sha256")
        return self


class OrchestratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: tuple[WorkerSpec, ...] = Field(min_length=1)
    roles: tuple[RoleNodeSpec, ...] = ()
    budget: BudgetSpec = BudgetSpec()
    router: RouterSpec = RouterSpec()
    shared_prefix: str = ""
    # Zero keeps the standard Conductor route. A positive value turns the
    # multi-agent route into that many parallel MoA proposals plus synthesis.
    moa_samples: int = Field(default=0, ge=0, le=16)

    @model_validator(mode="after")
    def _roles_reference_known_workers(self) -> OrchestratorSpec:
        worker_names = [worker.name for worker in self.workers]
        known = set(worker_names)
        if len(known) != len(worker_names):
            raise ValueError("worker names must be unique")
        for role in self.roles:
            if role.worker not in known:
                raise ValueError(
                    f"role {role.name!r} references unknown worker {role.worker!r}; "
                    f"known workers: {sorted(known)}"
                )
        return self
