"""Pydantic schema shared by the YAML loader and the decorator front-end (D7)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairyu.engine.openai_capabilities import resolve_openai_capabilities


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    engine_ref: str | None = Field(default=None, min_length=1)
    # A deployment-owned sandbox execution service (ECO-D1). Mutually exclusive
    # with engine_ref and with every generation factory field.
    executor_ref: str | None = Field(default=None, min_length=1)
    backend: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_openai_capabilities(self) -> WorkerSpec:
        if self.engine_ref is not None and self.executor_ref is not None:
            raise ValueError("a worker cannot declare both engine_ref and executor_ref")
        if self.engine_ref is not None or self.executor_ref is not None:
            kind = "engine_ref" if self.engine_ref is not None else "executor_ref"
            factory_fields = {
                "backend",
                "model",
                "base_url",
                "api_key_env",
                "options",
            } & self.model_fields_set
            if factory_fields:
                raise ValueError(
                    f"{kind} workers cannot also declare factory fields: "
                    f"{sorted(factory_fields)}"
                )
            return self
        if self.backend == "openai":
            resolve_openai_capabilities(
                self.options.get("upstream", "generic"),
                self.options.get("capabilities"),
            )
        return self


class RoleSamplingSpec(BaseModel):
    """Per-role sampling overrides (EO-D9). The selected final unit carries the
    caller's public intent and must not declare overrides; internal roles are
    still capped by ``internal_max_tokens``."""

    model_config = ConfigDict(frozen=True)

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    seed_offset: int | None = Field(default=None, ge=0)
    stop: tuple[str, ...] = ()


class ExecutionLimitsSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_time_s: float = Field(default=10.0, gt=0, le=120)
    cpu_time_s: float = Field(default=8.0, gt=0, le=120)
    memory_mb: int = Field(default=512, ge=32, le=4096)
    processes: int = Field(default=64, ge=1, le=1024)
    output_bytes: int = Field(default=32768, ge=1024, le=1048576)


class ExecutorRoleSpec(BaseModel):
    """Configuration for a sandbox execution role (ECO-D3)."""

    model_config = ConfigDict(frozen=True)

    code_from: tuple[str, ...] = Field(min_length=1)
    tests_from: tuple[str, ...] = Field(min_length=1)
    mode: Literal["single", "matrix"] = "single"
    limits: ExecutionLimitsSpec = ExecutionLimitsSpec()


class RoleNodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    worker: str
    prompt: str = ""
    role_type: str = "worker"
    depends_on: tuple[str, ...] = ()
    verifies: str | None = None
    sampling: RoleSamplingSpec | None = None
    executor: ExecutorRoleSpec | None = None
    # Appended after the prompt body on every attempt, including verifier
    # refinements — for worker-native scaffolding that must end the prompt.
    prompt_suffix: str = ""
    # Alternate body rendered for the selected final unit when the head is
    # absent or disabled for the call (tools, n>1, ...): the publisher then
    # writes the complete answer instead of continuing a committed opening
    # (issue #496). Uses the same placeholders and prompt_suffix.
    prompt_headless: str = ""
    # Declares that this role's rendered scaffold already closed the model's
    # private-reasoning span, so upstream reasoning-classified output with an
    # empty public text is the answer itself, not hidden deliberation.
    reasoning_closed: bool = False

    @model_validator(mode="after")
    def _executor_shape(self) -> RoleNodeSpec:
        if (self.role_type == "executor") != (self.executor is not None):
            raise ValueError(
                f"role {self.name!r}: role_type 'executor' and an executor block "
                "must be declared together"
            )
        if self.executor is not None:
            if self.sampling is not None:
                raise ValueError(f"executor role {self.name!r} cannot declare sampling")
            if self.prompt_headless or self.reasoning_closed:
                raise ValueError(
                    f"executor role {self.name!r} cannot declare prompt_headless "
                    "or reasoning_closed"
                )
            deps = set(self.depends_on)
            missing = (set(self.executor.code_from) | set(self.executor.tests_from)) - deps
            if missing:
                raise ValueError(
                    f"executor role {self.name!r} references roles outside its "
                    f"depends_on: {sorted(missing)}"
                )
        elif not self.prompt:
            raise ValueError(f"role {self.name!r} requires a prompt")
        return self


class BudgetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=16, ge=1)
    max_refine_depth: int = Field(default=2, ge=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    cost_per_1k_chars_usd: float | None = Field(default=None, gt=0)


class RouterThresholdSpec(BaseModel):
    """Validated operator overrides for the deterministic rule router."""

    model_config = ConfigDict(frozen=True)

    multi_step_markers: int = Field(default=3, ge=0)
    multi_agent_min_chars: int = Field(default=2000, ge=1)
    reasoning_keywords: int = Field(default=2, ge=0)
    math_symbols: int = Field(default=3, ge=0)
    tier2_min_chars: int = Field(default=600, ge=1)


class RouterSpec(BaseModel):
    """Immutable routing policy loaded from a calibrated artifact."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["rules", "calibrated"] = "rules"
    artifact: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_mode: Literal["auto", "auto-max"] = "auto"
    thresholds: RouterThresholdSpec | None = None

    @model_validator(mode="after")
    def _calibrated_artifact_is_pinned(self) -> RouterSpec:
        if self.kind == "calibrated" and (self.artifact is None or self.sha256 is None):
            raise ValueError("calibrated router requires artifact and sha256")
        if self.kind == "calibrated" and self.thresholds is not None:
            raise ValueError("calibrated router thresholds come from its pinned artifact")
        if self.kind == "rules" and (self.artifact is not None or self.sha256 is not None):
            raise ValueError("rules router cannot specify artifact or sha256")
        return self


class OrchestratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: tuple[WorkerSpec, ...] = Field(min_length=1)
    roles: tuple[RoleNodeSpec, ...] = ()
    # Optional second role DAG for requests outside the primary DAG's task
    # specialization (issue #509). Both profiles are full ensembles served
    # under the same model name; the orchestrator selects per request and no
    # profile ever routes to a single engine directly.
    general_roles: tuple[RoleNodeSpec, ...] = ()
    budget: BudgetSpec = BudgetSpec()
    router: RouterSpec = RouterSpec()
    shared_prefix: str = ""
    # Private planning/proposal/verification output is bounded independently
    # from the caller's public final-answer allowance.  Keeping this in the
    # orchestration policy makes latency/cost tuning portable across backends.
    internal_max_tokens: int = Field(default=1024, ge=1, le=32768)
    # Zero keeps the standard Conductor route. A positive value turns the
    # multi-agent route into that many parallel MoA proposals plus synthesis.
    moa_samples: int = Field(default=0, ge=0, le=16)
    # Completed pre-final stage outputs may be surfaced separately from the
    # answer. Hidden is the safe and backward-compatible default.
    expose_intermediate_outputs: bool = False

    @model_validator(mode="after")
    def _roles_reference_known_workers(self) -> OrchestratorSpec:
        worker_names = [worker.name for worker in self.workers]
        known = set(worker_names)
        if len(known) != len(worker_names):
            raise ValueError("worker names must be unique")
        executor_workers = {
            worker.name for worker in self.workers if worker.executor_ref is not None
        }
        if self.general_roles and not self.roles:
            raise ValueError("general_roles requires a primary roles DAG")
        for profile, roles in (("roles", self.roles), ("general_roles", self.general_roles)):
            for role in roles:
                if role.worker not in known:
                    raise ValueError(
                        f"{profile}: role {role.name!r} references unknown worker "
                        f"{role.worker!r}; known workers: {sorted(known)}"
                    )
                if (role.role_type == "executor") != (role.worker in executor_workers):
                    raise ValueError(
                        f"{profile}: role {role.name!r}: executor roles must use an "
                        "executor_ref worker and generation roles must not"
                    )
            heads = [role for role in roles if role.role_type == "head"]
            if len(heads) > 1:
                raise ValueError(f"{profile}: at most one role may declare role_type 'head'")
            if heads:
                head = heads[0]
                if head.depends_on:
                    raise ValueError(
                        f"{profile}: head role {head.name!r} cannot declare dependencies"
                    )
                if any(role.verifies == head.name for role in roles):
                    raise ValueError(f"{profile}: head role {head.name!r} cannot be verified")
                if not any(head.name in role.depends_on for role in roles):
                    raise ValueError(
                        f"{profile}: head role {head.name!r} must feed at least one "
                        "downstream role"
                    )
        return self
