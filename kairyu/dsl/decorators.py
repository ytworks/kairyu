"""Decorator front-end producing the same OrchestratorSpec as the YAML loader."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kairyu.dsl.spec import BudgetSpec, OrchestratorSpec, RoleNodeSpec, WorkerSpec


class AgentPool:
    """Collects workers, roles and a budget; ``to_spec()`` freezes them."""

    def __init__(self, shared_prefix: str = "") -> None:
        self._shared_prefix = shared_prefix
        self._workers: list[WorkerSpec] = []
        self._roles: list[RoleNodeSpec] = []
        self._budget = BudgetSpec()

    def worker(
        self,
        name: str,
        backend: str = "mock",
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        options: dict | None = None,
    ) -> WorkerSpec:
        spec = WorkerSpec(
            name=name,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            options=options or {},
        )
        self._workers.append(spec)
        return spec

    def budget(
        self,
        max_steps: int = 16,
        max_refine_depth: int = 2,
        max_cost_usd: float | None = None,
    ) -> BudgetSpec:
        self._budget = BudgetSpec(
            max_steps=max_steps, max_refine_depth=max_refine_depth, max_cost_usd=max_cost_usd
        )
        return self._budget

    def role(
        self,
        worker: str,
        name: str | None = None,
        role_type: str = "worker",
        depends_on: Sequence[str] = (),
        verifies: str | None = None,
    ) -> Callable[[Callable[[], str]], Callable[[], str]]:
        """Register the decorated function's returned string as a prompt template."""

        def decorator(template_fn: Callable[[], str]) -> Callable[[], str]:
            self._roles.append(
                RoleNodeSpec(
                    name=name or template_fn.__name__,
                    worker=worker,
                    prompt=template_fn(),
                    role_type=role_type,
                    depends_on=tuple(depends_on),
                    verifies=verifies,
                )
            )
            return template_fn

        return decorator

    def to_spec(self) -> OrchestratorSpec:
        return OrchestratorSpec(
            workers=tuple(self._workers),
            roles=tuple(self._roles),
            budget=self._budget,
            shared_prefix=self._shared_prefix,
        )
