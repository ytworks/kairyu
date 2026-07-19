"""YAML front-end for OrchestratorSpec plus the spec -> Orchestrator builder."""

from __future__ import annotations

from pathlib import Path

import yaml

from kairyu.dsl.spec import OrchestratorSpec, WorkerSpec
from kairyu.engine.backend import EngineBackend
from kairyu.engine.registry import create_backend
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import RoleSpec, chars_cost_model, zero_cost
from kairyu.orchestration.orchestrator import EngineDescriptor, Orchestrator


def load_spec(source: str | Path) -> OrchestratorSpec:
    """Load an OrchestratorSpec from a YAML file path or a YAML string."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source.strip()
                                    and Path(source).exists()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("orchestrator spec YAML must be a mapping at the top level")
    return OrchestratorSpec.model_validate(data)


def _build_worker(worker: WorkerSpec) -> EngineBackend:
    options = dict(worker.options)
    if worker.model is not None:
        options.setdefault("model", worker.model)
    if worker.base_url is not None:
        options.setdefault("base_url", worker.base_url)
    if worker.api_key_env is not None:
        options.setdefault("api_key_env", worker.api_key_env)
    return create_backend(worker.backend, **options)


def build_orchestrator(spec: OrchestratorSpec) -> Orchestrator:
    engines = {worker.name: _build_worker(worker) for worker in spec.workers}
    engine_descriptors = {
        worker.name: EngineDescriptor(
            backend_type=worker.backend,
            model=(
                worker.model
                if worker.model is not None
                else (
                    worker.options.get("model")
                    if isinstance(worker.options.get("model"), str)
                    else None
                )
            ),
        )
        for worker in spec.workers
    }
    roles = (
        tuple(
            RoleSpec(
                name=role.name,
                worker=role.worker,
                prompt=role.prompt,
                role_type=role.role_type,
                depends_on=role.depends_on,
                verifies=role.verifies,
            )
            for role in spec.roles
        )
        or None
    )
    budget = Budget(
        max_steps=spec.budget.max_steps,
        max_refine_depth=spec.budget.max_refine_depth,
        max_cost_usd=spec.budget.max_cost_usd,
    )
    rate = spec.budget.cost_per_1k_chars_usd
    cost_model = chars_cost_model(rate) if rate is not None else zero_cost
    return Orchestrator(
        engines=engines,
        roles=roles,
        budget=budget,
        shared_prefix=spec.shared_prefix,
        cost_model=cost_model,
        engine_descriptors=engine_descriptors,
    )
