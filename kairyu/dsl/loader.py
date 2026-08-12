"""YAML front-end for OrchestratorSpec plus the spec -> Orchestrator builder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from kairyu.dsl.spec import OrchestratorSpec, WorkerSpec
from kairyu.engine.backend import EngineBackend
from kairyu.engine.registry import create_backend
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import RoleSpec, chars_cost_model, zero_cost
from kairyu.orchestration.orchestrator import EngineDescriptor, Orchestrator
from kairyu.orchestration.router import RouteThresholds, RuleRouter, load_calibrated_router
from kairyu.sampling_params import SamplingParams


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
    if worker.engine_ref is not None:
        raise ValueError("engine_ref workers require deployment engine resolution")
    options = dict(worker.options)
    if worker.model is not None:
        options.setdefault("model", worker.model)
    if worker.base_url is not None:
        options.setdefault("base_url", worker.base_url)
    # OpenAICompatBackend defaults to OPENAI_API_KEY when this argument is
    # omitted. Preserve WorkerSpec's None so local node-to-node workers remain
    # explicitly keyless.
    if worker.backend == "openai" or worker.api_key_env is not None:
        options.setdefault("api_key_env", worker.api_key_env)
    return create_backend(worker.backend, **options)


def build_orchestrator(
    spec: OrchestratorSpec,
    *,
    engine_refs: Mapping[str, EngineBackend] | None = None,
) -> Orchestrator:
    available_refs = dict(engine_refs or {})
    engines: dict[str, EngineBackend] = {}
    owned_engines: list[EngineBackend] = []
    for worker in spec.workers:
        if worker.engine_ref is not None:
            try:
                engines[worker.name] = available_refs[worker.engine_ref]
            except KeyError as error:
                raise ValueError(
                    f"worker {worker.name!r} references unknown deployment engine "
                    f"{worker.engine_ref!r}"
                ) from error
        else:
            engine = _build_worker(worker)
            engines[worker.name] = engine
            owned_engines.append(engine)
    engine_descriptors = {
        worker.name: EngineDescriptor(
            backend_type=(
                type(engines[worker.name]).__name__
                if worker.engine_ref is not None
                else worker.backend
            ),
            model=(
                worker.engine_ref
                if worker.engine_ref is not None
                else (
                    worker.model
                    if worker.model is not None
                    else (
                        worker.options.get("model")
                        if isinstance(worker.options.get("model"), str)
                        else None
                    )
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
    router = (
        RuleRouter(
            RouteThresholds(**spec.router.thresholds.model_dump())
            if spec.router.thresholds is not None
            else RouteThresholds()
        )
        if spec.router.kind == "rules"
        else load_calibrated_router(
            spec.router.artifact or "",
            expected_sha256=spec.router.sha256 or "",
            target_mode=spec.router.target_mode,
        )
    )
    return Orchestrator(
        engines=engines,
        router=router,
        roles=roles,
        budget=budget,
        shared_prefix=spec.shared_prefix,
        sampling_params=SamplingParams(max_tokens=spec.internal_max_tokens),
        cost_model=cost_model,
        moa_samples=spec.moa_samples,
        engine_descriptors=engine_descriptors,
        owned_engines=owned_engines,
    )
