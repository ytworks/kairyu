"""DeploymentSpec: the YAML schema consumed by `kairyu serve` (design m7 D3).

Deliberately separate from ``ClusterSpec`` (m6 D1): ClusterSpec validates the
2-node TP/PP/P-D coherence domain; a serving deployment declares which models
the process serves and which remote replicas a pool fans out to. A replica
node's intra-node GPU layout may reference a ClusterSpec file — the two specs
compose, they do not merge.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.resolver import BaseResolver

from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits

_DEFAULT_PORT = 8000


class BackendSpec(BaseModel):
    """One engine backend: registry name + factory kwargs."""

    model_config = ConfigDict(frozen=True)

    backend: str
    options: dict = Field(default_factory=dict)
    health_url: str | None = Field(
        default=None,
        description=(
            "Readiness endpoint the prober GETs for this replica; defaults to "
            "<base_url minus /v1>/readyz for openai backends, None otherwise. "
            "Readiness (not liveness) so a drained/wedged node stays ejected (O3)."
        ),
    )

    def resolved_health_url(self) -> str | None:
        if self.health_url is not None:
            return self.health_url
        base = self.options.get("base_url")
        if not isinstance(base, str):
            return None
        root = base.rstrip("/")
        root = root.removesuffix("/v1")
        return f"{root}/readyz"


class PoolSpec(BaseModel):
    """A ReplicaPool of N backends served under one model name."""

    model_config = ConfigDict(frozen=True)

    replicas: tuple[BackendSpec, ...] = Field(min_length=1)
    unhealthy_after: int = Field(default=3, ge=1)
    queue_depth_threshold: int = Field(default=8, ge=0)
    probe_interval_s: float = Field(
        default=5.0,
        gt=0,
        description="Interval of the serve-layer health prober (m7 D4).",
    )


class ServerSection(ServerSettings):
    """Bind address plus the ServerSettings fields, flat in YAML."""

    host: str = "0.0.0.0"
    port: int = Field(default=_DEFAULT_PORT, ge=1, le=65535)


class OrchestratorSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: str = Field(description="Path to an OrchestratorSpec YAML (kairyu.dsl).")


class BatchSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: str
    max_concurrency: int = Field(default=4, ge=1)


class TenantLimitsSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests_per_minute: int = Field(default=600, ge=1)
    tokens_per_minute: int = Field(default=200_000, ge=1)


class TenantSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_tenant: str = "default"
    key_tenants: dict[str, str] = Field(default_factory=dict, repr=False)
    limits: dict[str, TenantLimitsSection] = Field(default_factory=dict)


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    server: ServerSection = ServerSection()
    engines: dict[str, BackendSpec] = Field(default_factory=dict)
    pools: dict[str, PoolSpec] = Field(default_factory=dict)
    # served model name -> HF Jinja chat template (inline text or *.jinja path);
    # per-MODEL, not per-replica — one map avoids engines-vs-pools ambiguity
    # and stays out of BackendSpec.options (factory kwargs). m9 D2.
    chat_templates: dict[str, str] = Field(default_factory=dict)
    orchestrator: OrchestratorSection | None = None
    # served auto-model name -> orchestrator spec; serves ANY number of named
    # orchestrations (e.g. kairyu-auto + kairyu-auto-max) from one YAML. The
    # legacy single `orchestrator:` key stays and is served as "kairyu-auto".
    orchestrators: dict[str, OrchestratorSection] = Field(default_factory=dict)
    batch: BatchSection | None = None
    tenants: TenantSection | None = None

    @model_validator(mode="after")
    def _validate(self) -> DeploymentSpec:
        if not self.engines and not self.pools:
            raise ValueError("deployment spec must declare at least one engine or pool")
        overlap = self.engines.keys() & self.pools.keys()
        if overlap:
            raise ValueError(
                f"names {sorted(overlap)} appear in both engines: and pools:; "
                "served model names must be unique"
            )
        unknown = self.chat_templates.keys() - self.engines.keys() - self.pools.keys()
        if unknown:
            raise ValueError(
                f"chat_templates for unknown models {sorted(unknown)}; "
                "keys must match engines: or pools: names"
            )
        if any(not name for name in self.orchestrators):
            raise ValueError("orchestrators: names must be non-empty strings")
        auto_overlap = self.orchestrators.keys() & (
            self.engines.keys() | self.pools.keys()
        )
        if auto_overlap:
            raise ValueError(
                f"orchestrators names {sorted(auto_overlap)} collide with "
                "engines:/pools: names; served model names must be unique"
            )
        if self.orchestrator is not None and "kairyu-auto" in self.orchestrators:
            raise ValueError(
                'both orchestrator: and orchestrators["kairyu-auto"] are set; '
                "declare kairyu-auto once (the legacy orchestrator: key is "
                'served as "kairyu-auto")'
            )
        if self.tenants is not None:
            TenantConfig.from_mapping(
                key_tenants=self.tenants.key_tenants,
                limits={
                    tenant: TenantLimits(
                        requests_per_minute=section.requests_per_minute,
                        tokens_per_minute=section.tokens_per_minute,
                    )
                    for tenant, section in self.tenants.limits.items()
                },
                default_tenant=self.tenants.default_tenant,
                resolved_api_keys=self.server.resolve_api_keys(),
            )
        return self


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that retains a path for duplicate-key diagnostics."""

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self._mapping_paths: dict[int, tuple[str, ...]] = {}

    def get_single_data(self):
        node = self.get_single_node()
        if node is None:
            return None
        self._index_paths(node, ())
        return self.construct_document(node)

    def _index_paths(self, node: Node, path: tuple[str, ...]) -> None:
        if isinstance(node, MappingNode):
            self._mapping_paths[id(node)] = path
            for key_node, value_node in node.value:
                segment = key_node.value if isinstance(key_node, ScalarNode) else "<key>"
                self._index_paths(value_node, (*path, segment))
        elif isinstance(node, SequenceNode):
            for index, value_node in enumerate(node.value):
                self._index_paths(value_node, (*path, f"[{index}]"))


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
):
    seen: set[object] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:
            continue
        if duplicate:
            path = ".".join(loader._mapping_paths.get(id(node), ())) or "<root>"
            raise ValueError(f"duplicate mapping key {key!r} at {path}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_deployment_spec(source: str | Path) -> DeploymentSpec:
    """Load a DeploymentSpec from a YAML file path or a YAML string."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source.strip() and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(data, dict):
        raise ValueError("deployment spec YAML must be a mapping at the top level")
    return DeploymentSpec.model_validate(data)
