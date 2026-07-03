"""DeploymentSpec: the YAML schema consumed by `kairyu serve` (design m7 D3).

Deliberately separate from ``ClusterSpec`` (m6 D1): ClusterSpec validates the
2-node TP/PP/P-D coherence domain; a serving deployment declares which models
the process serves and which remote replicas a pool fans out to. A replica
node's intra-node GPU layout may reference a ClusterSpec file — the two specs
compose, they do not merge.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairyu.entrypoints.server.settings import ServerSettings

_DEFAULT_PORT = 8000


class BackendSpec(BaseModel):
    """One engine backend: registry name + factory kwargs."""

    model_config = ConfigDict(frozen=True)

    backend: str
    options: dict = Field(default_factory=dict)
    health_url: str | None = Field(
        default=None,
        description=(
            "Health endpoint the prober GETs for this replica; defaults to "
            "<base_url minus /v1>/health for openai backends, None otherwise."
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
        return f"{root}/health"


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


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    server: ServerSection = ServerSection()
    engines: dict[str, BackendSpec] = Field(default_factory=dict)
    pools: dict[str, PoolSpec] = Field(default_factory=dict)
    orchestrator: OrchestratorSection | None = None
    batch: BatchSection | None = None

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
        return self


def load_deployment_spec(source: str | Path) -> DeploymentSpec:
    """Load a DeploymentSpec from a YAML file path or a YAML string."""
    import yaml

    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source.strip() and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("deployment spec YAML must be a mapping at the top level")
    return DeploymentSpec.model_validate(data)
