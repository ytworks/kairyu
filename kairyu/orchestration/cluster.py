"""Static cluster topology config: nodes, roles, endpoints (design m6 D1).

Two-node scope per Goal G2; validation enforces the PP-vs-P-D orthogonality
the design requires (a node is a PP stage or a P-D role, never both mixed in
one cluster). No Ray: rendezvous is torchrun-style from this spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_MAX_NODES = 2  # G2 non-goal: >2 nodes

NodeRole = Literal["replica", "prefill", "decode", "pp-stage"]


class NodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    host: str
    port: int = Field(default=8000, ge=1, le=65535)
    gpus: int = Field(default=8, ge=1)
    role: NodeRole
    pp_stage: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _stage_iff_pp_role(self) -> NodeSpec:
        if self.role == "pp-stage" and self.pp_stage is None:
            raise ValueError("pp-stage nodes require a pp_stage index")
        if self.role != "pp-stage" and self.pp_stage is not None:
            raise ValueError(f"pp_stage is only valid for pp-stage nodes, got role {self.role!r}")
        return self

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


class ClusterSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[NodeSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_topology(self) -> ClusterSpec:
        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique")
        if len(self.nodes) > _MAX_NODES:
            raise ValueError(f"Goal G2 covers at most 2 nodes, got {len(self.nodes)}")
        roles = {node.role for node in self.nodes}
        if "pp-stage" in roles and roles & {"prefill", "decode"}:
            raise ValueError("PP stages and P-D roles are orthogonal; do not mix them")
        if "pp-stage" in roles:
            stages = sorted(
                node.pp_stage for node in self.nodes if node.role == "pp-stage"
            )
            if stages != [0, 1]:
                raise ValueError(f"PP=2 requires exactly stages [0, 1], got {stages}")
        if "prefill" in roles and "decode" not in roles:
            raise ValueError("prefill nodes require at least one decode node")
        if "decode" in roles and "prefill" not in roles:
            raise ValueError("decode nodes require at least one prefill node")
        return self


def load_cluster_spec(source: str | Path) -> ClusterSpec:
    """Load a ClusterSpec from a YAML file path or a YAML string."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source.strip() and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("cluster spec YAML must be a mapping at the top level")
    return ClusterSpec.model_validate(data)
