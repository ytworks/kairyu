"""ClusterSpec: static 2-node topology config (design m6 D1)."""

from __future__ import annotations

import pytest

from kairyu.orchestration.cluster import ClusterSpec, NodeSpec, load_cluster_spec

_DP_YAML = """
nodes:
  - {name: node-a, host: 10.0.0.1, port: 8000, gpus: 8, role: replica}
  - {name: node-b, host: 10.0.0.2, port: 8000, gpus: 8, role: replica}
"""

_PD_YAML = """
nodes:
  - {name: node-a, host: 10.0.0.1, gpus: 8, role: prefill}
  - {name: node-b, host: 10.0.0.2, gpus: 8, role: decode}
"""

_PP_YAML = """
nodes:
  - {name: node-a, host: 10.0.0.1, gpus: 8, role: pp-stage, pp_stage: 0}
  - {name: node-b, host: 10.0.0.2, gpus: 8, role: pp-stage, pp_stage: 1}
"""


def test_loads_dp_pd_and_pp_topologies_from_yaml() -> None:
    dp = load_cluster_spec(_DP_YAML)
    assert tuple(node.role for node in dp.nodes) == ("replica", "replica")
    assert dp.nodes[0].endpoint == "10.0.0.1:8000"

    pd = load_cluster_spec(_PD_YAML)
    assert {node.role for node in pd.nodes} == {"prefill", "decode"}

    pp = load_cluster_spec(_PP_YAML)
    assert sorted(node.pp_stage for node in pp.nodes) == [0, 1]


def test_rejects_duplicate_node_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        ClusterSpec(
            nodes=(
                NodeSpec(name="a", host="h1", role="replica"),
                NodeSpec(name="a", host="h2", role="replica"),
            )
        )


def test_rejects_more_than_two_nodes() -> None:
    with pytest.raises(ValueError, match="2 nodes"):
        ClusterSpec(
            nodes=tuple(
                NodeSpec(name=f"n{i}", host=f"h{i}", role="replica") for i in range(3)
            )
        )


def test_pp_stage_field_required_iff_pp_role() -> None:
    with pytest.raises(ValueError, match="pp_stage"):
        NodeSpec(name="a", host="h", role="pp-stage")  # missing stage index
    with pytest.raises(ValueError, match="pp_stage"):
        NodeSpec(name="a", host="h", role="replica", pp_stage=0)  # stray stage index


def test_pp_and_pd_roles_are_orthogonal() -> None:
    # design m6 D1: a node is either a PP stage or a P-D role, never mixed
    with pytest.raises(ValueError, match="orthogonal"):
        ClusterSpec(
            nodes=(
                NodeSpec(name="a", host="h1", role="pp-stage", pp_stage=0),
                NodeSpec(name="b", host="h2", role="prefill"),
            )
        )


def test_pp_requires_exactly_stages_zero_and_one() -> None:
    with pytest.raises(ValueError, match="stages"):
        ClusterSpec(
            nodes=(
                NodeSpec(name="a", host="h1", role="pp-stage", pp_stage=0),
                NodeSpec(name="b", host="h2", role="pp-stage", pp_stage=2),
            )
        )


def test_prefill_and_decode_must_pair() -> None:
    with pytest.raises(ValueError, match="decode"):
        ClusterSpec(nodes=(NodeSpec(name="a", host="h1", role="prefill"),))
    with pytest.raises(ValueError, match="prefill"):
        ClusterSpec(nodes=(NodeSpec(name="a", host="h1", role="decode"),))


def test_rejects_non_mapping_yaml() -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_cluster_spec("- just\n- a list\n")
