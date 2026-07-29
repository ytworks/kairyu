import pytest
from pydantic import ValidationError

from kairyu.dsl.decorators import AgentPool
from kairyu.dsl.loader import build_orchestrator, load_spec

YAML_SPEC = """
shared_prefix: "SYS\\n"
moa_samples: 3
workers:
  - name: tier1
    backend: mock
  - name: tier2
    backend: mock
    options:
      responses:
        "[verifier]": "PASS"
roles:
  - name: planner
    worker: tier2
    prompt: "plan: {query}"
  - name: worker
    worker: tier1
    prompt: "do: {planner}"
    depends_on: [planner]
budget:
  max_steps: 8
  max_refine_depth: 1
"""


def test_yaml_spec_round_trip():
    spec = load_spec(YAML_SPEC)
    assert [w.name for w in spec.workers] == ["tier1", "tier2"]
    assert spec.workers[1].options["responses"]["[verifier]"] == "PASS"
    assert spec.roles[1].depends_on == ("planner",)
    assert spec.budget.max_steps == 8
    assert spec.shared_prefix == "SYS\n"
    assert spec.moa_samples == 3


def test_yaml_spec_from_file(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(YAML_SPEC)
    assert load_spec(path).budget.max_refine_depth == 1


def test_role_referencing_unknown_worker_rejected():
    bad = YAML_SPEC.replace("worker: tier1", "worker: ghost")
    with pytest.raises(ValidationError, match="ghost"):
        load_spec(bad)


def test_orchestrator_requires_at_least_one_worker():
    with pytest.raises(ValidationError, match="workers"):
        load_spec("workers: []")


def test_duplicate_worker_names_are_rejected():
    with pytest.raises(ValidationError, match="worker names must be unique"):
        load_spec(
            """
workers:
  - {name: repeated, backend: mock}
  - {name: repeated, backend: mock}
"""
        )


def test_decorator_pool_builds_equivalent_spec():
    pool = AgentPool(shared_prefix="SYS\n", moa_samples=3)
    pool.worker("tier1", backend="mock")
    pool.worker("tier2", backend="mock", options={"responses": {"[verifier]": "PASS"}})
    pool.budget(max_steps=8, max_refine_depth=1)

    @pool.role(worker="tier2")
    def planner():
        return "plan: {query}"

    @pool.role(worker="tier1", depends_on=("planner",))
    def worker():
        return "do: {planner}"

    assert pool.to_spec() == load_spec(YAML_SPEC)


async def test_build_orchestrator_runs_end_to_end():
    orchestrator = build_orchestrator(load_spec(YAML_SPEC))
    descriptor = orchestrator.describe_routing()
    assert descriptor["configured_engines"] == {
        "tier1": {"backend_type": "mock", "model": None},
        "tier2": {"backend_type": "mock", "model": None},
    }
    assert descriptor["moa_samples"] == 3
    assert descriptor["target_resolution"]["multi_agent"]["mode"] == "moa"
    result = await orchestrator.run(
        "First, plan the work. Then execute it. Finally, summarize the outcome."
    )
    assert result.route.target == "multi_agent"
    assert result.text


@pytest.mark.parametrize("value", [-1, 17])
def test_moa_samples_rejects_unsafe_values(value):
    with pytest.raises(ValidationError, match="moa_samples"):
        load_spec(YAML_SPEC.replace("moa_samples: 3", f"moa_samples: {value}"))


def test_build_openai_worker_preserves_keyless_auth(monkeypatch):
    captured = {}

    def fake_create_backend(name, **options):
        captured["name"] = name
        captured["options"] = options
        return object()

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", fake_create_backend)
    build_orchestrator(
        load_spec(
            """
workers:
  - name: local
    backend: openai
    model: default
    base_url: http://replica:8000/v1
    api_key_env: null
"""
        )
    )

    assert captured == {
        "name": "openai",
        "options": {
            "model": "default",
            "base_url": "http://replica:8000/v1",
            "api_key_env": None,
        },
    }


def test_openai_worker_profile_is_forwarded_and_validated(monkeypatch):
    captured = {}

    def fake_create_backend(name, **options):
        captured["name"] = name
        captured["options"] = options
        return object()

    monkeypatch.setattr("kairyu.dsl.loader.create_backend", fake_create_backend)
    build_orchestrator(
        load_spec(
            """
workers:
  - name: local
    backend: openai
    model: default
    base_url: http://replica:8000/v1
    api_key_env: null
    options: {upstream: kairyu}
"""
        )
    )
    assert captured["options"]["upstream"] == "kairyu"

    with pytest.raises(ValidationError, match="unknown OpenAI-compatible upstream"):
        load_spec(
            """
workers:
  - name: invalid
    backend: openai
    options: {upstream: inferred}
"""
        )
