"""m10a gates: dynamic membership, HRW remap property, drain, registry,
reconciler, tracing spans, helm render."""

import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

from kairyu.deploy.registry import (
    PoolReconciler,
    RegistryDiscovery,
    ReplicaRegistry,
    StaticDiscovery,
)
from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
)
from kairyu.orchestration.replica import ReplicaPool

pytestmark = pytest.mark.asyncio


class MockBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def stream(self, request):
        yield GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def shutdown(self) -> None:
        return None


def _request(session: str | None = None) -> GenerationRequest:
    hint = CacheHint(session_id=session) if session else None
    return GenerationRequest(
        request_id="req", prompt="p", sampling_params=SamplingParams(), cache_hint=hint
    )


class TestDynamicMembership:
    async def test_add_drain_remove_lifecycle(self):
        pool = ReplicaPool({"a": MockBackend(), "b": MockBackend()})
        pool.add_replica("c", MockBackend(), health_url="http://c/health")
        assert pool.replica_ids == ("a", "b", "c")
        assert pool.health_url("c") == "http://c/health"

        pool.drain("a")
        assert pool.is_draining("a")
        # drained replicas take no NEW placements
        for _ in range(8):
            await pool.generate(_request())
        assert pool.outstanding_by_id()["a"] == 0

        pool.remove_replica("a")
        assert pool.replica_ids == ("b", "c")
        with pytest.raises(ValueError, match="already"):
            pool.add_replica("b", MockBackend())

    async def test_remove_refuses_inflight_then_force(self):
        import asyncio

        class SlowBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self.release = asyncio.Event()

            async def generate(self, request):
                await self.release.wait()
                return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

        slow = SlowBackend()
        pool = ReplicaPool({"s": slow})
        task = asyncio.create_task(pool.generate(_request()))
        await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="in-flight"):
            pool.remove_replica("s")
        pool.remove_replica("s", force=True)
        slow.release.set()
        await task  # late completion on removed id is a no-op (A2)

    async def test_all_draining_raises(self):
        pool = ReplicaPool([MockBackend()])
        pool.drain("0")
        with pytest.raises(RuntimeError, match="eligible"):
            await pool.generate(_request())

    async def test_probe_never_clears_draining(self):
        pool = ReplicaPool([MockBackend(), MockBackend()])
        pool.drain("1")
        await pool.probe(1)  # legacy ordinal accepted
        assert pool.is_draining("1")


class TestHrwRemapProperty:
    def _mapping(self, pool: ReplicaPool, sessions: list[str]) -> dict[str, str]:
        return {
            session: pool._select(_request(session))[0] for session in sessions
        }

    async def test_removal_remaps_only_departed_sessions(self):
        backends = {str(i): MockBackend() for i in range(8)}
        pool = ReplicaPool(backends)
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        pool.remove_replica("3")
        after = self._mapping(pool, sessions)
        moved = [s for s in sessions if before[s] != after[s]]
        assert all(before[s] == "3" for s in moved)  # only its own sessions

    async def test_addition_remaps_about_one_over_n(self):
        pool = ReplicaPool({str(i): MockBackend() for i in range(8)})
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        pool.add_replica("8", MockBackend())
        after = self._mapping(pool, sessions)
        moved = sum(before[s] != after[s] for s in sessions)
        assert moved <= 400 / 9 * 1.6  # ~1/N with slack
        assert all(after[s] == "8" for s in sessions if before[s] != after[s])


class TestRegistryAndReconciler:
    def test_ttl_expiry(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register("r1", "http://r1/v1", ttl_s=10)
        assert registry.alive() == {"r1": "http://r1/v1"}
        clock["t"] = 9.0
        registry.heartbeat("r1")
        clock["t"] = 18.0
        assert registry.alive() == {"r1": "http://r1/v1"}
        clock["t"] = 30.0
        assert registry.alive() == {}
        with pytest.raises(KeyError):
            registry.heartbeat("ghost")

    def test_reconciler_adds_and_drain_removes(self):
        pool = ReplicaPool({"old": MockBackend()})
        members = {"old": "http://old/v1", "new": "http://new/v1"}
        source = StaticDiscovery(members)
        reconciler = PoolReconciler(
            pool, source, factory=lambda addr: (MockBackend(), f"{addr}/health")
        )
        result = reconciler.reconcile()
        assert result["added"] == ["new"]
        assert pool.replica_ids == ("old", "new")

        members.pop("old")
        source._members.pop("old")
        result = reconciler.reconcile()
        assert result["removed"] == ["old"]
        assert pool.replica_ids == ("new",)

    def test_reconciler_retries_inflight_removal(self):
        pool = ReplicaPool({"busy": MockBackend(), "idle": MockBackend()})
        pool._entries["busy"].outstanding = 1  # simulate in-flight
        source = StaticDiscovery({"idle": "http://idle/v1"})
        reconciler = PoolReconciler(pool, source, factory=lambda a: (MockBackend(), None))
        result = reconciler.reconcile()
        assert result["removed"] == []
        assert "busy" in result["draining"]
        assert pool.is_draining("busy")
        pool._entries["busy"].outstanding = 0
        result = reconciler.reconcile()
        assert result["removed"] == ["busy"]

    def test_registry_discovery_bridges(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register("r1", "http://r1/v1")
        assert RegistryDiscovery(registry).poll() == {"r1": "http://r1/v1"}


class TestTracing:
    async def test_spans_recorded_when_enabled(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from kairyu.telemetry import configure_tracing, traced_span

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        configure_tracing(True)
        try:
            with traced_span("kairyu.pool.place", {"replica_id": "3", "reason": "x"}):
                pass
        finally:
            configure_tracing(False)
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["kairyu.pool.place"]
        assert spans[0].attributes["replica_id"] == "3"

    async def test_disabled_is_noop_without_otel_import(self):
        from kairyu.telemetry import traced_span, tracing_enabled

        assert not tracing_enabled()
        with traced_span("anything") as span:
            assert span is None


async def test_kind_smoke_gates_default_and_gpu_chart_before_cluster_creation():
    lines = [
        line.strip()
        for line in Path("scripts/kind_smoke.sh").read_text(encoding="utf-8").splitlines()
    ]
    commands = [
        "helm lint deploy/helm/kairyu",
        (
            "helm lint deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml"
        ),
        "helm template kairyu deploy/helm/kairyu >/dev/null",
        (
            "helm template kairyu deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml >/dev/null"
        ),
    ]

    assert "set -euo pipefail" in lines
    for command in commands:
        assert lines.count(command) == 1
    command_positions = [lines.index(command) for command in commands]
    gate_call = lines.index("helm_gate")
    kind_create = lines.index('kind create cluster --name "$CLUSTER" --wait 120s')
    assert command_positions == sorted(command_positions)
    assert max(command_positions) < gate_call
    assert gate_call < kind_create
    assert 'if [[ "${1:-}" == "--helm-check" ]]; then' in lines[gate_call:kind_create]


async def test_helm_check_exits_after_four_helm_commands_without_cluster_side_effects(
    tmp_path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"

    helm = bin_dir / "helm"
    helm.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "helm $*" >>"$CALL_LOG"\n',
        encoding="utf-8",
    )
    helm.chmod(0o755)

    forbidden_command = (
        "#!/usr/bin/env bash\n"
        'name=${0##*/}\n'
        'printf \'%s\\n\' "$name $*" >>"$CALL_LOG"\n'
        "exit 97\n"
    )
    for name in ("kind", "docker", "kubectl", "curl"):
        command = bin_dir / name
        command.write_text(forbidden_command, encoding="utf-8")
        command.chmod(0o755)

    env = os.environ.copy()
    env["CALL_LOG"] = str(call_log)
    env["PATH"] = os.pathsep.join((str(bin_dir), env["PATH"]))
    result = subprocess.run(
        ["bash", "scripts/kind_smoke.sh", "--helm-check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "helm lint deploy/helm/kairyu",
        "helm lint deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml",
        "helm template kairyu deploy/helm/kairyu",
        (
            "helm template kairyu deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml"
        ),
    ]


async def test_ci_has_explicit_single_source_helm_schema_and_gpu_template_gate():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    helm_install = workflow.index("- uses: helm/kind-action@v1")
    named_gate = workflow.index("- name: Helm schema and GPU template gate")
    gate_call = workflow.index("run: bash scripts/kind_smoke.sh --helm-check")
    kind_smoke = workflow.index("- name: kind smoke (m10a D5)")

    assert helm_install < named_gate < gate_call < kind_smoke
    assert "helm lint" not in workflow
    assert "helm template" not in workflow


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_chart_renders():
    rendered = subprocess.run(
        ["helm", "template", "kairyu", "deploy/helm/kairyu"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "kind: Deployment" in rendered
    assert "path: /readyz" in rendered
    assert "mountPath: /etc/kairyu" in rendered  # the Dockerfile CMD path (A11)
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod_spec
    assert "runtimeClassName" not in pod_spec
    assert "tolerations" not in pod_spec
    assert "affinity" not in pod_spec
    container = pod_spec["containers"][0]
    assert all(mount["name"] != "model-storage" for mount in container["volumeMounts"])
    assert all(volume["name"] != "model-storage" for volume in pod_spec["volumes"])


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_chart_renders_placement_and_runtime_controls(tmp_path):
    node_selector = {
        "kairyu.ai/accelerator": "nvidia",
        "kubernetes.io/arch": "amd64",
    }
    tolerations = [
        {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        }
    ]
    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kairyu.ai/gpu-model",
                                "operator": "In",
                                "values": ["RTX-PRO-6000"],
                            }
                        ]
                    }
                ]
            }
        }
    }
    override = tmp_path / "placement.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "nodeSelector": node_selector,
                "runtimeClassName": "nvidia",
                "tolerations": tolerations,
                "affinity": affinity,
            }
        )
    )

    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(override),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]

    assert pod_spec["nodeSelector"] == node_selector
    assert pod_spec["runtimeClassName"] == "nvidia"
    assert pod_spec["tolerations"] == tolerations
    assert pod_spec["affinity"] == affinity


def test_helm_chart_config_is_a_valid_deployment_spec():
    """kind-smoke root cause (PR #16): the chart shipped 'models:' which is
    not a DeploymentSpec field — the pod crash-looped at validation. Pin the
    embedded config to the real schema, no helm binary needed."""
    from kairyu.deploy.spec import load_deployment_spec

    values = yaml.safe_load(open("deploy/helm/kairyu/values.yaml"))
    spec = load_deployment_spec(values["config"])
    assert spec.engines, "chart config must declare at least one engine"


def test_helm_gpu_values_define_real_engine_and_model_storage():
    from kairyu.deploy.spec import load_deployment_spec

    chart_dir = Path("deploy/helm/kairyu")
    defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
    gpu_values = yaml.safe_load((chart_dir / "values-gpu.yaml").read_text())

    assert defaults["modelStorage"] == {
        "enabled": False,
        "pvcName": "",
        "hostPath": "",
        "mountPath": "/models",
    }
    assert gpu_values["modelStorage"] == {
        "enabled": True,
        "pvcName": "",
        "hostPath": "/models",
        "mountPath": "/models",
    }

    spec = load_deployment_spec(gpu_values["config"])
    engine = spec.engines["default"]
    assert engine.backend == "kairyu"
    assert engine.backend != "mock"
    model_path = PurePosixPath(engine.options["model_path"])
    assert model_path.is_relative_to(PurePosixPath(gpu_values["modelStorage"]["mountPath"]))

    template = (chart_dir / "templates/deployment.yaml").read_text()
    assert template.count("{{- if .Values.modelStorage.enabled }}") == 2
    assert ".Values.modelStorage.pvcName" in template
    assert ".Values.modelStorage.hostPath" in template
    assert ".Values.modelStorage.mountPath" in template

    schema = json.loads((chart_dir / "values.schema.json").read_text())
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(defaults)
    assert set(schema["required"]) == set(defaults)
    assert schema["properties"]["replicaCount"]["minimum"] == 1
    gpu_schema = schema["definitions"]["resourceList"]["properties"]["nvidia.com/gpu"]
    assert gpu_schema == {"type": "integer", "minimum": 1}
    storage_schema = schema["properties"]["modelStorage"]
    assert storage_schema["additionalProperties"] is False
    assert set(storage_schema["required"]) == {
        "enabled",
        "pvcName",
        "hostPath",
        "mountPath",
    }
    enabled_rule = storage_schema["allOf"][0]
    assert enabled_rule["if"]["properties"]["enabled"]["const"] is True
    assert len(enabled_rule["then"]["oneOf"]) == 2
    assert storage_schema["properties"]["mountPath"]["pattern"] == "^/"
    assert storage_schema["properties"]["hostPath"]["anyOf"] == [
        {"const": ""},
        {"pattern": "^/"},
    ]


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_gpu_values_render_real_engine_and_model_storage():
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    deployment = next(document for document in documents if document.get("kind") == "Deployment")
    configmap = next(document for document in documents if document.get("kind") == "ConfigMap")

    pod_spec = deployment["spec"]["template"]["spec"]
    assert pod_spec["runtimeClassName"] == "nvidia"
    assert pod_spec["nodeSelector"] == {"kairyu.dev/gpu-profile": "pcie-gddr"}

    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    model_mount = next(
        mount for mount in container["volumeMounts"] if mount["mountPath"] == "/models"
    )
    assert model_mount["readOnly"] is True
    model_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == model_mount["name"]
    )
    assert model_volume["hostPath"]["path"] == "/models"
    assert "persistentVolumeClaim" not in model_volume

    config = yaml.safe_load(configmap["data"]["config.yaml"])
    engine = config["engines"]["default"]
    assert engine["backend"] == "kairyu"
    assert engine["backend"] != "mock"
    model_path = PurePosixPath(engine["options"]["model_path"])
    assert model_path.is_relative_to(PurePosixPath(model_mount["mountPath"]))


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_model_storage_can_render_an_existing_pvc(tmp_path):
    pvc_values = tmp_path / "pvc-model-storage.yaml"
    pvc_values.write_text(
        yaml.safe_dump(
            {
                "modelStorage": {
                    "pvcName": "kairyu-models",
                    "hostPath": "",
                }
            }
        )
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
            "-f",
            str(pvc_values),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]
    model_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "model-storage"
    )
    assert model_volume["persistentVolumeClaim"]["claimName"] == "kairyu-models"
    assert "hostPath" not in model_volume
    model_mount = next(
        mount
        for mount in pod_spec["containers"][0]["volumeMounts"]
        if mount["name"] == "model-storage"
    )
    assert model_mount == {
        "name": "model-storage",
        "mountPath": "/models",
        "readOnly": True,
    }


@pytest.mark.parametrize(
    "model_storage",
    [
        {
            "enabled": True,
            "pvcName": "",
            "hostPath": "",
            "mountPath": "/models",
        },
        {
            "enabled": True,
            "pvcName": "kairyu-models",
            "hostPath": "/models",
            "mountPath": "/models",
        },
    ],
    ids=["without-source", "pvc-and-host-path"],
)
@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_schema_rejects_invalid_model_storage(tmp_path, model_storage):
    invalid_values = tmp_path / "invalid-model-storage.yaml"
    invalid_values.write_text(yaml.safe_dump({"modelStorage": model_storage}))
    result = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
            "-f",
            str(invalid_values),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
