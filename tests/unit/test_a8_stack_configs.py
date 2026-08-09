"""Static G2 A8 stack contracts; these checks require neither Docker nor GPUs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "bench" / "deploy" / "qwen3-32b-multi-gpu"
COMPOSE_PATH = EXAMPLE_DIR / "a8-compose.yaml"
REPLICA_PATH = EXAMPLE_DIR / "a8-replica.yaml"
GATEWAY_PATH = EXAMPLE_DIR / "a8-gateway.yaml"
STACK_PATH = EXAMPLE_DIR / "a8-stack.sh"
IMAGE_EXPR = (
    "${KAIRYU_A8_IMAGE:?set KAIRYU_A8_IMAGE to the local image under test}"
)
IMAGE_ID_EXPR = (
    "${KAIRYU_A8_IMAGE_ID:?set the immutable sha256 image ID}"
)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _device_ids(service: dict) -> list[str]:
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert len(devices) == 1
    return devices[0]["device_ids"]


def _expand_cpuset(value: str) -> set[int]:
    cpus: set[int] = set()
    for field in value.split(","):
        start, separator, end = field.partition("-")
        if separator:
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(start))
    return cpus


def _write_fake_docker(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$KAIRYU_A8_RUN_DIR" > "$A8_RUN_DIR_LOG"\n'
        'printf "%s\\n" "$@" > "$A8_DOCKER_ARGS_LOG"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _stack_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_docker(fake_bin / "docker")
    args_log = tmp_path / "docker-args.log"
    run_dir_log = tmp_path / "run-dir.log"
    return (
        {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "KAIRYU_A8_IMAGE": "kairyu:a8-under-test",
            "KAIRYU_A8_IMAGE_ID": f"sha256:{'a' * 64}",
            "A8_DOCKER_ARGS_LOG": str(args_log),
            "A8_RUN_DIR_LOG": str(run_dir_log),
        },
        args_log,
        run_dir_log,
    )


def test_a8_replica_is_the_validated_qwen_tp4_contract() -> None:
    spec = load_deployment_spec(REPLICA_PATH)

    assert list(spec.engines) == ["qwen3-32b"]
    assert spec.pools == {}
    assert spec.server.host == "0.0.0.0"
    assert spec.server.port == 8000
    assert spec.server.api_keys_env is None
    assert spec.server.max_concurrency == 128
    assert spec.server.access_log is False
    engine = spec.engines["qwen3-32b"]
    assert engine.backend == "kairyu"
    assert engine.options == {
        "model_path": "/models/qwen3-32b",
        "tensor_parallel_size": 4,
        "num_pages": 8193,
        "max_model_len": 8192,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 16,
        "pipeline_depth": 5,
        "priority_age_s": 60.0,
        "decode_mode": "cuda_graph",
        "cuda_graph_max_batch": 16,
        "cuda_graph_max_pages": 512,
        "cuda_graph_warmup_iters": 3,
    }


def test_a8_gateway_is_a_two_replica_static_pool_with_bounded_evidence() -> None:
    spec = load_deployment_spec(GATEWAY_PATH)

    assert spec.engines == {}
    assert list(spec.pools) == ["qwen3-32b"]
    assert spec.server.max_concurrency == 256
    assert spec.server.access_log is True
    pool = spec.pools["qwen3-32b"]
    assert pool.discovery is None
    assert pool.unhealthy_after == 2
    assert pool.probe_interval_s == 1.0
    assert pool.queue_depth_threshold == 256
    assert pool.placement_log_path == "/evidence/placements.jsonl"
    assert [replica.health_url for replica in pool.replicas] == [
        "http://replica0:8000/readyz",
        "http://replica1:8000/readyz",
    ]
    assert [replica.options for replica in pool.replicas] == [
        {
            "base_url": "http://replica0:8000/v1",
            "model": "qwen3-32b",
            "api_key_env": None,
            "upstream": "kairyu",
        },
        {
            "base_url": "http://replica1:8000/v1",
            "model": "qwen3-32b",
            "api_key_env": None,
            "upstream": "kairyu",
        },
    ]
    limits = spec.tenants.limits["default"]
    assert limits.request_burst == 4096
    assert limits.token_burst == 50_000_000
    assert limits.max_in_flight == 256


def test_a8_compose_pins_gpu_cpu_ports_health_and_provenance() -> None:
    compose = _load_yaml(COMPOSE_PATH)
    assert compose["name"] == "kairyu-qwen3-32b-a8"
    services = compose["services"]
    assert set(services) == {"replica0", "replica1", "gateway"}

    for name, service in services.items():
        assert service["image"] == IMAGE_EXPR
        assert service["pull_policy"] == "never"
        assert "build" not in service
        assert service["labels"]["com.kairyu.a8.role"] == name
        assert service["labels"]["com.kairyu.a8.image-id"] == IMAGE_ID_EXPR
        health = service["healthcheck"]
        assert health["test"] == [
            "CMD",
            "curl",
            "-fsS",
            "http://127.0.0.1:8000/readyz",
        ]

    replica0 = services["replica0"]
    replica1 = services["replica1"]
    gateway = services["gateway"]
    assert _device_ids(replica0) == ["0", "1", "2", "3"]
    assert _device_ids(replica1) == ["4", "5", "6", "7"]
    assert "deploy" not in gateway
    assert replica0["ports"] == ["127.0.0.1:8200:8000"]
    assert replica1["ports"] == ["127.0.0.1:8201:8000"]
    assert gateway["ports"] == ["127.0.0.1:8202:8000"]
    assert gateway["depends_on"] == {
        "replica0": {"condition": "service_healthy"},
        "replica1": {"condition": "service_healthy"},
    }

    replica0_cpus = _expand_cpuset(replica0["cpuset"])
    replica1_cpus = _expand_cpuset(replica1["cpuset"])
    gateway_cpus = _expand_cpuset(gateway["cpuset"])
    assert not replica0_cpus & replica1_cpus
    assert not replica0_cpus & gateway_cpus
    assert not replica1_cpus & gateway_cpus
    assert replica0_cpus | replica1_cpus | gateway_cpus == set(range(128))
    assert gateway_cpus == {15, 31, 47, 63, 79, 95, 111, 127}

    for replica in (replica0, replica1):
        assert "./a8-replica.yaml:/etc/kairyu/config.yaml:ro" in replica["volumes"]
        assert "../../kairyu:/app/kairyu:ro" in replica["volumes"]
        assert "qwen3-32b:/models:ro" in replica["volumes"]
    assert gateway["volumes"] == [
        "./a8-gateway.yaml:/etc/kairyu/config.yaml:ro",
        "../../kairyu:/app/kairyu:ro",
        {
            "type": "bind",
            "source": (
                "${KAIRYU_A8_RUN_DIR:?"
                "run a8-stack.sh with a host run directory}"
            ),
            "target": "/evidence",
        },
    ]
    assert compose["volumes"]["qwen3-32b"] == {
        "external": True,
        "name": "kairyu-qwen3-32b_qwen3-32b",
    }


@pytest.mark.parametrize(
    ("overrides", "args", "message"),
    [
        ({"KAIRYU_A8_IMAGE": ""}, ("run", "config"), "must be nonblank"),
        ({"KAIRYU_A8_IMAGE": "   "}, ("run", "config"), "must be nonblank"),
        (
            {"KAIRYU_A8_IMAGE": "bad image"},
            ("run", "config"),
            "without whitespace",
        ),
        ({"KAIRYU_A8_IMAGE_ID": ""}, ("run", "config"), "sha256"),
        (
            {"KAIRYU_A8_IMAGE_ID": f"sha256:{'A' * 64}"},
            ("run", "config"),
            "lowercase hex",
        ),
        (
            {"KAIRYU_A8_IMAGE_ID": f"sha256:{'a' * 63}"},
            ("run", "config"),
            "64 lowercase hex",
        ),
    ],
)
def test_a8_stack_rejects_invalid_image_provenance(
    tmp_path: Path,
    overrides: dict[str, str],
    args: tuple[str, ...],
    message: str,
) -> None:
    env, _args_log, _run_dir_log = _stack_env(tmp_path)
    env.update(overrides)

    result = subprocess.run(
        [str(STACK_PATH), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_a8_stack_scopes_compose_and_preserves_caller_run_dir(
    tmp_path: Path,
) -> None:
    env, args_log, run_dir_log = _stack_env(tmp_path)
    run_dir = tmp_path / "evidence dir"
    run_dir.mkdir()
    sentinel = run_dir / "keep-me"
    sentinel.write_text("retained", encoding="utf-8")

    result = subprocess.run(
        [
            str(STACK_PATH),
            str(run_dir),
            "up",
            "-d",
            "--force-recreate",
            "--wait",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "retained"
    assert run_dir_log.read_text(encoding="utf-8").splitlines() == [
        str(run_dir.resolve())
    ]
    assert args_log.read_text(encoding="utf-8").splitlines() == [
        "compose",
        "--project-name",
        "kairyu-qwen3-32b-a8",
        "-f",
        str(COMPOSE_PATH),
        "up",
        "-d",
        "--force-recreate",
        "--wait",
    ]


def test_a8_stack_rejects_up_without_forced_recreation(
    tmp_path: Path,
) -> None:
    env, args_log, _run_dir_log = _stack_env(tmp_path)
    run_dir = tmp_path / "evidence"

    result = subprocess.run(
        [str(STACK_PATH), str(run_dir), "up", "-d", "--wait"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires --force-recreate" in result.stderr
    assert not args_log.exists()


def test_a8_stack_rejects_unsafe_placement_log_target(tmp_path: Path) -> None:
    env, args_log, _run_dir_log = _stack_env(tmp_path)
    run_dir = tmp_path / "evidence"
    run_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    (run_dir / "placements.jsonl").symlink_to(outside)

    result = subprocess.run(
        [str(STACK_PATH), str(run_dir), "config"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "absent or an existing regular file" in result.stderr
    assert not args_log.exists()
    assert not outside.exists()


@pytest.mark.parametrize(
    "scope_override",
    [
        "-p",
        "-pother",
        "--project-name=other",
        "-fother.yaml",
        "--file=other.yaml",
        "--project-directory=/tmp",
        "--env-file=other.env",
    ],
)
def test_a8_stack_rejects_compose_scope_overrides(
    tmp_path: Path,
    scope_override: str,
) -> None:
    env, args_log, _run_dir_log = _stack_env(tmp_path)
    run_dir = tmp_path / "evidence"

    result = subprocess.run(
        [str(STACK_PATH), str(run_dir), scope_override, "down"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must not override the A8 project or file scope" in result.stderr
    assert not args_log.exists()
    assert not run_dir.exists()
