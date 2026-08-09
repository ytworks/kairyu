"""Static G2 A9 TP8 stack contracts; these tests need no Docker or GPU."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "bench" / "deploy" / "qwen3-32b-multi-gpu"
COMPOSE_PATH = EXAMPLE_DIR / "a9-tp8-compose.yaml"
REPLICA_PATH = EXAMPLE_DIR / "a9-tp8-replica.yaml"
GATEWAY_PATH = EXAMPLE_DIR / "a9-tp8-gateway.yaml"
STACK_PATH = EXAMPLE_DIR / "a9-tp8-stack.sh"
A8_REPLICA_PATH = EXAMPLE_DIR / "a8-replica.yaml"
IMAGE_EXPR = (
    "${KAIRYU_A9_IMAGE:?set KAIRYU_A9_IMAGE to the immutable local image under test}"
)
IMAGE_ID_EXPR = (
    "${KAIRYU_A9_IMAGE_ID:?set the immutable sha256 image ID}"
)
RUNTIME_EXPR = (
    "${KAIRYU_A9_RUNTIME_DIR:?"
    "run a9-tp8-stack.sh to materialize the pinned A8 source}"
)
A8_SOURCE_COMMIT = "86d49223ffcdba6052428474bf0d9094c6791fed"
A8_REPLICA0_CPUS = set(range(0, 15)) | set(range(16, 31)) | set(
    range(64, 79)
) | set(range(80, 95))
A8_REPLICA1_CPUS = set(range(32, 47)) | set(range(48, 63)) | set(
    range(96, 111)
) | set(range(112, 127))
GATEWAY_CPUS = {15, 31, 47, 63, 79, 95, 111, 127}


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
        'printf "%s\\n" "$KAIRYU_A9_RUN_DIR" > "$A9_RUN_DIR_LOG"\n'
        'printf "%s\\n" "$KAIRYU_A9_RUNTIME_DIR" > "$A9_RUNTIME_DIR_LOG"\n'
        'printf "%s\\n" "$@" > "$A9_DOCKER_ARGS_LOG"\n',
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
            "KAIRYU_A9_IMAGE": "kairyu:a9-under-test",
            "KAIRYU_A9_IMAGE_ID": f"sha256:{'a' * 64}",
            "A9_DOCKER_ARGS_LOG": str(args_log),
            "A9_RUN_DIR_LOG": str(run_dir_log),
            "A9_RUNTIME_DIR_LOG": str(tmp_path / "runtime-dir.log"),
        },
        args_log,
        run_dir_log,
    )


def test_a9_tp8_replica_matches_a8_engine_contract_except_tp_degree() -> None:
    spec = load_deployment_spec(REPLICA_PATH)
    a8_spec = load_deployment_spec(A8_REPLICA_PATH)

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
        "tensor_parallel_size": 8,
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
    assert engine.options == {
        **a8_spec.engines["qwen3-32b"].options,
        "tensor_parallel_size": 8,
    }


def test_a9_gateway_is_one_replica_and_owns_the_formal_endpoint() -> None:
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
    assert len(pool.replicas) == 1
    assert pool.replicas[0].health_url == "http://tp8:8000/readyz"
    assert pool.replicas[0].options == {
        "base_url": "http://tp8:8000/v1",
        "model": "qwen3-32b",
        "api_key_env": None,
        "upstream": "kairyu",
    }
    limits = spec.tenants.limits["default"]
    assert limits.request_burst == 4096
    assert limits.token_burst == 50_000_000
    assert limits.max_in_flight == 256


def test_a9_compose_pins_topology_ports_and_provenance() -> None:
    compose = _load_yaml(COMPOSE_PATH)
    assert compose["name"] == "kairyu-qwen3-32b-a9-tp8"
    services = compose["services"]
    assert set(services) == {"tp8", "gateway"}

    for name, service in services.items():
        assert service["image"] == IMAGE_EXPR
        assert service["pull_policy"] == "never"
        assert "build" not in service
        assert service["labels"]["com.kairyu.a9.role"] == name
        assert service["labels"]["com.kairyu.a9.image-id"] == IMAGE_ID_EXPR
        health = service["healthcheck"]
        assert health["test"] == [
            "CMD",
            "curl",
            "-fsS",
            "http://127.0.0.1:8000/readyz",
        ]

    tp8 = services["tp8"]
    gateway = services["gateway"]
    assert _device_ids(tp8) == [str(index) for index in range(8)]
    assert "deploy" not in gateway
    assert tp8["ports"] == ["127.0.0.1:8300:8000"]
    assert gateway["ports"] == ["127.0.0.1:8301:8000"]
    assert gateway["depends_on"] == {"tp8": {"condition": "service_healthy"}}

    tp8_cpus = _expand_cpuset(tp8["cpuset"])
    gateway_cpus = _expand_cpuset(gateway["cpuset"])
    assert tp8_cpus == A8_REPLICA0_CPUS | A8_REPLICA1_CPUS
    assert gateway_cpus == GATEWAY_CPUS
    assert not tp8_cpus & gateway_cpus
    assert tp8_cpus | gateway_cpus == set(range(128))

    assert tp8["labels"]["com.kairyu.a9.host-gpu-ordinals"] == (
        "0,1,2,3,4,5,6,7"
    )
    assert gateway["labels"]["com.kairyu.a9.host-gpu-ordinals"] == ""
    assert tp8["volumes"] == [
        "./a9-tp8-replica.yaml:/etc/kairyu/config.yaml:ro",
        {
            "type": "bind",
            "source": RUNTIME_EXPR,
            "target": "/app/kairyu",
            "read_only": True,
        },
        "qwen3-32b:/models:ro",
    ]
    assert gateway["volumes"] == [
        "./a9-tp8-gateway.yaml:/etc/kairyu/config.yaml:ro",
        {
            "type": "bind",
            "source": RUNTIME_EXPR,
            "target": "/app/kairyu",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": (
                "${KAIRYU_A9_RUN_DIR:?"
                "run a9-tp8-stack.sh with a host run directory}"
            ),
            "target": "/evidence",
        },
    ]
    assert compose["volumes"]["qwen3-32b"] == {
        "external": True,
        "name": "kairyu-qwen3-32b_qwen3-32b",
    }
    assert all(
        any(
            isinstance(mount, dict)
            and mount.get("source") == RUNTIME_EXPR
            and mount.get("target") == "/app/kairyu"
            and mount.get("read_only") is True
            for mount in service.get("volumes", ())
        )
        for service in services.values()
    )


@pytest.mark.parametrize(
    ("overrides", "args", "message"),
    [
        ({"KAIRYU_A9_IMAGE": ""}, ("run", "config"), "must be nonblank"),
        ({"KAIRYU_A9_IMAGE": "   "}, ("run", "config"), "must be nonblank"),
        (
            {"KAIRYU_A9_IMAGE": "bad image"},
            ("run", "config"),
            "without whitespace",
        ),
        ({"KAIRYU_A9_IMAGE_ID": ""}, ("run", "config"), "sha256"),
        (
            {"KAIRYU_A9_IMAGE_ID": f"sha256:{'A' * 64}"},
            ("run", "config"),
            "lowercase hex",
        ),
        (
            {"KAIRYU_A9_IMAGE_ID": f"sha256:{'a' * 63}"},
            ("run", "config"),
            "64 lowercase hex",
        ),
    ],
)
def test_a9_stack_rejects_invalid_image_provenance(
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


def test_a9_stack_scopes_compose_and_preserves_caller_run_dir(
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
    assert (tmp_path / "runtime-dir.log").read_text(
        encoding="utf-8"
    ).splitlines() == [
        str(
            (
                run_dir.parent
                / f".{run_dir.name}.a8-source-{A8_SOURCE_COMMIT}"
                / "kairyu"
            ).resolve()
        )
    ]
    assert (
        run_dir.parent
        / f".{run_dir.name}.a8-source-{A8_SOURCE_COMMIT}"
        / "source.tar"
    ).is_file()
    assert args_log.read_text(encoding="utf-8").splitlines() == [
        "compose",
        "--project-name",
        "kairyu-qwen3-32b-a9-tp8",
        "-f",
        str(COMPOSE_PATH),
        "up",
        "-d",
        "--force-recreate",
        "--wait",
    ]


def test_a9_stack_rejects_up_without_forced_recreation(tmp_path: Path) -> None:
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


def test_a9_stack_rejects_unsafe_placement_log_target(tmp_path: Path) -> None:
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
def test_a9_stack_rejects_compose_scope_overrides(
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
    assert "must not override the A9 TP8 project or file scope" in result.stderr
    assert not args_log.exists()
    assert not run_dir.exists()
