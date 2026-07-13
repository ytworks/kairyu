"""Checked-in Compose files must resolve their deployment contracts locally."""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from kairyu.deploy.builder import build_app_from_config
from kairyu.deploy.spec import load_deployment_spec

COMPOSE_DIR = Path("deploy/compose")
COMPOSE_FILES = sorted(COMPOSE_DIR.glob("docker-compose*.yaml"))
WEBUI_COMPOSE = COMPOSE_DIR / "docker-compose.webui.yaml"
WEBUI_CONFIG = COMPOSE_DIR / "config.yaml"
CONTAINER_CONFIG = "/etc/kairyu/config.yaml"
VALIDATOR = Path("scripts/validate_compose_binds.py").resolve()
COMPOSE_SMOKE = Path("scripts/compose_smoke.sh")
CI_WORKFLOW = Path(".github/workflows/ci.yml")


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _run_validator(*compose_files: Path, cwd: Path):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *(str(path) for path in compose_files)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _literal_relative_binds(compose_file: Path):
    services = _load_yaml(compose_file).get("services")
    assert isinstance(services, dict), f"{compose_file} must declare services"
    for service_name, service in services.items():
        assert isinstance(service, dict)
        for volume in service.get("volumes", []):
            if isinstance(volume, str):
                source, separator, _target = volume.partition(":")
                if not separator or not source.startswith("."):
                    continue
            elif isinstance(volume, dict) and volume.get("type") == "bind":
                source = volume.get("source")
                if not isinstance(source, str) or Path(source).is_absolute():
                    continue
            else:
                continue
            if "$" not in source:
                yield service_name, source


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_checked_in_literal_relative_bind_sources_exist(compose_file):
    binds = list(_literal_relative_binds(compose_file))
    assert binds, f"{compose_file} has no literal relative bind sources"
    for service_name, source in binds:
        resolved = (compose_file.parent / source).resolve()
        assert resolved.exists(), (
            f"{compose_file}: service {service_name!r} has missing bind source "
            f"{source!r} ({resolved})"
        )


def test_webui_compose_pins_internal_v1_and_file_relative_config_mount():
    services = _load_yaml(WEBUI_COMPOSE)["services"]
    kairyu = services["kairyu"]
    webui = services["webui"]

    config_mounts = [
        volume
        for volume in kairyu["volumes"]
        if isinstance(volume, str) and f":{CONTAINER_CONFIG}" in volume
    ]
    assert config_mounts == [f"./config.yaml:{CONTAINER_CONFIG}:ro"]
    source = config_mounts[0].partition(":")[0]
    assert (WEBUI_COMPOSE.parent / source).resolve() == WEBUI_CONFIG.resolve()

    environment = webui["environment"]
    assert environment["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert all("MODEL" not in name for name in environment)


async def test_webui_mounted_config_builds_ready_app_and_discovers_default():
    assert WEBUI_CONFIG.is_file(), "WebUI DeploymentSpec is missing"
    spec = load_deployment_spec(WEBUI_CONFIG)
    assert list(spec.engines) == ["default"]
    assert spec.engines["default"].backend == "mock"
    assert spec.pools == {}
    assert spec.server.host == "0.0.0.0"
    assert spec.server.port == 8000
    assert spec.server.api_keys_env is None

    app = build_app_from_config(WEBUI_CONFIG)
    async with _client(app) as client:
        assert (await client.get("/readyz")).status_code == 200
        models = await client.get(
            "/v1/models", headers={"Authorization": "Bearer sk-local"}
        )

    assert models.status_code == 200
    assert [entry["id"] for entry in models.json()["data"]] == ["default"]


def test_validator_accepts_checked_in_inventory_from_any_cwd(tmp_path):
    result = _run_validator(
        *(path.resolve() for path in COMPOSE_FILES), cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr


def test_validator_accepts_valid_short_and_long_bind_syntax(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("engines:\n  default:\n    backend: mock\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "short": {
                        "volumes": [
                            "./config.yaml:/etc/kairyu/config.yaml:ro"
                        ]
                    },
                    "long": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "./data",
                                "target": "/data",
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 0, result.stderr


def test_validator_reports_every_missing_short_and_long_bind(tmp_path):
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "short-service": {"volumes": ["./missing-short:/short"]},
                    "long-service": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "./missing-long",
                                "target": "/long",
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 1
    for marker in (
        str(compose_file),
        "short-service",
        "./missing-short",
        "long-service",
        "./missing-long",
    ):
        assert marker in result.stderr


def test_validator_ignores_named_substituted_and_absolute_operator_sources(tmp_path):
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "operator": {
                        "volumes": [
                            "cache:/cache",
                            "${HOST_DIR:-/operator/data}:/substituted",
                            "/operator/data:/absolute",
                            {
                                "type": "bind",
                                "source": "${LONG_DIR:-/operator/data}",
                                "target": "/long-substituted",
                            },
                            {
                                "type": "bind",
                                "source": "/operator/data",
                                "target": "/long-absolute",
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 0, result.stderr


def test_validator_rejects_invalid_mounted_deployment_spec(tmp_path):
    (tmp_path / "invalid.yaml").write_text("engines: {}\n", encoding="utf-8")
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "kairyu": {
                        "volumes": [
                            "./invalid.yaml:/etc/kairyu/config.yaml:ro"
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_validator(compose_file, cwd=tmp_path.parent)
    assert result.returncode == 1
    assert str(compose_file) in result.stderr
    assert "kairyu" in result.stderr
    assert "./invalid.yaml" in result.stderr
    assert "invalid DeploymentSpec" in result.stderr


def test_validator_rejects_untracked_source_for_tracked_compose(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    compose_file = repo / "compose.yaml"
    compose_file.write_text(
        "services:\n"
        "  kairyu:\n"
        "    volumes:\n"
        "      - ./config.yaml:/etc/kairyu/config.yaml:ro\n",
        encoding="utf-8",
    )
    config = repo / "config.yaml"
    config.write_text("engines:\n  default:\n    backend: mock\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "compose.yaml"], check=True)

    untracked = _run_validator(compose_file, cwd=tmp_path)
    assert untracked.returncode == 1
    assert "./config.yaml" in untracked.stderr
    assert "not tracked" in untracked.stderr

    subprocess.run(["git", "-C", str(repo), "add", "config.yaml"], check=True)
    tracked = _run_validator(compose_file, cwd=tmp_path)
    assert tracked.returncode == 0, tracked.stderr


def test_validator_accepts_relative_bind_that_resolves_to_tracked_repo_root(tmp_path):
    repo = tmp_path / "repo"
    compose_dir = repo / "deploy" / "compose"
    compose_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    compose_file = compose_dir / "docker-compose.yaml"
    compose_file.write_text(
        "services:\n"
        "  workspace:\n"
        "    volumes:\n"
        "      - ../..:/workspace:ro\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "deploy/compose/docker-compose.yaml"],
        check=True,
    )

    result = _run_validator(compose_file, cwd=tmp_path)

    assert result.returncode == 0, result.stderr


def test_smoke_validation_failure_never_invokes_docker_cleanup(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    docker_marker = tmp_path / "docker-called"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/bin/sh\ntouch "$DOCKER_MARKER"\n', encoding="utf-8"
    )
    fake_docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_MARKER": str(docker_marker),
    }

    result = subprocess.run(
        ["/bin/bash", str(COMPOSE_SMOKE.resolve())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not docker_marker.exists()


def test_ci_and_default_smoke_fail_fast_through_validator():
    smoke = COMPOSE_SMOKE.read_text(encoding="utf-8")
    assert smoke.index("validate_compose_binds.py") < smoke.index('echo "== up =="')
    assert 'uv run --project "$REPO_ROOT" --no-dev python' in smoke

    compose_steps = _load_yaml(CI_WORKFLOW)["jobs"]["compose-smoke"]["steps"]
    validation_index = next(
        index
        for index, step in enumerate(compose_steps)
        if step.get("name") == "Validate Compose bind sources"
    )
    smoke_index = next(
        index
        for index, step in enumerate(compose_steps)
        if step.get("name") == "Compose smoke drill"
    )
    assert validation_index < smoke_index
    assert any(
        step.get("uses") == "astral-sh/setup-uv@v5"
        for step in compose_steps[:validation_index]
    )
    validation_command = compose_steps[validation_index]["run"]
    assert validation_command.startswith(
        "uv run --no-dev python scripts/validate_compose_binds.py"
    )
    assert "deploy/compose/docker-compose*.yaml" in validation_command
