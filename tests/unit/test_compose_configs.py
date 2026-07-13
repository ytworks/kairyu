"""Checked-in Compose files must resolve their deployment contracts locally."""

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


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


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
