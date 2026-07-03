"""`kairyu serve` CLI (design m7 D3)."""

import pytest

from kairyu.entrypoints import cli

DEPLOY_YAML = """
server:
  host: 127.0.0.1
  port: 8123
engines:
  m: { backend: mock }
"""


@pytest.fixture()
def config(tmp_path):
    path = tmp_path / "deploy.yaml"
    path.write_text(DEPLOY_YAML, encoding="utf-8")
    return path


def test_serve_runs_uvicorn_with_spec_address(monkeypatch, config):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    cli.main(["serve", str(config)])
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["app"].title == "kairyu"


def test_serve_flags_override_spec(monkeypatch, config):
    captured = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: captured.update(kw))
    cli.main(["serve", str(config), "--host", "0.0.0.0", "--port", "9000"])
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
