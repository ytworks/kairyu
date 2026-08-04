"""Validation-only CLI coverage for issue #230."""

from __future__ import annotations

import pytest
import uvicorn

from kairyu.deploy import builder as builder_module
from kairyu.dsl import loader as dsl_loader
from kairyu.engine import registry as registry_module
from kairyu.entrypoints import cli


def _write_valid_linked_deployment(tmp_path):
    orchestrator = tmp_path / "orchestrator.yaml"
    orchestrator.write_text(
        """
workers:
  - name: worker
    backend: mock
roles:
  - name: answer
    worker: worker
    prompt: "Answer: {query}"
""",
        encoding="utf-8",
    )
    (tmp_path / "chat.jinja").write_text(
        "{{ messages[-1].content }}",
        encoding="utf-8",
    )
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: { backend: mock }
orchestrators:
  answer: { spec: orchestrator.yaml }
chat_templates:
  local: chat.jinja
legacy_chat_models: [answer]
""",
        encoding="utf-8",
    )
    return deployment


def test_validate_accepts_linked_graph_without_startup_side_effects(
    tmp_path,
    monkeypatch,
    capsys,
):
    deployment = _write_valid_linked_deployment(tmp_path)

    def unexpected_startup(*_args, **_kwargs):
        raise AssertionError("validation must not construct or start runtime resources")

    monkeypatch.setattr(builder_module, "build_app_from_config", unexpected_startup)
    monkeypatch.setattr(dsl_loader, "build_orchestrator", unexpected_startup)
    monkeypatch.setattr(registry_module, "create_backend", unexpected_startup)
    monkeypatch.setattr(uvicorn, "run", unexpected_startup)

    cli.main(["validate", str(deployment)])

    captured = capsys.readouterr()
    assert captured.out.startswith("VALID ")
    assert captured.err == ""


def test_validate_collects_multiple_missing_references(tmp_path, capsys):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  local: { backend: mock }
orchestrators:
  first: { spec: missing-first.yaml }
  second: { spec: missing-second.yaml }
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["validate", str(deployment)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "missing-first.yaml" in captured.out
    assert "missing-second.yaml" in captured.out
    assert captured.out.count("ERROR [filesystem]") >= 2
    assert captured.err == ""


def test_validate_reports_incompatible_capability(tmp_path, capsys):
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text(
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://replica.test/v1
      model: model
      upstream: openai
      capabilities:
        allow_extra_args: [model]
        1: invalid
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["validate", str(deployment)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "engines.remote" in captured.out
    assert "(schema.invalid_backend_options)" in captured.out
    assert captured.err == ""


def test_validate_reports_invalid_root_path_without_traceback(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["validate", "bad\0path.yaml"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "\0" not in captured.out
    assert r"bad\x00path.yaml" in captured.out
    assert captured.err == ""
