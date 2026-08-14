"""Package-owned contracts shared by installed and repository benchmark CLIs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import evidence.reporting as reporting
from evals.targets import (
    parse_target_spec,
    resolve_api_key_env,
    target_api_key,
)
from evals.types import BenchConfig, BenchTarget
from evidence.reporting import (
    atomic_write_json,
    atomic_write_text,
    nearest_rank_percentile,
)


def test_target_parser_normalizes_url_and_records_only_env_name() -> None:
    target = parse_target_spec(
        "gateway = http://gateway.test:8000/ = model-a = BENCH_API_KEY"
    )

    assert target.name == "gateway"
    assert target.base_url == "http://gateway.test:8000/v1"
    assert target.model == "model-a"
    assert target.api_key_env == "BENCH_API_KEY"
    assert "secret" not in target.model_dump_json()


def test_direct_and_yaml_model_paths_share_target_semantics() -> None:
    target = BenchTarget(
        name="gateway",
        base_url=" http://gateway.test:8000/ ",
        model="model-a",
        api_key_env=" BENCH_API_KEY ",
    )

    assert target.base_url == "http://gateway.test:8000/v1"
    assert target.api_key_env == "BENCH_API_KEY"
    with pytest.raises(ValueError, match="environment-variable name"):
        BenchTarget(
            base_url="http://gateway.test",
            model="model-a",
            api_key_env="sk-literal-secret",
        )


@pytest.mark.parametrize(
    "spec",
    [
        "only-a-name",
        "=http://gateway.test=model",
        "name==model",
        "name=http://gateway.test=",
        "name=http://gateway.test=model=sk-literal-secret",
        "name=http://gateway.test=model=KEY=extra",
    ],
)
def test_target_parser_rejects_ambiguous_or_secret_like_specs(spec: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_target_spec(spec)
    assert "sk-literal-secret" not in str(exc_info.value)


def test_api_key_resolution_is_explicit_and_error_never_contains_value() -> None:
    assert resolve_api_key_env(None, environ={}) is None
    assert resolve_api_key_env("KEY", environ={"KEY": "private-value"}) == "private-value"
    assert resolve_api_key_env("KEY", environ={}) is None

    with pytest.raises(ValueError) as exc_info:
        resolve_api_key_env("KEY", environ={}, required=True)

    assert "KEY" in str(exc_info.value)
    assert "private-value" not in str(exc_info.value)


def test_configured_target_auth_fails_closed_and_redacts_model_errors() -> None:
    target = BenchTarget(
        base_url="http://gateway.test",
        model="model-a",
        api_key_env="MISSING_API_KEY",
    )
    with pytest.raises(ValueError, match="MISSING_API_KEY"):
        target_api_key(target, environ={})
    assert target_api_key(target, environ={}, required=False) is None

    secret = "sk-actual-secret"
    with pytest.raises(ValueError) as exc_info:
        BenchTarget(
            base_url="http://gateway.test",
            model="model-a",
            api_key_env=secret,
        )
    assert secret not in str(exc_info.value)

    with pytest.raises(ValueError) as nested_exc_info:
        BenchConfig.model_validate(
            {
                "suite": "core",
                "targets": [
                    {
                        "base_url": "http://gateway.test",
                        "model": "model-a",
                        "api_key_env": secret,
                    }
                ]
            }
        )
    assert secret not in str(nested_exc_info.value)


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [
        (range(1, 21), 0.95, 19),
        (range(1, 101), 0.99, 99),
        ([5, 1, 4, 2, 3], 0.5, 3),
        ([5, 1, 4, 2, 3], 1.0, 5),
    ],
)
def test_nearest_rank_percentile_has_one_explicit_definition(
    values,
    fraction: float,
    expected: int,
) -> None:
    assert nearest_rank_percentile(values, fraction) == expected


@pytest.mark.parametrize(("values", "fraction"), [([], 0.5), ([1], 0.0), ([1], 1.1)])
def test_nearest_rank_percentile_rejects_invalid_inputs(values, fraction) -> None:
    with pytest.raises(ValueError):
        nearest_rank_percentile(values, fraction)


def test_atomic_reporting_replaces_text_and_json_without_temp_files(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "nested" / "report.md"
    json_path = tmp_path / "nested" / "report.json"

    atomic_write_text(text_path, "first")
    atomic_write_text(text_path, "second")
    atomic_write_json(json_path, {"target": "模型", "score": 1})

    assert text_path.read_text(encoding="utf-8") == "second"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "target": "模型",
        "score": 1,
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_reporting_can_publish_without_replacing_existing_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    atomic_write_json(path, {"winner": 1}, overwrite=False, allow_nan=False)

    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"winner": 2}, overwrite=False, allow_nan=False)

    assert json.loads(path.read_text(encoding="utf-8")) == {"winner": 1}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_reporting_can_reject_non_strict_json_before_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"

    with pytest.raises(ValueError):
        atomic_write_json(path, {"value": float("nan")}, allow_nan=False)

    assert not path.exists()


def test_exclusive_reporting_rejects_a_swapped_source_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "report.json"
    outside = tmp_path / "outside.json"
    outside.write_text('{"attacker": true}', encoding="utf-8")
    real_link = os.link

    def swapped_link(source, target, *args, **kwargs):
        Path(source).unlink()
        Path(source).symlink_to(outside)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(reporting.os, "link", swapped_link)

    with pytest.raises(OSError):
        atomic_write_json(path, {"trusted": True}, overwrite=False, allow_nan=False)

    assert not path.exists()
    assert json.loads(outside.read_text(encoding="utf-8")) == {"attacker": True}


def test_exclusive_reporting_cleanup_failure_does_not_reclassify_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "report.json"
    real_unlink_if_identity = reporting._unlink_if_identity

    def fail_private_cleanup(candidate: Path, expected: tuple[int, int]) -> bool:
        if candidate != path and candidate.name.endswith(".tmp"):
            raise OSError("simulated private-temp cleanup failure")
        return real_unlink_if_identity(candidate, expected)

    monkeypatch.setattr(reporting, "_unlink_if_identity", fail_private_cleanup)

    assert atomic_write_json(
        path,
        {"published": True},
        overwrite=False,
        allow_nan=False,
    ) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"published": True}
