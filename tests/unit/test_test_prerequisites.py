from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import test_prerequisites


def _args(**overrides):
    values = {
        "min_gpus": 0,
        "require_nccl": False,
        "require_peer_access": False,
        "min_compute_capability": None,
        "require_module": [],
        "require_executable": [],
        "require_path": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_missing_prerequisite_is_not_run_and_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        test_prerequisites.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    result = test_prerequisites.evaluate(
        _args(require_module=["missing-package"])
    )
    exit_code = test_prerequisites.main(
        ["--require-module", "missing-package"]
    )

    assert result["status"] == "not-run"
    assert result["missing"] == ["python-module:missing-package:ImportError"]
    assert exit_code == test_prerequisites.NOT_RUN_EXIT
    assert '"status": "not-run"' in capsys.readouterr().out


def test_available_cuda_prerequisites_are_recorded(monkeypatch, tmp_path) -> None:
    required_path = tmp_path / "model"
    required_path.mkdir()
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(name=f"gpu-{index}"),
        get_device_capability=lambda index: (12, 0),
        can_device_access_peer=lambda source, target: (source, target) == (0, 1),
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        distributed=SimpleNamespace(is_nccl_available=lambda: True),
    )
    monkeypatch.setattr(
        test_prerequisites.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else SimpleNamespace(),
    )
    monkeypatch.setattr(
        test_prerequisites.shutil,
        "which",
        lambda name: f"/tools/{name}",
    )

    result = test_prerequisites.evaluate(
        _args(
            min_gpus=2,
            require_nccl=True,
            require_peer_access=True,
            min_compute_capability=(10, 0),
            require_module=["flashinfer"],
            require_executable=["ninja"],
            require_path=[required_path],
        )
    )

    assert result["status"] == "available"
    assert result["missing"] == []
    assert result["visible_gpu_count"] == 2
    assert result["visible_gpu_names"] == ["gpu-0", "gpu-1"]
    assert result["visible_compute_capabilities"] == [(12, 0), (12, 0)]
    assert result["executables"] == {"ninja": "/tools/ninja"}
    assert result["paths"] == {str(required_path): str(required_path.resolve())}


def test_compute_capability_below_requirement_is_not_run(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(name=f"gpu-{index}"),
        get_device_capability=lambda index: (9, 0) if index == 0 else (10, 0),
        can_device_access_peer=lambda source, target: True,
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        distributed=SimpleNamespace(is_nccl_available=lambda: True),
    )
    monkeypatch.setattr(
        test_prerequisites.importlib,
        "import_module",
        lambda name: fake_torch,
    )

    result = test_prerequisites.evaluate(
        _args(min_gpus=2, min_compute_capability=(10, 0))
    )

    assert result["status"] == "not-run"
    assert result["missing"] == [
        "cuda-compute-capability:10.0-required:9.0,10.0-visible"
    ]


def _load_pytest_controls():
    path = Path("tests/conftest.py")
    spec = importlib.util.spec_from_file_location("kairyu_test_controls", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fail_on_skip_changes_only_an_otherwise_successful_session() -> None:
    controls = _load_pytest_controls()

    class Reporter:
        message = ""

        def write_sep(self, _separator, message):
            self.message = message

    reporter = Reporter()
    config = SimpleNamespace(
        getoption=lambda name: name == "fail_on_skip",
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: reporter if name == "terminalreporter" else None
        ),
    )
    session = SimpleNamespace(config=config, exitstatus=pytest.ExitCode.OK)

    controls.pytest_configure(config)
    controls.pytest_runtest_logreport(SimpleNamespace(skipped=True))
    controls.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert "1 skipped selected test" in reporter.message


def test_fail_on_skip_leaves_a_session_without_skips_unchanged() -> None:
    controls = _load_pytest_controls()
    reporter = SimpleNamespace(stats={})
    config = SimpleNamespace(
        getoption=lambda name: name == "fail_on_skip",
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: reporter if name == "terminalreporter" else None
        ),
    )
    session = SimpleNamespace(config=config, exitstatus=pytest.ExitCode.OK)

    controls.pytest_configure(config)
    controls.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.OK


def test_fail_on_skip_rejects_a_real_skipped_pytest_session(tmp_path) -> None:
    shutil.copyfile("tests/conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_selected.py").write_text(
        "import pytest\n"
        "def test_selected():\n"
        "    pytest.skip('missing declared capability')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--fail-on-skip",
            "-q",
            "--no-cov",
            "test_selected.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED
    assert "1 skipped" in result.stdout
    assert "--fail-on-skip rejected 1 skipped selected test" in result.stdout


def test_fail_on_skip_does_not_depend_on_the_terminal_reporter(tmp_path) -> None:
    shutil.copyfile("tests/conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_selected.py").write_text(
        "import pytest\n"
        "def test_selected():\n"
        "    pytest.skip('missing declared capability')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:terminal",
            "--fail-on-skip",
            "--no-cov",
            "test_selected.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED
