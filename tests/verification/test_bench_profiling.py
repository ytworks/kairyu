"""Shared benchmark profiling is lazy, bounded, and non-overwriting."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import verification.l1.performance.profiling as profiling


class _FakeNativeProfiler:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.exited += 1


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.probes = 0

    def is_available(self) -> bool:
        self.probes += 1
        return self.available


class _FakeProfilerModule:
    class ProfilerActivity:
        CPU = object()
        CUDA = object()

    def __init__(self, native: _FakeNativeProfiler) -> None:
        self.native = native
        self.calls: list[dict[str, object]] = []

    def profile(self, **kwargs):
        self.calls.append(kwargs)
        return self.native


def _fake_torch(*, cuda_available: bool = False):
    native = _FakeNativeProfiler()
    profiler_module = _FakeProfilerModule(native)
    torch = SimpleNamespace(
        profiler=profiler_module,
        cuda=_FakeCuda(cuda_available),
    )
    return torch, profiler_module, native


class _ExportingProfiler:
    def __init__(self, payload: bytes | None = None, *, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.paths: list[Path] = []

    def export_chrome_trace(self, path: str) -> None:
        destination = Path(path)
        self.paths.append(destination)
        if self.payload is not None:
            destination.write_bytes(self.payload)
        if self.error is not None:
            raise self.error


class _SparseOversizeProfiler:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def export_chrome_trace(self, path: str) -> None:
        destination = Path(path)
        self.paths.append(destination)
        with destination.open("wb") as output:
            output.truncate(64 * 1024 * 1024 + 1)


def _trace_payload() -> bytes:
    return json.dumps(
        {
            "traceEvents": [
                {
                    "name": "serving-client",
                    "ph": "X",
                    "ts": 1,
                    "dur": 2,
                    "pid": 1,
                    "tid": 1,
                }
            ],
            "displayTimeUnit": "ns",
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _assert_directory_empty(path: Path) -> None:
    assert path.is_dir()
    assert list(path.iterdir()) == []


def test_disabled_scope_does_not_load_torch(monkeypatch) -> None:
    def unexpected_load():
        raise AssertionError("disabled profiling imported torch")

    monkeypatch.setattr(profiling, "_load_torch", unexpected_load)

    with profiling.profile_scope(False) as native:
        assert native is None

    with profiling.record_function_scope(False, "client.measurement"):
        pass


def test_module_import_and_disabled_scope_work_when_torch_is_absent(monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_torch(name: str, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("torch blocked by optionality test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    monkeypatch.delitem(sys.modules, "verification.l1.performance.profiling", raising=False)

    imported = __import__("verification.l1.performance.profiling", fromlist=["profiling"])
    with imported.profile_scope(False) as native:
        assert native is None


def test_serving_help_is_available_when_torch_is_absent(monkeypatch) -> None:
    module_name = "serving_bench_torch_optional"
    path = Path(__file__).parents[2] / "verification/l1/performance/serving_bench.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, module_name, module)

    spec.loader.exec_module(module)

    assert "--profile" in module.build_parser().format_help()


def test_enabled_scope_maps_all_native_options_and_yields_native_identity(
    monkeypatch,
) -> None:
    torch, profiler_module, native = _fake_torch(cuda_available=True)
    monkeypatch.setattr(profiling, "_load_torch", lambda: torch)

    scope = profiling.profile_scope(
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    assert scope is native
    with scope as retained:
        assert retained is native
        assert native.entered == 1
        assert native.exited == 0

    assert native.exited == 1
    assert len(profiler_module.calls) == 1
    call = profiler_module.calls[0]
    assert tuple(call["activities"]) == (
        profiler_module.ProfilerActivity.CPU,
        profiler_module.ProfilerActivity.CUDA,
    )
    assert call["acc_events"] is True
    assert call["record_shapes"] is True
    assert call["profile_memory"] is True
    assert call["with_stack"] is True
    assert torch.cuda.probes == 1


def test_cpu_scope_does_not_probe_cuda(monkeypatch) -> None:
    torch, profiler_module, _native = _fake_torch(cuda_available=False)
    monkeypatch.setattr(profiling, "_load_torch", lambda: torch)

    with profiling.profile_scope() as retained:
        assert retained is not None

    assert tuple(profiler_module.calls[0]["activities"]) == (
        profiler_module.ProfilerActivity.CPU,
    )
    assert torch.cuda.probes == 0


def test_cuda_scope_fails_instead_of_silently_falling_back(monkeypatch) -> None:
    torch, profiler_module, native = _fake_torch(cuda_available=False)
    monkeypatch.setattr(profiling, "_load_torch", lambda: torch)

    with pytest.raises(RuntimeError, match="CUDA.*unavailable|unavailable.*CUDA"):
        with profiling.profile_scope(activities=("cpu", "cuda")):
            raise AssertionError("unavailable CUDA entered the measured body")

    assert profiler_module.calls == []
    assert native.entered == 0
    assert torch.cuda.probes == 1


@pytest.mark.parametrize("activities", [(), ("gpu",), ("cpu", "xpu"), ("CPU",)])
def test_scope_rejects_empty_or_unknown_activities(monkeypatch, activities) -> None:
    torch, profiler_module, native = _fake_torch(cuda_available=True)
    monkeypatch.setattr(profiling, "_load_torch", lambda: torch)

    with pytest.raises(ValueError, match="activit"):
        with profiling.profile_scope(activities=activities):
            raise AssertionError("invalid activities entered the measured body")

    assert profiler_module.calls == []
    assert native.entered == 0


def test_profile_trace_path_uses_result_stem_and_scope() -> None:
    result = Path("bench/results/2026-08-06T120000.000001Z-serving.v2.json")

    assert profiling.profile_trace_path(result, scope="client-driver") == Path(
        "bench/results/2026-08-06T120000.000001Z-serving.v2."
        "client-driver.pt.trace.json"
    )


@pytest.mark.parametrize(
    "scope",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/scope",
        "nested\\scope",
        "/absolute",
        "C:drive",
        "scope with spaces",
        "scope\x00suffix",
        "x" * 129,
    ],
)
def test_profile_trace_path_rejects_unsafe_scope(scope: str) -> None:
    with pytest.raises(ValueError, match="scope"):
        profiling.profile_trace_path(Path("result.json"), scope=scope)


def test_writer_publishes_valid_trace_exclusively_with_hash_and_private_mode(
    tmp_path,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    payload = _trace_payload()
    exporter = _ExportingProfiler(payload)

    artifact = profiling.write_trace_artifact(
        exporter,
        destination,
        artifact_root=artifact_root,
    )

    assert artifact.path == destination
    assert artifact.size_bytes == len(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.format == profiling.TRACE_FORMAT
    assert artifact.format == "pytorch-chrome-trace-json-v1"
    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert len(exporter.paths) == 1
    assert exporter.paths[0].parent == destination.parent
    assert exporter.paths[0] != destination
    assert list(artifact_root.iterdir()) == [destination]


def test_writer_refuses_existing_artifact_without_overwrite_or_temp_leak(
    tmp_path,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    destination.write_bytes(b"retained evidence")
    exporter = _ExportingProfiler(_trace_payload())

    with pytest.raises(FileExistsError):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert destination.read_bytes() == b"retained evidence"
    assert list(artifact_root.iterdir()) == [destination]


def test_writer_refuses_destination_directory(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    destination.mkdir()

    with pytest.raises(OSError):
        profiling.write_trace_artifact(
            _ExportingProfiler(_trace_payload()),
            destination,
            artifact_root=artifact_root,
        )

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_writer_refuses_destination_symlink_without_touching_target(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside evidence")
    destination = artifact_root / "run.client.pt.trace.json"
    destination.symlink_to(outside)

    with pytest.raises((FileExistsError, ValueError)):
        profiling.write_trace_artifact(
            _ExportingProfiler(_trace_payload()),
            destination,
            artifact_root=artifact_root,
        )

    assert destination.is_symlink()
    assert outside.read_bytes() == b"outside evidence"
    assert list(artifact_root.iterdir()) == [destination]


def test_writer_rejects_lexical_escape_before_export(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / ".." / "escaped.pt.trace.json"
    exporter = _ExportingProfiler(_trace_payload())

    with pytest.raises(ValueError, match="artifact root|outside|escape|contain"):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert exporter.paths == []
    assert not (tmp_path / "escaped.pt.trace.json").exists()
    _assert_directory_empty(artifact_root)


def test_writer_rejects_symlinked_parent_escape_before_export(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "nested").symlink_to(outside, target_is_directory=True)
    destination = artifact_root / "nested" / "run.client.pt.trace.json"
    exporter = _ExportingProfiler(_trace_payload())

    with pytest.raises(ValueError, match="artifact root|outside|symlink|contain"):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert exporter.paths == []
    assert not (outside / destination.name).exists()
    assert list(artifact_root.iterdir()) == [artifact_root / "nested"]


def test_writer_loses_publish_race_without_replacing_winner(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    real_link = os.link

    def racing_link(source, target, *args, **kwargs):
        Path(target).write_bytes(b"race winner")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(profiling.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        profiling.write_trace_artifact(
            _ExportingProfiler(_trace_payload()),
            destination,
            artifact_root=artifact_root,
        )

    assert destination.read_bytes() == b"race winner"
    assert list(artifact_root.iterdir()) == [destination]


def test_writer_rejects_source_inode_content_race_before_publication(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    real_link = os.link
    replacement = json.dumps(
        {"traceEvents": [], "replacement": "unvalidated"}
    ).encode()

    def mutating_link(source, target, *args, **kwargs):
        Path(source).write_bytes(replacement)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(profiling.os, "link", mutating_link)

    with pytest.raises(RuntimeError, match="identity|validation|changed"):
        profiling.write_trace_artifact(
            _ExportingProfiler(_trace_payload()),
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_writer_hashes_the_same_bytes_that_passed_strict_validation(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    exporter = _ExportingProfiler(_trace_payload())
    real_validation = profiling._strict_json_object

    def mutate_after_validation(payload: bytes):
        result = real_validation(payload)
        exporter.paths[0].write_bytes(b"X" * len(payload))
        return result

    monkeypatch.setattr(profiling, "_strict_json_object", mutate_after_validation)

    with pytest.raises(RuntimeError, match="identity validation"):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_writer_rejects_source_inode_replacement_during_publication(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    real_link = os.link

    def replacing_link(source, target, *args, **kwargs):
        replacement = Path(source).with_suffix(".replacement")
        replacement.write_bytes(
            json.dumps({"traceEvents": [], "replacement": True}).encode()
        )
        os.replace(replacement, source)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(profiling.os, "link", replacing_link)

    with pytest.raises(RuntimeError, match="source changed|identity"):
        profiling.write_trace_artifact(
            _ExportingProfiler(_trace_payload()),
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_writer_rejects_fifo_export_without_blocking(tmp_path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO regression requires POSIX mkfifo")

    class FifoExporter:
        def export_chrome_trace(self, path: str) -> None:
            destination = Path(path)
            destination.unlink()
            os.mkfifo(destination)

    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"

    with pytest.raises(ValueError, match="regular file"):
        profiling.write_trace_artifact(
            FifoExporter(),
            destination,
            artifact_root=artifact_root,
        )

    _assert_directory_empty(artifact_root)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"{not-json",
        b"[]",
        b'{}',
        b'{"traceEvents":{}}',
        b'{"traceEvents":[],"traceEvents":[]}',
        b'{"traceEvents":[],"notFinite":NaN}',
        b'{"traceEvents":[],"overflow":1e999}',
        b'\xff',
    ],
)
def test_writer_rejects_empty_or_non_strict_trace_and_cleans_temp(
    tmp_path,
    payload: bytes,
) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"

    with pytest.raises(ValueError):
        profiling.write_trace_artifact(
            _ExportingProfiler(payload),
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_writer_rejects_trace_larger_than_64_mib_and_cleans_temp(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    exporter = _SparseOversizeProfiler()

    with pytest.raises(ValueError, match="64|size|large"):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_writer_cleans_partial_temp_when_export_fails(tmp_path) -> None:
    artifact_root = tmp_path / "profiles"
    artifact_root.mkdir()
    destination = artifact_root / "run.client.pt.trace.json"
    exporter = _ExportingProfiler(
        b'{"traceEvents":[',
        error=OSError("simulated profiler export failure"),
    )

    with pytest.raises(OSError, match="simulated profiler export failure"):
        profiling.write_trace_artifact(
            exporter,
            destination,
            artifact_root=artifact_root,
        )

    assert not destination.exists()
    _assert_directory_empty(artifact_root)


def test_remove_trace_artifact_only_unlinks_exact_untampered_publication(
    tmp_path,
) -> None:
    destination = tmp_path / "run.client.pt.trace.json"
    artifact = profiling.write_trace_artifact(
        _ExportingProfiler(_trace_payload()),
        destination,
        artifact_root=tmp_path,
    )

    assert profiling.remove_trace_artifact(artifact) is True
    assert profiling.remove_trace_artifact(artifact) is False


def test_remove_trace_artifact_refuses_modified_same_inode(tmp_path) -> None:
    destination = tmp_path / "run.client.pt.trace.json"
    artifact = profiling.write_trace_artifact(
        _ExportingProfiler(_trace_payload()),
        destination,
        artifact_root=tmp_path,
    )
    destination.write_bytes(json.dumps({"traceEvents": []}).encode())

    with pytest.raises(RuntimeError, match="modified"):
        profiling.remove_trace_artifact(artifact)

    assert destination.exists()


def test_remove_trace_artifact_restores_a_race_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "run.client.pt.trace.json"
    artifact = profiling.write_trace_artifact(
        _ExportingProfiler(_trace_payload()),
        destination,
        artifact_root=tmp_path,
    )
    replacement = tmp_path / "replacement"
    replacement_payload = b"concurrent winner"
    replacement.write_bytes(replacement_payload)
    real_replace = os.replace

    def replace_before_quarantine(source, target):
        if Path(source) == destination and str(target).endswith(".rollback"):
            real_replace(replacement, destination)
        return real_replace(source, target)

    monkeypatch.setattr(profiling.os, "replace", replace_before_quarantine)

    with pytest.raises(RuntimeError, match="replaced"):
        profiling.remove_trace_artifact(artifact)

    assert destination.read_bytes() == replacement_payload
    assert list(tmp_path.iterdir()) == [destination]


def test_real_cpu_profiler_trace_round_trips_through_writer(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    destination = tmp_path / "real.client.pt.trace.json"

    with profiling.profile_scope(
        activities=("cpu",),
        acc_events=True,
    ) as native:
        with profiling.record_function_scope(True, "client.measurement"):
            assert int((torch.ones(2) + 1).sum().item()) == 4
    artifact = profiling.write_trace_artifact(
        native,
        destination,
        artifact_root=tmp_path,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert artifact.size_bytes > 0
    assert payload["traceEvents"]
    assert any(
        event.get("name") == "client.measurement"
        for event in payload["traceEvents"]
        if isinstance(event, dict)
    )
