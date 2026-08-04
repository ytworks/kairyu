"""CPU contracts for the issue #333 process-isolation HTTP diagnostic."""

from __future__ import annotations

import copy
import importlib
import json
import re
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import g2_a6_vllm_bench as a6
from bench import issue_333_proc_http_bench as diagnostic
from bench import run_g2_a6_formal as formal


class _StableTokenizer:
    _piece = re.compile(r" p(\d{3})")

    def __init__(self, size: int = 256) -> None:
        self._pieces = [f" p{token_id:03d}" for token_id in range(size)]

    def vocab(self) -> list[str]:
        return list(self._pieces)

    def decode(self, token_ids) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def encode(self, text: str) -> tuple[int, ...]:
        result: list[int] = []
        position = 0
        for match in self._piece.finditer(text):
            if match.start() != position:
                raise ValueError("fixture text is not token aligned")
            result.append(int(match.group(1)))
            position = match.end()
        if position != len(text):
            raise ValueError("fixture text has an unmatched suffix")
        return tuple(result)


def _dataset_payload() -> str:
    records = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        f" p{index % 200:03d}"
                        f" p{(index + 1) % 200:03d}"
                        f" p{(index + 2) % 200:03d}"
                        f" p{(index + 3) % 200:03d}"
                    ),
                },
                {"from": "gpt", "value": "fixture response"},
            ]
        }
        for index in range(160)
    ]
    return json.dumps(records)


@pytest.fixture
def trace_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(_dataset_payload(), encoding="utf-8")
    dataset_sha = a6.sha256_file(dataset)
    monkeypatch.setattr(a6, "SHAREGPT_DATASET_SHA256", dataset_sha)
    tokenizer = _StableTokenizer()
    return {
        "schema_version": a6.TRACE_BUNDLE_SCHEMA,
        "benchmark": a6.benchmark_config(dataset_sha),
        "graph_warmup": a6.build_graph_warmup_trace(tokenizer),
        "traces": {
            "sharegpt": a6.build_sharegpt_trace(
                dataset,
                tokenizer,
                expected_dataset_sha256=dataset_sha,
            )
        },
    }


def _request_row(
    expected: dict[str, object],
    *,
    phase: str,
    start: int,
    end: int,
    completion_tokens: int,
    output_key: str,
) -> dict[str, object]:
    first = start + 10
    return {
        "type": "request",
        "phase": phase,
        "sequence": expected["sequence"],
        "attempts": 1,
        "prompt_text_sha256": expected["prompt_text_sha256"],
        "prompt_tokens": expected["prompt_tokens"],
        "prompt_token_ids_sha256": expected["prompt_token_ids_sha256"],
        "expected_completion_tokens": completion_tokens,
        "graph_batch_size": None,
        "graph_batch_sequence": None,
        "http_status": 200,
        "content_type": "text/event-stream; charset=utf-8",
        "response_request_header": f"request-{phase}-{expected['sequence']}",
        "response_id": f"completion-{phase}-{expected['sequence']}",
        "status": "success",
        "error_type": None,
        "done": True,
        "done_events": 1,
        "terminal_choice": True,
        "terminal_choice_events": 1,
        "finish_reasons": ["length"],
        "usage_events": 1,
        "usage": {
            "prompt_tokens": expected["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
        },
        "timing_ns": {
            "start": start,
            "first_token": first,
            "end": end,
            "ttft": first - start,
            "total": end - start,
        },
        "output_text_sha256": a6.sha256_text(output_key),
        "stream_payload_sha256": a6.sha256_text(f"stream-{output_key}"),
    }


def _request_rows(trace_bundle: dict[str, object]) -> list[dict[str, object]]:
    traces = trace_bundle["traces"]
    assert isinstance(traces, dict)
    trace = traces["sharegpt"]
    assert isinstance(trace, dict)
    trace_raw = trace["value"]
    assert isinstance(trace_raw, dict)
    warmup_expected = trace_raw["warmup"]
    measurement_expected = trace_raw["requests"]
    assert isinstance(warmup_expected, list)
    assert isinstance(measurement_expected, list)
    result: list[dict[str, object]] = []
    for sequence, expected in enumerate(warmup_expected):
        assert isinstance(expected, dict)
        start = 1_000 + sequence * 100
        result.append(
            _request_row(
                expected,
                phase="warmup",
                start=start,
                end=start + 50,
                completion_tokens=a6.WARMUP_MAX_TOKENS,
                output_key=f"warmup-{sequence}",
            )
        )

    graph = trace_bundle["graph_warmup"]
    assert isinstance(graph, dict)
    graph_raw = graph["value"]
    assert isinstance(graph_raw, dict)
    prompts = graph_raw["prompts"]
    assert isinstance(prompts, list)
    offset = 0
    anchor = 10_000
    for batch_size in a6.GRAPH_WARMUP_BATCH_SIZES:
        release = anchor
        for batch_sequence in range(batch_size):
            expected = prompts[offset]
            assert isinstance(expected, dict)
            start = release + batch_sequence + 1
            row = _request_row(
                expected,
                phase="graph_warmup",
                start=start,
                end=start + 50,
                completion_tokens=a6.GRAPH_WARMUP_COMPLETION_TOKENS,
                output_key=f"graph-{offset}",
            )
            row.update(
                {
                    "graph_batch_size": batch_size,
                    "graph_batch_sequence": batch_sequence,
                    "graph_warmup_release_ns": release,
                }
            )
            result.append(row)
            offset += 1
        anchor = max(int(row["timing_ns"]["end"]) for row in result[-batch_size:]) + 100

    release = 100_000
    for sequence, expected in enumerate(measurement_expected):
        assert isinstance(expected, dict)
        start = release + sequence + 1
        row = _request_row(
            expected,
            phase="measurement",
            start=start,
            end=start + 10_000,
            completion_tokens=a6.SHAREGPT_MAX_TOKENS,
            output_key=f"measurement-{sequence}",
        )
        row["burst_release_ns"] = release
        result.append(row)
    return result


def _backend(arm: str) -> dict[str, object]:
    return diagnostic.hashed_descriptor(diagnostic.expected_backend_projection(arm))


def _process_tree(arm: str) -> dict[str, object]:
    rows = [
        {
            "pid": 1,
            "ppid": 0,
            "pgid": 1,
            "state": "S",
            "comm": "kairyu",
            "argv": "kairyu serve",
        }
    ]
    if arm == "kairyu-proc":
        rows.append(
            {
                "pid": 2,
                "ppid": 1,
                "pgid": 2,
                "state": "S",
                "comm": "python",
                "argv": "python -c from multiprocessing.spawn import spawn_main",
            }
        )
        rows.extend(
            {
                "pid": pid,
                "ppid": 2,
                "pgid": 2,
                "state": "S",
                "comm": "python",
                "argv": "python -c from multiprocessing.spawn import spawn_main",
            }
            for pid in (3, 4, 5)
        )
    else:
        rows.extend(
            {
                "pid": pid,
                "ppid": 1,
                "pgid": 1,
                "state": "S",
                "comm": "python",
                "argv": "python -c from multiprocessing.spawn import spawn_main",
            }
            for pid in (2, 3, 4)
        )
    derived = diagnostic._derived_process_tree_shape(rows, arm=arm)
    assert derived is not None
    return diagnostic.hashed_descriptor({"arm": arm, "rows": rows, **derived})


def test_live_launch_attestation_rejects_unexpected_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "kairyu").mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("backend: kairyu\n", encoding="utf-8")
    image_id = "sha256:" + "a" * 64
    mounts = [
        {
            "Destination": "/models",
            "Type": "volume",
            "Name": "fixture-model",
            "RW": False,
        },
        {
            "Destination": "/workspace",
            "Type": "bind",
            "Source": str(repo.resolve()),
            "RW": False,
        },
        {
            "Destination": "/app/kairyu",
            "Type": "bind",
            "Source": str((repo / "kairyu").resolve()),
            "RW": False,
        },
        {
            "Destination": "/etc/kairyu/config.yaml",
            "Type": "bind",
            "Source": str(config_path.resolve()),
            "RW": False,
        },
        {
            "Destination": "/unexpected",
            "Type": "bind",
            "Source": str(tmp_path.resolve()),
            "RW": False,
        },
    ]
    inspected = [
        {
            "Id": "b" * 64,
            "Image": image_id,
            "Config": {
                "Image": image_id,
                "Entrypoint": ["/app/.venv/bin/kairyu"],
                "Cmd": ["serve", "/etc/kairyu/config.yaml"],
                "Env": [
                    "PYTHONDONTWRITEBYTECODE=1",
                    f"PYTHONPYCACHEPREFIX={diagnostic.PYCACHE_PREFIX}",
                    "KAIRYU_ATTENTION_BACKEND=flashinfer",
                ],
            },
            "HostConfig": {},
            "Mounts": mounts,
        }
    ]
    monkeypatch.setattr(
        formal,
        "run_command",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(inspected)),
    )

    with pytest.raises(diagnostic.DiagnosticRunError, match="unexpected mount"):
        diagnostic.container_launch_attestation(
            args=SimpleNamespace(
                model_volume="fixture-model",
                repo=repo,
                port=18080,
            ),
            name="fixture",
            cell=diagnostic.diagnostic_plan()[0],
            config_path=config_path,
            config_descriptor=diagnostic.hashed_descriptor({}),
            image_id=image_id,
            source_hashes={},
        )


def test_docker_command_pins_bridge_port_and_isolated_pycache(tmp_path: Path) -> None:
    command = diagnostic.docker_command(
        args=SimpleNamespace(
            model_volume="fixture-model",
            repo=tmp_path,
            port=18080,
        ),
        name="g2a6-issue333-fixture",
        config_path=tmp_path / "config.yaml",
        image_id="sha256:" + "a" * 64,
    )

    assert command[command.index("--network") + 1] == "bridge"
    assert command[command.index("-p") + 1] == "127.0.0.1:18080:8000"
    assert command[command.index("--tmpfs") + 1] == (
        f"{diagnostic.PYCACHE_PREFIX}:ro,nosuid,nodev,noexec,size=1m"
    )
    assert f"PYTHONPYCACHEPREFIX={diagnostic.PYCACHE_PREFIX}" in command


def test_gpu_quiescence_query_failure_is_not_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        formal,
        "run_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="injected nvidia-smi failure",
        ),
    )

    with pytest.raises(diagnostic.DiagnosticRunError, match="cannot query"):
        diagnostic._selected_gpu_quiescence(
            [SimpleNamespace(uuid=f"GPU-{index}") for index in range(diagnostic.TP_SIZE)]
        )


def test_runner_source_must_be_same_root_and_detached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(diagnostic.DiagnosticRunError, match="source root executing"):
        diagnostic.runner_source_hashes(tmp_path)

    monkeypatch.setattr(
        formal,
        "run_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="refs/heads/main\n",
            stderr="",
        ),
    )
    with pytest.raises(diagnostic.DiagnosticRunError, match="detached HEAD"):
        diagnostic.runner_source_hashes(diagnostic.DEFAULT_REPO)


def test_log_capture_failure_is_retained_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        formal,
        "run_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="partial-log\n",
            stderr="capture-error\n",
        ),
    )
    path = tmp_path / "logs" / "cell.log"

    with pytest.raises(diagnostic.DiagnosticRunError, match="retain Docker logs"):
        diagnostic._save_container_logs_strict(
            "g2a6-issue333-fixture",
            path,
            work_dir=tmp_path,
        )

    assert path.read_text(encoding="utf-8") == "partial-log\ncapture-error\n"


def test_cleanup_attempts_remove_and_gpu_wait_after_log_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_log(*_args, **_kwargs):
        calls.append("log")
        raise diagnostic.DiagnosticRunError("injected log failure")

    def remove(_name: str) -> dict[str, object]:
        calls.append("remove")
        return {"docker_rm_returncode": 0, "absent_after_remove": True}

    def wait(_inventory) -> dict[str, object]:
        calls.append("gpu-wait")
        return {
            "selected_gpu_uuids": [],
            "compute_applications": [],
            "polls": 1,
        }

    monkeypatch.setattr(diagnostic, "_save_container_logs_strict", fail_log)
    monkeypatch.setattr(diagnostic, "_remove_container_strict", remove)
    monkeypatch.setattr(diagnostic, "_wait_for_selected_gpu_quiescence", wait)

    log, removal, quiescence, errors = diagnostic._attempt_container_cleanup(
        name="g2a6-issue333-fixture",
        log_path=tmp_path / "logs" / "cell.log",
        work_dir=tmp_path,
        inventory=[],
    )

    assert calls == ["log", "remove", "gpu-wait"]
    assert log is None
    assert removal == {"docker_rm_returncode": 0, "absent_after_remove": True}
    assert quiescence == {
        "selected_gpu_uuids": [],
        "compute_applications": [],
        "polls": 1,
    }
    assert len(errors) == 1


def _provenance(
    cell: diagnostic.Cell,
    *,
    session_id: str,
) -> dict[str, object]:
    template = diagnostic.DEFAULT_KAIRYU_TEMPLATE.read_text(encoding="utf-8")
    text = diagnostic.render_kairyu_config(template, arm=cell.arm)
    configuration = diagnostic.configuration_source(text, arm=cell.arm)
    source_raw = configuration["value"]
    assert isinstance(source_raw, dict)
    source_hashes = {
        path: (f"{index + 1:x}" * 64)[:64]
        for index, path in enumerate(diagnostic.CRITICAL_SOURCE_FILES)
    }
    runner_hashes = {
        relative: diagnostic.sha256_file(diagnostic.DEFAULT_REPO / relative)
        for relative in diagnostic.RUNNER_SOURCE_FILES
    }
    image_id = "sha256:" + "a" * 64
    engine_imports = {
        relative: {
            "module": module,
            "path": f"/app/{relative}",
            "sha256": source_hashes[relative],
        }
        for relative, module in diagnostic.critical_source_modules().items()
    }
    launch = diagnostic.hashed_descriptor(
        {
            "container_id": f"{cell.order_index + 1:x}" * 64,
            "image_id": image_id,
            "entrypoint": ["/app/.venv/bin/kairyu"],
            "argv": ["serve", "/etc/kairyu/config.yaml"],
            "required_environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": diagnostic.PYCACHE_PREFIX,
                "KAIRYU_ATTENTION_BACKEND": "flashinfer",
            },
            "mounts": [
                {
                    "destination": "/app/kairyu",
                    "type": "bind",
                    "name": None,
                    "rw": False,
                },
                {
                    "destination": "/etc/kairyu/config.yaml",
                    "type": "bind",
                    "name": None,
                    "rw": False,
                },
                {
                    "destination": "/models",
                    "type": "volume",
                    "name": "fixture-model-volume",
                    "rw": False,
                },
                {
                    "destination": "/workspace",
                    "type": "bind",
                    "name": None,
                    "rw": False,
                },
            ],
            "host_config": {
                "ipc_mode": "host",
                "network_mode": "bridge",
                "tmpfs": {diagnostic.PYCACHE_PREFIX: "ro,nosuid,nodev,noexec,size=1m"},
                "memlock_soft": -1,
                "memlock_hard": -1,
                "published_host_ip": "127.0.0.1",
                "published_host_port": 18080,
                "gpu_driver": "",
                "gpu_count": 0,
                "gpu_device_ids": ["0", "1", "2", "3"],
                "gpu_capabilities": [["gpu"]],
                "gpu_options": {},
            },
            "source_files": source_hashes,
            "engine_imports": engine_imports,
            "python_import_policy": {
                "dont_write_bytecode": True,
                "pycache_prefix": diagnostic.PYCACHE_PREFIX,
                "prefix_empty_before_and_after_probe": True,
            },
            "configuration_text_sha256": source_raw["text_sha256"],
            "diagnostic_arm": cell.arm,
        }
    )
    uuids = [f"GPU-{index}" for index in range(4)]
    pci = [f"00000000:{index:02X}:00.0" for index in range(4)]
    return diagnostic.hashed_descriptor(
        {
            "schema_version": 1,
            "diagnostic_only": True,
            "session_id": session_id,
            "server_generation": f"generation-{cell.order_index}",
            "server_started_ns": 1_000_000 + cell.order_index,
            "source": {
                "repo_commit": "b" * 40,
                "source_clean": True,
                "critical_file_sha256": source_hashes,
                "runner_file_sha256": runner_hashes,
            },
            "host": {
                "id": "fixture-host",
                "environment": {
                    "measurement_session_id": "fixture-environment-session",
                    "measured_at_ns": 1,
                    "artifact_date": "2026-07-30",
                    "artifact": {
                        "path": "bench/results/env-2026-07-30.json",
                        "sha256": "c" * 64,
                    },
                    "physical_topology_sha256": "d" * 64,
                    "fabric": {
                        "kind": "fixture-fabric",
                        "raw_p2p_matrix_sha256": "e" * 64,
                        "minimum_peer_bandwidth_gbs": 1.0,
                    },
                },
                "pre_start_gpu_quiescence": {
                    "selected_gpu_uuids": uuids,
                    "compute_applications": [],
                },
            },
            "gpu": {
                "uuids": uuids,
                "pci_bus_ids": pci,
                "product_name": "fixture-gpu",
                "driver_version": "fixture-driver",
                "container_visible_uuids": uuids,
                "container_visible_pci_bus_ids": pci,
            },
            "model": a6.model_identity(),
            "checkpoint": formal.dry_checkpoint_attestation(a6),
            "image": {"ref": "kairyu:test", "id": image_id},
            "runtime_versions": {
                "cuda_version": "13.0",
                "nccl_version": "2.29.7",
                "package_versions": {
                    "torch": "2.9.0",
                    "flashinfer": "0.6.6",
                    "uvloop": "0.22.1",
                    "httptools": "0.8.0",
                    "uvicorn": "0.49.0",
                    "kairyu": "0.1.0",
                },
            },
            "container_launch": launch,
            "configuration_source": configuration,
            "diagnostic_arm": cell.arm,
        }
    )


def _cleanup(
    cell: diagnostic.Cell,
    provenance: dict[str, object],
) -> dict[str, object]:
    raw = provenance["value"]
    assert isinstance(raw, dict)
    uuids = raw["gpu"]["uuids"]
    return diagnostic.cleanup_attestation(
        cell=cell,
        provenance=provenance,
        log={
            "path": f"logs/{cell.key}.log",
            "sha256": f"{cell.order_index + 5:x}" * 64,
            "bytes": 10,
        },
        removal={"docker_rm_returncode": 0, "absent_after_remove": True},
        post_cleanup_quiescence={
            "selected_gpu_uuids": uuids,
            "compute_applications": [],
            "polls": 1,
        },
    )


def _rehash_launch_tamper(
    provenance: dict[str, object],
    mutate,
) -> dict[str, object]:
    raw = copy.deepcopy(provenance["value"])
    assert isinstance(raw, dict)
    launch = raw["container_launch"]
    assert isinstance(launch, dict)
    launch_raw = copy.deepcopy(launch["value"])
    assert isinstance(launch_raw, dict)
    mutate(launch_raw)
    raw["container_launch"] = diagnostic.hashed_descriptor(launch_raw)
    return diagnostic.hashed_descriptor(raw)


def _rehash_provenance_tamper(
    provenance: dict[str, object],
    mutate,
) -> dict[str, object]:
    raw = copy.deepcopy(provenance["value"])
    assert isinstance(raw, dict)
    mutate(raw)
    return diagnostic.hashed_descriptor(raw)


def _shards(
    tmp_path: Path,
    trace_bundle: dict[str, object],
    *,
    session_id: str = "fixture-session",
) -> list[Path]:
    result: list[Path] = []
    for cell in diagnostic.diagnostic_plan():
        request_rows = _request_rows(trace_bundle)
        timing = [row["timing_ns"] for row in request_rows if row.get("phase") == "measurement"]
        end_evidence = {
            "backend_after_warmup": _backend(cell.arm),
            "backend_after_measurement": _backend(cell.arm),
            "process_tree_after_warmup": _process_tree(cell.arm),
            "process_tree_after_measurement": _process_tree(cell.arm),
            "measurement_window_ns": {
                "start": min(int(item["start"]) for item in timing),
                "end": max(int(item["end"]) for item in timing),
            },
        }
        provenance = _provenance(cell, session_id=session_id)
        rows = diagnostic.measurement_rows(
            cell=cell,
            trace_bundle=trace_bundle,
            provenance=provenance,
            cleanup=_cleanup(cell, provenance),
            backend_ready=_backend(cell.arm),
            process_tree_ready=_process_tree(cell.arm),
            requests=request_rows,
            end_evidence=end_evidence,
        )
        path = tmp_path / f"{cell.key}.jsonl"
        diagnostic.write_measurement_shard(path, rows)
        result.append(path)
    return result


def _set_proc_measurement_ttft(shards: list[Path], ttft_ns: int) -> None:
    for path in shards:
        rows = diagnostic._read_jsonl(path)
        if rows[0].get("arm") != "kairyu-proc":
            continue
        for row in rows:
            if row.get("type") != "request" or row.get("phase") != "measurement":
                continue
            timing = row["timing_ns"]
            assert isinstance(timing, dict)
            timing["first_token"] = int(timing["start"]) + ttft_ns
            timing["ttft"] = ttft_ns
        diagnostic._write_jsonl(path, rows)


def _bind_fixture_logs(work_dir: Path, shards: list[Path]) -> None:
    for shard in shards:
        rows = diagnostic._read_jsonl(shard)
        cell = diagnostic._scenario_from_identity(rows[0])
        content = f"fixture log for {cell.scenario_id}\n"
        log_path = work_dir / "logs" / f"{cell.key}.log"
        diagnostic.atomic_write_text(log_path, content)
        provenance = rows[0]["provenance"]
        assert isinstance(provenance, dict)
        rows[-1]["cleanup"] = diagnostic.cleanup_attestation(
            cell=cell,
            provenance=provenance,
            log={
                "path": f"logs/{cell.key}.log",
                "sha256": diagnostic.sha256_file(log_path),
                "bytes": log_path.stat().st_size,
            },
            removal={"docker_rm_returncode": 0, "absent_after_remove": True},
            post_cleanup_quiescence={
                "selected_gpu_uuids": provenance["value"]["gpu"]["uuids"],
                "compute_applications": [],
                "polls": 1,
            },
        )
        diagnostic._write_jsonl(shard, rows)


def test_plan_is_exact_two_pair_fresh_server_abba() -> None:
    plan = diagnostic.diagnostic_plan()

    assert tuple(cell.arm for cell in plan) == (
        "kairyu",
        "kairyu-proc",
        "kairyu-proc",
        "kairyu",
    )
    assert [(cell.repeat, cell.arm) for cell in plan] == [
        (0, "kairyu"),
        (0, "kairyu-proc"),
        (1, "kairyu-proc"),
        (1, "kairyu"),
    ]
    assert len({cell.scenario_id for cell in plan}) == 4


def test_configs_differ_only_by_backend_and_keep_the_a6_tp4_shape() -> None:
    template = diagnostic.DEFAULT_KAIRYU_TEMPLATE.read_text(encoding="utf-8")
    inproc = diagnostic.render_kairyu_config(template, arm="kairyu")
    proc = diagnostic.render_kairyu_config(template, arm="kairyu-proc")

    assert inproc.replace("backend: kairyu", "backend: <arm>") == proc.replace(
        "backend: kairyu-proc", "backend: <arm>"
    )
    assert yaml_backend(inproc) == "kairyu"
    assert yaml_backend(proc) == "kairyu-proc"
    for text in (inproc, proc):
        for pin in (
            "tensor_parallel_size: 4",
            "num_pages: 8193",
            "max_model_len: 8192",
            "max_num_batched_tokens: 1024",
            "max_num_seqs: 16",
            "pipeline_depth: 5",
            "cuda_graph_max_batch: 16",
        ):
            assert pin in text


def yaml_backend(text: str) -> str:
    parsed = json.loads(json.dumps(__import__("yaml").safe_load(text)))
    return str(parsed["engines"]["qwen3-32b"]["backend"])


def test_backend_descriptor_fails_closed_on_proc_mislabel_or_tp() -> None:
    valid = _backend("kairyu-proc")
    wrong_label = copy.deepcopy(valid["value"])
    assert isinstance(wrong_label, dict)
    wrong_label["engine_backend"] = "kairyu"
    wrong_tp = copy.deepcopy(valid["value"])
    assert isinstance(wrong_tp, dict)
    wrong_tp["tensor_parallel_size"] = 1

    assert diagnostic.validate_backend_descriptor(valid, arm="kairyu-proc")
    assert not diagnostic.validate_backend_descriptor(
        diagnostic.hashed_descriptor(wrong_label),
        arm="kairyu-proc",
    )
    assert not diagnostic.validate_backend_descriptor(
        diagnostic.hashed_descriptor(wrong_tp),
        arm="kairyu-proc",
    )


def test_process_tree_recomputes_exact_proc_tp4_rank_and_group_shape() -> None:
    valid = _process_tree("kairyu-proc")
    assert diagnostic.validate_process_tree_descriptor(valid, arm="kairyu-proc")

    missing_rank = copy.deepcopy(valid["value"])
    assert isinstance(missing_rank, dict)
    missing_rank["rows"] = missing_rank["rows"][:-1]
    wrong_pgid = copy.deepcopy(valid["value"])
    assert isinstance(wrong_pgid, dict)
    wrong_pgid["rows"][2]["pgid"] = 99
    trusted_boolean = copy.deepcopy(valid["value"])
    assert isinstance(trusted_boolean, dict)
    trusted_boolean["live_spawn_worker_count"] = 1

    for tampered in (missing_rank, wrong_pgid, trusted_boolean):
        assert not diagnostic.validate_process_tree_descriptor(
            diagnostic.hashed_descriptor(tampered),
            arm="kairyu-proc",
        )


def test_process_tree_rejects_proc_session_hidden_in_inproc_arm() -> None:
    proc_style = copy.deepcopy(_process_tree("kairyu-proc")["value"])
    assert isinstance(proc_style, dict)
    proc_style["arm"] = "kairyu"
    proc_style["api_engine_process_split"] = False

    assert not diagnostic.validate_process_tree_descriptor(
        diagnostic.hashed_descriptor(proc_style),
        arm="kairyu",
    )


def test_process_tree_requires_all_live_spawn_workers() -> None:
    zombie = copy.deepcopy(_process_tree("kairyu-proc")["value"])
    assert isinstance(zombie, dict)
    zombie["rows"][3]["state"] = "Z"

    assert not diagnostic.validate_process_tree_descriptor(
        diagnostic.hashed_descriptor(zombie),
        arm="kairyu-proc",
    )


def test_process_tree_generation_identity_allows_only_state_to_vary() -> None:
    sleeping = _process_tree("kairyu-proc")
    running_raw = copy.deepcopy(sleeping["value"])
    assert isinstance(running_raw, dict)
    running_raw["rows"][2]["state"] = "R"
    running = diagnostic.hashed_descriptor(running_raw)

    assert diagnostic.validate_process_tree_descriptor(running, arm="kairyu-proc")
    assert diagnostic.process_tree_generation_identity(
        sleeping,
        arm="kairyu-proc",
    ) == diagnostic.process_tree_generation_identity(
        running,
        arm="kairyu-proc",
    )


def test_provenance_replay_rejects_rehashed_launch_env_mount_and_port_tamper() -> None:
    cell = diagnostic.diagnostic_plan()[0]
    valid = _provenance(cell, session_id="fixture-session")
    assert diagnostic.validate_provenance_descriptor(valid, cell=cell)

    def wrong_env(launch: dict[str, object]) -> None:
        launch["required_environment"] = {
            "PYTHONDONTWRITEBYTECODE": "0",
            "KAIRYU_ATTENTION_BACKEND": "flashinfer",
        }

    def writable_mount(launch: dict[str, object]) -> None:
        mounts = launch["mounts"]
        assert isinstance(mounts, list)
        mounts[0]["rw"] = True

    def missing_volume_name(launch: dict[str, object]) -> None:
        mounts = launch["mounts"]
        assert isinstance(mounts, list)
        model = next(item for item in mounts if item["destination"] == "/models")
        model["name"] = ""

    def public_ip(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["published_host_ip"] = "0.0.0.0"

    def invalid_port(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["published_host_port"] = 0

    def wrong_network(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["network_mode"] = "host"

    def missing_pycache_tmpfs(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["tmpfs"] = {}

    def invalid_gpu_request(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["gpu_capabilities"] = [["compute"]]

    def weakened_import_policy(launch: dict[str, object]) -> None:
        policy = launch["python_import_policy"]
        assert isinstance(policy, dict)
        policy["prefix_empty_before_and_after_probe"] = False

    for mutate in (
        wrong_env,
        writable_mount,
        missing_volume_name,
        public_ip,
        invalid_port,
        wrong_network,
        missing_pycache_tmpfs,
        invalid_gpu_request,
        weakened_import_policy,
    ):
        tampered = _rehash_launch_tamper(valid, mutate)
        assert not diagnostic.validate_provenance_descriptor(tampered, cell=cell)


def test_provenance_replay_rejects_environment_and_runner_hash_tamper() -> None:
    cell = diagnostic.diagnostic_plan()[0]
    valid = _provenance(cell, session_id="fixture-session")

    def empty_environment(raw: dict[str, object]) -> None:
        raw["host"]["environment"] = {}

    def wrong_runner_hash(raw: dict[str, object]) -> None:
        runner = raw["source"]["runner_file_sha256"]
        assert isinstance(runner, dict)
        runner[diagnostic.RUNNER_SOURCE_FILES[0]] = "f" * 64

    for mutate in (empty_environment, wrong_runner_hash):
        assert not diagnostic.validate_provenance_descriptor(
            _rehash_provenance_tamper(valid, mutate),
            cell=cell,
        )


def test_artifact_replay_requires_one_common_valid_host_port(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    rows = diagnostic._read_jsonl(shards[0])

    def another_valid_port(launch: dict[str, object]) -> None:
        host = launch["host_config"]
        assert isinstance(host, dict)
        host["published_host_port"] = 18081

    tampered = _rehash_launch_tamper(rows[0]["provenance"], another_valid_port)
    rows[0]["provenance"] = tampered
    rows[-1]["provenance"] = tampered
    diagnostic._write_jsonl(shards[0], rows)
    artifact_rows = diagnostic.assemble_rows(shards)
    artifact_rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "port-mismatch", artifact_rows)

    assert manifest["checks"]["source_image_model_gpu_config_provenance_exact"] is True
    assert (
        manifest["checks"]["backend_only_configuration_and_common_runtime_identity_exact"] is False
    )
    report = manifest["comparisons"]["report_only_material_reduction_classification"]
    assert report["classification"] == "insufficient_evidence"
    assert report["hypothesis_conclusion"] is None


def test_valid_artifact_replays_without_an_a6_performance_claim(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    rows = diagnostic.assemble_rows(_shards(tmp_path, trace_bundle))
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "artifact", rows)
    replayed = diagnostic.verify_artifact(
        tmp_path / "artifact",
        assert_integrity=True,
    )

    assert replayed == manifest
    assert manifest["evidence_valid"] is True
    assert manifest["diagnostic_only"] is True
    assert manifest["a6_gate_claimed"] is False
    assert manifest["external_workdir_logs_required"] is False
    assert manifest["performance_acceptance_threshold"] is None
    assert "passed" not in manifest
    assert all(manifest["checks"].values())
    paired = manifest["comparisons"]["paired_rounds"]
    assert [item["server_order"] for item in paired] == [
        ["kairyu", "kairyu-proc"],
        ["kairyu-proc", "kairyu"],
    ]
    assert manifest["comparisons"]["paired_median"]["goodput_rps"]["decimal"] == 1.0
    report = manifest["comparisons"]["report_only_material_reduction_classification"]
    assert report["criterion"] == "<= 0.90"
    assert report["classification"] == "no_material_reduction"
    assert report["binding"] is False
    method = manifest["diagnostics"]["method"]["report_only_material_reduction_classification"]
    assert method["predeclared_before_measurement"] is True
    assert method["binding"] is False


def test_live_artifact_rechecks_bound_workdir_logs(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _bind_fixture_logs(tmp_path, shards)
    rows = diagnostic.assemble_rows(
        shards,
        external_workdir_logs_required=True,
    )
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "artifact", rows)

    assert manifest["external_workdir_logs_required"] is True
    assert diagnostic.verify_artifact(tmp_path / "artifact", assert_integrity=True) == manifest

    target = tmp_path / "logs" / f"{diagnostic.diagnostic_plan()[0].key}.log"
    retained = target.read_bytes()
    target.write_bytes(retained + b"tamper")
    with pytest.raises(ValueError, match="differs from cleanup binding"):
        diagnostic.verify_artifact(tmp_path / "artifact")

    target.write_bytes(retained)
    target.unlink()
    with pytest.raises(ValueError, match="retained Docker log is absent"):
        diagnostic.verify_artifact(tmp_path / "artifact")


def test_assemble_rejects_incomplete_abba_matrix(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)

    with pytest.raises(ValueError, match="exact four-cell ABBA"):
        diagnostic.assemble_rows(shards[:-1])


def test_predeclared_p99_ceiling_classifies_exact_point_nine_as_material(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _set_proc_measurement_ttft(shards, 9)
    rows = diagnostic.assemble_rows(shards)
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "material", rows)
    report = manifest["comparisons"]["report_only_material_reduction_classification"]

    assert manifest["evidence_valid"] is True
    assert report["observed"] == {
        "numerator": 9,
        "denominator": 10,
        "decimal": 0.9,
    }
    assert report["classification"] == "material_reduction"
    assert report["material_reduction"] is True
    assert report["interpretation"] == ("dominant process/GIL-contention hypothesis supported")
    assert "net backend swap" in report["causal_scope"]


def test_material_classification_is_report_only_and_exact() -> None:
    at_ceiling = diagnostic.material_reduction_classification(
        Fraction(9, 10),
        evidence_valid=True,
    )
    above = diagnostic.material_reduction_classification(
        Fraction(901, 1000),
        evidence_valid=True,
    )

    assert at_ceiling["material_reduction"] is True
    assert above["material_reduction"] is False
    assert at_ceiling["binding"] is False
    assert at_ceiling["report_only"] is True


def test_material_classification_fails_closed_on_invalid_evidence() -> None:
    invalid = diagnostic.material_reduction_classification(
        Fraction(1, 2),
        evidence_valid=False,
    )

    assert invalid["observed"]["decimal"] == 0.5
    assert invalid["classification"] == "insufficient_evidence"
    assert invalid["material_reduction"] is False
    assert invalid["hypothesis_conclusion"] is None


def test_failed_request_cannot_produce_supported_hypothesis(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _set_proc_measurement_ttft(shards, 9)
    proc_path = next(path for path in shards if "kairyu-proc" in path.name)
    proc_rows = diagnostic._read_jsonl(proc_path)
    failed = next(
        row
        for row in proc_rows
        if row.get("type") == "request"
        and row.get("phase") == "measurement"
        and row.get("sequence") == 0
    )
    failed["status"] = "failure"
    failed["error_type"] = "InjectedFailure"
    diagnostic._write_jsonl(proc_path, proc_rows)
    rows = diagnostic.assemble_rows(shards)
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "failed-request", rows)
    report = manifest["comparisons"]["report_only_material_reduction_classification"]

    assert manifest["evidence_valid"] is False
    assert manifest["checks"]["strict_a6_sse_request_evidence_exact"] is False
    assert report["observed"]["decimal"] == 0.9
    assert report["classification"] == "insufficient_evidence"
    assert report["material_reduction"] is False
    assert report["hypothesis_conclusion"] is None


def test_backend_attestation_tamper_cannot_produce_supported_hypothesis(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _set_proc_measurement_ttft(shards, 9)
    proc_path = next(path for path in shards if "kairyu-proc" in path.name)
    proc_rows = diagnostic._read_jsonl(proc_path)
    ready = copy.deepcopy(proc_rows[0]["backend_ready"]["value"])
    assert isinstance(ready, dict)
    ready["tensor_parallel_size"] = 1
    proc_rows[0]["backend_ready"] = diagnostic.hashed_descriptor(ready)
    diagnostic._write_jsonl(proc_path, proc_rows)
    rows = diagnostic.assemble_rows(shards)
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "bad-attestation", rows)
    report = manifest["comparisons"]["report_only_material_reduction_classification"]

    assert manifest["evidence_valid"] is False
    assert manifest["checks"]["three_backend_and_process_attestations_per_cell_exact"] is False
    assert report["observed"]["decimal"] == 0.9
    assert report["classification"] == "insufficient_evidence"
    assert report["hypothesis_conclusion"] is None


def test_rehashed_process_restart_between_attestations_invalidates_evidence(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _set_proc_measurement_ttft(shards, 9)
    proc_path = next(path for path in shards if "kairyu-proc" in path.name)
    proc_rows = diagnostic._read_jsonl(proc_path)
    ready = proc_rows[0]["process_tree_ready"]
    restarted_raw = copy.deepcopy(proc_rows[-1]["process_tree_after_measurement"]["value"])
    assert isinstance(restarted_raw, dict)
    pid_map = {2: 20, 3: 21, 4: 22, 5: 23}
    for row in restarted_raw["rows"]:
        old_pid = int(row["pid"])
        if old_pid not in pid_map:
            continue
        row["pid"] = pid_map[old_pid]
        row["ppid"] = 1 if old_pid == 2 else 20
        row["pgid"] = 20
    restarted_raw["rows"].sort(key=lambda row: int(row["pid"]))
    derived = diagnostic._derived_process_tree_shape(
        restarted_raw["rows"],
        arm="kairyu-proc",
    )
    assert derived is not None
    restarted = diagnostic.hashed_descriptor(
        {"arm": "kairyu-proc", "rows": restarted_raw["rows"], **derived}
    )
    assert diagnostic.validate_process_tree_descriptor(restarted, arm="kairyu-proc")
    assert diagnostic.process_tree_generation_identity(
        ready,
        arm="kairyu-proc",
    ) != diagnostic.process_tree_generation_identity(
        restarted,
        arm="kairyu-proc",
    )
    proc_rows[-1]["process_tree_after_measurement"] = restarted
    diagnostic._write_jsonl(proc_path, proc_rows)

    rows = diagnostic.assemble_rows(shards)
    rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "restarted-process", rows)
    report = manifest["comparisons"]["report_only_material_reduction_classification"]

    assert manifest["checks"]["three_backend_and_process_attestations_per_cell_exact"] is True
    assert (
        manifest["checks"]["process_tree_generation_identity_stable_across_three_attestations"]
        is False
    )
    assert manifest["evidence_valid"] is False
    assert report["observed"]["decimal"] == 0.9
    assert report["classification"] == "insufficient_evidence"
    assert report["material_reduction"] is False
    assert report["hypothesis_conclusion"] is None


def test_rehashed_cleanup_failure_cannot_produce_supported_hypothesis(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    shards = _shards(tmp_path, trace_bundle)
    _set_proc_measurement_ttft(shards, 9)
    rows = diagnostic._read_jsonl(shards[-1])
    cleanup_raw = copy.deepcopy(rows[-1]["cleanup"]["value"])
    assert isinstance(cleanup_raw, dict)
    cleanup_raw["removal"]["absent_after_remove"] = False
    rows[-1]["cleanup"] = diagnostic.hashed_descriptor(cleanup_raw)
    diagnostic._write_jsonl(shards[-1], rows)

    artifact_rows = diagnostic.assemble_rows(shards)
    artifact_rows[0]["session_id"] = "fixture-session"
    manifest = diagnostic.write_artifact(tmp_path / "cleanup-failure", artifact_rows)
    report = manifest["comparisons"]["report_only_material_reduction_classification"]

    assert manifest["checks"]["logs_removal_and_post_cleanup_gpu_quiescence_exact"] is False
    assert manifest["evidence_valid"] is False
    assert report["observed"]["decimal"] == 0.9
    assert report["classification"] == "insufficient_evidence"
    assert report["hypothesis_conclusion"] is None


def test_tampered_output_parity_is_retained_as_invalid_evidence(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    rows = diagnostic.assemble_rows(_shards(tmp_path, trace_bundle))
    rows[0]["session_id"] = "fixture-session"
    target = next(
        row
        for row in rows
        if row.get("type") == "request"
        and row.get("arm") == "kairyu-proc"
        and row.get("phase") == "measurement"
        and row.get("sequence") == 0
    )
    target["output_text_sha256"] = "f" * 64
    manifest = diagnostic.assemble_manifest(
        [{**row, "row_index": index} for index, row in enumerate(rows)],
        raw_sha256="e" * 64,
    )

    assert manifest["evidence_valid"] is False
    assert manifest["checks"]["completion_output_hash_parity_across_all_four_cells"] is False
    report = manifest["comparisons"]["report_only_material_reduction_classification"]
    assert report["classification"] == "insufficient_evidence"
    assert report["hypothesis_conclusion"] is None


def test_manifest_hash_or_replay_tampering_fails_verification(
    tmp_path: Path,
    trace_bundle: dict[str, object],
) -> None:
    rows = diagnostic.assemble_rows(_shards(tmp_path, trace_bundle))
    rows[0]["session_id"] = "fixture-session"
    diagnostic.write_artifact(tmp_path / "artifact", rows)
    manifest_path = tmp_path / "artifact" / diagnostic.MANIFEST_NAME
    retained = json.loads(manifest_path.read_text(encoding="utf-8"))
    retained["diagnostic_only"] = False
    manifest_path.write_text(json.dumps(retained), encoding="utf-8")

    with pytest.raises(ValueError, match="independent raw replay"):
        diagnostic.verify_artifact(tmp_path / "artifact")


def test_parser_names_integrity_not_performance_gate() -> None:
    parser = diagnostic.build_parser()
    run = parser.parse_args(["run", "--dry-run", "--assert-integrity"])
    verify = parser.parse_args(["verify", "--artifact", "/tmp/evidence", "--assert-integrity"])

    assert run.command == "run"
    assert run.assert_integrity is True
    assert verify.command == "verify"
    assert verify.assert_integrity is True
    assert not any("gate" in action.dest for action in parser._actions)


def test_container_name_is_inside_formal_cleanup_safety_prefix() -> None:
    name = diagnostic._container_name("fixture-session", diagnostic.diagnostic_plan()[0])

    assert name.startswith("g2a6-issue333-")
    assert len(name) <= 120


def test_port_probe_rejects_listener_but_allows_closed_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Probe:
        def __init__(self, result: int) -> None:
            self.result = result

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def connect_ex(self, _address: tuple[str, int]) -> int:
            return self.result

    monkeypatch.setattr(diagnostic.socket, "socket", lambda *_args: Probe(0))
    assert diagnostic._port_available(18080) is False
    monkeypatch.setattr(diagnostic.socket, "socket", lambda *_args: Probe(111))
    assert diagnostic._port_available(18080) is True


def test_source_attestation_covers_proc_tp_runtime_plumbing() -> None:
    assert {
        "kairyu/engine/zmq_backend.py",
        "kairyu/engine/core/engine_service.py",
        "kairyu/engine/core/worker.py",
        "kairyu/engine/config_validation.py",
        "kairyu/engine/registry.py",
    }.issubset(diagnostic.CRITICAL_SOURCE_FILES)


def test_critical_source_module_mapping_is_exact_and_importable() -> None:
    expected = {
        relative: relative.removesuffix(".py").replace("/", ".")
        for relative in diagnostic.CRITICAL_SOURCE_FILES
    }

    assert diagnostic.critical_source_modules() == expected
    assert expected["kairyu/engine/core/worker.py"] == ("kairyu.engine.core.worker")
    assert expected["kairyu/engine/config_validation.py"] == ("kairyu.engine.config_validation")
    repo = Path(diagnostic.__file__).resolve().parents[1]
    for relative, module_name in expected.items():
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve() == (repo / relative).resolve()
