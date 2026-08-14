"""Focused CPU tests for the replayable G5 F4b agentic tier gate."""

from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from verification.fleet.correctness import agentic_kv_tier_f4b_bench as gate


def _checkpoint() -> dict[str, object]:
    weights = [
        {
            "file": f"model-{index:05d}-of-00017.safetensors",
            "bytes": index,
            "sha256": gate.sha256_json(f"weight-{index}"),
        }
        for index in range(1, 18)
    ]
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "architecture": {
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        },
        "weights": weights,
        "weights_sha256": gate.sha256_json(weights),
        "pinned_weights_rollup": gate.EXPECTED_MODEL_WEIGHTS_ROLLUP,
        "metadata_sha256": {
            "config.json": gate.EXPECTED_MODEL_CONFIG_SHA256,
            "tokenizer.json": gate.sha256_json("tokenizer"),
        },
        "model_config_sha256": gate.sha256_json({"model_type": "qwen3"}),
        "read_from": "/models/qwen3-32b",
    }


def _source() -> dict[str, object]:
    python_paths = sorted(
        path.relative_to(gate._REPO_ROOT).as_posix()
        for path in (gate._REPO_ROOT / "kairyu").rglob("*.py")
    )
    bound = {
        path: gate.sha256_json(path)
        for path in sorted(
            set(python_paths)
            | {
                gate.SOURCE_PATH,
                "Dockerfile.cuda",
                "pyproject.toml",
                "uv.lock",
            }
        )
    }
    engine = {path: bound[path] for path in python_paths}
    return {
        "commit": "1" * 40,
        "dirty": False,
        "source_kind": "git",
        "script_path": gate.SOURCE_PATH,
        "script_sha256": bound[gate.SOURCE_PATH],
        "bound_file_sha256": bound,
        "bound_files_sha256": gate.sha256_json(dict(sorted(bound.items()))),
        "kairyu_python_paths": python_paths,
        "engine_source_sha256": gate.sha256_json(dict(sorted(engine.items()))),
        "python": "fixture-python",
        "platform": "fixture-platform",
    }


def _quality_source() -> dict[str, object]:
    source = copy.deepcopy(_source())
    source["commit"] = "2" * 40
    source["bound_file_sha256"][gate.SOURCE_PATH] = gate.sha256_json(
        "quality-script"
    )
    source["script_sha256"] = source["bound_file_sha256"][gate.SOURCE_PATH]
    source["bound_files_sha256"] = gate.sha256_json(
        dict(sorted(source["bound_file_sha256"].items()))
    )
    return source


def _profile(source: dict[str, object]) -> dict[str, object]:
    value = {
        "schema_version": gate.PROFILE_SCHEMA_VERSION,
        "runtime_identity": {
            "tensor_parallel_size": gate.TP_SIZE,
            "page_size": gate.PAGE_SIZE,
            "engine_source_sha256": source["engine_source_sha256"],
        },
        "policy": {
            "rule": "median_lt_1_and_wins_at_least_8_of_9",
            "min_restore_tokens": 1024,
        },
        "evidence_sha256": gate.F4A_PROFILE_EVIDENCE_SHA256,
        "runtime_fingerprint": gate.F4A_PROFILE_RUNTIME_FINGERPRINT,
        "passed": True,
    }
    return {
        "file_sha256": gate.F4A_PROFILE_FILE_SHA256,
        "value": value,
        "value_sha256": gate.sha256_json(value),
    }


def _stats(value: int = 0, *, fallback: int = 0) -> dict[str, int]:
    return {
        "offload_pages": value,
        "offload_bypassed_pages": 0,
        "restore_pages": value,
        "restore_attempts": value,
        "restore_fallbacks": fallback,
        "ownership_failures": 0,
    }


def _prompt(session: int, turn: int) -> list[int]:
    prompt = [1] * gate.SHARED_TOKENS
    for prior_turn in range(turn + 1):
        prompt.extend([2 + session * gate.TURNS + prior_turn] * gate.TURN_TOKENS)
    return prompt


def _trace(requests: list[dict[str, object]]) -> dict[str, object]:
    shared = requests[0]["prompt_token_ids"][: gate.SHARED_TOKENS]
    return gate.hashed_descriptor(
        {
            "kind": "canonical-agent-tool-token-trace-v1",
            "tokenizer_sha256": gate.sha256_json("tokenizer"),
            "shared_prefix_tokens": gate.SHARED_TOKENS,
            "turn_tokens": gate.TURN_TOKENS,
            "shared_prefix_sha256": gate.sha256_json(shared),
            "rows": gate._trace_rows_from_requests(requests),
        }
    )


def _request(sequence: int, *, arm: str, tpot_ns: int = 100) -> dict[str, object]:
    session, turn = gate._request_identity(sequence)
    prompt = _prompt(session, turn)
    initial_value = sequence if arm == "on" else 0
    visible_value = sequence + 1 if arm == "on" else 0
    initial = _stats(initial_value)
    visible = _stats(visible_value)
    output = list(range(1000, 1000 + gate.OUTPUT_TOKENS))
    events = []
    for output_count in range(1, gate.OUTPUT_TOKENS + 1):
        events.append(
            {
                "step_index": output_count - 1,
                "elapsed_ns": output_count * tpot_ns,
                "emitted_update": True,
                "output_token_ids": output[:output_count],
                "output_count": output_count,
                "finished": output_count == gate.OUTPUT_TOKENS,
                "finish_reason": ("length" if output_count == gate.OUTPUT_TOKENS else None),
                "error_type": None,
                "tier_stats": copy.deepcopy(visible),
                "num_free_pages": 6 if output_count <= 16 else 5,
            }
        )
    cached = len(prompt) // 2 + (32 if arm == "on" else 0)
    return {
        "row_type": "request",
        "sequence": sequence,
        "session": session,
        "turn": turn,
        "request_id": f"f4b-s{session:02d}-t{turn:02d}",
        "status": "success",
        "retry_count": 0,
        "prompt_tokens": len(prompt),
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": gate.sha256_json(prompt),
        "output_token_ids": output,
        "usage": {
            "prompt_tokens": len(prompt),
            "cached_tokens": cached,
            "completion_tokens": gate.OUTPUT_TOKENS,
        },
        "initial_tier_stats": initial,
        "initial_num_free_pages": 6,
        "events": events,
        "final_tier_stats": visible,
        "final_num_free_pages": 5,
    }


def _hardware(cohort: str) -> dict[str, object]:
    base = 0 if cohort == "A" else gate.TP_SIZE
    return {
        "cohort": cohort,
        "hostname": "fixture",
        "gpus": [
            {
                "rank": rank,
                "device_uuid": f"GPU-{base + rank:064x}",
                "pci_bus_id": f"0000:{base + rank:02x}:00.0",
                "name": "fixture-gpu",
                "torch_name": "fixture-gpu",
                "compute_capability": [12, 0],
                "total_memory_bytes": 1,
            }
            for rank in range(gate.TP_SIZE)
        ],
        "driver_version": "fixture-driver",
        "torch_version": "fixture-torch",
        "torch_cuda_version": "fixture-cuda",
        "nccl_version": "fixture-nccl",
    }


def _runtime(
    arm: str,
    source: dict[str, object],
    profile: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "tensor_parallel_size": gate.TP_SIZE,
        "page_size": gate.PAGE_SIZE,
        "num_pages": gate.HBM_PAGES,
        "max_num_batched_tokens": gate.MAX_NUM_BATCHED_TOKENS,
        "max_model_len": gate.MAX_MODEL_LEN,
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "engine_source_sha256": source["engine_source_sha256"],
        "attention_backend_decision": {
            "requested": "auto",
            "resolved": "flashinfer",
            "source": "hw_profile",
            "components": {
                "prefill": "flashinfer",
                "decode": "flashinfer",
                "kv_mode": "paged-direct",
            },
            "rationale": "fixture retained hardware profile",
            "architecture": {
                "arch": "cuda",
                "device_name": "fixture-gpu",
                "sm": 120,
                "kernel_tier": "fa2",
            },
        },
        "kv_cache_dtype_requested": "auto",
        "kv_cache_dtype_resolved": "bfloat16",
        "dram_kv_tier_enabled": arm == "on",
        "dram_kv_tier_capacity_pages": gate.DRAM_PAGES if arm == "on" else 0,
        "dram_kv_tier_profile_sha256": (profile["file_sha256"] if profile is not None else None),
        "dram_kv_tier_min_restore_tokens": 1024 if arm == "on" else None,
        "arm": arm,
    }


def _shard(identity: tuple[int, str, str], index: int) -> list[dict[str, object]]:
    round_index, arm, cohort = identity
    source = _source()
    profile = _profile(source) if arm == "on" else None
    requests = [_request(sequence, arm=arm) for sequence in range(gate.REQUESTS_PER_SHARD)]
    start_ns = 1_000_000_000 + index * 2_000_000
    run_id = f"fixture-r{round_index}-{arm}-{cohort}"
    container_id = str(index + 4) * 64
    output_path = f"/evidence/round{round_index}-{arm}-{cohort}.jsonl"
    repo_digest = (
        "kairyu-f4a@sha256:"
        + gate.F4A_CALIBRATED_IMAGE_ID.removeprefix("sha256:")
    )
    command = [
        gate.SOURCE_PATH,
        "run",
        "--round",
        str(round_index),
        "--arm",
        arm,
        "--cohort",
        cohort,
        "--model-path",
        "/models/qwen3-32b",
        "--output",
        output_path,
        "--image-id",
        gate.F4A_CALIBRATED_IMAGE_ID,
        "--container-id",
        container_id,
        "--repo-digest",
        repo_digest,
    ]
    if profile is not None:
        command.extend(("--profile", "/evidence/f4a-tp4-profile.json"))
    return [
        {
            "row_type": "run",
            "schema_version": gate.SCHEMA_VERSION,
            "measurement_kind": gate.MEASUREMENT_KIND,
            "run_id": run_id,
            "output_path": output_path,
            "round": round_index,
            "arm": arm,
            "cohort": cohort,
            "config": gate.formal_config(),
            "trace": _trace(requests),
            "source": source,
            "checkpoint": _checkpoint(),
            "profile": profile,
            "container": {
                "container_id": container_id,
                "image_id": gate.F4A_CALIBRATED_IMAGE_ID,
                "repo_digests": [repo_digest],
                "command": command,
                "container_id_binding": {
                    "option": "--container-id",
                    "argument": container_id,
                    "resolved_container_id": container_id,
                },
                "hostname": container_id[:12],
            },
            "hardware": _hardware(cohort),
            "runtime": _runtime(arm, source, profile),
            "measurement_started_at": "2026-08-01T00:00:00+00:00",
            "measurement_started_unix_ns": start_ns,
        },
        *requests,
        {
            "row_type": "run_end",
            "run_id": run_id,
            "round": round_index,
            "arm": arm,
            "cohort": cohort,
            "status": "complete",
            "errors": [],
            "request_count": gate.REQUESTS_PER_SHARD,
            "source": source,
            "final_tier_stats": requests[-1]["final_tier_stats"],
            "measurement_ended_at": "2026-08-01T00:00:00+00:00",
            "measurement_ended_unix_ns": start_ns + 1_000_000,
        },
    ]


def _quality_logprob_records(output: list[int]) -> list[dict[str, object]]:
    records = []
    for position, selected in enumerate(output):
        top = [{"token_id": selected, "logprob": -0.125}]
        top.extend(
            {
                "token_id": 200_000 + position * 100 + index,
                "logprob": -1.0 - index,
            }
            for index in range(gate.QUALITY_TOP_LOGPROBS - 1)
        )
        records.append(
            {
                "position": position,
                "selected_token_id": selected,
                "selected_logprob": -0.125,
                "top_logprobs": top,
            }
        )
    return records


def _quality_shard(
    identity: tuple[str, str],
    index: int,
    *,
    parent_raw_sha256: str,
) -> list[dict[str, object]]:
    arm, cohort = identity
    source = _quality_source()
    profile = _profile(source) if arm == "on" else None
    requests = []
    for sequence in range(gate.REQUESTS_PER_SHARD):
        request = _request(sequence, arm=arm)
        requests.append(
            {
                "row_type": "quality_request",
                **{
                    key: request[key]
                    for key in (
                        "sequence",
                        "session",
                        "turn",
                        "request_id",
                        "status",
                        "retry_count",
                        "prompt_tokens",
                        "prompt_token_ids",
                        "prompt_token_ids_sha256",
                        "output_token_ids",
                        "usage",
                        "initial_tier_stats",
                        "final_tier_stats",
                    )
                },
                "logprob_records": _quality_logprob_records(
                    request["output_token_ids"]
                ),
            }
        )
    run_id = f"fixture-quality-{arm}-{cohort}"
    container_id = str(index + 8) * 64
    output_path = f"/evidence/quality-{arm}.jsonl"
    repo_digest = (
        "kairyu-f4a@sha256:"
        + gate.F4A_CALIBRATED_IMAGE_ID.removeprefix("sha256:")
    )
    command = [
        gate.SOURCE_PATH,
        "quality-run",
        "--arm",
        arm,
        "--cohort",
        cohort,
        "--performance-raw-sha256",
        parent_raw_sha256,
        "--model-path",
        "/models/qwen3-32b",
        "--output",
        output_path,
        "--image-id",
        gate.F4A_CALIBRATED_IMAGE_ID,
        "--container-id",
        container_id,
        "--repo-digest",
        repo_digest,
    ]
    if profile is not None:
        command.extend(("--profile", "/evidence/f4a-tp4-profile.json"))
    started_ns = 2_000_000_000 + index * 2_000_000
    return [
        {
            "row_type": "quality_run",
            "schema_version": gate.QUALITY_SCHEMA_VERSION,
            "measurement_kind": gate.QUALITY_MEASUREMENT_KIND,
            "run_id": run_id,
            "output_path": output_path,
            "arm": arm,
            "cohort": cohort,
            "parent_performance_raw_sha256": parent_raw_sha256,
            "config": gate.quality_config(),
            "trace": _trace(requests),
            "source": source,
            "checkpoint": _checkpoint(),
            "profile": profile,
            "container": {
                "container_id": container_id,
                "image_id": gate.F4A_CALIBRATED_IMAGE_ID,
                "repo_digests": [repo_digest],
                "command": command,
                "container_id_binding": {
                    "option": "--container-id",
                    "argument": container_id,
                    "resolved_container_id": container_id,
                },
                "hostname": container_id[:12],
            },
            "hardware": _hardware(cohort),
            "runtime": _runtime(arm, source, profile),
            "measurement_started_at": "2026-08-01T00:00:00+00:00",
            "measurement_started_unix_ns": started_ns,
        },
        *requests,
        {
            "row_type": "quality_run_end",
            "run_id": run_id,
            "arm": arm,
            "cohort": cohort,
            "status": "complete",
            "errors": [],
            "request_count": gate.REQUESTS_PER_SHARD,
            "source": source,
            "final_tier_stats": requests[-1]["final_tier_stats"],
            "measurement_ended_at": "2026-08-01T00:00:00+00:00",
            "measurement_ended_unix_ns": started_ns + 1_000_000,
        },
    ]


def _timestamp(unix_ns: int) -> str:
    seconds, fraction = divmod(unix_ns, 1_000_000_000)
    prefix = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{prefix}.{fraction:09d}Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_lifecycle_metadata(
    metadata: Path,
    shards: dict[tuple[object, ...], list[dict[str, object]]],
    *,
    plan: tuple[tuple[object, ...], ...],
    labels: dict[tuple[object, ...], str],
) -> Path:
    first_header = shards[plan[0]][0]
    image_id = first_header["container"]["image_id"]
    _write_json(
        metadata / gate.IMAGE_INSPECT_NAME,
        [
            {
                "Id": image_id,
                "RepoDigests": first_header["container"]["repo_digests"],
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": gate.F4A_CALIBRATED_IMAGE_REVISION
                    }
                },
            }
        ],
    )
    for identity in plan:
        label = labels[identity]
        rows = shards[identity]
        header = rows[0]
        end = rows[-1]
        container = header["container"]
        container_id = container["container_id"]
        measurement_start = header["measurement_started_unix_ns"]
        measurement_end = end["measurement_ended_unix_ns"]
        created_ns = measurement_start - 500_000
        started_ns = measurement_start - 100_000
        finished_ns = measurement_end + 100_000
        config = {
            "Hostname": container_id[:12],
            "Image": image_id,
            "Entrypoint": ["/app/.venv/bin/python"],
            "Cmd": container["command"],
            "WorkingDir": "/workspace/kairyu",
            "User": "1003:1003",
            "Env": [
                "GLOO_SOCKET_IFNAME=lo",
                "TRITON_CACHE_DIR=/evidence/triton-cache",
                "FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer-workspace",
            ],
        }
        host_config = {
            "NetworkMode": "bridge",
            "OomKillDisable": False,
            "DeviceRequests": [
                {
                    "Driver": "",
                    "Count": 0,
                    "DeviceIDs": list(gate.COHORT_DEVICE_IDS[str(identity[-1])]),
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
        }
        static = {
            "Id": container_id,
            "Image": image_id,
            "Created": _timestamp(created_ns),
            "Path": "/app/.venv/bin/python",
            "Args": container["command"],
            "Config": config,
            "HostConfig": host_config,
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/fixture/source",
                    "Destination": "/workspace/kairyu",
                    "RW": False,
                },
                {
                    "Type": "volume",
                    "Source": "/fixture/model-volume",
                    "Destination": "/models/qwen3-32b",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": f"/fixture/evidence/{label}",
                    "Destination": "/evidence",
                    "RW": True,
                },
                {
                    "Type": "bind",
                    "Source": f"/fixture/metadata/{label}",
                    "Destination": "/run/kairyu-meta",
                    "RW": False,
                },
            ],
        }
        created = {
            **copy.deepcopy(static),
            "State": {
                "Status": "created",
                "Running": False,
                "ExitCode": 0,
                "OOMKilled": False,
                "Error": "",
                "StartedAt": gate._DOCKER_ZERO_TIMESTAMP,
                "FinishedAt": gate._DOCKER_ZERO_TIMESTAMP,
            },
        }
        exited = {
            **copy.deepcopy(static),
            "State": {
                "Status": "exited",
                "Running": False,
                "ExitCode": 0,
                "OOMKilled": False,
                "Error": "",
                "StartedAt": _timestamp(started_ns),
                "FinishedAt": _timestamp(finished_ns),
            },
        }
        exited["Mounts"].reverse()
        exited["HostConfig"]["OomKillDisable"] = None
        label_dir = metadata / label
        label_dir.mkdir(parents=True)
        (label_dir / gate.CONTAINER_ID_NAME).write_text(container_id + "\n", encoding="utf-8")
        _write_json(label_dir / gate.CONTAINER_CREATED_NAME, [created])
        _write_json(label_dir / gate.CONTAINER_EXITED_NAME, [exited])
    return metadata


def _write_fixture_inputs(root: Path) -> tuple[list[Path], Path]:
    paths: list[Path] = []
    shards: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for index, identity in enumerate(gate.SHARD_PLAN):
        rows = _shard(identity, index)
        path = root / f"shard-{index}.jsonl"
        gate.write_jsonl(path, rows)
        paths.append(path)
        shards[identity] = rows

    metadata = _write_lifecycle_metadata(
        root / "metadata",
        shards,
        plan=gate.SHARD_PLAN,
        labels=gate.SHARD_LABELS,
    )
    return paths, metadata


def _fixture_paths(root: Path) -> list[Path]:
    return [
        root / f"shard-{index}.jsonl"
        for index, _identity in enumerate(gate.SHARD_PLAN)
    ]


def _copy_fixture_inputs(
    template: Path,
    destination: Path,
) -> tuple[list[Path], Path]:
    shutil.copytree(template, destination)
    return _fixture_paths(destination), destination / "metadata"


def _copy_fixture_metadata(
    template: Path,
    destination: Path,
) -> tuple[list[Path], Path]:
    shutil.copytree(template / "metadata", destination)
    return _fixture_paths(template), destination


def _write_quality_inputs(
    performance_artifact: Path,
    root: Path,
) -> tuple[list[Path], Path]:
    parent_raw_sha256 = gate.sha256_file(
        performance_artifact / gate.RAW_NAME
    )
    paths = []
    shards: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for index, identity in enumerate(gate.QUALITY_PLAN):
        path = root / f"quality-{identity[0]}.jsonl"
        rows = _quality_shard(
            identity,
            index,
            parent_raw_sha256=parent_raw_sha256,
        )
        gate.write_jsonl(
            path,
            rows,
        )
        paths.append(path)
        shards[identity] = rows
    metadata = _write_lifecycle_metadata(
        root / "metadata",
        shards,
        plan=gate.QUALITY_PLAN,
        labels=gate.QUALITY_SHARD_LABELS,
    )
    return paths, metadata


def _set_performance_first_output(path: Path, token_id: int) -> None:
    rows = gate.read_jsonl(path)
    request = rows[1]
    request["output_token_ids"][0] = token_id
    for event in request["events"]:
        if event["output_token_ids"]:
            event["output_token_ids"][0] = token_id
    gate.write_jsonl(path, rows)


def _set_quality_first_distribution(
    path: Path,
    *,
    selected_token_id: int,
    reciprocal_token_id: int,
) -> None:
    rows = gate.read_jsonl(path)
    request = rows[1]
    request["output_token_ids"][0] = selected_token_id
    record = request["logprob_records"][0]
    record["selected_token_id"] = selected_token_id
    record["selected_logprob"] = -0.125
    retained = [
        candidate
        for candidate in record["top_logprobs"]
        if candidate["token_id"]
        not in {selected_token_id, reciprocal_token_id}
    ]
    retained.extend(
        (
            {"token_id": selected_token_id, "logprob": -0.125},
            {"token_id": reciprocal_token_id, "logprob": -0.2},
        )
    )
    retained.sort(
        key=lambda candidate: (-candidate["logprob"], candidate["token_id"])
    )
    record["top_logprobs"] = retained[: gate.QUALITY_TOP_LOGPROBS]
    gate.write_jsonl(path, rows)


@pytest.fixture(scope="module")
def performance_inputs_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("f4b-performance-inputs")
    _write_fixture_inputs(root)
    return root


@pytest.fixture(scope="module")
def passing_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    performance_inputs_template: Path,
):
    root = tmp_path_factory.mktemp("f4b-passing-artifact")
    paths = _fixture_paths(performance_inputs_template)
    metadata = performance_inputs_template / "metadata"
    output = root / "artifact"
    manifest = gate.assemble_artifact(
        paths,
        output,
        metadata_dir=metadata,
        assert_gate=True,
    )
    rows = gate.read_jsonl(output / gate.RAW_NAME)
    return output, rows, manifest


@pytest.fixture(scope="module")
def quality_inputs_template(
    tmp_path_factory: pytest.TempPathFactory,
    passing_artifact,
) -> Path:
    performance_artifact, _rows, _manifest_value = passing_artifact
    root = tmp_path_factory.mktemp("f4b-quality-inputs")
    _write_quality_inputs(performance_artifact, root)
    return root


@pytest.fixture(scope="module")
def passing_quality_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    passing_artifact,
    quality_inputs_template: Path,
) -> tuple[Path, dict[str, object], str]:
    performance_artifact, _rows, _manifest_value = passing_artifact
    quality_paths = [
        quality_inputs_template / f"quality-{identity[0]}.jsonl"
        for identity in gate.QUALITY_PLAN
    ]
    output = tmp_path_factory.mktemp("f4b-passing-quality") / "sealed"
    original_performance_sha = gate.sha256_file(
        performance_artifact / gate.RAW_NAME
    )
    sealed = gate.seal_quality_artifact(
        performance_artifact,
        quality_paths,
        output,
        quality_metadata_dir=quality_inputs_template / "metadata",
        assert_gate=True,
    )
    return output, sealed, original_performance_sha


def _copy_quality_inputs(
    template: Path,
    destination: Path,
) -> tuple[list[Path], Path]:
    shutil.copytree(template, destination)
    paths = [
        destination / f"quality-{identity[0]}.jsonl"
        for identity in gate.QUALITY_PLAN
    ]
    return paths, destination / "metadata"


def _quality_template_rows(template: Path, plan_index: int) -> list[dict]:
    identity = gate.QUALITY_PLAN[plan_index]
    return gate.read_jsonl(template / f"quality-{identity[0]}.jsonl")


def _request_indices(arm: str) -> list[int]:
    block = gate.REQUESTS_PER_SHARD + 2
    return [
        1 + shard_index * block + 1 + sequence
        for shard_index, identity in enumerate(gate.SHARD_PLAN)
        if identity[1] == arm
        for sequence in range(gate.REQUESTS_PER_SHARD)
    ]


def _shard_start_index(shard_index: int) -> int:
    return 1 + shard_index * (gate.REQUESTS_PER_SHARD + 2)


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    return gate.recompute_manifest(rows, raw_sha256="f" * 64)


def _inspect_value(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list) and len(value) == 1
    return value[0]


def _replace_inspect(path: Path, value: dict[str, object]) -> None:
    _write_json(path, [value])


def test_trace_builder_has_exact_page_aligned_agentic_geometry() -> None:
    class Tokenizer:
        @staticmethod
        def encode(_text: str) -> tuple[int, ...]:
            return (7, 11, 13)

    trace, requests = gate.build_trace(Tokenizer(), tokenizer_sha256="a" * 64)

    assert trace["sha256"] == gate.sha256_json(trace["value"])
    assert len(requests) == gate.REQUESTS_PER_SHARD
    assert all(row["prompt_tokens"] % gate.PAGE_SIZE == 0 for row in requests)
    assert requests[0]["prompt_tokens"] == 2560
    assert requests[-1]["prompt_tokens"] == 6144


def test_synchronous_driver_records_update_free_prefill_step() -> None:
    class Cache:
        dram_tier_stats = _stats()
        num_free_pages = 6

    class Loop:
        def __init__(self) -> None:
            self.index = -1

        def prepare_prompt(self, prompt, params):
            return (prompt, params)

        def submit(self, *_args, **_kwargs) -> None:
            self.index = 0

        def has_work(self) -> bool:
            return 0 <= self.index <= gate.OUTPUT_TOKENS

        def step(self):
            self.index += 1
            if self.index == 1:
                return []
            count = self.index - 1
            update = SimpleNamespace(
                outputs=tuple(range(count)),
                text="",
                finished=count == gate.OUTPUT_TOKENS,
                finish_reason=("length" if count == gate.OUTPUT_TOKENS else None),
                error=None,
                num_prompt_tokens=2560,
                num_cached_tokens=0,
            )
            return [("f4b-s00-t00", update)]

    backend = SimpleNamespace(_loop=Loop(), _cache=Cache())
    row = gate._run_one_request(
        backend,
        {
            "sequence": 0,
            "session": 0,
            "turn": 0,
            "prompt_token_ids": [1] * 2560,
        },
        SimpleNamespace(),
    )

    assert row["events"][0]["emitted_update"] is False
    assert row["events"][0]["output_count"] == 0
    assert row["events"][-1]["output_count"] == gate.OUTPUT_TOKENS


def test_passing_artifact_replays_and_verifies(passing_artifact) -> None:
    output, _rows, manifest = passing_artifact

    assert manifest["passed"] is True
    assert "same_arm_cross_cohort_output_reproducibility_exact" not in manifest[
        "checks"
    ]
    assert gate.replay_raw(output / gate.RAW_NAME, assert_gate=True) == manifest
    assert gate.verify_artifact(output, assert_gate=True) == manifest
    assert manifest["diagnostics"]["cache_gain_percentage_points_pooled"] > 0
    assert manifest["checks"]["container_lifecycle_metadata_exact_and_sequential"] is True
    assert len(manifest["container_metadata"]["file_sha256"]) == 13
    assert (output / gate.CONTAINER_METADATA_DIR / gate.IMAGE_INSPECT_NAME).is_file()
    assert manifest["bound_source"]["commit"] == "1" * 40
    assert (
        manifest["calibrated_execution_image"]["revision"]
        == gate.F4A_CALIBRATED_IMAGE_REVISION
    )
    assert (
        manifest["calibrated_execution_image"]["revision"]
        != manifest["bound_source"]["commit"]
    )
    assert (
        manifest["calibrated_execution_image"]["image_id"]
        == gate.F4A_CALIBRATED_IMAGE_ID
    )
    assert (
        manifest["checks"]["prompt_order_identity_exact_across_all_arms"]
        is True
    )
    assert manifest["diagnostics"]["free_running_output_equality_is_binding"] is False
    assert manifest["verdict_revision"] == gate.VERDICT_REVISION


def test_free_running_output_difference_is_retained_as_diagnostic(
    passing_artifact,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    on_indices = _request_indices("on")
    targets = (on_indices[0], on_indices[gate.REQUESTS_PER_SHARD])
    replacement_token = rows[targets[0]]["output_token_ids"][0] + 1
    for target in targets:
        row = rows[target]
        row["output_token_ids"][0] = replacement_token
        for event in row["events"]:
            if event["output_token_ids"]:
                event["output_token_ids"][0] = replacement_token

    manifest = _manifest(rows)

    assert manifest["passed"] is True
    assert (
        manifest["checks"]["prompt_order_identity_exact_across_all_arms"]
        is True
    )
    agreement = manifest["diagnostics"]["output_agreement"]
    assert agreement["tier_on_cross_cohort"]["exact_request_matches"] == 128
    assert (
        manifest["diagnostics"]["same_arm_cross_cohort_output_reproducibility_exact"]
        is True
    )
    assert agreement["tier_off_vs_on_cohort_b"]["exact_request_matches"] == 127
    assert agreement["tier_off_vs_on_cohort_b"]["first_divergences"] == [
        {
            "sequence": 0,
            "session": 0,
            "turn": 0,
            "position": 0,
            "left_token_id": 1000,
            "right_token_id": replacement_token,
        }
    ]


def test_quality_companion_seals_and_replays_without_changing_performance_raw(
    passing_quality_artifact,
) -> None:
    output, sealed, original_performance_sha = passing_quality_artifact

    assert sealed["passed"] is True
    assert gate.sha256_file(output / gate.RAW_NAME) == original_performance_sha
    assert gate.verify_quality_artifact(
        output,
        assert_gate=True,
    ) == sealed
    assert sealed["quality"]["diagnostics"][
        "aligned_prefix_position_count"
    ] == gate.REQUESTS_PER_SHARD * gate.OUTPUT_TOKENS
    assert sealed["quality"]["diagnostics"]["first_divergence_count"] == 0
    assert sealed["quality"]["checks"][
        "quality_container_lifecycle_metadata_exact_and_sequential"
    ] is True
    assert (
        output
        / gate.QUALITY_CONTAINER_METADATA_DIR
        / gate.IMAGE_INSPECT_NAME
    ).is_file()


def test_quality_first_divergence_passes_sealed_replay_with_reciprocal_top64(
    performance_inputs_template: Path,
    tmp_path: Path,
) -> None:
    performance_paths, performance_metadata = _copy_fixture_inputs(
        performance_inputs_template,
        tmp_path / "performance-inputs",
    )
    for path, identity in zip(
        performance_paths,
        gate.SHARD_PLAN,
        strict=True,
    ):
        if identity[1] == "on":
            _set_performance_first_output(path, 1001)
    performance_artifact = tmp_path / "performance-artifact"
    gate.assemble_artifact(
        performance_paths,
        performance_artifact,
        metadata_dir=performance_metadata,
        assert_gate=True,
    )
    quality_paths, quality_metadata = _write_quality_inputs(
        performance_artifact,
        tmp_path / "quality-inputs",
    )
    _set_quality_first_distribution(
        quality_paths[0],
        selected_token_id=1000,
        reciprocal_token_id=1001,
    )
    _set_quality_first_distribution(
        quality_paths[1],
        selected_token_id=1001,
        reciprocal_token_id=1000,
    )
    output = tmp_path / "sealed"

    sealed = gate.seal_quality_artifact(
        performance_artifact,
        quality_paths,
        output,
        quality_metadata_dir=quality_metadata,
        assert_gate=True,
    )

    replayed = gate.replay_quality_raw(
        output / gate.RAW_NAME,
        output / gate.QUALITY_RAW_NAME,
        assert_gate=True,
    )
    assert replayed == sealed["quality"]
    assert replayed["diagnostics"]["first_divergence_count"] == 1
    assert replayed["diagnostics"][
        "first_divergence_cross_selected_max_abs_delta"
    ] == pytest.approx(0.075)
    assert replayed["checks"][
        "first_divergence_requests_observe_tier_on_restore"
    ] is True
    assert replayed["diagnostics"][
        "first_divergence_tier_on_restore_observations"
    ] == [
        {
            "sequence": 0,
            "restore_attempts_delta": 1,
            "restore_pages_delta": 1,
            "restore_fallbacks_delta": 0,
            "ownership_failures_delta": 0,
        }
    ]


def test_quality_common_prefix_material_logprob_delta_fails_sealed_manifest(
    passing_artifact,
    quality_inputs_template: Path,
    tmp_path: Path,
) -> None:
    performance_artifact, _rows, _manifest_value = passing_artifact
    quality_paths, quality_metadata = _copy_quality_inputs(
        quality_inputs_template,
        tmp_path / "quality-inputs",
    )
    rows = gate.read_jsonl(quality_paths[1])
    record = rows[1]["logprob_records"][0]
    record["selected_logprob"] = -0.5
    selected_token = record["selected_token_id"]
    next(
        candidate
        for candidate in record["top_logprobs"]
        if candidate["token_id"] == selected_token
    )["logprob"] = -0.5
    record["top_logprobs"].sort(
        key=lambda candidate: (-candidate["logprob"], candidate["token_id"])
    )
    gate.write_jsonl(quality_paths[1], rows)
    output = tmp_path / "sealed"

    sealed = gate.seal_quality_artifact(
        performance_artifact,
        quality_paths,
        output,
        quality_metadata_dir=quality_metadata,
        assert_gate=False,
    )

    assert sealed["passed"] is False
    assert sealed["quality"]["checks"][
        "aligned_prefix_selected_logprob_abs_delta_at_most_0_25"
    ] is False
    assert gate.replay_quality_raw(
        output / gate.RAW_NAME,
        output / gate.QUALITY_RAW_NAME,
    ) == sealed["quality"]
    with pytest.raises(gate.EvidenceError, match="quality"):
        gate.verify_quality_artifact(output, assert_gate=True)


def test_quality_tier_stats_parent_binding_rejects_continuous_tamper(
    passing_artifact,
    quality_inputs_template: Path,
    tmp_path: Path,
) -> None:
    performance_artifact, _rows, _manifest_value = passing_artifact
    quality_paths, quality_metadata = _copy_quality_inputs(
        quality_inputs_template,
        tmp_path / "quality-inputs",
    )
    rows = gate.read_jsonl(quality_paths[1])
    for sequence, request in enumerate(rows[1:-1]):
        if sequence:
            request["initial_tier_stats"]["offload_pages"] += 1
        request["final_tier_stats"]["offload_pages"] += 1
    rows[-1]["final_tier_stats"]["offload_pages"] += 1
    gate.write_jsonl(quality_paths[1], rows)

    sealed = gate.seal_quality_artifact(
        performance_artifact,
        quality_paths,
        tmp_path / "sealed",
        quality_metadata_dir=quality_metadata,
        assert_gate=False,
    )

    assert sealed["passed"] is False
    assert sealed["quality"]["checks"][
        "quality_outputs_cache_and_tier_stats_equal_parent_arms"
    ] is False


@pytest.mark.parametrize("failure", ("discontinuous", "decreased"))
def test_quality_tier_stats_invalid_sequence_is_rejected_during_parse(
    failure: str,
    quality_inputs_template: Path,
) -> None:
    rows = _quality_template_rows(quality_inputs_template, 1)
    if failure == "discontinuous":
        rows[2]["initial_tier_stats"]["offload_pages"] += 1
    else:
        rows[2]["final_tier_stats"]["offload_pages"] = 0

    with pytest.raises(gate.EvidenceError, match=failure):
        gate.parse_quality_shard(rows)


def test_quality_command_accepts_absolute_source_path(
    quality_inputs_template: Path,
) -> None:
    rows = _quality_template_rows(quality_inputs_template, 0)
    rows[0]["container"]["command"][0] = (
        "/workspace/kairyu/" + gate.SOURCE_PATH
    )

    assert gate.parse_quality_shard(rows)["identity"] == gate.QUALITY_PLAN[0]


def test_quality_lifecycle_exit_failure_is_rejected_before_seal(
    passing_artifact,
    quality_inputs_template: Path,
    tmp_path: Path,
) -> None:
    performance_artifact, _rows, _manifest_value = passing_artifact
    quality_paths, quality_metadata = _copy_quality_inputs(
        quality_inputs_template,
        tmp_path / "quality-inputs",
    )
    exited_path = (
        quality_metadata
        / gate.QUALITY_SHARD_LABELS[gate.QUALITY_PLAN[0]]
        / gate.CONTAINER_EXITED_NAME
    )
    exited = _inspect_value(exited_path)
    exited["State"]["ExitCode"] = 1
    _replace_inspect(exited_path, exited)

    with pytest.raises(gate.EvidenceError, match="clean exit"):
        gate.seal_quality_artifact(
            performance_artifact,
            quality_paths,
            tmp_path / "sealed",
            quality_metadata_dir=quality_metadata,
        )


def test_verify_quality_detects_retained_metadata_byte_tamper(
    passing_quality_artifact,
    tmp_path: Path,
) -> None:
    passing_output, _sealed, _performance_sha = passing_quality_artifact
    output = tmp_path / "sealed"
    shutil.copytree(passing_output, output)
    identifier = (
        output
        / gate.QUALITY_CONTAINER_METADATA_DIR
        / gate.QUALITY_SHARD_LABELS[gate.QUALITY_PLAN[0]]
        / gate.CONTAINER_ID_NAME
    )
    identifier.write_text(
        identifier.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(gate.EvidenceError, match="quality container metadata"):
        gate.verify_quality_artifact(output)


def test_verify_quality_rejects_duplicate_manifest_keys(
    passing_quality_artifact,
    tmp_path: Path,
) -> None:
    passing_output, _sealed, _performance_sha = passing_quality_artifact
    output = tmp_path / "sealed"
    shutil.copytree(passing_output, output)
    manifest_path = output / gate.QUALITY_MANIFEST_NAME
    retained = manifest_path.read_text(encoding="utf-8")
    assert retained.startswith("{")
    manifest_path.write_text(
        '{"passed":false,' + retained[1:],
        encoding="utf-8",
    )

    with pytest.raises(gate.EvidenceError, match="duplicate JSON key 'passed'"):
        gate.verify_quality_artifact(output)


def _one_position_quality_request(
    selected_token: int,
    selected_logprob: float,
    other_token: int,
    other_logprob: float,
) -> dict[str, object]:
    top = [
        {"token_id": selected_token, "logprob": selected_logprob},
        {"token_id": other_token, "logprob": other_logprob},
    ]
    top.extend(
        {
            "token_id": 10_000 + index,
            "logprob": -10.0 - index,
        }
        for index in range(gate.QUALITY_TOP_LOGPROBS - 2)
    )
    top.sort(key=lambda candidate: (-candidate["logprob"], candidate["token_id"]))
    return {
        "sequence": 0,
        "session": 0,
        "turn": 0,
        "output_token_ids": [selected_token],
        "logprob_records": [
            {
                "selected_logprob": selected_logprob,
                "top_logprobs": top,
            }
        ],
    }


def test_quality_first_divergence_uses_reciprocal_selected_logprobs() -> None:
    off = _one_position_quality_request(11, -1.0, 13, -1.2)
    on = _one_position_quality_request(13, -1.1, 11, -1.125)

    checks, diagnostics = gate._quality_distribution_comparison(
        [off],
        [on],
    )

    assert all(checks.values())
    assert diagnostics["first_divergence_count"] == 1
    assert diagnostics[
        "first_divergence_cross_selected_comparison_count"
    ] == 2
    assert diagnostics[
        "first_divergence_cross_selected_max_abs_delta"
    ] == pytest.approx(0.125)


@pytest.mark.parametrize("failure", ("delta", "missing"))
def test_quality_first_divergence_fails_material_or_missing_cross_view(
    failure: str,
) -> None:
    off = _one_position_quality_request(11, -1.0, 13, -1.2)
    on = _one_position_quality_request(13, -1.1, 11, -1.125)
    if failure == "delta":
        next(
            candidate
            for candidate in on["logprob_records"][0]["top_logprobs"]
            if candidate["token_id"] == 11
        )["logprob"] = -2.0
    else:
        on["logprob_records"][0]["top_logprobs"] = [
            candidate
            for candidate in on["logprob_records"][0]["top_logprobs"]
            if candidate["token_id"] != 11
        ]

    checks, _diagnostics = gate._quality_distribution_comparison(
        [off],
        [on],
    )

    if failure == "delta":
        assert (
            checks[
                "first_divergence_reciprocal_cross_selected_abs_delta_at_most_0_25"
            ]
            is False
        )
    else:
        assert (
            checks[
                "first_divergence_reciprocal_cross_selected_top64_complete"
            ]
            is False
        )


def test_same_arm_output_difference_is_nonbinding_artifact_diagnostic(
    passing_artifact,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    target = _request_indices("on")[0]
    row = rows[target]
    replacement_token = row["output_token_ids"][0] + 1
    row["output_token_ids"][0] = replacement_token
    for event in row["events"]:
        if event["output_token_ids"]:
            event["output_token_ids"][0] = replacement_token

    manifest = _manifest(rows)

    assert manifest["passed"] is True
    assert (
        manifest["diagnostics"]["same_arm_cross_cohort_output_reproducibility_exact"]
        is False
    )
    assert (
        manifest["diagnostics"][
            "same_arm_cross_cohort_output_reproducibility_is_binding"
        ]
        is False
    )


def test_cross_arm_prompt_difference_remains_binding(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    shard_index = 1
    start_index = _shard_start_index(shard_index)
    request_start = start_index + 1
    requests = rows[
        request_start : request_start + gate.REQUESTS_PER_SHARD
    ]
    for request in requests:
        request["prompt_token_ids"][0] = 99
        request["prompt_token_ids_sha256"] = gate.sha256_json(
            request["prompt_token_ids"]
        )
    rows[start_index]["trace"] = _trace(requests)

    manifest = _manifest(rows)

    assert manifest["passed"] is False
    assert (
        manifest["checks"]["prompt_order_identity_exact_across_all_arms"]
        is False
    )


def test_retained_f4a_docker_host_config_has_only_known_false_to_null_drift() -> None:
    root = (
        gate._REPO_ROOT
        / "bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01"
        / "container-metadata/tp4"
    )
    created = _inspect_value(root / gate.CONTAINER_CREATED_NAME)
    exited = _inspect_value(root / gate.CONTAINER_EXITED_NAME)
    created_host = created["HostConfig"]
    exited_host = exited["HostConfig"]
    differences = {
        key
        for key in set(created_host) | set(exited_host)
        if created_host.get(key) != exited_host.get(key)
    }

    assert differences == {"OomKillDisable"}
    assert gate._normalized_host_config(created_host, "created") == gate._normalized_host_config(
        exited_host, "exited"
    )


@pytest.mark.parametrize(
    "difference",
    (
        "image_revision_wrong",
        "image_revision_malformed",
        "image_revision_missing",
        "image_id",
        "container_id",
        "command",
        "hostname",
        "device_requests",
        "host_config_drift",
    ),
)
def test_container_metadata_identity_or_configuration_difference_fails_closed(
    performance_inputs_template: Path,
    tmp_path: Path,
    difference: str,
) -> None:
    paths, metadata = _copy_fixture_metadata(
        performance_inputs_template,
        tmp_path / "metadata",
    )
    label_dir = metadata / gate.SHARD_LABELS[gate.SHARD_PLAN[0]]
    if difference.startswith("image_revision"):
        path = metadata / gate.IMAGE_INSPECT_NAME
        image = _inspect_value(path)
        labels = image["Config"]["Labels"]
        if difference == "image_revision_wrong":
            labels["org.opencontainers.image.revision"] = "2" * 40
        elif difference == "image_revision_malformed":
            labels["org.opencontainers.image.revision"] = "not-a-commit"
        else:
            del labels["org.opencontainers.image.revision"]
        _replace_inspect(path, image)
    elif difference == "image_id":
        path = metadata / gate.IMAGE_INSPECT_NAME
        image = _inspect_value(path)
        image["Id"] = f"sha256:{'b' * 64}"
        _replace_inspect(path, image)
    elif difference == "container_id":
        path = label_dir / gate.CONTAINER_CREATED_NAME
        created = _inspect_value(path)
        created["Id"] = "9" * 64
        _replace_inspect(path, created)
    elif difference == "host_config_drift":
        path = label_dir / gate.CONTAINER_EXITED_NAME
        exited = _inspect_value(path)
        exited["HostConfig"]["NetworkMode"] = "none"
        _replace_inspect(path, exited)
    else:
        for name in (gate.CONTAINER_CREATED_NAME, gate.CONTAINER_EXITED_NAME):
            path = label_dir / name
            inspect = _inspect_value(path)
            if difference == "command":
                command = inspect["Args"]
                command[command.index("--arm") + 1] = "on"
                inspect["Config"]["Cmd"] = command
            elif difference == "hostname":
                inspect["Config"]["Hostname"] = "not-the-container-id"
            else:
                inspect["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["4", "5", "6", "7"]
            _replace_inspect(path, inspect)

    with pytest.raises(gate.EvidenceError, match="Docker"):
        gate.assemble_artifact(
            paths,
            tmp_path / "artifact",
            metadata_dir=metadata,
        )


@pytest.mark.parametrize(
    "difference",
    (
        "working_dir",
        "root_user",
        "source_rw",
        "model_type",
        "missing_meta",
        "source_drift",
        "shared_evidence",
        "shared_metadata",
        "triton_env",
        "flashinfer_env",
        "duplicate_triton_env",
    ),
)
def test_container_mount_or_execution_carrier_difference_fails_closed(
    performance_inputs_template: Path,
    tmp_path: Path,
    difference: str,
) -> None:
    paths, metadata = _copy_fixture_metadata(
        performance_inputs_template,
        tmp_path / "metadata",
    )
    one_shard_differences = {"source_drift", "shared_evidence", "shared_metadata"}
    identities = (
        (gate.SHARD_PLAN[1],) if difference in one_shard_differences else gate.SHARD_PLAN
    )
    for identity in identities:
        label = gate.SHARD_LABELS[identity]
        for name in (gate.CONTAINER_CREATED_NAME, gate.CONTAINER_EXITED_NAME):
            path = metadata / label / name
            inspect = _inspect_value(path)
            if difference == "working_dir":
                inspect["Config"]["WorkingDir"] = "/app"
            elif difference == "root_user":
                inspect["Config"]["User"] = "0:0"
            elif difference in {"triton_env", "flashinfer_env"}:
                name = (
                    "TRITON_CACHE_DIR"
                    if difference == "triton_env"
                    else "FLASHINFER_WORKSPACE_BASE"
                )
                environment = inspect["Config"]["Env"]
                entry_index = next(
                    index
                    for index, entry in enumerate(environment)
                    if entry.startswith(f"{name}=")
                )
                environment[entry_index] = f"{name}=/shared-cache"
            elif difference == "duplicate_triton_env":
                inspect["Config"]["Env"].append("TRITON_CACHE_DIR=/shared-cache")
            else:
                mounts = {
                    mount["Destination"]: mount for mount in inspect["Mounts"]
                }
                if difference == "source_rw":
                    mounts["/workspace/kairyu"]["RW"] = True
                elif difference == "model_type":
                    mounts["/models/qwen3-32b"]["Type"] = "bind"
                elif difference == "missing_meta":
                    inspect["Mounts"].remove(mounts["/run/kairyu-meta"])
                elif difference == "shared_evidence":
                    mounts["/evidence"]["Source"] = "/fixture/evidence/round0-off"
                elif difference == "shared_metadata":
                    mounts["/run/kairyu-meta"]["Source"] = "/fixture/metadata/round0-off"
                else:
                    mounts["/workspace/kairyu"]["Source"] = "/fixture/other-source"
            _replace_inspect(path, inspect)

    with pytest.raises(gate.EvidenceError, match="Docker"):
        gate.assemble_artifact(
            paths,
            tmp_path / "artifact",
            metadata_dir=metadata,
        )


@pytest.mark.parametrize(
    "difference",
    ("exit_code", "running", "measurement_outside", "lifecycle_overlap"),
)
def test_container_metadata_lifecycle_failure_is_rejected(
    performance_inputs_template: Path,
    tmp_path: Path,
    difference: str,
) -> None:
    paths, metadata = _copy_fixture_metadata(
        performance_inputs_template,
        tmp_path / "metadata",
    )
    first_label = gate.SHARD_LABELS[gate.SHARD_PLAN[0]]
    if difference == "lifecycle_overlap":
        label = gate.SHARD_LABELS[gate.SHARD_PLAN[1]]
        label_dir = metadata / label
        created_at = _timestamp(1_001_000_000)
        started_at = _timestamp(1_001_050_000)
        for name in (gate.CONTAINER_CREATED_NAME, gate.CONTAINER_EXITED_NAME):
            path = label_dir / name
            inspect = _inspect_value(path)
            inspect["Created"] = created_at
            if name == gate.CONTAINER_EXITED_NAME:
                inspect["State"]["StartedAt"] = started_at
            _replace_inspect(path, inspect)
    else:
        path = metadata / first_label / gate.CONTAINER_EXITED_NAME
        exited = _inspect_value(path)
        if difference == "exit_code":
            exited["State"]["ExitCode"] = 1
        elif difference == "running":
            exited["State"]["Running"] = True
        else:
            exited["State"]["FinishedAt"] = _timestamp(1_000_999_999)
        _replace_inspect(path, exited)

    with pytest.raises(gate.EvidenceError, match="Docker"):
        gate.assemble_artifact(
            paths,
            tmp_path / "artifact",
            metadata_dir=metadata,
        )


@pytest.mark.parametrize("missing", ("directory", "file"))
def test_assemble_requires_complete_container_metadata(
    performance_inputs_template: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    paths, metadata = _copy_fixture_metadata(
        performance_inputs_template,
        tmp_path / "metadata",
    )
    if missing == "directory":
        metadata = tmp_path / "missing-metadata"
    else:
        (metadata / gate.SHARD_LABELS[gate.SHARD_PLAN[0]] / gate.CONTAINER_EXITED_NAME).unlink()

    with pytest.raises(gate.EvidenceError, match="container metadata"):
        gate.assemble_artifact(
            paths,
            tmp_path / "artifact",
            metadata_dir=metadata,
        )


def test_equal_cache_rate_fails_pooled_and_matched_cohorts(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    for index in _request_indices("on"):
        row = dict(rows[index])
        usage = dict(row["usage"])
        usage["cached_tokens"] = usage["prompt_tokens"] // 2
        row["usage"] = usage
        rows[index] = row

    manifest = _manifest(rows)

    assert manifest["passed"] is False
    assert (
        manifest["checks"]["tier_on_prefix_hit_rate_strictly_higher_pooled_and_per_cohort"] is False
    )


def test_tpot_regression_fails_pooled_and_geometric_mean(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    for index in _request_indices("on"):
        row = dict(rows[index])
        events = []
        for event in row["events"]:
            replacement = dict(event)
            replacement["elapsed_ns"] = event["output_count"] * 120
            events.append(replacement)
        row["events"] = events
        rows[index] = row

    manifest = _manifest(rows)

    assert manifest["checks"]["pooled_tpot_p99_ratio_at_most_1_10"] is False
    assert manifest["checks"]["cohort_tpot_p99_ratio_geometric_mean_at_most_1_10"] is False


def test_tier_failure_counter_fails_activity_gate(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    target = _request_indices("on")[-1]
    row = copy.deepcopy(rows[target])
    for event in row["events"]:
        event["tier_stats"]["restore_fallbacks"] = 1
    row["final_tier_stats"]["restore_fallbacks"] = 1
    rows[target] = row
    # Keep the enclosing shard-end cumulative counter exact.
    rows[target + 1] = {
        **rows[target + 1],
        "final_tier_stats": row["final_tier_stats"],
    }

    manifest = _manifest(rows)

    assert manifest["checks"]["tier_activity_nonzero_without_restore_or_ownership_failure"] is False


def test_post_content_transfer_fails_critical_path_gate(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    target = _request_indices("on")[0]
    row = copy.deepcopy(rows[target])
    for event in row["events"][17:]:
        event["tier_stats"]["offload_pages"] += 1
    row["final_tier_stats"] = copy.deepcopy(row["events"][-1]["tier_stats"])
    rows[target] = row

    manifest = _manifest(rows)

    assert (
        manifest["checks"][
            "tier_transfers_absent_after_first_content_with_decode_allocation_proven"
        ]
        is False
    )


def test_missing_output17_page_drop_fails_nonvacuous_gate(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    target = _request_indices("on")[0]
    row = copy.deepcopy(rows[target])
    for event in row["events"][16:]:
        event["num_free_pages"] = 6
    row["final_num_free_pages"] = 6
    rows[target] = row

    manifest = _manifest(rows)

    assert (
        manifest["checks"][
            "tier_transfers_absent_after_first_content_with_decode_allocation_proven"
        ]
        is False
    )


def test_off_shard_rejects_a_profile(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    first_shard_start = dict(rows[1])
    first_shard_start["profile"] = copy.deepcopy(rows[1 + (gate.REQUESTS_PER_SHARD + 2)]["profile"])
    rows[1] = first_shard_start

    with pytest.raises(gate.EvidenceError, match="tier-off shard"):
        _manifest(rows)


def test_verify_detects_raw_tamper(
    passing_artifact,
    tmp_path: Path,
) -> None:
    output, _rows, _manifest_value = passing_artifact
    copied = tmp_path / "artifact"
    shutil.copytree(output, copied)
    raw = (copied / gate.RAW_NAME).read_text(encoding="utf-8")
    (copied / gate.RAW_NAME).write_text(
        raw.replace('"status":"success"', '"status":"failed"', 1),
        encoding="utf-8",
    )

    with pytest.raises(gate.EvidenceError):
        gate.verify_artifact(copied)


def test_verify_detects_retained_container_metadata_byte_tamper(
    passing_artifact,
    tmp_path: Path,
) -> None:
    output, _rows, _manifest_value = passing_artifact
    copied = tmp_path / "artifact"
    shutil.copytree(output, copied)
    identifier = (
        copied
        / gate.CONTAINER_METADATA_DIR
        / gate.SHARD_LABELS[gate.SHARD_PLAN[0]]
        / gate.CONTAINER_ID_NAME
    )
    identifier.write_text(identifier.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(gate.EvidenceError, match="retained container metadata"):
        gate.verify_artifact(copied)


@pytest.mark.parametrize("difference", ("descriptor_hash", "semantic_exit"))
def test_replay_revalidates_embedded_container_metadata(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    descriptor = rows[0]["container_lifecycle"]["shards"][0]["exited_inspect"]
    if difference == "descriptor_hash":
        descriptor["value_sha256"] = "0" * 64
    else:
        descriptor["value"][0]["State"]["ExitCode"] = 1
        descriptor["value_sha256"] = gate.sha256_json(descriptor["value"])

    with pytest.raises(gate.EvidenceError, match="metadata|Docker"):
        _manifest(rows)


def test_schema_exposes_all_offline_commands_and_fixed_capacity() -> None:
    schema = gate.schema_summary()

    assert schema["commands"] == [
        "run",
        "assemble",
        "quality-run",
        "seal-quality",
        "verify",
        "verify-quality",
        "replay",
        "replay-quality",
        "schema",
    ]
    assert schema["formal_config"]["hbm_pages"] == 1024
    assert schema["formal_config"]["dram_pages_when_enabled"] == 2048
    assert f"{gate.CONTAINER_METADATA_DIR}/" in schema["retained_files"]
    assert f"{gate.QUALITY_CONTAINER_METADATA_DIR}/" in schema["retained_files"]
    assert json.loads(gate.canonical_json(schema)) == schema


def test_assemble_cli_requires_metadata_dir() -> None:
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args(
            [
                "assemble",
                "--raw",
                "one.jsonl",
                "--output-dir",
                "artifact",
            ]
        )


def test_seal_quality_cli_requires_quality_metadata_dir() -> None:
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args(
            [
                "seal-quality",
                "--performance-artifact",
                "performance",
                "--quality-raw",
                "quality-off.jsonl",
                "--quality-raw",
                "quality-on.jsonl",
                "--output-dir",
                "artifact",
            ]
        )


@pytest.mark.parametrize("difference", ("attention", "kv_request"))
def test_valid_runtime_identity_difference_fails_cross_arm_gate(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(1)
    start = copy.deepcopy(rows[index])
    if difference == "attention":
        start["runtime"]["attention_backend_decision"]["rationale"] = (
            "different but individually valid rationale"
        )
    else:
        start["runtime"]["kv_cache_dtype_requested"] = "bfloat16"
    rows[index] = start

    manifest = _manifest(rows)

    assert manifest["checks"]["runtime_identity_equal_across_all_arms"] is False


def test_bound_source_commit_drift_fails_cross_arm_gate(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    start_index = _shard_start_index(1)
    end_index = start_index + gate.REQUESTS_PER_SHARD + 1
    for index in (start_index, end_index):
        rows[index]["source"]["commit"] = "3" * 40

    manifest = _manifest(rows)

    assert manifest["checks"]["bound_source_identity_exact_across_all_arms"] is False
    assert manifest["passed"] is False


@pytest.mark.parametrize("difference", ("image_id", "repo_digest"))
def test_execution_image_drift_is_rejected_across_arms(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = copy.deepcopy(original)
    start = rows[_shard_start_index(1)]
    command = start["container"]["command"]
    if difference == "image_id":
        value = f"sha256:{'b' * 64}"
        start["container"]["image_id"] = value
        command[command.index("--image-id") + 1] = value
    else:
        value = f"other@sha256:{'b' * 64}"
        start["container"]["repo_digests"] = [value]
        command[command.index("--repo-digest") + 1] = value

    with pytest.raises(gate.EvidenceError, match="image|Docker"):
        _manifest(rows)


@pytest.mark.parametrize(
    ("field", "value"),
    (("kv_cache_dtype_requested", ""), ("attention_backend_decision", {})),
)
def test_invalid_runtime_identity_is_rejected(
    passing_artifact,
    field: str,
    value: object,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(1)
    start = copy.deepcopy(rows[index])
    start["runtime"][field] = value
    rows[index] = start

    with pytest.raises(gate.EvidenceError, match="runtime"):
        _manifest(rows)


@pytest.mark.parametrize("difference", ("driver", "gpu_specification"))
def test_hardware_environment_or_gpu_specification_difference_fails_gate(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(1)
    start = copy.deepcopy(rows[index])
    if difference == "driver":
        start["hardware"]["driver_version"] = "different-valid-driver"
    else:
        for gpu in start["hardware"]["gpus"]:
            gpu["total_memory_bytes"] = 2
    rows[index] = start

    manifest = _manifest(rows)

    assert (
        manifest["checks"]["hardware_environment_and_gpu_specification_equal_across_all_arms"]
        is False
    )


def test_same_cohort_requires_every_gpu_row_to_match(passing_artifact) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(2)  # round1 tier-on cohort A
    start = copy.deepcopy(rows[index])
    start["hardware"]["gpus"][0]["pci_bus_id"] = "0000:ff:00.0"
    rows[index] = start

    manifest = _manifest(rows)

    assert manifest["checks"]["cohort_swap_and_disjoint_gpu_assignment_exact"] is False


@pytest.mark.parametrize("difference", ("hostname", "command_arm"))
def test_container_hostname_and_command_are_bound_to_raw_identity(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(0)
    start = copy.deepcopy(rows[index])
    if difference == "hostname":
        start["container"]["hostname"] = "not-a-container-prefix"
    else:
        command = start["container"]["command"]
        command[command.index("--arm") + 1] = "on"
    rows[index] = start

    with pytest.raises(gate.EvidenceError, match="container"):
        _manifest(rows)


@pytest.mark.parametrize(
    "difference",
    ("file_sha256", "evidence_sha256", "runtime_fingerprint", "engine_source_sha256"),
)
def test_retained_f4a_profile_exact_identity_is_pinned(
    passing_artifact,
    difference: str,
) -> None:
    _output, original, _manifest_value = passing_artifact
    rows = list(original)
    index = _shard_start_index(1)
    start = copy.deepcopy(rows[index])
    profile = start["profile"]
    if difference == "file_sha256":
        profile["file_sha256"] = "0" * 64
    elif difference == "engine_source_sha256":
        profile["value"]["runtime_identity"][difference] = "0" * 64
        profile["value_sha256"] = gate.sha256_json(profile["value"])
    else:
        profile["value"][difference] = "0" * 64
        profile["value_sha256"] = gate.sha256_json(profile["value"])
    rows[index] = start

    with pytest.raises(gate.EvidenceError, match="profile"):
        _manifest(rows)
