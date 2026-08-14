"""Formal Qwen3-32B TP8 evidence for issue #229 decode page-table caching.

The verdict is structural and parity based.  Wall-clock and CUDA-event timing
are retained for diagnosis, but never decide whether the cache is correct or
worth retaining.

Measure:

    uv run python verification/l1/performance/decode_page_table_cache_qwen.py \
      --model-path /models/qwen3-32b \
      --output bench/results/issue-229-page-table-cache-qwen3-32b-tp8.json \
      --assert-gate

Replay:

    uv run python verification/l1/performance/decode_page_table_cache_qwen.py \
      --verify bench/results/issue-229-page-table-cache-qwen3-32b-tp8.json \
      --model-path /models/qwen3-32b --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from verification.l1.performance import batched_prefill_qwen as _provenance
except ModuleNotFoundError:  # direct ``python bench/...py`` execution
    import batched_prefill_qwen as _provenance

SCHEMA_VERSION = 1
MEASUREMENT_KIND = "qwen3-32b-tp8-decode-page-table-cache"
TP = 8
NUM_PAGES = 4096
PAGE_SIZE = 16
REQUEST_COUNT = 8
PROMPT_LENGTHS = (256, 258, 260, 262, 264, 266, 268, 270)
MAX_NEW_TOKENS = 32
DECODE_STEPS = MAX_NEW_TOKENS - 1
DECODE_TOKENS = REQUEST_COUNT * DECODE_STEPS
VOCAB_SIZE = 151936
GRAPH_MAX_BATCH = 8
GRAPH_MAX_PAGES = 64
FORMAL_ROUNDS = 2

# These exact counts follow from Scheduler's one-token-ahead capacity reserve.
# At decode position p, a row exposes ceil((prompt + p + 1) / page_size) pages.
EXPECTED_BUILDS = 31
EXPECTED_ROWS = 248
EXPECTED_OWNED_ELEMENTS = 4456
EXPECTED_LEGACY_RECT_ELEMENTS = 4568
EXPECTED_LEGACY_ELEMENTS_WRITTEN = 9024
EXPECTED_GRAPH_FULL_ELEMENTS = 15872
EXPECTED_CACHE_ROWS_UPLOADED = 37
EXPECTED_CACHE_ELEMENTS_UPLOADED = 165
EXPECTED_CACHE_ROW_HITS = 211
EXPECTED_GRAPH_ROWS_COPIED = 23
EXPECTED_GRAPH_ELEMENTS_COPIED = 527
EXPECTED_GRAPH_ROW_HITS = 225

IMPLEMENTATION_FILES = (
    "verification/l1/performance/batched_prefill_qwen.py",
    "verification/l1/performance/decode_page_table_cache_qwen.py",
    "evals/profiling.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/engine/core/spec_runner.py",
    "kairyu/engine/core/step_executor.py",
    "kairyu/engine/core/tp_runner.py",
    "kairyu/engine/core/worker.py",
)
_REPO = Path(__file__).resolve().parents[3]

_checkpoint_manifest = _provenance._checkpoint_manifest
_checkpoint_snapshot_gate = _provenance._checkpoint_snapshot_gate
_checkpoint_gate = _provenance._checkpoint_gate
_hardware = _provenance._hardware
_hardware_snapshot_gate = _provenance._hardware_gate
_prompt_tokens = _provenance._prompt_tokens
_topology_snapshot_gate = _provenance._topology_gate


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_show(commit: str, name: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ("git", "show", f"{commit}:{name}"),
            cwd=_REPO,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _hex_digest(value: object, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_snapshot() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    files: dict[str, str] = {}
    files_match_commit = isinstance(commit, str)
    if isinstance(commit, str):
        for name in IMPLEMENTATION_FILES:
            current_path = _REPO / name
            committed = _git_show(commit, name)
            if committed is None or not current_path.is_file():
                files_match_commit = False
                continue
            files[name] = hashlib.sha256(committed).hexdigest()
            files_match_commit &= current_path.read_bytes() == committed
    return {
        "commit": commit,
        "tracked_dirty": None if dirty is None else bool(dirty),
        "files": files,
        "files_match_commit": (files_match_commit and set(files) == set(IMPLEMENTATION_FILES)),
    }


def _source_snapshot_gate(snapshot: object) -> bool:
    if not isinstance(snapshot, dict):
        return False
    commit = snapshot.get("commit")
    files = snapshot.get("files")
    if (
        not _hex_digest(commit, length=40)
        or snapshot.get("tracked_dirty") is not False
        or snapshot.get("files_match_commit") is not True
        or not isinstance(files, dict)
        or set(files) != set(IMPLEMENTATION_FILES)
        or any(not _hex_digest(digest) for digest in files.values())
    ):
        return False
    for name in IMPLEMENTATION_FILES:
        committed = _git_show(commit, name)
        path = _REPO / name
        if (
            committed is None
            or not path.is_file()
            or hashlib.sha256(committed).hexdigest() != files[name]
            or path.read_bytes() != committed
        ):
            return False
    return True


def _stable_pair_gate(
    pair: object,
    snapshot_gate,
) -> bool:
    if not isinstance(pair, dict):
        return False
    start = pair.get("measurement_start")
    end = pair.get("measurement_end")
    return (
        pair.get("stable_across_measurement") is True
        and start == end
        and snapshot_gate(start)
        and snapshot_gate(end)
    )


def _source_gate(source: object) -> bool:
    return _stable_pair_gate(source, _source_snapshot_gate)


def _hardware_gate(hardware: object) -> bool:
    return _stable_pair_gate(hardware, _hardware_snapshot_gate)


def _topology_gate(topology: object) -> bool:
    return _stable_pair_gate(
        topology,
        lambda rows: _topology_snapshot_gate(rows, TP),
    )


def _ensure_python_bin_on_path() -> None:
    """Make FlashInfer's JIT helper visible beside the active interpreter."""
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    if os.access(python_bin / "ninja", os.X_OK):
        os.environ["PATH"] = os.pathsep.join(part for part in (str(python_bin), path) if part)


def _prompt_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, length in enumerate(PROMPT_LENGTHS):
        tokens = _prompt_tokens(index, length, VOCAB_SIZE)
        rows.append(
            {
                "length": length,
                "first": list(tokens[:4]),
                "last": list(tokens[-4:]),
                "sha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return rows


def _prompt_gate(rows: object) -> bool:
    return rows == _prompt_rows()


def _requests(prefix: str):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return [
        EngineRequest(
            request_id=f"{prefix}-{index}",
            prompt_token_ids=_prompt_tokens(index, length, VOCAB_SIZE),
            max_new_tokens=MAX_NEW_TOKENS,
            ignore_eos=True,
            sampling=EngineSampling(temperature=0.0),
        )
        for index, length in enumerate(PROMPT_LENGTHS)
    ]


def _new_scheduler():
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler

    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    scratch = cache.reserve_scratch_page()
    if scratch != 0:
        raise RuntimeError(f"formal graph scratch page changed: expected 0, got {scratch}")
    return Scheduler(
        cache,
        max_num_batched_tokens=sum(PROMPT_LENGTHS),
        max_num_seqs=REQUEST_COUNT,
        page_size=PAGE_SIZE,
    )


def _normalize_outputs(
    outputs: dict[str, tuple[int, ...]],
    prefix: str,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    for index in range(REQUEST_COUNT):
        values = outputs.get(f"{prefix}-{index}", ())
        if len(values) != MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{prefix}-{index} produced {len(values)} tokens, expected {MAX_NEW_TOKENS}"
            )
        normalized.append([int(token) for token in values])
    return normalized


def _run_cell(runner, *, enabled: bool, prefix: str) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.engine_core import EngineCore

    runner.set_decode_page_table_cache_enabled(enabled)
    scheduler = _new_scheduler()
    engine = EngineCore(scheduler, runner)
    requests = _requests(prefix)
    for request in requests:
        engine.add_request(request)

    # The prompt burst is intentionally outside the decode measurement.
    prefill_finished = engine.step()
    if prefill_finished:
        raise RuntimeError(f"formal prefill unexpectedly finished requests: {prefill_finished}")
    if any(len(scheduler.states[request.request_id].outputs) != 1 for request in requests):
        raise RuntimeError("formal prefill did not produce exactly one token per row")

    runner.decode_page_table_cache_stats(reset=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    outputs = engine.run_to_completion()
    end.record()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    cuda_ms = float(start.elapsed_time(end))
    pre_release = list(runner.decode_page_table_cache_stats())
    try:
        normalized = _normalize_outputs(outputs, prefix)
    finally:
        for request in requests:
            runner.release(request.request_id)
    post_release = list(runner.decode_page_table_cache_stats())
    return {
        "outputs": normalized,
        "pre_release_rank_stats": pre_release,
        "post_release_rank_stats": post_release,
        "timing": {
            "binding": False,
            "wall_s": wall_s,
            "cuda_ms": cuda_ms,
            "tpot_s": wall_s / DECODE_TOKENS,
            "tokens_per_s": DECODE_TOKENS / wall_s,
        },
    }


def _order_for_round(round_index: int) -> tuple[str, ...]:
    if round_index % 2 == 0:
        return ("legacy", "cache", "cache", "legacy")
    return ("cache", "legacy", "legacy", "cache")


def _summary(values: list[float]) -> dict[str, float]:
    median = float(statistics.median(values))
    return {
        "median": median,
        "mad": float(statistics.median(abs(value - median) for value in values)),
    }


def _diagnostic_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("legacy", "cache"):
        selected = [sample for sample in samples if sample.get("mode") == mode]
        result[mode] = {
            metric: _summary([float(sample["timing"][metric]) for sample in selected])
            for metric in ("wall_s", "cuda_ms", "tpot_s", "tokens_per_s")
        }
    return result


def _run_pair(runner) -> dict[str, Any]:
    # Warm graph capture, the exact rollback path, and the grow-only cache.
    _run_cell(runner, enabled=False, prefix="warm-legacy")
    _run_cell(runner, enabled=True, prefix="warm-cache")

    samples: list[dict[str, Any]] = []
    orders: list[list[str]] = []
    for round_index in range(FORMAL_ROUNDS):
        order = _order_for_round(round_index)
        orders.append(list(order))
        for slot, mode in enumerate(order):
            cell = _run_cell(
                runner,
                enabled=(mode == "cache"),
                prefix=f"r{round_index}-s{slot}-{mode}",
            )
            samples.append(
                {
                    "round": round_index,
                    "slot": slot,
                    "mode": mode,
                    **cell,
                }
            )
    return {
        "orders": orders,
        "samples": samples,
        "diagnostic": {
            "binding": False,
            "summary": _diagnostic_summary(samples),
        },
    }


def _formal_config() -> dict[str, Any]:
    return {
        "tp": TP,
        "num_pages": NUM_PAGES,
        "page_size": PAGE_SIZE,
        "request_count": REQUEST_COUNT,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decode_steps": DECODE_STEPS,
        "rounds": FORMAL_ROUNDS,
        "graph_max_batch": GRAPH_MAX_BATCH,
        "graph_max_pages": GRAPH_MAX_PAGES,
        "timing_is_diagnostic": True,
    }


def _stats_identical(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    first = rows[0].get("stats")
    return isinstance(first, dict) and all(row.get("stats") == first for row in rows[1:])


def _rank_envelope_gate(rows: object) -> bool:
    if not isinstance(rows, list) or len(rows) != TP:
        return False
    return _stats_identical(rows) and all(
        isinstance(row, dict)
        and set(row) == {"rank", "world_size", "device", "stats"}
        and row["rank"] == rank
        and row["world_size"] == TP
        and row["device"] == f"cuda:{rank}"
        and isinstance(row["stats"], dict)
        for rank, row in enumerate(rows)
    )


def _dict_contains(actual: object, expected: dict[str, object]) -> bool:
    return isinstance(actual, dict) and all(
        actual.get(key) == value for key, value in expected.items()
    )


def _rank_stats_gate(
    rows: object,
    *,
    mode: str,
    released: bool,
) -> bool:
    if not _rank_envelope_gate(rows):
        return False
    assert isinstance(rows, list)
    stats = rows[0]["stats"]
    cache_enabled = mode == "cache"
    common = {
        "enabled": cache_enabled,
        "builds": EXPECTED_BUILDS,
        "rows": EXPECTED_ROWS,
        "host_page_ids_visited": EXPECTED_OWNED_ELEMENTS,
    }
    legacy = (
        {
            "legacy_outer_allocations": 0,
            "legacy_row_allocations": 0,
            "legacy_elements_written": 0,
            "legacy_owned_elements_uploaded": 0,
        }
        if cache_enabled
        else {
            "legacy_outer_allocations": EXPECTED_BUILDS,
            "legacy_row_allocations": EXPECTED_ROWS,
            "legacy_elements_written": EXPECTED_LEGACY_ELEMENTS_WRITTEN,
            "legacy_owned_elements_uploaded": EXPECTED_OWNED_ELEMENTS,
        }
    )
    cache = (
        {
            "storage_allocations": 0,
            "storage_elements_copied": 0,
            "rows_uploaded": EXPECTED_CACHE_ROWS_UPLOADED,
            "elements_uploaded": EXPECTED_CACHE_ELEMENTS_UPLOADED,
            "row_hits": EXPECTED_CACHE_ROW_HITS,
            "released_rows": REQUEST_COUNT if released else 0,
            "removed_rows": 0,
            "full_row_uploads": REQUEST_COUNT,
            "partial_row_uploads": (EXPECTED_CACHE_ROWS_UPLOADED - REQUEST_COUNT),
            "live_rows": 0 if released else REQUEST_COUNT,
            "row_capacity": REQUEST_COUNT,
            "page_capacity": 32,
        }
        if cache_enabled
        else {
            "storage_allocations": 0,
            "storage_elements_copied": 0,
            "rows_uploaded": 0,
            "elements_uploaded": 0,
            "row_hits": 0,
            "released_rows": 0,
            "removed_rows": 0,
            "full_row_uploads": 0,
            "partial_row_uploads": 0,
            "live_rows": 0,
            "row_capacity": REQUEST_COUNT,
            "page_capacity": 32,
        }
    )
    graph = (
        {
            "rows_copied": EXPECTED_GRAPH_ROWS_COPIED,
            "elements_copied": EXPECTED_GRAPH_ELEMENTS_COPIED,
            "row_hits": EXPECTED_GRAPH_ROW_HITS,
            "released_rows": REQUEST_COUNT if released else 0,
            "live_rows": 0 if released else REQUEST_COUNT,
        }
        if cache_enabled
        else {
            "rows_copied": EXPECTED_ROWS,
            "elements_copied": EXPECTED_GRAPH_FULL_ELEMENTS,
            "row_hits": 0,
            "released_rows": 0,
            "live_rows": 0,
        }
    )
    dispatch = {
        "graph_executions": DECODE_STEPS,
        "eager_fallbacks": 0,
        "captured_buckets": [REQUEST_COUNT],
    }
    return (
        _dict_contains(stats, common)
        and _dict_contains(stats, legacy)
        and _dict_contains(stats.get("cache"), cache)
        and _dict_contains(stats.get("graph"), graph)
        and _dict_contains(stats.get("graph_dispatch"), dispatch)
        and (
            not cache_enabled
            or (
                cache["elements_uploaded"] < EXPECTED_OWNED_ELEMENTS
                and graph["elements_copied"] < EXPECTED_GRAPH_FULL_ELEMENTS
                and cache["row_hits"] > 0
                and graph["row_hits"] > 0
            )
        )
    )


def _outputs_gate(outputs: object) -> bool:
    return (
        isinstance(outputs, list)
        and len(outputs) == REQUEST_COUNT
        and all(
            isinstance(row, list)
            and len(row) == MAX_NEW_TOKENS
            and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in row)
            for row in outputs
        )
    )


def _timing_gate(timing: object) -> bool:
    if not isinstance(timing, dict) or timing.get("binding") is not False:
        return False
    if set(timing) != {
        "binding",
        "wall_s",
        "cuda_ms",
        "tpot_s",
        "tokens_per_s",
    }:
        return False
    return all(
        type(timing.get(metric)) in (int, float)
        and math.isfinite(float(timing[metric]))
        and float(timing[metric]) > 0
        for metric in ("wall_s", "cuda_ms", "tpot_s", "tokens_per_s")
    )


def _measurements_gate(measurements: object) -> tuple[bool, bool, bool]:
    if not isinstance(measurements, dict):
        return False, False, False
    expected_orders = [list(_order_for_round(round_index)) for round_index in range(FORMAL_ROUNDS)]
    samples = measurements.get("samples")
    if (
        measurements.get("orders") != expected_orders
        or not isinstance(samples, list)
        or len(samples) != FORMAL_ROUNDS * 4
    ):
        return False, False, False

    legacy_ok = True
    cache_ok = True
    outputs: list[object] = []
    timing_ok = True
    for index, sample in enumerate(samples):
        round_index, slot = divmod(index, 4)
        mode = expected_orders[round_index][slot]
        if (
            not isinstance(sample, dict)
            or sample.get("round") != round_index
            or sample.get("slot") != slot
            or sample.get("mode") != mode
        ):
            return False, False, False
        outputs.append(sample.get("outputs"))
        timing_ok &= _timing_gate(sample.get("timing"))
        structural = _rank_stats_gate(
            sample.get("pre_release_rank_stats"),
            mode=mode,
            released=False,
        ) and _rank_stats_gate(
            sample.get("post_release_rank_stats"),
            mode=mode,
            released=True,
        )
        if mode == "legacy":
            legacy_ok &= structural
        else:
            cache_ok &= structural

    diagnostic = measurements.get("diagnostic")
    timing_ok &= (
        isinstance(diagnostic, dict)
        and diagnostic.get("binding") is False
        and diagnostic.get("summary") == _diagnostic_summary(samples)
    )
    parity = (
        bool(outputs)
        and _outputs_gate(outputs[0])
        and all(output == outputs[0] for output in outputs[1:])
    )
    return legacy_ok, cache_ok, parity and timing_ok


def _gate_policy() -> dict[str, Any]:
    return {
        "timing_binding": False,
        "binding_basis": [
            "clean_source_checkpoint_hardware_topology",
            "all_rank_complete_identical_structural_stats",
            "legacy_allocation_and_full_graph_copy_counts",
            "bounded_cache_and_reduced_upload_copy_counts",
            "zero_graph_fallback_and_exact_decode_dispatch",
            "exact_output_parity_and_request_release",
        ],
        "diagnostic_only": [
            "wall_s",
            "cuda_ms",
            "tpot_s",
            "tokens_per_s",
            "median",
            "mad",
        ],
    }


def _gate_policy_gate(payload: dict[str, Any]) -> bool:
    return payload.get("gate_policy") == _gate_policy()


def _evaluate(payload: dict[str, Any]) -> dict[str, bool]:
    legacy, cache, parity_and_timing = _measurements_gate(payload.get("measurements"))
    return {
        "schema_and_kind": (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("measurement_kind") == MEASUREMENT_KIND
        ),
        "formal_geometry": payload.get("config") == _formal_config(),
        "clean_committed_source_stable": _source_gate(payload.get("source")),
        "exact_qwen3_32b_checkpoint_stable": _checkpoint_gate(payload.get("checkpoint")),
        "eight_gpu_hardware_stable": _hardware_gate(payload.get("hardware")),
        "tp8_topology_stable": _topology_gate(payload.get("topology")),
        "prompt_provenance_complete": _prompt_gate(payload.get("prompts")),
        "legacy_path_exact": legacy,
        "cache_path_reduced_and_bounded": cache,
        "output_parity_release_and_timing_shape": parity_and_timing,
        "timing_strictly_non_binding": _gate_policy_gate(payload),
    }


def _measure(model_path: Path) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.worker import DistTPLauncher

    _ensure_python_bin_on_path()
    if not torch.cuda.is_available() or torch.cuda.device_count() < TP:
        raise RuntimeError(f"formal #229 evidence requires {TP} visible CUDA devices")

    source_start = _source_snapshot()
    if not _source_snapshot_gate(source_start):
        raise RuntimeError("formal evidence requires a clean commit containing every bound source")
    hardware_start = _hardware()
    if not _hardware_snapshot_gate(hardware_start):
        raise RuntimeError("formal evidence requires the reviewed eight-GPU hardware profile")
    model_path = model_path.resolve()
    checkpoint_start = _checkpoint_manifest(model_path)
    if not _checkpoint_snapshot_gate(checkpoint_start):
        raise RuntimeError("--model-path is not the exact reviewed Qwen3-32B checkpoint")

    launcher = DistTPLauncher(
        str(model_path),
        tp=TP,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=[],
        graph_scratch_page=0,
        graph_max_batch=GRAPH_MAX_BATCH,
        graph_max_pages=GRAPH_MAX_PAGES,
    )
    try:
        runner = launcher.runner
        topology_start = list(runner.sampling_ownership_metadata())
        measurements = _run_pair(runner)
        topology_end = list(runner.sampling_ownership_metadata())
    finally:
        launcher.shutdown()

    checkpoint_end = _checkpoint_manifest(model_path)
    hardware_end = _hardware()
    source_end = _source_snapshot()
    source = {
        "measurement_start": source_start,
        "measurement_end": source_end,
        "stable_across_measurement": source_start == source_end,
    }
    checkpoint = {
        "measurement_start": checkpoint_start,
        "measurement_end": checkpoint_end,
        "stable_across_measurement": checkpoint_start == checkpoint_end,
    }
    hardware = {
        "measurement_start": hardware_start,
        "measurement_end": hardware_end,
        "stable_across_measurement": hardware_start == hardware_end,
    }
    topology = {
        "measurement_start": topology_start,
        "measurement_end": topology_end,
        "stable_across_measurement": topology_start == topology_end,
    }
    if not _source_gate(source):
        raise RuntimeError("bound source changed while formal evidence was running")
    if not _checkpoint_gate(checkpoint):
        raise RuntimeError("checkpoint changed while formal evidence was running")
    if not _hardware_gate(hardware):
        raise RuntimeError("hardware profile changed while formal evidence was running")
    if not _topology_gate(topology):
        raise RuntimeError("TP topology changed while formal evidence was running")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": _formal_config(),
        "gate_policy": _gate_policy(),
        "source": source,
        "checkpoint": checkpoint,
        "hardware": hardware,
        "topology": topology,
        "prompts": _prompt_rows(),
        "measurements": measurements,
    }
    payload["checks"] = _evaluate(payload)
    payload["passed"] = all(payload["checks"].values())
    return payload


def _verify(
    artifact: Path,
    model_path: Path | None,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text())
    raw_checks = _evaluate(payload)
    stored_matches = payload.get("checks") == raw_checks and payload.get("passed") == all(
        raw_checks.values()
    )
    checks = dict(raw_checks)
    checks["stored_replay_matches"] = stored_matches
    checks["source_commit_replay"] = _source_gate(payload.get("source"))

    hardware = payload.get("hardware")
    current_hardware = _hardware()
    checks["hardware_replay"] = (
        _hardware_gate(hardware)
        and isinstance(hardware, dict)
        and current_hardware == hardware.get("measurement_start")
        and current_hardware == hardware.get("measurement_end")
    )
    checkpoint = payload.get("checkpoint")
    if model_path is None:
        checks["checkpoint_replay"] = False
    else:
        current_checkpoint = _checkpoint_manifest(model_path.resolve())
        checks["checkpoint_replay"] = (
            _checkpoint_gate(checkpoint)
            and isinstance(checkpoint, dict)
            and current_checkpoint == checkpoint.get("measurement_start")
            and current_checkpoint == checkpoint.get("measurement_end")
            and _checkpoint_snapshot_gate(current_checkpoint)
        )
    return {
        "artifact": str(artifact),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()

    if args.verify is not None:
        result = _verify(
            args.verify,
            Path(args.model_path) if args.model_path else None,
        )
    else:
        if args.model_path is None or args.output is None:
            parser.error("measurement requires --model-path and --output")
        result = _measure(Path(args.model_path))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(args.assert_gate and not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
