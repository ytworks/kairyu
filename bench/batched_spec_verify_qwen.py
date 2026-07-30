"""Formal Qwen3-32B TP8 evidence for issue #215 batched target verification.

The verdict is structural and parity-based.  Wall time, TPOT and throughput are
recorded from warmed ABBA/BAAB rounds only as diagnostics; OS scheduling jitter
can never decide this gate.

Measure:

    uv run python bench/batched_spec_verify_qwen.py \
      --model-path /models/qwen3-32b --tp 8 \
      --output bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8.json \
      --assert-gate

Replay:

    uv run python bench/batched_spec_verify_qwen.py \
      --verify bench/results/issue-215-batched-spec-verify-qwen3-32b-tp8.json \
      --model-path /models/qwen3-32b --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    from bench import batched_prefill_qwen as _provenance
except ModuleNotFoundError:  # direct ``python bench/...py`` execution
    import batched_prefill_qwen as _provenance

SCHEMA_VERSION = 1
MEASUREMENT_KIND = "qwen3-32b-tp-batched-speculative-verification"
REQUEST_COUNT = 8
SPECULATIVE_TOKENS = 3
MAX_NEW_TOKENS = 8
PATTERN_REPEATS = 4
FORMAL_ROUNDS = 2
NUM_PAGES = 4096
PAGE_SIZE = 16
GRAPH_MAX_BATCH = 64
GRAPH_MAX_PAGES = 64

# Each prompt is built by encoding one fixed fragment and repeating its exact
# token tuple.  That makes the initial n-gram draft deterministic without
# assuming that separately encoded whitespace has tokenizer-stable boundaries.
PROMPT_SPECS = (
    ("repetitive", "greek", " alpha beta gamma delta"),
    ("repetitive", "numbers", " one two three four"),
    ("repetitive", "colors", " red green blue amber"),
    ("code", "python-loop", " for i in range(4): total += i\n"),
    ("code", "python-branch", " if ready: value = value + 1\n"),
    ("code", "sql", " SELECT id FROM items WHERE active = true;\n"),
    ("json", "object", ' {"key":[1,2,3],"ok":true}\n'),
    ("json", "event", ' {"event":"tick","count":4}\n'),
)

IMPLEMENTATION_FILES = (
    "kairyu/engine/core/draft.py",
    "kairyu/engine/core/engine_core.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/prefill.py",
    "kairyu/engine/core/radix_kv.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/engine/core/spec_decode.py",
    "kairyu/engine/core/spec_runner.py",
    "kairyu/engine/core/step_executor.py",
    "kairyu/engine/core/step_input.py",
    "kairyu/engine/core/torch_runner.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/core/attention/__init__.py",
    "kairyu/engine/core/attention/flashinfer_gpu.py",
    "kairyu/engine/core/cuda_graph_gpu.py",
    "kairyu/engine/core/kv_pool.py",
    "kairyu/engine/tokenizer.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/models/attention.py",
    "kairyu/models/llama.py",
    "kairyu/models/loader.py",
    "bench/batched_prefill_qwen.py",
    "bench/batched_spec_verify_qwen.py",
)
_REPO = Path(__file__).resolve().parents[1]

# The reviewed checkpoint constants are shared with the #224 formal runner.
_EXPECTED_CONFIG_SHA256 = _provenance._EXPECTED_CONFIG_SHA256
_EXPECTED_INDEX_SHA256 = _provenance._EXPECTED_INDEX_SHA256
_EXPECTED_WEIGHTS_MANIFEST_SHA256 = _provenance._EXPECTED_WEIGHTS_MANIFEST_SHA256
_EXPECTED_WEIGHT_ROWS = _provenance._EXPECTED_WEIGHT_ROWS
_checkpoint_manifest = _provenance._checkpoint_manifest
_checkpoint_snapshot_gate = _provenance._checkpoint_snapshot_gate
_checkpoint_gate = _provenance._checkpoint_gate
_hardware = _provenance._hardware
_hardware_gate = _provenance._hardware_gate
_topology_gate = _provenance._topology_gate


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


def _git_show(commit: str, relative: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ("git", "show", f"{commit}:{relative}"),
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
        "files_match_commit": (
            files_match_commit and set(files) == set(IMPLEMENTATION_FILES)
        ),
    }


def _source_snapshot_is_clean_and_bound(snapshot: object) -> bool:
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
        or any(not _hex_digest(value) for value in files.values())
    ):
        return False
    for name in IMPLEMENTATION_FILES:
        committed = _git_show(commit, name)
        current_path = _REPO / name
        if (
            committed is None
            or not current_path.is_file()
            or files[name] != hashlib.sha256(committed).hexdigest()
            or current_path.read_bytes() != committed
        ):
            return False
    return True


def _source_gate(source: object) -> bool:
    if not isinstance(source, dict):
        return False
    start = source.get("measurement_start")
    end = source.get("measurement_end")
    return (
        source.get("stable_across_measurement") is True
        and start == end
        and _source_snapshot_is_clean_and_bound(start)
        and _source_snapshot_is_clean_and_bound(end)
    )


def _ensure_python_bin_on_path() -> None:
    """Make FlashInfer's JIT helper visible beside the active interpreter."""
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    ninja = python_bin / "ninja"
    if os.access(ninja, os.X_OK):
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(python_bin), path) if part
        )


def _prompt_rows(tokenizer) -> list[dict[str, Any]]:
    from kairyu.engine.core.draft import NGramDraftSource

    source = NGramDraftSource()
    rows: list[dict[str, Any]] = []
    for index, (category, name, fragment_text) in enumerate(PROMPT_SPECS):
        fragment = tuple(int(token) for token in tokenizer.encode(fragment_text))
        if len(fragment) < 4:
            raise RuntimeError(
                f"prompt fragment {name!r} encoded to only {len(fragment)} tokens"
            )
        prompt = fragment * PATTERN_REPEATS
        committed = fragment[0]
        expected_draft = tuple(
            source.propose(
                (*prompt, committed),
                max_draft=SPECULATIVE_TOKENS,
            )
        )
        if len(expected_draft) != SPECULATIVE_TOKENS:
            raise RuntimeError(
                f"prompt fragment {name!r} did not produce a complete n-gram draft: "
                f"{expected_draft}"
            )
        rows.append(
            {
                "index": index,
                "category": category,
                "name": name,
                "fragment_text": fragment_text,
                "fragment_token_ids": list(fragment),
                "pattern_repeats": PATTERN_REPEATS,
                "prompt_token_ids": list(prompt),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt, separators=(",", ":")).encode()
                ).hexdigest(),
                "draft_probe_committed": committed,
                "expected_ngram_draft": list(expected_draft),
            }
        )
    return rows


def _prompt_rows_gate(rows: object) -> bool:
    from kairyu.engine.core.draft import NGramDraftSource

    if not isinstance(rows, list) or len(rows) != REQUEST_COUNT:
        return False
    source = NGramDraftSource()
    required = {
        "index",
        "category",
        "name",
        "fragment_text",
        "fragment_token_ids",
        "pattern_repeats",
        "prompt_token_ids",
        "prompt_sha256",
        "draft_probe_committed",
        "expected_ngram_draft",
    }
    for index, (row, spec) in enumerate(zip(rows, PROMPT_SPECS, strict=True)):
        if not isinstance(row, dict) or set(row) != required:
            return False
        category, name, fragment_text = spec
        fragment = row["fragment_token_ids"]
        prompt = row["prompt_token_ids"]
        if (
            row["index"] != index
            or row["category"] != category
            or row["name"] != name
            or row["fragment_text"] != fragment_text
            or row["pattern_repeats"] != PATTERN_REPEATS
            or not isinstance(fragment, list)
            or len(fragment) < 4
            or any(type(token) is not int or not 0 <= token < 151936 for token in fragment)
            or prompt != fragment * PATTERN_REPEATS
            or row["draft_probe_committed"] != fragment[0]
        ):
            return False
        prompt_tuple = tuple(prompt)
        digest = hashlib.sha256(
            json.dumps(prompt_tuple, separators=(",", ":")).encode()
        ).hexdigest()
        draft = tuple(
            source.propose(
                (*prompt_tuple, fragment[0]),
                max_draft=SPECULATIVE_TOKENS,
            )
        )
        if (
            row["prompt_sha256"] != digest
            or row["expected_ngram_draft"] != list(draft)
            or len(draft) != SPECULATIVE_TOKENS
        ):
            return False
    return True


def _prompt_tokenizer_replay(rows: object, model_path: Path) -> bool:
    if not _prompt_rows_gate(rows):
        return False
    from kairyu.engine.tokenizer import resolve_tokenizer

    tokenizer = resolve_tokenizer(str(model_path))
    assert isinstance(rows, list)
    return all(
        list(tokenizer.encode(row["fragment_text"])) == row["fragment_token_ids"]
        for row in rows
    )


def _requests(prefix: str, prompt_rows: list[dict[str, Any]], *, max_new_tokens: int):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return [
        EngineRequest(
            request_id=f"{prefix}-{index}",
            prompt_token_ids=tuple(row["prompt_token_ids"]),
            max_new_tokens=max_new_tokens,
            sampling=EngineSampling(temperature=0.0),
        )
        for index, row in enumerate(prompt_rows)
    ]


def _new_scheduler(*, graph: bool, speculative_tokens: int):
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler

    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    if graph:
        scratch = cache.reserve_scratch_page()
        if scratch != 0:
            raise RuntimeError(f"formal graph scratch page changed: expected 0, got {scratch}")
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=4096,
        max_num_seqs=REQUEST_COUNT,
        page_size=PAGE_SIZE,
        speculative_tokens=speculative_tokens,
    )
    return scheduler


def _normalized_outputs(
    outputs: dict[str, tuple[int, ...]],
    prefix: str,
    *,
    expected_tokens: int,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    for index in range(REQUEST_COUNT):
        values = outputs.get(f"{prefix}-{index}", ())
        if len(values) != expected_tokens:
            raise RuntimeError(
                f"{prefix}-{index} produced {len(values)} tokens, "
                f"expected {expected_tokens}"
            )
        normalized.append([int(token) for token in values])
    return normalized


def _run_end_to_end(
    runner,
    *,
    enabled: bool,
    graph: bool,
    prefix: str,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.spec_runner import SpeculativeRunner

    runner.set_batched_verification_enabled(enabled)
    runner.verification_execution_stats(reset=True)
    scheduler = _new_scheduler(
        graph=graph,
        speculative_tokens=SPECULATIVE_TOKENS,
    )
    speculative = SpeculativeRunner(runner)
    engine = EngineCore(scheduler, speculative)
    requests = _requests(
        prefix,
        prompt_rows,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    for request in requests:
        engine.add_request(request)

    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    outputs = engine.run_to_completion()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    rank_stats = list(runner.verification_execution_stats())
    for request in requests:
        speculative.release(request.request_id)

    token_count = REQUEST_COUNT * MAX_NEW_TOKENS
    return {
        "outputs": _normalized_outputs(
            outputs,
            prefix,
            expected_tokens=MAX_NEW_TOKENS,
        ),
        "draft_proposed": speculative.draft_proposed,
        "draft_accepted": speculative.draft_accepted,
        "rank_stats": rank_stats,
        "timing": {
            "binding": False,
            "wall_s": wall_s,
            "tpot_s": wall_s / token_count,
            "tokens_per_s": token_count / wall_s,
        },
    }


def _structural_probe(
    runner,
    *,
    enabled: bool,
    graph: bool,
    prefix: str,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score exactly 8*(draft+bonus)=32 positions after one prompt prefill."""
    import torch

    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.spec_runner import _OverlayState

    runner.set_batched_verification_enabled(enabled)
    scheduler = _new_scheduler(
        graph=graph,
        speculative_tokens=SPECULATIVE_TOKENS,
    )
    requests = _requests(prefix, prompt_rows, max_new_tokens=MAX_NEW_TOKENS)
    for request in requests:
        scheduler.add_request(request)
    prefill = scheduler.schedule().scheduled
    if (
        len(prefill) != REQUEST_COUNT
        or any(not chunk.is_prefill for chunk in prefill)
    ):
        raise RuntimeError("structural probe did not form one complete prefill group")
    first = runner.execute(prefill, scheduler.states)
    scheduler.update(token_ids(first))
    verify = scheduler.schedule().scheduled
    if (
        len(verify) != REQUEST_COUNT
        or any(
            chunk.is_prefill
            or chunk.position != 1
            or chunk.num_tokens != SPECULATIVE_TOKENS + 1
            for chunk in verify
        )
    ):
        raise RuntimeError(f"structural probe verification geometry changed: {verify!r}")

    overlays: dict[str, object] = {}
    drafts: list[list[int]] = []
    for index, chunk in enumerate(verify):
        state = scheduler.states[chunk.request_id]
        row = prompt_rows[index]
        committed = int(row["draft_probe_committed"])
        draft = tuple(int(token) for token in row["expected_ngram_draft"])
        overlays[chunk.request_id] = _OverlayState(
            state,
            (committed, *draft),
        )
        drafts.append(list(draft))

    runner.verification_execution_stats(reset=True)
    targets = runner.execute(verify, overlays)
    torch.cuda.synchronize()
    rank_stats = list(runner.verification_execution_stats())
    normalized_targets = [
        [int(record.token_id) for record in targets[chunk.request_id]]
        for chunk in verify
    ]
    for request in requests:
        runner.release(request.request_id)
    return {
        "enabled": enabled,
        "schedule": [
            {
                "request_id": chunk.request_id,
                "position": chunk.position,
                "num_tokens": chunk.num_tokens,
                "is_prefill": chunk.is_prefill,
            }
            for chunk in verify
        ],
        "drafts": drafts,
        "targets": normalized_targets,
        "rank_stats": rank_stats,
    }


def _order_for_round(round_index: int) -> tuple[str, ...]:
    if round_index % 2 == 0:
        return ("sequential", "candidate", "candidate", "sequential")
    return ("candidate", "sequential", "sequential", "candidate")


def _summary(values: list[float]) -> dict[str, float]:
    median = float(statistics.median(values))
    return {
        "median": median,
        "mad": float(statistics.median(abs(value - median) for value in values)),
    }


def _diagnostic_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("sequential", "candidate"):
        selected = [sample for sample in samples if sample["mode"] == mode]
        result[mode] = {
            metric: _summary([float(sample["timing"][metric]) for sample in selected])
            for metric in ("wall_s", "tpot_s", "tokens_per_s")
        }
    return result


def _run_pair(
    runner,
    *,
    pair_name: str,
    graph: bool,
    prompt_rows: list[dict[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    # Warm both code paths (and every graph bucket reached by this workload).
    _run_end_to_end(
        runner,
        enabled=False,
        graph=graph,
        prefix=f"warm-{pair_name}-sequential",
        prompt_rows=prompt_rows,
    )
    _run_end_to_end(
        runner,
        enabled=True,
        graph=graph,
        prefix=f"warm-{pair_name}-candidate",
        prompt_rows=prompt_rows,
    )

    samples: list[dict[str, Any]] = []
    orders: list[list[str]] = []
    for round_index in range(rounds):
        order = _order_for_round(round_index)
        orders.append(list(order))
        for slot, mode in enumerate(order):
            cell = _run_end_to_end(
                runner,
                enabled=(mode == "candidate"),
                graph=graph,
                prefix=f"{pair_name}-r{round_index}-s{slot}-{mode}",
                prompt_rows=prompt_rows,
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


def _measure_launcher(
    model_path: Path,
    *,
    graph: bool,
    prompt_rows: list[dict[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    import gc

    import torch

    from kairyu.engine.core.worker import DistTPLauncher

    launcher = DistTPLauncher(
        str(model_path),
        tp=8,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=[],
        graph_scratch_page=0 if graph else None,
        graph_max_batch=GRAPH_MAX_BATCH if graph else 0,
        graph_max_pages=GRAPH_MAX_PAGES if graph else 0,
    )
    runner = None
    try:
        runner = launcher.runner
        pair_name = "cuda-graph" if graph else "eager"
        timing = _run_pair(
            runner,
            pair_name=pair_name,
            graph=graph,
            prompt_rows=prompt_rows,
            rounds=rounds,
        )
        if graph:
            # Capture the exact 32-row structural bucket before counters are
            # collected. End-to-end n-gram acceptance is intentionally not
            # assumed to visit every fixed structural geometry.
            _structural_probe(
                runner,
                enabled=True,
                graph=True,
                prefix="warm-structural-cuda-graph",
                prompt_rows=prompt_rows,
            )
        structural = {
            "sequential": _structural_probe(
                runner,
                enabled=False,
                graph=graph,
                prefix=f"structural-{pair_name}-sequential",
                prompt_rows=prompt_rows,
            ),
            "candidate": _structural_probe(
                runner,
                enabled=True,
                graph=graph,
                prefix=f"structural-{pair_name}-candidate",
                prompt_rows=prompt_rows,
            ),
        }
        topology = list(runner.sampling_ownership_metadata())
        result = {
            "timing": timing,
            "structural": structural,
            "topology": topology,
        }
    finally:
        try:
            launcher.shutdown()
        finally:
            runner = None
            del launcher
            gc.collect()
            torch.cuda.empty_cache()
    return result


def _micro_page_geometry(
    prompt_rows: list[dict[str, Any]],
) -> tuple[list[tuple[int, ...]], int]:
    page_lists: list[tuple[int, ...]] = []
    cursor = 0
    for row in prompt_rows:
        seq_len = len(row["prompt_token_ids"]) + SPECULATIVE_TOKENS + 1
        count = -(-seq_len // PAGE_SIZE)
        page_lists.append(tuple(range(cursor, cursor + count)))
        cursor += count
    return page_lists, cursor


def _micro_batches(prompt_rows, page_lists, *, device):
    import torch

    from kairyu.engine.core.prefill import (
        PrefillSequence,
        build_prefill_batch,
    )
    from kairyu.engine.core.step_executor import build_decode_batch

    prompt_sequences = []
    target_sequences = []
    flat_tokens: list[int] = []
    flat_positions: list[int] = []
    flat_pages: list[list[int]] = []
    flat_seq_lens: list[int] = []
    flat_write_from: list[int] = []
    geometry: list[dict[str, Any]] = []
    for index, (row, pages) in enumerate(
        zip(prompt_rows, page_lists, strict=True)
    ):
        prompt = tuple(int(token) for token in row["prompt_token_ids"])
        committed = int(row["draft_probe_committed"])
        draft = tuple(int(token) for token in row["expected_ngram_draft"])
        target_inputs = (committed, *draft)
        prompt_len = len(prompt)
        prompt_sequences.append(
            PrefillSequence(
                request_id=f"micro-prompt-{index}",
                token_ids=prompt,
                page_table=pages,
                chunk_start=0,
                seq_len=prompt_len,
                write_from=0,
            )
        )
        target_sequences.append(
            PrefillSequence(
                request_id=f"micro-target-{index}",
                token_ids=target_inputs,
                page_table=pages,
                chunk_start=prompt_len,
                seq_len=prompt_len + len(target_inputs),
                write_from=prompt_len,
            )
        )
        for offset, token in enumerate(target_inputs):
            flat_tokens.append(token)
            flat_positions.append(prompt_len + offset)
            flat_pages.append(list(pages))
            flat_seq_lens.append(prompt_len + offset + 1)
            flat_write_from.append(prompt_len)
        geometry.append(
            {
                "request_index": index,
                "prompt_length": prompt_len,
                "page_table": list(pages),
                "target_input_ids": list(target_inputs),
                "positions": list(range(prompt_len, prompt_len + len(target_inputs))),
                "seq_lens": list(
                    range(prompt_len + 1, prompt_len + len(target_inputs) + 1)
                ),
                "write_from": prompt_len,
            }
        )
    prompt_batch = build_prefill_batch(
        prompt_sequences,
        page_size=PAGE_SIZE,
        device=device,
    )
    ragged_target = build_prefill_batch(
        target_sequences,
        page_size=PAGE_SIZE,
        device=device,
    )
    decode_target = build_decode_batch(
        token_ids=torch.tensor(flat_tokens, dtype=torch.int64, device=device),
        positions=torch.tensor(flat_positions, dtype=torch.int64, device=device),
        page_lists=flat_pages,
        seq_lens=flat_seq_lens,
        max_pages=max(len(pages) for pages in page_lists),
        scratch_page=None,
        write_from=flat_write_from,
        device=device,
    )
    return prompt_batch, ragged_target, decode_target, geometry


def _micro_kv_payload(pool, geometry: list[dict[str, Any]]):
    import torch

    keys = []
    values = []
    for row in geometry:
        pages = row["page_table"]
        for position in row["positions"]:
            page = pages[position // PAGE_SIZE]
            slot = position % PAGE_SIZE
            keys.append(pool.k[:, page, slot])
            values.append(pool.v[:, page, slot])
    key_tensor = torch.stack(keys, dim=1).contiguous()
    value_tensor = torch.stack(values, dim=1).contiguous()
    raw = (
        key_tensor.view(torch.uint8).cpu().numpy().tobytes()
        + value_tensor.view(torch.uint8).cpu().numpy().tobytes()
    )
    return key_tensor, value_tensor, hashlib.sha256(raw).hexdigest()


def _poison_micro_target_slots(pool, geometry: list[dict[str, Any]]) -> None:
    """Make the final comparison prove that every declared target slot is written."""
    for row in geometry:
        pages = row["page_table"]
        for position in row["positions"]:
            page = pages[position // PAGE_SIZE]
            slot = position % PAGE_SIZE
            pool.k[:, page, slot].fill_(float("nan"))
            pool.v[:, page, slot].fill_(float("nan"))


def _micro_kv_diagnostics(
    decode_k,
    decode_v,
    ragged_k,
    ragged_v,
) -> dict[str, Any]:
    """Describe cross-kernel numerical drift without making it a pass criterion.

    FlashInfer's decode and prefill kernels use different BF16 reduction orders.
    The production correctness gate compares flattened verification with the
    sequential target path directly; this microcell only chooses between two
    execution strategies.  It therefore binds write coverage and selected
    tokens, while retaining these cross-kernel distances for inspection.
    """
    import torch

    decode = torch.cat(
        (decode_k.float().reshape(-1), decode_v.float().reshape(-1))
    )
    ragged = torch.cat(
        (ragged_k.float().reshape(-1), ragged_v.float().reshape(-1))
    )
    delta = decode - ragged
    ragged_rms = torch.sqrt(torch.mean(ragged.square()))
    denominator = max(float(ragged_rms.item()), torch.finfo(torch.float32).eps)
    cosine = torch.dot(decode, ragged) / (
        torch.linalg.vector_norm(decode)
        * torch.linalg.vector_norm(ragged)
    )
    return {
        "kv_allclose_2e2_diagnostic": bool(
            torch.allclose(decode_k, ragged_k, atol=2e-2, rtol=2e-2)
            and torch.allclose(decode_v, ragged_v, atol=2e-2, rtol=2e-2)
        ),
        "kv_max_abs_diff_diagnostic": float(delta.abs().max().item()),
        "kv_nrmse_diagnostic": float(
            torch.sqrt(torch.mean(delta.square())).item() / denominator
        ),
        "kv_cosine_similarity_diagnostic": float(cosine.item()),
    }


def _micro_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("flattened_decode", "native_ragged_prefill"):
        selected = [sample for sample in samples if sample["mode"] == mode]
        result[mode] = {
            metric: _summary([float(sample[metric]) for sample in selected])
            for metric in ("wall_s", "cuda_ms")
        }
    flattened = result["flattened_decode"]
    ragged = result["native_ragged_prefill"]

    def winner(metric: str) -> str:
        left = flattened[metric]["median"]
        right = ragged[metric]["median"]
        if left == right:
            return "tie"
        return "flattened_decode" if left < right else "native_ragged_prefill"

    result["comparison"] = {
        "wall_flattened_over_native_ratio": (
            flattened["wall_s"]["median"] / ragged["wall_s"]["median"]
        ),
        "cuda_flattened_over_native_ratio": (
            flattened["cuda_ms"]["median"] / ragged["cuda_ms"]["median"]
        ),
        "wall_winner": winner("wall_s"),
        "cuda_winner": winner("cuda_ms"),
    }
    return result


def _measure_native_ragged_microcomparison(
    model_path: Path,
    prompt_rows: list[dict[str, Any]],
    *,
    rounds: int,
) -> dict[str, Any]:
    """Compare both real Qwen paths on identical target rows and KV geometry."""
    import gc

    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.models.loader import load_model

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, config, _generation = load_model(
        model_path,
        dtype=torch.bfloat16,
        attention_backend=select_backend(probe()),
        target_device=device,
    )
    try:
        model = model.to(device)
        return _run_native_ragged_micro_loaded(
            model,
            config,
            prompt_rows,
            rounds=rounds,
            device=device,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_native_ragged_micro_loaded(
    model,
    config,
    prompt_rows: list[dict[str, Any]],
    *,
    rounds: int,
    device,
) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.kv_pool import PagedKVPool

    page_lists, num_pages = _micro_page_geometry(prompt_rows)

    def new_pool():
        return PagedKVPool(
            num_layers=config.num_hidden_layers,
            num_pages=num_pages,
            page_size=PAGE_SIZE,
            num_kv_heads=config.kv_cache_num_heads,
            head_dim=config.kv_cache_head_dim,
            dtype=torch.bfloat16,
            device=str(device),
            v_head_dim=config.kv_cache_v_head_dim,
        )

    decode_pool = new_pool()
    ragged_pool = new_pool()
    prompt_batch, ragged_target, decode_target, geometry = _micro_batches(
        prompt_rows,
        page_lists,
        device=device,
    )
    # Prime identical prompt KV outside the target-path timing.
    model.forward_prefill_batch(prompt_batch, decode_pool)
    model.forward_prefill_batch(prompt_batch, ragged_pool)
    torch.cuda.synchronize()

    def execute(mode: str):
        if mode == "flattened_decode":
            model.plan_decode_tensors(
                decode_pool,
                decode_target.page_tables,
                decode_target.seq_lens,
            )
            hidden = model.forward_decode_tensors(
                decode_target.token_ids,
                decode_target.positions,
                decode_pool,
                decode_target.page_tables,
                decode_target.seq_lens,
                decode_target.write_from,
            )
        else:
            hidden = model.forward_prefill_batch(ragged_target, ragged_pool)
        return torch.argmax(model.logits(hidden), dim=-1)

    # Compile/JIT and warm both paths before alternating measurement.
    flattened_tokens = execute("flattened_decode")
    ragged_tokens = execute("native_ragged_prefill")
    torch.cuda.synchronize()

    orders: list[list[str]] = []
    samples: list[dict[str, Any]] = []
    for round_index in range(rounds):
        order = (
            (
                "flattened_decode",
                "native_ragged_prefill",
                "native_ragged_prefill",
                "flattened_decode",
            )
            if round_index % 2 == 0
            else (
                "native_ragged_prefill",
                "flattened_decode",
                "flattened_decode",
                "native_ragged_prefill",
            )
        )
        orders.append(list(order))
        for slot, mode in enumerate(order):
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            selected = execute(mode)
            end.record()
            torch.cuda.synchronize()
            samples.append(
                {
                    "round": round_index,
                    "slot": slot,
                    "mode": mode,
                    "wall_s": time.perf_counter() - wall_start,
                    "cuda_ms": start.elapsed_time(end),
                    "selected_token_sha256": hashlib.sha256(
                        selected.cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )

    # Re-run both once after timing so parity and KV evidence describe the
    # artifact's final, fully synchronized state rather than one selected round.
    # Poisoning the declared target slots first makes a finite final payload
    # evidence of an actual write, rather than stale data from warm-up.
    _poison_micro_target_slots(decode_pool, geometry)
    _poison_micro_target_slots(ragged_pool, geometry)
    flattened_tokens = execute("flattened_decode")
    ragged_tokens = execute("native_ragged_prefill")
    torch.cuda.synchronize()
    decode_k, decode_v, decode_digest = _micro_kv_payload(
        decode_pool,
        geometry,
    )
    ragged_k, ragged_v, ragged_digest = _micro_kv_payload(
        ragged_pool,
        geometry,
    )
    token_rows = [
        list(map(int, row))
        for row in flattened_tokens.reshape(
            REQUEST_COUNT,
            SPECULATIVE_TOKENS + 1,
        ).cpu()
    ]
    ragged_rows = [
        list(map(int, row))
        for row in ragged_tokens.reshape(
            REQUEST_COUNT,
            SPECULATIVE_TOKENS + 1,
        ).cpu()
    ]
    decode_shape = {
        "k": list(decode_k.shape),
        "v": list(decode_v.shape),
    }
    ragged_shape = {
        "k": list(ragged_k.shape),
        "v": list(ragged_v.shape),
    }
    decode_finite = bool(
        torch.isfinite(decode_k).all() and torch.isfinite(decode_v).all()
    )
    ragged_finite = bool(
        torch.isfinite(ragged_k).all() and torch.isfinite(ragged_v).all()
    )
    diagnostics = _micro_kv_diagnostics(
        decode_k,
        decode_v,
        ragged_k,
        ragged_v,
    )
    result = {
        "available": True,
        "device": "cuda:0",
        "checkpoint_layout": {
            "architecture": config.architecture,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
        },
        "geometry": geometry,
        "parity": {
            "binding": True,
            "flattened_selected_tokens": token_rows,
            "native_ragged_selected_tokens": ragged_rows,
            "selected_tokens_equal": token_rows == ragged_rows,
            "flattened_kv_sha256": decode_digest,
            "native_ragged_kv_sha256": ragged_digest,
            "kv_digest_match_diagnostic": decode_digest == ragged_digest,
            "flattened_kv_shape": decode_shape,
            "native_ragged_kv_shape": ragged_shape,
            "kv_shapes_equal": decode_shape == ragged_shape,
            "flattened_target_slots_finite": decode_finite,
            "native_ragged_target_slots_finite": ragged_finite,
            **diagnostics,
        },
        "timing": {
            "binding": False,
            "orders": orders,
            "samples": samples,
            "summary": _micro_summary(samples),
        },
    }
    return result


def _backend_counts(rank_row: dict[str, Any]) -> tuple[int, int]:
    backend = rank_row["stats"].get("backend")
    if (
        not isinstance(backend, list)
        or len(backend) != 1
        or backend[0].get("type") != "FlashInferBackend"
    ):
        return -1, -1
    return int(backend[0].get("plans", -1)), int(backend[0].get("runs", -1))


def _structural_rank_gate(
    rows: object,
    *,
    enabled: bool,
    graph_runner: bool,
    layers: int,
) -> bool:
    positions = REQUEST_COUNT * (SPECULATIVE_TOKENS + 1)
    if not isinstance(rows, list) or len(rows) != 8:
        return False
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("rank") != rank
            or row.get("world_size") != 8
            or row.get("device") != f"cuda:{rank}"
            or not isinstance(row.get("stats"), dict)
        ):
            return False
        stats = row["stats"]
        if (
            "capability_gap" not in stats
            or stats["capability_gap"] is not None
        ):
            return False
        expected = (
            {
                "enabled": True,
                "requests": REQUEST_COUNT,
                "positions": positions,
                "model_calls": 1,
                "batched_groups": 1,
                "sequential_positions": 0,
                "graph_groups": 1 if graph_runner else 0,
            }
            if enabled
            else {
                "enabled": False,
                "requests": REQUEST_COUNT,
                "positions": positions,
                "model_calls": positions,
                "batched_groups": 0,
                "sequential_positions": positions,
                "graph_groups": 0,
            }
        )
        if any(stats.get(key) != value for key, value in expected.items()):
            return False
        graph = stats.get("graph")
        if graph_runner:
            if (
                not isinstance(graph, dict)
                or graph.get("eager_fallbacks") != 0
                or graph.get("graph_executions") != (1 if enabled else 0)
                or positions not in graph.get("captured_buckets", ())
            ):
                return False
        elif graph is not None:
            return False
        plans, runs = _backend_counts(row)
        expected_backend = (
            (1, 0)
            if enabled and graph_runner
            else (1, layers)
            if enabled
            else (positions, positions * layers)
        )
        if (plans, runs) != expected_backend:
            return False
    return True


def _structural_cell_gate(
    cell: object,
    *,
    prefix: str,
    enabled: bool,
    graph_runner: bool,
    layers: int,
) -> bool:
    if not isinstance(cell, dict):
        return False
    schedule = cell.get("schedule")
    drafts = cell.get("drafts")
    targets = cell.get("targets")
    if (
        cell.get("enabled") is not enabled
        or not isinstance(schedule, list)
        or len(schedule) != REQUEST_COUNT
        or not isinstance(drafts, list)
        or len(drafts) != REQUEST_COUNT
        or not isinstance(targets, list)
        or len(targets) != REQUEST_COUNT
    ):
        return False
    for index, row in enumerate(schedule):
        if row != {
            "request_id": f"{prefix}-{index}",
            "position": 1,
            "num_tokens": SPECULATIVE_TOKENS + 1,
            "is_prefill": False,
        }:
            return False
    if any(
        not isinstance(draft, list)
        or len(draft) != SPECULATIVE_TOKENS
        or any(type(token) is not int for token in draft)
        for draft in drafts
    ):
        return False
    if any(
        not isinstance(target, list)
        or len(target) != SPECULATIVE_TOKENS + 1
        or any(type(token) is not int or not 0 <= token < 151936 for token in target)
        for target in targets
    ):
        return False
    return _structural_rank_gate(
        cell.get("rank_stats"),
        enabled=enabled,
        graph_runner=graph_runner,
        layers=layers,
    )


def _end_to_end_rank_gate(
    rows: object,
    *,
    enabled: bool,
    graph_runner: bool,
    proposed: int,
) -> bool:
    if not isinstance(rows, list) or len(rows) != 8:
        return False
    reference_counts = None
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("rank") != rank
            or row.get("world_size") != 8
            or row.get("device") != f"cuda:{rank}"
            or not isinstance(row.get("stats"), dict)
        ):
            return False
        stats = row["stats"]
        requests = stats.get("requests")
        positions = stats.get("positions")
        if (
            "capability_gap" not in stats
            or stats["capability_gap"] is not None
            or stats.get("enabled") is not enabled
            or type(requests) is not int
            or requests < 1
            or positions != proposed + requests
        ):
            return False
        counts = (
            requests,
            positions,
            stats.get("model_calls"),
            stats.get("batched_groups"),
            stats.get("sequential_positions"),
            stats.get("graph_groups"),
        )
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            return False
        if enabled:
            if (
                type(stats.get("batched_groups")) is not int
                or stats["batched_groups"] < 1
                or stats.get("model_calls") != stats["batched_groups"]
                or stats.get("sequential_positions") != 0
                or stats.get("graph_groups")
                != (stats["batched_groups"] if graph_runner else 0)
            ):
                return False
        elif (
            stats.get("model_calls") != positions
            or stats.get("batched_groups") != 0
            or stats.get("sequential_positions") != positions
            or stats.get("graph_groups") != 0
        ):
            return False
        graph = stats.get("graph")
        if graph_runner:
            if not isinstance(graph, dict) or graph.get("eager_fallbacks") != 0:
                return False
        elif graph is not None:
            return False
    return True


def _timing_pair_gate(
    pair: object,
    *,
    rounds: int,
    graph_runner: bool,
) -> bool:
    if not isinstance(pair, dict) or set(pair) != {"orders", "samples", "diagnostic"}:
        return False
    expected_orders = [list(_order_for_round(index)) for index in range(rounds)]
    samples = pair.get("samples")
    diagnostic = pair.get("diagnostic")
    if (
        pair.get("orders") != expected_orders
        or not isinstance(samples, list)
        or len(samples) != rounds * 4
        or not isinstance(diagnostic, dict)
        or diagnostic.get("binding") is not False
        or set(diagnostic) != {"binding", "summary"}
    ):
        return False
    for index, sample in enumerate(samples):
        round_index, slot = divmod(index, 4)
        mode = expected_orders[round_index][slot]
        if (
            not isinstance(sample, dict)
            or sample.get("round") != round_index
            or sample.get("slot") != slot
            or sample.get("mode") != mode
            or type(sample.get("draft_proposed")) is not int
            or sample["draft_proposed"] < 1
            or type(sample.get("draft_accepted")) is not int
            or not 0 <= sample["draft_accepted"] <= sample["draft_proposed"]
            or not isinstance(sample.get("outputs"), list)
        ):
            return False
        timing = sample.get("timing")
        if (
            not isinstance(timing, dict)
            or set(timing) != {"binding", "wall_s", "tpot_s", "tokens_per_s"}
            or timing.get("binding") is not False
            or any(
                type(timing.get(metric)) not in (float, int)
                or timing[metric] <= 0
                for metric in ("wall_s", "tpot_s", "tokens_per_s")
            )
            or not _end_to_end_rank_gate(
                sample.get("rank_stats"),
                enabled=(mode == "candidate"),
                graph_runner=graph_runner,
                proposed=sample["draft_proposed"],
            )
        ):
            return False
    return diagnostic.get("summary") == _diagnostic_summary(samples)


def _parity_gate(measurements: object) -> bool:
    if not isinstance(measurements, dict):
        return False
    cells: list[dict[str, Any]] = []
    structural_targets: list[object] = []
    for runner_name in ("eager", "cuda_graph"):
        runner = measurements.get(runner_name)
        if not isinstance(runner, dict):
            return False
        timing = runner.get("timing")
        structural = runner.get("structural")
        if not isinstance(timing, dict) or not isinstance(structural, dict):
            return False
        samples = timing.get("samples")
        if not isinstance(samples, list):
            return False
        cells.extend(samples)
        for mode in ("sequential", "candidate"):
            cell = structural.get(mode)
            if not isinstance(cell, dict):
                return False
            structural_targets.append(cell.get("targets"))
    if not cells or not structural_targets:
        return False
    first_outputs = cells[0].get("outputs")
    first_proposed = cells[0].get("draft_proposed")
    first_accepted = cells[0].get("draft_accepted")
    return (
        isinstance(first_outputs, list)
        and len(first_outputs) == REQUEST_COUNT
        and all(
            isinstance(output, list)
            and len(output) == MAX_NEW_TOKENS
            and all(type(token) is int and 0 <= token < 151936 for token in output)
            for output in first_outputs
        )
        and type(first_proposed) is int
        and first_proposed > 0
        and type(first_accepted) is int
        and all(
            cell.get("outputs") == first_outputs
            and cell.get("draft_proposed") == first_proposed
            and cell.get("draft_accepted") == first_accepted
            for cell in cells
        )
        and all(target == structural_targets[0] for target in structural_targets)
    )


def _native_ragged_micro_gate(
    cell: object,
    prompts: object,
    *,
    rounds: int,
) -> bool:
    if (
        not isinstance(cell, dict)
        or set(cell)
        != {
            "available",
            "device",
            "checkpoint_layout",
            "geometry",
            "parity",
            "timing",
        }
        or cell.get("available") is not True
        or cell.get("device") != "cuda:0"
        or cell.get("checkpoint_layout")
        != {
            "architecture": "Qwen3ForCausalLM",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        }
        or not isinstance(prompts, list)
        or len(prompts) != REQUEST_COUNT
    ):
        return False
    expected_pages, _ = _micro_page_geometry(prompts)
    geometry = cell.get("geometry")
    if not isinstance(geometry, list) or len(geometry) != REQUEST_COUNT:
        return False
    for index, (row, prompt, pages) in enumerate(
        zip(geometry, prompts, expected_pages, strict=True)
    ):
        prompt_len = len(prompt["prompt_token_ids"])
        inputs = [
            prompt["draft_probe_committed"],
            *prompt["expected_ngram_draft"],
        ]
        if row != {
            "request_index": index,
            "prompt_length": prompt_len,
            "page_table": list(pages),
            "target_input_ids": inputs,
            "positions": list(
                range(prompt_len, prompt_len + SPECULATIVE_TOKENS + 1)
            ),
            "seq_lens": list(
                range(prompt_len + 1, prompt_len + SPECULATIVE_TOKENS + 2)
            ),
            "write_from": prompt_len,
        }:
            return False
    parity = cell.get("parity")
    if not isinstance(parity, dict) or set(parity) != {
        "binding",
        "flattened_selected_tokens",
        "native_ragged_selected_tokens",
        "selected_tokens_equal",
        "flattened_kv_sha256",
        "native_ragged_kv_sha256",
        "kv_digest_match_diagnostic",
        "flattened_kv_shape",
        "native_ragged_kv_shape",
        "kv_shapes_equal",
        "flattened_target_slots_finite",
        "native_ragged_target_slots_finite",
        "kv_allclose_2e2_diagnostic",
        "kv_max_abs_diff_diagnostic",
        "kv_nrmse_diagnostic",
        "kv_cosine_similarity_diagnostic",
    }:
        return False
    flattened = parity["flattened_selected_tokens"]
    ragged = parity["native_ragged_selected_tokens"]
    expected_kv_shape = {
        "k": [64, REQUEST_COUNT * (SPECULATIVE_TOKENS + 1), 8, 128],
        "v": [64, REQUEST_COUNT * (SPECULATIVE_TOKENS + 1), 8, 128],
    }
    if (
        parity["binding"] is not True
        or parity["selected_tokens_equal"] is not True
        or flattened != ragged
        or not isinstance(flattened, list)
        or len(flattened) != REQUEST_COUNT
        or any(
            not isinstance(row, list)
            or len(row) != SPECULATIVE_TOKENS + 1
            or any(type(token) is not int or not 0 <= token < 151936 for token in row)
            for row in flattened
        )
        or not _hex_digest(parity["flattened_kv_sha256"])
        or not _hex_digest(parity["native_ragged_kv_sha256"])
        or parity["kv_digest_match_diagnostic"]
        is not (
            parity["native_ragged_kv_sha256"]
            == parity["flattened_kv_sha256"]
        )
        or parity["flattened_kv_shape"] != expected_kv_shape
        or parity["native_ragged_kv_shape"] != expected_kv_shape
        or parity["kv_shapes_equal"] is not True
        or parity["flattened_target_slots_finite"] is not True
        or parity["native_ragged_target_slots_finite"] is not True
        or type(parity["kv_allclose_2e2_diagnostic"]) is not bool
        or type(parity["kv_max_abs_diff_diagnostic"]) not in (int, float)
        or not 0 <= parity["kv_max_abs_diff_diagnostic"] < float("inf")
        or type(parity["kv_nrmse_diagnostic"]) not in (int, float)
        or not 0 <= parity["kv_nrmse_diagnostic"] < float("inf")
        or type(parity["kv_cosine_similarity_diagnostic"]) not in (int, float)
        or not -1 <= parity["kv_cosine_similarity_diagnostic"] <= 1
    ):
        return False
    timing = cell.get("timing")
    if (
        not isinstance(timing, dict)
        or set(timing) != {"binding", "orders", "samples", "summary"}
        or timing["binding"] is not False
    ):
        return False
    expected_orders = []
    for round_index in range(rounds):
        expected_orders.append(
            [
                "flattened_decode",
                "native_ragged_prefill",
                "native_ragged_prefill",
                "flattened_decode",
            ]
            if round_index % 2 == 0
            else [
                "native_ragged_prefill",
                "flattened_decode",
                "flattened_decode",
                "native_ragged_prefill",
            ]
        )
    samples = timing.get("samples")
    if (
        timing.get("orders") != expected_orders
        or not isinstance(samples, list)
        or len(samples) != rounds * 4
    ):
        return False
    sample_digests: set[str] = set()
    for index, sample in enumerate(samples):
        round_index, slot = divmod(index, 4)
        if (
            not isinstance(sample, dict)
            or sample.get("round") != round_index
            or sample.get("slot") != slot
            or sample.get("mode") != expected_orders[round_index][slot]
            or type(sample.get("wall_s")) not in (int, float)
            or sample["wall_s"] <= 0
            or type(sample.get("cuda_ms")) not in (int, float)
            or sample["cuda_ms"] <= 0
            or not _hex_digest(sample.get("selected_token_sha256"))
        ):
            return False
        sample_digests.add(sample["selected_token_sha256"])
    return (
        len(sample_digests) == 1
        and timing.get("summary") == _micro_summary(samples)
    )


def _timing_policy_gate(payload: dict[str, Any]) -> bool:
    policy = payload.get("gate_policy")
    config = payload.get("config", {})
    return (
        isinstance(config, dict)
        and config.get("timing_is_diagnostic") is True
        and policy
        == {
            "timing_binding": False,
            "binding_basis": [
                "source_checkpoint_hardware_provenance",
                "all_rank_model_call_and_position_counts",
                "zero_cuda_graph_fallback",
                "end_to_end_and_target_token_parity",
                "single_gpu_flattened_vs_native_ragged_token_and_kv_write_coverage",
            ],
            "diagnostic_only": [
                "wall_s",
                "tpot_s",
                "tokens_per_s",
                "cuda_ms",
                "median",
                "mad",
                "kv_digest_match_diagnostic",
                "kv_allclose_2e2_diagnostic",
                "kv_max_abs_diff_diagnostic",
                "kv_nrmse_diagnostic",
                "kv_cosine_similarity_diagnostic",
            ],
        }
    )


def _evaluate(payload: dict[str, Any]) -> dict[str, bool]:
    config = payload.get("config", {})
    checkpoint = payload.get("checkpoint", {})
    checkpoint_start = (
        checkpoint.get("measurement_start", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    layers = checkpoint_start.get("layout", {}).get("num_hidden_layers")
    measurements = payload.get("measurements", {})
    eager = measurements.get("eager", {}) if isinstance(measurements, dict) else {}
    graph = (
        measurements.get("cuda_graph", {})
        if isinstance(measurements, dict)
        else {}
    )
    eager_structural = eager.get("structural", {}) if isinstance(eager, dict) else {}
    graph_structural = graph.get("structural", {}) if isinstance(graph, dict) else {}
    tp = config.get("tp") if isinstance(config, dict) else None
    rounds = config.get("rounds") if isinstance(config, dict) else None
    topology = payload.get("topology", {})
    return {
        "schema_and_kind": (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("measurement_kind") == MEASUREMENT_KIND
        ),
        "formal_geometry": (
            config
            == {
                "tp": 8,
                "num_pages": NUM_PAGES,
                "page_size": PAGE_SIZE,
                "request_count": REQUEST_COUNT,
                "speculative_tokens": SPECULATIVE_TOKENS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "pattern_repeats": PATTERN_REPEATS,
                "rounds": FORMAL_ROUNDS,
                "graph_max_batch": GRAPH_MAX_BATCH,
                "graph_max_pages": GRAPH_MAX_PAGES,
                "timing_is_diagnostic": True,
            }
        ),
        "qwen3_32b_tp8": (
            tp == 8 and layers == 64 and _checkpoint_gate(checkpoint)
        ),
        "clean_committed_source_stable": _source_gate(payload.get("source")),
        "prompt_and_draft_provenance": _prompt_rows_gate(payload.get("prompts")),
        "sequential_eager_counts": _structural_cell_gate(
            eager_structural.get("sequential"),
            prefix="structural-eager-sequential",
            enabled=False,
            graph_runner=False,
            layers=layers,
        )
        if type(layers) is int
        else False,
        "batched_eager_counts": _structural_cell_gate(
            eager_structural.get("candidate"),
            prefix="structural-eager-candidate",
            enabled=True,
            graph_runner=False,
            layers=layers,
        )
        if type(layers) is int
        else False,
        "sequential_graph_counts": _structural_cell_gate(
            graph_structural.get("sequential"),
            prefix="structural-cuda-graph-sequential",
            enabled=False,
            graph_runner=True,
            layers=layers,
        )
        if type(layers) is int
        else False,
        "batched_graph_counts_without_fallback": _structural_cell_gate(
            graph_structural.get("candidate"),
            prefix="structural-cuda-graph-candidate",
            enabled=True,
            graph_runner=True,
            layers=layers,
        )
        if type(layers) is int
        else False,
        "eager_abba_baab_complete": _timing_pair_gate(
            eager.get("timing"),
            rounds=rounds,
            graph_runner=False,
        )
        if type(rounds) is int
        else False,
        "graph_abba_baab_complete": _timing_pair_gate(
            graph.get("timing"),
            rounds=rounds,
            graph_runner=True,
        )
        if type(rounds) is int
        else False,
        "end_to_end_and_target_parity": _parity_gate(measurements),
        "tp_topology_complete": (
            isinstance(topology, dict)
            and _topology_gate(topology.get("eager"), 8)
            and _topology_gate(topology.get("cuda_graph"), 8)
            and topology.get("eager") == topology.get("cuda_graph")
        ),
        "hardware_profile_complete": _hardware_gate(payload.get("hardware")),
        "timing_strictly_non_binding": _timing_policy_gate(payload),
        "native_ragged_microcomparison_exact": _native_ragged_micro_gate(
            payload.get("native_ragged_microcomparison"),
            payload.get("prompts"),
            rounds=rounds,
        )
        if type(rounds) is int
        else False,
    }


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    import gc

    import torch

    from kairyu.engine.tokenizer import resolve_tokenizer

    _ensure_python_bin_on_path()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.tp:
        raise RuntimeError(f"tp={args.tp} requires {args.tp} visible CUDA devices")
    if (
        args.tp != 8
        or args.num_pages != NUM_PAGES
        or args.page_size != PAGE_SIZE
        or args.rounds != FORMAL_ROUNDS
    ):
        raise RuntimeError(
            "formal #215 evidence requires --tp 8 --num-pages 4096 "
            "--page-size 16 --rounds 2"
        )
    source_start = _source_snapshot()
    if not _source_snapshot_is_clean_and_bound(source_start):
        raise RuntimeError(
            "formal evidence requires a clean commit containing every bound source"
        )
    hardware = _hardware()
    if not _hardware_gate(hardware):
        raise RuntimeError(
            "formal evidence requires exactly eight visible RTX PRO 6000 "
            "Blackwell Server Edition GPUs"
        )
    model_path = Path(args.model_path).resolve()
    checkpoint_start = _checkpoint_manifest(model_path)
    if not _checkpoint_snapshot_gate(checkpoint_start):
        raise RuntimeError("--model-path is not the exact reviewed Qwen3-32B checkpoint")
    prompt_rows = _prompt_rows(resolve_tokenizer(str(model_path)))

    eager = _measure_launcher(
        model_path,
        graph=False,
        prompt_rows=prompt_rows,
        rounds=args.rounds,
    )
    graph = _measure_launcher(
        model_path,
        graph=True,
        prompt_rows=prompt_rows,
        rounds=args.rounds,
    )
    # The full single-GPU model follows two TP launchers in the same rank-0
    # process.  Clear any cached shard/graph allocator blocks before loading it.
    gc.collect()
    torch.cuda.empty_cache()
    native_ragged_microcomparison = _measure_native_ragged_microcomparison(
        model_path,
        prompt_rows,
        rounds=args.rounds,
    )

    checkpoint_end = _checkpoint_manifest(model_path)
    checkpoint = {
        "measurement_start": checkpoint_start,
        "measurement_end": checkpoint_end,
        "stable_across_measurement": checkpoint_start == checkpoint_end,
    }
    if not _checkpoint_gate(checkpoint):
        raise RuntimeError("checkpoint content changed while formal evidence was running")
    source_end = _source_snapshot()
    source = {
        "measurement_start": source_start,
        "measurement_end": source_end,
        "stable_across_measurement": source_start == source_end,
    }
    if not _source_gate(source):
        raise RuntimeError("bound source or commit changed while formal evidence was running")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "tp": args.tp,
            "num_pages": args.num_pages,
            "page_size": args.page_size,
            "request_count": REQUEST_COUNT,
            "speculative_tokens": SPECULATIVE_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "pattern_repeats": PATTERN_REPEATS,
            "rounds": args.rounds,
            "graph_max_batch": GRAPH_MAX_BATCH,
            "graph_max_pages": GRAPH_MAX_PAGES,
            "timing_is_diagnostic": True,
        },
        "gate_policy": {
            "timing_binding": False,
            "binding_basis": [
                "source_checkpoint_hardware_provenance",
                "all_rank_model_call_and_position_counts",
                "zero_cuda_graph_fallback",
                "end_to_end_and_target_token_parity",
                "single_gpu_flattened_vs_native_ragged_token_and_kv_write_coverage",
            ],
            "diagnostic_only": [
                "wall_s",
                "tpot_s",
                "tokens_per_s",
                "cuda_ms",
                "median",
                "mad",
                "kv_digest_match_diagnostic",
                "kv_allclose_2e2_diagnostic",
                "kv_max_abs_diff_diagnostic",
                "kv_nrmse_diagnostic",
                "kv_cosine_similarity_diagnostic",
            ],
        },
        "source": source,
        "checkpoint": checkpoint,
        "hardware": hardware,
        "prompts": prompt_rows,
        "topology": {
            "eager": eager.pop("topology"),
            "cuda_graph": graph.pop("topology"),
        },
        "measurements": {
            "eager": eager,
            "cuda_graph": graph,
        },
        "native_ragged_microcomparison": native_ragged_microcomparison,
    }
    payload["checks"] = _evaluate(payload)
    payload["passed"] = all(payload["checks"].values())
    return payload


def _verify(artifact: Path, model_path: Path | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text())
    raw_checks = _evaluate(payload)
    stored_matches = (
        payload.get("checks") == raw_checks
        and payload.get("passed") == all(raw_checks.values())
    )
    checks = dict(raw_checks)
    checks["stored_replay_matches"] = stored_matches
    checks["source_commit_replay"] = _source_gate(payload.get("source"))
    checks["hardware_replay"] = (
        _hardware_gate(payload.get("hardware"))
        and _hardware() == payload.get("hardware")
    )
    if model_path is not None:
        replay = _checkpoint_manifest(model_path.resolve())
        checkpoint = payload.get("checkpoint", {})
        checks["checkpoint_replay"] = (
            isinstance(checkpoint, dict)
            and replay == checkpoint.get("measurement_start")
            and replay == checkpoint.get("measurement_end")
            and _checkpoint_snapshot_gate(replay)
        )
        checks["prompt_tokenizer_replay"] = _prompt_tokenizer_replay(
            payload.get("prompts"),
            model_path.resolve(),
        )
    else:
        checks["checkpoint_replay"] = False
        checks["prompt_tokenizer_replay"] = False
    return {
        "artifact": str(artifact),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num-pages", type=int, default=NUM_PAGES)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
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
        if not args.model_path or args.output is None:
            parser.error("measurement requires --model-path and --output")
        result = _measure(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.assert_gate and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
