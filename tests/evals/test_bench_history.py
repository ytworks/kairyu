"""Source-bound append-only scoreboard history and resume preflight."""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import stat
import threading
from pathlib import Path

import pytest

from evals import history
from evals.aggregate import build_scoreboard
from evals.sampling import combine_attempts, sampling_metrics, sampling_summary
from evals.store import ResultStore
from evals.types import ItemResult, PairResult, SamplingAttemptResult


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(identity: dict) -> str:
    return hashlib.sha256(_canonical(identity)).hexdigest()


def _commit(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:40]


def _metadata(
    run_id: str = "run-1",
    *,
    seed: str = "a",
    created_at: str = "2026-08-05T00:00:00+00:00",
    available: bool = True,
) -> dict:
    config = {
        "suite": "core",
        "offline_fixtures": False,
        "execution": {
            "runner": "local",
            "image": None,
            "cpus": 1.0,
            "pids_limit": 64,
            "disk_mb": 256,
            "pull_policy": "never",
        },
        "targets": [
            {
                "name": "target-a",
                "model": "model-a",
                "base_url": "http://127.0.0.1:8000/v1",
            }
        ],
        "only": ["gsm8k"],
        "run_id": run_id,
        "results_dir": "ignored/location",
        "cache_dir": None,
        "rerun": False,
        "download": False,
        "progress": True,
    }
    immutable = {
        key: value
        for key, value in config.items()
        if key not in {"run_id", "results_dir", "cache_dir", "rerun", "download", "progress"}
    }
    identity = {
        "config": immutable,
        "adapters": [
            {
                "name": "gsm8k",
                "dataset": "openai/gsm8k",
                "revision": seed,
                "sha256": hashlib.sha256(f"dataset:{seed}".encode()).hexdigest(),
                "assets": [],
                "history_provenance": {
                    "complete": True,
                    "reason": None,
                    "requires_content_addressed_execution": False,
                },
                "evaluation_protocol": {
                    "resources": [
                        {
                            "package": package,
                            "resource": resource,
                            "sha256": hashlib.sha256(
                                f"{package}/{resource}".encode()
                            ).hexdigest(),
                        }
                        for package, resource in (
                            ("evals.adapters", "gsm8k.py"),
                            ("evals.adapters", "base.py"),
                            ("evals", "aggregate.py"),
                            ("evals", "cache.py"),
                            ("evals", "runner.py"),
                            ("evals", "types.py"),
                            ("evals", "targets.py"),
                        )
                    ],
                    "distributions": [],
                    "executables": [],
                },
            }
        ],
    }
    environment = {
        "git_commit": _commit(seed),
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": hashlib.sha256(f"tree:{seed}".encode()).hexdigest(),
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "created_at": created_at,
        "execution": {
            "runner": "local",
            "available": available,
            "availability_detail": "available" if available else "temporarily absent",
        },
    }
    return {
        "fingerprint": _fingerprint(identity),
        "identity": identity,
        "config": config,
        "environment": environment,
        "run_id": run_id,
    }


def _scoreboard(
    metadata: dict,
    *,
    score: float = 0.5,
    status: str = "completed",
) -> dict:
    return {
        "schema_version": 1,
        "fingerprint": metadata["fingerprint"],
        "run_id": metadata["run_id"],
        "suite": metadata["config"]["suite"],
        "environment": metadata["environment"],
        "config": metadata["config"],
        "benchmarks": ["gsm8k"],
        "display_names": {"gsm8k": "GSM8K"},
        "targets": ["target-a"],
        "cells": {
            "gsm8k": {
                "target-a": {
                    "status": status,
                    "score": score,
                    "n": 2,
                    "n_scored": 2,
                    "confidence_interval": None,
                    "reason": None,
                    "comparable": True,
                    "incomparable_reasons": [],
                    "cross_run_policy": "allowed",
                    "cross_run_reason": None,
                    "footnotes": [],
                }
            }
        },
        "footnotes": [],
    }


def _refresh_metadata_identity(metadata: dict) -> None:
    metadata["identity"]["config"] = {
        key: value
        for key, value in metadata["config"].items()
        if key not in {"run_id", "results_dir", "cache_dir", "rerun", "download", "progress"}
    }
    metadata["fingerprint"] = _fingerprint(metadata["identity"])


def _sampling_metadata() -> dict:
    metadata = _metadata()
    metadata["config"]["attempts"] = 2
    metadata["config"]["sampling_protocol"] = "grouped-chat-seeds-v1"
    metadata["identity"]["adapters"][0]["binary_outcomes"] = True
    metadata["identity"]["adapters"][0]["evaluation_protocol"]["resources"].append(
        {
            "package": "evals",
            "resource": "sampling.py",
            "sha256": hashlib.sha256(b"evals/sampling.py").hexdigest(),
        }
    )
    _refresh_metadata_identity(metadata)
    return metadata


def _pair(
    metadata: dict,
    *,
    score: float | int = 0.5,
    status: str = "completed",
    reason: str | None = None,
) -> PairResult:
    return PairResult(
        benchmark="gsm8k",
        target="target-a",
        run_fingerprint=metadata["fingerprint"],
        source_identity=history.source_identity(metadata["environment"]),
        status=status,
        reason=reason,
        metrics={"score": score, "n_total": 2, "n_scored": 2},
    )


def _sampling_pair(metadata: dict) -> PairResult:
    seeds = (0, 1)
    items = tuple(
        combine_attempts(
            item_id,
            tuple(
                SamplingAttemptResult(
                    attempt=index + 1,
                    seed=seed,
                    result=ItemResult(
                        item_id=item_id,
                        status="completed",
                        score=score,
                    ),
                )
                for index, (seed, score) in enumerate(zip(seeds, scores, strict=True))
            ),
        )
        for item_id, scores in (("item-a", (1.0, 0.0)), ("item-b", (1.0, 1.0)))
    )
    summary = sampling_summary(items, seeds=seeds, declared_binary=True)
    return PairResult(
        benchmark="gsm8k",
        target="target-a",
        run_fingerprint=metadata["fingerprint"],
        source_identity=history.source_identity(metadata["environment"]),
        status="completed",
        metrics={
            "score": summary["mean"],
            "n_total": len(items),
            "n_scored": len(items),
            "n_unjudged": 0,
            "n_skipped": 0,
            "n_failed": 0,
            "sampling_complete": 1,
            **sampling_metrics(summary),
        },
        items=items,
        methodology={
            "temperature": 0.0,
            "sampling": {
                "mode": "adapter",
                "attempts": len(seeds),
                "seeds": list(seeds),
                "seed_source": "generated from zero",
                "wire_omitted": [],
                "recommended_defaults_attested": False,
            }
        },
    )


def _sampling_scoreboard(metadata: dict, pair: PairResult) -> dict:
    return build_scoreboard(
        run_id=metadata["run_id"],
        suite=metadata["config"]["suite"],
        config=metadata["config"],
        environment=metadata["environment"],
        fingerprint=metadata["fingerprint"],
        pairs=[pair],
        targets=["target-a"],
    )


def _append_worker(results_dir: str, index: int) -> None:
    run_id = f"run-{index}"
    metadata = _metadata(run_id, seed=f"worker-{index}")
    ResultStore(results_dir, run_id).append_scoreboard_index(
        _scoreboard(metadata), metadata, (_pair(metadata),)
    )


def _publish_under_fixed_lock(
    results_dir: str,
    raw: bytes,
    ready,
    release,
) -> None:
    root = Path(results_dir)
    descriptor = os.open(root / history.LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("test did not release history writer")
        (root / history.INDEX_NAME).write_bytes(raw)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _rewrite_entries(path: Path, entries: list[dict]) -> None:
    path.write_bytes(b"".join(_canonical(entry) + b"\n" for entry in entries))


def _rehash(entry: dict) -> None:
    entry["record_sha256"] = history._record_digest(entry)


def test_append_load_find_and_canonical_framing(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")

    path = store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))

    assert path == tmp_path / history.INDEX_NAME
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    entries = store.load_scoreboard_index()
    assert len(entries) == 1
    entry = entries[0]
    assert raw == _canonical(entry) + b"\n"
    assert entry["git_commit"] == metadata["environment"]["git_commit"]
    assert entry["fingerprint"] == metadata["fingerprint"]
    assert entry["run"] == metadata
    assert entry["scoreboard"] == _scoreboard(metadata)
    assert entry["previous_record_sha256"] is None
    assert store.find_scoreboard_entry("run-1") == entry
    assert store.find_scoreboard_entry("absent") is None


def test_fresh_sampling_claim_is_recomputed_then_omitted_from_schema1_index(tmp_path):
    metadata = _sampling_metadata()
    pair = _sampling_pair(metadata)
    board = _sampling_scoreboard(metadata, pair)
    cell = board["cells"]["gsm8k"]["target-a"]
    assert "sampling_sensitivity" in cell

    store = ResultStore(tmp_path, "run-1")
    store.append_scoreboard_index(board, metadata, (pair,))
    entry = store.load_scoreboard_index()[0]

    assert "sampling_sensitivity" in cell  # caller-owned board was not mutated
    assert "sampling_sensitivity" not in entry["scoreboard"]["cells"]["gsm8k"]["target-a"]
    assert set(entry["pairs"][0]) == history._PAIR_FIELDS


@pytest.mark.parametrize("mutation", ["missing", "forged-pass-at-k", "forged-spread"])
def test_fresh_history_rejects_sampling_claim_that_disagrees_with_raw_pair(
    tmp_path, mutation
):
    metadata = _sampling_metadata()
    pair = _sampling_pair(metadata)
    board = _sampling_scoreboard(metadata, pair)
    cell = board["cells"]["gsm8k"]["target-a"]
    if mutation == "missing":
        del cell["sampling_sensitivity"]
    elif mutation == "forged-pass-at-k":
        cell["sampling_sensitivity"]["pass_at_k"][0]["estimate"] = 0.0
    else:
        cell["sampling_sensitivity"]["sample_sd"] = 0.0

    with pytest.raises(ValueError, match="sampling sensitivity"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(board, metadata, (pair,))
    assert not (tmp_path / history.INDEX_NAME).exists()


def test_stored_schema1_rejects_rehashed_detached_sampling_claim(tmp_path):
    metadata = _sampling_metadata()
    pair = _sampling_pair(metadata)
    board = _sampling_scoreboard(metadata, pair)
    entry = history.build_entry(
        board,
        metadata,
        pairs=(pair,),
        previous_record_sha256=None,
    )
    entry["scoreboard"]["cells"]["gsm8k"]["target-a"]["sampling_sensitivity"] = {
        "pass_at_k": [{"k": 1, "estimate": 0.0}]
    }
    _rehash(entry)
    path = tmp_path / history.INDEX_NAME
    _rewrite_entries(path, [entry])

    with pytest.raises(ValueError, match="stored scoreboard cell .* sampling sensitivity"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


@pytest.mark.parametrize("trigger", ["marker", "target-policy", "raw-evidence"])
def test_fresh_sampling_protocol_requires_sampling_evaluator_identity(tmp_path, trigger):
    metadata = _metadata()
    if trigger == "marker":
        metadata["config"]["attempts"] = 2
        metadata["config"]["sampling_protocol"] = "grouped-chat-seeds-v1"
        _refresh_metadata_identity(metadata)
        pair = _pair(metadata)
        board = _scoreboard(metadata)
    elif trigger == "target-policy":
        metadata["config"]["targets"][0]["temperature"] = 0.6
        _refresh_metadata_identity(metadata)
        pair = _pair(metadata)
        board = _scoreboard(metadata)
    else:
        metadata["config"]["attempts"] = 1
        _refresh_metadata_identity(metadata)
        pair = _sampling_pair(metadata)
        board = _sampling_scoreboard(metadata, pair)

    with pytest.raises(ValueError, match="sampling evaluator resource.*sampling.py"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(board, metadata, (pair,))



def test_legacy_unmarked_multi_trial_history_keeps_ordinary_generative_pairs(tmp_path):
    metadata = _metadata()
    metadata["config"]["attempts"] = 4
    _refresh_metadata_identity(metadata)
    pair = _pair(metadata)
    board = _scoreboard(metadata)

    entry = history.build_entry(
        board,
        metadata,
        pairs=(pair,),
        previous_record_sha256=None,
    )
    _rewrite_entries(tmp_path / history.INDEX_NAME, [entry])

    assert ResultStore(tmp_path, "reader").load_scoreboard_index() == (entry,)


@pytest.mark.parametrize("mutation", ["attempts", "seed", "mode"])
def test_fresh_history_binds_sampling_methodology_to_run_target_config(
    tmp_path, mutation
):
    metadata = _sampling_metadata()
    if mutation == "attempts":
        metadata["config"]["attempts"] = 4
    elif mutation == "seed":
        metadata["config"]["targets"][0]["seed"] = 10
    else:
        metadata["config"]["targets"][0]["sampling_mode"] = "recommended"
    _refresh_metadata_identity(metadata)
    pair = _sampling_pair(metadata)

    with pytest.raises(ValueError, match="sampling methodology .*run config|target config"):
        history.build_entry(
            _sampling_scoreboard(metadata, pair),
            metadata,
            pairs=(pair,),
            previous_record_sha256=None,
        )


def test_fresh_history_rejects_impossible_recommended_target_policy():
    metadata = _sampling_metadata()
    target = metadata["config"]["targets"][0]
    target["sampling_mode"] = "recommended"
    target["temperature"] = 0.7
    _refresh_metadata_identity(metadata)
    pair = _sampling_pair(metadata)

    with pytest.raises(ValueError, match="expected sampling configuration is invalid"):
        history.build_entry(
            _sampling_scoreboard(metadata, pair),
            metadata,
            pairs=(pair,),
            previous_record_sha256=None,
        )


def test_stored_multi_attempt_record_requires_sampling_evaluator_identity(tmp_path):
    metadata = _sampling_metadata()
    pair = _sampling_pair(metadata)
    entry = history.build_entry(
        _sampling_scoreboard(metadata, pair),
        metadata,
        pairs=(pair,),
        previous_record_sha256=None,
    )
    resources = entry["run"]["identity"]["adapters"][0]["evaluation_protocol"]["resources"]
    resources[:] = [resource for resource in resources if resource["resource"] != "sampling.py"]
    fingerprint = _fingerprint(entry["run"]["identity"])
    entry["fingerprint"] = fingerprint
    entry["run"]["fingerprint"] = fingerprint
    entry["scoreboard"]["fingerprint"] = fingerprint
    entry["pairs"][0]["run_fingerprint"] = fingerprint
    _rehash(entry)
    _rewrite_entries(tmp_path / history.INDEX_NAME, [entry])

    with pytest.raises(ValueError, match="sampling evaluator resource.*sampling.py"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_stored_sampling_field_requires_sampling_evaluator_identity(tmp_path):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata),
        metadata,
        pairs=(_pair(metadata),),
        previous_record_sha256=None,
    )
    entry["scoreboard"]["cells"]["gsm8k"]["target-a"]["sampling_sensitivity"] = {}
    _rehash(entry)
    _rewrite_entries(tmp_path / history.INDEX_NAME, [entry])

    with pytest.raises(ValueError, match="sampling evaluator resource.*sampling.py"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def _persist_indexed_pair(tmp_path):
    metadata = _metadata()
    pair = _pair(metadata)
    store = ResultStore(tmp_path, "run-1")
    store.save_pair(pair)
    store.append_scoreboard_index(_scoreboard(metadata), metadata, (pair,))
    entry = store.find_scoreboard_entry("run-1")
    assert entry is not None
    return store, entry, pair


def test_load_indexed_pair_revalidates_complete_raw_binding(tmp_path):
    store, entry, pair = _persist_indexed_pair(tmp_path)

    assert store.load_indexed_pair(entry, "gsm8k", "target-a") == pair


def test_load_indexed_pair_rejects_missing_and_complete_binding_tamper(tmp_path):
    metadata = _metadata()
    pair = _pair(metadata)
    missing_store = ResultStore(tmp_path / "missing", "run-1")
    missing_store.append_scoreboard_index(_scoreboard(metadata), metadata, (pair,))
    missing_entry = missing_store.find_scoreboard_entry("run-1")
    assert missing_entry is not None

    with pytest.raises(ValueError, match="indexed pair .* is missing"):
        missing_store.load_indexed_pair(missing_entry, "gsm8k", "target-a")

    store, entry, original = _persist_indexed_pair(tmp_path / "tampered")
    changed = original.model_copy(update={"annotations": ("changed after indexing",)})
    store.save_pair(changed)

    with pytest.raises(ValueError, match="complete indexed binding"):
        store.load_indexed_pair(entry, "gsm8k", "target-a")


def test_load_indexed_pair_requires_entry_from_authoritative_current_index(tmp_path):
    store, entry, original = _persist_indexed_pair(tmp_path)
    forged_pair = original.model_copy(update={"annotations": ("forged binding",)})
    forged_entry = history.build_entry(
        entry["scoreboard"],
        entry["run"],
        pairs=(forged_pair,),
        previous_record_sha256=entry["previous_record_sha256"],
    )
    store.save_pair(forged_pair)

    with pytest.raises(ValueError, match="authoritative current scoreboard-index"):
        store.load_indexed_pair(forged_entry, "gsm8k", "target-a")


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            b'{"benchmark":"gsm8k","benchmark":"gsm8k"}',
            id="duplicate-key",
        ),
        pytest.param(b'{"metrics":{"score":NaN}}', id="non-finite"),
        pytest.param(
            None,
            id="excessive-nesting",
        ),
    ],
)
def test_load_indexed_pair_wraps_malformed_raw_json_as_value_error(tmp_path, raw):
    store, entry, pair = _persist_indexed_pair(tmp_path)
    if raw is None:
        deep_value = ("[" * 2_000) + "0" + ("]" * 2_000)
        raw = pair.model_dump_json().replace(
            '"methodology":{}',
            f'"methodology":{{"deep":{deep_value}}}',
        ).encode()
    store.pair_path("gsm8k", "target-a").write_bytes(raw)

    with pytest.raises(ValueError, match="strict JSON|supported nesting depth"):
        store.load_indexed_pair(entry, "gsm8k", "target-a")


def test_load_indexed_pair_rejects_wrong_entry_run_and_unknown_cell(tmp_path):
    store, entry, _ = _persist_indexed_pair(tmp_path)
    malformed = json.loads(json.dumps(entry))
    malformed["unexpected"] = True

    with pytest.raises(ValueError, match="record fields"):
        store.load_indexed_pair(malformed, "gsm8k", "target-a")

    with pytest.raises(ValueError, match="does not match indexed scoreboard entry"):
        ResultStore(tmp_path, "run-2").load_indexed_pair(
            entry, "gsm8k", "target-a"
        )

    with pytest.raises(ValueError, match="no unique pair binding"):
        store.load_indexed_pair(entry, "unknown-benchmark", "target-a")
    with pytest.raises(ValueError, match="no unique pair binding"):
        store.load_indexed_pair(entry, "gsm8k", "unknown-target")


def test_load_indexed_pair_rejects_duplicate_binding_and_symlink(tmp_path):
    store, entry, pair = _persist_indexed_pair(tmp_path / "duplicate")
    duplicate = json.loads(json.dumps(entry))
    duplicate["pairs"].append(json.loads(json.dumps(duplicate["pairs"][0])))
    _rehash(duplicate)

    with pytest.raises(ValueError, match="duplicate pair binding"):
        store.load_indexed_pair(duplicate, "gsm8k", "target-a")

    symlink_store, symlink_entry, _ = _persist_indexed_pair(tmp_path / "symlink")
    pair_path = symlink_store.pair_path("gsm8k", "target-a")
    outside = tmp_path / "outside-pair.json"
    outside.write_text(pair.model_dump_json(), encoding="utf-8")
    before = outside.read_bytes()
    pair_path.unlink()
    pair_path.symlink_to(outside)

    with pytest.raises(ValueError, match="cannot be opened safely"):
        symlink_store.load_indexed_pair(symlink_entry, "gsm8k", "target-a")

    assert outside.read_bytes() == before


def test_load_indexed_pair_pins_intermediate_directory_before_final_open(
    tmp_path,
    monkeypatch,
):
    store, entry, _ = _persist_indexed_pair(tmp_path / "results")
    pair_path = store.pair_path("gsm8k", "target-a")
    renamed_directory = pair_path.parent.with_name(f"{pair_path.parent.name}-original")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_pair = outside_directory / pair_path.name
    outside_pair.write_bytes(b"external bytes must never be read")
    outside_stat = outside_pair.stat()

    real_open = os.open
    real_read = os.read
    swapped = False
    external_reads: list[int] = []

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and path == pair_path.name and dir_fd is not None:
            pair_path.parent.rename(renamed_directory)
            pair_path.parent.symlink_to(outside_directory, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def recording_read(descriptor, size):
        opened = os.fstat(descriptor)
        if (
            opened.st_dev == outside_stat.st_dev
            and opened.st_ino == outside_stat.st_ino
        ):
            external_reads.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(os, "read", recording_read)

    with pytest.raises(ValueError, match="path changed while reading"):
        store.load_indexed_pair(entry, "gsm8k", "target-a")

    assert swapped is True
    assert external_reads == []
    assert outside_pair.read_bytes() == b"external bytes must never be read"


def test_exact_duplicate_is_noop_but_conflict_preserves_bytes(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")
    board = _scoreboard(metadata)
    path = store.append_scoreboard_index(board, metadata, (_pair(metadata),))
    before = path.read_bytes()

    assert store.append_scoreboard_index(board, metadata, (_pair(metadata),)) == path
    assert path.read_bytes() == before


def test_numeric_type_change_is_not_a_canonical_duplicate(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")
    integer_board = _scoreboard(metadata, score=1)
    path = store.append_scoreboard_index(integer_board, metadata, (_pair(metadata, score=1),))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="different evidence"):
        store.append_scoreboard_index(
            _scoreboard(metadata, score=1.0), metadata, (_pair(metadata, score=1.0),)
        )

    assert path.read_bytes() == before

    with pytest.raises(ValueError, match="different evidence"):
        store.append_scoreboard_index(
            _scoreboard(metadata, score=0.75), metadata, (_pair(metadata, score=0.75),)
        )
    assert path.read_bytes() == before


def test_two_entries_form_a_valid_hash_chain(tmp_path):
    first = _metadata("run-1", seed="first")
    second = _metadata("run-2", seed="second")
    ResultStore(tmp_path, "run-1").append_scoreboard_index(
        _scoreboard(first), first, (_pair(first),)
    )
    ResultStore(tmp_path, "run-2").append_scoreboard_index(
        _scoreboard(second), second, (_pair(second),)
    )

    entries = ResultStore(tmp_path, "reader").load_scoreboard_index()
    assert entries[1]["previous_record_sha256"] == entries[0]["record_sha256"]


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"{}",
        b"{}\r\n",
        b'{"x":NaN}\n',
        b'{"x":1,"x":2}\n',
        b"\xff\n",
    ],
    ids=["blank", "torn", "crlf", "nonfinite", "duplicate-key", "utf8"],
)
def test_strict_loader_rejects_invalid_jsonl_framing(tmp_path, raw):
    path = tmp_path / history.INDEX_NAME
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="scoreboard history"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()

    assert path.read_bytes() == raw


def test_unknown_schema_and_record_tamper_are_rejected(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")
    path = store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))
    original = json.loads(path.read_text().strip())

    unknown = json.loads(json.dumps(original))
    unknown["index_schema_version"] = 2
    _rewrite_entries(path, [unknown])
    with pytest.raises(ValueError, match="schema"):
        store.load_scoreboard_index()

    changed = json.loads(json.dumps(original))
    changed["scoreboard"]["cells"]["gsm8k"]["target-a"]["score"] = 0.9
    _rewrite_entries(path, [changed])
    with pytest.raises(ValueError, match="record_sha256"):
        store.load_scoreboard_index()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda entry: entry.__setitem__("index_schema_version", 1.0),
            "schema",
        ),
        (
            lambda entry: entry["scoreboard"].__setitem__("schema_version", 1.0),
            "schema_version",
        ),
        (
            lambda entry: entry["scoreboard"]["cells"]["gsm8k"]["target-a"].__setitem__(
                "status", "bogus"
            ),
            "status",
        ),
        (
            lambda entry: entry["scoreboard"]["cells"]["gsm8k"]["target-a"].__setitem__(
                "status", []
            ),
            "status",
        ),
    ],
)
def test_rehashed_schema_types_and_cell_status_still_fail_closed(tmp_path, mutation, match):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata), metadata, pairs=(_pair(metadata),), previous_record_sha256=None
    )
    mutation(entry)
    _rehash(entry)
    _rewrite_entries(tmp_path / history.INDEX_NAME, [entry])

    with pytest.raises(ValueError, match=match):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_loader_wraps_escaped_lone_surrogate_as_history_error(tmp_path):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata), metadata, pairs=(_pair(metadata),), previous_record_sha256=None
    )
    raw = _canonical(entry).replace(b'"GSM8K"', b'"\\ud800"', 1) + b"\n"
    (tmp_path / history.INDEX_NAME).write_bytes(raw)

    with pytest.raises(ValueError, match="scoreboard history"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_chain_detects_middle_deletion_and_reordering(tmp_path):
    for index in range(3):
        metadata = _metadata(f"run-{index}", seed=f"seed-{index}")
        ResultStore(tmp_path, metadata["run_id"]).append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )
    path = tmp_path / history.INDEX_NAME
    entries = [json.loads(line) for line in path.read_text().splitlines()]

    _rewrite_entries(path, [entries[0], entries[2]])
    with pytest.raises(ValueError, match="chain"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()

    _rewrite_entries(path, [entries[1], entries[0], entries[2]])
    with pytest.raises(ValueError, match="chain"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_forged_duplicate_key_and_run_id_are_rejected(tmp_path):
    first_meta = _metadata("run-1", seed="first")
    first = history.build_entry(
        _scoreboard(first_meta),
        first_meta,
        pairs=(_pair(first_meta),),
        previous_record_sha256=None,
    )
    duplicate_key = json.loads(json.dumps(first))
    duplicate_key["previous_record_sha256"] = first["record_sha256"]
    _rehash(duplicate_key)
    path = tmp_path / history.INDEX_NAME
    _rewrite_entries(path, [first, duplicate_key])

    with pytest.raises(ValueError, match="duplicate scoreboard key"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()

    second_meta = _metadata("run-1", seed="second")
    duplicate_run = history.build_entry(
        _scoreboard(second_meta),
        second_meta,
        pairs=(_pair(second_meta),),
        previous_record_sha256=first["record_sha256"],
    )
    _rewrite_entries(path, [first, duplicate_run])
    with pytest.raises(ValueError, match="duplicate run_id"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_rehashed_identity_matrix_and_pair_tampering_are_rejected(tmp_path):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata), metadata, pairs=(_pair(metadata),), previous_record_sha256=None
    )
    path = tmp_path / history.INDEX_NAME

    matrix = json.loads(json.dumps(entry))
    matrix["scoreboard"]["benchmarks"] = ["other"]
    matrix["scoreboard"]["display_names"] = {"other": "Other"}
    matrix["scoreboard"]["cells"] = {"other": matrix["scoreboard"]["cells"]["gsm8k"]}
    matrix["pairs"][0]["benchmark"] = "other"
    _rehash(matrix)
    _rewrite_entries(path, [matrix])
    with pytest.raises(ValueError, match="identity adapters"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()

    pair = json.loads(json.dumps(entry))
    pair["pairs"][0]["run_fingerprint"] = "f" * 64
    _rehash(pair)
    _rewrite_entries(path, [pair])
    with pytest.raises(ValueError, match="pair .* fingerprint"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()

    missing_display_names = json.loads(json.dumps(entry))
    missing_display_names["scoreboard"].pop("display_names")
    _rehash(missing_display_names)
    _rewrite_entries(path, [missing_display_names])
    with pytest.raises(ValueError, match="display_names"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda metadata: metadata["environment"].__setitem__("source_tree_clean", False),
            "clean",
        ),
        (
            lambda metadata: metadata["environment"].__setitem__(
                "source_kind", "unverified-package"
            ),
            "git-checkout",
        ),
        (
            lambda metadata: metadata["environment"].__setitem__("git_commit", "abc123"),
            "full lowercase",
        ),
        (
            lambda metadata: metadata["environment"].__setitem__("source_tree_sha256", "bad"),
            "SHA-256",
        ),
    ],
)
def test_index_requires_clean_full_git_source(tmp_path, mutation, match):
    metadata = _metadata()
    mutation(metadata)
    store = ResultStore(tmp_path, "run-1")

    with pytest.raises(ValueError, match=match):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))

    assert not (tmp_path / history.INDEX_NAME).exists()


def test_append_rejects_scoreboard_and_pair_inconsistency(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")

    wrong_fingerprint = _scoreboard(metadata)
    wrong_fingerprint["fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="scoreboard fingerprint"):
        store.append_scoreboard_index(wrong_fingerprint, metadata, (_pair(metadata),))

    wrong_targets = _scoreboard(metadata)
    wrong_targets["targets"] = ["other"]
    wrong_targets["cells"]["gsm8k"] = {"other": wrong_targets["cells"]["gsm8k"].pop("target-a")}
    with pytest.raises(ValueError, match="config target labels"):
        store.append_scoreboard_index(wrong_targets, metadata, (_pair(metadata),))

    bad_pair = _pair(metadata).model_copy(update={"run_fingerprint": "f" * 64})
    with pytest.raises(ValueError, match="pair .* fingerprint"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (bad_pair,))

    bad_source = _pair(metadata).model_copy(
        update={
            "source_identity": {
                **history.source_identity(metadata["environment"]),
                "source_tree_clean": False,
            }
        }
    )
    with pytest.raises(ValueError, match="clean run source identity"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (bad_source,))

    out_of_range = _scoreboard(metadata, score=1.5)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        store.append_scoreboard_index(out_of_range, metadata, (_pair(metadata),))

    with pytest.raises(ValueError, match="complete PairResult evidence"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata)

    wrong_score = _pair(metadata, score=0.25)
    with pytest.raises(ValueError, match="summary disagrees"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (wrong_score,))

    huge_score = _pair(metadata, score=10**4000)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (huge_score,))

    wrong_status = _pair(metadata, status="partial")
    with pytest.raises(ValueError, match="summary disagrees"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (wrong_status,))


@pytest.mark.parametrize(
    "display_names",
    [None, {}, {"gsm8k": ""}, {"gsm8k": 1}],
    ids=["missing", "empty", "empty-name", "non-string-name"],
)
def test_append_requires_complete_nonempty_display_names(tmp_path, display_names):
    metadata = _metadata()
    board = _scoreboard(metadata)
    if display_names is None:
        board.pop("display_names")
    else:
        board["display_names"] = display_names

    with pytest.raises(ValueError, match="display_names"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            board,
            metadata,
            (_pair(metadata),),
        )


@pytest.mark.parametrize(
    "footnotes",
    [{}, 1, ["valid", 2]],
    ids=["mapping", "scalar", "non-string-item"],
)
def test_append_requires_string_list_scoreboard_footnotes(tmp_path, footnotes):
    metadata = _metadata()
    board = _scoreboard(metadata)
    board["footnotes"] = footnotes

    with pytest.raises(ValueError, match="footnotes"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            board,
            metadata,
            (_pair(metadata),),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("available", "yes"), ("available", None), ("availability_detail", 1)],
)
def test_append_requires_typed_execution_observation(tmp_path, field, value):
    metadata = _metadata()
    metadata["environment"]["execution"][field] = value

    with pytest.raises(ValueError, match=field):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata),
            metadata,
            (_pair(metadata),),
        )


def test_append_revalidates_model_copy_and_nested_item_schema(tmp_path):
    metadata = _metadata()
    base = _pair(metadata)
    item = ItemResult(item_id="item", status="completed", score=0.5)
    malformed: dict[str, object] = {
        "item-status": base.model_copy(
            update={"items": (item.model_copy(update={"status": "bogus"}),)}
        ),
        "item-id": base.model_copy(
            update={"items": (item.model_copy(update={"item_id": 1}),)}
        ),
        "item-details": base.model_copy(
            update={"items": (item.model_copy(update={"details": []}),)}
        ),
        "methodology": base.model_copy(update={"methodology": []}),
        "annotations": base.model_copy(update={"annotations": "not-a-list"}),
        "started-at": base.model_copy(update={"started_at": 1}),
        "finished-at": base.model_copy(update={"finished_at": []}),
        "comparable": base.model_copy(update={"comparable": 1}),
        "schema-version": base.model_copy(update={"schema_version": 1.0}),
    }
    raw_with_extra_item_field = base.model_dump(mode="json")
    raw_with_extra_item_field["items"] = [
        {
            "item_id": "item",
            "status": "completed",
            "score": 0.5,
            "forged": True,
        }
    ]
    malformed["nested-extra"] = raw_with_extra_item_field

    for name, pair in malformed.items():
        with pytest.raises(ValueError, match="PairResult schema"):
            ResultStore(tmp_path / name, "run-1").append_scoreboard_index(
                _scoreboard(metadata),
                metadata,
                (pair,),
            )


@pytest.mark.parametrize("score", [True, "1.0"], ids=["boolean", "numeric-string"])
def test_append_rejects_coercible_raw_pair_metric_scores(tmp_path, score):
    metadata = _metadata()
    raw_pair = _pair(metadata, score=1.0).model_dump(mode="json")
    raw_pair["metrics"]["score"] = score

    with pytest.raises(ValueError, match="PairResult schema"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata, score=1.0),
            metadata,
            (raw_pair,),
        )


@pytest.mark.parametrize("field", ["details", "methodology"])
def test_append_rejects_nested_nonfinite_pair_evidence(tmp_path, field):
    metadata = _metadata()
    pair = _pair(metadata)
    if field == "details":
        pair = pair.model_copy(
            update={
                "items": (
                    ItemResult(
                        item_id="bad",
                        status="completed",
                        score=0.5,
                        details={"nested": float("nan")},
                    ),
                )
            }
        )
    else:
        pair = pair.model_copy(update={"methodology": {"nested": float("nan")}})

    with pytest.raises(ValueError, match="invalid evidence: .* must be finite"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata),
            metadata,
            (pair,),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        pytest.param(
            lambda metadata: metadata["identity"]["adapters"][0].pop(
                "evaluation_protocol"
            ),
            "evaluation_protocol",
            id="missing-protocol",
        ),
        pytest.param(
            lambda metadata: metadata["identity"]["adapters"][0][
                "evaluation_protocol"
            ].__setitem__("resources", []),
            "evaluator resources",
            id="missing-resources",
        ),
        pytest.param(
            lambda metadata: metadata["identity"]["adapters"][0][
                "evaluation_protocol"
            ]["resources"][0].__setitem__("sha256", "not-a-digest"),
            "SHA-256",
            id="bad-resource-digest",
        ),
        pytest.param(
            lambda metadata: metadata["identity"]["adapters"][0][
                "evaluation_protocol"
            ].__setitem__(
                "distributions",
                [
                    {
                        "name": "external-harness",
                        "installed": True,
                        "version": "1.0",
                        "content_verified": False,
                        "detail": "unreadable",
                    }
                ],
            ),
            "not content verified",
            id="unverified-distribution",
        ),
    ],
)
def test_history_revalidates_complete_evaluator_methodology(tmp_path, mutation, match):
    metadata = _metadata()
    mutation(metadata)
    _refresh_metadata_identity(metadata)

    with pytest.raises(ValueError, match=match):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


@pytest.mark.parametrize("status", ["completed", "partial"])
@pytest.mark.parametrize(
    ("name", "complete"),
    [("gsm8k", True)],
    ids=["source-complete"],
)
def test_scored_dataset_rows_require_content_bound_cache_identity(
    tmp_path, status, name, complete
):
    metadata = _metadata()
    identity = metadata["identity"]["adapters"][0]
    identity["name"] = name
    identity["history_provenance"]["complete"] = complete
    identity["history_provenance"]["reason"] = (
        None if complete else "external harness runtime remains unresolved"
    )
    identity.pop("sha256")
    identity.pop("assets")
    identity["unavailable"] = True
    metadata["config"]["only"] = [name]
    _refresh_metadata_identity(metadata)
    board = _scoreboard(metadata, status=status)
    board["benchmarks"] = [name]
    board["display_names"] = {name: name}
    board["cells"] = {name: board["cells"].pop("gsm8k")}
    pair = _pair(metadata, status=status).model_copy(update={"benchmark": name})

    with pytest.raises(ValueError, match="content-bound dataset"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(board, metadata, (pair,))


def test_skipped_dataset_row_may_record_an_unavailable_cache(tmp_path):
    metadata = _metadata()
    identity = metadata["identity"]["adapters"][0]
    identity.pop("sha256")
    identity.pop("assets")
    identity["unavailable"] = True
    _refresh_metadata_identity(metadata)
    board = _scoreboard(metadata, status="skipped")
    cell = board["cells"]["gsm8k"]["target-a"]
    cell.update({"score": None, "n": 0, "n_scored": None, "reason": "cache unavailable"})
    pair = _pair(metadata, status="skipped", reason="cache unavailable").model_copy(
        update={"metrics": {"score": None, "n_total": 0}}
    )

    assert ResultStore(tmp_path, "run-1").append_scoreboard_index(
        board, metadata, (pair,)
    ).is_file()


@pytest.mark.parametrize(
    "assets",
    [
        [{"path": "../escape", "sha256": "a" * 64}],
        [{"path": "assets/chart.png", "sha256": "bad"}],
        [
            {"path": "assets/chart.png", "sha256": "a" * 64},
            {"path": "assets/chart.png", "sha256": "a" * 64},
        ],
        [{"path": "assets//chart.png", "sha256": "a" * 64}],
    ],
    ids=["unsafe-path", "bad-digest", "duplicate", "non-canonical-path"],
)
def test_history_rejects_malformed_asset_identity_lists(tmp_path, assets):
    metadata = _metadata()
    metadata["identity"]["adapters"][0]["assets"] = assets
    _refresh_metadata_identity(metadata)

    with pytest.raises(ValueError, match="asset"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda identity: identity.update({"dataset": None, "revision": None}),
        lambda identity: identity.update({"unavailable": False}),
        lambda identity: identity.update({"unavailable": True}),
        lambda identity: (identity.pop("sha256"), identity.pop("assets")),
    ],
    ids=["dataset-spoof", "false-unavailable", "conflicting-unavailable", "undisclosed"],
)
def test_history_rejects_inconsistent_dataset_availability_identity(
    tmp_path, mutation
):
    metadata = _metadata()
    mutation(metadata["identity"]["adapters"][0])
    _refresh_metadata_identity(metadata)

    with pytest.raises(ValueError, match="dataset identity|unavailable|availability"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


def test_history_rejects_path_shadowed_evaluator_executable(tmp_path):
    metadata = _metadata()
    protocol = metadata["identity"]["adapters"][0]["evaluation_protocol"]
    protocol["distributions"] = [
        {
            "name": "external-harness",
            "installed": True,
            "version": "1.0",
            "content_verified": True,
            "file_count": 1,
            "content_sha256": "a" * 64,
        }
    ]
    protocol["executables"] = [
        {
            "name": "harness",
            "found": True,
            "content_verified": True,
            "content_sha256": "b" * 64,
            "owned_by_distribution": [],
        }
    ]
    _refresh_metadata_identity(metadata)

    with pytest.raises(ValueError, match="PATH-shadowed"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


@pytest.mark.parametrize("offline_value", [True, None], ids=["enabled", "omitted"])
def test_history_requires_explicit_real_run_config(tmp_path, offline_value):
    metadata = _metadata()
    if offline_value is None:
        metadata["config"].pop("offline_fixtures")
    else:
        metadata["config"]["offline_fixtures"] = offline_value
    _refresh_metadata_identity(metadata)

    with pytest.raises(ValueError, match="offline_fixtures.*explicitly false"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


def test_history_rejects_failed_run_evidence(tmp_path):
    metadata = _metadata()

    with pytest.raises(ValueError, match="completed-run evidence"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata, status="failed"),
            metadata,
            (_pair(metadata, status="failed"),),
        )


@pytest.mark.parametrize(
    ("name", "requires_execution", "policy", "reason"),
    [
        pytest.param(
            "custom-remote-eval",
            False,
            "withheld_unresolved_runtime",
            "runtime dataset, image, or executable content is unresolved",
            id="unresolved-runtime",
        ),
        pytest.param(
            "custom-code-eval",
            True,
            "withheld_unpinned_execution",
            "code execution lacks a resolved immutable image and platform identity",
            id="unbound-code-execution",
        ),
    ],
)
def test_history_withholds_only_unresolved_runtime_cells(
    tmp_path, name, requires_execution, policy, reason
):
    metadata = _metadata()
    identity = metadata["identity"]["adapters"][0]
    identity["name"] = name
    if name == "custom-remote-eval":
        identity["history_provenance"].update(
            {
                "complete": False,
                "reason": "runtime remains unresolved",
            }
        )
    identity["history_provenance"]["requires_content_addressed_execution"] = (
        requires_execution
    )
    metadata["config"]["only"] = [name]
    _refresh_metadata_identity(metadata)
    board = _scoreboard(metadata)
    board["benchmarks"] = [name]
    board["display_names"] = {name: name}
    board["cells"] = {name: board["cells"].pop("gsm8k")}
    pair = _pair(metadata).model_copy(update={"benchmark": name})

    with pytest.raises(ValueError, match="cross-run policy"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            board, metadata, (pair,)
        )

    cell = board["cells"][name]["target-a"]
    cell["cross_run_policy"] = policy
    cell["cross_run_reason"] = reason
    pair = pair.model_copy(
        update={"cross_run_policy": policy, "cross_run_reason": reason}
    )
    path = ResultStore(tmp_path, "run-1").append_scoreboard_index(
        board, metadata, (pair,)
    )

    assert path.is_file()


def test_content_addressed_execution_policy_requires_config_environment_binding(tmp_path):
    metadata = _metadata()
    identity = metadata["identity"]["adapters"][0]
    identity["name"] = "custom-code-eval"
    identity["history_provenance"]["requires_content_addressed_execution"] = True
    metadata["config"]["only"] = ["custom-code-eval"]
    metadata["environment"]["execution"] = {
        "runner": "docker",
        "image": "registry.example/bench@sha256:" + "1" * 64,
        "resolved_image_id": "sha256:" + "2" * 64,
        "image_os": "linux",
        "image_architecture": "amd64",
        "image_variant": None,
        "cpus": 1.0,
        "pids_limit": 64,
        "disk_mb": 256,
        "pull_policy": "never",
        "available": True,
        "availability_detail": "available",
    }
    _refresh_metadata_identity(metadata)
    board = _scoreboard(metadata)
    board["benchmarks"] = ["custom-code-eval"]
    board["display_names"] = {"custom-code-eval": "Custom Code Eval"}
    board["cells"] = {"custom-code-eval": board["cells"].pop("gsm8k")}
    pair = _pair(metadata).model_copy(update={"benchmark": "custom-code-eval"})

    with pytest.raises(ValueError, match="runner disagrees"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(board, metadata, (pair,))

    metadata["config"]["execution"] = {
        "runner": "docker",
        "image": "registry.example/bench@sha256:" + "1" * 64,
        "cpus": 1.0,
        "pids_limit": 64,
        "disk_mb": 256,
        "pull_policy": "never",
    }
    _refresh_metadata_identity(metadata)
    board = _scoreboard(metadata)
    board["benchmarks"] = ["custom-code-eval"]
    board["display_names"] = {"custom-code-eval": "Custom Code Eval"}
    board["cells"] = {"custom-code-eval": board["cells"].pop("gsm8k")}
    pair = _pair(metadata).model_copy(update={"benchmark": "custom-code-eval"})

    assert ResultStore(tmp_path, "run-1").append_scoreboard_index(
        board, metadata, (pair,)
    ).is_file()


@pytest.mark.parametrize("field", ["score", "confidence"], ids=["score", "wilson"])
def test_rehashed_huge_numbers_fail_as_validation_errors(tmp_path, field):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata), metadata, pairs=(_pair(metadata),), previous_record_sha256=None
    )
    cell = entry["scoreboard"]["cells"]["gsm8k"]["target-a"]
    binding = entry["pairs"][0]
    huge = 10**4000
    if field == "score":
        cell["score"] = huge
        binding["score"] = huge
    else:
        interval = {
            "method": "wilson",
            "confidence": huge,
            "successes": 1,
            "trials": 2,
            "lower": 0.1,
            "upper": 0.9,
        }
        cell["confidence_interval"] = interval
        binding["confidence_interval"] = interval
    _rehash(entry)
    _rewrite_entries(tmp_path / history.INDEX_NAME, [entry])

    with pytest.raises(ValueError, match="scoreboard history"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_identity_hash_and_disclosed_config_are_recomputed(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")
    metadata["identity"]["adapters"][0]["revision"] = "mutated"
    with pytest.raises(ValueError, match="fingerprint"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))

    metadata = _metadata()
    metadata["config"]["only"] = ["other"]
    with pytest.raises(ValueError, match="identity config"):
        store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))


def test_tracked_config_requires_opaque_extra_body_and_rejects_credential_fields(tmp_path):
    metadata = _metadata()
    target = metadata["config"]["targets"][0]
    target["extra_body_json"] = '{"vendor_access_token":"dummy-body-token"}'
    _refresh_metadata_identity(metadata)
    with pytest.raises(ValueError, match="opaque SHA-256"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )

    metadata = _metadata()
    metadata["config"]["targets"][0]["vendor_access_token"] = "dummy-token"
    _refresh_metadata_identity(metadata)
    with pytest.raises(ValueError, match="credential-capable"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )

    metadata = _metadata()
    metadata["config"]["targets"][0]["api_key_env"] = "sk-live-secret-value"
    _refresh_metadata_identity(metadata)
    with pytest.raises(ValueError, match="environment-variable name"):
        ResultStore(tmp_path, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )


def test_normal_token_count_fields_and_redacted_extra_body_are_trackable(tmp_path):
    metadata = _metadata()
    target = metadata["config"]["targets"][0]
    target["max_output_tokens"] = 8192
    target["extra_body_json"] = "sha256:" + "a" * 64
    _refresh_metadata_identity(metadata)

    path = ResultStore(tmp_path, "run-1").append_scoreboard_index(
        _scoreboard(metadata), metadata, (_pair(metadata),)
    )

    assert path.is_file()


def test_key_preflight_handles_resume_rerun_and_collisions(tmp_path):
    metadata = _metadata()
    store = ResultStore(tmp_path, "run-1")
    store.ensure_scoreboard_key_available(metadata)
    store.append_scoreboard_index(_scoreboard(metadata), metadata, (_pair(metadata),))

    store.ensure_scoreboard_key_available(metadata)
    with pytest.raises(ValueError, match="already indexed"):
        store.ensure_scoreboard_key_available(metadata, rerun=True)

    same_key_other_run = json.loads(json.dumps(metadata))
    same_key_other_run["run_id"] = "run-2"
    same_key_other_run["config"]["run_id"] = "run-2"
    with pytest.raises(ValueError, match="already indexed"):
        ResultStore(tmp_path, "run-2").ensure_scoreboard_key_available(same_key_other_run)

    same_run_other_key = _metadata("run-1", seed="other")
    with pytest.raises(ValueError, match="run_id"):
        store.ensure_scoreboard_key_available(same_run_other_key)


def test_initialize_returns_first_metadata_and_ignores_only_volatile_environment(
    tmp_path,
):
    store = ResultStore(tmp_path, "run-1")
    first = _metadata(created_at="2026-08-05T00:00:00+00:00", available=False)
    resumed = _metadata(created_at="2026-08-06T00:00:00+00:00", available=True)

    assert store.initialize_run(first) == first
    assert store.initialize_run(resumed) == first

    changed = _metadata(created_at="2026-08-07T00:00:00+00:00")
    changed["environment"]["source_tree_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stable environment"):
        store.initialize_run(changed)


def test_resume_treats_resolved_docker_identity_as_unknown_only_while_unavailable(tmp_path):
    metadata = _metadata()
    image = "registry.example/bench@sha256:" + "1" * 64
    metadata["config"]["execution"] = {
        "runner": "docker",
        "image": image,
        "cpus": 1.0,
        "pids_limit": 64,
        "disk_mb": 256,
        "pull_policy": "never",
    }
    metadata["environment"]["execution"] = {
        "runner": "docker",
        "image": image,
        "resolved_image_id": "sha256:" + "2" * 64,
        "image_os": "linux",
        "image_architecture": "amd64",
        "image_variant": None,
        "cpus": 1.0,
        "pids_limit": 64,
        "disk_mb": 256,
        "pull_policy": "never",
        "available": True,
        "availability_detail": "available",
    }
    _refresh_metadata_identity(metadata)
    store = ResultStore(tmp_path, "run-1")
    store.initialize_run(metadata)

    unavailable = json.loads(json.dumps(metadata))
    unavailable_execution = unavailable["environment"]["execution"]
    for field in (
        "resolved_image_id",
        "image_os",
        "image_architecture",
        "image_variant",
    ):
        unavailable_execution.pop(field)
    unavailable_execution["available"] = False
    unavailable_execution["availability_detail"] = "daemon temporarily unavailable"
    store.require_resumable_environment(unavailable["environment"])
    assert store.initialize_run(unavailable) == metadata

    changed = json.loads(json.dumps(metadata))
    changed["environment"]["execution"]["resolved_image_id"] = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="stable environment"):
        store.require_resumable_environment(changed["environment"])


def test_initialize_rejects_metadata_for_another_run_id(tmp_path):
    store = ResultStore(tmp_path, "run-1")

    with pytest.raises(ValueError, match="persisted run metadata"):
        store.initialize_run(_metadata("run-2"))

    assert not store.run_dir.exists()


def test_early_environment_preflight_is_noop_for_new_run_and_rejects_before_io(
    tmp_path,
):
    store = ResultStore(tmp_path / "results", "run-1")
    metadata = _metadata()
    store.require_resumable_environment(metadata["environment"])
    assert not store.run_dir.exists()

    store.initialize_run(metadata)
    resumed = _metadata(created_at="later", available=False)
    store.require_resumable_environment(resumed["environment"])

    changed = json.loads(json.dumps(resumed["environment"]))
    changed["python"] = "3.11.9"
    with pytest.raises(ValueError, match="stable environment"):
        store.require_resumable_environment(changed)


def test_index_and_lock_symlinks_never_read_or_write_external_files(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    external_index = tmp_path / "external-index"
    external_index.write_bytes(b"external-index-bytes")
    (root / history.INDEX_NAME).symlink_to(external_index)

    with pytest.raises(ValueError, match="unsafe artifact"):
        ResultStore(root, "reader").load_scoreboard_index()
    assert external_index.read_bytes() == b"external-index-bytes"

    (root / history.INDEX_NAME).unlink()
    (root / history.LOCK_NAME).unlink(missing_ok=True)
    external_lock = tmp_path / "external-lock"
    external_lock.write_bytes(b"external-lock-bytes")
    (root / history.LOCK_NAME).symlink_to(external_lock)
    with pytest.raises(ValueError, match="unsafe lock"):
        ResultStore(root, "reader").load_scoreboard_index()
    metadata = _metadata()
    with pytest.raises(ValueError, match="unsafe artifact|lock"):
        ResultStore(root, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )
    assert external_lock.read_bytes() == b"external-lock-bytes"


def test_read_only_load_does_not_create_a_missing_lock(tmp_path):
    metadata = _metadata()
    ResultStore(tmp_path, "run-1").append_scoreboard_index(
        _scoreboard(metadata), metadata, (_pair(metadata),)
    )
    (tmp_path / history.LOCK_NAME).unlink()
    before = {path.name for path in tmp_path.iterdir()}

    assert len(ResultStore(tmp_path, "reader").load_scoreboard_index()) == 1

    assert {path.name for path in tmp_path.iterdir()} == before
    assert not (tmp_path / history.LOCK_NAME).exists()


def test_reader_can_share_lock_that_is_not_openable_for_write(tmp_path):
    metadata = _metadata()
    ResultStore(tmp_path, "run-1").append_scoreboard_index(
        _scoreboard(metadata), metadata, (_pair(metadata),)
    )
    lock = tmp_path / history.LOCK_NAME
    lock.chmod(0o400)

    assert ResultStore(tmp_path, "reader").load_scoreboard_index()[0]["run_id"] == "run-1"


def test_fifo_index_is_rejected_without_blocking(tmp_path):
    os.mkfifo(tmp_path / history.INDEX_NAME)

    with pytest.raises(ValueError, match="unsafe artifact"):
        ResultStore(tmp_path, "reader").load_scoreboard_index()


def test_pinned_root_dirfd_never_creates_lock_through_replacement_symlink(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    moved = tmp_path / "moved-results"
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="results root changed"):
        with history._opened_results_root(root, create=False) as root_fd:
            root.rename(moved)
            root.symlink_to(outside, target_is_directory=True)
            with history._root_lock(root_fd, exclusive=True):
                pass

    assert (moved / history.LOCK_NAME).is_file()
    assert not (outside / history.LOCK_NAME).exists()


def test_symlinked_results_root_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "results"
    root.symlink_to(outside, target_is_directory=True)
    metadata = _metadata()

    with pytest.raises(ValueError, match="symlinked results root"):
        ResultStore(root, "reader").load_scoreboard_index()
    with pytest.raises(ValueError, match="symlinked or unsafe results root"):
        ResultStore(root, "run-1").append_scoreboard_index(
            _scoreboard(metadata), metadata, (_pair(metadata),)
        )
    assert not (outside / history.INDEX_NAME).exists()


def test_atomic_failure_preserves_existing_prefix(tmp_path, monkeypatch):
    first = _metadata("run-1", seed="first")
    store = ResultStore(tmp_path, "run-1")
    path = store.append_scoreboard_index(_scoreboard(first), first, (_pair(first),))
    before = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("simulated atomic failure")

    monkeypatch.setattr(history, "_atomic_replace_index", fail_write)
    second = _metadata("run-2", seed="second")
    with pytest.raises(OSError, match="simulated atomic failure"):
        ResultStore(tmp_path, "run-2").append_scoreboard_index(
            _scoreboard(second), second, (_pair(second),)
        )
    assert path.read_bytes() == before


def test_append_fsyncs_the_results_directory(tmp_path, monkeypatch):
    calls: list[bool] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    metadata = _metadata()
    ResultStore(tmp_path, "run-1").append_scoreboard_index(
        _scoreboard(metadata), metadata, (_pair(metadata),)
    )

    assert True in calls


def test_concurrent_processes_append_complete_unique_records(tmp_path):
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_append_worker, args=(str(tmp_path), index)) for index in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    entries = ResultStore(tmp_path, "reader").load_scoreboard_index()
    assert len(entries) == 8
    assert {entry["run_id"] for entry in entries} == {f"run-{index}" for index in range(8)}
    assert len({entry["record_sha256"] for entry in entries}) == 8


def test_reader_waits_on_existing_lock_before_concluding_index_is_empty(tmp_path):
    metadata = _metadata()
    entry = history.build_entry(
        _scoreboard(metadata), metadata, pairs=(_pair(metadata),), previous_record_sha256=None
    )
    raw = _canonical(entry) + b"\n"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_publish_under_fixed_lock,
        args=(str(tmp_path), raw, ready, release),
    )
    process.start()
    assert ready.wait(timeout=5)

    loaded: list[tuple[dict, ...]] = []
    done = threading.Event()

    def load() -> None:
        loaded.append(ResultStore(tmp_path, "reader").load_scoreboard_index())
        done.set()

    thread = threading.Thread(target=load)
    thread.start()
    assert not done.wait(timeout=0.1)
    release.set()
    thread.join(timeout=5)
    process.join(timeout=5)

    assert process.exitcode == 0
    assert not thread.is_alive()
    assert loaded == [(entry,)]
