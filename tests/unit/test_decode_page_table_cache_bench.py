from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "bench" / "decode_page_table_cache_qwen.py"
_SPEC = importlib.util.spec_from_file_location(
    "decode_page_table_cache_qwen",
    _PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BENCH)

_SOURCE = {"bound": "source"}
_CHECKPOINT = {"bound": "checkpoint"}
_HARDWARE = {"bound": "hardware"}
_TOPOLOGY = {"bound": "topology"}


def _stats(mode: str, *, released: bool) -> dict[str, object]:
    cache_enabled = mode == "cache"
    return {
        "enabled": cache_enabled,
        "builds": BENCH.EXPECTED_BUILDS,
        "rows": BENCH.EXPECTED_ROWS,
        "host_page_ids_visited": BENCH.EXPECTED_OWNED_ELEMENTS,
        "legacy_outer_allocations": (0 if cache_enabled else BENCH.EXPECTED_BUILDS),
        "legacy_row_allocations": (0 if cache_enabled else BENCH.EXPECTED_ROWS),
        "legacy_elements_written": (0 if cache_enabled else BENCH.EXPECTED_LEGACY_ELEMENTS_WRITTEN),
        "legacy_owned_elements_uploaded": (0 if cache_enabled else BENCH.EXPECTED_OWNED_ELEMENTS),
        "cache": {
            "storage_allocations": 0,
            "storage_elements_copied": 0,
            "rows_uploaded": (BENCH.EXPECTED_CACHE_ROWS_UPLOADED if cache_enabled else 0),
            "elements_uploaded": (BENCH.EXPECTED_CACHE_ELEMENTS_UPLOADED if cache_enabled else 0),
            "row_hits": (BENCH.EXPECTED_CACHE_ROW_HITS if cache_enabled else 0),
            "released_rows": (BENCH.REQUEST_COUNT if cache_enabled and released else 0),
            "removed_rows": 0,
            "full_row_uploads": (BENCH.REQUEST_COUNT if cache_enabled else 0),
            "partial_row_uploads": (
                BENCH.EXPECTED_CACHE_ROWS_UPLOADED - BENCH.REQUEST_COUNT if cache_enabled else 0
            ),
            "live_rows": (0 if released or not cache_enabled else BENCH.REQUEST_COUNT),
            "row_capacity": BENCH.REQUEST_COUNT,
            "page_capacity": 32,
        },
        "graph": {
            "rows_copied": (
                BENCH.EXPECTED_GRAPH_ROWS_COPIED if cache_enabled else BENCH.EXPECTED_ROWS
            ),
            "elements_copied": (
                BENCH.EXPECTED_GRAPH_ELEMENTS_COPIED
                if cache_enabled
                else BENCH.EXPECTED_GRAPH_FULL_ELEMENTS
            ),
            "row_hits": (BENCH.EXPECTED_GRAPH_ROW_HITS if cache_enabled else 0),
            "released_rows": (BENCH.REQUEST_COUNT if cache_enabled and released else 0),
            "live_rows": (0 if released or not cache_enabled else BENCH.REQUEST_COUNT),
        },
        "graph_dispatch": {
            "graph_executions": BENCH.DECODE_STEPS,
            "eager_fallbacks": 0,
            "captured_buckets": [BENCH.REQUEST_COUNT],
        },
    }


def _rank_rows(mode: str, *, released: bool) -> list[dict[str, object]]:
    stats = _stats(mode, released=released)
    return [
        {
            "rank": rank,
            "world_size": BENCH.TP,
            "device": f"cuda:{rank}",
            "stats": copy.deepcopy(stats),
        }
        for rank in range(BENCH.TP)
    ]


def _outputs() -> list[list[int]]:
    return [
        [
            16 + ((row * 997 + position * 131) % (BENCH.VOCAB_SIZE - 16))
            for position in range(BENCH.MAX_NEW_TOKENS)
        ]
        for row in range(BENCH.REQUEST_COUNT)
    ]


def _measurements() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    orders: list[list[str]] = []
    for round_index in range(BENCH.FORMAL_ROUNDS):
        order = BENCH._order_for_round(round_index)
        orders.append(list(order))
        for slot, mode in enumerate(order):
            wall_s = 1.0 + 0.1 * len(samples)
            cuda_ms = 900.0 + 10.0 * len(samples)
            samples.append(
                {
                    "round": round_index,
                    "slot": slot,
                    "mode": mode,
                    "outputs": _outputs(),
                    "pre_release_rank_stats": _rank_rows(
                        mode,
                        released=False,
                    ),
                    "post_release_rank_stats": _rank_rows(
                        mode,
                        released=True,
                    ),
                    "timing": {
                        "binding": False,
                        "wall_s": wall_s,
                        "cuda_ms": cuda_ms,
                        "tpot_s": wall_s / BENCH.DECODE_TOKENS,
                        "tokens_per_s": BENCH.DECODE_TOKENS / wall_s,
                    },
                }
            )
    return {
        "orders": orders,
        "samples": samples,
        "diagnostic": {
            "binding": False,
            "summary": BENCH._diagnostic_summary(samples),
        },
    }


@pytest.fixture
def artifact(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(BENCH, "_source_gate", lambda value: value == _SOURCE)
    monkeypatch.setattr(
        BENCH,
        "_checkpoint_gate",
        lambda value: value == _CHECKPOINT,
    )
    monkeypatch.setattr(
        BENCH,
        "_hardware_gate",
        lambda value: value == _HARDWARE,
    )
    monkeypatch.setattr(
        BENCH,
        "_topology_gate",
        lambda value: value == _TOPOLOGY,
    )
    payload: dict[str, object] = {
        "schema_version": BENCH.SCHEMA_VERSION,
        "measurement_kind": BENCH.MEASUREMENT_KIND,
        "created_at": "2026-07-30T00:00:00+00:00",
        "config": BENCH._formal_config(),
        "gate_policy": BENCH._gate_policy(),
        "source": copy.deepcopy(_SOURCE),
        "checkpoint": copy.deepcopy(_CHECKPOINT),
        "hardware": copy.deepcopy(_HARDWARE),
        "topology": copy.deepcopy(_TOPOLOGY),
        "prompts": BENCH._prompt_rows(),
        "measurements": _measurements(),
    }
    payload["checks"] = BENCH._evaluate(payload)
    payload["passed"] = all(payload["checks"].values())
    return payload


def test_synthetic_formal_artifact_passes_all_binding_checks(
    artifact: dict[str, object],
) -> None:
    checks = BENCH._evaluate(artifact)

    assert checks
    assert all(checks.values())
    assert artifact["passed"] is True


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (
            lambda payload: payload["measurements"]["samples"][0]["outputs"][0].__setitem__(
                0, 151935
            ),
            "output_parity_release_and_timing_shape",
        ),
        (
            lambda payload: payload["measurements"]["samples"][0]["pre_release_rank_stats"][
                1
            ].__setitem__("rank", 0),
            "legacy_path_exact",
        ),
        (
            lambda payload: payload["measurements"]["samples"][1]["pre_release_rank_stats"][0][
                "stats"
            ]["cache"].__setitem__(
                "elements_uploaded",
                BENCH.EXPECTED_OWNED_ELEMENTS,
            ),
            "cache_path_reduced_and_bounded",
        ),
        (
            lambda payload: payload["measurements"]["samples"][1]["post_release_rank_stats"][0][
                "stats"
            ]["graph"].__setitem__("live_rows", 1),
            "cache_path_reduced_and_bounded",
        ),
        (
            lambda payload: payload.__setitem__("source", {"bound": "wrong"}),
            "clean_committed_source_stable",
        ),
    ],
    ids=("output", "rank", "counter", "release", "source"),
)
def test_binding_tampering_is_rejected(
    artifact: dict[str, object],
    mutation,
    failed_check: str,
) -> None:
    tampered = copy.deepcopy(artifact)
    mutation(tampered)

    checks = BENCH._evaluate(tampered)

    assert checks[failed_check] is False
    assert not all(checks.values())


def test_extreme_timing_is_valid_but_never_binding(
    artifact: dict[str, object],
) -> None:
    measurements = artifact["measurements"]
    samples = measurements["samples"]
    for index, sample in enumerate(samples, start=1):
        wall_s = float(index) * 1.0e12
        sample["timing"] = {
            "binding": False,
            "wall_s": wall_s,
            "cuda_ms": wall_s * 1.0e3,
            "tpot_s": wall_s / BENCH.DECODE_TOKENS,
            "tokens_per_s": BENCH.DECODE_TOKENS / wall_s,
        }
    measurements["diagnostic"]["summary"] = BENCH._diagnostic_summary(samples)

    assert all(BENCH._evaluate(artifact).values())


def test_verify_recomputes_checks_instead_of_trusting_stored_verdict(
    artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact["checks"] = {"forged": True}
    artifact["passed"] = True
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(BENCH, "_hardware", lambda: _HARDWARE)
    monkeypatch.setattr(
        BENCH,
        "_checkpoint_manifest",
        lambda _path: _CHECKPOINT,
    )
    monkeypatch.setattr(
        BENCH,
        "_checkpoint_snapshot_gate",
        lambda value: value == _CHECKPOINT,
    )

    replay = BENCH._verify(path, tmp_path)

    assert replay["checks"]["stored_replay_matches"] is False
    assert replay["passed"] is False


def test_geometry_constants_match_scheduler_reservation_formula() -> None:
    owned = 0
    rectangular = 0
    for position in range(1, BENCH.DECODE_STEPS + 1):
        page_counts = [
            -(-(length + position + 1) // BENCH.PAGE_SIZE) for length in BENCH.PROMPT_LENGTHS
        ]
        owned += sum(page_counts)
        rectangular += BENCH.REQUEST_COUNT * max(page_counts)

    assert owned == BENCH.EXPECTED_OWNED_ELEMENTS
    assert rectangular == BENCH.EXPECTED_LEGACY_RECT_ELEMENTS
    assert rectangular + owned == BENCH.EXPECTED_LEGACY_ELEMENTS_WRITTEN
    assert (
        BENCH.DECODE_STEPS * BENCH.REQUEST_COUNT * BENCH.GRAPH_MAX_PAGES
        == BENCH.EXPECTED_GRAPH_FULL_ELEMENTS
    )


def test_production_cache_and_graph_replay_match_formal_constants() -> None:
    from kairyu.engine.core.step_executor import (
        DecodePageTableCache,
        DecodeRowOwner,
        FakeGraphBackend,
        GraphStepExecutor,
        build_decode_batch,
    )

    owners = [DecodeRowOwner(f"request-{index}") for index in range(BENCH.REQUEST_COUNT)]

    def geometry(position: int):
        counts = [-(-(length + position + 1) // BENCH.PAGE_SIZE) for length in BENCH.PROMPT_LENGTHS]
        pages = [
            tuple(range(1 + row * 100, 1 + row * 100 + count)) for row, count in enumerate(counts)
        ]
        return counts, pages

    cache = DecodePageTableCache("cpu")
    # Prewarm only storage, exactly as the formal warm cache cell does.
    for position in range(1, BENCH.DECODE_STEPS + 1):
        counts, pages = geometry(position)
        cache.materialize(
            pages,
            max_pages=max(counts),
            scratch_page=0,
            row_owners=owners,
        )
    for owner in owners:
        cache.release(owner.request_id)
    cache.stats(reset=True)

    def decode(batch):
        return batch.token_ids

    candidate_graph = GraphStepExecutor(
        decode,
        FakeGraphBackend(),
        BENCH.GRAPH_MAX_BATCH,
        scratch_page=0,
        max_pages=BENCH.GRAPH_MAX_PAGES,
    )
    for position in range(1, BENCH.DECODE_STEPS + 1):
        counts, pages = geometry(position)
        candidate_graph.execute_decode(
            build_decode_batch(
                token_ids=[1] * BENCH.REQUEST_COUNT,
                positions=[length + position - 1 for length in BENCH.PROMPT_LENGTHS],
                page_lists=pages,
                seq_lens=[length + position for length in BENCH.PROMPT_LENGTHS],
                max_pages=max(counts),
                scratch_page=0,
                page_table_cache=cache,
                row_owners=owners,
            )
        )

    cache_stats = cache.stats()
    graph_stats = candidate_graph.page_table_execution_stats()
    dispatch = candidate_graph.page_table_dispatch_stats()
    assert cache_stats == {
        "storage_allocations": 0,
        "storage_elements_copied": 0,
        "rows_uploaded": BENCH.EXPECTED_CACHE_ROWS_UPLOADED,
        "elements_uploaded": BENCH.EXPECTED_CACHE_ELEMENTS_UPLOADED,
        "row_hits": BENCH.EXPECTED_CACHE_ROW_HITS,
        "released_rows": 0,
        "removed_rows": 0,
        "full_row_uploads": BENCH.REQUEST_COUNT,
        "partial_row_uploads": (BENCH.EXPECTED_CACHE_ROWS_UPLOADED - BENCH.REQUEST_COUNT),
        "live_rows": BENCH.REQUEST_COUNT,
        "row_capacity": BENCH.REQUEST_COUNT,
        "page_capacity": 32,
    }
    assert graph_stats == {
        "rows_copied": BENCH.EXPECTED_GRAPH_ROWS_COPIED,
        "elements_copied": BENCH.EXPECTED_GRAPH_ELEMENTS_COPIED,
        "row_hits": BENCH.EXPECTED_GRAPH_ROW_HITS,
        "released_rows": 0,
        "live_rows": BENCH.REQUEST_COUNT,
    }
    assert dispatch == {
        "graph_executions": BENCH.DECODE_STEPS,
        "eager_fallbacks": 0,
        "captured_buckets": (BENCH.REQUEST_COUNT,),
    }
    for owner in owners:
        cache.release(owner.request_id)
        candidate_graph.release(owner.request_id)
    assert cache.stats()["live_rows"] == 0
    assert candidate_graph.page_table_execution_stats()["live_rows"] == 0

    legacy_graph = GraphStepExecutor(
        decode,
        FakeGraphBackend(),
        BENCH.GRAPH_MAX_BATCH,
        scratch_page=0,
        max_pages=BENCH.GRAPH_MAX_PAGES,
    )
    for position in range(1, BENCH.DECODE_STEPS + 1):
        counts, pages = geometry(position)
        legacy_graph.execute_decode(
            build_decode_batch(
                token_ids=[1] * BENCH.REQUEST_COUNT,
                positions=[position] * BENCH.REQUEST_COUNT,
                page_lists=pages,
                seq_lens=[1] * BENCH.REQUEST_COUNT,
                max_pages=max(counts),
                scratch_page=0,
            )
        )
    assert legacy_graph.page_table_execution_stats() == {
        "rows_copied": BENCH.EXPECTED_ROWS,
        "elements_copied": BENCH.EXPECTED_GRAPH_FULL_ELEMENTS,
        "row_hits": 0,
        "released_rows": 0,
        "live_rows": 0,
    }
    assert legacy_graph.page_table_dispatch_stats() == {
        "graph_executions": BENCH.DECODE_STEPS,
        "eager_fallbacks": 0,
        "captured_buckets": (BENCH.REQUEST_COUNT,),
    }
