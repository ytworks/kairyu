#!/usr/bin/env python3
"""Reproducible CPU benchmark for bursty EngineLoop producer operations.

Run:
    uv run python verification/l1/performance/op_queue_bench.py --operations 100000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from kairyu import SamplingParams
from kairyu.engine.core.scheduler import EngineRequest
from kairyu.engine.engine_loop import (
    EngineLoop,
    _AbortBatch,
    _AddBatch,
    _RequestTrack,
    engine_sampling_from,
)
from kairyu.engine.tokenizer import IncrementalDetokenizer


class _Tokenizer:
    eos_token_id = None

    def encode(self, _text: str) -> tuple[int, ...]:
        return (1,)

    def decode(self, token_ids) -> str:
        return "x" * len(token_ids)

    def vocab(self) -> list[str]:
        return ["x"]


class _Scheduler:
    states: dict[str, object] = {}

    def abort(self, _request_id: str) -> None:
        pass

    def forget(self, _request_id: str) -> None:
        pass


class _Runner:
    def release(self, _request_id: str) -> None:
        pass


class _LegacyEngineLoop(EngineLoop):
    """Exact pre-#218 producer queue behavior for the benchmark baseline."""

    def submit(self, request_id: str, prompt: str, params: SamplingParams) -> None:
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=self.tokenize_prompt(prompt),
            max_new_tokens=params.max_tokens or 16,
            eos_token_id=self._default_eos,
            stop_token_ids=tuple(params.stop_token_ids or ())
            + self._default_stop_ids,
            min_tokens=params.min_tokens,
            ignore_eos=params.ignore_eos,
            sampling=engine_sampling_from(params),
        )
        track = _RequestTrack(
            detok=IncrementalDetokenizer(self._tokenizer),
            stops=tuple(params.stop or ()),
            num_prompt_tokens=len(engine_request.prompt_token_ids),
        )
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        self._active_request_ids.add(request_id)
        self._ops.append(("add", (engine_request, track)))

    def abort(self, request_id: str) -> None:
        if request_id in self._active_request_ids:
            self._ops.append(("abort", request_id))


def _new_loop(loop_type=EngineLoop) -> EngineLoop:
    # The legacy subclass intentionally installs its old tuple queue at runtime.
    loop = loop_type(_Tokenizer(), _Scheduler(), _Runner())
    if loop_type is _LegacyEngineLoop:
        loop._ops = deque()
    return loop
    return EngineLoop(_Tokenizer(), _Scheduler(), _Runner())


def _run_partitioned(operations: int, producers: int, fn) -> None:
    if producers == 1:
        for index in range(operations):
            fn(index)
        return

    def run_partition(producer: int) -> None:
        for index in range(producer, operations, producers):
            fn(index)

    with ThreadPoolExecutor(max_workers=producers) as pool:
        list(pool.map(run_partition, range(producers)))


def _measure_adds(
    operations: int, producers: int, loop_type=EngineLoop
) -> dict[str, float | int]:
    loop = _new_loop(loop_type)
    params = SamplingParams(max_tokens=1)
    start = time.perf_counter()
    _run_partitioned(
        operations,
        producers,
        lambda index: loop.submit(f"request-{index}", "p", params),
    )
    elapsed = time.perf_counter() - start
    with loop._ops_lock:
        queued_containers = len(loop._ops)
        queued_ids = sum(
            len(batch.requests) if isinstance(batch, _AddBatch) else 1
            for batch in loop._ops
        )
    loop.close()
    return {
        "elapsed_seconds": elapsed,
        "operations_per_second": operations / elapsed,
        "queued_operation_containers": queued_containers,
        "queued_ids": queued_ids,
    }


def _measure_duplicate_aborts(
    operations: int, producers: int, loop_type=EngineLoop
) -> dict[str, float | int]:
    loop = _new_loop(loop_type)
    loop.submit("target", "p", SamplingParams(max_tokens=1))
    start = time.perf_counter()
    _run_partitioned(
        operations,
        producers,
        lambda _index: loop.abort("target"),
    )
    elapsed = time.perf_counter() - start
    with loop._ops_lock:
        queued_containers = len(loop._ops)
        queued_abort_ids = sum(
            (
                len(batch.request_ids)
                if isinstance(batch, _AbortBatch)
                else int(isinstance(batch, tuple) and batch[0] == "abort")
            )
            for batch in loop._ops
        )
    loop.close()
    return {
        "elapsed_seconds": elapsed,
        "operations_per_second": operations / elapsed,
        "queued_operation_containers_including_add": queued_containers,
        "queued_abort_ids": queued_abort_ids,
    }


def _summarize(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    result = dict(samples[0])
    for key in ("elapsed_seconds", "operations_per_second"):
        result[key] = statistics.median(float(sample[key]) for sample in samples)
    result["repeats"] = len(samples)
    return result


def _comparison(
    measure,
    operations: int,
    producers: int,
    repeats: int,
) -> dict[str, object]:
    legacy = _summarize(
        [
            measure(operations, producers, _LegacyEngineLoop)
            for _ in range(repeats)
        ]
    )
    coalesced = _summarize(
        [measure(operations, producers, EngineLoop) for _ in range(repeats)]
    )
    return {
        "legacy": legacy,
        "coalesced": coalesced,
        "elapsed_speedup": (
            float(legacy["elapsed_seconds"])
            / float(coalesced["elapsed_seconds"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=int, default=100_000)
    parser.add_argument("--producers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.operations < 1 or args.producers < 1 or args.repeats < 1:
        parser.error("operations, producers, and repeats must be positive")

    result = {
        "operations": args.operations,
        "producers": args.producers,
        "add_burst": _comparison(
            _measure_adds,
            args.operations,
            args.producers,
            args.repeats,
        ),
        "duplicate_abort_burst": _comparison(
            _measure_duplicate_aborts,
            args.operations,
            args.producers,
            args.repeats,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
