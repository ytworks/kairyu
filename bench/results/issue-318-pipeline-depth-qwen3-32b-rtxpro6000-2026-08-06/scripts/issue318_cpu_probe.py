from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field

from kairyu import SamplingParams
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import Scheduler
from kairyu.engine.engine_loop import EngineLoop
from kairyu.engine.prompt import TokensPrompt


class _Tokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        raise AssertionError("the probe submits TokensPrompt directly")

    def decode(self, token_ids) -> str:
        return ""

    def vocab(self):
        return range(1_000_000)


@dataclass
class Stats:
    normal_calls: int = 0
    strict_calls: int = 0
    empty_normal: int = 0
    empty_strict: int = 0
    operation_calls: int = 0
    cap_choices: int = 0
    normal_choices: int = 0
    strict_choices: int = 0
    submissions: int = 0
    resolves: int = 0
    rows: int = 0
    device_tokens: int = 0
    prefill_rows: int = 0
    decode_rows: int = 0
    mixed_submissions: int = 0
    prefill_submissions: int = 0
    decode_only_submissions: int = 0
    row_histogram: Counter[int] = field(default_factory=Counter)
    token_histogram: Counter[int] = field(default_factory=Counter)
    pending_start_histogram: Counter[int] = field(default_factory=Counter)
    pending_end_histogram: Counter[int] = field(default_factory=Counter)
    normal_row_histogram: Counter[int] = field(default_factory=Counter)
    strict_row_histogram: Counter[int] = field(default_factory=Counter)
    admission_histogram: Counter[int] = field(default_factory=Counter)
    fragmented_events: list[dict] = field(default_factory=list)
    public_steps: int = 0
    finished_updates: int = 0


class _Handle:
    def __init__(self, runner, scheduled, states):
        self.runner = runner
        self.scheduled = scheduled
        self.states = states

    def result(self):
        return self.runner.resolve(self.scheduled, self.states)


class _Runner:
    def __init__(self, stats: Stats):
        self.stats = stats
        self.plan_source = "unknown"
        self.loop = None
        self.public_step = -1

    def submit(self, _step_index, scheduled, states):
        scheduled = tuple(scheduled)
        rows = len(scheduled)
        tokens = sum(chunk.num_tokens for chunk in scheduled)
        prefill = sum(chunk.is_prefill for chunk in scheduled)
        decode = rows - prefill
        self.stats.submissions += 1
        self.stats.rows += rows
        self.stats.device_tokens += tokens
        self.stats.prefill_rows += prefill
        self.stats.decode_rows += decode
        self.stats.row_histogram[rows] += 1
        self.stats.token_histogram[tokens] += 1
        if self.plan_source == "strict":
            self.stats.strict_row_histogram[rows] += 1
        else:
            self.stats.normal_row_histogram[rows] += 1
        if rows < 16:
            self.stats.fragmented_events.append(
                {
                    "public_step": self.public_step,
                    "source": self.plan_source,
                    "pending_before_submit": (
                        len(self.loop._pending_steps) if self.loop is not None else None
                    ),
                    "rows": rows,
                    "prefill": prefill,
                    "decode": decode,
                    "request_ids": [chunk.request_id for chunk in scheduled],
                }
            )
        if prefill and decode:
            self.stats.mixed_submissions += 1
        elif prefill:
            self.stats.prefill_submissions += 1
        else:
            self.stats.decode_only_submissions += 1
        return _Handle(self, scheduled, states)

    def resolve(self, scheduled, states):
        self.stats.resolves += 1
        sampled = {}
        for chunk in scheduled:
            if chunk.is_prefill and not states[chunk.request_id].prefill_done:
                continue
            count = 1 if chunk.is_prefill else chunk.num_tokens
            sampled[chunk.request_id] = tuple(
                SampledToken(1000 + ((chunk.position + offset) % 1000))
                for offset in range(count)
            )
        return sampled


class LegacyCapLoop(EngineLoop):
    """Current loop shell with HEAD's pre-#318 dynamic total-depth policy."""

    def _schedule_operation(self):
        admission_depth = min(self._pipeline_depth, 2)
        if any(step.contains_prefill for step in self._pending_steps):
            limit = admission_depth
        else:
            has_prefill_work = getattr(self._scheduler, "has_prefill_work", None)
            limit = (
                admission_depth
                if not callable(has_prefill_work) or has_prefill_work()
                else self._pipeline_depth
            )
        if len(self._pending_steps) >= limit:
            return None, "unsupported_scheduler"
        return self._scheduler.schedule, None


def _prompts(shape: str) -> list[tuple[int, ...]]:
    prompts = []
    for index in range(128):
        if shape == "one":
            length = 1
        elif shape == "future":
            length = 32 + index
        elif shape == "varied":
            length = 4 + ((index * 73) % 1021)
        else:
            raise ValueError(shape)
        base = 10_000 + index * 2_000
        prompts.append(tuple(range(base, base + length)))
    return prompts


def run_once(mode: str, shape: str) -> tuple[dict, float]:
    stats = Stats()
    scheduler = Scheduler(
        RadixKVCache(num_pages=8192, page_size=16),
        max_num_batched_tokens=1024,
        max_num_seqs=16,
        page_size=16,
    )
    original_normal = scheduler.schedule
    original_strict = scheduler.schedule_decode_only

    def normal():
        stats.normal_calls += 1
        runner.plan_source = "normal"
        waiting_before = set(scheduler.waiting_ids)
        plan = original_normal()
        admitted = len(waiting_before - set(scheduler.waiting_ids))
        if admitted:
            stats.admission_histogram[admitted] += 1
        if not plan.scheduled:
            stats.empty_normal += 1
        return plan

    def strict():
        stats.strict_calls += 1
        runner.plan_source = "strict"
        plan = original_strict()
        if not plan.scheduled:
            stats.empty_strict += 1
        return plan

    scheduler.schedule = normal
    scheduler.schedule_decode_only = strict
    runner = _Runner(stats)
    loop_type = EngineLoop if mode == "candidate" else LegacyCapLoop
    loop = loop_type(_Tokenizer(), scheduler, runner, pipeline_depth=5)
    runner.loop = loop
    original_operation = loop._schedule_operation

    def operation():
        stats.operation_calls += 1
        selected, reason = original_operation()
        if selected is None:
            stats.cap_choices += 1
        elif selected is scheduler.schedule_decode_only:
            stats.strict_choices += 1
        else:
            stats.normal_choices += 1
        return selected, reason

    loop._schedule_operation = operation
    params = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    for index, prompt in enumerate(_prompts(shape)):
        loop.submit(f"r{index}", TokensPrompt(prompt), params)

    started = time.perf_counter_ns()
    try:
        while loop.has_work():
            runner.public_step = stats.public_steps
            stats.pending_start_histogram[len(loop._pending_steps)] += 1
            updates = loop.step()
            stats.public_steps += 1
            stats.pending_end_histogram[len(loop._pending_steps)] += 1
            stats.finished_updates += sum(update.finished for _, update in updates)
    finally:
        loop.close()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if stats.finished_updates != 128:
        raise AssertionError(stats.finished_updates)
    result = {
        "mode": mode,
        "shape": shape,
        "elapsed_ms": elapsed_ms,
        "public_steps": stats.public_steps,
        "normal_calls": stats.normal_calls,
        "strict_calls": stats.strict_calls,
        "empty_normal": stats.empty_normal,
        "empty_strict": stats.empty_strict,
        "operation_calls": stats.operation_calls,
        "cap_choices": stats.cap_choices,
        "normal_choices": stats.normal_choices,
        "strict_choices": stats.strict_choices,
        "submissions": stats.submissions,
        "resolves": stats.resolves,
        "rows": stats.rows,
        "device_tokens": stats.device_tokens,
        "prefill_rows": stats.prefill_rows,
        "decode_rows": stats.decode_rows,
        "mixed_submissions": stats.mixed_submissions,
        "prefill_submissions": stats.prefill_submissions,
        "decode_only_submissions": stats.decode_only_submissions,
        "mean_rows": stats.rows / stats.submissions,
        "row_histogram": dict(sorted(stats.row_histogram.items())),
        "token_histogram": dict(sorted(stats.token_histogram.items())),
        "pending_start_histogram": dict(sorted(stats.pending_start_histogram.items())),
        "pending_end_histogram": dict(sorted(stats.pending_end_histogram.items())),
        "normal_row_histogram": dict(sorted(stats.normal_row_histogram.items())),
        "strict_row_histogram": dict(sorted(stats.strict_row_histogram.items())),
        "admission_histogram": dict(sorted(stats.admission_histogram.items())),
        "fragmented_event_count": len(stats.fragmented_events),
        "fragmented_events_head": stats.fragmented_events[:20],
        "fragmented_events_tail": stats.fragmented_events[-20:],
    }
    return result, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("one", "future", "varied"), default="future")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    # Untimed warm-up exercises imports/caches before alternating measured arms.
    for mode in ("legacy", "candidate"):
        run_once(mode, args.shape)
    rows = []
    timings = {"legacy": [], "candidate": []}
    last = {}
    for repeat in range(args.repeats):
        order = ("legacy", "candidate") if repeat % 2 == 0 else ("candidate", "legacy")
        for mode in order:
            result, elapsed = run_once(mode, args.shape)
            timings[mode].append(elapsed)
            last[mode] = result
            rows.append({"repeat": repeat, "mode": mode, "elapsed_ms": elapsed})
    output = {
        "workload": {
            "requests": 128,
            "max_num_seqs": 16,
            "max_tokens": 128,
            "token_budget": 1024,
            "page_size": 16,
            "shape": args.shape,
        },
        "timings": rows,
        "median_ms": {mode: statistics.median(values) for mode, values in timings.items()},
        "candidate_over_legacy": (
            statistics.median(timings["candidate"])
            / statistics.median(timings["legacy"])
        ),
        "counts": last,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
