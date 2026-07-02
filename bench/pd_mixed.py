"""Gate A10/B3 harness: P-D disaggregation vs colocated on a mixed workload.

Mixed trace = long prefills (batch jobs) + short-prompt long-decode requests
(the latency-SLO side A10 protects). Runs the trace through the PDCoordinator
(prefill core + decode core + KVHandoff) and through a single colocated
chunked-prefill core, verifying output equivalence and reporting structural
metrics: engine steps per side, handoff count, decode-side cache hits. On CPU
this pins the protocol; the GPU phase adds TPOT/TTFT percentiles and the
handoff-latency budget to the same results shape (with --prefill-node /
--decode-node pointing at real roles for B3).

Run: uv run python bench/pd_mixed.py
"""

from __future__ import annotations

import argparse
import json
import random

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.pd import LocalKVHandoff, PDCoordinator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

_VOCAB = 50_000
_SEED = 20260702


class _ToyRunner:
    def __init__(self) -> None:
        self.steps = 0

    def execute(self, scheduled, states):
        self.steps += 1
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                sampled[chunk.request_id] = (seed + 31 * chunk.position) % _VOCAB
        return sampled


def _mixed_trace(num_long: int, num_short: int) -> list[EngineRequest]:
    rng = random.Random(_SEED)
    trace: list[EngineRequest] = []
    for i in range(num_long):
        prompt_len = rng.randrange(256, 512)
        trace.append(
            EngineRequest(
                request_id=f"long{i}",
                prompt_token_ids=tuple(rng.randrange(_VOCAB) for _ in range(prompt_len)),
                max_new_tokens=4,
            )
        )
    for i in range(num_short):
        trace.append(
            EngineRequest(
                request_id=f"short{i}",
                prompt_token_ids=tuple(rng.randrange(_VOCAB) for _ in range(rng.randrange(8, 32))),
                max_new_tokens=32,
            )
        )
    rng.shuffle(trace)
    return trace


def _make_pair(num_pages: int, budget: int, **kwargs) -> tuple[Scheduler, RadixKVCache]:
    kv = RadixKVCache(num_pages=num_pages, page_size=16)
    return Scheduler(kv, max_num_batched_tokens=budget, page_size=16, **kwargs), kv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-long-prefills", type=int, default=8)
    parser.add_argument("--num-decode-slo", type=int, default=24)
    parser.add_argument("--prefill-budget", type=int, default=1024,
                        help="P-D prefill chunk budget (<=1024 per design m6 D4)")
    parser.add_argument("--prefill-node", default="local")
    parser.add_argument("--decode-node", default="local")
    args = parser.parse_args()

    trace = _mixed_trace(args.num_long_prefills, args.num_decode_slo)

    # colocated baseline: one chunked-prefill core
    colocated_scheduler, _ = _make_pair(num_pages=8192, budget=2048)
    colocated_runner = _ToyRunner()
    colocated = EngineCore(colocated_scheduler, colocated_runner)
    for request in trace:
        colocated.add_request(request)
    baseline = colocated.run_to_completion()

    # P-D split: prefill core + decode core, page-granular handoff
    prefill_scheduler, _ = _make_pair(num_pages=8192, budget=args.prefill_budget)
    decode_scheduler, decode_kv = _make_pair(
        num_pages=8192, budget=2048, pd_separation=True, decode_token_budget=256
    )
    prefill_runner, decode_runner = _ToyRunner(), _ToyRunner()
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_scheduler,
        prefill_runner=prefill_runner,
        decode_scheduler=decode_scheduler,
        decode_runner=decode_runner,
        handoff=LocalKVHandoff(decode_kv),
    )
    for request in trace:
        coordinator.add_request(request)
    disaggregated = coordinator.run_to_completion()

    equivalent = disaggregated == baseline
    result = {
        "config": {
            "harness": "cpu-toy (GPU phase adds TTFT/TPOT percentiles, same shape)",
            "prefill_node": args.prefill_node,
            "decode_node": args.decode_node,
            "num_long_prefills": args.num_long_prefills,
            "num_decode_slo": args.num_decode_slo,
            "prefill_chunk_budget": args.prefill_budget,
            "seed": _SEED,
        },
        "requests": len(trace),
        "output_equivalent_to_colocated": equivalent,
        "failed_requests": list(coordinator.failed_requests),
        "colocated_engine_steps": colocated_runner.steps,
        "pd_prefill_steps": prefill_runner.steps,
        "pd_decode_steps": decode_runner.steps,
        "decode_side_kv_hit_rate": round(decode_kv.hit_rate, 4),
    }
    print(json.dumps(result, indent=2))
    print(f"equivalence: {'OK' if equivalent else 'MISMATCH'}")


if __name__ == "__main__":
    main()
