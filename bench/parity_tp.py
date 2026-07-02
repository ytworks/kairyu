"""Gate A1/A2 harness: greedy token parity across TP degrees (goal G2, m5 §6).

Runs the same fixed prompts through the kairyu engine at each TP degree and
compares outputs pairwise against the base degree, with the overlap pipeline
ON and OFF. On CPU this exercises the real TPModelRunner broadcast/divergence
path over FakeCommunicator ranks (deterministic — parity must be exact); the
GPU phase runs the identical script against real ranks, where A2 allows
match-rate >=99% + logprob tolerance instead of bit-exactness.

Run: uv run python bench/parity_tp.py --tp 1,2 --num-prompts 64
"""

from __future__ import annotations

import argparse
import json
import random

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.overlap import OverlapEngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree

_VOCAB = 50_000
_SEED = 20260702


class _ToyRunner:
    """Deterministic CPU forward (kairyu_backend's toy runner)."""

    def execute(self, scheduled, states):
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                sampled[chunk.request_id] = (seed + 31 * chunk.position) % _VOCAB
        return sampled


def _fixed_prompts(count: int) -> list[EngineRequest]:
    rng = random.Random(_SEED)
    return [
        EngineRequest(
            request_id=f"p{i}",
            prompt_token_ids=tuple(rng.randrange(_VOCAB) for _ in range(rng.randrange(8, 96))),
            max_new_tokens=16,
        )
        for i in range(count)
    ]


def _make_runner(tp: int):
    if tp == 1:
        return _ToyRunner()
    validate_tp_degree(tp)
    comms = FakeCommunicator.create_group(tp)
    return TPModelRunner(rank_runners=tuple(_ToyRunner() for _ in range(tp)), comms=comms)


def _run(tp: int, overlap: bool, prompts: list[EngineRequest]) -> dict[str, tuple[int, ...]]:
    kv = RadixKVCache(num_pages=4096, page_size=16)
    scheduler = Scheduler(kv, max_num_batched_tokens=2048, page_size=16)
    runner = _make_runner(tp)
    core = OverlapEngineCore(scheduler, runner) if overlap else EngineCore(scheduler, runner)
    for request in prompts:
        core.add_request(request)
    return core.run_to_completion()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", default="1,2", help="comma-separated TP degrees; first is base")
    parser.add_argument("--num-prompts", type=int, default=64)
    args = parser.parse_args()
    degrees = [int(value) for value in args.tp.split(",")]
    if len(degrees) < 2:
        raise SystemExit("need at least a base degree and one comparison degree")

    prompts = _fixed_prompts(args.num_prompts)
    results = {}
    for overlap in (False, True):
        base = _run(degrees[0], overlap, prompts)
        for degree in degrees[1:]:
            candidate = _run(degree, overlap, prompts)
            matches = sum(base[rid] == candidate[rid] for rid in base)
            results[f"tp{degree}_vs_tp{degrees[0]}_overlap_{'on' if overlap else 'off'}"] = {
                "prompts": len(base),
                "exact_match": matches,
                "match_rate": round(matches / len(base), 4),
            }

    config = {
        "harness": "cpu-toy (GPU phase: real ranks, same script)",
        "tp_degrees": degrees,
        "num_prompts": args.num_prompts,
        "seed": _SEED,
    }
    print(json.dumps({"config": config, "parity": results}, indent=2))
    worst = min(entry["match_rate"] for entry in results.values())
    verdict = "OK" if worst == 1.0 else "CHECK (A2 allows >=0.99 on GPU)"
    print(f"worst match rate: {worst:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
