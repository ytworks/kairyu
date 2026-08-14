"""KV hit-rate measurement on a synthetic multi-turn, 50%-shared-prefix workload.

Runs the real RadixKVCache (kairyu/engine/core) against the acceptance workload
shape: a system prompt shared by all sessions (~50% of prompt tokens) plus
per-session growing conversation history. Prints only measured values.

This validates the KV-manager half of the M2 acceptance criterion
("共有prefix比率50%のワークロードでKVキャッシュヒット率>80%"); the end-to-end
engine measurement repeats this on GPU with real attention kernels.

Run: uv run python verification/fleet/performance/multiturn_prefix.py
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from kairyu.engine.core.radix_kv import RadixKVCache

PAGE_SIZE = 16
NUM_PAGES = 8192
NUM_SESSIONS = 64
TURNS_PER_SESSION = 8
SYSTEM_PROMPT_TOKENS = 512  # shared prefix across every session
TURN_TOKENS = 128  # user turn + assistant reply appended per turn
WORKLOAD_SEED = 42


@dataclass(frozen=True)
class WorkloadPrompt:
    """One request in the fixed A7 workload geometry.

    ``prompt_token_ids`` are synthetic IDs used only by this CPU RadixKV
    simulator.  Real-engine harnesses should preserve ``sequence``, ``session``,
    ``turn``, ``prompt_tokens``, and ``expected_cached_tokens`` while replacing
    the IDs with text encoded by the checkpoint's tokenizer.
    """

    sequence: int
    session: int
    turn: int
    prompt_token_ids: tuple[int, ...]
    expected_cached_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


def fixed_workload() -> tuple[WorkloadPrompt, ...]:
    """Build the deterministic 64-session, 8-turn A7 workload.

    The theoretical cached count assumes serialized requests, no eviction, and
    a cache that retains every completed prompt: the first request is cold,
    the other first-turn requests reuse the shared system prefix, and later
    turns reuse each session's complete preceding prompt.
    """

    rng = random.Random(WORKLOAD_SEED)
    system_prompt = tuple(range(SYSTEM_PROMPT_TOKENS))
    histories: dict[int, tuple[int, ...]] = {
        session: () for session in range(NUM_SESSIONS)
    }
    workload: list[WorkloadPrompt] = []
    for turn in range(TURNS_PER_SESSION):
        order = list(range(NUM_SESSIONS))
        rng.shuffle(order)
        for session in order:
            new_turn = tuple(
                rng.randrange(10_000, 1_000_000) for _ in range(TURN_TOKENS)
            )
            prompt = system_prompt + histories[session] + new_turn
            expected_cached = (
                0
                if not workload
                else SYSTEM_PROMPT_TOKENS + turn * TURN_TOKENS
            )
            workload.append(
                WorkloadPrompt(
                    sequence=len(workload),
                    session=session,
                    turn=turn,
                    prompt_token_ids=prompt,
                    expected_cached_tokens=expected_cached,
                )
            )
            histories[session] += new_turn
    return tuple(workload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # M5 topology labels (design m5 D6); recorded in the run config for G2 §8.
    # The GPU phase wires them into engine behavior (A7/A8 measurement modes).
    parser.add_argument("--replicas", type=int, default=1,
                        help="DP replica count for the A8 affinity mode (results label)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="TP degree for the A7 hit-rate mode (results label)")
    parser.add_argument("--pd", action="store_true",
                        help="measure under the prefill-decode split (results label)")
    return parser


def build_run_config(args: argparse.Namespace) -> dict:
    """Run config embedded in results so files carry topology (G2 §8, design m5 D6)."""
    return {
        "sessions": NUM_SESSIONS,
        "turns_per_session": TURNS_PER_SESSION,
        "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
        "turn_tokens": TURN_TOKENS,
        "replicas": args.replicas,
        "tensor_parallel": args.tensor_parallel,
        "pd": args.pd,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(f"config={json.dumps(build_run_config(args))}")
    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)

    total_prompt_tokens = 0
    for request in fixed_workload():
        allocation = cache.allocate(request.prompt_token_ids)
        cache.free(allocation)
        total_prompt_tokens += request.prompt_tokens

    shared_ratio = SYSTEM_PROMPT_TOKENS / (
        SYSTEM_PROMPT_TOKENS + (TURNS_PER_SESSION / 2 + 0.5) * TURN_TOKENS
    )
    print(
        f"sessions={NUM_SESSIONS} turns={TURNS_PER_SESSION} "
        f"system_prompt={SYSTEM_PROMPT_TOKENS}tok turn={TURN_TOKENS}tok "
        f"(shared-prefix ratio ~{shared_ratio:.0%} at mid-conversation)"
    )
    print(f"total prompt tokens processed: {total_prompt_tokens}")
    print(f"KV cache hit rate: {cache.hit_rate:.1%}")
    print(f"acceptance target >80%: {'MET' if cache.hit_rate > 0.80 else 'NOT MET'}")


if __name__ == "__main__":
    main()
