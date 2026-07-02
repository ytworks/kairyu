"""KV hit-rate measurement on a synthetic multi-turn, 50%-shared-prefix workload.

Runs the real RadixKVCache (kairyu/engine/core) against the acceptance workload
shape: a system prompt shared by all sessions (~50% of prompt tokens) plus
per-session growing conversation history. Prints only measured values.

This validates the KV-manager half of the M2 acceptance criterion
("共有prefix比率50%のワークロードでKVキャッシュヒット率>80%"); the end-to-end
engine measurement repeats this on GPU with real attention kernels.

Run: uv run python bench/multiturn_prefix.py
"""

from __future__ import annotations

import random

from kairyu.engine.core.radix_kv import RadixKVCache

PAGE_SIZE = 16
NUM_PAGES = 8192
NUM_SESSIONS = 64
TURNS_PER_SESSION = 8
SYSTEM_PROMPT_TOKENS = 512  # shared prefix across every session
TURN_TOKENS = 128  # user turn + assistant reply appended per turn


def main() -> None:
    rng = random.Random(42)
    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    system_prompt = tuple(range(SYSTEM_PROMPT_TOKENS))
    histories: dict[int, tuple[int, ...]] = {s: () for s in range(NUM_SESSIONS)}

    total_prompt_tokens = 0
    # interleave sessions the way concurrent serving would
    for _turn in range(TURNS_PER_SESSION):
        order = list(range(NUM_SESSIONS))
        rng.shuffle(order)
        for session in order:
            new_turn = tuple(
                rng.randrange(10_000, 1_000_000) for _ in range(TURN_TOKENS)
            )
            prompt = system_prompt + histories[session] + new_turn
            allocation = cache.allocate(prompt)
            cache.free(allocation)
            histories[session] = histories[session] + new_turn
            total_prompt_tokens += len(prompt)

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
