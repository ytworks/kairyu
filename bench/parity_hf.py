"""Gate 1: teacher-forced next-token agreement vs HF transformers (runbook §1).

Free-running greedy comparison cannot answer "are our kernels right?". Once one
token differs the trajectories separate and every later token is compared against
a prefix the other side never saw, so a single moved near-tie reads the same as a
broken shard. Measured on Qwen3-32B, free-running tp=1 agreed with HF on 6/8
sequences — with both sides producing correct, fluent text and parting ways only
in open-ended continuation after the answer.

Teacher forcing removes that. Both sides are given the SAME prefix at every
position and asked only for the next token, so a disagreement is a disagreement
about that one prediction and nothing compounds.

The reference is HF's own greedy continuation, which makes ``reference[k]``
exactly HF's argmax given ``prompt + reference[:k]``. The engine is then asked
for one token at each of those prefixes, through the ordinary request path, so
paged KV, the attention backend and any TP sharding are all in the measurement.

Run:
  uv run python bench/parity_hf.py --model-path /models/qwen3-32b --tp 1
  uv run python bench/parity_hf.py --model-path /models/qwen3-32b --tp 8 \\
      --reference bench/results/hf-reference-<model>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.parity_tp import _TEXT_PROMPTS

_TOLERANCE = 0.99  # runbook §1 Gate 1 / G2 A2


def _build_reference(model_path: str, prompts: list[str], positions: int) -> dict:
    """HF greedy continuations. Freed before the engine loads — a 32B in bf16
    twice over does not fit on one card."""
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        .to("cuda:0")
        .eval()
    )
    reference = {}
    for index, text in enumerate(prompts):
        encoded = tokenizer(text, return_tensors="pt")
        ids = encoded.input_ids.to("cuda:0")
        with torch.no_grad():
            generated = model.generate(
                ids,
                attention_mask=encoded.attention_mask.to("cuda:0"),
                max_new_tokens=positions,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        reference[f"p{index}"] = {
            "prompt": text,
            "prompt_ids": ids[0].tolist(),
            "continuation": generated[0][ids.shape[1] :].tolist(),
        }
    del model
    torch.cuda.empty_cache()
    return reference


def _engine_next_tokens(
    model_path: str, tp: int, reference: dict, num_pages: int, page_size: int
) -> dict[str, list[int]]:
    """One token per teacher-forced prefix, through the ordinary request path."""
    from bench.parity_tp import _real_runner
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    runner, teardown = _real_runner(model_path, tp, num_pages, page_size)
    try:
        cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        scheduler = Scheduler(cache, max_num_batched_tokens=2048, page_size=page_size)
        core = EngineCore(scheduler, runner)
        index = {}
        for name, entry in reference.items():
            prompt_ids = entry["prompt_ids"]
            for position in range(len(entry["continuation"])):
                request_id = f"{name}@{position}"
                index[request_id] = (name, position)
                core.add_request(
                    EngineRequest(
                        request_id,
                        tuple(prompt_ids) + tuple(entry["continuation"][:position]),
                        max_new_tokens=1,
                        sampling=EngineSampling(),
                    )
                )
        outputs = core.run_to_completion()
    finally:
        teardown()

    predicted: dict[str, list[int]] = {name: [] for name in reference}
    for request_id, (name, position) in sorted(index.items(), key=lambda kv: kv[1]):
        tokens = list(outputs.get(request_id, ()))
        # a request that produced nothing is a disagreement, not a skip
        predicted[name].append(tokens[0] if tokens else -1)
        del position
    return predicted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--positions", type=int, default=16, help="teacher-forced steps")
    parser.add_argument("--num-pages", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--reference",
        type=Path,
        help="reuse (or create) the HF reference here; skips reloading a 32B "
        "just to reproduce the same greedy continuation",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    prompts = list(_TEXT_PROMPTS[: args.num_prompts])
    if len(prompts) < args.num_prompts:
        raise SystemExit(
            f"--num-prompts {args.num_prompts} exceeds the {len(_TEXT_PROMPTS)} fixed prompts"
        )

    if args.reference and args.reference.exists():
        reference = json.loads(args.reference.read_text())
        reference = {k: v for k, v in reference.items() if int(k[1:]) < args.num_prompts}
        for entry in reference.values():
            entry["continuation"] = entry["continuation"][: args.positions]
    else:
        reference = _build_reference(args.model_path, prompts, args.positions)
        if args.reference:
            args.reference.parent.mkdir(parents=True, exist_ok=True)
            args.reference.write_text(json.dumps(reference, indent=2) + "\n")

    predicted = _engine_next_tokens(
        args.model_path, args.tp, reference, args.num_pages, args.page_size
    )

    total = 0
    agreed = 0
    per_prompt = {}
    disagreements = []
    for name, entry in reference.items():
        expected = entry["continuation"]
        actual = predicted[name]
        hits = sum(1 for a, b in zip(actual, expected, strict=False) if a == b)
        total += len(expected)
        agreed += hits
        per_prompt[name] = {
            "positions": len(expected),
            "agreed": hits,
            "rate": round(hits / len(expected), 4) if expected else 0.0,
        }
        for position, (a, b) in enumerate(zip(actual, expected, strict=False)):
            if a != b:
                disagreements.append(
                    {"prompt": name, "position": position, "engine": a, "hf": b}
                )

    rate = round(agreed / total, 4) if total else 0.0
    from kairyu.engine.core.hw_profile import probe

    profile = probe()
    payload = {
        "gate": "runbook §1 Gate 1 — teacher-forced next-token agreement vs HF transformers",
        "config": {
            "model_path": args.model_path,
            "tensor_parallel_size": args.tp,
            "num_prompts": len(reference),
            "positions": args.positions,
            "dtype": "bfloat16" if profile.arch == "cuda" else "float32",
            "hardware": {
                "arch": profile.arch,
                "device_name": profile.device_name,
                "device_count": profile.device_count,
            },
            "method": (
                "both sides receive the identical prefix at every position; only "
                "the next token is compared, so trajectory divergence cannot compound"
            ),
        },
        "agreement": {
            "positions": total,
            "agreed": agreed,
            "rate": rate,
            "threshold": _TOLERANCE,
            "verdict": "PASS" if rate >= _TOLERANCE else "FAIL",
        },
        "per_prompt": per_prompt,
        # kept in full: which positions disagree is the diagnostic, and a summary
        # rate alone cannot tell a scattered near-tie from a systematic fault
        "disagreements": disagreements,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"written: {args.out}")
    print(
        f"teacher-forced agreement: {agreed}/{total} = {rate:.4f} "
        f"(threshold {_TOLERANCE}) -> {payload['agreement']['verdict']}"
    )
    return 0 if rate >= _TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
