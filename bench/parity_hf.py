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

_TOLERANCE = 0.99  # runbook §1 Gate 1 / G2 A2 match rate
_TOP_K = 20  # HF ranks recorded per position; below this a pick is not a tie
#: Floor for the tie threshold when the reference turns out to be perfectly
#: self-consistent. m2 §2.5 and G2 A2 both ask for a "logprob tolerance" without
#: fixing a number, and picking one by assertion is how a gate ends up measuring
#: its own threshold: the first attempt used 0.1 nats, but bf16 logits quantize
#: the gaps to multiples of ~0.125 at these magnitudes, so 0.1 could only ever
#: classify 0.0 as a tie and everything else as a fault. The tolerance is
#: therefore MEASURED — see `_reference_noise_floor` — and this is only the
#: fallback.
_MIN_TIE_GAP = 0.125


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
        continuation = generated[0][ids.shape[1] :].tolist()
        # One teacher-forced forward over prompt+continuation yields HF's
        # distribution at EVERY position at once. Recorded top-k, not just the
        # argmax, because the question a disagreement raises is "how far below
        # did HF rank what we picked?" — and a rate alone cannot answer it.
        with torch.no_grad():
            full = torch.tensor([ids[0].tolist() + continuation], device="cuda:0")
            logits = model(full).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        prompt_length = ids.shape[1]
        distributions = []
        for step in range(len(continuation)):
            # position predicting continuation[step] is the token before it
            row = log_probs[prompt_length + step - 1]
            values, indices = torch.topk(row, k=min(_TOP_K, row.shape[-1]))
            distributions.append(
                {str(int(t)): round(float(v), 5) for t, v in zip(indices, values, strict=True)}
            )
        reference[f"p{index}"] = {
            "prompt": text,
            "prompt_ids": ids[0].tolist(),
            "continuation": continuation,
            "hf_top_logprobs": distributions,
        }
    del model
    torch.cuda.empty_cache()
    return reference


def _reference_noise_floor(reference: dict) -> dict:
    """How well HF agrees with ITSELF, and at what logprob gap.

    `generate()` and a teacher-forced forward over the same sequence are two
    code paths through the same weights. In bf16 they do not always pick the
    same token. Every gate here is stated relative to that: an engine cannot be
    required to match a reference more closely than the reference matches
    itself, and a "substantive" threshold below the gap the reference produces
    on its own disagreements is measuring quantization, not correctness.
    """
    positions = 0
    inconsistent = 0
    gaps: list[float] = []
    for entry in reference.values():
        rows = entry.get("hf_top_logprobs") or []
        for index, token in enumerate(entry["continuation"]):
            if index >= len(rows):
                break
            row = rows[index]
            positions += 1
            if not row:
                continue
            best = max(row, key=lambda key: row[key])
            if int(best) != token:
                inconsistent += 1
                own = row.get(str(token))
                if own is not None:
                    gaps.append(round(row[best] - own, 5))
    return {
        "positions": positions,
        "self_inconsistent": inconsistent,
        "self_agreement_rate": round(1 - inconsistent / positions, 4) if positions else 1.0,
        "max_gap_at_self_disagreement": round(max(gaps), 5) if gaps else 0.0,
        "method": (
            "HF greedy generate() vs HF teacher-forced argmax over the same "
            "sequence: two paths through one set of weights"
        ),
    }


class _RecordingRunner:
    """Pass-through runner that keeps the first ``SampledToken`` per request.

    Transparent via ``__getattr__`` so ``release``/``shutdown`` still reach the
    real runner — a ``DistTPModelRunner`` that never sees ``release`` leaks its
    per-request sampler state on every rank.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.sampled: dict[str, object] = {}

    def execute(self, scheduled, states):
        out = self._inner.execute(scheduled, states)
        for request_id, tokens in (out or {}).items():
            if tokens and request_id not in self.sampled:
                self.sampled[request_id] = tokens[0]
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _engine_next_tokens(
    model_path: str, tp: int, reference: dict, num_pages: int, page_size: int
) -> dict[str, list]:
    """One token per teacher-forced prefix, through the ordinary request path."""
    from bench.parity_tp import _real_runner
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    inner, teardown = _real_runner(model_path, tp, num_pages, page_size)
    # EngineCore.step() returns finished request ids; the SampledToken that
    # carries the logprobs is local to it and discarded. Wrapping the runner is
    # the only seam that sees them without changing the engine.
    recorder = _RecordingRunner(inner)
    runner = recorder
    sampled = recorder.sampled
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
                        # run_to_completion() returns token ids only; step() hands
                        # back SampledToken, which is where the logprobs live
                        sampling=EngineSampling(logprobs=_TOP_K),
                    )
                )
        core.run_to_completion()
    finally:
        teardown()

    predicted: dict[str, list] = {name: [] for name in reference}
    for request_id, (name, _position) in sorted(index.items(), key=lambda kv: kv[1]):
        # a request that produced nothing is a disagreement, not a skip
        predicted[name].append(sampled.get(request_id))
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

    noise_floor = _reference_noise_floor(reference)
    # the tolerance is the reference's own instability, never tighter
    tie_gap = max(_MIN_TIE_GAP, noise_floor["max_gap_at_self_disagreement"])

    predicted = _engine_next_tokens(
        args.model_path, args.tp, reference, args.num_pages, args.page_size
    )

    total = 0
    agreed = 0
    per_prompt = {}
    disagreements = []
    logprob_deltas: list[float] = []
    for name, entry in reference.items():
        expected = entry["continuation"]
        actual = predicted[name]
        distributions = entry.get("hf_top_logprobs") or [{}] * len(expected)
        hits = 0
        for position, (token, hf_token) in enumerate(
            zip(actual, expected, strict=False)
        ):
            hf_row = distributions[position] if position < len(distributions) else {}
            engine_token = token.token_id if token is not None else None
            if engine_token == hf_token:
                hits += 1
                # agreement is not the whole story: the same choice can still
                # sit on a differently-shaped distribution
                if token is not None and token.logprob is not None:
                    hf_value = hf_row.get(str(hf_token))
                    if hf_value is not None:
                        logprob_deltas.append(abs(token.logprob - hf_value))
                continue
            hf_best = hf_row.get(str(hf_token))
            hf_for_engine_choice = hf_row.get(str(engine_token))
            # A2's own words: "reduction order shifts argmax ties". A disagreement
            # is a tie-break iff HF ITSELF ranked the two near-equally. Picking a
            # token HF put far below is a fault, not rounding — and only the gap
            # in HF's distribution can tell the two apart.
            gap = (
                round(hf_best - hf_for_engine_choice, 5)
                if hf_best is not None and hf_for_engine_choice is not None
                else None
            )
            disagreements.append(
                {
                    "prompt": name,
                    "position": position,
                    "engine": engine_token,
                    "hf": hf_token,
                    "hf_logprob_gap": gap,
                    "outside_hf_top_k": hf_for_engine_choice is None,
                    "tie_break": gap is not None and gap <= tie_gap,
                }
            )
        total += len(expected)
        agreed += hits
        per_prompt[name] = {
            "positions": len(expected),
            "agreed": hits,
            "rate": round(hits / len(expected), 4) if expected else 0.0,
        }

    rate = round(agreed / total, 4) if total else 0.0
    substantive = [d for d in disagreements if not d["tie_break"]]
    max_delta = round(max(logprob_deltas), 5) if logprob_deltas else 0.0
    mean_delta = (
        round(sum(logprob_deltas) / len(logprob_deltas), 5) if logprob_deltas else 0.0
    )
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
        # An engine cannot be required to match the reference more closely than
        # the reference matches itself. Reported first because it reframes every
        # number under it.
        "reference_noise_floor": noise_floor,
        "agreement": {
            "positions": total,
            "agreed": agreed,
            "rate": rate,
            "threshold": _TOLERANCE,
            "verdict": "PASS" if rate >= _TOLERANCE else "FAIL",
            "reference_self_agreement_rate": noise_floor["self_agreement_rate"],
            "at_or_above_reference_self_agreement": (
                rate >= noise_floor["self_agreement_rate"]
            ),
        },
        # A2 asks for a match rate AND a logprob tolerance. The rate alone cannot
        # separate "reduction order shifted a tie" from "the shard is wrong", and
        # the docs never fixed a number for the second half, so it is defined here
        # by what A2 actually claims: a disagreement is acceptable only where HF
        # itself ranked the two tokens within the MEASURED tie gap.
        "logprob_tolerance": {
            "tie_gap_nats": tie_gap,
            "tie_gap_source": (
                "measured from the reference's own self-disagreements"
                if tie_gap > _MIN_TIE_GAP
                else "floor (bf16 logprob resolution); reference was self-consistent"
            ),
            "top_k": _TOP_K,
            "agreeing_positions_max_abs_delta": max_delta,
            "agreeing_positions_mean_abs_delta": mean_delta,
            "disagreements": len(disagreements),
            "tie_breaks": len(disagreements) - len(substantive),
            "substantive": len(substantive),
            "verdict": "PASS" if not substantive else "FAIL",
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
        f"reference noise floor: HF agrees with itself on "
        f"{noise_floor['positions'] - noise_floor['self_inconsistent']}/"
        f"{noise_floor['positions']} = {noise_floor['self_agreement_rate']:.4f}"
    )
    print(
        f"teacher-forced agreement: {agreed}/{total} = {rate:.4f} "
        f"(threshold {_TOLERANCE}) -> {payload['agreement']['verdict']}"
    )
    print(
        f"logprob tolerance: {len(substantive)} substantive disagreement(s), "
        f"{len(disagreements) - len(substantive)} tie-break(s) within {tie_gap} nats; "
        f"max |delta| on agreeing positions {max_delta} "
        f"-> {payload['logprob_tolerance']['verdict']}"
    )
    # both halves must hold: a high rate with one badly-wrong pick is not a pass,
    # and a tie-break-only failure is not the same defect as a broken shard
    return 0 if rate >= _TOLERANCE and not substantive else 1


if __name__ == "__main__":
    raise SystemExit(main())
