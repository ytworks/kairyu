"""Teacher-forced next-token agreement vs HF transformers.

This is the formal HF-relative measurement for G2 A2 and the teacher-forced
half of G2 A1. A1 additionally requires full Llama-3.1-8B continuations with
the overlap pipeline ON and OFF, which ``bench/parity_tp.py`` records and
``bench/gate_a1.py`` assembles. A2 instead requires the fixed 64-prompt
Llama-3.3-70B FP8 measurement at TP=2/4/8; ``bench/gate_a2.py`` assembles those
three runs with the shared HF reference and fails closed on incomplete evidence.

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
  uv run python bench/parity_hf.py --model-path /models/llama-3.3-70b-fp8 \\
      --tp 2 --num-prompts 64 --positions 16 \\
      --reference bench/results/hf-reference-llama33-70b-fp8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.parity_tp import _TEXT_PROMPTS, _code_provenance

#: G2 A2, amended 2026-07-25: the bar is the REFERENCE's own self-agreement, not
#: a fixed percentage. HF's `generate()` and a teacher-forced forward over the
#: same sequence disagree with each other in bf16 (0.9805 on Qwen3-32B), and an
#: engine cannot be required to match a reference more closely than the reference
#: matches itself. The old fixed 0.99 is kept only as a reported reference point.
_REPORTED_REFERENCE_RATE = 0.99
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
#: Absolute logprob agreement required at positions where BOTH sides picked the
#: same token. This is the half of A2's "logprob tolerance" that a tie
#: classification cannot express: an engine can agree on every argmax while its
#: distribution is wildly different, and review [P1] on #131 demonstrated exactly
#: that — a mock returning logprob -100.0 against HF's -1.0 scored
#: `max_abs_delta: 99.0` and still reported PASS. Derived from the reference's
#: own bf16 resolution: gaps quantize at ~0.125, so twice that is the smallest
#: bound that is not measuring quantization.
_MAX_LOGPROB_DELTA = 0.25


_REFERENCE_SCHEMA = 5
#: Read size for the checkpoint digest. Large enough that hashlib releases the
#: GIL for the update, which is what makes the thread pool below worth having.
_HASH_CHUNK = 8 << 20


def _prompt_ids(tokenizer, text: str) -> list[int]:
    """Use the production Kairyu prompt contract: no implicit BOS/EOS.

    ``HFTokenizer.encode`` deliberately passes ``add_special_tokens=False``.
    The reference must do the same or it scores a different prefix while
    claiming to evaluate the full-continuation prompts from ``parity_tp.py``.
    """
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def _checkpoint_weight_digests(root: Path) -> dict[str, str]:
    """SHA-256 of every weight file's COMPLETE contents, one digest per file.

    The previous version hashed each shard's safetensors header plus four fixed
    4 KB windows through the payload, on the theory that different weights differ
    in the sampled bytes. They need not: an edit anywhere in the ~99.99% of a
    shard that no window covers leaves the fingerprint identical, so a reference
    cache built from DIFFERENT weights was accepted (review [P2] on #131). A
    sampled fingerprint cannot be the basis for cache safety, because the bytes it
    skips are exactly the ones a swap changes.

    So the whole file is read. On the 64 GB Qwen3-32B checkpoint this is ~20 s
    against a run that loads the same bytes onto a GPU anyway.

    Kept per file rather than merged into one opaque number: a safetensors
    shard's SHA-256 is the same digest the Hub publishes as that blob's LFS oid,
    so every entry below can be checked against the upstream revision with
    `sha256sum` and nothing from this repo. That is what makes committed evidence
    identify its checkpoint reproducibly (G2 §8) rather than by a path on one
    machine.
    """
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    weight_files = sorted(
        list(root.glob("*.safetensors")) + list(root.glob("*.safetensors.index.json"))
    )
    if not weight_files:
        raise SystemExit(f"{root} has no safetensors weights to fingerprint")

    def digest(path: Path) -> str:
        accumulator = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                accumulator.update(chunk)
        return accumulator.hexdigest()

    with ThreadPoolExecutor(max_workers=min(8, len(weight_files))) as pool:
        digests = list(pool.map(digest, weight_files))
    return {
        path.name: value for path, value in zip(weight_files, digests, strict=True)
    }


def _provenance(model_path: str, prompts: list[str], positions: int) -> dict:
    """What a cached reference must match before it may be scored against.

    Reuse was fail-open: the cache was keyed on nothing, so a file from another
    model, tokenizer or prompt set was accepted, and a SHORT continuation was
    scored as-is while `config.positions` still reported what was asked for
    (review [P1] on #131).
    """
    import hashlib

    import transformers

    root = Path(model_path)
    # config.json alone does NOT identify a checkpoint: a fine-tune, or any other
    # weights with the same architecture, hashes identically. Neither do names,
    # sizes and second-precision mtimes — a swap preserving those reads as the
    # same checkpoint, which is what the first attempt did while claiming to pin
    # "the actual bytes"; nor do sampled byte windows, which the second attempt
    # used and which miss any edit that falls between them (review [P2] on #131).
    weight_digests = _checkpoint_weight_digests(root)
    # a rollup for the console and for a quick eyeball across result files; the
    # per-file digests above are the identity, and are what `_load_reference`
    # actually compares
    weight_id = hashlib.sha256(
        "".join(
            f"{name}:{value}\n" for name, value in sorted(weight_digests.items())
        ).encode()
    ).hexdigest()[:16]
    config_path = root / "config.json"
    raw_config = json.loads(config_path.read_text())
    checkpoint = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    # the tokenizer's SERIALIZATION, not just its vocab: normalizer and
    # pre-tokenizer differences change tokenization while leaving get_vocab()
    # identical
    tokenizer_files = sorted(
        root.glob("tokenizer*.json")
    ) + sorted(root.glob("special_tokens_map.json"))
    tokenizer_id = hashlib.sha256(
        b"".join(f.read_bytes() for f in tokenizer_files)
    ).hexdigest()[:16] if tokenizer_files else None
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    vocab_id = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()[:16]
    # the actual ids the run will feed, so a tokenizer change that alters
    # tokenization is caught even if every file hash somehow matched
    prompt_hash = hashlib.sha256(
        json.dumps([_prompt_ids(tokenizer, text) for text in prompts]).encode()
    ).hexdigest()[:16]
    return {
        "schema": _REFERENCE_SCHEMA,
        "checkpoint_config_sha256": checkpoint,
        "checkpoint_contract": {
            "architectures": raw_config.get("architectures"),
            "hidden_size": raw_config.get("hidden_size"),
            "intermediate_size": raw_config.get("intermediate_size"),
            "num_hidden_layers": raw_config.get("num_hidden_layers"),
            "num_attention_heads": raw_config.get("num_attention_heads"),
            "num_key_value_heads": raw_config.get("num_key_value_heads"),
            "vocab_size": raw_config.get("vocab_size"),
            "torch_dtype": raw_config.get("torch_dtype"),
            "quantization_config": raw_config.get("quantization_config"),
        },
        # rollup of the per-file digests below, not an independent measurement
        "checkpoint_weights_sha256": weight_id,
        # full SHA-256 per weight file: checkable against the Hub's LFS oid for
        # the same blob, so the evidence names its checkpoint without depending
        # on `model_path` pointing at the same bytes tomorrow
        "checkpoint_weight_files": weight_digests,
        "tokenizer_files_sha256": tokenizer_id,
        "tokenizer_vocab_sha256": vocab_id,
        "prompt_token_ids_sha256": prompt_hash,
        "num_prompts": len(prompts),
        "positions": positions,
        # pinned so a checkpoint's own generation_config cannot quietly turn the
        # reference into sampling and pass its noise off as bf16 tie-breaking
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_new_tokens": positions,
            "max_new_tokens": positions,
        },
    }


def _load_reference(path: Path, expected: dict) -> dict:
    """Accept a cache only if it is the same measurement, and complete."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "provenance" not in payload:
        raise SystemExit(
            f"{path} predates the provenance envelope (schema {_REFERENCE_SCHEMA}); "
            "delete it and let the run rebuild the reference"
        )
    found = payload["provenance"]
    # checked first and by name: it is the identity of the weights, and a raw
    # dict-vs-dict message would be seventeen unreadable digests
    want_files = expected.get("checkpoint_weight_files") or {}
    found_files = found.get("checkpoint_weight_files") or {}
    if found_files != want_files:
        differing = sorted(
            (set(want_files) ^ set(found_files))
            | {
                name
                for name in set(want_files) & set(found_files)
                if want_files[name] != found_files[name]
            }
        )
        raise SystemExit(
            f"{path} was built from different checkpoint weights: "
            f"{', '.join(differing[:4])}{' …' if len(differing) > 4 else ''} "
            "differ from the checkpoint being scored. A reference is HF's output "
            "for a specific set of weights and cannot survive a swap."
        )
    for key, want in expected.items():
        if key == "checkpoint_weight_files":
            continue
        if found.get(key) != want:
            raise SystemExit(
                f"{path} does not match this run: {key} is {found.get(key)!r}, "
                f"expected {want!r}. A cache from another model, tokenizer or "
                "prompt set cannot be scored against."
            )
    entries = payload["reference"]
    if len(entries) != expected["num_prompts"]:
        raise SystemExit(
            f"{path} holds {len(entries)} prompts, expected {expected['num_prompts']}"
        )
    for name, entry in entries.items():
        if len(entry["continuation"]) != expected["positions"]:
            raise SystemExit(
                f"{path}: {name} has {len(entry['continuation'])} continuation "
                f"tokens, expected {expected['positions']} — a short cache would be "
                "scored while config.positions still reported the requested count"
            )
        rows = entry.get("hf_top_logprobs")
        # Without this the noise floor and the tolerance both go vacuous: empty
        # rows give positions=0, self-agreement 1.0, zero logprob samples and a
        # max delta of 0.0 — which reads as a clean PASS (review [P1] on #131).
        if not isinstance(rows, list) or len(rows) != expected["positions"]:
            raise SystemExit(
                f"{path}: {name} has {len(rows) if isinstance(rows, list) else 'no'} "
                f"top-logprob rows, expected {expected['positions']}"
            )
        for index, (row, token) in enumerate(zip(rows, entry["continuation"], strict=True)):
            if not row:
                raise SystemExit(f"{path}: {name} position {index} has an empty row")
            if str(token) not in row:
                raise SystemExit(
                    f"{path}: {name} position {index} does not rank its own "
                    f"reference token {token} — the gap for a disagreement there "
                    "could not be computed, so it would score as a free pass"
                )
    return entries


def _build_reference(
    model_path: str,
    prompts: list[str],
    positions: int,
    device_map: str | None = None,
    batch_size: int = 1,
) -> dict:
    """HF greedy continuations, optionally layer-dispatched across GPUs.

    ``device_map=auto`` is the 70B path: Accelerate distributes the checkpoint
    without changing HF's model implementation, and compressed-tensors keeps
    the checkpoint's FP8 weights/scales intact. The model is freed before the
    Kairyu TP candidate loads.
    """
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    load_kwargs = {"dtype": torch.bfloat16}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, **load_kwargs
    )
    if device_map is None:
        model = model.to("cuda:0")
    model = model.eval()
    input_device = model.get_input_embeddings().weight.device
    if batch_size < 1:
        raise ValueError(f"reference batch_size must be positive, got {batch_size}")
    if batch_size > 1:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    reference = {}
    for batch_start in range(0, len(prompts), batch_size):
        texts = prompts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=batch_size > 1,
        )
        ids = encoded.input_ids.to(input_device)
        attention_mask = encoded.attention_mask.to(input_device)
        with torch.no_grad():
            generated = model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=positions,
                min_new_tokens=positions,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        for batch_index, text in enumerate(texts):
            index = batch_start + batch_index
            prompt_ids = _prompt_ids(tokenizer, text)
            continuation = generated[batch_index][ids.shape[1] :].tolist()
            # One teacher-forced forward over prompt+continuation yields HF's
            # distribution at EVERY position at once. Recorded top-k, not just
            # argmax, because the question a disagreement raises is "how far
            # below did HF rank what we picked?"
            with torch.no_grad():
                full = torch.tensor(
                    [prompt_ids + continuation], device=input_device
                )
                logits = model(full).logits[0].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            prompt_length = len(prompt_ids)
            distributions = []
            for step in range(len(continuation)):
                # position predicting continuation[step] is the token before it
                row = log_probs[prompt_length + step - 1]
                values, indices = torch.topk(
                    row, k=min(_TOP_K, row.shape[-1])
                )
                distributions.append(
                    {
                        str(int(token)): round(float(value), 5)
                        for token, value in zip(indices, values, strict=True)
                    }
                )
            reference[f"p{index}"] = {
                "prompt": text,
                "prompt_ids": prompt_ids,
                "continuation": continuation,
                "hf_top_logprobs": distributions,
            }
    del model
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
    return reference


def decide(
    *,
    agreed: int,
    total: int,
    reference_self_agreement: float,
    max_abs_delta: float,
    substantive: int,
    missing_samples: int,
) -> tuple[bool, str]:
    """The verdict, as a pure function so it can be tested on its own inputs.

    Extracted because the first tests for these rules asserted that certain
    STRINGS appeared in the source — which cannot tell a live rule from dead code
    (review [P2] on #131). Every comparison here is on raw values; rounding is
    display only.
    """
    rate = agreed / total if total else 0.0
    if rate < reference_self_agreement:
        return False, (
            f"agreement {rate:.6f} is below the reference's own "
            f"{reference_self_agreement:.6f}"
        )
    if substantive:
        return False, f"{substantive} substantive disagreement(s)"
    if max_abs_delta > _MAX_LOGPROB_DELTA:
        return False, (
            f"max |logprob delta| {max_abs_delta} exceeds {_MAX_LOGPROB_DELTA}"
        )
    if missing_samples:
        return False, (
            f"{missing_samples} agreeing position(s) had no comparable logprob; "
            "a comparison that could not be made is not one that passed"
        )
    return True, "within the reference's noise floor on both halves"


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
        # unrounded: the comparison must not be decided by a display artefact
        "raw_self_agreement_rate": (1 - inconsistent / positions) if positions else 1.0,
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


def _teacher_forced_waves(reference: dict):
    """Yield independent prefixes without co-scheduling nested ownership.

    Every position for one prompt extends the previous position. Scheduling
    those nested prefixes in the same engine step can make one request write a
    page that a longer request already treats as cached. That is not a valid
    serving batch. A position wave still batches all independent prompts while
    preserving the exact teacher-forced inputs.
    """

    positions = max(
        (len(entry["continuation"]) for entry in reference.values()),
        default=0,
    )
    for position in range(positions):
        yield tuple(
            (
                name,
                position,
                tuple(entry["prompt_ids"])
                + tuple(entry["continuation"][:position]),
            )
            for name, entry in reference.items()
            if position < len(entry["continuation"])
        )


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
        for wave in _teacher_forced_waves(reference):
            for name, position, prompt_ids in wave:
                request_id = f"{name}@{position}"
                index[request_id] = (name, position)
                core.add_request(
                    EngineRequest(
                        request_id,
                        prompt_ids,
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
        help="reuse (or create) the HF reference here; skips reloading the "
        "reference model for every TP degree",
    )
    parser.add_argument(
        "--reference-device-map",
        choices=("auto",),
        help="HF/Accelerate device map used only while creating the reference",
    )
    parser.add_argument("--reference-batch-size", type=int, default=1)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="create/validate the reference and exit before loading Kairyu",
    )
    parser.add_argument("--checkpoint-repo")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.reference_only and args.reference is None:
        parser.error("--reference-only requires --reference")
    if bool(args.checkpoint_repo) != bool(args.checkpoint_revision):
        parser.error(
            "--checkpoint-repo and --checkpoint-revision must be provided together"
        )

    prompts = list(_TEXT_PROMPTS[: args.num_prompts])
    if len(prompts) < args.num_prompts:
        raise SystemExit(
            f"--num-prompts {args.num_prompts} exceeds the {len(_TEXT_PROMPTS)} fixed prompts"
        )

    provenance = _provenance(args.model_path, prompts, args.positions)
    if args.reference and args.reference.exists():
        # no truncation, no key filtering: a cache either IS this measurement or
        # it is rejected. Silently trimming it is how a 1-prompt file got scored
        # while config.positions still claimed 16.
        reference = _load_reference(args.reference, provenance)
    else:
        reference = _build_reference(
            args.model_path,
            prompts,
            args.positions,
            device_map=args.reference_device_map,
            batch_size=args.reference_batch_size,
        )
        if args.reference:
            import torch
            import transformers

            args.reference.parent.mkdir(parents=True, exist_ok=True)
            args.reference.write_text(
                json.dumps(
                    {
                        "schema_version": _REFERENCE_SCHEMA,
                        "provenance": provenance,
                        "reference_runtime": {
                            "backend": "HF transformers",
                            "code": _code_provenance(),
                            "transformers": transformers.__version__,
                            "torch": torch.__version__,
                            "torch_cuda": torch.version.cuda,
                            "dtype": "bfloat16",
                            "device_map": args.reference_device_map,
                            "batch_size": args.reference_batch_size,
                            "visible_device_count": torch.cuda.device_count(),
                            "checkpoint_repo": args.checkpoint_repo,
                            "checkpoint_revision": args.checkpoint_revision,
                        },
                        "reference": reference,
                    },
                    indent=2,
                )
                + "\n"
            )

    noise_floor = _reference_noise_floor(reference)
    if args.reference_only:
        print(json.dumps({"reference_noise_floor": noise_floor}, indent=2))
        print(f"written: {args.reference}")
        return 0
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
    missing_deltas: list[dict] = []
    raw_positions: list[dict] = []
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
            engine_logprob = token.logprob if token is not None else None
            engine_top_logprobs = (
                {str(token_id): value for token_id, value in token.top_logprobs}
                if token is not None and token.top_logprobs is not None
                else None
            )
            raw_position = {
                "prompt": name,
                "position": position,
                "reference_token": hf_token,
                "engine_token": engine_token,
                "agreement": engine_token == hf_token,
                "reference_token_logprob": hf_row.get(str(hf_token)),
                "engine_token_logprob": engine_logprob,
                "reference_top_logprobs": hf_row,
                "engine_top_logprobs": engine_top_logprobs,
            }
            if engine_token == hf_token:
                hits += 1
                # agreement is not the whole story: the same choice can still sit
                # on a differently-shaped distribution. A MISSING sample is not a
                # free pass — silently skipping it is how a run with zero deltas
                # reported max_abs_delta 0.0 and PASSed.
                hf_value = hf_row.get(str(hf_token))
                if token is None or token.logprob is None or hf_value is None:
                    missing_deltas.append({"prompt": name, "position": position})
                    raw_position["absolute_logprob_delta"] = None
                else:
                    delta = abs(token.logprob - hf_value)
                    logprob_deltas.append(delta)
                    raw_position["absolute_logprob_delta"] = delta
                raw_positions.append(raw_position)
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
            raw_position["reference_logprob_gap"] = gap
            raw_position["outside_reference_top_k"] = hf_for_engine_choice is None
            raw_position["tie_break"] = gap is not None and gap <= tie_gap
            raw_positions.append(raw_position)
        total += len(expected)
        agreed += hits
        per_prompt[name] = {
            "positions": len(expected),
            "agreed": hits,
            "rate": round(hits / len(expected), 4) if expected else 0.0,
        }

    # RAW ratio for every comparison; rounding is display only. 296/299 rounds to
    # 0.99 and would pass a >= 0.99 test while the real value is 0.98997
    # (review [P1] on #131).
    raw_rate = agreed / total if total else 0.0
    rate = round(raw_rate, 4)
    substantive = [d for d in disagreements if not d["tie_break"]]
    raw_max_delta = max(logprob_deltas) if logprob_deltas else 0.0
    max_delta = round(raw_max_delta, 5)
    passed, verdict_reason = decide(
        agreed=agreed,
        total=total,
        reference_self_agreement=noise_floor["raw_self_agreement_rate"],
        max_abs_delta=raw_max_delta,
        substantive=len(substantive),
        missing_samples=len(missing_deltas),
    )
    tolerance_ok = not substantive and not missing_deltas and (
        raw_max_delta <= _MAX_LOGPROB_DELTA
    )
    mean_delta = (
        round(sum(logprob_deltas) / len(logprob_deltas), 5) if logprob_deltas else 0.0
    )
    from bench.parity_tp import _gpu_runtime_provenance
    from kairyu.engine.core.hw_profile import probe

    profile = probe()
    payload = {
        "schema_version": 5,
        "measurement": (
            "teacher-forced next-token agreement vs HF transformers "
            "(formal G2 A2 measurement; G2 A1 additionally requires full "
            "greedy continuations with overlap ON and OFF)"
        ),
        "config": {
            "model_path": args.model_path,
            "checkpoint_repo": args.checkpoint_repo,
            "checkpoint_revision": args.checkpoint_revision,
            "reference_provenance": provenance,
            "code": _code_provenance(),
            "tensor_parallel_size": args.tp,
            "num_prompts": len(reference),
            "positions": args.positions,
            "dtype": "bfloat16" if profile.arch == "cuda" else "float32",
            "hardware": {
                "arch": profile.arch,
                "device_name": profile.device_name,
                "device_count": profile.device_count,
                "runtime": (
                    _gpu_runtime_provenance()
                    if profile.arch == "cuda"
                    else None
                ),
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
            "threshold": noise_floor["raw_self_agreement_rate"],
            "threshold_source": "the reference's own self-agreement (G2 A2, amended 2026-07-25)",
            "historical_fixed_threshold": _REPORTED_REFERENCE_RATE,
            "verdict": "PASS" if raw_rate >= noise_floor["raw_self_agreement_rate"] else "FAIL",
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
            "max_abs_delta_bound": _MAX_LOGPROB_DELTA,
            "within_delta_bound": raw_max_delta <= _MAX_LOGPROB_DELTA,
            "expected_samples": agreed,
            "collected_samples": len(logprob_deltas),
            "missing_samples": missing_deltas,
            # BOTH halves: agreeing on every argmax while the distribution is far
            # off is not a logprob tolerance pass
            # a comparison that could not be made is not a comparison that passed
            "verdict": "PASS" if tolerance_ok else "FAIL",
        },
        "per_prompt": per_prompt,
        "raw_positions": raw_positions,
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
        f"(bar = reference self-agreement {noise_floor['self_agreement_rate']:.4f}) "
        f"-> {payload['agreement']['verdict']}"
    )
    print(
        f"logprob tolerance: {len(substantive)} substantive disagreement(s), "
        f"{len(disagreements) - len(substantive)} tie-break(s) within {tie_gap} nats; "
        f"max |delta| on agreeing positions {max_delta} "
        f"-> {payload['logprob_tolerance']['verdict']}"
    )
    # both halves must hold: a high rate with one badly-wrong pick is not a pass,
    # and neither half is a fixed number — each is measured against the reference
    print(f"verdict: {verdict_reason}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
