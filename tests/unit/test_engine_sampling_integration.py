"""Full-engine sampling integration (m8 D2): greedy equivalence with a Sampler
attached, seeded-sampling determinism, and structured output through the
engine with a char-level vocab + real xgrammar."""

import json

import pytest

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner

PAGE = 4

# char-level vocab large enough for JSON structural tokens; the last slot is
# the designated EOS — a completed grammar terminates by sampling it (m8 D2)
_CHARS = [chr(c) for c in range(32, 127)] + ["<eos>"]
_EOS_ID = len(_CHARS) - 1


def _render(tokens: tuple[int, ...]) -> str:
    return "".join(_CHARS[t] for t in tokens if t != _EOS_ID)


def _engine(sampler: Sampler | None, vocab: int = 128, num_pages: int = 128):
    model = TinyAttentionLM(vocab=vocab, seed=0)
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=PAGE)
    runner = TorchPagedRunner(model, num_pages=num_pages, page_size=PAGE, sampler=sampler)
    return EngineCore(scheduler, runner), model


def test_sampler_greedy_matches_plain_runner():
    prompt = (5, 9, 2, 11, 7)
    plain, model = _engine(sampler=None)
    plain.add_request(EngineRequest("a", prompt, max_new_tokens=8))
    reference = plain.run_to_completion()["a"]
    assert reference == model.reference_greedy(prompt, 8)

    sampled_engine, _ = _engine(sampler=Sampler())
    sampled_engine.add_request(
        EngineRequest("a", prompt, max_new_tokens=8, sampling=EngineSampling(temperature=0.0))
    )
    assert sampled_engine.run_to_completion()["a"] == reference


def test_seeded_sampling_is_reproducible_through_engine():
    prompt = (3, 1, 4, 1, 5)
    results = []
    for _ in range(2):
        engine, _ = _engine(sampler=Sampler())
        engine.add_request(
            EngineRequest(
                "a",
                prompt,
                max_new_tokens=6,
                sampling=EngineSampling(temperature=1.0, seed=42),
            )
        )
        results.append(engine.run_to_completion()["a"])
    assert results[0] == results[1]


def test_different_seeds_diverge_through_engine():
    prompt = (3, 1, 4, 1, 5)
    outs = []
    for seed in (1, 2, 3):
        engine, _ = _engine(sampler=Sampler())
        engine.add_request(
            EngineRequest(
                "a",
                prompt,
                max_new_tokens=8,
                sampling=EngineSampling(temperature=2.0, seed=seed),
            )
        )
        outs.append(engine.run_to_completion()["a"])
    assert len(set(outs)) > 1


# bounded-output schemas: an untrained greedy toy model can loop forever in
# unbounded regions (free strings / digit runs), so the engine gate uses
# schemas whose every position is tightly constrained
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        {
            "type": "object",
            "properties": {"tag": {"type": "string", "enum": ["a", "b"]}},
            "required": ["tag"],
        },
        {
            "type": "object",
            "properties": {"x": {"type": "boolean"}, "y": {"type": "boolean"}},
            "required": ["x", "y"],
        },
    ],
)
def test_structured_output_through_engine_yields_valid_json(schema):
    pytest.importorskip("xgrammar")
    sampler = Sampler(vocab_provider=lambda: _CHARS)
    engine, _ = _engine(sampler=sampler, vocab=len(_CHARS))
    engine.add_request(
        EngineRequest(
            "a",
            (1, 2, 3),
            max_new_tokens=200,
            eos_token_id=_EOS_ID,
            sampling=EngineSampling(temperature=0.0, json_schema=schema),
        )
    )
    tokens = engine.run_to_completion()["a"]
    text = _render(tokens)
    parsed = json.loads(text)  # grammar termination finished the request cleanly
    assert isinstance(parsed, dict)


def test_json_mode_masks_from_the_first_token():
    # end-to-end json_mode with an untrained model can babble inside free
    # strings until max_tokens (model quality, not grammar correctness) — the
    # engine gate asserts the mask constrains the very first sampled token
    pytest.importorskip("xgrammar")
    sampler = Sampler(vocab_provider=lambda: _CHARS)
    engine, _ = _engine(sampler=sampler, vocab=len(_CHARS))
    engine.add_request(
        EngineRequest(
            "a",
            (4, 5, 6),
            max_new_tokens=8,
            eos_token_id=_EOS_ID,
            sampling=EngineSampling(temperature=0.0, json_mode=True),
        )
    )
    tokens = engine.run_to_completion()["a"]
    assert _CHARS[tokens[0]] in '{["tfn-0123456789'  # a legal JSON value start


def test_grammar_finish_reason_is_stop():
    pytest.importorskip("xgrammar")
    sampler = Sampler(vocab_provider=lambda: _CHARS)
    model = TinyAttentionLM(vocab=len(_CHARS), seed=0)
    cache = RadixKVCache(num_pages=128, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=PAGE)
    runner = TorchPagedRunner(model, num_pages=128, page_size=PAGE, sampler=sampler)
    engine = EngineCore(scheduler, runner)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    engine.add_request(
        EngineRequest(
            "a",
            (1, 2, 3),
            max_new_tokens=200,
            eos_token_id=_EOS_ID,
            sampling=EngineSampling(temperature=0.0, json_schema=schema),
        )
    )
    engine.run_to_completion()
    assert scheduler.finish_reason("a") == "stop"
