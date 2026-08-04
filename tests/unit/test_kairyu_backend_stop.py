"""Backend-level D1 behavior: tokenizer option, stop strings with SSE-safe
holdback, finish_reason plumbing, radix commit on stop (design m8 D1)."""

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.tokenizer import ToyTokenizer


def _request(request_id: str, prompt: str, **sampling) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(**sampling),
    )


class _EchoTokenizer(ToyTokenizer):
    """Decodes token ids to single letters so stop strings are constructible."""

    _ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def decode(self, token_ids) -> str:
        return "".join(self._ALPHABET[t % 26] for t in token_ids)


async def test_default_backend_unchanged():
    backend = KairyuBackend(num_pages=256)
    result = await backend.generate(_request("r1", "hello world", max_tokens=4))
    assert result.finished
    assert result.completions[0].finish_reason == "length"
    assert result.completions[0].text.startswith("tok")


async def test_tokenizer_instance_accepted():
    backend = KairyuBackend(num_pages=256, tokenizer=_EchoTokenizer())
    result = await backend.generate(_request("r1", "hello", max_tokens=3))
    assert len(result.completions[0].text) == 3  # one letter per token


def test_bad_tokenizer_path_fails_at_construction(tmp_path):
    with pytest.raises(ValueError, match="tokenizer"):
        KairyuBackend(num_pages=64, tokenizer=str(tmp_path / "nope"))


async def test_stop_string_truncates_text_and_sets_reason():
    backend = KairyuBackend(num_pages=256, tokenizer=_EchoTokenizer())
    # _ToyRunner output tokens are deterministic; generate once to learn the text
    probe = await backend.generate(_request("p", "hello", max_tokens=8))
    text = probe.completions[0].text
    stop = text[2:4]  # a substring that appears mid-stream
    result = await backend.generate(
        _request("r", "hello", max_tokens=8, stop=stop)
    )
    completion = result.completions[0]
    assert completion.finish_reason == "stop"
    assert stop not in completion.text
    assert completion.text == text[: text.index(stop)]


async def test_stop_token_id_is_hidden_from_text_but_retained_for_accounting():
    tokenizer = _EchoTokenizer()
    backend = KairyuBackend(num_pages=256, tokenizer=tokenizer)
    probe = await backend.generate(_request("p", "hello", max_tokens=8))
    stop_id = probe.completions[0].token_ids[1]

    partials = [
        partial
        async for partial in backend.stream(
            _request("r", "hello", max_tokens=8, stop_token_ids=(stop_id,))
        )
    ]
    final = partials[-1]
    completion = final.completions[0]

    assert completion.finish_reason == "stop"
    assert completion.token_ids == probe.completions[0].token_ids[:2]
    assert completion.text == tokenizer.decode(completion.token_ids[:-1])
    assert tokenizer.decode((stop_id,)) not in completion.text
    assert all(
        tokenizer.decode((stop_id,)) not in partial.completions[0].text
        for partial in partials
    )
    assert final.usage is not None
    assert final.usage.completion_tokens == 2


async def test_native_incremental_decoder_never_sees_the_stop_token():
    tokenizer = ToyTokenizer()
    backend = KairyuBackend(num_pages=256, tokenizer=tokenizer)
    probe = await backend.generate(_request("native-probe", "hello", max_tokens=6))
    stop_id = probe.completions[0].token_ids[1]

    partials = [
        partial
        async for partial in backend.stream(
            _request(
                "native-stop",
                "hello",
                max_tokens=6,
                stop_token_ids=(stop_id,),
            )
        )
    ]
    final = partials[-1]
    completion = final.completions[0]

    assert completion.token_ids == probe.completions[0].token_ids[:2]
    assert completion.text == tokenizer.decode(completion.token_ids[:-1])
    assert all(f"tok{stop_id}" not in partial.text for partial in partials)


async def test_stop_string_never_leaks_partial_prefix_in_stream():
    backend = KairyuBackend(num_pages=256, tokenizer=_EchoTokenizer())
    probe = await backend.generate(_request("p", "hello", max_tokens=8))
    text = probe.completions[0].text
    stop = text[3:6]
    partials = []
    async for partial in backend.stream(_request("r", "hello", max_tokens=8, stop=stop)):
        partials.append(partial.completions[0].text)
    final = partials[-1]
    assert final == text[: text.index(stop)]
    # no intermediate ever showed the stop string or text beyond the cut
    for chunk in partials:
        assert stop not in chunk
        assert final.startswith(chunk)


async def test_stop_finish_preserves_radix_reuse():
    backend = KairyuBackend(num_pages=256, tokenizer=_EchoTokenizer())
    probe = await backend.generate(_request("p", "warm cache prompt", max_tokens=8))
    stop = probe.completions[0].text[2:4]
    await backend.generate(_request("r1", "warm cache prompt", max_tokens=8, stop=stop))
    hits_before = backend._cache.hit_rate
    await backend.generate(_request("r2", "warm cache prompt", max_tokens=2))
    assert backend._cache.hit_rate >= hits_before  # prefix survived the stop finish


async def test_stream_finish_reason_length_unchanged():
    backend = KairyuBackend(num_pages=256)
    last = None
    async for partial in backend.stream(_request("r", "hello", max_tokens=3)):
        last = partial
    assert last is not None and last.finished
    assert last.completions[0].finish_reason == "length"


async def test_concurrent_with_and_without_stop():
    import asyncio

    backend = KairyuBackend(num_pages=256, tokenizer=_EchoTokenizer())
    probe = await backend.generate(_request("p", "alpha", max_tokens=6))
    stop = probe.completions[0].text[1:3]
    stopped, plain = await asyncio.gather(
        backend.generate(_request("s", "alpha", max_tokens=6, stop=stop)),
        backend.generate(_request("q", "beta prompt", max_tokens=4)),
    )
    assert stopped.completions[0].finish_reason == "stop"
    assert plain.completions[0].finish_reason == "length"
