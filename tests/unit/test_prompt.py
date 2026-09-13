from __future__ import annotations

import threading

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import (
    AdmissionUpperBound,
    CacheHint,
    GenerationRequest,
    admission_upper_bound,
    backend_admission_upper_bound_async,
)
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
    prompt_from_wire,
    prompt_kind,
    prompt_text,
    prompt_to_wire,
    supplied_prompt_token_ids,
    token_prompt_fingerprint,
)


@pytest.mark.parametrize(
    "prompt",
    ["legacy", TextPrompt("typed"), TemplatedPrompt("templated")],
)
def test_text_prompt_round_trip_is_lossless(prompt):
    decoded = prompt_from_wire(prompt_to_wire(prompt))

    assert decoded == prompt
    assert type(decoded) is type(prompt)
    assert prompt_kind(decoded) == "text"
    assert prompt_text(decoded) in {"legacy", "typed", "templated"}


def test_tokens_prompt_round_trip_and_defensive_copy():
    source = [0, 7, 2**64 - 1]
    prompt = TokensPrompt(source, prompt="display only")
    source[0] = 99

    wire = prompt_to_wire(prompt)
    decoded = prompt_from_wire(wire)
    wire["prompt_token_ids"][0] = 42

    assert prompt.prompt_token_ids == (0, 7, 2**64 - 1)
    assert decoded == prompt
    assert decoded.prompt_token_ids == (0, 7, 2**64 - 1)


@pytest.mark.parametrize(
    ("token_ids", "error"),
    [
        ([], ValueError),
        ([True], TypeError),
        ([1.0], TypeError),
        ([-1], ValueError),
        ([2**64], ValueError),
    ],
)
def test_tokens_prompt_rejects_ambiguous_or_out_of_range_ids(token_ids, error):
    with pytest.raises(error):
        TokensPrompt(token_ids)


def test_token_prompt_helpers_and_fingerprint_ignore_display_text():
    without_text = TokensPrompt([9, 3, 1])
    with_text = TokensPrompt([9, 3, 1], prompt="not cache identity")

    assert prompt_kind(with_text) == "tokens"
    assert prompt_text(without_text) is None
    assert prompt_text(with_text) == "not cache identity"
    assert supplied_prompt_token_ids(with_text) == (9, 3, 1)
    assert token_prompt_fingerprint(without_text) == token_prompt_fingerprint(with_text)
    assert token_prompt_fingerprint(with_text) != token_prompt_fingerprint(
        TokensPrompt([1, 3, 9])
    )
    assert len(token_prompt_fingerprint(TokensPrompt([0, 2**64 - 1]))) == 64


def test_multimodal_data_round_trip_is_lossless_and_json_is_canonical():
    prompt = MultimodalPrompt(
        base=TokensPrompt([10, 11], prompt="caption"),
        items=[
            MultimodalItem("image", "uri", "https://example.test/image.png"),
            MultimodalItem("audio", "base64", "AAEC"),
            MultimodalItem("image", "bytes", b"\x00\xff"),
            MultimodalItem("metadata", "json", '{ "z": 2, "a": [1, true] }'),
        ],
    )

    decoded = prompt_from_wire(prompt_to_wire(prompt))

    assert decoded == prompt
    assert decoded.items[2].data == b"\x00\xff"
    assert decoded.items[3].data == '{"a":[1,true],"z":2}'
    assert prompt_kind(decoded) == "multimodal"
    assert prompt_text(decoded) is None
    assert supplied_prompt_token_ids(decoded) == (10, 11)


def test_json_multimodal_canonicalization_preserves_exact_decimal_values():
    item = MultimodalItem(
        "metadata",
        "json",
        '{"small":0.12345678901234567890123456789,"large":123456789012345678901}',
    )

    assert item.data == (
        '{"large":123456789012345678901,'
        '"small":0.12345678901234567890123456789}'
    )


def test_multimodal_prompt_defensively_copies_items():
    source = [MultimodalItem("image", "bytes", b"image")]
    prompt = MultimodalPrompt("describe", source)
    source.clear()

    assert prompt.items == (MultimodalItem("image", "bytes", b"image"),)


def test_multimodal_chat_layout_round_trip_preserves_roles_and_part_order():
    prompt = MultimodalPrompt(
        base=TextPrompt("display-only text"),
        items=(
            MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),
            MultimodalItem("image", "uri", "data:image/png;base64,AQID"),
        ),
        messages=(
            MultimodalMessage(
                "system",
                (MultimodalMessagePart("text", text="Answer with the color."),),
            ),
            MultimodalMessage(
                "user",
                (
                    MultimodalMessagePart("text", text="first="),
                    MultimodalMessagePart("item", item_index=0),
                    MultimodalMessagePart("text", text=" second="),
                    MultimodalMessagePart("item", item_index=1),
                ),
            ),
        ),
    )

    decoded = prompt_from_wire(prompt_to_wire(prompt))

    assert decoded == prompt
    assert [part.type for part in decoded.messages[1].content] == [
        "text",
        "item",
        "text",
        "item",
    ]
    assert [
        part.item_index
        for part in decoded.messages[1].content
        if part.type == "item"
    ] == [0, 1]


@pytest.mark.parametrize(
    "messages",
    [
        (
            MultimodalMessage(
                "user",
                (MultimodalMessagePart("item", item_index=1),),
            ),
        ),
        (
            MultimodalMessage(
                "user",
                (
                    MultimodalMessagePart("item", item_index=0),
                    MultimodalMessagePart("item", item_index=0),
                ),
            ),
        ),
    ],
)
def test_multimodal_chat_layout_cannot_drop_duplicate_or_misindex_items(messages):
    with pytest.raises(ValueError, match="reference every item exactly once"):
        MultimodalPrompt(
            base="display",
            items=(MultimodalItem("image", "bytes", b"image"),),
            messages=messages,
        )


@pytest.mark.parametrize(
    "item",
    [
        lambda: MultimodalItem("", "uri", "https://example.test"),
        lambda: MultimodalItem("image", "unknown", "value"),
        lambda: MultimodalItem("image", "bytes", "not bytes"),
        lambda: MultimodalItem("image", "uri", b"not text"),
        lambda: MultimodalItem("image", "base64", "%%%invalid"),
        lambda: MultimodalItem("metadata", "json", '{"duplicate":1,"duplicate":2}'),
        lambda: MultimodalItem("metadata", "json", '{"number":NaN}'),
    ],
)
def test_multimodal_item_rejects_invalid_or_ambiguous_data(item):
    with pytest.raises((TypeError, ValueError)):
        item()


def test_multimodal_prompt_requires_a_nonempty_typed_item_sequence():
    with pytest.raises(ValueError):
        MultimodalPrompt("text", [])
    with pytest.raises(TypeError):
        MultimodalPrompt("text", ["untyped"])
    with pytest.raises(TypeError):
        MultimodalPrompt(MultimodalPrompt("nested", [MultimodalItem("x", "bytes", b"x")]), [])


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "unknown"},
        {"prompt": "missing kind"},
        {"kind": "text", "prompt": "x", "extra": True},
        {"kind": "tokens", "prompt_token_ids": [1]},
        {"kind": "tokens", "prompt_token_ids": [1], "prompt": None, "extra": True},
        {"kind": "multimodal", "base": "x"},
        {
            "kind": "multimodal",
            "base": "x",
            "items": [
                {
                    "modality": "image",
                    "encoding": "bytes",
                    "data": b"x",
                    "extra": True,
                }
            ],
        },
    ],
)
def test_prompt_from_wire_rejects_unknown_missing_and_extra_fields(wire):
    with pytest.raises((TypeError, ValueError)):
        prompt_from_wire(wire)


@pytest.mark.parametrize("wire", [None, [], 1, b"text"])
def test_prompt_from_wire_rejects_untagged_non_strings(wire):
    with pytest.raises(TypeError):
        prompt_from_wire(wire)


def test_text_prefix_fingerprint_cannot_be_attached_to_non_text_prompt():
    with pytest.raises(ValueError, match="text prefix"):
        GenerationRequest(
            request_id="invalid-cache-domain",
            prompt=TokensPrompt((1, 2, 3)),
            sampling_params=SamplingParams(max_tokens=1),
            cache_hint=CacheHint(
                session_id="session",
                prefix_fingerprint="0" * 16,
            ),
        )


def test_token_admission_counts_exact_ids_not_optional_display_text():
    short = GenerationRequest(
        request_id="short-display",
        prompt=TokensPrompt((1, 2, 3), prompt="x"),
        sampling_params=SamplingParams(max_tokens=5),
    )
    long = GenerationRequest(
        request_id="long-display",
        prompt=TokensPrompt((1, 2, 3), prompt="x" * 100_000),
        sampling_params=SamplingParams(max_tokens=5),
    )

    assert admission_upper_bound(short).tokens == 8
    assert admission_upper_bound(long) == admission_upper_bound(short)


def test_immutable_sampling_extensions_remain_admission_serializable():
    request = GenerationRequest(
        request_id="json-extension",
        prompt="hello",
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"response_format": {"type": "json_object"}},
        ),
    )

    assert admission_upper_bound(request).tokens > 0


async def test_backend_admission_override_runs_off_event_loop():
    class AdmissionBackend:
        def __init__(self):
            self.thread_id = None

        def admission_upper_bound(self, request):
            self.thread_id = threading.get_ident()
            return AdmissionUpperBound(tokens=7, refundable_on_exact_usage=True)

    backend = AdmissionBackend()
    request = GenerationRequest(
        request_id="async-admission",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=1),
    )
    event_loop_thread = threading.get_ident()

    bound = await backend_admission_upper_bound_async(backend, request)

    assert bound.tokens == 7
    assert backend.thread_id != event_loop_thread


def test_multimodal_admission_never_guesses_processor_token_count():
    request = GenerationRequest(
        request_id="mm",
        prompt=MultimodalPrompt(
            "describe",
            (MultimodalItem("image", "bytes", b"image"),),
        ),
        sampling_params=SamplingParams(max_tokens=1),
    )

    with pytest.raises(ValueError, match="processed token count"):
        admission_upper_bound(request)
