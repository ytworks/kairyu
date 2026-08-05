"""Kairyu continuation log-likelihood extension gates."""

from __future__ import annotations

import math

import httpx
import pytest

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.prompt import TokensPrompt
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput, TokenLogprob
from tests.server._legacy_chat import create_legacy_app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _valid_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "score",
        "prompt": "Question:",
        "kairyu_continuation": " AB",
        "max_tokens": None,
        "temperature": 0,
        "logprobs": 0,
    }
    body.update(overrides)
    return body


def _valid_completion(**overrides: object) -> CompletionOutput:
    values: dict[str, object] = {
        "index": 0,
        # Continuation-only detokenization may be context dependent. The wire
        # response must echo the scored input after token evidence is exact.
        "text": "backend-local rendering",
        "token_ids": (21, 22),
        "finish_reason": "length",
        "logprob_content": (
            TokenLogprob(" A", 21, -0.25, bytes_=(32, 65)),
            TokenLogprob("B", 22, -1.5, bytes_=(66,)),
        ),
    }
    values.update(overrides)
    return CompletionOutput(**values)


class _ScoringBackend:
    def __init__(
        self,
        *,
        completions: tuple[CompletionOutput, ...] | None = None,
        tokenized: object = ((11, 12), (21, 22)),
        tokenize_error: Exception | None = None,
        prompt_token_ids: object = (11, 12),
    ) -> None:
        self.completions = completions or (_valid_completion(),)
        self.tokenized = tokenized
        self.tokenize_error = tokenize_error
        self.prompt_token_ids = prompt_token_ids
        self.tokenize_calls = 0
        self.requests = []

    def tokenize_loglikelihood(self, context: str, continuation: str):
        self.tokenize_calls += 1
        assert context == "Question:"
        assert continuation == " AB"
        if self.tokenize_error is not None:
            raise self.tokenize_error
        return self.tokenized

    async def generate(self, request):
        self.requests.append(request)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=self.completions,
            usage=GenerationUsage(prompt_tokens=2, completion_tokens=2),
            prompt_token_ids=self.prompt_token_ids,
        )

    async def shutdown(self) -> None:
        return None


class _OrdinaryBackend:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="ordinary",
                    token_ids=(7,),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(prompt_tokens=1, completion_tokens=1),
        )

    async def shutdown(self) -> None:
        return None


async def test_continuation_loglikelihood_dispatches_exact_forced_tokens():
    backend = _ScoringBackend()
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json=_valid_body(priority=-17),
        )

    assert response.status_code == 200
    assert backend.tokenize_calls == 1
    assert len(backend.requests) == 1
    dispatched = backend.requests[0]
    assert dispatched.prompt == TokensPrompt((11, 12), prompt="Question:")
    assert dispatched.priority == -17
    assert dispatched.sampling_params.max_tokens == 2
    assert dispatched.sampling_params.temperature == 0.0
    assert dispatched.sampling_params.logprobs == 0
    assert dispatched.sampling_params.ignore_eos is True
    assert dispatched.sampling_params.forced_token_ids == (21, 22)
    assert dispatched.sampling_params.skip_special_tokens is False

    payload = response.json()
    assert payload["mode"] == "loglikelihood"
    assert payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "prompt_tokens_details": None,
    }
    assert payload["choices"] == [
        {
            "index": 0,
            "text": " AB",
            "logprobs": {
                "tokens": [" A", "B"],
                "token_logprobs": [-0.25, -1.5],
                "top_logprobs": None,
                "text_offset": [0, 2],
            },
            "finish_reason": "length",
            "prompt_token_ids": [11, 12],
            "continuation_token_ids": [21, 22],
        }
    ]


async def test_ordinary_completion_wire_shape_is_unchanged():
    backend = _OrdinaryBackend()
    app = create_legacy_app(engines={"ordinary": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "ordinary", "prompt": "hello", "max_tokens": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "mode" not in payload
    assert "prompt_token_ids" not in payload["choices"][0]
    assert "continuation_token_ids" not in payload["choices"][0]
    assert backend.requests[0].prompt == "hello"
    assert backend.requests[0].sampling_params.forced_token_ids is None


async def test_continuation_loglikelihood_unsupported_model_is_stable_400():
    backend = _OrdinaryBackend()
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "kairyu_continuation is not supported by model 'score'"
    )
    assert backend.requests == []


async def test_orchestrated_model_is_capability_400_but_unknown_model_stays_404():
    worker = _OrdinaryBackend()
    app = create_legacy_app(
        engines={},
        orchestrators={"auto": Orchestrator({"worker": worker})},
    )

    async with _client(app) as client:
        unsupported = await client.post(
            "/v1/completions",
            json=_valid_body(model="auto"),
        )
        ordinary = await client.post(
            "/v1/completions",
            json={"model": "auto", "prompt": "hello"},
        )
        unknown = await client.post(
            "/v1/completions",
            json=_valid_body(model="typo"),
        )

    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["message"] == (
        "kairyu_continuation is not supported by model 'auto'"
    )
    assert ordinary.status_code == 404
    assert unknown.status_code == 404
    assert worker.requests == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            _valid_body(prompt=""),
            "requires one non-empty string prompt",
            id="empty-prompt",
        ),
        pytest.param(
            _valid_body(prompt=["Question:"]),
            "requires one non-empty string prompt",
            id="prompt-batch",
        ),
        pytest.param(
            _valid_body(kairyu_continuation=""),
            "must be a non-empty string",
            id="empty-continuation",
        ),
        pytest.param(_valid_body(stream=True), "does not support streaming", id="stream"),
        pytest.param(_valid_body(n=2), "requires n=1", id="n"),
        pytest.param(
            {
                key: value
                for key, value in _valid_body().items()
                if key != "max_tokens"
            },
            "requires max_tokens=null",
            id="implicit-max-tokens",
        ),
        pytest.param(
            _valid_body(max_tokens=1),
            "requires max_tokens=null",
            id="finite-max-tokens",
        ),
        pytest.param(_valid_body(logprobs=1), "requires logprobs=0", id="logprobs"),
        pytest.param(
            _valid_body(temperature=1),
            "requires temperature=0",
            id="temperature",
        ),
        pytest.param(_valid_body(top_p=0.5), "requires top_p=1.0", id="top-p"),
        pytest.param(_valid_body(top_k=5), "requires top_k=-1", id="top-k"),
        pytest.param(_valid_body(min_p=0.1), "requires min_p=0.0", id="min-p"),
        pytest.param(
            _valid_body(min_tokens=1),
            "requires min_tokens=0",
            id="min-tokens",
        ),
        pytest.param(
            _valid_body(presence_penalty=1),
            "requires presence_penalty=0.0",
            id="presence-penalty",
        ),
        pytest.param(
            _valid_body(frequency_penalty=1),
            "requires frequency_penalty=0.0",
            id="frequency-penalty",
        ),
        pytest.param(
            _valid_body(repetition_penalty=1.1),
            "requires repetition_penalty=1.0",
            id="repetition-penalty",
        ),
        pytest.param(_valid_body(seed=7), "does not support seed", id="seed"),
        pytest.param(
            _valid_body(ignore_eos=True),
            "controls ignore_eos internally",
            id="ignore-eos",
        ),
        pytest.param(
            _valid_body(skip_special_tokens=False),
            "controls skip_special_tokens internally",
            id="skip-special-tokens",
        ),
        pytest.param(_valid_body(stop="x"), "does not support stop", id="stop"),
        pytest.param(
            _valid_body(stop_token_ids=[1]),
            "does not support stop_token_ids",
            id="stop-token-ids",
        ),
        pytest.param(
            _valid_body(logit_bias={"21": 1}),
            "unsupported request fields: logit_bias",
            id="unknown-extra",
        ),
    ],
)
async def test_continuation_loglikelihood_rejects_misleading_surface(
    body: dict[str, object], message: str
):
    backend = _ScoringBackend()
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=body)

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert backend.tokenize_calls == 0
    assert backend.requests == []


async def test_continuation_boundary_failure_is_client_error_before_dispatch():
    backend = _ScoringBackend(tokenize_error=ValueError("private tokenizer detail"))
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "kairyu_continuation does not align with a model token boundary"
    )
    assert "private tokenizer detail" not in response.text
    assert backend.requests == []


@pytest.mark.parametrize(
    "tokenized",
    [
        pytest.param(((11, 12),), id="not-a-pair"),
        pytest.param(((11, 12), ()), id="empty-continuation"),
        pytest.param(((11, 12), (True,)), id="boolean-token"),
        pytest.param(((11, 12), (-1,)), id="negative-token"),
    ],
)
async def test_malformed_tokenizer_evidence_is_sanitized_502(tokenized: object):
    backend = _ScoringBackend(tokenized=tokenized)
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "upstream backend error (RuntimeError)",
        "type": "upstream_error",
        "code": "backend_error",
    }
    assert backend.requests == []


@pytest.mark.parametrize(
    "prompt_token_ids",
    [
        pytest.param((), id="missing"),
        pytest.param("11,12", id="not-a-token-sequence"),
        pytest.param((True, 12), id="boolean-token"),
        pytest.param((11.0, 12), id="float-token"),
        pytest.param((11, 99), id="wrong-token-ids"),
    ],
)
async def test_malformed_backend_prompt_evidence_is_sanitized_502(
    prompt_token_ids: object,
):
    backend = _ScoringBackend(prompt_token_ids=prompt_token_ids)
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"
    assert "loglikelihood backend" not in response.text
    assert len(backend.requests) == 1


@pytest.mark.parametrize(
    "completions",
    [
        pytest.param((_valid_completion(token_ids=(21, 99)),), id="wrong-token-ids"),
        pytest.param(
            (_valid_completion(index=False),),
            id="boolean-choice-index",
        ),
        pytest.param(
            (_valid_completion(token_ids=(21.0, 22)),),
            id="float-forced-token-id",
        ),
        pytest.param(
            (_valid_completion(finish_reason="stop"),),
            id="wrong-finish-reason",
        ),
        pytest.param((_valid_completion(logprob_content=None),), id="missing-logprobs"),
        pytest.param(
            (
                _valid_completion(
                    logprob_content=(
                        TokenLogprob(" A", 99, -0.25),
                        TokenLogprob("B", 22, -1.5),
                    )
                ),
            ),
            id="misaligned-logprob",
        ),
        pytest.param(
            (
                _valid_completion(
                    logprob_content=(
                        TokenLogprob(" A", 21.0, -0.25),
                        TokenLogprob("B", 22, -1.5),
                    )
                ),
            ),
            id="float-logprob-token-id",
        ),
        pytest.param(
            (
                _valid_completion(
                    logprob_content=(
                        TokenLogprob(" A", 21, False),
                        TokenLogprob("B", 22, -1.5),
                    )
                ),
            ),
            id="boolean-logprob",
        ),
        pytest.param(
            (
                _valid_completion(
                    logprob_content=(
                        TokenLogprob(" A", 21, math.nan),
                        TokenLogprob("B", 22, -1.5),
                    )
                ),
            ),
            id="non-finite-logprob",
        ),
        pytest.param(
            (
                _valid_completion(
                    logprob_content=(
                        TokenLogprob(" A", 21, 0.01),
                        TokenLogprob("B", 22, -1.5),
                    )
                ),
            ),
            id="positive-logprob",
        ),
        pytest.param((_valid_completion(), _valid_completion(index=1)), id="two-choices"),
    ],
)
async def test_malformed_scoring_evidence_is_sanitized_502(
    completions: tuple[CompletionOutput, ...],
):
    backend = _ScoringBackend(completions=completions)
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"
    assert "loglikelihood backend" not in response.text
    assert len(backend.requests) == 1


@pytest.mark.parametrize(
    "completions",
    [
        pytest.param(
            (
                _valid_completion(
                    token_ids=(True, 22),
                    logprob_content=(
                        TokenLogprob(" A", 1, -0.25),
                        TokenLogprob("B", 22, -1.5),
                    ),
                ),
            ),
            id="boolean-forced-token-id",
        ),
        pytest.param(
            (
                _valid_completion(
                    token_ids=(1, 22),
                    logprob_content=(
                        TokenLogprob(" A", True, -0.25),
                        TokenLogprob("B", 22, -1.5),
                    ),
                ),
            ),
            id="boolean-logprob-token-id",
        ),
    ],
)
async def test_boolean_token_ids_do_not_equal_integer_backend_evidence(
    completions: tuple[CompletionOutput, ...],
):
    backend = _ScoringBackend(
        completions=completions,
        tokenized=((11, 12), (1, 22)),
    )
    app = create_legacy_app(engines={"score": backend})

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=_valid_body())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"
    assert "loglikelihood backend" not in response.text
    assert len(backend.requests) == 1
