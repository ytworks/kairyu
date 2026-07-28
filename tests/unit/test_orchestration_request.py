import asyncio

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
)
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.outputs import CompletionOutput, TokenLogprob
from kairyu.sampling_params import SamplingParams

COMPLEX = (
    "First research the choices. Then make a plan. After that implement it. "
    "Finally verify the result end to end."
)

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        },
    },
}
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    },
)


class IntentBackend:
    def __init__(self, *, latency_s: float = 0.0) -> None:
        self.requests = []
        self.latency_s = latency_s

    def _result(self, request):
        if "[verifier]" in request.prompt:
            texts = ["PASS"]
        else:
            texts = [f"answer-{index}" for index in range(request.sampling_params.n)]
        completions = tuple(
            CompletionOutput(
                index=index,
                text=text,
                token_ids=(index,),
                finish_reason="stop",
                logprob_content=(
                    (
                        TokenLogprob(
                            token=text,
                            token_id=index,
                            logprob=-0.1,
                        ),
                    )
                    if request.sampling_params.logprobs is not None
                    else None
                ),
            )
            for index, text in enumerate(texts)
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            usage=GenerationUsage(
                prompt_tokens=3,
                completion_tokens=len(completions),
            ),
        )

    async def generate(self, request):
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        self.requests.append(request)
        return self._result(request)

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


def _call(prompt: str, *, n: int = 3, temperature: float = 0.25):
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=0.8,
        max_tokens=17,
        presence_penalty=0.2,
        frequency_penalty=-0.3,
        stop=("END",),
        seed=41,
        logprobs=2,
        extra_args={"response_format": SCHEMA},
    )
    return OrchestrationRequest(
        prompt=prompt,
        sampling_params=params,
        tools=TOOLS,
        tool_choice="required",
        response_format=SCHEMA,
    )


def test_internal_sampling_removes_only_final_output_intent():
    call = _call("hello")
    public = call.sampling_params.clone(best_of=5)
    call = OrchestrationRequest(
        prompt=call.prompt,
        sampling_params=public,
        tools=call.tools,
        tool_choice=call.tool_choice,
        response_format=call.response_format,
    )

    internal = call.internal_sampling_params()

    assert internal.n == 1
    assert internal.best_of is None
    assert internal.logprobs is None
    assert internal.extra_args == {}
    assert internal.temperature == 0.25
    assert internal.top_p == 0.8
    assert internal.max_tokens == 17
    assert internal.presence_penalty == 0.2
    assert internal.frequency_penalty == -0.3
    assert internal.stop == ("END",)
    assert internal.seed == 41
    assert call.sampling_params.n == 3
    assert call.sampling_params.best_of == 5
    assert call.sampling_params.extra_args["response_format"] == SCHEMA


def test_internal_sampling_caps_private_work_without_changing_final_intent():
    call = _call("hello")
    public = call.sampling_params.clone(max_tokens=8192)
    call = OrchestrationRequest(
        prompt=call.prompt,
        sampling_params=public,
        tools=call.tools,
        tool_choice=call.tool_choice,
        response_format=call.response_format,
    )

    internal = call.internal_sampling_params(max_tokens_cap=1024)

    assert internal.max_tokens == 1024
    assert call.sampling_params.max_tokens == 8192


def test_native_tool_prompt_is_added_once_and_honors_template_ownership():
    request = GenerationRequest(
        request_id="tools",
        prompt="question",
        sampling_params=SamplingParams(),
        tools=TOOLS,
        tool_choice="required",
    )

    rendered = prompt_with_tool_intent(request)

    assert rendered.count("Available functions:") == 1
    assert '"name":"lookup"' in rendered
    assert "must call" in rendered
    assert (
        prompt_with_tool_intent(
            GenerationRequest(
                request_id="templated",
                prompt=rendered,
                sampling_params=SamplingParams(),
                tools=TOOLS,
                tool_choice="required",
                tools_in_prompt=True,
            )
        )
        == rendered
    )


async def test_direct_route_preserves_complete_final_intent():
    backend = IntentBackend()
    orchestrator = Orchestrator({"tier1": backend, "tier2": backend})

    result = await orchestrator.run(_call("hello"))

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.sampling_params == _call("hello").sampling_params
    assert request.tools == TOOLS
    assert request.tool_choice == "required"
    assert [completion.index for completion in result.completions] == [0, 1, 2]
    assert all(completion.logprob_content for completion in result.completions)


async def test_conductor_applies_complete_intent_only_to_final_role():
    backend = IntentBackend()
    orchestrator = Orchestrator({"tier1": backend, "tier2": backend})

    result = await orchestrator.run(_call(COMPLEX))

    final = next(request for request in backend.requests if "[synthesizer]" in request.prompt)
    internal = [request for request in backend.requests if request is not final]
    assert final.sampling_params == _call(COMPLEX).sampling_params
    assert final.tools == TOOLS
    assert final.tool_choice == "required"
    assert len(result.completions) == 3
    assert internal
    assert all(request.sampling_params.n == 1 for request in internal)
    assert all(request.sampling_params.logprobs is None for request in internal)
    assert all(request.sampling_params.extra_args == {} for request in internal)
    assert all(request.tools == () and request.tool_choice is None for request in internal)


async def test_orchestrator_private_token_policy_caps_only_internal_roles():
    backend = IntentBackend()
    orchestrator = Orchestrator(
        {"tier1": backend, "tier2": backend},
        sampling_params=SamplingParams(max_tokens=8),
    )

    await orchestrator.run(_call(COMPLEX))

    final = next(request for request in backend.requests if "[synthesizer]" in request.prompt)
    internal = [request for request in backend.requests if request is not final]
    assert final.sampling_params.max_tokens == 17
    assert all(request.sampling_params.max_tokens == 8 for request in internal)
    assert orchestrator.describe_routing()["internal_max_tokens"] == 8


async def test_moa_preserves_seeded_proposals_and_complete_synthesis_intent():
    backend = IntentBackend()
    orchestrator = Orchestrator(
        {"tier1": backend, "tier2": backend},
        moa_samples=3,
    )

    result = await orchestrator.run(_call(COMPLEX))

    proposals = sorted(
        (request for request in backend.requests if "-propose-" in request.request_id),
        key=lambda request: request.request_id,
    )
    synthesis = next(
        request for request in backend.requests if request.request_id.endswith("-synthesize")
    )
    assert [request.sampling_params.seed for request in proposals] == [41, 42, 43]
    assert all(request.sampling_params.n == 1 for request in proposals)
    assert all(request.sampling_params.logprobs is None for request in proposals)
    assert all(request.sampling_params.extra_args == {} for request in proposals)
    assert synthesis.sampling_params == _call(COMPLEX).sampling_params
    assert synthesis.tools == TOOLS
    assert synthesis.tool_choice == "required"
    assert len(result.completions) == 3


async def test_concurrent_calls_do_not_mix_request_intent():
    backend = IntentBackend(latency_s=0.01)
    orchestrator = Orchestrator({"tier1": backend})
    first = _call("first", n=2, temperature=0.1)
    second = _call("second", n=3, temperature=1.7)

    first_result, second_result = await asyncio.gather(
        orchestrator.run(first),
        orchestrator.run(second),
    )

    by_prompt = {request.prompt: request for request in backend.requests}
    assert by_prompt["first"].sampling_params.temperature == 0.1
    assert by_prompt["first"].sampling_params.n == 2
    assert by_prompt["second"].sampling_params.temperature == 1.7
    assert by_prompt["second"].sampling_params.n == 3
    assert len(first_result.completions) == 2
    assert len(second_result.completions) == 3
