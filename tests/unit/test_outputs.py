import pytest

from kairyu import CompletionOutput, RequestOutput


def _completion(index: int = 0, text: str = "hello") -> CompletionOutput:
    return CompletionOutput(
        index=index,
        text=text,
        token_ids=(1, 2, 3),
        cumulative_logprob=-0.5,
        finish_reason="stop",
    )


def test_completion_output_surface_matches_vllm():
    out = _completion()
    assert out.index == 0
    assert out.text == "hello"
    assert out.token_ids == (1, 2, 3)
    assert out.cumulative_logprob == -0.5
    assert out.logprobs is None
    assert out.finish_reason == "stop"
    assert out.stop_reason is None


def test_request_output_surface_matches_vllm():
    request_output = RequestOutput(
        request_id="req-1",
        prompt="hi",
        prompt_token_ids=(7, 8),
        outputs=(_completion(),),
    )
    assert request_output.request_id == "req-1"
    assert request_output.prompt == "hi"
    assert request_output.prompt_token_ids == (7, 8)
    assert request_output.outputs[0].text == "hello"
    assert request_output.finished is True
    assert request_output.metrics is None


def test_outputs_are_immutable():
    out = _completion()
    with pytest.raises(AttributeError):
        out.text = "changed"
