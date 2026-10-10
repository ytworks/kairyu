"""A fake vLLM upstream whose chat and ``/tokenize`` counts follow vLLM's rules.

Counts depend on every field a chat template renders, merged the way vLLM
0.30's ``ChatCompletionRequest`` and ``TokenizeChatRequest`` merge them, so a
token count equals billed usage only when Kairyu tokenizes what it generates.
"""

from __future__ import annotations

import json

import httpx

from kairyu.engine.openai_backend import OpenAICompatBackend


def _rendered_tokens(messages, tools, template_kwargs) -> int:
    return len(json.dumps([messages, tools or [], template_kwargs], sort_keys=True))


def _chat_template_kwargs(body: dict) -> dict:
    caller = body.get("chat_template_kwargs") or {}
    effort = body.get("reasoning_effort")
    extra = {
        "add_generation_prompt": body.get("add_generation_prompt", True),
        "continue_final_message": body.get("continue_final_message", False),
        "reasoning_effort": effort,
    }
    if effort is not None and "enable_thinking" not in caller:
        extra["enable_thinking"] = effort != "none"
    return caller | {k: v for k, v in extra.items() if v not in (None, "auto")}


def _tokenize_template_kwargs(body: dict) -> dict:
    return (body.get("chat_template_kwargs") or {}) | {
        "add_generation_prompt": body.get("add_generation_prompt", True),
        "continue_final_message": body.get("continue_final_message", False),
    }


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path == "/tokenize":
        count = (
            len(body["prompt"])
            if "prompt" in body
            else _rendered_tokens(
                body["messages"], body.get("tools"), _tokenize_template_kwargs(body)
            )
        )
        return httpx.Response(
            200, json={"count": count, "max_model_len": 1 << 20, "tokens": []}
        )
    assert request.url.path == "/v1/chat/completions"
    prompt_tokens = _rendered_tokens(
        body["messages"], body.get("tools"), _chat_template_kwargs(body)
    )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-fake",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
        },
    )


def fake_vllm_backend(model: str = "m") -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="http://vllm.test/v1",
        model=model,
        api_key_env=None,
        upstream="vllm",
        transport=httpx.MockTransport(_handler),
    )
