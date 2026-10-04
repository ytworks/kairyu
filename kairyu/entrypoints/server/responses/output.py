"""Responses output items, terminal status, and metering of an executed chat."""

from __future__ import annotations

import uuid

from fastapi import Request

from kairyu.entrypoints.server.chat_service import ChatRequestError, ExecutedChat
from kairyu.entrypoints.server.metering import record_state_usage
from kairyu.entrypoints.server.responses.envelope import usage_payload
from kairyu.entrypoints.server.responses.request import ResponsesRequest
from kairyu.entrypoints.server.responses.tools import namespace_names


def record_execution(
    http_request: Request,
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> None:
    usage = usage_payload(
        execution.result.prompt,
        execution.result.completions,
        execution.result.usage,
    )
    owner = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        cached_tokens=usage["input_tokens_details"]["cached_tokens"],
        reservation=getattr(http_request.state, "tenant_admission", None),
        usage_exact=execution.result.usage is not None,
    )


def output_items(
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> list[dict]:
    if not execution.response.choices:
        return []
    message = execution.response.choices[0].message.model_dump(mode="json")
    return output_items_from_message(request, message)


def output_items_from_message(request: ResponsesRequest, message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    if calls:
        namespaces = namespace_names(request.tools)
        items = []
        for call in calls:
            function = call.get("function") or {}
            call_name = function.get("name") or ""
            namespace_name = namespaces.get(call_name)
            item = {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": call.get("id") or "",
                "type": "function_call",
                "name": (
                    namespace_name[1] if namespace_name is not None else call_name
                ),
                "arguments": function.get("arguments") or "",
                "status": "completed",
            }
            if namespace_name is not None:
                item["namespace"] = namespace_name[0]
            items.append(item)
        return items
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return [
        {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    ]


def validate_parallel_tool_calls(
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> None:
    if request.parallel_tool_calls:
        return
    calls = sum(
        len(choice.message.tool_calls or ())
        for choice in execution.response.choices
    )
    if calls > 1:
        raise ChatRequestError(
            "upstream model emitted multiple calls while parallel_tool_calls=false",
            status_code=502,
            code="parallel_tool_calls_not_satisfied",
            error_type="upstream_error",
            execution=execution,
        )


def terminal_status(execution: ExecutedChat) -> tuple[str, dict | None]:
    return terminal_status_for(
        completion.finish_reason for completion in execution.result.completions
    )


def terminal_status_for(finish_reasons) -> tuple[str, dict | None]:
    if any(reason in {"length", "max_tokens"} for reason in finish_reasons):
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def apply_terminal_item_status(output: list[dict], status: str) -> None:
    if status != "incomplete":
        return
    for item in output:
        if item["type"] == "message":
            item["status"] = "incomplete"
