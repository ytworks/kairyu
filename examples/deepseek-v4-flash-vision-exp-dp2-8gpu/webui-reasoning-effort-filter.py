"""
title: Reasoning Effort
description: Select the DeepSeek reasoning effort (low/high/max) from a dropdown in Chat Controls.
version: 0.2.0
"""

from typing import Literal

from pydantic import BaseModel, Field

# Open WebUI v0.12.3 reaches Kairyu over the Responses API (api_type
# "responses" in compose.yaml): it converts this chat body with
# convert_to_responses_payload, which forwards unknown top-level keys verbatim
# and replays saved output items as they were stored. Kairyu's /v1/responses is
# the strict OpenAI contract, so the inlet leaves only Responses fields: the
# effort becomes reasoning.effort, sampling keys the Responses API lacks are
# dropped, saved output items keep their Responses fields, and store is false
# because the Chat UI sends the whole conversation every turn.
_NON_RESPONSES_FIELDS = (
    "chat_template_kwargs",
    "top_k",
    "min_p",
    "seed",
    "repeat_penalty",
    "repeat_last_n",
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "tfs_z",
    "use_mmap",
    "use_mlock",
    "think",
    "format",
    "num_keep",
    "num_ctx",
    "num_batch",
    "num_thread",
    "num_gpu",
    "keep_alive",
)
_OUTPUT_ITEM_FIELDS = {
    "message": ("type", "role", "content"),
    "reasoning": ("type", "id", "summary", "content", "encrypted_content", "status"),
    "function_call": ("type", "id", "call_id", "name", "arguments"),
    "function_call_output": ("type", "call_id", "output"),
}


def _responses_body(body: dict) -> dict:
    effort = body.pop("reasoning_effort", None)
    if effort:
        body["reasoning"] = {"effort": effort}
    for field in _NON_RESPONSES_FIELDS:
        body.pop(field, None)
    body["store"] = False
    for message in body.get("messages") or ():
        output = message.get("output")
        if isinstance(output, list):
            message["output"] = [
                {key: item[key] for key in _OUTPUT_ITEM_FIELDS[item["type"]] if key in item}
                for item in output
                if isinstance(item, dict) and item.get("type") in _OUTPUT_ITEM_FIELDS
            ]
    return body


class Filter:
    """Open WebUI global filter exposing the L3 effort knob as a dropdown.

    The pinned Open WebUI v0.12.3 renders enum-typed user valves as a
    ``<select>`` in Chat Controls, replacing the stock free-text Advanced
    Params field. The selection is forwarded as ``reasoning.effort`` on the
    Responses body the inlet keeps to Responses fields (m11 D7 amendment
    2026-10-17); the levels are the
    checkpoint encoder's own vocabulary (``encoding_dsv4.py``), and ``default``
    leaves the field out so the server's non-thinking direct-chat default
    applies.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description=(
                "Reasoning effort for DeepSeek-V4-Flash-Vision-Exp. "
                "default = direct chat without thinking."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        valves = (__user__ or {}).get("valves")
        effort = getattr(valves, "reasoning_effort", None) if valves else None
        if effort == "default":
            body.pop("reasoning_effort", None)
        elif effort:
            body["reasoning_effort"] = effort
        return _responses_body(body)
