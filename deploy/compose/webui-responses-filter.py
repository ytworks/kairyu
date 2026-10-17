"""
title: Responses API
description: Keep only OpenAI Responses fields in the Chat UI's requests to Kairyu.
version: 0.1.0
"""

from pydantic import BaseModel

# The compose demo's Open WebUI reaches Kairyu over the Responses API (api_type
# "responses" in docker-compose.webui.yaml): it converts its chat body with
# convert_to_responses_payload, which forwards unknown top-level keys verbatim
# and replays saved output items with only some local fields removed. Kairyu's
# /v1/responses is the strict OpenAI contract, so the inlet leaves only
# Responses fields: the Chat Controls effort becomes reasoning.effort, sampling
# keys the Responses API lacks are dropped, saved output items keep their
# Responses fields, and store is false because the UI sends the whole
# conversation every turn.
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
    """Open WebUI global filter: the Chat UI's body as an OpenAI Responses body."""

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        return _responses_body(body)
