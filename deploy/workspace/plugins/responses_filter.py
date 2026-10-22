"""
title: Responses API
description: Keep only OpenAI Responses fields in the Chat UI's requests to Kairyu.
version: 1.0.0
"""

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

# L4 (deploy/workspace) installs this one filter in every workspace. Open WebUI
# v0.12.3 reaches Kairyu over the Responses API (api_type "responses"): it
# converts its chat body with convert_to_responses_payload, which forwards
# unknown top-level keys verbatim and replays saved output items as they were
# stored. Kairyu's /v1/responses is the strict OpenAI contract, so the inlet
# leaves only Responses fields: the effort becomes reasoning.effort, sampling
# keys the Responses API lacks are dropped, saved output items keep their
# Responses fields, and store is false because the Chat UI sends the whole
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
# The effort levels this workspace offers (the model author's vocabulary,
# "default" first) come from the Chat UI's environment; none, no dropdown.
_EFFORT_LEVELS = tuple(json.loads(os.environ.get("KAIRYU_WORKSPACE_EFFORT_LEVELS") or "[]"))


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
    """Open WebUI global filter: the Chat UI's body as an OpenAI Responses body.

    With effort levels, Open WebUI renders the enum-typed user valve as a
    ``<select>`` in Chat Controls; the choice reaches Kairyu as
    ``reasoning.effort``, and ``default`` omits it so the model's own default
    applies (also over a ``reasoning_effort`` in the default model params).
    """

    class Valves(BaseModel):
        pass

    if _EFFORT_LEVELS:

        class UserValves(BaseModel):
            reasoning_effort: Literal[_EFFORT_LEVELS] = Field(
                default=_EFFORT_LEVELS[0],
                description=os.environ.get("KAIRYU_WORKSPACE_EFFORT_HELP", ""),
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
