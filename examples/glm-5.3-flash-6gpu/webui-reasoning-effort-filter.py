"""
title: Reasoning Effort
description: Select GLM-5.3-Flash reasoning effort; default is the model's own (max).
version: 0.1.0
"""

from typing import Literal

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI global filter that turns the effort knob into a dropdown.

    Open WebUI v0.11.0 renders an enum-typed user valve as a ``<select>`` in
    Chat Controls. The choice is sent as the OpenAI ``reasoning_effort`` body
    field in the model author's vocabulary (low, high, max); ``default`` omits
    it so the chat template's default (max) applies. The model author asks
    chat clients to pass ``clear_thinking=true``: earlier turns' reasoning is
    dropped from the prompt, the current turn's is kept.
    """

    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        reasoning_effort: Literal["default", "low", "high", "max"] = Field(
            default="default",
            description="Reasoning effort for GLM-5.3-Flash. default = max (the model's default).",
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
        # The effort is the public top-level field; the only template switch
        # the chat surface sends is the model author's chat setting.
        body["chat_template_kwargs"] = {"clear_thinking": True}
        return body
