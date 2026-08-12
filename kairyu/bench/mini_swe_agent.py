"""Small mini-swe-agent compatibility shims loaded only by its subprocess."""

from __future__ import annotations

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

_OPENAI_CHAT_MESSAGE_FIELDS = frozenset(
    {"role", "content", "name", "tool_call_id", "tool_calls"}
)


class OpenAICompatTextbasedModel(LitellmTextbasedModel):
    """Keep LiteLLM response-only extensions out of later chat requests."""

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        return [
            {
                key: value
                for key, value in message.items()
                if key in _OPENAI_CHAT_MESSAGE_FIELDS
            }
            for message in prepared
        ]


__all__ = ["OpenAICompatTextbasedModel"]
