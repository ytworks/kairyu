"""Remote compaction (Codex v2): the sealed summary codec and its output item.

A terminal ``compaction_trigger`` input item requests exactly one compaction
output item whose ``encrypted_content`` is opaque to the client and echoed back
verbatim on a later turn.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from types import MappingProxyType

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.responses.errors import BufferedFailure
from kairyu.entrypoints.server.responses.sealing import (
    SealedItemError,
    SealingKeyRing,
    SealPurpose,
)

_COMPACTION_INSTRUCTION_ITEM = MappingProxyType(
    {
        "type": "message",
        "role": "user",
        "content": (
            "Summarize the conversation above for a compacted continuation. "
            "Capture the goals, decisions, constraints, tool results, and any "
            "unfinished work so the conversation can continue from this summary "
            "alone."
        ),
    }
)


class CompactionCodec:
    """Tenant-bound sealed compaction summaries (``kst2`` purpose compaction).

    Compaction fails closed (m20 D6): a token the key ring cannot open is a
    400 ``invalid_encrypted_content``, never a silently dropped summary.
    """

    def __init__(self, keys: SealingKeyRing) -> None:
        self._keys = keys

    def encode(self, summary: str, *, owner: str) -> str:
        return self._keys.seal(SealPurpose.COMPACTION, {"summary": summary}, owner=owner)

    def decode(self, token: object, *, owner: str, param: str | None = None) -> str:
        try:
            item = self._keys.open(token, purpose=SealPurpose.COMPACTION, owner=owner)
        except SealedItemError as error:
            raise _refused(error.reason, param) from None
        summary = item.payload.get("summary")
        if not isinstance(summary, str):
            raise _refused("was not issued by this server", param)
        return summary


def _refused(reason: str, param: str | None) -> ChatRequestError:
    return ChatRequestError(
        f"compaction encrypted_content {reason}, so the compacted conversation "
        "cannot be restored; start a new session (in Codex: /new)",
        code="invalid_encrypted_content",
        param=param,
    )


def extract_compaction_trigger(items: Sequence[dict]) -> bool:
    for index, item in enumerate(items):
        if item["type"] == "compaction_trigger" and index != len(items) - 1:
            raise ChatRequestError("compaction_trigger must be the final input item")
    return bool(items) and items[-1]["type"] == "compaction_trigger"


def compaction_prompt_items(work_items: Sequence[dict]) -> list[dict]:
    """Return ``work_items`` followed by a fresh copy of the summary instruction."""
    return [*work_items, dict(_COMPACTION_INSTRUCTION_ITEM)]


def compaction_output_from_message(
    message: dict,
    *,
    compaction_codec: CompactionCodec,
    owner: str,
) -> list[dict]:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise BufferedFailure.from_payload(
            {
                "message": "upstream model returned an empty compaction summary",
                "type": "upstream_error",
                "code": "compaction_failed",
            },
            502,
        )
    return [
        {
            "type": "compaction",
            "id": f"cmp_{uuid.uuid4().hex[:24]}",
            "encrypted_content": compaction_codec.encode(content, owner=owner),
            "status": "completed",
        }
    ]
