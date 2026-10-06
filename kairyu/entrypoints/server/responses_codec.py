"""Tenant-bound sealed tokens and remote-compaction helpers for Responses."""

from __future__ import annotations

import base64
import binascii
import os
import uuid
from collections.abc import Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.responses_protocol import ResponsesError, _BufferedFailure

# Remote compaction v2 (Codex against an OpenAI-shaped provider): a terminal
# compaction_trigger input item requests exactly one compaction output item
# whose encrypted_content is opaque to the client and echoed back verbatim.
_COMPACTION_TOKEN_PREFIX = "kcp1."
_COMPACTION_AAD_PREFIX = b"kairyu.responses.compaction.v1\0"
_COMPACTION_NONCE_BYTES = 12
_REASONING_TOKEN_PREFIX = "krs1."
_REASONING_AAD_PREFIX = b"kairyu.responses.reasoning.v1\0"
_COMPACTION_INSTRUCTION_ITEM = {
    "type": "message",
    "role": "user",
    "content": (
        "Summarize the conversation above for a compacted continuation. "
        "Capture the goals, decisions, constraints, tool results, and any "
        "unfinished work so the conversation can continue from this summary "
        "alone."
    ),
}


class _CompactionCodec:
    """Tenant-bound authenticated encryption for opaque Responses state.

    One deployment key seals two token kinds whose prefixes and associated
    data differ, so neither can be opened as the other: remote-compaction
    summaries (``kcp1.``) and reasoning carried by stateless clients through
    ``include: ["reasoning.encrypted_content"]`` (``krs1.``).
    """

    def __init__(
        self,
        key: bytes,
        *,
        kind: str = "compaction",
        token_prefix: str = _COMPACTION_TOKEN_PREFIX,
        aad_prefix: bytes = _COMPACTION_AAD_PREFIX,
    ) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("Responses compaction key must be exactly 32 bytes")
        self._cipher = AESGCM(key)
        self._kind = kind
        self._token_prefix = token_prefix
        self._aad_prefix = aad_prefix

    def _associated_data(self, owner: str) -> bytes:
        return self._aad_prefix + owner.encode("utf-8")

    def issued(self, token: object) -> bool:
        return isinstance(token, str) and token.startswith(self._token_prefix)

    def encode(self, summary: str, *, owner: str) -> str:
        nonce = os.urandom(_COMPACTION_NONCE_BYTES)
        sealed = self._cipher.encrypt(
            nonce,
            summary.encode("utf-8"),
            self._associated_data(owner),
        )
        encoded = base64.urlsafe_b64encode(nonce + sealed).rstrip(b"=")
        return self._token_prefix + encoded.decode("ascii")

    def decode(self, token: object, *, owner: str) -> str:
        if not isinstance(token, str) or not token:
            raise ChatRequestError(
                f"{self._kind} encrypted_content must be a non-empty string"
            )
        if not token.startswith(self._token_prefix):
            raise ChatRequestError(
                f"{self._kind} encrypted_content was not issued by this server"
            )
        encoded = token.removeprefix(self._token_prefix)
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = base64.b64decode(padded, altchars=b"-_", validate=True)
            if len(payload) < _COMPACTION_NONCE_BYTES + 16:
                raise ValueError("sealed token is too short")
            plaintext = self._cipher.decrypt(
                payload[:_COMPACTION_NONCE_BYTES],
                payload[_COMPACTION_NONCE_BYTES:],
                self._associated_data(owner),
            )
            return plaintext.decode("utf-8")
        except (binascii.Error, InvalidTag, UnicodeDecodeError, ValueError):
            raise ChatRequestError(
                f"{self._kind} encrypted_content was not issued by this server"
            ) from None


def reasoning_codec(key: bytes) -> _CompactionCodec:
    return _CompactionCodec(
        key,
        kind="reasoning",
        token_prefix=_REASONING_TOKEN_PREFIX,
        aad_prefix=_REASONING_AAD_PREFIX,
    )


def _extract_compaction_trigger(items: Sequence[dict]) -> bool:
    for index, item in enumerate(items):
        if item["type"] == "compaction_trigger" and index != len(items) - 1:
            raise ResponsesError(
                "compaction_trigger must be the final input item",
                param=f"input[{index}]",
                code="invalid_value",
            )
    return bool(items) and items[-1]["type"] == "compaction_trigger"


def _compaction_output_from_message(
    message: dict,
    *,
    compaction_codec: _CompactionCodec,
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
        raise _BufferedFailure(
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
