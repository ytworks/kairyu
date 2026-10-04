"""Canonical Responses input items: validation and normalization of ``input``."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Sequence

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.responses.compaction import CompactionCodec

_SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}
_CODEX_INTERNAL_ITEM_FIELDS = {
    "internal_chat_message_metadata_passthrough",
    "encrypted_function_args",
}


def canonical_input(
    payload: str | list[dict],
    *,
    compaction_codec: CompactionCodec,
    owner: str,
) -> list[dict]:
    if isinstance(payload, str):
        return [{"type": "message", "role": "user", "content": payload}]
    if not isinstance(payload, list):
        raise ChatRequestError("input must be a string or an array of input items")
    items: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ChatRequestError(f"input[{index}] must be an object")
        kind = item.get("type")
        if kind is None and "role" in item:
            kind = "message"
        # Codex-internal passthrough metadata rides input items verbatim when
        # Codex addresses an OpenAI-shaped base URL (the Harbor/Terminal-Bench
        # setup); it scrubs them for custom providers. Accepted and dropped on
        # every item kind.
        item = {
            key: value
            for key, value in item.items()
            if key not in _CODEX_INTERNAL_ITEM_FIELDS
        }
        if kind == "message":
            unknown = set(item) - {
                "type",
                "role",
                "content",
                "status",
                "id",
                "phase",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            role = item.get("role")
            if role not in _SUPPORTED_ROLES:
                raise ChatRequestError(
                    f"input[{index}].role must be user, assistant, system, or developer"
                )
            content = item.get("content", "")
            content_text(content, path=f"input[{index}].content")
            phase = item.get("phase")
            if phase not in (None, "commentary", "final_answer"):
                raise ChatRequestError(
                    f"input[{index}].phase must be commentary or final_answer"
                )
            canonical = {
                "type": "message",
                "role": role,
                "content": copy.deepcopy(content),
            }
            if phase is not None:
                canonical["phase"] = phase
            items.append(canonical)
            continue
        if kind == "function_call":
            unknown = set(item) - {
                "type",
                "id",
                "call_id",
                "name",
                "namespace",
                "arguments",
                "status",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            namespace = item.get("namespace")
            if not isinstance(name, str) or not name:
                raise ChatRequestError(f"input[{index}].name must be a non-empty string")
            if not isinstance(call_id, str) or not call_id:
                raise ChatRequestError(f"input[{index}].call_id must be a non-empty string")
            if not isinstance(arguments, str):
                raise ChatRequestError(f"input[{index}].arguments must be a JSON string")
            if namespace is not None and (
                not isinstance(namespace, str) or not namespace
            ):
                raise ChatRequestError(
                    f"input[{index}].namespace must be a non-empty string"
                )
            items.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": call_id,
                    "name": name,
                    "namespace": namespace,
                    "arguments": arguments,
                    "status": item.get("status") or "completed",
                }
            )
            continue
        if kind == "function_call_output":
            unknown = set(item) - {"type", "id", "call_id", "output", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise ChatRequestError(f"input[{index}].call_id must be a non-empty string")
            if isinstance(output, list):
                # Codex sends structured tool output (e.g. view_image) as an
                # array of content items instead of a plain string.
                output = _function_output_text(
                    output, path=f"input[{index}].output"
                )
            elif not isinstance(output, str):
                raise ChatRequestError(
                    f"input[{index}].output must be a string or an array of output parts"
                )
            items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )
            continue
        if kind == "reasoning":
            # Codex echoes prior reasoning items back with the history
            # (encrypted_content may be null or foreign). Kairyu emits no
            # reasoning output item and renders none into the prompt, so the
            # echo is accepted for wire compatibility and dropped.
            unknown = set(item) - {
                "type",
                "id",
                "summary",
                "content",
                "encrypted_content",
                "status",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            continue
        if kind in {"compaction", "compaction_summary"}:
            unknown = set(item) - {"type", "id", "encrypted_content", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            # Fail-fast on foreign or cross-tenant tokens; render re-decodes it.
            compaction_codec.decode(
                item.get("encrypted_content"),
                owner=owner,
                param=f"input[{index}].encrypted_content",
            )
            items.append(
                {
                    "type": "compaction",
                    "encrypted_content": item["encrypted_content"],
                }
            )
            continue
        if kind == "compaction_trigger":
            unknown = set(item) - {"type", "id", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            items.append({"type": "compaction_trigger"})
            continue
        raise ChatRequestError(
            f"input[{index}].type {kind!r} is not supported; "
            "use message, function_call, or function_call_output"
        )
    return items


def _function_output_text(parts: list, *, path: str) -> str:
    texts: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ChatRequestError(f"{path}[{index}] must be an object")
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ChatRequestError(f"{path}[{index}].text must be a string")
            texts.append(text)
            continue
        raise ChatRequestError(
            f"{path}[{index}].type {kind!r} is not supported; "
            "this model accepts text tool output only"
        )
    return "".join(texts)


def content_text(content: object, *, path: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ChatRequestError(f"{path} must be a string or an array of text parts")
    parts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ChatRequestError(f"{path}[{index}] must be an object")
        kind = part.get("type")
        if kind not in {"input_text", "output_text", "text"}:
            raise ChatRequestError(f"{path}[{index}].type {kind!r} is not supported")
        allowed = (
            {"type", "text", "annotations", "logprobs"}
            if kind == "output_text"
            else {"type", "text"}
        )
        unknown = set(part) - allowed
        if unknown:
            raise ChatRequestError(
                f"{path}[{index}] has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ChatRequestError(f"{path}[{index}].text must be a string")
        parts.append(text)
    return "".join(parts)


def validate_function_outputs(items: Sequence[dict]) -> None:
    pending: set[str] = set()
    consumed: set[str] = set()
    for item in items:
        if item["type"] == "function_call":
            call_id = item["call_id"]
            if call_id in pending:
                raise ChatRequestError(f"duplicate function call_id {call_id!r}")
            pending.add(call_id)
        elif item["type"] == "function_call_output":
            call_id = item["call_id"]
            if call_id not in pending:
                raise ChatRequestError(
                    f"input function_call_output references unknown call_id {call_id!r}"
                )
            if call_id in consumed:
                raise ChatRequestError(
                    f"input function_call_output repeats call_id {call_id!r}"
                )
            consumed.add(call_id)
