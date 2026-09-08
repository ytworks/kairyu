"""Example-local vLLM middleware: reserve output space for the two evidence roots."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import yaml

_LOGGER = logging.getLogger("vllm.entrypoints.openai.api_server")
_MAX_BUFFER_BYTES = 32 * 1024 * 1024


CHECKLIST_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^R[1-9][0-9]*$"},
            "priority": {"type": "string", "enum": ["minimum", "optional"]},
            **{
                name: {"type": "string", "minLength": 1}
                for name in ("requirement", "acceptance_criterion", "source")
            },
        },
        "required": ["id", "priority", "requirement", "acceptance_criterion", "source"],
        "additionalProperties": False,
    },
}


def budget_payload(
    payload: object,
    *,
    prefix: str,
    suffix: str,
    thinking_budget: int,
    output_schema: dict | None = CHECKLIST_SCHEMA,
):
    """Recognize the entire shipped role template, never a marker in user data."""
    if not isinstance(payload, dict) or payload.get("model") != "qwen3.8-27b":
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return None
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, list):
        if not all(
            isinstance(part, dict) and part.get("type") in ("text", "image_url") for part in content
        ):
            return None
        texts = [part.get("text") for part in content if part["type"] == "text"]
        if len(texts) != 1:
            return None
        content = texts[0]
    if (
        not isinstance(content, str)
        or not content.startswith(prefix)
        or not content.endswith(suffix)
    ):
        return None
    maximum = payload.get("max_tokens")
    if type(maximum) is not int or maximum < 2:
        return None
    # Short caller allowances still retain at least half for the body. This
    # does not enlarge the caller's or the role's total token allowance.
    limit = min(thinking_budget, maximum // 2)
    changed = {**payload, "thinking_token_budget": limit}
    if output_schema is not None:
        changed["structured_outputs"] = {"json": output_schema}
    return changed


class RequirementsBudgetMiddleware:
    """Standard vLLM --middleware hook; no change to the Kairyu framework."""

    def __init__(self, app, *, config_dir: Path | None = None):
        self.app = app
        directory = config_dir or Path(__file__).resolve().parent
        spec = yaml.safe_load((directory / "auto-max.yaml").read_text())
        metadata = json.loads((directory / "example.json").read_text())
        self.policies = []
        for name, schema in (("requirements", CHECKLIST_SCHEMA), ("image_description", None)):
            roles = [role for role in spec["roles"] if role["name"] == name]
            if len(roles) != 1 or roles[0].get("reasoning_effort") != "high":
                raise ValueError(f"{name} middleware requires the fixed medium role")
            prompt = roles[0]["prompt"]
            if prompt.count("{query}") != 1:
                raise ValueError(f"{name} prompt must contain exactly one query slot")
            prefix, suffix = prompt.split("{query}")
            if not prefix or f"[{name}]" not in suffix:
                raise ValueError(f"{name} prompt must have an unambiguous role suffix")
            budget = metadata["orchestration"][f"{name}_thinking_token_budget"]
            total = roles[0]["sampling"]["max_tokens"]
            if type(budget) is not int or not 0 < budget < total:
                raise ValueError(f"{name} thinking budget must leave completion tokens")
            self.policies.append(
                (
                    name,
                    {
                        "prefix": prefix,
                        "suffix": suffix,
                        "thinking_budget": budget,
                        "output_schema": schema,
                    },
                )
            )

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return
        messages = []
        size = 0

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                await self.app(scope, replay, send)
                return
            size += len(message.get("body", b""))
            if size > _MAX_BUFFER_BYTES:
                await self.app(scope, replay, send)
                return
            if not message.get("more_body", False):
                break
        body = b"".join(message.get("body", b"") for message in messages)
        changed = None
        matched_role = None
        try:
            payload = json.loads(body)
            for name, policy in self.policies:
                changed = budget_payload(payload, **policy)
                if changed is not None:
                    matched_role = name
                    break
        except (TypeError, ValueError):
            changed = None
        if changed is not None:
            body = json.dumps(changed, ensure_ascii=False, separators=(",", ":")).encode()
            messages[:] = [{"type": "http.request", "body": body, "more_body": False}]
            headers = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() not in (b"content-length", b"transfer-encoding")
            ]
            scope = {**scope, "headers": [*headers, (b"content-length", str(len(body)).encode())]}
            _LOGGER.info(
                "Kairyu %s budget: thinking=%d max_tokens=%d seed=%s messages_sha256=%s",
                matched_role,
                changed["thinking_token_budget"],
                changed["max_tokens"],
                changed.get("seed"),
                hashlib.sha256(
                    json.dumps(changed["messages"], ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
            )
        await self.app(scope, replay, send)
