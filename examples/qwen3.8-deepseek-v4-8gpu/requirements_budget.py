"""Example-local vLLM middleware: reserve output space for the extractor only."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import yaml

_LOGGER = logging.getLogger("vllm.entrypoints.openai.api_server")
_MAX_BUFFER_BYTES = 32 * 1024 * 1024


def budget_payload(payload: object, *, prefix: str, suffix: str, thinking_budget: int):
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
    return {**payload, "thinking_token_budget": limit}


class RequirementsBudgetMiddleware:
    """Standard vLLM --middleware hook; no change to the Kairyu framework."""

    def __init__(self, app, *, config_dir: Path | None = None):
        self.app = app
        directory = config_dir or Path(__file__).resolve().parent
        spec = yaml.safe_load((directory / "auto-max.yaml").read_text())
        roles = [role for role in spec["roles"] if role["name"] == "requirements"]
        if len(roles) != 1 or roles[0].get("reasoning_effort") != "high":
            raise ValueError("requirements middleware requires the fixed medium role")
        prompt = roles[0]["prompt"]
        if prompt.count("{query}") != 1:
            raise ValueError("requirements prompt must contain exactly one query slot")
        self.prefix, self.suffix = prompt.split("{query}")
        if not self.prefix or "[requirements]" not in self.suffix:
            raise ValueError("requirements prompt must have an unambiguous role suffix")
        metadata = json.loads((directory / "example.json").read_text())
        self.budget = metadata["orchestration"]["requirements_thinking_token_budget"]
        total = roles[0]["sampling"]["max_tokens"]
        if type(self.budget) is not int or not 0 < self.budget < total:
            raise ValueError("requirements thinking budget must leave completion tokens")

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
        try:
            changed = budget_payload(
                json.loads(body),
                prefix=self.prefix,
                suffix=self.suffix,
                thinking_budget=self.budget,
            )
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
                "Kairyu requirements budget: thinking=%d max_tokens=%d seed=%s messages_sha256=%s",
                changed["thinking_token_budget"],
                changed["max_tokens"],
                changed.get("seed"),
                hashlib.sha256(
                    json.dumps(changed["messages"], ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
            )
        await self.app(scope, replay, send)
