"""Example-local vLLM hook: evidence-root budgets and the audit wire format."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from string import Formatter

import yaml

_LOGGER = logging.getLogger("vllm.entrypoints.openai.api_server")
_MAX_BUFFER_BYTES = 32 * 1024 * 1024

_AUDIT_ROW = (
    r"R[1-9][0-9]* \| (satisfied|unsatisfied|unverifiable|unsupported)"
    r" \| evidence: [^\n]+ \| correction: [^\n]+"
)
AUDIT_REGEX = r"(PASS|FAIL)\n" + _AUDIT_ROW + r"(\n" + _AUDIT_ROW + r")*"
AUDIT_RETRY_SUFFIX = (
    "\nYour previous verification ran out of output budget "
    "before emitting a verdict. Answer immediately: PASS or "
    "FAIL on the first line, then at most three short bullet points."
)


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


def _payload_content(payload: object) -> str | None:
    if not isinstance(payload, dict) or payload.get("model") not in {
        "qwen3.8-27b",
        "deepseek-v4.1-flash",
    }:
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
    return content if isinstance(content, str) else None


def template_pattern(prompt: str, *, retry: bool = False) -> re.Pattern:
    """Match every literal of the shipped template, including its closing text."""
    parts = []
    for literal, field, format_spec, conversion in Formatter().parse(prompt):
        parts.append(re.escape(literal))
        if field is not None:
            if format_spec or conversion:
                raise ValueError("role templates cannot format or convert input slots")
            parts.append(r"[\s\S]*?")
    if retry:
        parts.append("(?:" + re.escape(AUDIT_RETRY_SUFFIX) + ")?")
    return re.compile("".join(parts))


class RequirementsBudgetMiddleware:
    """Standard vLLM --middleware hook; no change to the Kairyu framework."""

    def __init__(self, app, *, config_dir: Path | None = None):
        self.app = app
        directory = config_dir or Path(__file__).resolve().parent
        spec = yaml.safe_load((directory / "auto-max.yaml").read_text())
        roles = spec["roles"] + [r for p in spec["profiles"] for r in p["roles"]]
        requirements = next(r for r in roles if r["name"] == "requirements")
        audit = next(r for r in roles if r["name"] == "audit")
        if requirements["worker"] != "tier2" or requirements["reasoning_effort"] != "high":
            raise ValueError("requirements must run on DeepSeek at fixed native high")
        if audit["worker"] != "tier1" or audit["reasoning_effort"] != "high":
            raise ValueError("audit must run on Qwen at fixed medium (spec high)")
        self.policies = []
        for role in roles:
            if role["worker"].startswith("tier2") or role["name"] == "audit":
                for key in ("prompt", "prompt_headless"):
                    if key in role:
                        self.policies.append(
                            (
                                role["name"],
                                template_pattern(role[key], retry=role["name"] == "audit"),
                            )
                        )

    def transform(self, payload):
        content = _payload_content(payload)
        if content is None:
            return None, None
        for name, pattern in self.policies:
            expected = "qwen3.8-27b" if name == "audit" else "deepseek-v4.1-flash"
            if payload["model"] != expected:
                continue
            # Synthesis repair appends verifier feedback. Require the entire
            # base template first; a lone role marker is never sufficient.
            matched = pattern.fullmatch(content)
            if matched is None and name == "synthesis":
                repair = (
                    pattern.pattern
                    + r"\n\nPrevious attempt:\n[\s\S]*?"
                    + r"\n\nVerifier feedback:\n[\s\S]*?"
                    + r"\n\nRevise the answer addressing the feedback\."
                )
                matched = re.fullmatch(repair, content)
            if matched is None:
                continue
            if name == "audit":
                return {**payload, "structured_outputs": {"regex": AUDIT_REGEX}}, name
            kwargs = payload.get("chat_template_kwargs") or {}
            changed = {
                **payload,
                "chat_template_kwargs": {
                    **kwargs,
                    "thinking": name != "deepseek_answer",
                    "enable_thinking": name != "deepseek_answer",
                },
            }
            if payload.get("reasoning_effort") in ("low", "high", "max"):
                changed["chat_template_kwargs"]["reasoning_effort"] = payload["reasoning_effort"]
            if name == "requirements":
                changed["reasoning_effort"] = "high"
                changed["chat_template_kwargs"]["reasoning_effort"] = "high"
                maximum = payload.get("max_tokens")
                if type(maximum) is int and maximum >= 2:
                    changed["thinking_token_budget"] = min(4096, maximum // 2)
                changed["structured_outputs"] = {"json": CHECKLIST_SCHEMA}
            if name in {"synthesis", "deepseek_think_answer"}:
                maximum = payload.get("max_tokens")
                if type(maximum) is int and maximum >= 2:
                    changed["thinking_token_budget"] = maximum - min(256, maximum // 2)
            # Pinned v4.1 engine parser exposes <think>/</think> boundary IDs;
            # vLLM's thinking budget processor can reserve checklist body tokens.
            # Native high still selects official high75, independent of the cap.
            return changed, name
        return None, None

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
            changed, matched_role = self.transform(payload)
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
            messages_hash = hashlib.sha256(
                json.dumps(changed["messages"], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            _LOGGER.info(
                "Kairyu role hook: role=%s max_tokens=%s effort=%s messages_sha256=%s",
                matched_role,
                changed.get("max_tokens"),
                changed.get("reasoning_effort"),
                messages_hash,
            )
        await self.app(scope, replay, send)
