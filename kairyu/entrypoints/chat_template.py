"""Chat templating: HF Jinja templates + the legacy concatenator (m9 D2).

``ChatTemplate`` matches transformers' ``_compile_jinja_template`` exactly —
``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
extensions=[loopcontrols])`` with HF's ``tojson`` (``ensure_ascii=False``;
Jinja's builtin html-escapes ``<>&'``) and the ``raise_exception`` /
``strftime_now`` globals — anything less breaks byte-match with
``apply_chat_template``. Assistant ``tool_calls.arguments`` arriving as JSON
strings (the OpenAI wire form) are parsed to dicts before rendering (HF
convention; Qwen templates ``| tojson`` them).

``render_chat`` (the legacy role concatenator) stays the default when no
template is configured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


def render_chat(messages: Sequence[dict]) -> str:
    """Legacy minimal renderer: role-prefixed concatenation (pre-m9 default)."""
    lines = [
        f"{message['role']}: {flatten_content(message.get('content'))[0]}"
        for message in messages
    ]
    return "\n".join(lines) + "\nassistant:"


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _raise_exception(message: str) -> None:
    raise ValueError(f"chat template error: {message}")


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def flatten_content(content) -> tuple[str, bool]:
    """Content-parts -> (text, has_images) (m11 A10). Strings pass through."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    texts: list[str] = []
    has_images = False
    for part in content:
        data = part if isinstance(part, dict) else part.model_dump()
        if data.get("type") == "text":
            texts.append(data.get("text") or "")
        elif data.get("type") == "image_url":
            has_images = True
    return "".join(texts), has_images


def _normalize_message(message: Mapping[str, object]) -> dict:
    """OpenAI wire form -> template form: tool_calls.arguments str -> dict."""
    normalized = dict(message)
    content = normalized.get("content")
    if content is not None and not isinstance(content, str):
        # text-only HF templates must never see part lists (m11 A10)
        normalized["content"], _ = flatten_content(content)
    tool_calls = normalized.get("tool_calls")
    if isinstance(tool_calls, list):
        fixed_calls = []
        for call in tool_calls:
            call = dict(call) if isinstance(call, Mapping) else call
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, Mapping):
                function = dict(function)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass  # leave malformed arguments as-is; template decides
                call["function"] = function
            fixed_calls.append(call)
        normalized["tool_calls"] = fixed_calls
    return normalized


class ChatTemplate:
    """One compiled HF-compatible chat template (per served model, m9 D2)."""

    def __init__(self, source: str, special_tokens: Mapping[str, str] | None = None) -> None:
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        environment.filters["tojson"] = _tojson
        environment.globals["raise_exception"] = _raise_exception
        environment.globals["strftime_now"] = _strftime_now
        self._template = environment.from_string(source)
        self._special_tokens = dict(special_tokens or {})
        self._special_tokens.setdefault("bos_token", "")
        self._special_tokens.setdefault("eos_token", "")

    @classmethod
    def load(
        cls, source_or_path: str, special_tokens: Mapping[str, str] | None = None
    ) -> ChatTemplate:
        """Inline template text, or a ``*.jinja`` path (spec config surface)."""
        if source_or_path.endswith(".jinja"):
            path = Path(source_or_path)
            if not path.is_file():
                raise ValueError(f"chat template file not found: {source_or_path}")
            return cls(path.read_text(), special_tokens)
        return cls(source_or_path, special_tokens)

    def render(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        return self._template.render(
            messages=[_normalize_message(message) for message in messages],
            # None (not []) when absent: templates gate on `tools is not none`
            tools=list(tools) if tools else None,
            add_generation_prompt=add_generation_prompt,
            **self._special_tokens,
        )
