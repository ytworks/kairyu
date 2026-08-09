"""Chat templating: HF Jinja templates + an explicit legacy renderer (m9 D2).

``ChatTemplate`` matches transformers' ``_compile_jinja_template`` exactly —
``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
extensions=[AssistantTracker, loopcontrols])`` with HF's ``tojson`` (``ensure_ascii=False``;
Jinja's builtin html-escapes ``<>&'``) and the ``raise_exception`` /
``strftime_now`` globals — anything less breaks byte-match with
``apply_chat_template``. Assistant ``tool_calls.arguments`` arriving as JSON
strings (the OpenAI wire form) are parsed to dicts before rendering (HF
convention; Qwen templates ``| tojson`` them).

``render_chat`` remains available for explicitly selected compatibility paths;
production model templates are represented by ``ChatTemplate``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from kairyu.sampling_params import PROMPT_CARRIER_EXTRA_ARGS

_HF_TEMPLATE_CONTROL_VARIABLES = frozenset(
    {
        "messages",
        "tools",
        "documents",
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "return_assistant_tokens_mask",
        "conversations",
    }
)
_HF_STANDARD_SPECIAL_TOKEN_VARIABLES = frozenset(
    {
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    }
)
_LLAMA_PROTOCOL_TEMPLATE_KWARGS = frozenset(
    {"builtin_tools", "custom_tools", "tools_in_user_message"}
)


class ToolCallProtocol(StrEnum):
    """Model-native tool syntax attested by a server-owned chat template."""

    GENERIC = "generic"
    LLAMA = "llama"
    QWEN = "qwen"
    DEEPSEEK_V4 = "deepseek_v4"


def _tool_call_protocol(source: str | Mapping[str, str]) -> ToolCallProtocol:
    """Infer only syntax explicitly declared by the configured template text.

    The served model name is deliberately absent from this decision.  Aliases
    of one configured template therefore retain its protocol, while a client
    cannot opt into a parser by choosing a suggestive model identifier.
    """

    sources = (source,) if isinstance(source, str) else tuple(source.values())
    sources = tuple(re.sub(r"{#.*?#}", "", template, flags=re.DOTALL) for template in sources)
    has_llama = any("<|python_tag|>" in template for template in sources)
    has_qwen = any(
        "<function=" in template and "<parameter=" in template and "</function>" in template
        for template in sources
    )
    has_deepseek_v4 = any("｜DSML｜tool_calls" in template for template in sources)
    if sum((has_llama, has_qwen, has_deepseek_v4)) > 1:
        return ToolCallProtocol.GENERIC
    if has_llama:
        return ToolCallProtocol.LLAMA
    if has_qwen:
        return ToolCallProtocol.QWEN
    if has_deepseek_v4:
        return ToolCallProtocol.DEEPSEEK_V4
    return ToolCallProtocol.GENERIC


def _reject_prompt_carriers(
    message: Mapping[str, object],
    *,
    index: int,
) -> None:
    carriers = PROMPT_CARRIER_EXTRA_ARGS.intersection(message)
    if carriers:
        raise ValueError(
            f"message {index} contains alternate prompt carriers "
            f"{sorted(carriers)}; pass input through PromptInput"
        )


def render_chat(messages: Sequence[dict]) -> str:
    """Legacy minimal renderer: role-prefixed concatenation (pre-m9 default)."""
    lines = []
    for index, message in enumerate(messages):
        _reject_prompt_carriers(message, index=index)
        text, has_images = flatten_content(message.get("content"))
        if has_images:
            raise ValueError(
                f"message {index} contains image input that a text chat "
                "renderer cannot execute; use MultimodalPrompt"
            )
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            rendered_calls = []
            for call in tool_calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping):
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                rendered_calls.append(
                    "<tool_call>"
                    + json.dumps(
                        {
                            "name": function.get("name"),
                            "arguments": arguments,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "</tool_call>"
                )
            text += "".join(rendered_calls)
        lines.append(f"{message['role']}: {text}")
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


def _iter_validated_content_parts(content):
    """Yield normalized tagged parts after applying the shared strict checks."""

    allowed_fields = {"type", "text", "image_url"}
    for index, part in enumerate(content):
        if isinstance(part, Mapping):
            data = dict(part)
        else:
            dump = getattr(part, "model_dump", None)
            if not callable(dump):
                raise ValueError(
                    f"content part {index} must be a tagged object, "
                    f"got {type(part).__name__}"
                )
            data = dump()
        extra = set(data) - allowed_fields
        if extra:
            raise ValueError(
                f"content part {index} has unsupported fields: {sorted(extra)}"
            )
        kind = data.get("type")
        if kind == "text":
            text = data.get("text")
            if type(text) is not str or data.get("image_url") is not None:
                raise ValueError(
                    f"content part {index} type 'text' requires only a string text payload"
                )
            yield "text", text, None
        elif kind == "image_url":
            image_url = data.get("image_url")
            if (
                not isinstance(image_url, Mapping)
                or type(image_url.get("url")) is not str
                or not image_url["url"]
                or data.get("text") is not None
            ):
                raise ValueError(
                    f"content part {index} type 'image_url' requires only a "
                    "non-empty URL payload"
                )
            yield "image_url", None, image_url
        else:
            raise ValueError(f"content part {index} has unsupported type {kind!r}")


def flatten_content(content) -> tuple[str, bool]:
    """Content-parts -> (text, has_images) without dropping unknown payloads."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    texts: list[str] = []
    has_images = False
    for kind, text, _image_url in _iter_validated_content_parts(content):
        if kind == "text":
            assert text is not None
            texts.append(text)
        else:
            has_images = True
    return "".join(texts), has_images


def _normalize_message(message: Mapping[str, object], *, index: int) -> dict:
    """OpenAI wire form -> template form: tool_calls.arguments str -> dict."""
    _reject_prompt_carriers(message, index=index)
    normalized = dict(message)
    content = normalized.get("content")
    if content is not None and not isinstance(content, str):
        # text-only HF templates must never see part lists (m11 A10)
        normalized["content"], has_images = flatten_content(content)
        if has_images:
            raise ValueError(
                f"message {index} contains image input that a text chat "
                "template cannot execute; use MultimodalPrompt"
            )
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
    """Compiled HF-compatible chat template(s) for one served model (m9 D2)."""

    def __init__(
        self,
        source: str | Mapping[str, str],
        special_tokens: Mapping[str, str] | None = None,
        *,
        _tokenizer_metadata_authoritative: bool = False,
    ) -> None:
        from contextlib import contextmanager

        import jinja2
        import jinja2.ext
        import jinja2.meta
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        class AssistantTracker(Extension):
            """HF's ``generation`` tag; Kairyu renders without mask tracking."""

            tags = {"generation"}

            def __init__(self, environment):
                super().__init__(environment)
                environment.extend(activate_tracker=self.activate_tracker)
                self._rendered_blocks = None
                self._generation_indices = None

            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(
                    ["name:endgeneration"],
                    drop_needle=True,
                )
                return jinja2.nodes.CallBlock(
                    self.call_method("_generation_support"),
                    [],
                    [],
                    body,
                ).set_lineno(lineno)

            @jinja2.pass_eval_context
            def _generation_support(self, _context, caller):
                rendered = caller()
                if self._rendered_blocks is not None:
                    start = len("".join(self._rendered_blocks))
                    self._generation_indices.append((start, start + len(rendered)))
                return rendered

            @contextmanager
            def activate_tracker(self, rendered_blocks, generation_indices):
                if self._rendered_blocks is not None:
                    raise ValueError("AssistantTracker is already active")
                self._rendered_blocks = rendered_blocks
                self._generation_indices = generation_indices
                try:
                    yield
                finally:
                    self._rendered_blocks = None
                    self._generation_indices = None

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[AssistantTracker, jinja2.ext.loopcontrols],
        )
        environment.filters["tojson"] = _tojson
        environment.globals["raise_exception"] = _raise_exception
        environment.globals["strftime_now"] = _strftime_now
        referenced_variables: set[str] = set()

        def compile_source(template_source: str):
            parsed = environment.parse(template_source)
            referenced_variables.update(
                jinja2.meta.find_undeclared_variables(parsed)
            )
            return environment.from_string(template_source)

        if isinstance(source, str):
            self._tool_call_protocol = _tool_call_protocol(source)
            self._tool_call_protocols = None
            if not source.strip():
                raise ValueError("chat template source must be non-empty")
            self._template = compile_source(source)
            self._templates = None
        elif isinstance(source, Mapping):
            if not source:
                raise ValueError("chat template mapping must not be empty")
            invalid_names = [name for name in source if not isinstance(name, str)]
            if invalid_names:
                raise TypeError("chat template names must be strings")
            empty_names = [name for name in source if not name.strip()]
            if empty_names:
                raise ValueError("chat template names must be non-empty")
            invalid_sources = [
                name for name, template_source in source.items()
                if not isinstance(template_source, str)
            ]
            if invalid_sources:
                raise TypeError(
                    "chat template sources must be strings; invalid templates: "
                    f"{sorted(invalid_sources)}"
                )
            empty_sources = [
                name for name, template_source in source.items() if not template_source.strip()
            ]
            if empty_sources:
                raise ValueError(
                    "chat template sources must be non-empty; invalid templates: "
                    f"{sorted(empty_sources)}"
                )
            self._tool_call_protocol = None
            self._tool_call_protocols = {
                name: _tool_call_protocol(template_source)
                for name, template_source in source.items()
            }
            self._template = None
            self._templates = {
                name: compile_source(template_source)
                for name, template_source in source.items()
            }
        else:
            raise TypeError("chat template source must be a string or mapping")
        self._special_tokens = dict(special_tokens or {})
        if any(not isinstance(name, str) for name in self._special_tokens):
            raise TypeError("special-token names must be strings")
        invalid_token_values = sorted(
            name
            for name, value in self._special_tokens.items()
            if not isinstance(value, str)
        )
        if invalid_token_values:
            raise TypeError(
                "special-token values must be strings; invalid tokens: "
                f"{invalid_token_values}"
            )
        reserved_tokens = self._special_tokens.keys() & _HF_TEMPLATE_CONTROL_VARIABLES
        if reserved_tokens:
            raise ValueError(
                "special-token metadata conflicts with HF chat-template control "
                f"variables: {sorted(reserved_tokens)}"
            )
        self._referenced_special_token_variables = frozenset(
            name
            for name in referenced_variables
            if name in _HF_STANDARD_SPECIAL_TOKEN_VARIABLES
            or name.endswith("_token")
        )
        self._tokenizer_metadata_authoritative = (
            _tokenizer_metadata_authoritative
        )

    @classmethod
    def from_tokenizer_metadata(
        cls,
        source: str | Mapping[str, str],
        special_tokens: Mapping[str, str],
    ) -> ChatTemplate:
        """Compile a checkpoint-owned template with authoritative token metadata.

        An absent token variable then has the same undefined semantics as
        Transformers. Explicit templates must instead resolve every referenced
        token variable so missing metadata cannot silently become empty text.
        """

        return cls(
            source,
            special_tokens,
            _tokenizer_metadata_authoritative=True,
        )

    @property
    def referenced_special_token_variables(self) -> frozenset[str]:
        """Tokenizer-owned variables referenced by any compiled template."""

        return self._referenced_special_token_variables

    @property
    def tool_call_protocol(self) -> ToolCallProtocol:
        """Protocol for a single template, or generic for a named collection."""

        return self._tool_call_protocol or ToolCallProtocol.GENERIC

    def tool_call_protocol_for_tools(self, *, has_tools: bool) -> ToolCallProtocol:
        """Protocol of the named template that this request will execute."""

        if self._tool_call_protocol is not None:
            return self._tool_call_protocol
        assert self._tool_call_protocols is not None
        if has_tools and "tool_use" in self._tool_call_protocols:
            return self._tool_call_protocols["tool_use"]
        if "default" in self._tool_call_protocols:
            return self._tool_call_protocols["default"]
        available = ", ".join(sorted(self._tool_call_protocols))
        raise ValueError(
            "chat template mapping has no 'default' template for this request; "
            f"available templates: {available}"
        )

    @property
    def unresolved_special_token_variables(self) -> frozenset[str]:
        """Referenced tokenizer variables absent from the supplied metadata."""

        return self._referenced_special_token_variables - frozenset(
            self._special_tokens
        )

    @property
    def unverified_special_token_variables(self) -> frozenset[str]:
        """Unresolved variables lacking checkpoint-authoritative provenance."""

        if self._tokenizer_metadata_authoritative:
            return frozenset()
        return self.unresolved_special_token_variables

    @classmethod
    def load(
        cls,
        source_or_path: str | Mapping[str, str],
        special_tokens: Mapping[str, str] | None = None,
    ) -> ChatTemplate:
        """Load inline/path template text, either singly or in a named mapping."""
        def load_source(source: str) -> str:
            if not isinstance(source, str):
                raise TypeError("chat template source must be a string")
            if source.endswith(".jinja"):
                path = Path(source)
                if not path.is_file():
                    raise ValueError(f"chat template file not found: {source}")
                return path.read_text(encoding="utf-8")
            return source

        if isinstance(source_or_path, Mapping):
            return cls(
                {name: load_source(source) for name, source in source_or_path.items()},
                special_tokens,
            )
        return cls(load_source(source_or_path), special_tokens)

    def _select_template(self, *, has_tools: bool):
        if self._template is not None:
            return self._template
        assert self._templates is not None
        if has_tools and "tool_use" in self._templates:
            return self._templates["tool_use"]
        if "default" in self._templates:
            return self._templates["default"]
        available = ", ".join(sorted(self._templates))
        raise ValueError(
            "chat template mapping has no 'default' template for this request; "
            f"available templates: {available}"
        )

    def render(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        add_generation_prompt: bool = True,
        template_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        normalized_tools = None if tools is None else list(tools)
        trusted = {
            **self._special_tokens,
            "messages": [
                _normalize_message(message, index=index)
                for index, message in enumerate(messages)
            ],
            # None (not []) when absent: templates gate on `tools is not none`
            "tools": normalized_tools,
            # Kairyu has no documents request surface yet, but HF always
            # supplies this control variable as None when it is omitted.
            "documents": None,
            "add_generation_prompt": add_generation_prompt,
        }
        custom = dict(template_kwargs or {})
        protocol_reserved = (
            _LLAMA_PROTOCOL_TEMPLATE_KWARGS
            if self.tool_call_protocol_for_tools(has_tools=normalized_tools is not None)
            is ToolCallProtocol.LLAMA
            else frozenset()
        )
        reserved = custom.keys() & (
            trusted.keys()
            | _HF_TEMPLATE_CONTROL_VARIABLES
            | _HF_STANDARD_SPECIAL_TOKEN_VARIABLES
            | self._referenced_special_token_variables
            | protocol_reserved
        )
        if reserved:
            raise ValueError(
                "chat_template_kwargs contains reserved template variables: "
                f"{sorted(reserved)}"
            )
        template = self._select_template(has_tools=normalized_tools is not None)
        return template.render(**trusted, **custom)
