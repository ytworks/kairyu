"""Tokenizer seam: protocol, toy default, HF impl, incremental detokenizer (m8 D1).

``ToyTokenizer`` preserves the pre-M8 word-hash behavior (readable ``tok<N>``
renderings, clearly not model output) and stays the default so every existing
test and example runs unchanged. ``HFTokenizer`` wraps the ``tokenizers``
library behind a deferred import (same pattern as structured.py). The
``IncrementalDetokenizer`` emits only text that can no longer change — it holds
back incomplete UTF-8 sequences so an SSE stream never shows U+FFFD.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

_TOY_VOCAB = 50_000
_REPLACEMENT_CHAR = "�"
# eos candidates probed when tokenizer_config.json is absent, most-specific first
_COMMON_EOS_TOKENS = ("<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>")
_STANDARD_SPECIAL_TOKEN_FIELDS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)
_TOKENIZER_BEHAVIOR_FLAGS = {"add_bos_token", "add_eos_token"}
GrammarVocabType = Literal["raw", "byte_fallback", "byte_level"]


def _normalize_special_token(
    value: object,
    *,
    field: str,
    source: str = "tokenizer_config.json",
) -> str | None:
    """Normalize Hugging Face's string-or-AddedToken config representation."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
        raise ValueError(f"{source} {field!r} AddedToken must contain a non-empty string 'content'")
    raise ValueError(
        f"{source} {field!r} must be a string, null, or "
        f"an AddedToken object; got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TokenizerChatMetadata:
    """Chat templates and named special tokens stored beside a tokenizer.

    A single template is normalized to the ``default`` name, which lets the
    deployment layer use one selection path for both single- and multi-template
    tokenizers.  The mapping proxies keep this frozen value immutable beyond
    the dataclass attributes themselves.
    """

    templates: Mapping[str, str]
    special_tokens: Mapping[str, str]
    template_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", MappingProxyType(dict(self.templates)))
        object.__setattr__(
            self,
            "special_tokens",
            MappingProxyType(dict(self.special_tokens)),
        )


@dataclass(frozen=True)
class GrammarVocabulary:
    """Encoded tokenizer vocabulary plus the metadata xgrammar must decode with.

    Hugging Face tokenizer vocabularies are not uniformly literal text:
    byte-level BPE represents a space as ``Ġ`` and byte-fallback tokenizers use
    strings such as ``<0x1B>``. Treating either as raw text can leave a valid
    grammar with no legal tokens. ``vocab_size`` is the model lm-head width,
    which may include padding ids absent from the tokenizer vocabulary.
    """

    encoded_vocab: list[str]
    vocab_type: GrammarVocabType = "raw"
    vocab_size: int | None = None
    add_prefix_space: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ValueError("grammar vocabulary size must be positive")


@runtime_checkable
class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def vocab(self) -> list[str]: ...


class TokenDecodeStream(Protocol):
    """Optional tokenizer-native incremental decode contract.

    Implementations return only the text contributed by ``token_ids`` and must
    retain any decoder state needed across calls. Tokenizers without this
    capability continue through ``IncrementalDetokenizer``'s exact full-prefix
    fallback.
    """

    def push(self, token_ids: Sequence[int]) -> str: ...


def _stable_hash(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")


class _ToyDecodeStream:
    """O(delta) stream for ToyTokenizer's context-free space joining."""

    def __init__(self) -> None:
        self._has_tokens = False

    def push(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        text = " ".join(f"tok{token_id}" for token_id in token_ids)
        if self._has_tokens:
            text = f" {text}"
        self._has_tokens = True
        return text


class ToyTokenizer:
    """Word-hash placeholder tokenizer (process-stable: sha256, never hash())."""

    eos_token_id: int | None = None

    def encode(self, text: str) -> tuple[int, ...]:
        words = text.split()
        if not words:
            return (0,)
        return tuple(_stable_hash(word) % _TOY_VOCAB for word in words)

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def vocab(self) -> list[str]:
        return [f"tok{i}" for i in range(_TOY_VOCAB)]

    def new_decode_stream(self) -> TokenDecodeStream | None:
        # A subclass may override decode() while inheriting this factory. The
        # Toy chunk format would then disagree with its public full decode, so
        # capability detection must fall back unless both implementations match.
        if type(self).decode is not ToyTokenizer.decode:
            return None
        return _ToyDecodeStream()


def _import_tokenizers():
    try:
        import tokenizers
    except ImportError as error:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "HF tokenizer support requires the 'tokenizers' package (uv sync --extra hf)"
        ) from error
    return tokenizers


def _nodes_with_type(value: object, wanted: str):
    """Yield tokenizer.json nodes of ``wanted`` type, including sequences."""
    if isinstance(value, dict):
        if value.get("type") == wanted:
            yield value
        for child in value.values():
            yield from _nodes_with_type(child, wanted)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes_with_type(child, wanted)


def _grammar_metadata(backend: dict) -> tuple[GrammarVocabType, bool]:
    """Detect the public xgrammar vocabulary metadata from tokenizer.json."""
    decoder = backend.get("decoder")
    vocab_type: GrammarVocabType
    if next(_nodes_with_type(decoder, "ByteLevel"), None) is not None:
        vocab_type = "byte_level"
    elif next(_nodes_with_type(decoder, "ByteFallback"), None) is not None:
        vocab_type = "byte_fallback"
    else:
        vocab_type = "raw"

    pre_tokenizer = backend.get("pre_tokenizer")
    byte_level = next(_nodes_with_type(pre_tokenizer, "ByteLevel"), None)
    if byte_level is not None:
        add_prefix_space = bool(byte_level.get("add_prefix_space", False))
    else:
        metaspace = next(_nodes_with_type(pre_tokenizer, "Metaspace"), None)
        add_prefix_space = (
            metaspace is not None and metaspace.get("prepend_scheme", "always") != "never"
        )
    return vocab_type, add_prefix_space


class HFTokenizer:
    """Wraps a Hugging Face ``tokenizer.json`` (file or containing directory).

    ``eos_token`` is read from a sibling ``tokenizer_config.json`` when present.
    Both Hugging Face's string and AddedToken object representations are
    supported. Missing or null values are probed from common EOS token names.
    """

    def __init__(self, path: str | Path, eos_token: str | None = None) -> None:
        tokenizers = _import_tokenizers()
        file, config = _locate_tokenizer_files(Path(path))
        self._tok = tokenizers.Tokenizer.from_file(str(file))
        self._grammar_vocab_type, self._grammar_add_prefix_space = _grammar_metadata(
            json.loads(self._tok.to_str())
        )
        if eos_token is not None and not isinstance(eos_token, str):
            raise TypeError("eos_token must be a string or None")
        if eos_token is None and config is not None:
            payload = _load_json_object(config, label="tokenizer_config.json")
            eos_token = _normalize_special_token(payload.get("eos_token"), field="eos_token")
        self.eos_token_id = self._resolve_eos(eos_token)
        self._vocab: list[str] | None = None

    def _resolve_eos(self, eos_token: str | None) -> int | None:
        candidates = (eos_token,) if eos_token is not None else _COMMON_EOS_TOKENS
        for candidate in candidates:
            token_id = self._tok.token_to_id(candidate)
            if token_id is not None:
                return token_id
        return None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tok.decode(list(token_ids), skip_special_tokens=True)

    def vocab(self) -> list[str]:
        if self._vocab is None:
            size = self._tok.get_vocab_size()
            table = [""] * size
            for token, token_id in self._tok.get_vocab().items():
                if token_id < size:
                    table[token_id] = token
            self._vocab = table
        return self._vocab

    def grammar_vocabulary(self) -> GrammarVocabulary:
        """Return cached token strings with their tokenizer-native encoding."""
        return GrammarVocabulary(
            self.vocab(),
            vocab_type=self._grammar_vocab_type,
            add_prefix_space=self._grammar_add_prefix_space,
        )

    def new_decode_stream(self) -> TokenDecodeStream | None:
        """Use the Rust tokenizer's stateful decoder when the version supports it.

        ``tokenizers`` added ``DecodeStream`` after Kairyu's original minimum
        dependency. Older installations stay byte-correct through the generic
        fallback instead of failing at startup.
        """
        if type(self).decode is not HFTokenizer.decode:
            return None
        tokenizers = _import_tokenizers()
        stream_type = getattr(tokenizers.decoders, "DecodeStream", None)
        if stream_type is None:
            return None
        return _HFDecodeStream(self._tok, stream_type(skip_special_tokens=True))


class _HFDecodeStream:
    """Adapter from tokenizers.DecodeStream's optional chunks to our contract."""

    def __init__(self, tokenizer: object, stream: object) -> None:
        self._tokenizer = tokenizer
        self._stream = stream

    def push(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        chunk = self._stream.step(self._tokenizer, list(token_ids))
        return chunk if chunk is not None else ""


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_chat_template(name: object, template: object, *, source: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source} contains an empty or non-string chat template name")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{source} chat template {name!r} must be a non-empty string")
    return name, template


def _chat_templates_from_config(
    config: Mapping[str, object],
    *,
    source: str,
) -> dict[str, str]:
    if "chat_template" not in config or config["chat_template"] is None:
        return {}

    raw = config["chat_template"]
    if isinstance(raw, str):
        name, template = _validate_chat_template("default", raw, source=source)
        return {name: template}

    templates: dict[str, str] = {}
    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f"{source} chat_template mapping must not be empty")
        entries = raw.items()
    elif isinstance(raw, list):
        if not raw:
            raise ValueError(f"{source} chat_template list must not be empty")
        normalized_entries: list[tuple[object, object]] = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict) or "name" not in entry or "template" not in entry:
                raise ValueError(
                    f"{source} chat_template entry {index} must contain 'name' and 'template'"
                )
            normalized_entries.append((entry["name"], entry["template"]))
        entries = normalized_entries
    else:
        raise ValueError(f"{source} chat_template must be a string, mapping, legacy list, or null")

    for raw_name, raw_template in entries:
        name, template = _validate_chat_template(
            raw_name,
            raw_template,
            source=source,
        )
        if name in templates:
            raise ValueError(f"{source} contains duplicate chat template {name!r}")
        templates[name] = template
    return templates


def _apply_named_special_tokens(
    target: dict[str, str],
    payload: Mapping[str, object],
    *,
    source: str,
    preserve_fields: frozenset[str] = frozenset(),
) -> None:
    """Overlay named token values from one HF configuration object."""
    named_fields = {
        key
        for key in payload
        if key in _STANDARD_SPECIAL_TOKEN_FIELDS
        or (key.endswith("_token") and key not in _TOKENIZER_BEHAVIOR_FLAGS)
    }
    for field in sorted(named_fields):
        if field in preserve_fields:
            continue
        token = _normalize_special_token(payload[field], field=field, source=source)
        if token is None:
            target.pop(field, None)
        else:
            target[field] = token

    extra = payload.get("extra_special_tokens")
    if extra is None or isinstance(extra, list):
        # Lists are valid HF metadata but have no names to expose to Jinja.
        return
    if not isinstance(extra, dict):
        raise ValueError(f"{source} 'extra_special_tokens' must be a mapping, list, or null")
    for raw_name, raw_token in extra.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{source} 'extra_special_tokens' contains an empty token name")
        token = _normalize_special_token(raw_token, field=raw_name, source=source)
        if token is None:
            raise ValueError(
                f"{source} 'extra_special_tokens' value for {raw_name!r} must not be null"
            )
        target[raw_name] = token


def load_tokenizer_chat_metadata(
    path: str | Path,
    *,
    include_templates: bool = True,
) -> TokenizerChatMetadata:
    """Load local Hugging Face chat metadata without importing Transformers.

    Dedicated Jinja files collectively replace the legacy
    ``tokenizer_config.json`` template value, matching Transformers' loading
    precedence. Modern configs containing ``added_tokens_decoder`` own their
    named tokens; for legacy configs, ``special_tokens_map.json`` overlays the
    config exactly as Transformers 5.12 does. ``include_templates=False`` is
    used by an explicit deployment override so malformed unused checkpoint
    templates cannot defeat that higher-precedence policy; special tokens are
    still loaded for the override's render context.
    """
    source_path = Path(path)
    if source_path.is_dir():
        tokenizer_root = source_path
    elif source_path.is_file():
        tokenizer_root = source_path.parent
    else:
        raise ValueError(f"tokenizer metadata path does not exist: {source_path}")
    config_candidate = tokenizer_root / "tokenizer_config.json"
    config_file = config_candidate if config_candidate.is_file() else None
    config: dict[str, object] = (
        _load_json_object(config_file, label="tokenizer_config.json")
        if config_file is not None
        else {}
    )

    templates: dict[str, str] = {}
    template_sources: list[str] = []
    dedicated_files: list[tuple[str, Path]] = []
    if include_templates:
        default_file = tokenizer_root / "chat_template.jinja"
        if default_file.is_file():
            dedicated_files.append(("default", default_file))
        # Transformers 5.12 stores non-default named templates in this
        # directory (``CHAT_TEMPLATE_DIR`` in transformers.utils.hub).
        template_dir = tokenizer_root / "additional_chat_templates"
        if template_dir.is_dir():
            dedicated_files.extend(
                (template_file.stem, template_file)
                for template_file in sorted(template_dir.glob("*.jinja"))
                if template_file.is_file()
            )

    for raw_name, template_file in dedicated_files:
        name, template = _validate_chat_template(
            raw_name,
            template_file.read_text(encoding="utf-8"),
            source=str(template_file),
        )
        if name in templates:
            raise ValueError(f"duplicate dedicated chat template {name!r}")
        templates[name] = template
        template_sources.append(str(template_file))

    if include_templates and not dedicated_files:
        templates = _chat_templates_from_config(
            config,
            source="tokenizer_config.json",
        )
        if templates and config_file is not None:
            template_sources.append(f"{config_file}:chat_template")

    special_tokens: dict[str, str] = {}
    _apply_named_special_tokens(
        special_tokens,
        config,
        source="tokenizer_config.json",
    )
    special_tokens_map_file = tokenizer_root / "special_tokens_map.json"
    if (
        "added_tokens_decoder" not in config
        and special_tokens_map_file.is_file()
    ):
        # Transformers extracts direct model-specific ``*_token`` values from
        # tokenizer_config before applying the legacy map, so those config
        # values retain priority. Standard fields and the map's
        # ``extra_special_tokens`` entries are applied later and do override.
        config_model_specific_fields = {
            key
            for key, value in config.items()
            if key.endswith("_token")
            and key not in _STANDARD_SPECIAL_TOKEN_FIELDS
            and key not in _TOKENIZER_BEHAVIOR_FLAGS
            # At this point Transformers 5.12 has not recursively converted
            # serialized AddedToken objects yet. Only direct string values are
            # extracted early enough to beat a legacy map entry of the same
            # name; a top-level AddedToken JSON object is overwritten later.
            and isinstance(value, str)
        }
        config_extra = config.get("extra_special_tokens")
        if isinstance(config_extra, dict):
            config_model_specific_fields.update(config_extra)
        _apply_named_special_tokens(
            special_tokens,
            _load_json_object(
                special_tokens_map_file,
                label="special_tokens_map.json",
            ),
            source="special_tokens_map.json",
            preserve_fields=frozenset(config_model_specific_fields),
        )

    return TokenizerChatMetadata(
        templates=templates,
        special_tokens=special_tokens,
        template_sources=tuple(template_sources),
    )


def _locate_tokenizer_files(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        file = path / "tokenizer.json"
        config: Path | None = path / "tokenizer_config.json"
    else:
        file = path
        config = path.parent / "tokenizer_config.json"
    if not file.is_file():
        raise ValueError(f"no tokenizer.json at {path}")
    assert config is not None
    return file, (config if config.is_file() else None)


def resolve_tokenizer(tokenizer: str | Tokenizer) -> Tokenizer:
    """Resolve the backend's ``tokenizer=`` option; fails fast on a bad path."""
    if not isinstance(tokenizer, str):
        return tokenizer
    if tokenizer == "toy":
        return ToyTokenizer()
    try:
        return HFTokenizer(tokenizer)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"could not load tokenizer from {tokenizer!r}: {error}") from error


def grammar_vocabulary(
    tokenizer: Tokenizer,
    *,
    model_vocab_size: int | None = None,
) -> GrammarVocabulary:
    """Build xgrammar input while preserving legacy/custom tokenizer support."""
    factory = getattr(tokenizer, "grammar_vocabulary", None)
    if callable(factory):
        vocabulary = factory()
        if not isinstance(vocabulary, GrammarVocabulary):
            raise TypeError("grammar_vocabulary() must return GrammarVocabulary")
    else:
        vocabulary = GrammarVocabulary(tokenizer.vocab())
    if model_vocab_size is not None:
        if len(vocabulary.encoded_vocab) > model_vocab_size:
            raise ValueError(
                f"tokenizer vocab ({len(vocabulary.encoded_vocab)}) exceeds the "
                f"model's vocab_size ({model_vocab_size})"
            )
        vocabulary = replace(vocabulary, vocab_size=model_vocab_size)
    return vocabulary


class IncrementalDetokenizer:
    """Per-request streaming detokenizer: emits only never-retracted text.

    ``push`` returns the cumulative *stable* text; trailing replacement
    characters (incomplete UTF-8 across byte-level token boundaries) are held
    back until later tokens complete them. ``finalize`` returns the full
    decode of everything pushed.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._stable = ""
        stream_factory = getattr(tokenizer, "new_decode_stream", None)
        self._stream: TokenDecodeStream | None = (
            stream_factory() if callable(stream_factory) else None
        )

    def push(self, token_ids: Sequence[int]) -> str:
        self._ids.extend(token_ids)
        if self._stream is not None:
            # Native streams own decoder context and withhold incomplete UTF-8,
            # so only the arriving delta is processed. Appending their chunks
            # is never-retract by contract.
            self._stable += self._stream.push(token_ids)
            return self._stable

        # Safe compatibility path for arbitrary Tokenizer implementations that
        # cannot promise context-correct incremental chunks.
        text = self._tokenizer.decode(tuple(self._ids))
        stable = text.rstrip(_REPLACEMENT_CHAR)
        # never retract: only advance when the new stable text extends the old
        if len(stable) > len(self._stable) and stable.startswith(self._stable):
            self._stable = stable
        return self._stable

    def finalize(self) -> str:
        # One exact O(n) decode preserves the historical final-output contract
        # and flushes a genuinely incomplete byte sequence as U+FFFD.
        return self._tokenizer.decode(tuple(self._ids))

    @property
    def uses_native_stream(self) -> bool:
        """Whether pushes use bounded native incremental work."""
        return self._stream is not None
