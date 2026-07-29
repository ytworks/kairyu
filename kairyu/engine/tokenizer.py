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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

_TOY_VOCAB = 50_000
_REPLACEMENT_CHAR = "�"
# eos candidates probed when tokenizer_config.json is absent, most-specific first
_COMMON_EOS_TOKENS = ("<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>")
GrammarVocabType = Literal["raw", "byte_fallback", "byte_level"]


def _normalize_special_token(value: object, *, field: str) -> str | None:
    """Normalize Hugging Face's string-or-AddedToken config representation."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
        raise ValueError(
            f"tokenizer_config.json {field!r} AddedToken must contain "
            "a non-empty string 'content'"
        )
    raise ValueError(
        f"tokenizer_config.json {field!r} must be a string, null, or "
        f"an AddedToken object; got {type(value).__name__}"
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
            payload = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("tokenizer_config.json must contain a JSON object")
            eos_token = _normalize_special_token(
                payload.get("eos_token"), field="eos_token"
            )
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
