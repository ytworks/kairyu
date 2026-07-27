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
from pathlib import Path
from typing import Protocol, runtime_checkable

_TOY_VOCAB = 50_000
_REPLACEMENT_CHAR = "�"
# eos candidates probed when tokenizer_config.json is absent, most-specific first
_COMMON_EOS_TOKENS = ("<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>")


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
            "HF tokenizer support requires the 'tokenizers' package "
            "(uv sync --extra hf)"
        ) from error
    return tokenizers


class HFTokenizer:
    """Wraps a Hugging Face ``tokenizer.json`` (file or containing directory).

    ``eos_token`` is read from a sibling ``tokenizer_config.json`` when present,
    otherwise probed from common EOS token names.
    """

    def __init__(self, path: str | Path, eos_token: str | None = None) -> None:
        tokenizers = _import_tokenizers()
        file, config = _locate_tokenizer_files(Path(path))
        self._tok = tokenizers.Tokenizer.from_file(str(file))
        if eos_token is None and config is not None:
            eos_token = json.loads(config.read_text()).get("eos_token")
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
