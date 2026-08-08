"""xgrammar-backed structured output enforcement (goal L1: xgrammar統合).

Thin wrapper over xgrammar's grammar-compile → token-bitmask flow so the
ModelRunner only needs: mask_logits() before sampling, accept() after. Works
with any vocab (the CPU toy vocab in tests, the real tokenizer on GPU).
Import of xgrammar is deferred so kairyu works without it installed.
"""

from __future__ import annotations

import json

from kairyu.engine.tokenizer import GrammarVocabulary


def _import_xgrammar():
    try:
        import xgrammar
    except ImportError as error:  # pragma: no cover - exercised only without xgrammar
        raise RuntimeError("structured output requires xgrammar (pip install xgrammar)") from error
    return xgrammar


class XGrammarEnforcer:
    """Per-request grammar state: masks logits to grammar-legal tokens.

    ``stop_token_id`` (the tokenizer's EOS) is how a completed grammar
    terminates: once the JSON value is complete the bitmask allows the stop
    token, accepting it flips ``is_terminated()``. Without one, a completed
    grammar has no legal continuation and generation cannot end cleanly.
    """

    def __init__(
        self,
        vocab: list[str] | GrammarVocabulary,
        json_schema: dict | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        structural_tag: dict | None = None,
        stop_token_id: int | None = None,
    ) -> None:
        xgr = self._xgr = _import_xgrammar()
        stop_ids = [stop_token_id] if stop_token_id is not None else None
        if isinstance(vocab, GrammarVocabulary):
            vocab_type = {
                "raw": xgr.VocabType.RAW,
                "byte_fallback": xgr.VocabType.BYTE_FALLBACK,
                "byte_level": xgr.VocabType.BYTE_LEVEL,
            }[vocab.vocab_type]
            tokenizer_info = xgr.TokenizerInfo(
                vocab.encoded_vocab,
                vocab_type,
                vocab_size=vocab.vocab_size,
                stop_token_ids=stop_ids,
                add_prefix_space=vocab.add_prefix_space,
            )
        else:
            # Public tests and third-party runners that provide literal token
            # strings retain the pre-metadata RAW behavior.
            tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=stop_ids)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        formats = sum(
            value is not None
            for value in (json_schema, regex, grammar, structural_tag)
        )
        if formats > 1:
            raise ValueError("structured output accepts exactly one grammar format")
        if json_schema is not None:
            # strict format (no free whitespace): removes degenerate unbounded
            # whitespace runs — matches vLLM's disable_any_whitespace guidance
            compiled = compiler.compile_json_schema(json.dumps(json_schema), any_whitespace=False)
        elif regex is not None:
            compiled = compiler.compile_regex(regex)
        elif grammar is not None:
            compiled = compiler.compile_grammar(grammar)
        elif structural_tag is not None:
            compiled = compiler.compile_structural_tag(structural_tag)
        else:
            compiled = compiler.compile_builtin_json_grammar()
        self._matcher = xgr.GrammarMatcher(compiled)
        self._vocab_size = tokenizer_info.vocab_size
        self._bitmask = xgr.allocate_token_bitmask(1, self._vocab_size)
        self._device_bitmasks = {}

    def mask_logits(self, logits):
        """Set grammar-illegal token logits to -inf (in place); returns logits."""
        self._matcher.fill_next_token_bitmask(self._bitmask)
        bitmask = self._bitmask
        if logits.device.type != "cpu":
            key = (logits.device.type, logits.device.index)
            device_bitmask = self._device_bitmasks.get(key)
            if device_bitmask is None:
                device_bitmask = self._bitmask.to(device=logits.device)
                self._device_bitmasks[key] = device_bitmask
            else:
                device_bitmask.copy_(self._bitmask)
            bitmask = device_bitmask
        self._xgr.apply_token_bitmask_inplace(logits.view(1, -1), bitmask)
        return logits

    def accept(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def is_terminated(self) -> bool:
        return self._matcher.is_terminated()
