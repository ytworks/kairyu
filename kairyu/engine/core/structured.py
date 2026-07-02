"""xgrammar-backed structured output enforcement (goal L1: xgrammar統合).

Thin wrapper over xgrammar's grammar-compile → token-bitmask flow so the
ModelRunner only needs: mask_logits() before sampling, accept() after. Works
with any vocab (the CPU toy vocab in tests, the real tokenizer on GPU).
Import of xgrammar is deferred so kairyu works without it installed.
"""

from __future__ import annotations

import json


def _import_xgrammar():
    try:
        import xgrammar
    except ImportError as error:  # pragma: no cover - exercised only without xgrammar
        raise RuntimeError(
            "structured output requires xgrammar (pip install xgrammar)"
        ) from error
    return xgrammar


class XGrammarEnforcer:
    """Per-request grammar state: masks logits to grammar-legal tokens."""

    def __init__(self, vocab: list[str], json_schema: dict | None = None) -> None:
        xgr = self._xgr = _import_xgrammar()
        tokenizer_info = xgr.TokenizerInfo(vocab)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        if json_schema is not None:
            compiled = compiler.compile_json_schema(json.dumps(json_schema))
        else:
            compiled = compiler.compile_builtin_json_grammar()
        self._matcher = xgr.GrammarMatcher(compiled)
        self._vocab_size = tokenizer_info.vocab_size
        self._bitmask = xgr.allocate_token_bitmask(1, self._vocab_size)

    def mask_logits(self, logits):
        """Set grammar-illegal token logits to -inf (in place); returns logits."""
        self._matcher.fill_next_token_bitmask(self._bitmask)
        self._xgr.apply_token_bitmask_inplace(logits.view(1, -1), self._bitmask)
        return logits

    def accept(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def is_terminated(self) -> bool:
        return self._matcher.is_terminated()
