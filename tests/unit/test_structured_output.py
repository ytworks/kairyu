import pytest

xgr = pytest.importorskip("xgrammar")
torch = pytest.importorskip("torch")

from kairyu.engine.core.structured import XGrammarEnforcer  # noqa: E402

# a tiny vocab sufficient to build JSON fragments
VOCAB = ["{", "}", "[", "]", '"', "a", "b", "1", "2", ":", ",", " ", "true", "null"]


def test_json_grammar_masks_invalid_first_tokens():
    enforcer = XGrammarEnforcer(vocab=VOCAB)
    logits = torch.zeros(len(VOCAB))
    masked = enforcer.mask_logits(logits.clone())
    open_brace = VOCAB.index("{")
    close_brace = VOCAB.index("}")
    assert masked[open_brace] != float("-inf")  # '{' can start a JSON value
    assert masked[close_brace] == float("-inf")  # '}' cannot


def test_accepting_tokens_advances_grammar_state():
    enforcer = XGrammarEnforcer(vocab=VOCAB)
    for token in ["{", '"', "a", '"', ":", "1", "}"]:
        assert enforcer.accept(VOCAB.index(token)) is True
    # grammar-invalid continuation is rejected mid-sequence too
    fresh = XGrammarEnforcer(vocab=VOCAB)
    assert fresh.accept(VOCAB.index("{")) is True
    assert fresh.accept(VOCAB.index(":")) is False
    # after a complete JSON object, starting a new value is illegal
    masked = enforcer.mask_logits(torch.zeros(len(VOCAB)))
    assert masked[VOCAB.index("{")] == float("-inf")


def test_schema_constrained_grammar_compiles():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    enforcer = XGrammarEnforcer(vocab=VOCAB, json_schema=schema)
    logits = torch.zeros(len(VOCAB))
    masked = enforcer.mask_logits(logits.clone())
    assert masked[VOCAB.index("{")] != float("-inf")
