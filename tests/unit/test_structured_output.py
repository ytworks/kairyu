import pytest

xgr = pytest.importorskip("xgrammar")
torch = pytest.importorskip("torch")

from kairyu.engine.core.structured import XGrammarEnforcer  # noqa: E402
from kairyu.engine.tokenizer import GrammarVocabulary  # noqa: E402

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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regex": "a+"},
        {"grammar": 'root ::= "a"'},
        {
            "structural_tag": {
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "a"},
            }
        },
    ],
)
def test_extended_grammar_compilers_mask_to_the_requested_prefix(kwargs):
    enforcer = XGrammarEnforcer(vocab=VOCAB, **kwargs)

    masked = enforcer.mask_logits(torch.zeros(len(VOCAB)))

    assert masked[VOCAB.index("a")] != float("-inf")
    assert masked[VOCAB.index("b")] == float("-inf")


def test_enforcers_reuse_compiler_for_the_same_grammar_vocabulary(monkeypatch):
    vocab = GrammarVocabulary([*VOCAB, "cache-probe"])
    real_compiler = xgr.GrammarCompiler
    constructed = 0

    def counting_compiler(tokenizer_info):
        nonlocal constructed
        constructed += 1
        return real_compiler(tokenizer_info)

    monkeypatch.setattr(xgr, "GrammarCompiler", counting_compiler)

    XGrammarEnforcer(vocab, regex="a+")
    XGrammarEnforcer(vocab, regex="b+")

    assert constructed == 1


def test_byte_level_space_marker_remains_legal_after_schema_colon():
    # Qwen's tokenizer stores a space as Ġ. Interpreting this vocabulary as RAW
    # leaves no legal token after the schema's canonical `": ` prefix, causing
    # an all--inf argmax of token 0 and the production invariant failure.
    vocab = ["!", '{"', "answer", '":', "Ġ", "4", "}", "<eos>"]
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    raw = XGrammarEnforcer(
        GrammarVocabulary(vocab, vocab_type="raw", vocab_size=len(vocab)),
        json_schema=schema,
        stop_token_id=7,
    )
    byte_level = XGrammarEnforcer(
        GrammarVocabulary(vocab, vocab_type="byte_level", vocab_size=len(vocab)),
        json_schema=schema,
        stop_token_id=7,
    )
    for token_id in (1, 2, 3):
        assert raw.accept(token_id)
        assert byte_level.accept(token_id)

    raw_masked = raw.mask_logits(torch.zeros(len(vocab)))
    assert torch.isfinite(raw_masked).sum().item() == 0
    byte_masked = byte_level.mask_logits(torch.zeros(len(vocab)))
    assert torch.isfinite(byte_masked).nonzero().flatten().tolist() == [4]

    for token_id in (4, 5, 6, 7):
        assert byte_level.accept(token_id)
    assert byte_level.is_terminated()
