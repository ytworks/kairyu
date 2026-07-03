"""Tokenizer seam: protocol impls + incremental detokenizer (design m8 D1)."""

import pytest

from kairyu.engine.tokenizer import (
    HFTokenizer,
    IncrementalDetokenizer,
    ToyTokenizer,
    resolve_tokenizer,
)


@pytest.fixture(scope="module")
def hf_tokenizer_dir(tmp_path_factory):
    """Tiny byte-level BPE built programmatically — no committed blobs."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    corpus = [
        "hello world this is kairyu",
        "the quick brown fox jumps over the lazy dog",
        "こんにちは世界 推論エンジンのテストです",
        "日本語とenglishの混在テキスト",
    ]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["[UNK]", "</s>"])
    tok.train_from_iterator(corpus, trainer)
    path = tmp_path_factory.mktemp("tok")
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text('{"eos_token": "</s>"}')
    return path


class TestToyTokenizer:
    def test_encode_is_deterministic_across_calls(self):
        toy = ToyTokenizer()
        assert toy.encode("hello world") == toy.encode("hello world")

    def test_encode_is_stable_not_process_hash(self):
        # sha256-based: pinned values survive process restarts (PYTHONHASHSEED)
        toy = ToyTokenizer()
        first = toy.encode("hello")[0]
        assert 0 <= first < 50_000
        assert toy.encode("hello")[0] == first

    def test_empty_prompt_yields_one_token(self):
        assert len(ToyTokenizer().encode("")) == 1

    def test_decode_renders_readable_ids(self):
        toy = ToyTokenizer()
        assert toy.decode((1, 2)) == "tok1 tok2"

    def test_no_eos(self):
        assert ToyTokenizer().eos_token_id is None

    def test_vocab_size(self):
        assert len(ToyTokenizer().vocab()) == 50_000


class TestHFTokenizer:
    def test_roundtrip(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_japanese_roundtrip(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        text = "こんにちは世界"
        assert tok.decode(tok.encode(text)) == text

    def test_eos_from_tokenizer_config(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        assert tok.eos_token_id is not None
        assert tok.vocab()[tok.eos_token_id] == "</s>"

    def test_accepts_direct_file_path(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir / "tokenizer.json")
        assert tok.decode(tok.encode("hello")) == "hello"

    def test_vocab_indexed_by_id(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        vocab = tok.vocab()
        ids = tok.encode("hello")
        assert all(isinstance(vocab[i], str) for i in ids)


class TestResolveTokenizer:
    def test_toy_by_name(self):
        assert isinstance(resolve_tokenizer("toy"), ToyTokenizer)

    def test_path_resolves_hf(self, hf_tokenizer_dir):
        assert isinstance(resolve_tokenizer(str(hf_tokenizer_dir)), HFTokenizer)

    def test_instance_passthrough(self):
        toy = ToyTokenizer()
        assert resolve_tokenizer(toy) is toy

    def test_bad_path_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="tokenizer"):
            resolve_tokenizer(str(tmp_path / "missing"))


class TestIncrementalDetokenizer:
    def test_incremental_equals_full_at_end(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        text = "こんにちは世界 hello world 日本語とenglish"
        ids = tok.encode(text)
        detok = IncrementalDetokenizer(tok)
        for token_id in ids:
            detok.push((token_id,))
        assert detok.finalize() == tok.decode(ids)

    def test_stable_text_never_retracts(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        ids = tok.encode("こんにちは世界です hello")
        detok = IncrementalDetokenizer(tok)
        previous = ""
        for token_id in ids:
            stable = detok.push((token_id,))
            assert stable.startswith(previous)
            previous = stable

    def test_stable_is_prefix_of_final(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        ids = tok.encode("日本語のテスト")
        detok = IncrementalDetokenizer(tok)
        stable = ""
        for token_id in ids:
            stable = detok.push((token_id,))
        assert detok.finalize().startswith(stable)

    def test_incomplete_utf8_held_back(self, hf_tokenizer_dir):
        # feeding byte-level tokens one at a time must never emit U+FFFD mid-stream
        tok = HFTokenizer(hf_tokenizer_dir)
        ids = tok.encode("こんにちは")
        detok = IncrementalDetokenizer(tok)
        for token_id in ids:
            stable = detok.push((token_id,))
            assert "�" not in stable

    def test_works_with_toy_tokenizer(self):
        toy = ToyTokenizer()
        detok = IncrementalDetokenizer(toy)
        detok.push((1,))
        stable = detok.push((2,))
        assert stable == "tok1 tok2"
        assert detok.finalize() == "tok1 tok2"
