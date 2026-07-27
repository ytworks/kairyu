"""Tokenizer seam: protocol impls + incremental detokenizer (design m8 D1)."""

import pytest

from kairyu.engine.tokenizer import (
    HFTokenizer,
    IncrementalDetokenizer,
    ToyTokenizer,
    _grammar_metadata,
    grammar_vocabulary,
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

    def test_grammar_vocabulary_preserves_byte_level_metadata(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        model_vocab_size = len(tok.vocab()) + 7

        vocabulary = grammar_vocabulary(tok, model_vocab_size=model_vocab_size)

        assert vocabulary.encoded_vocab is tok.vocab()
        assert vocabulary.vocab_type == "byte_level"
        assert vocabulary.vocab_size == model_vocab_size
        assert vocabulary.add_prefix_space is False

    def test_grammar_metadata_detects_nested_byte_fallback_and_metaspace(self):
        backend = {
            "decoder": {
                "type": "Sequence",
                "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}],
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": "▁",
                "prepend_scheme": "first",
            },
        }

        assert _grammar_metadata(backend) == ("byte_fallback", True)


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
        assert detok.uses_native_stream

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
        assert detok.uses_native_stream

    def test_native_stream_accepts_multi_token_commits(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        ids = tok.encode("こんにちは世界 hello world 日本語")
        detok = IncrementalDetokenizer(tok)

        for start in range(0, len(ids), 3):
            stable = detok.push(ids[start : start + 3])
            assert tok.decode(ids[: start + 3]).startswith(stable)

        assert detok.finalize() == tok.decode(ids)

    def test_hf_stream_matches_wordpiece_decoder_and_skips_specials(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordPiece(
                vocab={
                    "[UNK]": 0,
                    "hello": 1,
                    "##s": 2,
                    "world": 3,
                    "[SEP]": 4,
                },
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.WordPiece(prefix="##", cleanup=True)
        raw.add_special_tokens(["[SEP]"])
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        ids = (1, 2, 4, 3)
        detok = IncrementalDetokenizer(tok)

        stable = ""
        for token_id in ids:
            current = detok.push((token_id,))
            assert current.startswith(stable)
            stable = current

        assert detok.finalize() == tok.decode(ids) == "hellos world"

    def test_hf_stream_matches_metaspace_decoder(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"<unk>": 0, "▁hello": 1, "▁world": 2, "x": 3},
                unk_token="<unk>",
            )
        )
        raw.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        ids = (1, 3, 2)
        detok = IncrementalDetokenizer(tok)

        for token_id in ids:
            detok.push((token_id,))

        assert detok.finalize() == tok.decode(ids) == "hellox world"

    def test_long_native_stream_has_linear_decode_work(self):
        class _CountingStream:
            def __init__(self, owner):
                self._owner = owner

            def push(self, token_ids):
                self._owner.incremental_work += len(token_ids)
                return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

        class _CountingTokenizer:
            eos_token_id = None

            def __init__(self):
                self.incremental_work = 0
                self.full_decode_work = 0

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                self.full_decode_work += len(token_ids)
                return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

            def vocab(self):
                return []

            def new_decode_stream(self):
                return _CountingStream(self)

        token_count = 4096
        tok = _CountingTokenizer()
        detok = IncrementalDetokenizer(tok)
        for token_id in range(token_count):
            detok.push((token_id,))

        assert len(detok.finalize()) == token_count
        assert tok.incremental_work == token_count
        assert tok.full_decode_work == token_count
        assert tok.incremental_work + tok.full_decode_work == 2 * token_count

    def test_tokenizer_without_stream_uses_exact_safe_fallback(self):
        class _ContextTokenizer:
            eos_token_id = None

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                return "|".join(str(token_id) for token_id in token_ids)

            def vocab(self):
                return []

        tok = _ContextTokenizer()
        detok = IncrementalDetokenizer(tok)

        assert not detok.uses_native_stream
        assert detok.push((1,)) == "1"
        assert detok.push((2, 3)) == "1|2|3"
        assert detok.finalize() == "1|2|3"

    def test_decode_override_does_not_inherit_incompatible_native_stream(self):
        class _CustomToy(ToyTokenizer):
            def decode(self, token_ids):
                return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

        tok = _CustomToy()
        detok = IncrementalDetokenizer(tok)

        assert not detok.uses_native_stream
        assert detok.push((1, 2, 3)) == tok.decode((1, 2, 3))

    def test_hf_without_decode_stream_uses_safe_fallback(self, hf_tokenizer_dir, monkeypatch):
        from tokenizers import decoders

        monkeypatch.delattr(decoders, "DecodeStream")
        tok = HFTokenizer(hf_tokenizer_dir)
        ids = tok.encode("こんにちは世界 hello")
        detok = IncrementalDetokenizer(tok)

        assert not detok.uses_native_stream
        for token_id in ids:
            stable = detok.push((token_id,))
            assert tok.decode(ids).startswith(stable)
        assert detok.finalize() == tok.decode(ids)

    def test_fallback_holds_incomplete_replacement_suffix(self):
        class _SplitUtf8Tokenizer:
            eos_token_id = None

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                if tuple(token_ids) == (1,):
                    return "�"
                if tuple(token_ids) == (1, 2):
                    return "あ"
                return ""

            def vocab(self):
                return []

        detok = IncrementalDetokenizer(_SplitUtf8Tokenizer())

        assert detok.push((1,)) == ""
        assert detok.push((2,)) == "あ"
        assert detok.finalize() == "あ"
