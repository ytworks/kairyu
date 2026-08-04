"""Tokenizer seam: protocol impls + incremental detokenizer (design m8 D1)."""

import json

import pytest

from kairyu.engine.tokenizer import (
    HFTokenizer,
    IncrementalDetokenizer,
    ToyTokenizer,
    _grammar_metadata,
    grammar_vocabulary,
    load_tokenizer_chat_metadata,
    resolve_tokenizer,
)


def _write_word_level_tokenizer(path, config):
    """Write a small tokenizer whose EOS candidates have stable IDs."""
    from tokenizers import Tokenizer, models

    raw = Tokenizer(
        models.WordLevel(
            vocab={
                "[UNK]": 0,
                "</s>": 1,
                "<｜end▁of▁sentence｜>": 2,
            },
            unk_token="[UNK]",
        )
    )
    raw.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


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

    def test_eos_from_added_token_config(self, tmp_path):
        path = _write_word_level_tokenizer(
            tmp_path,
            {
                "eos_token": {
                    "__type": "AddedToken",
                    "content": "<｜end▁of▁sentence｜>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                }
            },
        )

        tok = HFTokenizer(path)

        assert tok.eos_token_id == 2
        assert tok.vocab()[tok.eos_token_id] == "<｜end▁of▁sentence｜>"

    @pytest.mark.parametrize("config", [{}, {"eos_token": None}])
    def test_missing_or_null_eos_probes_common_tokens(self, tmp_path, config):
        tok = HFTokenizer(_write_word_level_tokenizer(tmp_path, config))

        assert tok.eos_token_id == 1

    @pytest.mark.parametrize(
        "eos_token",
        [
            {"__type": "AddedToken"},
            {"content": 7},
            {"content": ""},
            7,
            ["</s>"],
        ],
    )
    def test_rejects_malformed_eos_config(self, tmp_path, eos_token):
        path = _write_word_level_tokenizer(tmp_path, {"eos_token": eos_token})

        with pytest.raises(ValueError, match="eos_token"):
            HFTokenizer(path)

    def test_explicit_eos_takes_precedence_over_added_token_config(self, tmp_path):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"eos_token": {"content": "<｜end▁of▁sentence｜>"}},
        )

        tok = HFTokenizer(path, eos_token="</s>")

        assert tok.eos_token_id == 1

    def test_rejects_non_object_tokenizer_config(self, tmp_path):
        path = _write_word_level_tokenizer(tmp_path, [])

        with pytest.raises(ValueError, match="must contain a JSON object"):
            HFTokenizer(path)

    def test_rejects_non_string_explicit_eos(self, tmp_path):
        path = _write_word_level_tokenizer(tmp_path, {})

        with pytest.raises(TypeError, match="eos_token must be a string or None"):
            HFTokenizer(path, eos_token={"content": "</s>"})  # type: ignore[arg-type]

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


class TestTokenizerChatMetadata:
    @pytest.mark.parametrize(
        ("chat_template", "expected"),
        [
            ("{{ bos_token }}hello", {"default": "{{ bos_token }}hello"}),
            (
                {"default": "default-template", "tool_use": "tool-template"},
                {"default": "default-template", "tool_use": "tool-template"},
            ),
            (
                [
                    {"name": "default", "template": "legacy-default"},
                    {"name": "tool_use", "template": "legacy-tool"},
                ],
                {"default": "legacy-default", "tool_use": "legacy-tool"},
            ),
        ],
    )
    def test_loads_all_tokenizer_config_template_formats(
        self,
        tmp_path,
        chat_template,
        expected,
    ):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"chat_template": chat_template},
        )

        metadata = load_tokenizer_chat_metadata(path)

        assert metadata.templates == expected
        assert metadata.template_sources == (f"{path / 'tokenizer_config.json'}:chat_template",)

    @pytest.mark.parametrize(
        "chat_template",
        [
            "",
            "   ",
            {},
            [],
            7,
            {"": "template"},
            {"default": ""},
            [{"name": "default"}],
            [{"name": "default", "template": 7}],
            [
                {"name": "default", "template": "first"},
                {"name": "default", "template": "second"},
            ],
        ],
    )
    def test_rejects_empty_duplicate_or_malformed_config_templates(
        self,
        tmp_path,
        chat_template,
    ):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"chat_template": chat_template},
        )

        with pytest.raises(ValueError, match="chat_template|chat template"):
            load_tokenizer_chat_metadata(path)

    def test_dedicated_templates_collectively_override_config(self, tmp_path):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"chat_template": "ignored-config-template"},
        )
        default_file = path / "chat_template.jinja"
        default_file.write_text("dedicated-default", encoding="utf-8")
        template_dir = path / "additional_chat_templates"
        template_dir.mkdir()
        tool_file = template_dir / "tool_use.jinja"
        tool_file.write_text("dedicated-tool", encoding="utf-8")

        metadata = load_tokenizer_chat_metadata(path / "tokenizer.json")

        assert metadata.templates == {
            "default": "dedicated-default",
            "tool_use": "dedicated-tool",
        }
        assert metadata.template_sources == (str(default_file), str(tool_file))

    def test_named_dedicated_template_alone_still_overrides_config(self, tmp_path):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"chat_template": "ignored-config-template"},
        )
        template_dir = path / "additional_chat_templates"
        template_dir.mkdir()
        (template_dir / "tool_use.jinja").write_text(
            "dedicated-tool",
            encoding="utf-8",
        )

        metadata = load_tokenizer_chat_metadata(path)

        assert metadata.templates == {"tool_use": "dedicated-tool"}

    def test_rejects_duplicate_dedicated_default(self, tmp_path):
        path = _write_word_level_tokenizer(tmp_path, {})
        (path / "chat_template.jinja").write_text("root-default", encoding="utf-8")
        template_dir = path / "additional_chat_templates"
        template_dir.mkdir()
        (template_dir / "default.jinja").write_text(
            "directory-default",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate dedicated"):
            load_tokenizer_chat_metadata(path)

    def test_rejects_empty_dedicated_template(self, tmp_path):
        path = _write_word_level_tokenizer(tmp_path, {})
        (path / "chat_template.jinja").write_text("\n", encoding="utf-8")

        with pytest.raises(ValueError, match="non-empty"):
            load_tokenizer_chat_metadata(path)

    @pytest.mark.parametrize("modern", [False, True])
    def test_special_token_precedence_matches_transformers_5_12(
        self,
        tmp_path,
        modern,
    ):
        config = {
            "bos_token": "<new-bos>",
            "eos_token": "<new-eos>",
            "unk_token": None,
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            "image_token": "<new-image>",
            "object_token": {
                "__type": "AddedToken",
                "content": "<new-object>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "extra_special_tokens": {
                "audio_token": "<new-audio>",
                "sensor_token": "<new-sensor>",
                "video_marker": "<video>",
            },
            "add_bos_token": False,
            "add_eos_token": True,
        }
        if modern:
            config["added_tokens_decoder"] = {}
        path = _write_word_level_tokenizer(
            tmp_path,
            config,
        )
        (path / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": "<old-bos>",
                    "eos_token": "<old-eos>",
                    "unk_token": "<old-unk>",
                    "image_token": "<old-image>",
                    "object_token": "<old-object>",
                    "sensor_token": "<old-sensor>",
                    "extra_special_tokens": {
                        "audio_token": "<old-audio>",
                        "map_only_token": "<map-only>",
                    },
                }
            ),
            encoding="utf-8",
        )

        metadata = load_tokenizer_chat_metadata(path)
        expected = {
            "audio_token": "<new-audio>" if modern else "<old-audio>",
            "bos_token": "<new-bos>" if modern else "<old-bos>",
            "cls_token": "<cls>",
            "eos_token": "<new-eos>" if modern else "<old-eos>",
            # Direct model-specific config tokens are extracted before the
            # legacy map and retain priority in Transformers 5.12.
            "image_token": "<new-image>",
            "mask_token": "<mask>",
            "object_token": "<new-object>" if modern else "<old-object>",
            "pad_token": "<pad>",
            "sep_token": "<sep>",
            "sensor_token": "<new-sensor>",
            "video_marker": "<video>",
        }
        if modern:
            assert "unk_token" not in expected
        else:
            expected["map_only_token"] = "<map-only>"
            expected["unk_token"] = "<old-unk>"
        assert metadata.special_tokens == expected

        transformers = pytest.importorskip("transformers")
        reference = transformers.PreTrainedTokenizerFast.from_pretrained(
            path,
            local_files_only=True,
        )
        for name, value in expected.items():
            assert getattr(reference, name) == value

    @pytest.mark.parametrize(
        "config",
        [
            {"bos_token": 7},
            {"image_token": {"content": ""}},
            {"extra_special_tokens": "<image>"},
            {"extra_special_tokens": {"image_token": None}},
        ],
    )
    def test_rejects_malformed_named_special_tokens(self, tmp_path, config):
        path = _write_word_level_tokenizer(tmp_path, config)

        with pytest.raises(ValueError, match="token"):
            load_tokenizer_chat_metadata(path)

    def test_accepts_unnamed_extra_special_token_list(self, tmp_path):
        path = _write_word_level_tokenizer(
            tmp_path,
            {"extra_special_tokens": ["<one>", {"content": "<two>"}]},
        )

        metadata = load_tokenizer_chat_metadata(path)

        assert metadata.special_tokens == {}

    def test_metadata_does_not_require_fast_tokenizer_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            '{"chat_template": "template"}',
            encoding="utf-8",
        )

        metadata = load_tokenizer_chat_metadata(tmp_path)

        assert metadata.templates == {"default": "template"}


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
