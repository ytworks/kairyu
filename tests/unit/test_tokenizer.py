"""Tokenizer seam: protocol impls + incremental detokenizer (design m8 D1)."""

import json
import random

import pytest

from kairyu.engine.tokenizer import (
    HFTokenizer,
    IncrementalDetokenizer,
    ToyTokenizer,
    _grammar_metadata,
    _HFDecodeStream,
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
        "絵文字 😀 🐉 と CJK 世界のストリーミング",
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


def _write_byte_fallback_tokenizer(path):
    """Write a deterministic byte-fallback tokenizer with a special token."""
    from tokenizers import Tokenizer, decoders, models

    tokens = (
        "[UNK]",
        "a",
        "z",
        "<0xE3>",
        "<0x81>",
        "<0x82>",  # あ
        "<0xE4>",
        "<0xB8>",
        "<0x96>",  # 世
        "<0xE7>",
        "<0x95>",
        "<0x8C>",  # 界
        "<0xF0>",
        "<0x9F>",
        "<0x98>",
        "<0x80>",  # 😀
        "<0x90>",
        "<0x89>",  # 🐉
        "<special>",
    )
    raw = Tokenizer(
        models.WordLevel(
            vocab={token: token_id for token_id, token in enumerate(tokens)},
            unk_token="[UNK]",
        )
    )
    raw.decoder = decoders.ByteFallback()
    raw.add_special_tokens(["<special>"])
    raw.save(str(path / "tokenizer.json"))
    return path


@pytest.fixture
def byte_fallback_tokenizer(tmp_path):
    return HFTokenizer(_write_byte_fallback_tokenizer(tmp_path))


def _ids_for_tokens(tokenizer, *tokens):
    by_token = {token: token_id for token_id, token in enumerate(tokenizer.vocab())}
    return tuple(by_token[token] for token in tokens)


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

    def test_special_token_policy_is_a_noop_for_decode_and_native_stream(self):
        toy = ToyTokenizer()
        ids = (1, 2, 3)
        expected = "tok1 tok2 tok3"

        assert toy.decode(ids, skip_special_tokens=True) == expected
        assert toy.decode(ids, skip_special_tokens=False) == expected
        for skip_special_tokens in (True, False):
            detok = IncrementalDetokenizer(
                toy,
                skip_special_tokens=skip_special_tokens,
            )
            assert detok.uses_native_stream
            assert detok.push(ids[:1]) == "tok1"
            assert detok.push(ids[1:]) == expected
            assert detok.finalize() == expected

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

    def test_vocab_preserves_sparse_valid_token_ids(self, tmp_path):
        from tokenizers import Tokenizer, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "raw-piece": 5, "<special>": 9},
                unk_token="[UNK]",
            )
        )
        raw.add_special_tokens(["<special>"])
        raw.save(str(tmp_path / "tokenizer.json"))

        vocab = HFTokenizer(tmp_path).vocab()

        assert len(vocab) == 10
        assert vocab[5] == "raw-piece"
        assert vocab[9] == "<special>"
        assert vocab[1:5] == [""] * 4

    def test_vocab_rejects_pathological_sparse_id_amplification(self, tmp_path):
        from tokenizers import Tokenizer, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "far-away": 100_000},
                unk_token="[UNK]",
            )
        )
        raw.save(str(tmp_path / "tokenizer.json"))

        with pytest.raises(ValueError, match="IDs are too sparse"):
            HFTokenizer(tmp_path).vocab()

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

    def test_hf_fixed_seed_streaming_matches_full_decode(self, hf_tokenizer_dir):
        tok = HFTokenizer(hf_tokenizer_dir)
        rng = random.Random(361)
        atoms = ("hello", " ", "世界", "こんにちは", "😀", "🐉", "日本語", "kairyu")

        for _ in range(80):
            text = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 24)))
            ids = tok.encode(text)
            detok = IncrementalDetokenizer(tok)
            previous = ""
            offset = 0
            while offset < len(ids):
                width = rng.randint(1, 7)
                current = detok.push(ids[offset : offset + width])
                assert current.startswith(previous)
                previous = current
                offset += width

            final = detok.finalize()
            assert final.startswith(previous)
            assert final == tok.decode(ids)
            assert detok.finalize() == final

    def test_toy_fixed_seed_streaming_matches_full_decode(self):
        tok = ToyTokenizer()
        rng = random.Random(361)

        for _ in range(80):
            ids = tuple(rng.randrange(50_000) for _ in range(rng.randint(0, 128)))
            detok = IncrementalDetokenizer(tok)
            previous = ""
            offset = 0
            while offset < len(ids):
                width = rng.randint(1, 11)
                current = detok.push(ids[offset : offset + width])
                assert current.startswith(previous)
                previous = current
                offset += width

            assert detok.finalize() == tok.decode(ids)
            assert detok.finalize().startswith(previous)

    def test_byte_fallback_valid_cjk_emoji_and_special_match_full_decode(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        ids = _ids_for_tokens(
            tok,
            "a",
            "<0xE3>",
            "<0x81>",
            "<0x82>",
            "<0xE4>",
            "<0xB8>",
            "<0x96>",
            "<0xE7>",
            "<0x95>",
            "<0x8C>",
            "<0xF0>",
            "<0x9F>",
            "<0x98>",
            "<0x80>",
            "<special>",
            "<0xF0>",
            "<0x9F>",
            "<0x90>",
            "<0x89>",
            "z",
        )
        rng = random.Random(361)

        for _ in range(40):
            detok = IncrementalDetokenizer(tok)
            previous = ""
            offset = 0
            while offset < len(ids):
                width = rng.randint(1, 6)
                current = detok.push(ids[offset : offset + width])
                assert current.startswith(previous)
                previous = current
                offset += width

            assert detok.finalize() == tok.decode(ids) == "aあ世界😀🐉z"

    def test_hf_decode_and_native_stream_honor_special_token_policy(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        ids = _ids_for_tokens(tok, "a", "<special>", "<0xE3>", "<0x81>", "<0x82>", "z")

        assert tok.decode(ids, skip_special_tokens=True) == "aあz"
        assert tok.decode(ids, skip_special_tokens=False) == "a<special>あz"

        for skip_special_tokens, expected in (
            (True, "aあz"),
            (False, "a<special>あz"),
        ):
            detok = IncrementalDetokenizer(
                tok,
                skip_special_tokens=skip_special_tokens,
            )
            assert detok.uses_native_stream
            previous = ""
            for token_id in ids:
                current = detok.push((token_id,))
                assert current.startswith(previous)
                previous = current
            assert detok.finalize() == expected
            assert detok.finalize() == tok.decode(
                ids,
                skip_special_tokens=skip_special_tokens,
            )

    @pytest.mark.parametrize(
        "terminal_tokens",
        [
            ("<0xE3>",),
            ("<0xE3>", "<0x81>"),
            ("<0xF0>", "<0x9F>", "<0x98>"),
        ],
    )
    def test_byte_fallback_incomplete_terminal_is_flushed_without_retraction(
        self,
        byte_fallback_tokenizer,
        terminal_tokens,
    ):
        tok = byte_fallback_tokenizer
        ids = _ids_for_tokens(tok, "a", *terminal_tokens)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push(ids)
        final = detok.finalize()

        assert stable == "a"
        assert final.startswith(stable)
        assert final == tok.decode(ids)
        assert detok.finalize() == final

    def test_byte_fallback_terminal_flush_preserves_published_multibyte_text(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        # U+3041 followed by an incomplete leading byte. tokenizers' separate
        # full decode replaces the entire contiguous byte run, but the native
        # stream has already correctly published U+3041 and must not retract it.
        ids = _ids_for_tokens(tok, "<0xE3>", "<0x81>", "<0x81>", "<0xE3>")
        detok = IncrementalDetokenizer(tok)

        stable = detok.push(ids)
        final = detok.finalize()
        separate_full_decode = tok.decode(ids)

        assert stable == "ぁ"
        assert final == "ぁ�"
        assert final.startswith(stable)
        assert not separate_full_decode.startswith(stable)
        assert final != separate_full_decode

    def test_byte_fallback_malformed_interior_run_recovers_without_retraction(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        ids = _ids_for_tokens(
            tok,
            "<0xE3>",
            "<0x81>",
            "<0x81>",  # ぁ, already published by DecodeStream
            "<0xE3>",  # incomplete next character
            "a",  # makes DecodeStream's full window disagree with ぁ
            "<0xE4>",
            "<0xB8>",
            "<0x96>",  # 世, verifies the restarted stream remains usable
            "z",
        )
        detok = IncrementalDetokenizer(tok)
        previous = ""

        for token_id in ids:
            current = detok.push((token_id,))
            assert current.startswith(previous)
            previous = current

        final = detok.finalize()
        separate_full_decode = tok.decode(ids)

        assert final == "ぁ�a世z"
        assert final.startswith(previous)
        assert not separate_full_decode.startswith("ぁ")
        assert detok.finalize() == final

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

    def test_hf_ctc_no_growth_pending_token_is_not_duplicated(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "a": 1, "b": 2, "|": 3, "<pad>": 4},
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.CTC(
            pad_token="<pad>",
            word_delimiter_token="|",
            cleanup=True,
        )
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push((1, 1))

        assert stable == "a"
        assert detok.finalize() == tok.decode((1, 1)) == "a"

    @pytest.mark.parametrize(
        ("ids", "stable", "expected"),
        [
            ((3, 4, 3), "", "��"),
            ((1, 3, 4, 3), "a", "a��"),
        ],
    )
    def test_hf_ctc_terminal_replacements_retain_no_growth_separator(
        self,
        tmp_path,
        ids,
        stable,
        expected,
    ):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "a": 1, "|": 2, "�": 3, "<pad>": 4},
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.CTC(
            pad_token="<pad>",
            word_delimiter_token="|",
            cleanup=True,
        )
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        detok = IncrementalDetokenizer(tok)

        assert detok.push(ids) == stable
        assert detok.finalize() == tok.decode(ids) == expected

    def test_hf_ctc_preterminal_repeats_do_not_grow_wrapper_flush(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "a": 1, "|": 2, "�": 3, "<pad>": 4},
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.CTC(
            pad_token="<pad>",
            word_delimiter_token="|",
            cleanup=True,
        )
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        original_decode = tok.decode
        decode_calls = []

        def counting_decode(token_ids):
            decode_calls.append(tuple(token_ids))
            return original_decode(token_ids)

        tok.decode = counting_decode  # type: ignore[method-assign]
        detok = IncrementalDetokenizer(tok)

        assert detok.push((1,) * 4096 + (3,)) == "a"
        decode_calls.clear()
        assert detok.finalize() == "a�"
        assert decode_calls == [(1, 3), (1,)]

    def test_hf_adapter_does_not_feed_special_ids_to_native_stream(self):
        class _RecordingStream:
            def __init__(self):
                self.calls = []

            def step(self, _tokenizer, token_id):
                self.calls.append(token_id)
                return "a" if token_id == 1 else None

        stream = _RecordingStream()
        adapter = _HFDecodeStream(
            object(),
            lambda: stream,
            lambda token_ids: "a�" if tuple(token_ids) == (1, 2) else "�",
            supports_replacement_flush=True,
            special_token_ids=frozenset({99}),
        )

        assert adapter.push((1, *(99,) * 4096, 2)) == "a"
        assert stream.calls == [1, 2]
        assert adapter.finalize_suffix() == "�"

    def test_hf_adapter_recovers_legacy_invalid_prefix_message(self):
        streams = []

        class _Stream:
            def __init__(self, generation):
                self.generation = generation

            def step(self, _tokenizer, token_id):
                if self.generation == 0:
                    if token_id == 1:
                        return "a"
                    if token_id == 2:
                        return None
                    raise Exception("Invalid prefix encountered")
                return "z"

        def stream_factory():
            stream = _Stream(len(streams))
            streams.append(stream)
            return stream

        adapter = _HFDecodeStream(
            object(),
            stream_factory,
            lambda token_ids: "�" if tuple(token_ids) == (2,) else "",
            supports_replacement_flush=True,
            special_token_ids=frozenset(),
        )

        assert adapter.push((1, 2, 3)) == "a�z"
        assert len(streams) == 2
        assert adapter.finalize_suffix() == ""

    def test_hf_generic_replacement_token_is_flushed_at_terminal(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"[UNK]": 0, "a": 1, "�": 2},
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.Fuse()
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push((1, 2))

        assert stable == "a"
        assert detok.finalize() == tok.decode((1, 2)) == "a�"

    def test_hf_wordpiece_replacement_flush_preserves_contextual_space(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordPiece(
                vocab={"[UNK]": 0, "a": 1, "�": 2},
                unk_token="[UNK]",
            )
        )
        raw.decoder = decoders.WordPiece(prefix="##", cleanup=True)
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push((1, 2))

        assert stable == "a"
        assert detok.finalize() == tok.decode((1, 2)) == "a �"

    def test_hf_metaspace_replacement_flush_preserves_contextual_space(self, tmp_path):
        from tokenizers import Tokenizer, decoders, models

        raw = Tokenizer(
            models.WordLevel(
                vocab={"<unk>": 0, "▁a": 1, "▁�": 2},
                unk_token="<unk>",
            )
        )
        raw.decoder = decoders.Metaspace(
            replacement="▁",
            prepend_scheme="always",
            split=True,
        )
        raw.save(str(tmp_path / "tokenizer.json"))
        tok = HFTokenizer(tmp_path)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push((1, 2))

        assert stable == "a"
        assert detok.finalize() == tok.decode((1, 2)) == "a �"

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

    def test_hf_native_finalize_decodes_only_bounded_terminal_suffix(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        a_id, incomplete_id = _ids_for_tokens(tok, "a", "<0xE3>")
        original_decode = tok.decode
        decode_calls = []

        def counting_decode(token_ids):
            decode_calls.append(tuple(token_ids))
            return original_decode(token_ids)

        tok.decode = counting_decode  # type: ignore[method-assign]
        ids = (a_id,) * 4096 + (incomplete_id,)
        detok = IncrementalDetokenizer(tok)

        stable = detok.push(ids)
        final = detok.finalize()

        assert stable == "a" * 4096
        assert final == original_decode(ids)
        assert decode_calls == [
            (a_id, incomplete_id),
            (incomplete_id,),
        ]
        assert detok.finalize() == final
        assert decode_calls == [
            (a_id, incomplete_id),
            (incomplete_id,),
        ]

    def test_hf_terminal_tracking_excludes_long_special_token_runs(
        self,
        byte_fallback_tokenizer,
    ):
        tok = byte_fallback_tokenizer
        a_id, special_id, incomplete_id = _ids_for_tokens(
            tok,
            "a",
            "<special>",
            "<0xE3>",
        )
        original_decode = tok.decode
        decode_calls = []

        def counting_decode(token_ids):
            decode_calls.append(tuple(token_ids))
            return original_decode(token_ids)

        tok.decode = counting_decode  # type: ignore[method-assign]
        special_run = (special_id,) * 4096

        detok = IncrementalDetokenizer(tok)
        assert detok.push((a_id, *special_run)) == "a"
        assert detok.finalize() == "a"
        assert decode_calls == []

        detok = IncrementalDetokenizer(tok)
        assert detok.push((a_id, *special_run, incomplete_id)) == "a"
        assert detok.finalize() == "a�"
        assert decode_calls == [
            (a_id, incomplete_id),
            (incomplete_id,),
        ]
        assert detok.finalize() == "a�"
        assert decode_calls == [
            (a_id, incomplete_id),
            (incomplete_id,),
        ]

    def test_push_only_custom_stream_keeps_accumulated_output_compatibility(self):
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

    def test_legacy_custom_signatures_remain_compatible_when_specials_are_requested(self):
        class _LegacyStream:
            def push(self, token_ids):
                return "".join(str(token_id) for token_id in token_ids)

        class _LegacyTokenizer:
            eos_token_id = None

            def __init__(self):
                self.decode_calls = []
                self.stream_calls = 0

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                self.decode_calls.append(tuple(token_ids))
                return "".join(str(token_id) for token_id in token_ids)

            def vocab(self):
                return []

            def new_decode_stream(self):
                self.stream_calls += 1
                return _LegacyStream()

        tok = _LegacyTokenizer()
        detok = IncrementalDetokenizer(tok, skip_special_tokens=False)

        assert detok.push((1, 2, 3)) == "123"
        assert detok.finalize() == "123"
        assert tok.stream_calls == 1
        assert tok.decode_calls == [(1, 2, 3)]

    def test_keyword_aware_custom_tokenizer_receives_false_policy(self):
        class _PolicyStream:
            def __init__(self, skip_special_tokens):
                self._skip_special_tokens = skip_special_tokens

            def push(self, token_ids):
                return "".join(
                    "<special>" if token_id == 1 else str(token_id)
                    for token_id in token_ids
                    if not (self._skip_special_tokens and token_id == 1)
                )

        class _PolicyTokenizer:
            eos_token_id = None

            def __init__(self):
                self.decode_policies = []
                self.stream_policies = []

            def encode(self, text):
                return ()

            def decode(self, token_ids, *, skip_special_tokens=True):
                self.decode_policies.append(skip_special_tokens)
                return "".join(
                    "<special>" if token_id == 1 else str(token_id)
                    for token_id in token_ids
                    if not (skip_special_tokens and token_id == 1)
                )

            def vocab(self):
                return ["0", "<special>", "2"]

            def new_decode_stream(self, *, skip_special_tokens=True):
                self.stream_policies.append(skip_special_tokens)
                return _PolicyStream(skip_special_tokens)

        tok = _PolicyTokenizer()
        detok = IncrementalDetokenizer(tok, skip_special_tokens=False)

        assert detok.push((0, 1, 2)) == "0<special>2"
        assert detok.finalize() == "0<special>2"
        assert tok.stream_policies == [False]
        assert tok.decode_policies == [False]

    def test_false_policy_bypasses_legacy_default_skip_custom_stream(self):
        class _DefaultSkipStream:
            def push(self, token_ids):
                return "".join(
                    str(token_id)
                    for token_id in token_ids
                    if token_id != 1
                )

        class _MixedTokenizer:
            eos_token_id = None

            def __init__(self):
                self.decode_policies = []
                self.stream_calls = 0

            def encode(self, text):
                return ()

            def decode(self, token_ids, *, skip_special_tokens=True):
                self.decode_policies.append(skip_special_tokens)
                return "".join(
                    "<special>" if token_id == 1 else str(token_id)
                    for token_id in token_ids
                    if not (skip_special_tokens and token_id == 1)
                )

            def vocab(self):
                return ["0", "<special>", "2"]

            def new_decode_stream(self):
                self.stream_calls += 1
                return _DefaultSkipStream()

        tok = _MixedTokenizer()
        detok = IncrementalDetokenizer(tok, skip_special_tokens=False)

        assert not detok.uses_native_stream
        assert detok.push((0, 1)) == "0<special>"
        assert detok.push((2,)) == "0<special>2"
        assert detok.finalize() == "0<special>2"
        assert tok.stream_calls == 0
        assert tok.decode_policies == [False, False]

    def test_push_only_custom_stream_finalization_never_retracts_and_is_idempotent(self):
        class _PublishedStream:
            def push(self, token_ids):
                return "published" if token_ids else ""

            def finalize(self):
                raise AssertionError("an unrelated legacy finalize must not be called")

        class _DivergingTokenizer:
            eos_token_id = None

            def __init__(self):
                self.decode_calls = 0

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                self.decode_calls += 1
                return "different-final-text"

            def vocab(self):
                return []

            def new_decode_stream(self):
                return _PublishedStream()

        tok = _DivergingTokenizer()
        detok = IncrementalDetokenizer(tok)

        stable = detok.push((1,))
        final = detok.finalize()

        assert stable == "published"
        assert final == stable
        assert detok.finalize() == final
        assert tok.decode_calls == 1

    def test_push_only_custom_stream_can_flush_a_safe_terminal_suffix(self):
        class _HeldTailStream:
            def push(self, token_ids):
                return "".join(str(token_id) for token_id in token_ids[:-1])

        class _Tokenizer:
            eos_token_id = None

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                return "".join(str(token_id) for token_id in token_ids)

            def vocab(self):
                return []

            def new_decode_stream(self):
                return _HeldTailStream()

        detok = IncrementalDetokenizer(_Tokenizer())

        assert detok.push((1, 2, 3)) == "12"
        assert detok.finalize() == "123"

    def test_finalizable_custom_stream_uses_suffix_without_full_decode(self):
        class _FinalizableStream:
            def __init__(self):
                self.finalize_calls = 0

            def push(self, token_ids):
                return "".join(str(token_id) for token_id in token_ids[:-1])

            def finalize_suffix(self):
                self.finalize_calls += 1
                return "!"

        class _Tokenizer:
            eos_token_id = None

            def __init__(self):
                self.stream = _FinalizableStream()

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                raise AssertionError("finalizable streams must not decode full history")

            def vocab(self):
                return []

            def new_decode_stream(self):
                return self.stream

        tok = _Tokenizer()
        detok = IncrementalDetokenizer(tok)

        assert detok.push((1, 2, 3)) == "12"
        assert detok.finalize() == "12!"
        assert detok.finalize() == "12!"
        assert tok.stream.finalize_calls == 1

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

    def test_fallback_finalize_never_retracts_or_decodes_again(self):
        class _DivergingTokenizer:
            eos_token_id = None

            def __init__(self):
                self.decode_calls = 0

            def encode(self, text):
                return ()

            def decode(self, token_ids):
                self.decode_calls += 1
                return "published" if tuple(token_ids) == (1,) else "different"

            def vocab(self):
                return []

        tok = _DivergingTokenizer()
        detok = IncrementalDetokenizer(tok)

        assert detok.push((1,)) == "published"
        assert detok.push((2,)) == "published"
        assert detok.finalize() == "published"
        assert detok.finalize() == "published"
        assert tok.decode_calls == 2

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

    def test_hf_without_decode_stream_fallback_honors_false_special_policy(
        self,
        byte_fallback_tokenizer,
        monkeypatch,
    ):
        from tokenizers import decoders

        monkeypatch.delattr(decoders, "DecodeStream")
        tok = byte_fallback_tokenizer
        ids = _ids_for_tokens(tok, "a", "<special>", "<0xE3>", "<0x81>", "<0x82>", "z")
        detok = IncrementalDetokenizer(tok, skip_special_tokens=False)

        assert not detok.uses_native_stream
        previous = ""
        for token_id in ids:
            current = detok.push((token_id,))
            assert current.startswith(previous)
            previous = current
        assert detok.finalize() == "a<special>あz"
        assert detok.finalize() == tok.decode(ids, skip_special_tokens=False)

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
