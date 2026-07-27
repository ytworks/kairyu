"""The three false-passes review [P1] on #131 demonstrated, pinned.

Each of these let a run report PASS while the thing being gated was wrong, and
each is checkable without a GPU.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _decide(**overrides):
    from bench.parity_hf import decide

    base = dict(
        agreed=100,
        total=100,
        reference_self_agreement=0.98,
        max_abs_delta=0.0,
        substantive=0,
        missing_samples=0,
    )
    base.update(overrides)
    return decide(**base)


def test_a_clean_run_passes():
    passed, _reason = _decide()
    assert passed is True


def test_rounding_cannot_lift_a_rate_over_the_bar():
    """296/299 = 0.98997 rounds to 0.99 and used to clear a >= 0.99 bar."""
    raw = 296 / 299
    assert round(raw, 4) == 0.99 and raw < 0.99  # display vs reality

    passed, reason = _decide(agreed=296, total=299, reference_self_agreement=0.99)
    assert passed is False
    assert "below the reference" in reason

    # and the same numbers pass when the reference really is that unstable
    assert _decide(agreed=296, total=299, reference_self_agreement=raw)[0] is True


def test_a_large_logprob_delta_fails_even_with_every_argmax_agreeing():
    """The reported false pass: agreement 1.0, max delta 99.0, verdict PASS."""
    passed, reason = _decide(max_abs_delta=99.0)
    assert passed is False
    assert "logprob delta" in reason


def test_a_delta_at_the_bound_passes_and_just_over_it_fails():
    from bench.parity_hf import _MAX_LOGPROB_DELTA

    assert _decide(max_abs_delta=_MAX_LOGPROB_DELTA)[0] is True
    assert _decide(max_abs_delta=_MAX_LOGPROB_DELTA + 1e-9)[0] is False


def test_a_substantive_disagreement_fails():
    passed, reason = _decide(substantive=1)
    assert passed is False
    assert "substantive" in reason


def test_a_missing_logprob_sample_is_not_a_free_pass():
    """Positions with no comparable logprob used to be dropped, so a run with
    ZERO samples reported max delta 0.0 and passed."""
    passed, reason = _decide(missing_samples=1)
    assert passed is False
    assert "could not be made" in reason


def _reference_file(tmp_path: Path, provenance: dict, entries: dict) -> Path:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"provenance": provenance, "reference": entries}))
    return path


def _provenance(**overrides) -> dict:
    base = {
        "schema": 5,
        "checkpoint_config_sha256": "aaaa",
        "checkpoint_weights_sha256": "aaab",
        "checkpoint_weight_files": {"model.safetensors": "a" * 64},
        "tokenizer_files_sha256": "bbbb",
        "tokenizer_vocab_sha256": "bbbc",
        "prompt_token_ids_sha256": "cccc",
        "num_prompts": 2,
        "positions": 4,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
    }
    base.update(overrides)
    return base


def _entries(count: int, positions: int) -> dict:
    return {
        f"p{i}": {
            "prompt": f"prompt {i}",
            "prompt_ids": [1, 2],
            "continuation": list(range(positions)),
            # each row must rank its own reference token, which is what a real
            # teacher-forced reference always does
            "hf_top_logprobs": [
                {str(step): -0.1, "999": -3.0} for step in range(positions)
            ],
        }
        for i in range(count)
    }


def test_a_matching_cache_is_accepted(tmp_path):
    from bench import parity_hf

    want = _provenance()
    path = _reference_file(tmp_path, want, _entries(2, 4))
    assert len(parity_hf._load_reference(path, want)) == 2


def test_reference_prompt_ids_match_kairyu_without_implicit_bos():
    from bench.parity_hf import _prompt_ids

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert text == "hello"
            assert add_special_tokens is False
            return type("Encoded", (), {"input_ids": [7, 8]})()

    assert _prompt_ids(Tokenizer(), "hello") == [7, 8]


def test_a_cache_from_another_checkpoint_is_rejected(tmp_path):
    from bench import parity_hf

    path = _reference_file(
        tmp_path, _provenance(checkpoint_config_sha256="dead"), _entries(2, 4)
    )
    with pytest.raises(SystemExit, match="checkpoint_config_sha256"):
        parity_hf._load_reference(path, _provenance())


def test_a_cache_from_another_prompt_set_is_rejected(tmp_path):
    from bench import parity_hf

    path = _reference_file(
        tmp_path, _provenance(prompt_token_ids_sha256="dead"), _entries(2, 4)
    )
    with pytest.raises(SystemExit, match="prompt_token_ids_sha256"):
        parity_hf._load_reference(path, _provenance())


def test_a_short_cache_is_rejected_rather_than_scored(tmp_path):
    """The reported case: 1 prompt x 1 token was scored while config.positions
    still said 16."""
    from bench import parity_hf

    path = _reference_file(tmp_path, _provenance(), _entries(1, 1))
    with pytest.raises(SystemExit, match="holds 1 prompts"):
        parity_hf._load_reference(path, _provenance())


def test_a_cache_with_short_continuations_is_rejected(tmp_path):
    from bench import parity_hf

    path = _reference_file(tmp_path, _provenance(), _entries(2, 2))
    with pytest.raises(SystemExit, match="continuation"):
        parity_hf._load_reference(path, _provenance())


def test_a_pre_envelope_cache_is_rejected(tmp_path):
    from bench import parity_hf

    path = tmp_path / "old.json"
    path.write_text(json.dumps(_entries(2, 4)))  # the old bare-dict format
    with pytest.raises(SystemExit, match="provenance envelope"):
        parity_hf._load_reference(path, _provenance())


def test_the_harness_is_formal_for_a2_but_scopes_the_extra_a1_evidence():
    from bench import parity_hf

    doc = parity_hf.__doc__ or ""
    assert "formal HF-relative measurement for G2 A2" in doc
    assert "A1 additionally requires full" in doc


def test_a_reference_row_missing_its_own_token_is_rejected(tmp_path):
    """Without the token ranked, a disagreement there has no computable gap and
    would score as a free pass."""
    from bench import parity_hf

    entries = _entries(2, 4)
    entries["p0"]["hf_top_logprobs"][2] = {"999": -0.1}  # its own token 2 absent
    path = _reference_file(tmp_path, _provenance(), entries)
    with pytest.raises(SystemExit, match="does not rank its own"):
        parity_hf._load_reference(path, _provenance())


def test_a_reference_with_empty_logprob_rows_is_rejected(tmp_path):
    """The minimal repro: empty rows gave positions=0 and self-agreement 1.0."""
    from bench import parity_hf

    entries = _entries(2, 4)
    entries["p0"]["hf_top_logprobs"][0] = {}
    path = _reference_file(tmp_path, _provenance(), entries)
    with pytest.raises(SystemExit, match="empty row"):
        parity_hf._load_reference(path, _provenance())


def test_a_reference_with_too_few_logprob_rows_is_rejected(tmp_path):
    from bench import parity_hf

    entries = _entries(2, 4)
    entries["p0"]["hf_top_logprobs"] = entries["p0"]["hf_top_logprobs"][:2]
    path = _reference_file(tmp_path, _provenance(), entries)
    with pytest.raises(SystemExit, match="top-logprob rows"):
        parity_hf._load_reference(path, _provenance())


def test_a_cache_from_other_weights_is_rejected(tmp_path):
    """config.json alone does not identify a checkpoint — a fine-tune hashes the
    same. The weight fingerprint is what separates them."""
    from bench import parity_hf

    path = _reference_file(
        tmp_path, _provenance(checkpoint_weights_sha256="dead"), _entries(2, 4)
    )
    with pytest.raises(SystemExit, match="checkpoint_weights_sha256"):
        parity_hf._load_reference(path, _provenance())


def test_a_cache_from_another_tokenizer_serialization_is_rejected(tmp_path):
    """Same vocab, different normalizer/pre-tokenizer: get_vocab() cannot see it."""
    from bench import parity_hf

    path = _reference_file(
        tmp_path, _provenance(tokenizer_files_sha256="dead"), _entries(2, 4)
    )
    with pytest.raises(SystemExit, match="tokenizer_files_sha256"):
        parity_hf._load_reference(path, _provenance())


def _write_local_tokenizer(directory) -> None:
    """A minimal tokenizer written to disk, not fetched.

    `AutoTokenizer.from_pretrained("gpt2")` put a Hub download inside an unmarked
    unit test: with an empty cache and HF_HUB_OFFLINE=1 it errored with
    LocalEntryNotFoundError (review [P1] on #131). The provenance fingerprint only
    needs a tokenizer that loads and tokenizes deterministically.
    """
    import json

    vocab = {f"<{index}>": index for index in range(64)}
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": "<0>",
                },
            }
        )
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>"})
    )


def test_a_cache_from_other_weight_files_is_rejected(tmp_path):
    """The per-file digests are the checkpoint's identity, so a difference in any
    one of them stops the reference being scored — and says which file."""
    from bench import parity_hf

    path = _reference_file(
        tmp_path,
        _provenance(checkpoint_weight_files={"model.safetensors": "b" * 64}),
        _entries(2, 4),
    )
    with pytest.raises(SystemExit, match="model.safetensors"):
        parity_hf._load_reference(path, _provenance())


@pytest.fixture(scope="module")
def two_checkpoints(tmp_path_factory):
    """Same architecture and config, different weights, size and mtime preserved.

    Review [P2] on #131: hashing name/size/mtime accepts exactly this swap as the
    same checkpoint, while claiming to pin the actual bytes.
    """
    import os
    import shutil

    import torch

    transformers = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    )
    root = tmp_path_factory.mktemp("fingerprint")
    first, second = root / "a", root / "b"

    torch.manual_seed(1)
    transformers.LlamaForCausalLM(config).to(torch.float32).eval().save_pretrained(
        first, safe_serialization=True
    )
    _write_local_tokenizer(first)
    shutil.copytree(first, second)

    original = (second / "model.safetensors").stat()
    torch.manual_seed(2)
    transformers.LlamaForCausalLM(config).to(torch.float32).eval().save_pretrained(
        second, safe_serialization=True
    )
    os.utime(second / "model.safetensors", (original.st_atime, original.st_mtime))
    return str(first), str(second)


def test_different_weights_are_not_the_same_checkpoint(two_checkpoints):
    from bench.parity_hf import _provenance

    first, second = two_checkpoints
    a, b = _provenance(first, ["hello"], 1), _provenance(second, ["hello"], 1)

    # the architecture is identical, so config alone cannot tell them apart
    assert a["checkpoint_config_sha256"] == b["checkpoint_config_sha256"]
    # nor can size/mtime — they were deliberately preserved
    import pathlib

    stats = [pathlib.Path(p, "model.safetensors").stat() for p in (first, second)]
    assert stats[0].st_size == stats[1].st_size
    assert int(stats[0].st_mtime) == int(stats[1].st_mtime)
    # the weight fingerprint must still separate them
    assert a["checkpoint_weights_sha256"] != b["checkpoint_weights_sha256"]


def test_the_same_checkpoint_fingerprints_stably(two_checkpoints):
    from bench.parity_hf import _provenance

    first, _second = two_checkpoints
    assert _provenance(first, ["hello"], 1) == _provenance(first, ["hello"], 1)


def test_every_weight_file_digest_is_the_whole_file(two_checkpoints):
    """Not a sample of it.

    The recorded digest must equal `sha256sum` over the file — which is also the
    oid the Hub publishes for that safetensors blob, so committed evidence can be
    tied back to a revision by anyone, without this script.
    """
    import hashlib

    from bench.parity_hf import _provenance

    first, _second = two_checkpoints
    digests = _provenance(first, ["hello"], 1)["checkpoint_weight_files"]
    assert digests, "no weight file was fingerprinted at all"
    for name, digest in digests.items():
        expected = hashlib.sha256((Path(first) / name).read_bytes()).hexdigest()
        assert digest == expected, f"{name} is not a full-content digest"


def _withdrawn_sampled_windows(path: Path) -> list[tuple[int, int]]:
    """The byte ranges the withdrawn sampling fingerprint actually read.

    Header, then 4 KB at 0%/33%/66%/99% of the payload. Reproduced here so the
    test can prove the byte it edits is one that fingerprint never looked at,
    rather than asserting it.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
    windows = [(0, 8 + header_length)]
    for fraction in (0.0, 0.33, 0.66, 0.99):
        offset = 8 + header_length + int((size - 8 - header_length) * fraction)
        start = min(offset, max(size - 4096, 0))
        windows.append((start, start + 4096))
    return windows


@pytest.fixture(scope="module")
def checkpoint_edited_between_the_windows(tmp_path_factory):
    """One payload byte changed, in a region no sampled window covered.

    `two_checkpoints` re-initialises every tensor, so the sampled windows move
    too and a sampling fingerprint separates them anyway. This is the case it
    does NOT catch, and the one review [P2] on #131 raised: a targeted edit
    between the windows left `checkpoint_weights_sha256` identical, so a
    reference cache built from different weights was accepted.

    The model is sized so the payload is megabytes — with a few kilobytes of
    weights the four windows cover the whole file and there is nothing to miss.
    """
    import os
    import shutil

    import torch

    transformers = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size=4096, hidden_size=256, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    root = tmp_path_factory.mktemp("windows")
    original, edited = root / "a", root / "b"

    torch.manual_seed(3)
    transformers.LlamaForCausalLM(config).to(torch.float32).eval().save_pretrained(
        original, safe_serialization=True
    )
    _write_local_tokenizer(original)
    shutil.copytree(original, edited)

    target = edited / "model.safetensors"
    before = target.stat()
    with target.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
    payload_start = 8 + header_length
    # halfway through the payload: between the 33% and 66% windows
    offset = payload_start + (before.st_size - payload_start) // 2
    with target.open("r+b") as handle:
        handle.seek(offset)
        (byte,) = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([byte ^ 0xFF]))
    os.utime(target, (before.st_atime, before.st_mtime))
    assert target.stat().st_size == before.st_size
    return str(original), str(edited), offset


def test_a_weight_edit_between_the_sampled_windows_is_still_a_different_checkpoint(
    checkpoint_edited_between_the_windows,
):
    from bench.parity_hf import _provenance

    original, edited, offset = checkpoint_edited_between_the_windows
    windows = _withdrawn_sampled_windows(Path(original) / "model.safetensors")
    # the premise: this byte is one the sampling fingerprint never read, so it
    # reported the two checkpoints as identical
    assert not any(start <= offset < end for start, end in windows)

    a = _provenance(original, ["hello"], 1)
    b = _provenance(edited, ["hello"], 1)
    assert a["checkpoint_config_sha256"] == b["checkpoint_config_sha256"]
    assert a["checkpoint_weights_sha256"] != b["checkpoint_weights_sha256"]
    assert a["checkpoint_weight_files"] != b["checkpoint_weight_files"]


def test_such_a_cache_is_refused_rather_than_scored(
    checkpoint_edited_between_the_windows, tmp_path
):
    """End to end: the reference from one of them cannot be scored against the
    other, which is the safety the fingerprint exists to provide."""
    from bench import parity_hf

    original, edited, _offset = checkpoint_edited_between_the_windows
    built = parity_hf._provenance(original, ["hello"], 4)
    path = _reference_file(tmp_path, built, _entries(1, 4))

    with pytest.raises(SystemExit, match="different checkpoint weights"):
        parity_hf._load_reference(path, parity_hf._provenance(edited, ["hello"], 4))
