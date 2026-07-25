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
        "schema": 2,
        "checkpoint_config_sha256": "aaaa",
        "checkpoint_weights_sha256": "aaab",
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


def test_the_harness_does_not_claim_to_be_the_formal_gate():
    from bench import parity_hf

    assert "NOT runbook §1 Gate 1" in (parity_hf.__doc__ or "")


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
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(first)
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
