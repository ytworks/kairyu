"""The three false-passes review [P1] on #131 demonstrated, pinned.

Each of these let a run report PASS while the thing being gated was wrong, and
each is checkable without a GPU.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_rounding_cannot_lift_a_rate_over_the_bar():
    """296/299 = 0.98997 rounds to 0.99 and used to pass a >= 0.99 test."""
    from bench import parity_hf

    raw = 296 / 299
    assert round(raw, 4) == 0.99  # the display value
    assert raw < 0.99  # the real one
    # the module must compare the raw ratio; a rounded threshold is display only
    source = Path(parity_hf.__file__).read_text()
    assert "raw_rate >= noise_floor[\"raw_self_agreement_rate\"]" in source
    assert "raw_max_delta <= _MAX_LOGPROB_DELTA" in source


def test_a_large_logprob_delta_fails_even_with_every_argmax_agreeing():
    """A mock agreeing on every token while its logprobs are 99 nats off scored
    `max_abs_delta: 99.0` and still reported PASS."""
    from bench import parity_hf

    assert parity_hf._MAX_LOGPROB_DELTA < 99.0
    # both halves are required, so argmax agreement alone cannot carry a pass
    source = Path(parity_hf.__file__).read_text()
    assert (
        '"PASS"\n                if not substantive and raw_max_delta <= _MAX_LOGPROB_DELTA'
        in source
    )


def _reference_file(tmp_path: Path, provenance: dict, entries: dict) -> Path:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"provenance": provenance, "reference": entries}))
    return path


def _provenance(**overrides) -> dict:
    base = {
        "schema": 2,
        "checkpoint_config_sha256": "aaaa",
        "tokenizer_vocab_sha256": "bbbb",
        "prompt_set_sha256": "cccc",
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
            "hf_top_logprobs": [{"0": -0.1} for _ in range(positions)],
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

    path = _reference_file(tmp_path, _provenance(prompt_set_sha256="dead"), _entries(2, 4))
    with pytest.raises(SystemExit, match="prompt_set_sha256"):
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

    source = Path(parity_hf.__file__).read_text()
    assert "NOT runbook §1 Gate 1" in source
    assert "DIAGNOSTIC" in source
