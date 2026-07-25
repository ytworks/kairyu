"""The toy harness must run the configuration it reports."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(*args: str):
    return subprocess.run(
        [sys.executable, "bench/parity_tp.py", *args],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )


def test_max_new_tokens_is_honoured_not_just_recorded(tmp_path):
    """Review [P2] on #130: `_fixed_prompts` hardcoded 16 while the config
    recorded the CLI value, so the result described a run that never happened."""
    out = tmp_path / "result.json"
    result = _run(
        "--tp", "1,2", "--num-prompts", "1", "--max-new-tokens", "2",
        "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    assert payload["config"]["max_new_tokens"] == 2
    for entry in payload["parity"].values():
        assert entry["tokens"] == 2, "reported config and measured tokens disagree"


def test_a_non_positive_prompt_count_is_rejected():
    result = _run("--tp", "1,2", "--num-prompts", "0")
    assert result.returncode != 0
    assert "must be positive" in result.stderr


def test_the_toy_harness_still_gates_on_exactness(tmp_path):
    """Deterministic ranks: sharding must not move a single token."""
    out = tmp_path / "toy.json"
    result = _run("--tp", "1,2", "--num-prompts", "4", "--out", str(out))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert all(e["match_rate"] == 1.0 for e in payload["parity"].values())
    assert "toy harness must be exact" in result.stdout


def test_the_exact_gate_compares_counts_not_a_rounded_rate():
    """Review [P2] on #130: 20000/20001 rounds to 1.0 at four places, so an
    `== 1.0` gate on `match_rate` would pass a run with a real mismatch."""
    near_miss = 20_000 / 20_001
    assert round(near_miss, 4) == 1.0  # what the display value shows
    assert near_miss < 1.0  # what is true

    source = (REPO / "bench" / "parity_tp.py").read_text()
    assert 'entry["exact_match"] == entry["prompts"]' in source
    assert 'worst == 1.0' not in source


def test_the_result_carries_the_counts_the_gate_uses(tmp_path):
    out = tmp_path / "toy.json"
    result = _run("--tp", "1,2", "--num-prompts", "5", "--out", str(out))
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    for entry in payload["parity"].values():
        assert entry["prompts"] == 5
        assert entry["exact_match"] == 5


def _fake_checkpoint(root: Path) -> Path:
    """A checkpoint-shaped directory: one safetensors shard plus the config and
    tokenizer files provenance is derived from."""
    root.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    ).encode()
    (root / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * 16
    )
    (root / "config.json").write_text(json.dumps({"_name_or_path": "acme/toy-1b"}))
    (root / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 8}))
    return root


def test_provenance_identifies_the_checkpoint_by_content_not_path(tmp_path):
    """Review [P2] on #130: the result recorded only a machine-local
    `model_path`, which identifies nothing once the run leaves the box. G2 §8
    wants the checkpoint, tokenizer and running commit recorded instead."""
    sys.path.insert(0, str(REPO / "bench"))
    try:
        import parity_tp
    finally:
        sys.path.pop(0)

    first = _fake_checkpoint(tmp_path / "a")
    provenance = parity_tp._checkpoint_provenance(str(first))

    assert provenance["name"] == "acme/toy-1b"
    assert provenance["tokenizer_sha256"]["tokenizer_config.json"]
    assert provenance["metadata_sha256"]["config.json"]
    assert len(provenance["weights_sha256"]) == 64

    # the same bytes under a different path are the same checkpoint
    same = _fake_checkpoint(tmp_path / "b")
    assert parity_tp._checkpoint_provenance(str(same))["weights_sha256"] == (
        provenance["weights_sha256"]
    )

    # different weights are a different checkpoint even at the same path
    (first / "model.safetensors").write_bytes(
        (first / "model.safetensors").read_bytes() + b"\1"
    )
    assert parity_tp._checkpoint_provenance(str(first))["weights_sha256"] != (
        provenance["weights_sha256"]
    )


def test_a_checkpoint_without_weights_is_rejected(tmp_path):
    """Silently writing evidence with no checkpoint identity is what the review
    objected to, so the harness refuses rather than recording a bare path."""
    sys.path.insert(0, str(REPO / "bench"))
    try:
        import parity_tp
    finally:
        sys.path.pop(0)

    empty = tmp_path / "empty"
    empty.mkdir()
    for bad in (str(empty), str(tmp_path / "missing")):
        try:
            parity_tp._checkpoint_provenance(bad)
        except SystemExit as exit_:
            assert "provenance" in str(exit_) or "not a directory" in str(exit_)
        else:
            raise AssertionError(f"expected a refusal for {bad}")


def test_toy_results_record_the_running_commit(tmp_path):
    """Even the toy harness has to say which code produced the numbers."""
    out = tmp_path / "result.json"
    result = _run(
        "--tp", "1,2", "--num-prompts", "1", "--max-new-tokens", "2",
        "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    code = json.loads(out.read_text())["config"]["code"]
    assert code["commit"] and len(code["commit"]) == 40
    assert code["dirty"] in (True, False)
    # untracked scratch dirs must not be reported as modified code
    assert isinstance(code["untracked_files"], int)


def _parity_tp():
    sys.path.insert(0, str(REPO / "bench"))
    try:
        import parity_tp

        return parity_tp
    finally:
        sys.path.pop(0)


def test_equal_aggregates_are_not_sequence_equality():
    """Review [P1] on #145: the harness compared each overlap mode against its
    own TP1 base and dropped the outputs, then the PR concluded "ON reproduces
    OFF exactly" from two aggregate rows agreeing.

    These two runs diverge on DIFFERENT prompts at the same depth, so every
    aggregate the old result recorded — exact_match, tokens, token_match_rate,
    median_first_divergence — is identical. Only comparing the outputs sees it.
    """
    parity_tp = _parity_tp()

    base = {"p0": (1, 2, 3), "p1": (4, 5, 6)}
    off = {"p0": (1, 9, 9), "p1": (4, 5, 6)}   # p0 diverges at position 1
    on = {"p0": (1, 2, 3), "p1": (4, 9, 9)}    # p1 diverges at position 1

    against_base_off = parity_tp._compare(base, off)
    against_base_on = parity_tp._compare(base, on)
    for field in ("exact_match", "prompts", "tokens", "token_match_rate",
                  "median_first_divergence"):
        assert against_base_off[field] == against_base_on[field], field

    # ...and yet the two modes did not produce the same output
    direct = parity_tp._compare(off, on)
    assert direct["exact_match"] == 0
    assert direct["first_mismatch"]["request_id"] == "p0"
    assert direct["first_mismatch"]["position"] == 1


def test_identical_runs_compare_exact():
    parity_tp = _parity_tp()
    same = {"p0": (1, 2, 3), "p1": (4, 5, 6)}
    result = parity_tp._compare(same, dict(same))
    assert result["exact_match"] == result["prompts"] == 2
    assert result["first_mismatch"] is None
    assert result["token_match_rate"] == 1.0


def test_the_sweep_records_and_gates_the_overlap_comparison(tmp_path):
    """The ON-vs-OFF comparison must reach the evidence file, not just stdout."""
    out = tmp_path / "result.json"
    result = _run(
        "--tp", "1,2", "--num-prompts", "2", "--max-new-tokens", "2",
        "--out", str(out),
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    equality = payload["overlap_equality"]
    assert set(equality) == {"tp1_overlap_on_vs_off", "tp2_overlap_on_vs_off"}
    for entry in equality.values():
        assert entry["exact_match"] == entry["prompts"]
        assert entry["first_mismatch"] is None
    assert "overlap ON vs OFF" in result.stdout
