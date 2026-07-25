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
