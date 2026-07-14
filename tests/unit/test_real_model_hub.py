"""Opt-in networked parity (m12 D6 secondary): `pytest -m hf_hub`."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.parity_real_model import compare_greedy_tokens


@pytest.mark.parametrize(
    ("ours", "reference", "expected", "diagnostic"),
    [
        pytest.param([], [], True, "exact token match", id="both-empty"),
        pytest.param([], [1], False, "length mismatch", id="empty-vs-nonempty"),
        pytest.param([1], [], False, "length mismatch", id="nonempty-vs-empty"),
        pytest.param(
            [1], [1, 2, 3], False, "length mismatch", id="truncated-prefix-old-false-success"
        ),
        pytest.param([1, 2], [1, 3], False, "token mismatch", id="token-mismatch"),
        pytest.param(
            [10, 2], [10, 2, 3], False, "length mismatch", id="early-eos"
        ),
        pytest.param([1, 2, 3], [1, 2, 3], True, "exact token match", id="exact"),
    ],
)
def test_compare_greedy_tokens_requires_complete_exact_match(
    ours, reference, expected, diagnostic
):
    matched, detail = compare_greedy_tokens(ours, reference)

    assert matched is expected
    assert diagnostic in detail


@pytest.mark.hf_hub
def test_qwen25_0_5b_greedy_parity():
    script = Path(__file__).parents[2] / "scripts" / "parity_real_model.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
