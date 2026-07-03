"""Opt-in networked parity (m12 D6 secondary): `pytest -m hf_hub`."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.hf_hub
def test_qwen25_0_5b_greedy_parity():
    script = Path(__file__).parents[2] / "scripts" / "parity_real_model.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stdout + result.stderr
