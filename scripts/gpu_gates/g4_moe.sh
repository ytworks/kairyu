#!/usr/bin/env bash
# Goal G4 gates: MoE/MLA on GPUs (EP + MTP + quantized experts).
source "$(dirname "$0")/_lib.sh"
run uv run pytest tests/unit/test_moe_mla_parity.py -v --no-cov
run uv run pytest tests/dist/test_distributed.py::test_ep2_moe_block_matches_single_process -v
run uv run pytest tests/unit/test_eagle_mtp.py -v --no-cov
