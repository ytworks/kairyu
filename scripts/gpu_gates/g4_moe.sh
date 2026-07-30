#!/usr/bin/env bash
# Goal G4: CPU MoE/MLA/MTP parity plus real two-GPU EP parity.
source "$(dirname "$0")/_lib.sh"
preflight --require-module transformers
pytest_no_skip tests/unit/test_moe_mla_parity.py -v --no-cov
pytest_no_skip tests/dist/test_distributed.py::test_ep2_moe_block_matches_single_process -v
pytest_no_skip tests/unit/test_eagle_mtp.py -v --no-cov
preflight --min-gpus 2 --require-nccl
pytest_no_skip -m gpu tests/gpu/test_moe_parallel_nccl.py -v --no-cov
