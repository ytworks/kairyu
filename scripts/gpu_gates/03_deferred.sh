#!/usr/bin/env bash
# Runbook §3: remaining GPU-deferred seams (graphs, streams).
source "$(dirname "$0")/_lib.sh"
preflight --min-gpus 4 --min-compute-capability 10.0 \
  --require-nccl --require-peer-access \
  --require-module flashinfer --require-module transformers \
  --require-module triton --require-executable ninja \
  --require-executable nvidia-smi
pytest_no_skip -m gpu tests/gpu tests/unit/test_pd_factory.py -v
