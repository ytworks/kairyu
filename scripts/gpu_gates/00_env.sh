#!/usr/bin/env bash
# Runbook §0: environment audit + env record.
source "$(dirname "$0")/_lib.sh"
run nvidia-smi
run python -c "import torch; assert torch.cuda.is_available()"
run uv sync --frozen --extra gpu --extra hf --extra fleet
run uv run python -c "from kairyu.engine.core.hw_profile import probe, write_env_record, EnvRecord; import torch; p=probe(); print(p)"
