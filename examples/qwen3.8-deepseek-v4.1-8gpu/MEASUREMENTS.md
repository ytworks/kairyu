# V4.1 ensemble evidence

Status: **CPU implementation validation only; GPU validation pending.**

The user deferred all GPU testing on 2026-09-12 because the hardware is in use.
No new GPU inference, service startup/shutdown, performance matrix, model quality
evaluation or deployment restart was performed for this example.

## What is established

Local validation on 2026-09-12: **202 tests passed** in one invocation:

```sh
python -m pytest tests/unit/test_v41_ensemble_*.py \
  tests/unit/test_tiered_frontier_examplectl.py \
  tests/unit/test_deepseek_v41_example.py \
  tests/unit/test_replica_examplectl.py --no-cov -q
ruff check .
python scripts/check_progress_size.py
```

Docker Compose `config --quiet` passed using the lifecycle-generated environment
without starting containers. Configuration digest agreement and verification
command listing passed. An independent review's stale-deployment provenance
finding was fixed and re-reviewed; no actionable P1/P2 remained in that review.
The full repository test suite is left to CI; the tests above are the selected
local regression set. These are CPU/static checks only.

- The fixed existing V4.1 image and checkpoint source were inspected read-only.
  Exact image, source paths/hashes and constraints are in [L1-NOTES.md](L1-NOTES.md)
  and `example.json`. This is provenance and a static feasibility assessment.
- CPU contracts exercise deployment loading and Compose rendering; the real
  Conductor's effort/dependency/image/repair behavior with scripted engines;
  native OpenAI-compatible tool/image forwarding with an HTTP mock transport;
  role-specific mode/schema/budget middleware; lifecycle isolation and runtime
  attestation; stage coverage and measurement provenance.
- Requirement is fixed DeepSeek high, while Qwen retains its existing effort
  mapping. The removed image-description stage is not part of the new DAG.

## What remains unmeasured

TP2×DP3/EP6 initialization, memory fit, collective/kernel correctness, actual
native tokenizer/schema/thinking-budget enforcement, image understanding, tool
behavior on generated model outputs, restart/cancellation/co-residency, long
context, concurrency, latency and throughput all remain open.

The sibling TP8 V4.1 result and PR #595's Qwen Requirement measurements are **not**
measurements of this example. No old outputs, throughput numbers or baseline
fallbacks were copied here. The 1M DeepSeek and 256K Qwen settings are candidate
limits, not verified usable capacity for this combined deployment.

## Next evidence record

Follow [README.md](README.md#deferred-gpu-validation-execution-order). Record
checkout SHA, exact runtime image IDs, checkpoint attestation, configuration
hash, request/response/trace artifacts, first failures and explicit skipped
gates. Preserve protocol results separately from model/task quality and human
semantic review. Mark only completed gates as passed; a completed implementation
or a healthy process alone is insufficient to close the remaining gates.
