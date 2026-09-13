# qwen3.8-deepseek-v4.1-8gpu evidence

Status: **CPU contracts only; no GPU evidence yet.**

Nothing in this file establishes that the 6-GPU DeepSeek topology starts,
produces finite output, fits its memory, or meets the performance gate. The
sibling examples' measurements (`deepseek-v4.1-flash-8gpu` on TP8,
`qwen3.8-deepseek-v4-8gpu` on V4) are not evidence for this topology and are
not reused here.

## CPU evidence

- `tests/unit/test_v41_tiered_examplectl.py` (see the PR checks): the served
  YAML loads through the real DSL loader and deployment loader, matches
  `example.json`, and the shipped DAG runs end-to-end on the real Conductor
  with scripted engines (ordering and inputs of every role, image forwarding,
  headless tool turns, FAIL → refine → PASS, exhaustion, inconclusive
  re-audit); the verification row validator and TTFT gate arithmetic are
  exercised on synthetic rows.
- `ruff check .` and the repository unit suite pass on the PR head.

## L1 selection procedure (to be executed on the GPU host)

Candidates are measured in order with `verify.sh native`; the first candidate
that passes every gate is adopted and recorded in `example.json`,
`kairyu.yaml`, and `compose.yaml`. Failures are kept here verbatim.

| # | Topology | Engram | `gpu-memory-utilization` | `max-num-batched-tokens` | Status |
|---|---|---|---|---|---|
| 1 | TP2 × attention-DP3, EP6 | GPU-resident | 0.95 | 8192 | **FAILED** 2026-09-13 17:22 UTC: every worker loads 83.92 GiB of weights per GPU (36–38 s), then all three DP engines raise `ValueError: No available memory for the cache blocks` after profiling (0.95 is equivalent to 0.9429 with CUDA-graph memory profiling); the container never became healthy. Evidence: `verification-results/20260913T172300Z-l1-candidate1/{worker.log,launch.log,summary.txt}` |
| 2 | TP2 × attention-DP3, EP6 | CPU offload | 0.90 | 16384 | **FAILED** 2026-09-13 17:29 UTC: starts (weights 52.62 GiB/GPU, two 15.74 GiB Engram tables per rank in pinned host memory, 21.42 GiB KV/GPU = 8,527,200 tokens per DP engine, 8.13× 1M concurrency), gateway readiness passes, but the **first real request** kills the service: worker DP2/TP0 raises `CUDA error: an illegal memory access` inside a Triton kernel and `NCCL error` in `sp_reduce_scatter` (the TP sequence-parallel path), all engines die. Kairyu published the 27-token Qwen head alone with `finish_reason: stop` while every DeepSeek stage traced `failed` — client-level success is not evidence. Evidence: `verification-results/20260913T172900Z-l1-candidate2/{worker.log,launch-1.log,launch-2.log,first-request.json,first-request-result/,summary.txt}` |
| 3 | TP1 × attention-DP6, EP6 | CPU offload only (GPU-resident is arithmetically infeasible at TP1: 2 × 15.74 GiB tables plus ≥ 52.6 GiB of weights per GPU) | 0.90 | 16384 | running |

Constraints established by source inspection (closed PR #598 notes, checkpoint
`config.json`): TP6 is invalid (64 attention heads and 8 output groups are
not divisible by 6); DSpark is off (its 128 draft experts do not divide
EP6); TP2 × PP3 is not viable (shared indexer / compressed-KV caches across
pipeline stages).

Known risk: on the unpatched pinned image, the closed PR #598 campaign
observed NaN log-probabilities and corrupt text with candidate 2 (masked
sparse-KV gathers reading slot 0). Per the approved plan, if every candidate
fails the finite-output gate on the unpatched image, this example stops and
reports; no patched image or new patch layer is used.

Gates per candidate (`verify.sh native`): startup log excerpt (weights, KV
capacity), idle memory snapshot, rendered thinking/effort encodings via
`/tokenize` (75 default, 50/75/100, chat mode), 12 concurrent probes per
mode on all DP ranks with finite log-probabilities and exact answers, tool
call, image, native cancellation (running/waiting gauges return to zero),
fixed-256-token matrix c1/8/16/32 × 32 with memory peaks, completed-answer
matrix c1/8/16/32 × 32 for the generic and coding datasets at the default
(high) effort, and a service restart.

## Public gates (to be executed)

| Gate | Command | Result |
|---|---|---|
| Judged product, generic | `verify.sh serving-auto-max` | not run |
| Judged product, coding | `verify.sh serving-auto-max-coding` | not run |
| Forced ensemble, generic + coding | `verify.sh serving-ensemble` | not run |
| Tool calling (900 s turn) on both models | `verify.sh tool-calling` | not run |
| Images on both models (headless JSON proves DeepSeek saw the image) | `verify.sh vision` | not run |
| Public cancellation (early and during the withheld remainder) | `verify.sh cancellation` | not run |
| Normal restart | `verify.sh restart` | not run |
| Issue #599 saved request | `verify.sh issue-599` | not run (request file location pending) |
| Long inputs (Qwen boundary ensemble; DeepSeek-direct 32K/256K/~1M) | `verify.sh long-input` | not run |
| Chat UI browser gate (shared script, tiered phase) | `verify.sh browser` | not run |

Each serving row writes `row-serving.json` (per-request timing, finish
reason, usage, content, raw trace), `routes.json`, `stages.json`,
`placement.json`, `problems.json`, `memory.json`; the run writes
`ttft-gate.json` and `run.json` (served-config SHA-256, image IDs, commands,
mounted-file hashes, git commit).
