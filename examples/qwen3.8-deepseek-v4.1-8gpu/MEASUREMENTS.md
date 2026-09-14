# qwen3.8-deepseek-v4.1-8gpu evidence

Status: **L1 native gates PASS (27/27) on this example's overlay; public gates in progress.** The sibling's unpatched SM120 overlay corrupts output on every six-GPU topology (candidates 1–5 below); this example now ships PR #598's masked-KV / top-p overlay recipe (`vllm-sm120.Dockerfile`, `patch_masked_kv.py`, `patch_top_p.py`) and `run.sh` builds it on any host.

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
| 3 | TP1 × attention-DP6, EP6 | CPU offload only (GPU-resident is arithmetically infeasible at TP1: 2 × 15.74 GiB tables plus ≥ 52.6 GiB of weights per GPU) | 0.90 | 16384 | **FAILED** 2026-09-13 17:36 UTC: starts (weights 57.37 GiB/GPU, 9.68 GiB KV/GPU = 3,855,545 tokens per DP engine), readiness passes, the rendered prompts are correct (`Reasoning Effort: 75` default, 50/75/100 by effort, chat mode closes the span), but only the first two native probes answer `323` with finite log-probabilities; every later probe returns NaN log-probabilities (HTTP 400 `Out of range float values … nan`) and, without logprobs, empty visible text after 1,000–4,096 generated tokens (3/6 hit the cap). No worker crash. Evidence: `verification-results/20260913T173600Z-l1-candidate3/{probes.json,worker.log,launch.log,summary.txt}` |
| 4 | TP2 × attention-DP3, EP6 (added: tests whether the corruption depends on Engram offload) | GPU-resident, activation footprint reduced (`max-num-batched-tokens 2048`, `max-num-seqs 8`) | 0.97 | 2048 | **FAILED** 2026-09-13 17:48 UTC: starts (weights 83.79 GiB/GPU, 2.97 GiB KV/GPU = 1,722,859 tokens per DP engine), readiness passes, but 2/24 native probes are correct; 16/18 thinking probes return NaN log-probabilities and all 6 chat-mode probes return garbage text (e.g. `##归当归# Flexible…`). Evidence: `verification-results/20260913T174800Z-l1-candidate4/{probes.json,worker.log,launch.log,summary.txt}` |
| 5 | TP1 × attention-DP6, EP6, `--attention-backend FLASHMLA_SPARSE_DSV41` (config-only alternative backend) | CPU offload | 0.90 | 16384 | **FAILED** 2026-09-13 23:30 UTC: every worker dies in CUDA-graph memory profiling with `ValueError: No common block size for 64.` — the FlashMLA sparse backend does not support the 64-token pages SM120 requires. Evidence: `verification-results/20260913T233000Z-l1-candidate5/{worker.log,launch.log,summary.txt}` |
| 6 | TP2 × attention-DP3, EP6 on **this example's overlay** (PR #598 recipe: masked sparse-KV gathers read a zero row; seeded top-p keeps the forced thinking terminator) | CPU offload | 0.90 | 16384 | **ADOPTED** — `verify.sh native` PASS 27/27 (below) |

**Owner decision (2026-09-14):** the unpatched sibling overlay cannot serve six
GPUs correctly (candidates 1–5), and PR #598 had already shown that its
masked-KV overlay does. Restricting this example to the sibling's image was
not an owner requirement; it is dropped. This example now owns the overlay
recipe (Dockerfile + two patch scripts, byte-identical to the ones inside the
image PR #598 measured, SHA-256 pinned in `example.json`) and `run.sh up`
builds the parent and child overlays on any host, attesting the image by the
patch-script content rather than by image ID. Kairyu is unchanged.

Constraints established by source inspection (closed PR #598 notes, checkpoint
`config.json`): TP6 is invalid (64 attention heads and 8 output groups are
not divisible by 6); DSpark is off (its 128 draft experts do not divide
EP6); TP2 × PP3 is not viable (shared indexer / compressed-KV caches across
pipeline stages).

Gates per candidate (`verify.sh native`): startup log excerpt (weights, KV
capacity), idle memory snapshot, rendered thinking/effort encodings via
`/tokenize` (75 default, 50/75/100, chat mode), 12 concurrent probes per
mode on all DP ranks with finite log-probabilities and exact answers, tool
call, image, native cancellation (running/waiting gauges return to zero),
fixed-256-token matrix c1/8/16/32 × 32 with memory peaks, completed-answer
matrix c1/8/16/32 × 32 for the generic and coding datasets at the default
(high) effort, and a service restart.

## Native L1 evidence (candidate 6, `verify.sh native`, run `20260913T235125Z-native`, PASS 27/27)

Runtime: image `local/vllm-openai:deepseek-v41-sm120-masked-kv-budget` (ID
`sha256:18dad57d…`, patch scripts attested by SHA-256), vLLM
`0.1.dev20904+g179dd0fa9`, FlashInfer `60b49158` + masked-KV zero row, Qwen
`vllm/vllm-openai:v0.23.0`. Configuration: `--tensor-parallel-size 2
--data-parallel-size 3 --enable-expert-parallel --engram-config
'{"cpu_offload":true}' --gpu-memory-utilization 0.90 --max-num-batched-tokens
16384 --max-num-seqs 32 --max-model-len 1048576 --kv-cache-dtype fp8 --block-size
64 --attention-config '{"indexer_kv_dtype":"mxfp4"}' --moe-backend marlin`, no
DSpark, no `--default-chat-template-kwargs`.

- Startup: weights 52.62 GiB per GPU; two Engram tables of 15.74 GiB per rank in
  pinned host memory; 21.42 GiB KV per GPU = 8,527,200 tokens per DP engine
  (8.13× a 1,048,576-token request); three engines.
- Rendering (`/tokenize` → `/detokenize`): default and `high` render
  `Reasoning Effort: 75`, `low` 50, `max` 100; `enable_thinking=false` renders
  `<｜Assistant｜></think>` with no effort line.
- Probes (12 concurrent per variant across the three DP engines, `17 * 19`,
  `logprobs` + `top_logprobs 3`): default/low/high/max thinking and chat mode
  all return exactly `323`, `finish_reason stop`, every log-probability finite.
  The first-request-only success of the unpatched image (candidates 3–4) does
  not recur.
- Tool call (bash function) and image (solid red PNG → "red") through the
  native endpoint: PASS.
- Native cancellation: stream closed after the first delta with the engine
  gauge at 1; running + waiting returned to 0 after 0.23 s; follow-up answered.
- Fixed 256-token matrix (reasoning tokens count; `ignore_eos`; 32 requests
  per row):

| c | requests | output tok/s | model TTFT p50 (ms) | TPOT p50 (ms) | DeepSeek GPU peak (MiB, GPUs 0–5) |
|---|---|---|---|---|---|
| 1 | 32/32 | 55.1 | 937 | 14.5 | 91,011–91,371 |
| 8 | 32/32 | 203.5 | 5,061 | 19.7 | 93,071–93,171 |
| 16 | 32/32 | 267.7 | 6,994 | 31.7 | 93,071–93,171 |
| 32 | 32/32 | 365.3 | 10,960 | 44.6 | 93,071–93,171 |

- Completed-answer baselines (natural completion, `max_tokens 65536`, default
  = high effort, temperature 1.0 / top_p 0.95; every request `stop` with
  visible content — the denominators the public TTFT gate compares against):

| workload | c | requests | first visible content p50 (ms) | completion p50 (ms) | completion p99 (ms) | output tok/s |
|---|---|---|---|---|---|---|
| generic 8K-token prompt | 1 | 32/32 | 14,747 | 18,347 | 45,117 | 67.1 |
| generic | 8 | 32/32 | 24,744 | 30,540 | 77,985 | 299.1 |
| generic | 16 | 32/32 | 36,679 | 40,814 | 94,663 | 453.6 |
| generic | 32 | 32/32 | 47,909 | 54,235 | 72,366 | 642.0 |
| coding 1.5K-token prompt | 1 | 32/32 | 24,680 | 27,702 | 113,894 | 69.5 |
| coding | 8 | 32/32 | 38,132 | 43,626 | 222,425 | 376.1 |
| coding | 16 | 32/32 | 60,023 | 63,814 | 160,155 | 479.6 |
| coding | 32 | 32/32 | 78,217 | 87,221 | 212,364 | 556.5 |

- Memory: idle GPU 0–5 89.5–89.8 GiB used, GPU 6–7 (Qwen) 88.2 GiB; under load
  GPUs 0–5 peak 93.2 GiB. Host: the DeepSeek container's resident set sums to
  ≈233 GiB (VmRSS over its processes, dominated by the ≈189 GiB of pinned
  Engram tables), Qwen containers ≈6.1 GiB each, gateway ≈0.27 GiB.
- Service restart (`docker compose restart deepseek`): healthy and answering
  again after 170.8 s.

Artifacts: `verification-results/20260913T235125Z-native/{run.json,requests.jsonl,native/*}`
(`native.json`, `rendered-prompts.json`, `startup-excerpt.log`, `memory-idle.json`,
per-row `row-serving.json` + `memory.json`).

## First forced-ensemble request on candidate 6 (`20260913T2348…-l2-smoke-overlay`)

`kairyu-ensemble-max`, one request: all 11 roles traced `success`, audit PASS
on the first attempt, first visible content 464 ms (Qwen head), completion
206 s, public 1,094 tokens (head 27 + final 1,067), orchestration output
16,882 tokens; the four policy answers ran 2 + 2 on the Qwen replicas. The
answer ended with the required literal. The head/remainder seam lacked a
space, which is why the final role now demands a leading blank line.

## Public gates on candidate 6 (served config: overlay image, specs at commit `4386969c`+)

Row validation for every public sample: HTTP 200, `finish_reason` stop/tool_calls,
visible output, trace `status == success` for the published route's final unit
and (primary) for all 11 roles, no stage at its cap except the head's designed
256, Qwen placement from the pool log; the paired DeepSeek-direct row is a valid
denominator only when all 32 requests end with `stop` and visible content.

### `serving-auto-max` — judged product, generic 8K-token prompts (run `20260914T022900Z`, PASS)

| c | ok | routes (judge fallbacks) | judge p50 | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | audit | Qwen placement | gate: product vs DeepSeek-direct p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | qwen_direct 32 (0) | 237 ms | 2,295 / 2,315 ms | 7.9 / 11.6 s | 258 s | 32.3 | 8,465 | — | 49 / 15 (serial, not gated) | 2,295 vs 15,816 ms → PASS |
| 8 | 32/32 | qwen_direct 30, primary 2 (2) | 320 ms | 6,422 / 14,069 ms | 25.8 / 604.5 s | 648 s | 21.1 | 100,087 | PASS 2, FAIL 2 (one sample refined twice) | 36 / 36 | 11,953 (primary p50) vs 22,049 ms → PASS |
| 16 | 32/32 | qwen_direct 25, primary 7 (7) | 1,111 ms | 18,024 / 44,730 ms | 62.1 / 789.7 s | 841 s | 25.1 | 293,550 | PASS 6, FAIL 6, 1 exhausted (last attempt published) | 47 / 45 | 32,057 (qwen_direct p50) vs 28,674 ms → PASS (1.12×) |
| 32 | 32/32 | qwen_direct 32 (0) | 2,235 ms | 65,070 / 82,481 ms | 88.8 / 94.0 s | 94 s | 95.8 | 9,137 | — | 32 / 32 | 65,070 vs 45,023 ms → PASS (1.45×) |

- The judge chose `QWEN` for every generic prompt; every primary run at c8/c16
  came from a judge timeout (5 s) under Qwen load, i.e. Kairyu's designed
  fallback. Those ensembles completed all 11 roles with audits (primary route
  completion p50 541 s at c8, 631 s at c16). Per-stage duration p50 at c16:
  requirements 86 s, independent 44 s, policies 35 s, answer 151 s (medium
  Qwen, ~4.7K tokens), synthesis 31 s, final 26 s, audit 70 s; Kairyu-side
  queue wait is 0 (queueing happens inside vLLM).
- All four paired DeepSeek-direct rows completed 32/32 with `stop`.
- GPU peak under load: DeepSeek GPUs 93.7 GiB, Qwen GPUs 95.6 GiB.
- Artifacts: `verification-results/20260914T022900Z-serving-auto-max/serving-auto-max/{generic-c*,deepseek-direct-c*}/`, `ttft-gate.json`.

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
