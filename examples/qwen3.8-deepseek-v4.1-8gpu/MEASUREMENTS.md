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

## Public gates on candidate 6, served config A (overlay image; specs at commits `4386969c`–`34104902`, before the audit amendment)

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

### `serving-auto-max-coding` — judged product, coding 2.9K-token prompts (run `20260914T031731Z`, rows PASS, TTFT gate not applicable)

| c | ok | routes (judge fallbacks) | judge p50 | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | audit | Qwen placement | gate: product vs DeepSeek-direct p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | qwen_think_medium 32 (0) | 288 ms | 32,792 / 224,772 ms | 48.8 / 238.8 s | 1,805 s | 45.0 | 81,477 | — | 49 / 15 (serial, not gated) | not applicable (baseline invalid: 31/32) |
| 8 | 32/32 | qwen_think_medium 32 (0) | 366 ms | 24,772 / 319,235 ms | 38.5 / 324.5 s | 456 s | 203.8 | 93,155 | — | 26 / 38 | not applicable (baseline invalid: 31/32) |
| 16 | 32/32 | qwen_think_medium 32 (0) | 895 ms | 37,401 / 335,532 ms | 54.2 / 352.0 s | 367 s | 304.5 | 111,798 | — | 36 / 28 | not applicable (58,156 ms baseline valid) |
| 32 | 32/32 | qwen_think_medium 32 (0) | 1,726 ms | 36,010 / 411,919 ms | 52.6 / 430.6 s | 431 s | 198.6 | 85,707 | — | 32 / 32 | not applicable (82,309 ms baseline valid) |

- The judge chose `QWEN_THINK` for all 128 coding prompts with no timeout
  fallback, so no ensemble ran in this matrix and the TTFT gate has nothing
  to compare: `ttft_gated_profiles` (inherited from the V4 tiered example)
  covers `primary`, `qwen_direct` and `deepseek_direct` only, because a
  thinking route's first visible text follows the model's own thinking by
  design. For reference only, the thinking route's first-visible-content
  p50 is 0.64× (c16) and 0.44× (c32) of the paired DeepSeek-direct value.
- Every public answer ended with `stop`; no stage reached its cap (largest
  Qwen answer 17,696 tokens at c32, under the route's user cap 65,536).
  The p99 tail (225–412 s) is single prompts on which Qwen thinks for
  10–18K tokens; the c1 wall of 1,805 s is those tails run serially.
- Paired DeepSeek-direct (default = high effort, `max_tokens 65536`):
  c1 and c8 each finished 31/32 — one sample per row (index 1, index 9,
  the same RLE-decoder task) spent the full 65,536 tokens thinking
  (`length`, ≈201K reasoning characters, no visible text, 946 s / 996 s),
  so those two denominators are recorded as invalid rather than used;
  their completed-31 first-visible-content p50 is 35,977 ms / 39,790 ms.
  c16 and c32 completed 32/32 (`stop`): 58,156 ms / 82,309 ms, completion
  p50 65.7 s / 87.5 s, 500.9 / 483.9 output tok/s.
- Memory peaks across the matrix: DeepSeek GPUs 0–3 93,151 MiB, GPUs 4–5
  93,733 MiB; Qwen GPUs 6–7 95,601–95,603 MiB; DeepSeek container RSS
  ≈233 GiB, Qwen containers ≈6.3 GiB each, gateway ≈0.31 GiB.
- Artifacts: `verification-results/20260914T031731Z-serving-auto-max-coding/serving-auto-max-coding/{coding-c*,deepseek-direct-c*}/`, `ttft-gate.json`.

### `serving-ensemble` — forced ensemble, generic c1 (run `20260914T051628Z`, gate PASS; chain stopped after this row by owner decision)

| c | ok | routes | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | audit (first attempt → outcome) | Qwen placement | gate: product vs DeepSeek-direct p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | primary 32 | 2,031 / 2,059 ms | 520.8 / 746.1 s | 17,069 s | 4.8 | 1,443,246 | PASS 13; FAIL 19 → PASS after 1 refinement 10, PASS after 2 refinements 1, exhausted after three FAILs (last attempt published) 8 | 80 / 80 | 2,031 vs 18,239 ms → PASS (0.11×); baseline 32/32 `stop` |

- All 11 roles `success` on every request; every public answer `stop`; no
  stage at its cap except the head's designed 256 (8 requests). Per-stage
  duration p50 / p99 (ms): head 6,259 / 7,686; requirements 53,692 / 99,576;
  independent 26,983 / 53,013; policies 34,809 / 62,450; answer_1..4
  ≈104,700–118,000 / 225,000–310,000 (Qwen medium, 4.1–4.9K tokens p50);
  synthesis 37,160 / 85,608; final 25,232 / 95,882 (60 attempts); audit
  79,633 / 188,630 (60 attempts). Kairyu-side queue wait 0–1 ms.
- Memory peaks: DeepSeek GPUs 93,151–93,733 MiB, Qwen GPUs 95,601–95,603
  MiB; DeepSeek container RSS ≈233 GiB.
- The warm-up (4 requests at c4) completed 4/4 `stop`, first visible content
  ≈8.0 s, completion 468–891 s.
- Artifacts: `verification-results/20260914T051628Z-serving-ensemble/serving-ensemble/generic/{generic-c1,deepseek-direct-c1}/`,
  `ttft-gate.json` (no top-level `run.json`: the chain was stopped after
  this row; the served config is the one attested by run
  `20260914T031731Z`, same container starts and file hashes).

## Audit diagnosis and prompt amendment (V41T-D2 amendment, commit `e8d7ba8e`)

The public API exposes no audit text, so three of the eight exhausted requests
(cases 2, 7, 12 of the row above) were replayed once on the same served
config through `kairyu-ensemble-max` with the intermediate outputs read from
`reasoning_content` (`gate-logs-run5/diag-audit/`). Findings:

- The benchmark row label `Run <run id>, case N` was extracted by
  `requirements` as minimum requirement R1 "execute the run". The audit then
  FAILed answers that said no run was performed ("N/A without a fixture";
  "Status: NOT_EXECUTED_EXTERNALLY") and PASSed answers that invented an
  execution result ("The case-2 run is accepted as PASS, verified, low risk";
  "Run report … executed … Dry-run result code: PASS"). In the row above 25
  of 32 published answers carry such an invented result; the single answer
  that stated plainly that nothing was executed was FAILed three times.
- The published length was not the cause: visible answers were 1,440–2,180
  characters (head ≈200 tokens + remainder ≈200–300 tokens) against the
  requested "approximately 256 output tokens"; the large per-attempt
  `completion_tokens` (900–6,700) are the final role's private thinking.
- Amendment: the audit accepts a run/execute/test/measure/verify claim as
  evidence only with the matching tool call and result in the conversation
  and otherwise FAILs it as fabrication; a requirement demanding an action
  the turn cannot perform is satisfied by an answer that says so plainly
  and delivers everything else; `synthesis` and `final` (streamed and
  headless) carry the same rule; the verification datasets label the row
  identity as "identifier only, not an instruction". Kairyu unchanged; CPU
  tests 36/36.
- Replay after the amendment (same three prompts, old label wording,
  gateway restarted 10:43 UTC; `gate-logs-run5/diag-audit-fixed/`): all
  three PASSed on the first audit (316 / 405 / 488 s), each stating that the
  run cannot be executed in this turn and inventing no result.
- Consequence: every public gate is re-run on the revised served config
  (config B, below). The config-A rows above stay as evidence.

## Public gates on served config B (specs at commit `e8d7ba8e`)

- Run 6 (gateway from the example branch, restarted 10:43 UTC): aborted by
  the gateway deadlock (section below).
- Run 7 (same gateway restarted 15:33 UTC, host watchdog): stopped after
  about an hour by owner decision so the fix could be served instead.
- From the run-8 forced-ensemble generic c8 row on, `benchmark.py` also
  stores each sample's streamed `reasoning_content` (the exposed
  intermediate outputs, including every audit verdict text) as
  `reasoning`, so audit outcomes can be read from the artifacts instead of
  replays; the measurement itself is unchanged.
- Run 8 (from 15:46 UTC): the `kairyu` container was rebuilt from the
  deadlock fix (PR #603, commit `e22b545e`, which contains this branch
  plus the five re-entrant locks) and recreated in place; the L1 engines
  kept running. The verification tool runs from that checkout, so its
  `run.json` records `git_commit` `e22b545e`; the example files it serves
  and executes are byte-identical to `e8d7ba8e`. The watchdog stays armed
  to record any recurrence.


### Run 8 — `serving-auto-max`, judged product, generic 8K-token prompts (run `20260914T154637Z`, PASS; served `git_commit` `e22b545e`, served-config SHA `b5efddba…`)

| c | ok | routes (judge fallbacks) | judge p50 | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | audit | Qwen placement | gate: product vs DeepSeek-direct p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | qwen_direct 32 (0) | 239 ms | 2,294 / 2,350 ms | 8.2 / 11.9 s | 283 s | 33.4 | 9,588 | — | 47 / 17 (serial, not gated) | 2,294 vs 17,219 ms → PASS |
| 8 | 32/32 | qwen_direct 31, primary 1 (1) | 321 ms | 6,432 / 13,919 ms | 27.4 / 554.8 s | 582 s | 19.2 | 51,623 | exhausted 1 (three FAILs, last attempt published) | 34 / 34 | 12,293 (primary) vs 22,902 ms → PASS |
| 16 | 32/32 | qwen_direct 27, primary 5 (5) | 1,258 ms | 18,617 / 38,296 ms | 56.0 / 733.2 s | 803 s | 27.3 | 227,523 | PASS 5 (3 first attempt, 1 after one refinement, 1 after two) | 41 / 43 | 25,487 (primary) vs 31,670 ms → PASS |
| 32 | 32/32 | qwen_direct 32 (0) | 2,369 ms | 51,974 / 74,251 ms | 78.6 / 86.2 s | 86 s | 115.5 | 10,091 | — | 32 / 32 | 51,974 vs 43,871 ms → PASS (1.18×) |

- No gateway hang across the matrix (the watchdog recorded nothing); every
  paired DeepSeek-direct row completed 32/32 with `stop`.
- All six judge-fallback ensembles completed all 11 roles; one of the six
  exhausted its two refinements (published after three FAILs). The audit
  text is not exposed by the public API; if the forced-ensemble rows show
  the same rate, a few requests will be replayed with intermediate outputs
  after the chain.
- Artifacts: `verification-results/20260914T154637Z-serving-auto-max/`.

### Run 8 — `serving-auto-max-coding`, judged product, coding 2.9K-token prompts (run `20260914T163330Z`, rows PASS, TTFT gate not applicable)

| c | ok | routes (judge fallbacks) | judge p50 | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | Qwen placement | DeepSeek-direct denominator (32/32 `stop`) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | qwen_think_medium 32 (0) | 292 ms | 19,133 / 194,713 ms | 35.2 / 213.9 s | 1,459 s | 44.9 | 65,735 | 49 / 15 (serial, not gated) | 32,678 ms (thinking route reference: 0.59×) |
| 8 | 32/32 | qwen_think_medium 32 (0) | 365 ms | 26,589 / 236,056 ms | 39.7 / 257.4 s | 408 s | 166.8 | 68,193 | 34 / 30 | 59,910 ms (0.44×) |
| 16 | 32/32 | qwen_think_medium 32 (0) | 367 ms | 33,914 / 99,360 ms | 47.1 / 125.3 s | 152 s | 401.1 | 61,223 | 32 / 32 | 51,286 ms (0.66×) |
| 32 | 32/32 | qwen_think_medium 32 (0) | 2,062 ms | 50,255 / 252,136 ms | 66.2 / 266.2 s | 266 s | 273.6 | 73,028 | 32 / 32 | 84,721 ms (0.59×) |

- As on config A, the judge chose `QWEN_THINK` for all 128 coding prompts
  with no timeout fallback, so no gated route served a request and the
  gate has nothing to compare; every public answer ended with `stop` and
  no stage reached a cap (largest Qwen answer 11,371 tokens). This time
  all four DeepSeek-direct denominators completed 32/32 with `stop`
  (the config-A thinking runaway on the RLE task did not recur; sampling
  at temperature 1.0 is not deterministic across runs).
- Artifacts: `verification-results/20260914T163330Z-serving-auto-max-coding/`.

### Gate chain run 6 (`20260914T105208Z-serving-auto-max`) — aborted by a Kairyu gateway deadlock

| c | ok | routes (judge fallbacks) | judge p50 | first visible content p50 / p99 | completion p50 / p99 | wall | public tok/s | internal output tokens | audit | Qwen placement | gate: product vs DeepSeek-direct p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32/32 | qwen_direct 32 (0) | 241 ms | 2,299 / 2,349 ms | 8.9 / 11.2 s | 291 s | 33.7 | 9,936 | — | 46 / 18 (serial, not gated) | 2,299 vs 14,705 ms → PASS |
| 8 | 32/32 | qwen_direct 29, primary 3 (3) | 320 ms | 6,458 / 13,926 ms | 27.3 / 895.5 s | 931 s | 21.1 | 148,388 | PASS 2, exhausted 1 | 38 / 38 | 12,187 (primary p50) vs 21,322 ms → PASS |
| 16 | 2/32 | qwen_direct 2, 30 never answered | 1,106 ms | — | — | 14,587 s (client timeout) | — | — | — | 18 / 18 | row FAIL (30 requests hung) |

- At 11:26:53 UTC, after the first two c16 answers, the gateway stopped
  serving everything — chat completions, `/readyz`, the compose health
  probe — while its process stayed alive (state S, 205 threads, 0 % CPU,
  291 MB RSS, 51 open files, listen socket up). The 30 remaining requests
  hung until the client's 14,400 s timeout; the chain then moved on and was
  stopped by hand at 15:31 UTC. The engines were healthy throughout (the
  paired DeepSeek-direct rows and the Qwen replicas kept answering).
- Cause (py-spy dump of the live process, `gate-logs-run6/gateway-hang-20260914T1126Z-pyspy.txt`):
  the event-loop thread is blocked in
  `kairyu/engine/openai_backend.py` — `_peek_prepared_payload` holds
  `self._prepared_payloads_lock` (a non-re-entrant `threading.Lock`) when a
  garbage-collection pass fires the weakref callback `discard` of a dead
  `GenerationRequest`, and `discard` tries to take the same lock on the
  same thread: a self-deadlock that freezes the whole gateway. The module-
  level `_SHARED_PREPARED_PAYLOADS_LOCK` has the same pattern. This is a
  `main` framework defect independent of this example (the code has been in
  place since `1fff59db`, 2026-08-06); it depends on GC timing, which is why
  the identical c16 row passed on config A.
- Handling: the gateway was restarted (15:33 UTC; served config unchanged),
  a host-side watchdog (`gate-logs-run7/gateway_watchdog.sh`) now dumps the
  stacks and restarts the gateway if `/readyz` fails for two minutes, so a
  recurrence costs minutes instead of hours and is recorded, and every
  public gate is re-run as chain run 7. The framework fix (re-entrant lock
  or a callback that never blocks on the lock) needs the owner's
  authorization as a separate change; nothing in `kairyu/` is modified by
  this example.
- Artifacts: `verification-results/20260914T105208Z-serving-auto-max/serving-auto-max/{generic-c*,deepseek-direct-c*}/`, `ttft-gate.json`, `gate-logs-run6/`.

## Public gate status

| Gate | Command | Result |
|---|---|---|
| Judged product, generic | `verify.sh serving-auto-max` | PASS on config B, run 8 (`20260914T154637Z`); config A PASS (`20260914T022900Z`) |
| Judged product, coding | `verify.sh serving-auto-max-coding` | rows PASS, gate not applicable, on config B run 8 (`20260914T163330Z`) and config A (`20260914T031731Z`) |
| Forced ensemble, generic + coding | `verify.sh serving-ensemble` | config A generic c1 PASS (`20260914T051628Z`, stopped for diagnosis); config B re-run in progress |
| Tool calling (900 s turn) on both models | `verify.sh tool-calling` | not run |
| Images on both models (headless JSON proves DeepSeek saw the image) | `verify.sh vision` | not run |
| Public cancellation (early and during the withheld remainder) | `verify.sh cancellation` | not run |
| Normal restart | `verify.sh restart` | not run |
| Issue #599 saved request | `verify.sh issue-599` | queued in run 6 (request file found: `kairyu-bench/results/deepswe-full-3w-20260913-r1/progress-detail/api-failure-investigation/request.json`) |
| Long inputs (Qwen boundary ensemble; DeepSeek-direct 32K/256K/~1M) | `verify.sh long-input` | not run |
| Chat UI browser gate (shared script, tiered phase) | `verify.sh browser` | not run |

Each serving row writes `row-serving.json` (per-request timing, finish
reason, usage, content, raw trace), `routes.json`, `stages.json`,
`placement.json`, `problems.json`, `memory.json`; the run writes
`ttft-gate.json` and `run.json` (served-config SHA-256, image IDs, commands,
mounted-file hashes, git commit).
