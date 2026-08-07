# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (master roadmap `docs/roadmap.md`, accepted 2026-07-03): serve a
Fugu-class orchestration product — multi-model auto-routing plus agent
ensemble/synthesis behind one OpenAI-compatible API and a chat UI — from an
on-prem DC of thousands of GPUs across **two first-class hardware profiles**:
NVLink-HBM nodes (8× H100-class, TP-first) and PCIe-GDDR nodes (RTX PRO 6000
Blackwell, DP-first / PP for capacity / EP over RDMA). TTFT/TPOT/goodput must
beat frontier APIs as measured by the committed harness (G6 gate P-C1).

- L1 stays Kairyu's own engine (kernel libraries like FlashInfer are used;
  scheduler, radix KV, spec decode, orchestration are ours). Layering:
  L3 Interface / L2 Orchestration / L1 Engines.
- Target model classes: small dense ~14B (latency tier), mid dense ~70B,
  mid MoE 100–300B, frontier MoE 500B+ (multi-node EP).
- Parallelism/quantization strategy is derived per measured hardware profile,
  never assumed. Details: `docs/goals/g2..g6-*.md`.

## Current Status

Snapshot date: 2026-08-06. Hardware context: all GPU evidence so far is on
8× RTX PRO 6000 Blackwell (SM120), PCIe-only interconnect (P2P 30–37 GB/s);
NVLink-HBM (H100-class) formal gates still need hardware. Evidence lives in
`bench/results/` (see `index.json`); decisions and rationale in `docs/design/`.

### Milestones

| Milestone | Status |
|---|---|
| M1 Orchestration+Interface | Complete: Router/Conductor/MoA, vLLM-compat API, OpenAI server, DSL |
| M2 Core engine | CPU half done; unified EngineLoop + device sampling GPU-validated; NVLink perf gates pending |
| M3 Spec/graphs/P-D | n-gram spec, CUDA-graph serving, intra-node P-D production GPU-validated; EAGLE runtime deferred to G4 |
| M4 Router learning | Implemented CPU-only, design reviewed |
| M5 Intra-node multi-GPU | CPU half done; TP/DP/P-D plumbing live; GPU phase per runbook |
| M6 Inter-node multi-GPU | CPU half done; production stage-sharded PP remains a roadmap item |
| M7 Productionization | CPU half done: serve CLI, gateway, batch, compose smoke; `kairyu validate` preflight |
| M8 Engine CPU core | Complete (amended 2026-08-04) |
| M9 Truthful API | Complete |
| M10a/M10b Fleet base + KV routing | Complete |
| M11 Product surface + tenancy | Complete |
| M12 Dense model zoo | Complete; generation-default contract amended 2026-08-04 |
| M13 Attention backends | Complete; FA3/FA4 added, `auto` stays FlashInfer (measured faster on SM120) |
| M14 Quant compute | GPU-validated: FP8/INT8/AWQ/GPTQ/NVFP4 production-dispatch, fail-closed |
| M15–M18 MoE/MLA, distributed, graphs/drafts, KV transport | Complete |

### Formal gates

- G2 A1: complete — TP1/2 logprob-agreement closure on Llama-3.1-8B
- G2 A2: complete — TP2/4/8 closure on Llama-3.3-70B FP8-dynamic
- G2 A6 (perf vs vLLM): **open** — TP4 ShareGPT 0.466× SLO-goodput HTTP; matrix deferred while gap closes
- G2 A7: closed — >80% KV cache-hit rates on Qwen3-32B TP4/TP8, direct and gateway
- G2 A8 (DP scaling): `passed: false` (1.7993× vs 1.9× threshold); owner accepted as explicit closure deviation
- G2 A9: closed — DP=2×TP4 vs TP8 production-topology report on Qwen3-32B
- A12 (batch-invariance determinism, #360): closed — exact-match verdict passed on Qwen3-32B TP8
- B7 (KV answer-equivalence, #373): operator implemented and portable-validated; additive over F2/F4
- G4 MoE: M-A1 formal FAIL retained; M-A2 complete; M-A3 scope-closed by owner deviation (perf gate stays FAIL)
- G4 E-KV: FP8-E4M3 KV **FAIL** on Qwen3-32B long-context; `fp8_e4m3` startup rejected, BF16 KV fail-closed
- G5: F1a–F1d, F2a–F2d, F4a, F4b all closed; F4c decided (keep per-replica RadixKV + F2 routing, thresholded revisit)
- F5a/b/c (priority, noisy-neighbor, SLO admission): closed
- G6: P-A, P-B1–P-B4, P-C2/C3/C4 green (incl. Open WebUI P-B3 browser gate); remaining P-C gates continue
- #150 LiveCodeBench TP8 gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Fully device-side sampling, penalties, spec verification, page-table caching; incremental detokenization and stop matching
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: `kairyu bench run` (Fugu + core suites), hash-chained quality history, config A/B and quant sweeps
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- EAGLE-3 runtime integration remains a G4 follow-up; FP8-E4M3 KV disabled pending offline calibration
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-07 — [amendment] Cumulative engine state advances by deltas
- What: overlapping step snapshots now freeze append-only outputs by reference plus length; TP/EP sync uses epoch/length and output/page tails; output presentation caches cumulative token/logprob content while exposing immutable-length internal views; KV allocation pages are concatenated once.
- Why: copying and rescanning each request's full generated history on every step made host work quadratic in completion length.
- Refs: issue #324; m5 D2; m8 D1/D6; `kairyu/engine/{core/{frozen_prefix,step_input,scheduler,radix_kv,spec_runner},engine_loop,kairyu_backend}.py`

### 2026-08-07 — [amendment] Engine presentation leaves the core step thread
- What: all public prompt paths now prepare text outside the engine step owner; production detokenization and in-process delivery or process-wire event/msgpack work use one bounded serial output lane, overlapping at most one next raw step while ROUTER I/O stays on its socket thread; HF production requires native `DecodeStream`.
- Why: burst tokenization and cumulative presentation work serialized the next model step and inflated TTFT tail latency; bounded one-ahead execution preserves stop-string scheduler safe points without adding a new protocol or configurable queue.
- Refs: issue #327; m8 D1/D6; `kairyu/engine/{engine_loop,kairyu_backend,zmq_backend,core/engine_service,tokenizer}.py`

### 2026-08-07 — [amendment] Prefill host preparation scales by physical pages
- What: ragged prefill now validates cross-row KV ownership in one physical-page pass and packs its typed metadata into one fresh buffer, using a single pinned H2D copy on CUDA.
- Why: token-by-token ownership maps, row-pair intersections, and pageable per-field uploads added avoidable TTFT before every batched prefill model call.
- Refs: issue #322; m13 D1; `kairyu/engine/core/prefill.py`; `tests/unit/test_batched_prefill.py`

### 2026-08-07 — [amendment] Prefill budget forms bounded work-conserving cohorts
- What: native schedulers now share post-decode prefill budget across a bounded leading cohort of equal-priority partial prompts (two by default), expose `max_num_partial_prefills`, and permit one cache-safe, completion-only immediate-successor admission past a KV-blocked head; deferred P-D may retain one peer token to overlap its copy.
- Why: serial long-prefill chunks and unbounded head-of-line KV blocking inflated small-prompt TTFT, while unrestricted skip-ahead could starve the head or create recompute-preemption thrash.
- Refs: issue #328; m11 D6/A11; `kairyu/engine/core/scheduler.py`; `tests/unit/test_{scheduler,scheduler_waiting_queue}.py`

### 2026-08-07 — [amendment] Direct chat activates predictive TTFT admission
- What: added opt-in `server.ttft_slo_s`; validated direct interactive chat now includes known ingress elapsed time, atomically admits, batch-defers only on routes attesting running-decode isolation (otherwise sheds), observes the first successfully sent visible SSE delta, releases leases at the outer ASGI boundary, and exports the controller snapshot through six Prometheus gauges.
- Why: the validated F5c controller had no production call site, so requests predicted to miss the TTFT SLO still consumed serving capacity and reduced SLO-goodput.
- Refs: issue #340; `kairyu/entrypoints/server/{app,middleware,metrics,settings,slo}.py`; `kairyu/deploy/spec.py`; `tests/server/test_slo_admission_integration.py`; `docs/design/m11-product.md`

### 2026-08-06 — [progress] Pipeline-depth hypothesis closes as a measured negative
- What: retained twelve Qwen3-32B TP4 HTTP diagnostics and CPU plan-shape evidence, reverted the experimental deep strict-decode tail and metric, and kept the existing two-step admission/prefill horizon.
- Why: the initial depth-five candidate regressed throughput by 3–4%, its cohort-preserving refinement returned only to noisy parity, and the historical 35.98% result compared depth one with five rather than isolating benefit beyond depth two.
- Refs: issue #318; `bench/results/issue-318-pipeline-depth-qwen3-32b-rtxpro6000-2026-08-06/`

### 2026-08-06 — [amendment] CUDA graph coverage moves to readiness-time defaults
- What: supported real CUDA models now resolve omitted decode policy to CUDA graphs, size graph batch/page coverage from serving limits, pre-capture every bucket before single/TP/attention-DP EP readiness with bounded rank preflight and rollback, and export monotonic single/process/pool eager fallbacks to Prometheus; CPU, P-D, replicated EP, custom, and MLA paths stay eager by capability.
- Why: lazy first-use capture blocked live traffic and the former batch/page defaults left common requests permanently eager without an operator-visible counter.
- Refs: issue #320; m17 A26-A28; `kairyu/engine/{kairyu_backend,core/{step_executor,model_runner,worker}}.py`; `kairyu/entrypoints/server/metrics.py`; `tests/gpu/test_ep_attention_dp_cuda_graph_startup_gpu.py`

### 2026-08-06 — [progress] Offloaded-route endpoint context no longer trusts FastAPI's id cache
- What: `_OffloadedRequestBodyRoute` now computes the validation-error endpoint context once from the live endpoint at route build instead of calling FastAPI's `_extract_endpoint_context`, and the flaky ZMQ cancelled-submit test waits on wall-clock time for the prompt worker thread.
- Why: FastAPI 0.139.0 caches endpoint context by `id(func)` without holding a reference, so a garbage-collected endpoint's recycled address serves a stale context (observed as `chat_completions` reported for `probe` in CI); the ZMQ test's bare loop yields gave the worker thread no time on loaded runners.
- Refs: PR #423; `kairyu/entrypoints/server/app.py`; `tests/server/test_prompt_offload.py`; `tests/unit/test_zmq_backend.py`

### 2026-08-06 — [progress] CPU microbenchmark gate retries timing-ratio noise
- What: the same-run CPU gate re-runs a benchmark (≤3 attempts total) when its only failing checks are timing ratios; structural failures and benchmark errors still fail immediately, and thresholds plus the report schema consumed by the nightly series are unchanged.
- Why: shared-runner jitter tripped `scheduler.priority_speedup` (0.87 vs the 0.9 floor) on a docs-only PR; a real hot-path regression keeps failing on every attempt, so retries preserve the smoke alarm while removing the flake.
- Refs: PR #422 CI run 31094215441; `scripts/cpu_microbench_gate.py`; `tests/unit/test_cpu_microbench_gate.py`

### 2026-08-06 — [design] PROGRESS.md restructured for minimal session-start tokens
- What: archived the verbose Current Status to `docs/progress/archive/status-2026-08-06.md` and all older Change Log entries to `docs/progress/archive/change-log.md` (both verbatim); PROGRESS.md now holds a compact Product/Current Status/Change Log. Size budgets and archiving live in `.claude/rules/progress-log.md` + `docs/progress/archiving.md`, enforced by a SessionStart hook running `scripts/check_progress_size.py`.
- Why: PROGRESS.md and the progress rules are loaded every session; at ~514 KB they burned the context they were meant to save.
- Refs: `docs/progress/archive/`; `.claude/settings.json`; `scripts/check_progress_size.py`
