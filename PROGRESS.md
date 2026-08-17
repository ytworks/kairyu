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

Snapshot date: 2026-08-17. Hardware context: all GPU evidence so far is on
8× RTX PRO 6000 Blackwell (SM120), PCIe-only interconnect (P2P 30–37 GB/s);
NVLink-HBM (H100-class) formal gates still need hardware. Evidence lives in
`bench/results/` (see `index.json`); decisions and rationale in `docs/design/`.

### Milestones

| Milestone | Status |
|---|---|
| M1 Orchestration+Interface | Complete: Router/Conductor/MoA, vLLM-compat API, OpenAI server, DSL |
| M2 Core engine | CPU half done; unified EngineLoop + device sampling GPU-validated; NVLink perf gates pending |
| M3 Spec/graphs/P-D | n-gram spec, CUDA-graph serving, intra-node P-D GPU-validated; EAGLE-3/MTP greedy serving integrated |
| M4 Router learning | Implemented CPU-only, design reviewed |
| M5 Intra-node multi-GPU | CPU half done; TP/DP/P-D plumbing live; GPU phase per runbook |
| M6 Inter-node multi-GPU | CPU half done; production stage-sharded PP remains a roadmap item |
| M7 Productionization | CPU half done: serve CLI, gateway, batch, compose smoke; `kairyu validate` preflight |
| M8 Engine CPU core | Complete (amended 2026-08-08) |
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
- #356 real-checkpoint quant parity: evidence complete — INT8 PASS; AWQ/GPTQ formal FAIL retained with SHA-bound same-GPU oracle replay isolating checkpoint quantization loss
- B7 (KV answer-equivalence, #373): operator implemented and portable-validated; additive over F2/F4
- G4 MoE: M-A1 formal FAIL retained; M-A2 complete; M-A3 scope-closed by owner deviation (perf gate stays FAIL); dense BF16 MoE uses sort-by-expert grouped GEMM and fixed-capacity EP transport, then combines returned rows in fixed FP32 order before one model-dtype cast
- G4 E-KV: unit-scale and calibrated per-layer K/V FP8-E4M3 re-bakes **FAIL** retained; calibrated cache metrics/logprobs pass but 16K/32K exact tokens and decode envelope do not; `fp8_e4m3` startup rejected
- G5: F1a–F1d, F2a–F2d, F4a, F4b all closed; F4c decided (keep per-replica RadixKV + F2 routing, thresholded revisit)
- F5a/b/c (priority, noisy-neighbor, SLO admission): closed
- G6: P-A, P-B1–P-B4, P-C2/C3/C4 green (incl. Open WebUI P-B3 browser gate); remaining P-C gates continue
- #150 TP8 long-generation stability gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Incremental architecture-state paths for Qwen3.6 and DeepSeek V4 plus an explicit recompute diagnostic mode; DeepSeek EP2/4/8 Attention-DP and direct packed-FP4 execution are implemented, with SM120 single-kernel and two-rank NCCL smokes green
- Device-side sampling, penalties, spec verification, page-table caching; TP step headers sleep on Gloo while fixed-layout delta payloads use the bounded NCCL model group and rare controls remain Gloo objects; structured masks stay on CUDA with only selected IDs returned to the host matcher; deterministic n-gram/EAGLE-3/MTP drafts preserve T>0 and penalized sampling
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example is a coding-first product: an L1 fleet of four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker, under a nine-role coding L2 DAG plus a seven-role general ensemble profile auto-selected for agent/format-constrained and non-code turns (never a direct single-engine route) where a Qwen head streams the public answer opening from t=0 (~0.3 s at c1; semantic-TTFT gate ≤2× the DeepSeek-direct row), Qwen test/proposal fan-out feeds a networkless CPU sandbox over a Unix-socket transport, and a verified DeepSeek continuation streams the remainder after the committed opening; non-coding requests skip execution locally. Launcher readiness proves the DAG, executor binding, and a two-input 384-dimensional embedding response. Image chat still reaches only Qwen roles; composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- Frontier full-checkpoint 262K/1M correctness/performance evidence, DeepSeek EP4/EP8 topology lock, CUDA Graph pointer stability, MTP/DSpark selection, 30-minute soak, and failure recovery remain open
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-17 — [amendment] Tiered Qwen MTP remains candidate-only
- What: removed MTP-3 from the selected four-replica Qwen deployment while
  retaining its c1/c4/c8 measurements as candidate evidence; compose, metadata,
  tests, and operator docs now agree on no speculation.
- Why: the public product admits 256 requests and gates c16/c32, but the new
  role-shaped run stopped at c8; the matching Qwen TP1 c16/c32 rows regressed,
  so the deployed high-concurrency envelope was not proven safe.
- Refs: #509; ECO-D4 2026-08-17 Qwen MTP concurrency amendment;
  `examples/qwen3.8-deepseek-v4-8gpu/`

### 2026-08-17 — [design] General ensemble profile and Terminal-Bench latency fixes
- What: `general_roles` — a second full-ensemble DAG under one served model,
  deterministically selected (tools/format-demand/non-code → general; code
  authoring keeps the coding DAG; never a single-engine route); example gains a
  7-role all-model general profile; verifier T 0.0→0.6/top_p 0.95 (greedy
  thinking looped: 52% of verdicts burned the 4096 cap → reverify); Qwen
  prompts lead with a shared REQUEST block (prefix reuse); measured MTP-3
  adopted on Qwen workers (c1 +43.9%, c4/c8 +26%, lossless).
- Why: issue #509 — 12/27 Terminal-Bench trials hit the 900 s agent timeout at
  p50 63.9 s/request; coding contracts idled on agent JSON turns.
- Refs: #509; ECO-D6 + ECO-D4 2026-08-17 amendment; `kairyu/orchestration/`;
  `examples/qwen3.8-deepseek-v4-8gpu/`; its MEASUREMENTS.md

### 2026-08-16 — [amendment] Agent tool-turn latency and cancellation (issue #495)
- What: tool-turn FAIL/refine loop removed (verifier/synthesis prompts declare
  the publisher emits the tool call; inconclusive verdicts re-verify once,
  never auto-FAIL; verifier cap 2048→4096); a complete committed opening is
  declined via a stripped NO_CONTINUATION sentinel; conversation-head
  KV-affinity session; client-disconnect cancels the non-streaming AUTO DAG
  (499); `kairyu_conductor_stage_seconds` per-stage metric; `/v1/models`
  advertises `max_model_len`; example timeouts bounded (600 s / 1800 s);
  after a still-timing-out rerun, a plain-text structured-format demand
  (Terminus-2 "format as JSON") now disables the head like tools do, and the
  example refine depth drops 2→1.
- Why: a traced minimal tool turn spent 20.8 s of 25.8 s in three thinking
  verifier passes; the rerun showed head prose corrupting JSON replies and
  ~11-minute single responses against the 900 s Terminal-Bench budget.
- Refs: #495, #501 (+review PRs #502–#507), ECO-D4 2026-08-16 amendment;
  `kairyu/orchestration/`; `kairyu/entrypoints/server/`;
  `examples/qwen3.8-deepseek-v4-8gpu/{auto-max,kairyu}.yaml`

### 2026-08-16 — [amendment] OpenAI Chat Completions output contract (issue #496)
- What: omitted output limits are context-bound (16-token fallback removed,
  admission reserves max_model_len); a caller limit is one budget across
  head+continuation with honest finish_reason; standard usage now means the
  public request/completion (orchestration_* keep #196 cumulative totals,
  metering still bills them); empty final output retries once then 502;
  `reasoning_closed`/`prompt_headless` role contracts; GET /v1/models/{id}.
- Why: issue #496 — orchestrated responses ignored caller limits, reported
  deterministic "length", and lost post-tool answers to `reasoning_content`.
- Refs: #496, #458, m11 2026-08-16 amendment, ECO-D4 2026-08-16 amendment

### 2026-08-15 — [amendment] Sandbox reaps escaped submission descendants
- What: the coding executor now acts as a child subreaper and sweeps/reaps
  same-UID runner descendants not owned by another live submission on every
  completion path.
- Why: timeout-only process-group cleanup allowed `setsid()` children to retain
  resources and mutate later execution evidence after an accepted result.
- Refs: ECO-D1; `examples/qwen3.8-deepseek-v4-8gpu/sandbox/runner.py`

### 2026-08-15 — [amendment] Executor deadline includes queue admission
- What: deployment executor config now declares queue allowance; the client
  shares one wall-plus-queue deadline across retries and the runner rejects
  admission when the residual budget cannot cover the declared wall limit.
- Why: the previous 8s queue plus 10s execution exceeded the 15s client
  deadline, discarding valid work as `unavailable` under contention.
- Refs: ECO-D1; `kairyu/orchestration/execution.py`; PR #488 review
