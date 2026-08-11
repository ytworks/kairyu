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

Snapshot date: 2026-08-12. Hardware context: all GPU evidence so far is on
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
- #150 LiveCodeBench TP8 gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Incremental architecture-state paths for Qwen3.6 and DeepSeek V4 plus an explicit recompute diagnostic mode; DeepSeek EP2/4/8 Attention-DP and direct packed-FP4 execution are implemented, with SM120 single-kernel and two-rank NCCL smokes green
- Device-side sampling, penalties, spec verification, page-table caching; TP step headers sleep on Gloo while fixed-layout delta payloads use the bounded NCCL model group and rare controls remain Gloo objects; structured masks stay on CUDA with only selected IDs returned to the host matcher; deterministic n-gram/EAGLE-3/MTP drafts preserve T>0 and penalized sampling
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: Accuracy/Core/Quantization/Structured/Long Context suites, eight-model sourced Accuracy comparison with cell-level provenance, target-only streamed TTFT/TPS including exact public-vs-internal orchestration token rates, hash-chained quality history, config A/B and quant sweeps; shared fail-closed evidence replay mechanics
- The example surface includes measured RTX PRO 6000 deployments for 8-GPU DeepSeek and one-GPU Qwen3.6 FP8, plus a contract-tested tiered Qwen TP1x4 + DeepSeek TP4/EP4 L2 stack entering runtime validation; its externally bound no-auth Open WebUI defaults to quality-first `kairyu-auto-max` and calls only loopback-exposed Kairyu L3, which owns pinned vLLM L1. Qwen's no-MTP/16K-batch configuration completed its final serving and LiveCodeBench-20 gates after MTP and 32K-batch candidates failed performance or stability checks, with all persistent data and compilation caches on NVMe
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- Frontier full-checkpoint 262K/1M quality/performance evidence, DeepSeek EP4/EP8 topology lock, CUDA Graph pointer stability, MTP/DSpark selection, 30-minute soak, and failure recovery remain open
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-12 — [progress] Ordinary synthesis fails to preserve the quality floor
- What: Ordinary-DeepSeek MoA-3 scored 0/3 completed pilot tasks, making the fourth task irrelevant to its maximum possible 1/4 score versus direct DeepSeek's clean 2/4. The quality candidate returns to private-thinking MoA-3 with a generic 2048-token internal allowance to eliminate the previously measured 1024-token boundary-exhaustion tail before re-running L3 performance and quality gates.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,auto-max.yaml}`; run ID `terminalbench-selection-moa3-vs-deepseek-20260812`; PR #471

### 2026-08-12 — [progress] MoA-2 fails the quality-selection gate
- What: The performance-winning ordinary-chat MoA-2 candidate scored 1/3 completed Terminal-Bench 2.1 pilot tasks and failed the fourth request with `BadGatewayError`, below direct DeepSeek's clean 2/4 baseline. MoA-2 is rejected; the already performance-qualified MoA-3 ordinary-chat candidate advances to the same fixed four-task gate.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,auto-max-chat.yaml}`; run ID `terminalbench-selection-moa2-vs-deepseek-20260812`; PR #471

### 2026-08-12 — [progress] MoA-2 chat synthesis wins the L3 performance gate
- What: The ordinary-DeepSeek MoA-2 candidate completed c1/c8/c16/c32 with 32/32 non-empty answers and valid traces at every row; versus MoA-3 it preserves c1 and cuts median TTFT by 17.1%/21.0%/20.7% at c8/c16/c32. The quality pilot now compares only this winner with direct DeepSeek, and validates raw zero-failed/unjudged/skipped/error evidence on a clean source tree.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,benchmark.py}`; run ID `l3-auto-max-chat-moa2-public-v1-20260812`; PR #471

### 2026-08-12 — [progress] Tiered quality candidate reduces fan-out, not proposal depth
- What: The 512-token private-cap A/B retained 32/32 valid L3 responses but did not improve c1/c8 median TTFT beyond run noise, so the quality candidate restores the 1024-token allowance and changes ordinary-chat synthesis from MoA-3 to MoA-2 for the next performance/quality gate.
- Refs: run IDs `l3-auto-max-chat-moa3-public-v1-20260812`, `l3-auto-max-chat-moa3-private512-public-v1-20260812`; `examples/qwen3.6-deepseek-v4-8gpu/`; PR #471

### 2026-08-12 — [amendment] L2 bounds private work independently of public output
- What: Orchestrator YAML and decorator specs now expose a validated, backend-neutral `internal_max_tokens` policy that caps private planning/proposal/verification generations without reducing the caller's final-answer budget. The tiered MoA-3 chat-synthesis candidate pins 512 after its reliable but 1024-token L3 matrix showed excessive c16/c32 latency.
- Why: Private proposal length is an operator latency/quality tradeoff and must remain portable policy rather than a model- or example-specific core-code branch.
- Refs: M11 D2; `kairyu/dsl/`; `examples/qwen3.6-deepseek-v4-8gpu/auto-max-chat.yaml`; run ID `l3-auto-max-chat-moa3-public-v1-20260812`; PR #471
