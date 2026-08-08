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

Snapshot date: 2026-08-08. Hardware context: all GPU evidence so far is on
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
- Device-side sampling, penalties, spec verification, page-table caching; TP step headers sleep on Gloo while fixed-layout delta payloads use the bounded NCCL model group and rare controls remain Gloo objects; structured masks stay on CUDA with only selected IDs returned to the host matcher; deterministic n-gram/EAGLE-3/MTP drafts preserve T>0 and penalized sampling
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
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-08 — [amendment] Grouped-MoE probe failures always fall back
- What: capability and plan-warmup probes catch backend-specific ordinary exceptions and decline the derived grouped pack.
- Why: cuDNN graph-support errors do not consistently inherit `RuntimeError`; an optional optimization must not abort otherwise supported reference serving.
- Refs: issue #331; PR #453 Fable 5 re-review; M15 A12; `kairyu/models/grouped_moe.py`

### 2026-08-08 — [amendment] Grouped MoE is capability-probed and decode-bounded
- What: correcting the preceding #331 entry, grouped plans are power-of-two bucketed and prewarmed through 8,192 total rows before graph capture; larger EP prefills and unsupported/custom runtimes keep the reference transport/math.
- Why: unconditional fixed capacity multiplied prefill memory/FLOPs by EP degree, while late cuDNN plan/workspace growth could invalidate captured graph pointers.
- Refs: issue #331; PR #453 Fable 5 review; M15 A11; M16 D3; `kairyu/models/grouped_moe.py`

### 2026-08-08 — [amendment] Dense MoE uses grouped GEMM and fixed EP capacity
- What: CUDA BF16 experts use canonical-storage packs, device-side sort/offsets, and two FlashInfer cuDNN grouped GEMMs; generic EP uses fixed peer-capacity forward/reverse all-to-all buffers with host-constant splits.
- Why: data-dependent `unique`/`nonzero` expert loops and count `.item()`/`.tolist()` forced up to one host synchronization and small GEMM per expert, preventing CUDA graph capture.
- Refs: issue #331; M15 A10; M16 D3; `kairyu/models/{grouped_moe,moe,moe_parallel}.py`

### 2026-08-08 — [amendment] TP step headers remain on sleeping Gloo transport
- What: correcting the preceding #323 entry, a two-word Gloo tensor header frames every transaction; only encoded `StepDelta` payloads use the bounded NCCL model group.
- Why: posting a process-lifetime NCCL receive while idle burns resources and can strand orphaned GPU workers after rank-0 death; a Gloo tensor header avoids pickle while retaining sleeping TCP liveness.
- Refs: issue #323; PR #452 Fable 5 review; M16 D4; `kairyu/engine/core/worker.py`

### 2026-08-08 — [amendment] TP step control uses a dedicated NCCL tensor channel
- What: hot-path `StepDelta` state is losslessly encoded into int64 tensors on a long-idle NCCL subgroup; rare controls remain Gloo objects and model collectives retain their short fail-fast subgroup.
- Why: removing per-step pickle/TCP overhead must not make healthy idle workers time out or weaken bounded model-operation failure detection.
- Refs: issue #323; M16 D4; `kairyu/engine/core/{step_input,worker}.py`

### 2026-08-08 — [amendment] Structured generation uses the device sampling seam
- What: regex, EBNF, and structural-tag formats join JSON grammars; native strict tools compile parser-matched schemas (one call for Llama); both layouts synchronously isolate malformed requests; per-process vocabulary/compiler caches avoid rebuilds without compile-time global locking; CUDA masks avoid full logits-row D2H.
- Why: structured and strict-tool requests must be correctness invariants without forcing every token through the vocabulary-sized CPU sampling path.
- Refs: issue #363; M8 D2; `kairyu/engine/{backend.py,core/{structured,sampler}.py}`

### 2026-08-08 — [amendment] Deterministic drafts preserve sampled distributions
- What: n-gram/EAGLE/MTP argmax drafts use point-mass rejection verification for T>0; per-row history slicing also enables penalties while grammar/forced continuations keep the safe bypass.
- Why: one target draw exactly realizes acceptance plus residual correction for q(t)=1, avoiding vocabulary-sized draft/target probability transfer.
- Refs: issue #358; M8 D4; `kairyu/engine/core/{spec_decode,spec_runner,model_runner}.py`

### 2026-08-08 — [amendment] FP8 audit separates arithmetic noise from error
- What: calibrated dequant auditing permits a documented 2^-48 relative FP64 comparison slack while retaining exact stored-byte, finite, range, and physical E4M3 error checks.
- Why: byte-perfect writes could exceed the theoretical error boundary by a few FP64 ulps, making the calibrated gate unsatisfiable independently of model quality.
- Refs: issue #357; PR #449 Fable 5 review; `bench/fp8_kv_g4_ekv_bench.py`; `tests/bench/test_fp8_kv_g4_ekv_bench.py`
