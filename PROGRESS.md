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

Snapshot date: 2026-08-10. Hardware context: all GPU evidence so far is on
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
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: Accuracy/Core/Quantization/Structured/Long Context suites, six-model sourced Accuracy comparison, target-only streamed TTFT/TPS, hash-chained quality history, config A/B and quant sweeps; shared fail-closed evidence replay mechanics
- Frontier example surface rebuilt around Qwen 1-GPU, DeepSeek 8-GPU, and combined 8-GPU environments with process-only configuration, pinned revisions/images, bind-backed external model storage, credential-safe offline model attestations, unified lifecycle/benchmark CLIs, throughput-oriented concurrency 16, a deterministic Qwen LiveCodeBench 30-item performance diagnostic, and per-attempt reports; the DeepSeek vLLM arm currently needs a local source build of `jasl/vllm@aa0d513` plus eager execution, disabled DeepGEMM, and disabled FP4 indexer caching on SM120, and has no completed 64-item verdict
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

### 2026-08-11 — [progress] DeepSeek vLLM reaches SM120 inference without a full verdict
- What: Kairyu L3 served a local CUDA 13 source build of `jasl/vllm@aa0d513` after the pinned v0.26 image failed SM120 scaled-MM; startup also disables unsupported FP4 indexer caching and allows the long engine-ready phase. The requested 64-item LiveCodeBench run was stopped before a pair result was persisted, so it has no accuracy, TTFT, or TPS verdict.
- Refs: PR #467; `examples/deepseek-v4-flash-0731-8gpu/{compose.yaml,example.json,vllm-gateway.yaml}`

### 2026-08-10 — [progress] DeepSeek vLLM arm selects the SM120 fallback path
- What: The reference arm now pins the v0.26.0 CUDA 13.0 image and disables DeepGEMM so RTX PRO 6000 uses the guarded TileLang MHC fallback; gateway deployment attestation pins the same digest.
- Refs: vllm-project/vllm#50576; `examples/deepseek-v4-flash-0731-8gpu/{compose.yaml,example.json,vllm-gateway.yaml}`

### 2026-08-10 — [progress] DeepSeek vLLM reference arm bypasses broken Inductor compilation
- What: The standalone DeepSeek V4 8-GPU vLLM arm now uses eager execution instead of the failing piecewise Inductor path; a compose contract test keeps the workaround scoped to that arm.
- Refs: vllm-project/vllm#40821; `examples/deepseek-v4-flash-0731-8gpu/compose.yaml`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-10 — [progress] Qwen example gains a fixed 30-item performance diagnostic
- What: The full-dataset example default remains unchanged; a separate entrypoint fixes LiveCodeBench selection to `limit=30`, `seed=0`, and concurrency 16, while the shared controller records those CLI overrides in both backend runs.
- Refs: PR #465; `examples/_shared/benchctl.py`; `examples/qwen3.6-27b-1gpu/bench-livecodebench-30.sh`

### 2026-08-10 — [amendment] External benchmark concurrency increases to sixteen
- What: Main, judge, serving, and all frontier example clients now use sixteen simultaneous in-flight requests; combined orchestration applies the same external limit in addition to its internal replica and proposal fan-out.
- Why: The owner selected the higher request wave after observed KV-cache headroom showed that eight did not saturate the single-GPU engine.
- Refs: supersedes the concurrency-eight and concurrency-one entries below; PR #465; `kairyu/bench/`; `bench/{serving_bench.py,tiered_auto_bench.py,configs/}`; `examples/`

### 2026-08-10 — [amendment] External benchmark concurrency returns to eight
- What: Main, judge, serving, and all frontier example clients now use eight simultaneous in-flight requests; combined orchestration also applies the same external limit in addition to its internal replica and proposal fan-out.
- Why: The owner selected maximum aggregate per-GPU continuous-batching throughput over single-request isolation after a concurrency-1 LiveCodeBench run exposed unacceptable long-tail wall time.
- Refs: supersedes the two concurrency-1 entries below; PR #465; `kairyu/bench/`; `bench/{serving_bench.py,tiered_auto_bench.py,configs/}`; `examples/`

### 2026-08-10 — [progress] Every generic external benchmark client is serial by default
- What: Judge calls, the standalone serving runner, and checked-in accuracy, core, structured, and quantization configs now join the main runner and frontier examples at concurrency 1; load-balanced and intentional stress runs can still opt into higher values explicitly.
- Why: A separate judge semaphore and older sample configs could silently restore parallel load even when the primary benchmark client was serialized.
- Refs: PR #465; `kairyu/bench/types.py`; `bench/{serving_bench.py,configs/}`

### 2026-08-10 — [progress] Frontier benchmarks default to one external request
- What: The generic runner and all Qwen, DeepSeek, and combined example commands now default to and explicitly record external concurrency 1; combined orchestration retains its independent four-replica Qwen pool and internal proposal fan-out.
- Why: Without an explicitly load-balanced target, concurrent client requests distort single-engine accuracy, TTFT, and TPS comparisons and impose unintended GPU load.
- Refs: PR #465; `kairyu/bench/{types.py,adapters/base.py}`; `examples/{_shared,qwen3.6-27b-1gpu,deepseek-v4-flash-0731-8gpu,qwen3.6-deepseek-v4-8gpu}/`; `bench/tiered_auto_bench.py`

### 2026-08-10 — [progress] Frontier quality runs survive long reasoning phases
- What: Benchmark CLI now accepts a positive generation read timeout, and Qwen, DeepSeek, and combined frontier examples pin a one-day allowance so reasoning-only streams cannot be retried or failed by the generic 600-second limit before final code appears.
- Refs: PR #465; `kairyu/bench/{cli,config}.py`; `examples/{_shared,qwen3.6-27b-1gpu,deepseek-v4-flash-0731-8gpu,qwen3.6-deepseek-v4-8gpu}/`

### 2026-08-10 — [progress] Benchmark CLI preserves explicit output budgets
- What: `kairyu bench run` now accepts a positive `--max-output-tokens` override and applies it to every CLI-declared target, so frontier example budgets reach adapters instead of failing at argument parsing or reverting to 8,192 tokens.
- Refs: PR #465; `kairyu/bench/{cli,config,types}.py`; `tests/bench/test_bench_config.py`
