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

Snapshot date: 2026-08-11. Hardware context: all GPU evidence so far is on
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
- Benchmark/eval tooling: Accuracy/Core/Quantization/Structured/Long Context suites, eight-model sourced Accuracy comparison with cell-level provenance, target-only streamed TTFT/TPS, hash-chained quality history, config A/B and quant sweeps; shared fail-closed evidence replay mechanics
- The example surface includes measured RTX PRO 6000 deployments for 8-GPU DeepSeek and one-GPU Qwen3.6 FP8; both route Open WebUI through Kairyu L3 to pinned vLLM L1. Qwen's no-MTP/16K-batch configuration completed its final serving and LiveCodeBench-20 gates after MTP and 32K-batch candidates failed performance or stability checks, with all persistent data and compilation caches on NVMe
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

### 2026-08-11 — [progress] One-GPU Qwen3.6 example completes measured gate
- What: The one-command Qwen3.6-27B FP8 stack on one RTX PRO 6000 completed its final unique-prefix 8K/256 matrix (41.43 output tok/s with 190.82 ms median TTFT at c1; 842.35 output tok/s with 1.247 s median TTFT at c32) and deterministic LiveCodeBench subset (11/20, 55.0%, 20/20 scored, zero request errors/retries/unmeasured rows). No-MTP won TTFT at every tested concurrency and aggregate throughput at c16/c32; MTP-5 crashed at c16 and the 32K prefill budget exhausted VRAM at c32.
- Refs: PR #469; `examples/qwen3.6-27b-1gpu/MEASUREMENTS.md`; run IDs `final-no-mtp-20260811` and `qwen36-fp8-no-mtp-lcb20-20260811`

### 2026-08-11 — [progress] One-GPU Qwen3.6 vLLM example enters runtime validation
- What: A second example adds one-command Open WebUI -> Kairyu L3 -> pinned vLLM L1 serving for the official Qwen3.6-27B FP8 checkpoint on one selected RTX PRO 6000; model/UI persistence is fail-closed below `/mnt/nvme`, and independent/all serving plus deterministic 20-row LiveCodeBench runners are contract-tested. Runtime tuning and measured evidence remain in progress.
- Refs: `examples/qwen3.6-27b-1gpu/`; `tests/unit/test_frontier_examplectl.py`

### 2026-08-11 — [progress] Accuracy comparisons cite every reference score
- What: Accuracy reports now compare local runs with the final eight-model benchmark matrix, preserve missing values, and attach source markers, dates, source class, conditions, and notes to selected and alternate scores in Markdown and JSON.
- Refs: `kairyu/bench/{reference.py,compare.py}`; `tests/bench/test_bench_compare.py`; `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`

### 2026-08-11 — [design] Accuracy reports bind reference scores to sources
- What: The Accuracy comparison will use the supplied eight-model matrix and render a source marker, linked source metadata, measurement condition, and notes for every selected or alternate published score; missing values remain missing and old DeepSeek snapshots do not fill the 0731 column.
- Why: A numeric comparison without cell-level provenance obscures provider, third-party, harness, and version differences and cannot be audited from the generated report.
- Refs: `docs/superpowers/specs/2026-08-11-accuracy-reference-sources-design.md`; `kairyu/bench/{reference.py,compare.py}`

### 2026-08-11 — [progress] Full DeepSeek-V4 LiveCodeBench evidence completes
- What: The clean-cache TP8+EP8 DSpark-5 example completed and scored all 1,055 LiveCodeBench release_v6 problems: 759 passed (71.9431%), all target timings measured, and zero request errors, retries, or unmeasured rows. The warm serving matrix reached 168.01 output tok/s at c1 with 220.10 ms median TTFT and 972.93 output tok/s at c32.
- Refs: PR #468; `examples/deepseek-v4-flash-0731-8gpu/MEASUREMENTS.md`; run IDs `livecodebench-full-tp8-ep8-dspark5-clean-sse16m-20260811` and `tp8-ep8-dspark5-warm-20260811`

### 2026-08-11 — [progress] Frontier benchmark streams support 32K-token evidence
- What: The strict benchmark SSE reader retains its fail-closed bound but raises it from 1 MiB to 16 MiB so 32K-token DeepSeek streams are not rejected solely by per-token JSON framing. The example runner now also rejects a nominally successful CLI run unless all 1,055 LiveCodeBench rows have measured, error-free evidence.
- Refs: PR #468; `kairyu/bench/streaming.py`; `examples/deepseek-v4-flash-0731-8gpu/benchmark.py`; full-run diagnosis on 2026-08-11

### 2026-08-11 — [progress] Kairyu-owned vLLM prompts bypass a second chat template
- What: Pre-rendered `TemplatedPrompt` requests sent to a vLLM-compatible backend now use `/completions` for both buffered and streaming generation. This preserves the checkpoint's single prompt boundary, prevents DeepSeek chat mode from being wrapped into thinking mode again, and returns final answer text to Open WebUI.
- Refs: PR #468; `kairyu/engine/openai_backend.py`; `tests/unit/test_openai_backend.py`; real DeepSeek-V4 direct and streaming probes on 2026-08-11
