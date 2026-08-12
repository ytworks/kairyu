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
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; calibrated auto-max routes prompts beyond the tier1 input envelope directly to tier2; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: 12-slot Accuracy plus Core/Quantization/Structured/Long Context suites, eight-model sourced Accuracy comparison with cell-level provenance, target-only streamed TTFT/TPS including exact public-vs-internal orchestration token rates, hash-chained quality history, config A/B and quant sweeps; SWE-bench Verified uses mini-SWE-agent's official 250-step `verified` flow plus the official harness with fail-closed denominators; Terminal-Bench keeps resumable raw Harbor jobs and bounds every agent phase to two effective hours
- The example surface includes measured RTX PRO 6000 deployments for 8-GPU DeepSeek and one-GPU Qwen3.6 FP8, plus a measured tiered Qwen TP1x4 + DeepSeek TP4/EP4 stack; its no-auth external Open WebUI defaults to quality-first MoA-3 `kairyu-auto-max` and calls only loopback Kairyu L3. The selected policy cleared direct DeepSeek 3/4 vs 2/4 on the fixed pilot and its all-89 single-attempt Terminal-Bench run scored 60/89 with Harbor's zero-inclusive Mean; all persistent data and compilation caches are on NVMe
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

### 2026-08-12 — [progress] Docker executor tolerates loaded-daemon creation
- What: The untrusted-code runner keeps 10-second Docker control calls but allows 30 seconds for asynchronous container creation, retaining the late-create ownership handoff and delegated cleanup guarantee.
- Refs: `kairyu/bench/execution.py`; `tests/bench/test_bench_exec_runners.py`

### 2026-08-12 — [progress] Benchmark cache reuses unchanged verification
- What: One suite run now reuses a normalized cache's digest, parsed asset references, and manifest while its exact file identities stay unchanged; adapter-owned assets remain rehashed on every readiness check and any data/manifest mutation invalidates the reuse.
- Refs: `kairyu/bench/cache.py`; `tests/bench/test_bench_download_resilience.py`

### 2026-08-12 — [amendment] Auto-max respects the tier1 input envelope
- What: Calibrated auto-max now applies its token/character capacity guards before forcing multi-agent routing, sending oversized prompts directly to tier2 instead of dispatching impossible tier1 proposals.
- Why: Long-context Accuracy inputs exceeded Qwen's 262K context while remaining valid for the example's 1M-context DeepSeek tier, causing every forced proposal to fail upstream.
- Refs: `kairyu/orchestration/router.py`; `tests/unit/test_router.py`

### 2026-08-12 — [amendment] LiveCodeBench Pro honors its declared case range
- What: LiveCodeBench Pro now scores the complete numbered `1..n_cases` range, records paired files above that range as ignored extras, and still rejects missing or unpaired declared cases.
- Why: The pinned `2112B` archive declares 57 judge cases but retains paired sample files 58 and 59; treating those samples as denominator drift made the full official archive unusable.
- Refs: `kairyu/bench/adapters/livecodebench_pro.py`; `tests/bench/test_bench_lcb_datasets.py`; `docs/benchmarks.md`

### 2026-08-12 — [amendment] MRCR accepts the corrected official 500-row slice
- What: MRCR keeps exact `o200k_base` selection and a fail-closed 500-row denominator while retaining the corrected source's observed 106/96/98/100/100 per-bin distribution.
- Why: The pinned December 2025 source repairs moved examples across adjacent exact-token boundaries, so enforcing the dataset card's original 100-per-bin shape rejected the complete corrected slice.
- Refs: `kairyu/bench/adapters/mrcr.py`; `tests/bench/test_bench_{mcq_adapters,download_hf}.py`; `docs/benchmarks.md`

### 2026-08-12 — [amendment] SWE-bench fixes selection before generation
- What: SWE-bench now persists the official ordered selection before generation, evaluates that full set even when a prediction is omitted, accepts the official v4.1 completed/error report overlap with error precedence, and retains redacted stage logs plus raw upstream evidence; failed runs resume while explicit reruns use isolated artifacts.
- Why: mini-SWE-agent can exit successfully after a worker omits its prediction, so deriving `--instance_ids` from `preds.json` could shrink the denominator and deleting work directories could discard the only audit and resume evidence.
- Refs: issue #472; `kairyu/bench/adapters/swebench.py`; `kairyu/bench/adapters/base.py`; `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`

### 2026-08-12 — [progress] SWE-bench Verified launches through Accuracy
- What: Accuracy now runs the 500-task Verified test split through mini-SWE-agent's standard 250-step configuration and the official SWE-bench evaluator, preserves every selected task in the resolved-rate denominator, retains auditable commands and raw evidence, and extends the sourced comparison report with Fable 5's published result without claiming one-trial/five-trial parity.
- Refs: issue #472; `kairyu/bench/adapters/swebench*.py`; `docs/benchmarks.md`; `tests/bench/test_bench_agentic*.py`; `tests/bench/test_bench_compare.py`
