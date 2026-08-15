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

Snapshot date: 2026-08-15. Hardware context: all GPU evidence so far is on
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
- The tiered RTX PRO example is a coding-first product: an unchanged L1 fleet (four Qwen3.8 TP1 vLLM workers + the measured DeepSeek TP4/EP4 DSpark worker) under a nine-role L2 DAG where a Qwen head streams the public answer opening from t=0 (~0.3 s at c1; semantic-TTFT gate ≤2× the DeepSeek-direct row), Qwen test/proposal fan-out feeds a networkless CPU sandbox over a Unix-socket transport, and a verified DeepSeek continuation streams the remainder after the committed opening; non-coding requests skip execution locally. Launcher readiness proves the DAG, executor binding, and a two-input 384-dimensional embedding response. Image chat still reaches only Qwen roles; composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
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

### 2026-08-15 — [amendment] Executor deadline includes queue admission
- What: deployment executor config now declares queue allowance; the client
  shares one wall-plus-queue deadline across retries and the runner rejects
  admission when the residual budget cannot cover the declared wall limit.
- Why: the previous 8s queue plus 10s execution exceeded the 15s client
  deadline, discarding valid work as `unavailable` under contention.
- Refs: ECO-D1; `kairyu/orchestration/execution.py`; PR #488 review

### 2026-08-15 — [amendment] Sandbox transport removes the reverse network path
- What: replaced the bidirectional internal Docker bridge with a shared Unix
  domain socket and `network_mode: none` on the executor; Kairyu mounts the
  socket volume read-only and the deployment executor client supports UDS.
- Why: an internal bridge blocks external routing but still lets hostile
  submitted code connect back to Kairyu, violating ECO-D1's no-egress
  boundary.
- Refs: PR #488 review; ECO-D1; `kairyu/orchestration/execution.py`;
  `examples/qwen3.8-deepseek-v4-8gpu/{compose.yaml,sandbox/runner.py}`

### 2026-08-15 — [design] Coding orchestration with head streaming and sandbox execution
- What: replaced the tiered example's seven-role DAG with a nine-role coding
  DAG on the unchanged L1 fleet; added L2 mechanisms for a t=0 public head
  stream + verified continuation, incremental reasoning streaming, per-role
  sampling/prompt suffixes, and deployment-owned sandbox executors run as
  untrusted-data DAG stages; TTFT gate: public p50 ≤2× DeepSeek-direct per
  concurrency (live c1 smoke ~0.3 s vs the 1.56 s budget).
- Why: frontier-level coding accuracy needs ensemble + execution-grounded
  verification, but the old collect-then-synthesize DAG left first public
  tokens ~10× over the agreed 2×-DeepSeek TTFT budget; only E2E is
  unconstrained, so the answer opening streams while the ensemble runs.
- Refs: ECO-D1..D5 in `docs/design/example-coding-orchestration.md`;
  `kairyu/orchestration/{conductor,execution}.py`;
  `examples/qwen3.8-deepseek-v4-8gpu/`

### 2026-08-15 — [progress] Frontier examples move to Qwen3.8
- What: replaced both Qwen3.6 example surfaces with attested Qwen3.8-27B-FP8,
  selected a measured 32K/no-MTP/piecewise-graph vLLM v0.23.0 L1 envelope,
  ported it unchanged to four TP1 replicas, and passed an eight-GPU product
  chat plus embedding smoke without changing the L2 role DAG or L3 contract.
- Why: DeepSeek remains on the measured `aa0d513027` DSpark build because the
  official v0.23.0 runtime cannot serve the 0731 checkpoint's draft path.
- Refs: `examples/qwen3.8-{27b-1gpu,deepseek-v4-8gpu}/`;
  `examples/qwen3.8-27b-1gpu/MEASUREMENTS.md`

### 2026-08-14 — [progress] Accuracy removal and verification migration completed
- What: removed the named Accuracy suite and its code, fixtures, dependencies,
  tests, examples, and active documentation; retained four explicit eval suites
  and moved Kairyu serving/correctness gates to checkout-only verification.
- Refs: PR #486; `evals/`; `verification/`; `evidence/`;
  `docs/design/verification-framework.md`

### 2026-08-14 — [progress] Accuracy implementation removed
- What: removed the externally migrated Accuracy adapters, judge and published
  comparison paths, fixtures, dependencies, and tests; retained eval tests now
  live under `tests/evals/` and require an explicit retained suite.
- Refs: PR #486; `evals/`; `tests/evals/`; `pyproject.toml`

### 2026-08-14 — [progress] Verification framework split implemented
- What: moved 68 Kairyu gates into checkout-only `verification/` by scope and
  kind, added a strict registry/runner and neutral `evidence/` contracts, and
  kept all 205 tracked legacy result files byte-identical at `bench/results/`.
- Refs: PR #486; `verification/registry.toml`;
  `docs/design/verification-framework.md`

