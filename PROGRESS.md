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
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-medium, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. The ensemble (`primary`) profile is the dual-track policy-ensemble L2 DAG (DTO-D1..D12, amended by DTO-D14) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline Qwen thinking-medium (DTO-D14) `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). The sandbox executor stays deployed but unreferenced. Last green verify.sh runs 20260825T161729Z (coding) and 20260825T173343Z (generic) on the DTO-D8..D14 served config: coding TTFT rows all not_applicable (the judge routes every coding request to the ungated qwen_think_medium route), generic route-aware stage validation green. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Replica-pool scale-out examples (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 and DeepSeek TP4+EP4 x 2 behind one public model each; `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see their MEASUREMENTS.md); two vision replica examples (FN-D9 amendment 2026-09-04: DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2, Qwen3.8-Flash-Next-FP8 TP4 x 2 on a shared upstream-main SM120 overlay image, Chat UI reasoning-effort dropdown, `verify.sh vision`) are GPU-verified (2026-09-04: pins locked, serving/tool-calling/vision gates PASS, MEASUREMENTS.md written); the Qwen example serves without the recipe's MTP k=3 because prefix caching + MTP corrupts batched output on this vLLM revision (vllm#53912)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- V41C six-plus-two critical ensemble implementation is started from main: budgets, terminal errors and the typed chat backend contract are implemented. Six-GPU numerical preflight remains FAIL; native AUTO/capacity contracts, topology selection and mandatory primary performance gates remain pending (`docs/design/v41-critical-ensemble.md`).
- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- Frontier full-checkpoint 262K/1M correctness/performance evidence, DeepSeek EP4/EP8 topology lock, CUDA Graph pointer stability, MTP/DSpark selection, 30-minute soak, and failure recovery remain open
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Qwen3.8-Flash-Next MTP speculative decoding stays off in `qwen3.8-flash-next-dp2-8gpu` until upstream fixes vllm#53912 (prefix caching + MTP output corruption on hybrid GDN); single-stream decode 104 vs 175 tok/s
- DTO-D15 (2026-08-26) changed the served tiered-example config: verify.sh coding/generic gates and the digest re-pin are pending before the example status can be claimed green again
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-09-13 — [progress] V41C: retain failed six-GPU numerical preflight
- What: main-pinned TP2/DP3/EP6 preparation stopped on one numerical-reference outlier. A same-fixture eight-head slice is bit-exact to the 32-head slice; both retain FAIL. Original services are healthy. Native non-thinking and explicit 50/75/100 budgets pass CPU rendering checks using existing settings.
- Why: source feasibility and successful service restoration cannot substitute for candidate startup or GPU correctness; preserve exact failed evidence and distinguish a reference discrepancy from a head-shape defect.
- Refs: PR #601; V41C-D5; existing ensemble `MEASUREMENTS.md` and `measurements/20260913T134938Z-v41-preflight/`.

### 2026-09-13 — [progress] V41C: preserve native chat at the backend boundary
- What: add immutable ChatPrompt/ChatMessage and native OpenAI wire support for conversation, reasoning, tool history and ordered media. Reuse existing media lifecycle; unsupported engines reject explicitly. Missing-usage failures retain reserved charges and the original error. Related validation passes 545 tests.
- Why: text flattening and the existing media-only carrier cannot preserve an ordinary assistant/tool conversation; AUTO derivation and exact capacity accounting remain separate work.
- Refs: PR #601; V41C-D4 and validation record in `docs/design/v41-critical-ensemble.md`.

### 2026-09-13 — [progress] V41C: separate configured private generation budgets
- What: Conductor and MoA use their configured internal ceiling independently of public completion MAX. Final intent and the no-private-ceiling fallback remain unchanged. Three small-public-MAX regressions fail before the fix; combined related validation passes 588 tests.
- Why: a short requested public answer must not truncate planning, proposals or verification; resource reservations must reflect the actual configured private work.
- Refs: PR #601; V41C-D2 and validation record in `docs/design/v41-critical-ensemble.md`.

### 2026-09-13 — [progress] V41C: terminal AUTO errors retain their cause
- What: preserve selected-final/verifier failures through Conductor and unary/SSE APIs; safe upstream 400/422 remain validation errors, including after a committed head. Internal draft text cannot leak as a failed final stream's fallback. Related tests pass; capacity prevention and GPU gates remain pending.
- Why: a known upstream rejection must not become an unrelated empty-output 502 or successful head-only response; retain partial public output and known usage while preserving optional-role best-so-far policy.
- Refs: PR #601; issue #599; V41C-D3 in `docs/design/v41-critical-ensemble.md`.

### 2026-09-13 — [design] V41C: six-plus-two critical ensemble implementation
- What: accept four Qwen candidates on two replicas plus an independent V4.1 candidate, DeepSeek requirements/review/reconstruction/audit, and incremental draft implementation from main. Admission records cover private-budget separation and terminal error propagation; remaining contracts and GPU gates are open.
- Why: preserve full input, reasoning budgets and required stages while replacing the legacy V4 topology without restoring closed example-specific infrastructure.
- Refs: `docs/design/v41-critical-ensemble.md`; approved plan `2026-09-13-v41-six-gpu-critical-ensemble.md`; issues/PRs #595, #598, #599, #600.
