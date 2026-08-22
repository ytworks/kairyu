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
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-low, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. Not GPU-measured. The ensemble (`primary`) profile is the unchanged dual-track policy-ensemble L2 DAG (DTO-D1..D12) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline thinking-DeepSeek `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). No general profile or judge; the sandbox executor stays deployed but unreferenced. Last green verify.sh run 20260818T025710Z (TTFT PASS c1/8/16/32, c32 1.87×→0.67×) predates DTO-D8..D12 — GPU re-verification pending. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
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

### 2026-08-22 — [design] Qwen-judged five-route selection for the tiered example (DTO-D13)
- What: L2 DSL `general_roles`/`CODE|GENERAL` judge generalized to N named
  `profiles` + `profile_judge {choices: [{profile,label,criteria}], fallback}`
  (verdict attached pre-admission; image requests offered only image-capable
  profiles; head-disable short-circuit and keyword fallback removed); the
  final unit may declare sampling (style = policy, max_tokens = cap); DSL
  token bounds → 393216. Example: Qwen non-thinking judge → qwen_direct /
  qwen_think_low / deepseek_direct (`tier2-direct`) / deepseek_think /
  ensemble; route-aware verify gates. Ensemble + Chat UI cap unchanged.
- Why: owner decision 2026-08-22 — fastest route that still answers well.
- Refs: DTO-D13; docs/design/example-dual-track-orchestration.md; examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-20 — [design] Peer synthesis + audit loop, image_description, halved DeepSeek budgets (DTO-D10..D12)
- What: tiered example `compose` → `synthesis` weighing critique + 4 policy
  answers as peers; new thinking-DeepSeek `audit` verifier on the final unit
  (PASS/FAIL, `max_refine_depth: 2`, remainder published after the verdict);
  Qwen `image_description` (`requires: image`) feeds the DeepSeek roles on
  image requests only; DeepSeek tiers 16384/65536/131072 → 8192/32768/65536,
  ceiling/Chat UI/harness cap 131072 → 65536; budget `{19, 2}`; 11 roles.
- Why: owner decisions (2026-08-20); Terminal-Bench turns timing out against
  the 900 s agent budget on the serial DeepSeek chain. GPU gates not re-run.
- Refs: DTO-D10..D12; examples/qwen3.8-deepseek-v4-8gpu/; tests/unit/test_tiered_frontier_examplectl.py

### 2026-08-20 — [design] Verified final unit streams deferred; image-conditional roles
- What: a verifier on the final unit is accepted in `Conductor.stream`: the
  unary verify/refine loop runs first and the text is published once after
  the committed head (n>1 skips the verifier, `skipped:intent`); the DTO-D9
  floor retry runs before the verdict and refinements reopen the scaffold.
  Roles gain `requires: image`: skipped entirely (`skipped:condition`) on
  text requests, dependents render the slot as ""; head/final/verifier/
  executor cannot be conditional; the worker must accept images.
- Why: owner decision for the tiered example (synthesis audit loop, image
  description stage — DTO-D10/D11); amends m11 A5 and DTO-D9 (#547).
- Refs: kairyu/{orchestration,dsl,deploy}; docs/design/m11-product.md

### 2026-08-20 — [design] Public-output floor for the thinking final unit (DTO-D9)
- What: spec-level `public_output_floor` + role `reasoning_close_tag`: the
  final unit's attempt 0 thinks within budget−floor, and the existing bounded
  empty-output re-dispatch continues the captured reasoning after a forced
  close with the reserve, reclaimed as public. Example: floor 256 on compose.
  Fixes #546–#549: direct-API bounds; verified-final/MoA rejected; n>1 no floor.
- Why: issue #542 — a small caller max_tokens was eaten whole by thinking
  and 502'd after full DAG cost; owner chose option 1. GPU gates not re-run
  (floor is 0.2% of the recorded 131072-cap runs).
- Refs: DTO-D9; issue #542; kairyu/{dsl,orchestration}; examples/qwen3.8-deepseek-v4-8gpu/

### 2026-08-20 — [progress] Chat UI reasoning-effort dropdown provisioned by the launcher
- What: the tiered example's `up` installs and self-verifies a globally
  active Open WebUI filter whose enum user valve renders Reasoning Effort as
  a Chat Controls dropdown (default/low/high/max) forwarded as
  `reasoning_effort`; re-up refreshes content without flipping activation off.
- Why: owner — the DTO-D6 Chat UI knob must be selectable, and the pinned
  Open WebUI v0.11.0 (and upstream main) only offers a free-text field.
- Refs: DTO-D6; examples/qwen3.8-deepseek-v4-8gpu/{control.py,
  webui-reasoning-effort-filter.py,README.md}

### 2026-08-20 — [amendment] Chat effort aliases match the pinned UI
- What: /v1/chat/completions now accepts the same OpenAI-style aliases as
  Responses and normalizes minimal/low→low, medium/high→high, xhigh/max→max.
  This corrects the native-level-only Chat contract in the DTO-D8 entry below.
- Why: pinned Open WebUI v0.11.0 initializes Reasoning Effort to medium;
  the prior strict low|high|max schema rejected its default request with 422.
- Refs: DTO-D8 (revised); PR #536 review; kairyu/entrypoints/server/protocol.py

### 2026-08-20 — [amendment] DTO-D8 budgets raised to the vendor starting values
- What: DeepSeek effort tiers 4096/16384/32768 → 16384/65536/131072
  (fallback = high tier), internal_max_tokens → 131072, DSL token bounds
  32768 → 131072, Chat UI default and harness public max_tokens → 131072.
  Corrects the entry below; admission upper bound now ~32× the pre-DTO-D8
  ceiling.
- Why: owner — the TTFT gate is head-based with a paired same-concurrency
  DeepSeek-direct denominator and E2E is unconstrained by design, so the
  ≤2×-DeepSeek budget never constrained DeepSeek role budgets; use the
  vendor-recommended starting values outright.
- Refs: DTO-D8 (revised); examples/qwen3.8-deepseek-v4-8gpu/; kairyu/dsl/spec.py

### 2026-08-20 — [design] Vendor-official sampling + effort-graded token budgets (DTO-D8)
- What: RoleSamplingSpec gains top_k/min_p/presence/repetition penalties and
  `max_tokens_by_effort` {low,high,max} (inherit roles; think+answer budget,
  min()'d vs public cap/internal ceiling). auto-max: official Qwen sampling,
  DeepSeek T→1.0 + caps 4096/16384/32768 per L3 effort, internal_max_tokens
  →32768; /v1/responses maps reasoning.effort (minimal..xhigh normalized).
  GPU gates not re-run — re-verify next window.
- Why: owner + vendor research — effort scales the token budget while
  sampling stays effort-invariant; T=1.0 stays anti-greedy (#509 rationale).
- Refs: DTO-D8; PR #536; kairyu/{dsl,orchestration,entrypoints/server};
  examples/qwen3.8-deepseek-v4-8gpu/
