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
- M20 (2026-10-05, G6 P-C2 reopened): full OpenAI Responses API compatibility so
  Codex, the OpenAI SDKs and the Agents SDK run unmodified
  (`docs/design/m20-responses-compat.md`).

## Current Status

Snapshot date: 2026-10-05. Hardware context: all GPU evidence so far is on
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
| M20 Responses compatibility | In progress: Phase 0 (WP-00–05) done; Phase 1: 06/06b/12a/07 done |

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
- G6: P-A, P-B1–P-B4, P-C3/C4 green (incl. Open WebUI P-B3 browser gate); P-C2 reopened for M20; remaining P-C gates continue
- #150 TP8 long-generation stability gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Incremental architecture-state paths for Qwen3.6 and DeepSeek V4 plus an explicit recompute diagnostic mode; DeepSeek EP2/4/8 Attention-DP and direct packed-FP4 execution are implemented, with SM120 single-kernel and two-rank NCCL smokes green
- Device-side sampling, penalties, spec verification, page-table caching; TP step headers sleep on Gloo while fixed-layout delta payloads use the bounded NCCL model group and rare controls remain Gloo objects; structured masks stay on CUDA with only selected IDs returned to the host matcher; deterministic n-gram/EAGLE-3/MTP drafts preserve T>0 and penalized sampling
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API; System One (Jev wire API) at `/v1/systemone` outside ReplicaPool (m11 D8), GPU-verified with OpenJev in `openjev-diffusiongemma-26b-1gpu` (512-token think-first chat, Jev-style playground)
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering; Helm supports immutable images, split-role labels, safe rollout/drain, hardened Pods, and ServiceMonitor plus the kind CI drill
- AsyncRequest v1 (opt-in `async_requests`): PostgreSQL-backed non-streaming Chat with tenant-scoped status/result/cancel, lease-fenced workers, shared queue telemetry and bounded request/audit retention; the retention-inclusive three-gateway Kind gate runs in F1c CI. Runner State v1 contracts (Kubernetes observation/reconciliation, fenced drain, failure-domain backoff, PostgreSQL leader lease, WP3.1 scaling policies, WP3.2 decision log) exist as a library
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-medium, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. The ensemble (`primary`) profile is the dual-track policy-ensemble L2 DAG (DTO-D1..D12, amended by DTO-D14) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline Qwen thinking-medium (DTO-D14) `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). The sandbox executor stays deployed but unreferenced. Last green verify.sh runs 20260825T161729Z (coding) and 20260825T173343Z (generic) on the DTO-D8..D14 served config: coding TTFT rows all not_applicable (the judge routes every coding request to the ungated qwen_think_medium route), generic route-aware stage validation green. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Replica-pool scale-out examples (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 and DeepSeek TP4+EP4 x 2 behind one public model each; `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see their MEASUREMENTS.md); two vision replica examples (FN-D9 amendment 2026-09-04: DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2, Qwen3.8-Flash-Next-FP8 TP4 x 2 on a shared upstream-main SM120 overlay image, Chat UI reasoning-effort dropdown, `verify.sh vision`) are GPU-verified (2026-09-04: pins locked, serving/tool-calling/vision gates PASS, MEASUREMENTS.md written); the Qwen example serves without the recipe's MTP k=3 because prefix caching + MTP corrupts batched output on this vLLM revision (vllm#53912)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
- DeepSeek V4.1 Flash six-GPU example (FN-D9 six-GPU amendment, 2026-09-30): one DP6/EP6 replica on GPUs 0–5 with the 8-GPU example's L2/L3 structure and its own scripts; official-first L1 (pinned vLLM nightly + SM120 overlay, Engram offload, 4K batch / 0.92 from the recipe's memory-bound arm, DSpark 5 with full verification). Serving 102 / 591 / 718 tok/s at c1/c32/c64; gate evidence in its `MEASUREMENTS.md`.
- Qwen3.8 + DeepSeek-V4.1 ensemble example (DTO-D16/D17, 2026-10-01): V4.1 DP6/EP6 (GPU 0–5, the six-GPU example's L1) + Qwen TP1 × 2 (GPU 6, 7). A Qwen judge picks one of two routes: thinking DeepSeek, or the dual-track ensemble with two policies and a three-candidate synthesis. Every role takes images natively; DeepSeek uses the official V4.1 encoder with per-request effort, and the example's overlay continues the floor's assistant prefill. ENSEMBLE criteria loosened and judge fallback moved to `deepseek_think` (DTO-D17 amendments). GPU-verified 2026-10-01: all gates pass, including the ensemble TTFT gate at c1–c32 (c32 at 95.8 % of the limit); 4 of 128 coding requests exceed the 900 s turn envelope after two audit refinements.
- Checklist-verified answers example (m1 D8 / VCO-D1..D6, 2026-10-01): DeepSeek-V4.1 DP6/EP6 (GPU 0-5) writes, OpenJev x 2 (GPU 6, 7) judges through System One, Kairyu L2 checklist verifiers (Jev-shaped questions, tau_hi, an acceptance read; no rule-based checks since PR #619) publish `kairyu_verification` with every answer; tau_hi 0.9966 at alpha 0.10 on InFoBench expert labels (held-out upper bound 8.7 %). Per-claim G1 (OpenJev claim support) failed calibration on RAGTruth/PRM800K/FEVER and is advisory (VCO-D11); the guarantee covers tau_hi requirements plus deterministic grounding checks. Two-stage extraction, a source/action-only state builder and format-scoped units (VCO-D12/D13). All GPU gates pass on `e81db571` (2026-10-02): implicit recall 0.875, InFoBench gold recall 0.972, serving guaranteed 44-53 % (was 25-38 %), routed verified 65 %; owner latency target (p50 <= 180 s) not met (InFoBench p50 154-342 s). Evidence in its `MEASUREMENTS.md`.
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed. Ruled out as causes: process split (#333, `no_material_reduction`, ratio 0.92 vs ≤0.90) and admission depth beyond two steps (#318, `no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV remains disabled after its calibrated re-bake failed exact-output and decode-envelope checks
- Frontier full-checkpoint 262K/1M correctness/performance evidence, DeepSeek EP4/EP8 topology lock, CUDA Graph pointer stability, MTP/DSpark selection, 30-minute soak, and failure recovery remain open
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Runner autoscaling WP3.1–WP3.7 and model-cache WP4.1–WP4.7/D3.1–D3.20 are fail-closed and CPU-tested (signed artifact identity, verified node cache, fenced pre-stage/startup/admission, live PostgreSQL/Kubernetes/Kueue authority, leader-fenced CRD publication, production runtime assembly); deployment, runtime instrumentation, and live acceptance remain open
- Qwen3.8-Flash-Next MTP speculative decoding stays off in `qwen3.8-flash-next-dp2-8gpu` until upstream fixes vllm#53912 (prefix caching + MTP output corruption on hybrid GDN); single-stream decode 104 vs 175 tok/s
- DTO-D15 (2026-08-26) changed the served tiered-example config: verify.sh coding/generic gates and the digest re-pin are pending before the example status can be claimed green again
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-10-05 — [amendment] Responses error contract; Codex-retryable backpressure (M20 WP-07)
- What: every OpenAI-envelope error carries nullable `param` (Chat errors, backend failures, middleware via `error_classifier.render_openai`). `/v1/responses*`, `/v1/conversations*`, `/v1/alpha*` render framework 422→400 (`param` from loc), bad JSON 400 `invalid_json`, 404 `Invalid URL (…)`, 405 and middleware errors in the envelope; transient overload/tenant quota there is 503 `slow_down` + `Retry-After`/`retry-after-ms` (refill-derived wait), a reservation above the bucket capacity 429 `tenant_budget_too_small`; unknown `previous_response_id` 400 `previous_response_not_found` (tenant-blind); AUTO delegated errors re-rendered; in-band generation failure `server_error`. Chat/Messages keep their 429s.
- Why: owner decision O-2: Codex treats any 429 as terminal but retries 503 `slow_down` after `Retry-After` (matrix `backpressure-retry` passes on 0.160.0; 0.153.4 ignores Retry-After); spec-required `param` (M-ST-5/6), D-d.
- Refs: m7 D5 amendment 2026-10-05; m20 D10, D-d, D-g; `kairyu/entrypoints/server/{error_classifier,middleware,tenancy}.py`, `responses/{errors,framework_errors}.py`

### 2026-10-05 — [amendment] Prompt overflow is `context_length_exceeded` on every surface (M20 WP-04)
- What: typed `ContextLengthExceededError` + one `resolve_output_budget` (engine loop, `kairyu-proc` preflight with the child's model-config limit); upstream 400 overflow bodies classified with a fixed public message; L3 `error_classifier` renders Chat 400 `param:"messages"`, Messages "prompt is too long", Responses unary 400 `param:"input"` and in-band `response.failed{context_length_exceeded}` on pre-dispatch and buffered/tool stream paths incl. AUTO (an AUTO direct route sends the client prompt). Internal orchestration-stage overflow stays a server error (`internal_stage_context_overflow`). Known limits: AUTO `_stream_orchestrator` streams → WP-18; live-path item events → WP-17a; upstream Chat/Messages streams stay in-band (deviation).
- Why: Codex compacts only on in-band `context_length_exceeded`; overflow was 400 `code:null`, a vLLM 502, or an AUTO pre-stream 400; the proc preflight assumed 16 output tokens.
- Refs: m9 D6 amendment 2026-10-05; m20 D10 (WP-04 part) pending the m20 doc; `kairyu/engine/{request_errors,openai_errors,model_limits}.py`; `kairyu/entrypoints/server/{error_classifier,chat_errors,responses_errors}.py`

### 2026-10-05 — [design] M20 Responses compatibility proposed; schema gate (WP-01)
- What: m20 (Proposed) records D1–D22, deviations D-a..D-j, the WP roadmap, the admission table, the gap matrix and the m11 supersession table; g6 P-C2 reopened. WP-01 gates every test: pinned OpenAPI closure, ASGI recorder, `divergences.toml` (`CONTRACT_STRICT=1`), opt-in wire capture, strict SDK.
- Why: owner decisions 2026-10-05 (full scope, O-1..O-5) after the audit (80 + 31 gaps, six Codex P0).
- Refs: `docs/design/m20-responses-compat.md`; m11 D4 pointer amendment; `tests/contracts/`; `scripts/vendor_openai_schema.py`

### 2026-10-05 — [amendment] Responses: Codex P0 fixes (M20 WP-03)
- What: `/v1/responses` drops the 1024 default output cap and the 4096 compaction cap (remaining context, #496); every SSE path repeats `response.in_progress` after 15 s data silence instead of `: keep-alive` comments; live/indexed hosted `web_search` is accepted, echoed, and hidden; startup warns when a tenant token bucket cannot hold a model's `max_model_len` reservation.
- Why: audit vs codex-rs rust-v0.160.0: Codex never sends `max_output_tokens` and retries `incomplete` 5x, resets its 300 s idle timer only on data events, and declares live web search under full-access sandboxes (terminal 400).
- Refs: m11 D4 "Codex P0 amendment"; `kairyu/entrypoints/server/{responses_service,stream_util,tenant_budget}.py`; `docs/deployment.md`

### 2026-10-04 — [design] Verified DAG drops agent-turn wording (PR #619)
- What: extractors no longer read `{tools}` or target "this one message"; adoption asks "is this point necessary to answer the request?" without tools; the summary covers earlier turns; the repair has no tool-call instructions; extractor limits back to 32,768.
- Why: owner decision: tool requests take the verified-tool route, so the guarantee route assumes a complete answer.
- Refs: VCO-D17 in `docs/design/example-verified-checklist-orchestration.md`; example `verified.yaml`, `verified-always.yaml`

### 2026-10-04 — [design] Verified example: tool route replaces step verification (PR #619)
- What: Jev routes a request that requires a tool call to TOOL: one DeepSeek call at max effort with the caller's tools, unverified (kairyu-verified THINK/TOOL/VERIFIED; kairyu-verified-always TOOL/VERIFIED). The STEP route and `verified_step` profile are removed.
- Why: owner decision. Verification failed correct intermediate agent steps and repairs jumped to the final move; step verification did not remove that risk.
- Refs: VCO-D17 (supersedes VCO-D16) in `docs/design/example-verified-checklist-orchestration.md`; example `verified.yaml`, `verified-always.yaml`

### 2026-10-03 — [design] Verified example judges an agent turn as one step (PR #619 S3)
- What: new profile `verified_step` and Jev route label STEP (kairyu-verified: THINK/STEP/VERIFIED; kairyu-verified-always: STEP/VERIFIED, no think route). Step points come from the task and the latest tool results; coverage and acceptance ask whether the reply is a sound next step (A0) over request, recent conversation (bounded `query`), summary and reply. Repairs in both profiles rewrite the same message in the draft's frame (B1); a step repair gets only points read below 0.5 (B2). Extractors may use 65,536 tokens (C2). STEP thresholds are placeholders until labelled DeepSWE turns (V3).
- Why: complete-answer criteria failed sound mid-task turns and their repairs drifted to submission (closed PR #618: 73/83 turns hit the refinement limit).
- Refs: PR #619; plan `docs/superpowers/plans/2026-10-03-jev-verified-minimal.md` (A0, B1, B2, C2); example `verified.yaml`, `verified-always.yaml`
