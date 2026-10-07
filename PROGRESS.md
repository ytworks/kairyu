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

Snapshot date: 2026-09-30. Hardware context: all GPU evidence so far is on
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
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, OpenAI-spec Responses API (live reasoning/tool streams, store endpoints, in-band overflow; m11 D4); System One (Jev wire API) at `/v1/systemone` outside ReplicaPool (m11 D8), GPU-verified with OpenJev in `openjev-diffusiongemma-26b-1gpu` (512-token think-first chat, Jev-style playground)
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI (0.160 GPU-verified 2026-10-06 on DeepSeek-V4.1 and an AUTO model) and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering; Helm supports immutable images, split-role labels, safe rollout/drain, hardened Pods, and ServiceMonitor plus the kind CI drill
- AsyncRequest v1 (opt-in `async_requests`): PostgreSQL-backed non-streaming Chat with tenant-scoped status/result/cancel, lease-fenced workers, shared queue telemetry and bounded request/audit retention; the retention-inclusive three-gateway Kind gate runs in F1c CI. Runner State v1 contracts (Kubernetes observation/reconciliation, fenced drain, failure-domain backoff, PostgreSQL leader lease, WP3.1 scaling policies, WP3.2 decision log) exist as a library
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- The tiered RTX PRO example (DTO-D13, 2026-08-22) puts a bounded Qwen non-thinking route judge in front of five profiles — four single-call direct routes (Qwen non-thinking, Qwen thinking-medium, DeepSeek non-thinking on the re-added `tier2-direct` pool, DeepSeek thinking at the L3 effort; official per-mode sampling fixed on the final unit, vendor-official caps 131072/393216) and the ensemble — selecting per request with fallback to the ensemble; the L2 DSL now has N named `profiles` + a judge with spec-defined `choices`, final-unit sampling overrides (caps min()'d with the caller), and route-aware serving gates. The ensemble (`primary`) profile is the dual-track policy-ensemble L2 DAG (DTO-D1..D12, amended by DTO-D14) over four Qwen3.8 TP1 vLLM workers (no MTP pending c16/c32 evidence) + the measured DeepSeek TP4/EP4 DSpark worker: a Qwen head streams the public opening from t=0 (semantic-TTFT gate ≤2× DeepSeek-direct, inherited); one thinking DeepSeek call writes 4 maximally different policies fanned out to 4 policy-bound Qwen answers in parallel while thinking DeepSeek critically refines a quick Qwen draft; thinking DeepSeek `synthesis` weighs the 5 candidates as peers and writes one better answer, and an inline Qwen thinking-medium (DTO-D14) `audit` (PASS/FAIL, ≤2 refinements, last attempt published on exhaustion) gates the streamed remainder (DTO-D10); a Qwen `image_description` stage runs on image requests only and feeds the text-only DeepSeek roles (DTO-D11); DeepSeek budgets halved to 8192/32768/65536 with a 65536 ceiling and Chat UI default for the Terminal-Bench 900 s turn envelope (DTO-D12). The sandbox executor stays deployed but unreferenced. Last green verify.sh runs 20260825T161729Z (coding) and 20260825T173343Z (generic) on the DTO-D8..D14 served config: coding TTFT rows all not_applicable (the judge routes every coding request to the ungated qwen_think_medium route), generic route-aware stage validation green. Composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
- Replica-pool scale-out examples (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 and DeepSeek TP4+EP4 x 2 behind one public model each; `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see their MEASUREMENTS.md); two vision replica examples (FN-D9 amendment 2026-09-04: DeepSeek-V4-Flash-Vision-Exp TP4+EP4 x 2, Qwen3.8-Flash-Next-FP8 TP4 x 2 on a shared upstream-main SM120 overlay image, Chat UI reasoning-effort dropdown, `verify.sh vision`) are GPU-verified (2026-09-04: pins locked, serving/tool-calling/vision gates PASS, MEASUREMENTS.md written); the Qwen example serves without the recipe's MTP k=3 because prefix caching + MTP corrupts batched output on this vLLM revision (vllm#53912)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
- DeepSeek V4.1 Flash six-GPU example (FN-D9 six-GPU amendment, 2026-09-30): one DP6/EP6 replica on GPUs 0–5 with the 8-GPU example's L2/L3 structure and its own scripts; official-first L1 (pinned vLLM nightly + SM120 overlay, Engram offload, 4K batch / 0.92 from the recipe's memory-bound arm, DSpark 5 with full verification). Serving 102 / 591 / 718 tok/s at c1/c32/c64; gate evidence in its `MEASUREMENTS.md`.
- Qwen3.8 + DeepSeek-V4.1 ensemble example (DTO-D16/D17, 2026-10-01): V4.1 DP6/EP6 (GPU 0–5, the six-GPU example's L1) + Qwen TP1 × 2 (GPU 6, 7). A Qwen judge picks one of two routes: thinking DeepSeek, or the dual-track ensemble with two policies and a three-candidate synthesis. Every role takes images natively; DeepSeek uses the official V4.1 encoder with per-request effort, and the example's overlay continues the floor's assistant prefill. ENSEMBLE criteria loosened and judge fallback moved to `deepseek_think` (DTO-D17 amendments). GPU-verified 2026-10-01: all gates pass, including the ensemble TTFT gate at c1–c32 (c32 at 95.8 % of the limit); 4 of 128 coding requests exceed the 900 s turn envelope after two audit refinements.
- Winnow-routed DeepSeek example `deepseek-v4.1-winnow-8gpu` (VCO-D18, 2026-10-06, PR #640; replaces the checklist-verified OpenJev example, whose evidence stays at `df109a6b`): DeepSeek-V4.1 DP6/EP6 (GPU 0-5), Winnow-12B Q8_0 x 2 (GPU 6 route judge, GPU 7 judgments). Winnow routes each request through System One to VERIFIED or THINK (the caller's effort). VERIFIED is three waves (VCO-D19, PR #641): five DeepSeek drafts beside DeepSeek's MECE requirements (max effort), one Winnow read judging each draft's adoptability and each draft x requirement, then a critical DeepSeek answer; drafts and answer run at the caller's effort. Replies target the next step and drafts carry structured tool calls since 2026-10-07; all nine GPU gates pass with it (verified-route p50 181.6 s, serving c1/c16 p50 77.4/147.6 s; replayed DeepSWE turns 10/10 structured tool calls). Before that change the 2026-10-07 layout (no Qwen, two Winnow replicas) also passed all nine: verified-route p50 141.6 s, serving c1/c16 p50 98.3/151.4 s, serving-routed c16 VERIFIED/THINK p50 282/2.9 s; DeepSWE r1 on the Qwen layout scored 8 of 24 before it was stopped. Open WebUI and the answer page carry over; evidence in its `MEASUREMENTS.md`.
- GGUF via llama.cpp (LCP-D1..D6, 2026-10-04): `upstream: llamacpp` on the `openai` backend; CPU contract gate green on stock b11391; Winnow-12B Q8_0 examples (1 GPU, DP8 ReplicaPool + System One, System One playground) pass all GPU gates (2026-10-05) with llama.cpp's Gemma 4 `required` fix `f072b10` backported
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
- Runner autoscaling WP3.1–WP3.7 and model-cache WP4.1–WP4.7/D3.1–D3.20
  are fail-closed and CPU-tested, including signed artifact identity, verified
  node cache, fenced pre-stage/startup/admission, live PostgreSQL/Kubernetes/Kueue
  authority, leader-fenced CRD publication, and production runtime assembly.
  Deployment, runtime instrumentation, and live acceptance remain open.
- Qwen3.8-Flash-Next MTP speculative decoding stays off in `qwen3.8-flash-next-dp2-8gpu` until upstream fixes vllm#53912 (prefix caching + MTP output corruption on hybrid GDN); single-stream decode 104 vs 175 tok/s
- DTO-D15 (2026-08-26) changed the served tiered-example config: verify.sh coding/generic gates and the digest re-pin are pending before the example status can be claimed green again
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-10-07 — [amendment] Verified replies target the next step; drafts carry structured tool calls (PR #641)
- What: drafts/requirements/answer define the reply as the next assistant message (on agent turns, the move needed now); each draft carries `tool_calls` and reads the caller's tools; judgments also read the conversation; stage reports leave `reasoning_content`. DeepSWE on the two-Winnow layout stopped at 0/113 scored. All nine GPU gates pass.
- Why: drafts wrote tool calls as text and the answer copied it (3/47 turns rejected by the agent); requirements read the task as the request every turn, so agents re-explored instead of stepping; replayed stage reports grew every request.
- Refs: VCO-D19 amendment 2026-10-07 (next step); example `verified.yaml`

### 2026-10-07 — [amendment] Verdict waits validated across exclusions; two-Winnow layout GPU-verified (PR #641)
- What: review fixes to the m1 D8 wait-for-a-parallel-branch amendment: waits join the cycle check; a waited unit's dependencies must precede the target on every request; a dependency counts as done only if, for each head/image exclusion combination, it precedes the target or is excluded (otherwise it is waited for). All nine GPU gates pass on DeepSeek max requirements with `winnow-route`/`winnow-judge`.
- Why: four review findings: verdicts waiting on each other's targets, and image-conditional or head exclusions shifting waves, hung runs or let verdicts read missing outputs.
- Refs: m1 D8 amendment 2026-10-06 (extended); VCO-D19 amendment 2026-10-07; example `MEASUREMENTS.md`

### 2026-10-07 — [amendment] Verified requirements on DeepSeek max; Qwen replaced by a second Winnow (PR #641)
- What: `requirements` moves from Qwen (low) to DeepSeek (max, full conversation). Qwen leaves the example (renamed `deepseek-v4.1-winnow-8gpu`); GPU 6 hosts a second Winnow-12B: `winnow-route` judges only the route, `winnow-judge` (GPU 7) only the judgments. The `{conversation_without_reasoning}` placeholder is withdrawn (no user left). GPU gates pending.
- Why: owner decision after DeepSWE r1 (8 of 24 scored, stopped): Qwen requirements was the wave-1 bottleneck (median 107 s vs drafts 36 s); separate replicas keep judgments from queueing the route decision.
- Refs: VCO-D19 amendment 2026-10-07; m1 D8 withdrawal note; supersedes the two entries below

### 2026-10-07 — [amendment] Correction: Winnow keeps replayed reasoning (PR #641)
- What: corrects the entry below: only Qwen `requirements` drops replayed `reasoning_content`; DeepSeek and Winnow (route judge and `judgments` as configured) are unchanged.
- Why: owner decision: do not drop the reasoning for Winnow.
- Refs: entry below; plan `docs/superpowers/plans/2026-10-07-verified-requirements-context.md`

### 2026-10-07 — [amendment] Qwen requirements reads the conversation without replayed reasoning (PR #641)
- What: new role placeholder `{conversation_without_reasoning}` (`{conversation}` minus assistant `reasoning_content`); the example's Qwen `requirements` uses it, DeepSeek roles keep the full conversation.
- Why: DeepSWE r1 replayed Kairyu's stage reports; Qwen `requirements` overflowed 262,144 tokens and 87/160 VERIFIED turns had no Winnow judgment. Owner: only Qwen and Winnow do without reasoning; DeepSeek needs it.
- Refs: m1 D8 amendment 2026-10-07; VCO-D19 amendment 2026-10-07; plan `docs/superpowers/plans/2026-10-07-verified-requirements-context.md`

### 2026-10-06 — [design] Verified route in three waves; verifiers may wait for a parallel branch (VCO-D19, PR #641)
- What: VERIFIED = wave 1 DeepSeek five drafts (one call, caller's effort) beside Qwen requirements (MECE, ≤16, low effort); wave 2 one Winnow read per draft (adoptable?) and per draft x requirement (met?); wave 3 DeepSeek (caller's effort) writes the best answer from all of them critically. Framework: a verifier may read a unit running beside its target; the verdict waits for it (deadlock-free validation).
- Why: owner design; without the framework change wave 1 could not run in parallel (owner authorized the change).
- Refs: VCO-D19; m1 D8 amendment 2026-10-06; plan `docs/superpowers/plans/2026-10-06-winnow-verified-three-wave.md`

### 2026-10-06 — [design] Verified example rebuilt as Winnow-routed answers (VCO-D18, PR #640)
- What: `deepseek-v4.1-openjev-verified-8gpu` → `deepseek-v4.1-qwen3.8-winnow-8gpu`: DeepSeek GPU 0-5, Qwen3.8-27B GPU 6 (idle pool), Winnow-12B GPU 7. Winnow judges THINK/VERIFIED with the old criteria; VERIFIED = one max-effort DeepSeek call. Verified-tool route, checklist DAG, calibration and OpenJev removed; UIs kept; gates rebuilt with the c1/c4/c8/c16 plan.
- Why: owner restarts the verified route from a plain max-effort answer on a layout that also hosts Qwen and Winnow.
- Refs: VCO-D18 (supersedes VCO-D17), `docs/superpowers/plans/2026-10-06-deepseek-v41-winnow-routed-8gpu.md`

### 2026-10-05 — [amendment] OpenAI Responses compatibility in L3 (m11 D4)
- What: `/v1/responses` follows the pinned OpenAI spec, so Codex 0.160 and the SDKs work unmodified. It adds:
  - the OpenAI error envelope with `param` (400/404/405, `previous_response_not_found`, 503 `slow_down` for transient overload)
  - omitted `max_output_tokens` meaning the remaining context
  - live reasoning, text and tool-call streams with preamble and in-band gates
  - `response.in_progress` snapshot heartbeats
  - reasoning items with `krs1.` tokens, replayed into `reasoning_content`
  - in-band `context_length_exceeded`
  - retrieve, delete, `input_items`, `input_tokens`, cancel and `compact`
  - input images and web-search declarations

  Unsupported fields are typed 400s.
- Why: Codex retried every 1024-capped turn, aborted on comment keep-alives, never compacted after an HTTP 400, and lost reasoning and preambles. An earlier attempt that widened into L1/L2 was closed, so this one only maps existing capabilities in L3.
- Refs: m11 D4 amendment 2026-10-05; `kairyu/entrypoints/server/responses_*.py`; `tests/server/test_responses_{contract,stream,inputs}.py`

### 2026-10-05 — [amendment] llama.cpp `n` limited to 1; Winnow example fixes (PR #620 review)
- What: `upstream: llamacpp` rejects `n > 1` before dispatch (`max_n=1`), and the contract gate's `n`-above-slots row is removed. In the Winnow examples, `down`/`status`/`logs` no longer need 30 GiB free; the playground sends WebP as PNG; the streaming tool-call gate assembles the deltas instead of searching for the `tool_calls` key.
- Why: owner review. llama-server reports usage per candidate (the first candidate's when unary, one usage chunk per candidate when streaming), so Kairyu under-reported and under-billed completion tokens. System One forwards images untouched, and Winnow's build cannot decode WebP. Plain text chunks also carry `"tool_calls": null`.
- Refs: LCP-D2 in `docs/design/llamacpp-upstream.md`; PR #620 review 5409395128

### 2026-10-05 — [amendment] Winnow examples GPU-verified; Gemma 4 `required` backport (PR #620)
- What: both Winnow-12B examples pass all six `verify.sh` gates on RTX PRO 6000 (1 GPU; DP8 on 8 GPUs). The examples add llama.cpp `f072b10` as a fifth winnow-server patch, registered in Winnow's own `runtime.lock.json`. They also add a Jev-style System One playground on `:3001`, and the UIs now listen on all interfaces. Kairyu code is unchanged.
- Why: at b11036 the Gemma 4 grammar ignores `tool_choice: "required"`. The named-tool adaptation (LCP-D3) then got text, and Kairyu failed closed with 502. The source re-read had missed this. Owner approved the backport (example-owned runtime), the playground and the public binds.
- Refs: LCP-D3 and the Winnow amendment in `docs/design/llamacpp-upstream.md`; `examples/winnow-12b-q8-*/MEASUREMENTS.md`
