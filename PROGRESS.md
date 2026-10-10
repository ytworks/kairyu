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
- Quyet-routed DeepSeek example `deepseek-v4.1-quyet-8gpu` (VCO-D21/D22, PR #645; renamed from `deepseek-v4.1-winnow-8gpu`): DeepSeek-V4.1 DP6/EP6 (GPU 0-5), Quyet-1.0-Large x 2 (vLLM bf16 + the example's System One adapter; GPU 6 route judge, GPU 7 judgments and form check), used within Quyet's 6,000-token state. One public model `kairyu-verified-tool`; routing (TOOL/THINK) and the think route as in VCO-D20. The verified tool route is rebuilt: 10-16 DeepSeek candidate calls in one request, six Quyet judgments per candidate, a DeepSeek move that batches independent safe calls, and a seven-question Quyet form check with up to two DeepSeek fixes (`kairyu_verification` reports it); requirements removed; `kairyu/` unchanged. GPU 2026-10-11: 8 of 9 gates pass (serving p50 c1/c16 74.0/131.1 s vs Winnow 137.2/208.6 s, every reply a structured call, 0 Quyet states cut); `verified-tool-route` fails 2/12 and the real-turn route replay misroutes 11/64 (Winnow 1/64), final submit turns routed THINK; fix pending owner decision. The Winnow layout's last gates (2026-10-09, all nine pass) stay in its `MEASUREMENTS.md`; its DeepSWE r2 stopped at 24 passes of 42 scored (`deepswe-verified-tool-4w-20261009-r2`).
- GGUF via llama.cpp (LCP-D1..D6, 2026-10-04): `upstream: llamacpp` on the `openai` backend; CPU contract gate green on stock b11391; Winnow-12B Q8_0 examples (1 GPU, DP8 ReplicaPool + System One, System One playground) pass all GPU gates (2026-10-05) with llama.cpp's Gemma 4 `required` fix `f072b10` backported
- Quyet-1.0-Large System One example `quyet-1.0-large-1gpu` (FN-D9 Quyet amendment, PR #642, 2026-10-10): one GPU, System One only (Jev-family decision model, no chat): Kairyu publishes `/v1/systemone`; an example-owned adapter keeps the `quyet` package's prompt and calibration and reads option-letter logprobs from an internal, batch-invariant vLLM v0.31.0. All nine gates pass: JevBench public 231 through Kairyu 211 correct vs 209 for the package's own run (Brier 0.141/0.142, ECE 0.038/0.042, p50 0.10 s), 308/309 confident answers keep the package's top option, TypeSafe's SDK works (`models.list()` lacks Kairyu's `release_date`). Evidence in its `MEASUREMENTS.md`.
- GLM-5.3-Flash six-GPU example `glm-5.3-flash-6gpu` (FN-D9 GLM amendment, PR #644, 2026-10-11): official FP8 on stock vLLM v0.31.0 as one TP2 x DP3 / EP6 replica with MTP 3 on GPUs 0-5 (four GPUs left no 1M KV pool), `deepseek-v4.1-flash-6gpu` L2/L3, `kairyu/` unchanged. All nine gates pass (run `20261011-r2`): 70.6 / 298.3 / 455.7 tok/s at c1 / c16 / c64, needle at 1,039,890 tokens, 80 max-effort answers complete. Evidence in its `MEASUREMENTS.md`.
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

### 2026-10-11 — [design] Quyet replaces Winnow; verified tool route judges candidates and checks form (VCO-D21/D22, PR #645)
- What: `deepseek-v4.1-winnow-8gpu` -> `deepseek-v4.1-quyet-8gpu`: two Quyet-1.0-Large replicas (as in `quyet-1.0-large-1gpu`) replace Winnow, states bounded to Quyet's 6,000 tokens (route 12,000 chars, judgments 8,000, form check 6,000). TOOL route: 10-16 DeepSeek candidate calls (one request), 6 Quyet judgments each (32 per read), a DeepSeek move batching independent safe calls, 7-question Quyet form check (p >= 0.5) with up to 2 DeepSeek fixes; requirements removed. Routing and think route unchanged; `kairyu/` unchanged.
- Why: owner decisions: decisions on a calibrated Jev-family model within its documented input; choose the move needed now from many judged candidates, cover more safe work per move, and fix the reply's form (announced calls, calls as text, finishing alone) before publishing. Threshold 0.5 because VCO-D16's 0.99 repaired near-passes and changed moves.
- Refs: VCO-D21, VCO-D22; plan `docs/superpowers/plans/2026-10-11-deepseek-v4.1-quyet-8gpu.md`

### 2026-10-11 — [progress] Quyet example: 8 of 9 GPU gates pass; routing misses final agent turns (PR #645)
- What: l1, routing, think-route, effort, fallback, serving (72/72 structured calls; p50 c1/c16 74.0/131.1 s), serving-routed, browser pass; `verified-tool-route` fails 2/12 (`own-check` routed THINK). Route replay of 64 real DeepSWE turns: 11 routed THINK (Winnow 1), all final turns before submit. Judgment reads were refused (400) until candidates went into the question text.
- Refs: VCO-D21/D22; example `MEASUREMENTS.md`

### 2026-10-11 — [progress] GLM-5.3-Flash six-GPU example: all nine GPU gates pass (PR #644)
- What: `examples/glm-5.3-flash-6gpu` serves official FP8 GLM-5.3-Flash on stock vLLM v0.31.0 as one TP2 x DP3 / EP6 replica with MTP 3 (GPUs 0-5) behind the `deepseek-v4.1-flash-6gpu` L2/L3; `kairyu/` unchanged. Run `20261011-r2`: 70.6 / 298.3 / 455.7 tok/s at c1 / c16 / c64, needle found at 1,039,890 tokens, 80 max-effort answers complete.
- Why: planned on four GPUs; the fit probe left 1.81 GiB of KV memory per GPU on TP4 against 7.56 GiB for a 1M request, so the owner-approved six-GPU fallback. The Chat UI sends only the effort: Kairyu's legacy chat path rejects `chat_template_kwargs` on text requests.
- Refs: FN-D9 GLM amendment; plan `docs/superpowers/plans/2026-10-10-glm-5.3-flash-example.md`; example `MEASUREMENTS.md`

### 2026-10-10 — [amendment] Token-count review fixes (m9 D1, PR #643 review)
- What: `ReplicaPool` counts tokens on placeable replicas only; vLLM chat counts decline tools under `tool_choice: none`.
- Why: a drained replica with an older config turned count_tokens/input_tokens into 400 for inputs the pool generates; vLLM's `--exclude-tools-when-tool-choice-none` drops those tools from generation but not from `/tokenize`, and Kairyu cannot see the flag.
- Refs: m9 D1 amendment 2026-10-10; `kairyu/orchestration/replica.py`, `kairyu/engine/openai_backend.py`

### 2026-10-10 — [progress] Token counts equal billed prompt tokens on live vLLM stacks (#621, PR #643)
- What: V1 `qwen3.8-27b-1gpu` and V2 `deepseek-v4.1-flash-8gpu`: count_tokens / input_tokens equal the billed input tokens in all 14 cases (text, system+tools, tool transcript, effort high); the main build undercounted them by 12-234 and 30-212 tokens.
- Refs: m9 D1 amendment 2026-10-10; PR #643

### 2026-10-10 — [amendment] Token counts tokenize what generation sends (m9 D1, #621, PR #643)
- What: `/v1/messages/count_tokens` and `/v1/responses/input_tokens` pass the generation `GenerationRequest` to the backend; vLLM upstreams build the `/tokenize` chat body from the dispatch payload (messages, tools, template kwargs incl. vLLM's `reasoning_effort`/`enable_thinking` merge). Native/mock counts unchanged; images and llama.cpp stay declined.
- Why: on vLLM upstreams without a Kairyu template the count tokenized a rendered string without the chat template (too low) and with a tool-intent suffix generation never sends (either direction); clients sizing context from it got a wrong budget.
- Refs: m9 D1 amendment 2026-10-10; plan `docs/superpowers/plans/2026-10-10-issue-621-count-tokens.md`; `kairyu/engine/openai_backend.py`

### 2026-10-10 — [progress] Quyet-1.0-Large System One example: all nine GPU gates pass (PR #642)
- What: `examples/quyet-1.0-large-1gpu` serves the Jev-family decision model as System One only (no chat): example-owned adapter on the `quyet` package with a batch-invariant vLLM v0.31.0 as an internal pool (`public_models`); `kairyu/` unchanged. Gates rebuilt from TypeSafe's documented usage (reference, attest, parity, JevBench, fan-out, consistency, SDK, throughput, overload); run `20261010-s1-r2` passes all nine.
- Why: owner direction: Jev models are decision APIs, so chat, tool-call and image surfaces were removed. vLLM's kernels differ from transformers in the tail (median probability difference 0.0001, max 0.22), so parity is judged on prompts, confident decisions (>= 99 %) and JevBench quality; batch invariance makes repeats identical at about 10 % latency.
- Refs: FN-D9 Quyet amendment; plan `docs/superpowers/plans/2026-10-10-quyet-large-1gpu-example.md`; example `MEASUREMENTS.md`
