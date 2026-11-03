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
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, OpenAI-spec Responses API (live reasoning/tool streams, store endpoints, in-band overflow; m11 D4); Anthropic Messages returns and accepts a direct engine's thinking blocks (m11 D4 amendment 2026-10-19); System One (Jev wire API) at `/v1/systemone` outside ReplicaPool (m11 D8), GPU-verified through `quyet-1.0-large-1gpu` and the two Quyet 8-GPU examples
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI (0.160 GPU-verified 2026-10-06 on DeepSeek-V4.1 and an AUTO model) and IDE tool-calling work end-to-end, including AUTO models over /v1/responses (#530)
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering; Helm supports immutable images, split-role labels, safe rollout/drain, hardened Pods, and ServiceMonitor plus the kind CI drill
- AsyncRequest v1 (opt-in `async_requests`): PostgreSQL-backed non-streaming Chat with tenant-scoped status/result/cancel, lease-fenced workers, shared queue telemetry and bounded request/audit retention; the retention-inclusive three-gateway Kind gate runs in F1c CI. Runner State v1 contracts (Kubernetes observation/reconciliation, fenced drain, failure-domain backoff, PostgreSQL leader lease, WP3.1 scaling policies, WP3.2 decision log) exist as a library
- Checkout-only eval tooling retains explicit Core, Quantization, Structured Output, and Long Context suites with hash-chained quality history, config A/B comparisons, and quantization sweeps; Kairyu correctness and performance gates are owned by `verification/`, not evals
- Replica-pool scale-out example (FN-D9, 2026-09-01): Qwen3.8 TP1 x 8 behind one public model (`qwen3.8-27b-dp8-8gpu`); `verify.sh serving` proves the even per-replica split from the pool placement log and `verify.sh tool-calling` proves OpenAI tool calls on every replica (see its MEASUREMENTS.md)
- DeepSeek V4.1 Flash single-replica example (FN-D9 amendment, 2026-09-11) is GPU-verified on TP8/EP8 SM120 with the V4 ReplicaPool/API/UI structure and official thinking-high default; bounded L1 comparisons select DSpark 5, 16K batching and NCCL. The 320-request matrix, reasoning/tool/vision/cancellation, normal restart and retrieval through 1,039,909 prompt tokens pass; exact evidence and limitations are in its `MEASUREMENTS.md`.
- DeepSeek V4.1 Flash six-GPU example (FN-D9 six-GPU amendment, 2026-09-30): one DP6/EP6 replica on GPUs 0–5 with the 8-GPU example's L2/L3 structure and its own scripts; official-first L1 (pinned vLLM nightly + SM120 overlay, Engram offload, 4K batch / 0.92 from the recipe's memory-bound arm, DSpark 5 with full verification). Serving 102 / 591 / 718 tok/s at c1/c32/c64; gate evidence in its `MEASUREMENTS.md`. Since PR #647 its chats reach vLLM as native messages (`upstream_chat_models`): all nine gates and the pre-benchmark check pass again (2026-10-12/13; c1/c32/c64 103.9 / 586.7 / 825.1 tok/s; tool-result gate turn asks for a needed call).
- Qwen3.8 + DeepSeek-V4.1 ensemble example (DTO-D16/D17, 2026-10-01): V4.1 DP6/EP6 (GPU 0–5, the six-GPU example's L1) + Qwen TP1 × 2 (GPU 6, 7). A Qwen judge picks one of two routes: thinking DeepSeek, or the dual-track ensemble with two policies and a three-candidate synthesis. Every role takes images natively; DeepSeek uses the official V4.1 encoder with per-request effort, and the example's overlay continues the floor's assistant prefill. ENSEMBLE criteria loosened and judge fallback moved to `deepseek_think` (DTO-D17 amendments). GPU-verified 2026-10-01: all gates pass, including the ensemble TTFT gate at c1–c32 (c32 at 95.8 % of the limit); 4 of 128 coding requests exceed the 900 s turn envelope after two audit refinements.
- Quyet-routed DeepSeek example `deepseek-v4.1-quyet-8gpu` (VCO-D21/D22, PR #645; renamed from `deepseek-v4.1-winnow-8gpu`): DeepSeek-V4.1 DP6/EP6 (GPU 0-5), Quyet-1.0-Large x 2 (vLLM bf16 + the example's System One adapter; GPU 6 route judge, GPU 7 judgments and form check), used within Quyet's 6,000-token state. One public model `kairyu-verified-tool`; routing (TOOL/THINK) and the think route as in VCO-D20. The verified tool route is rebuilt: 10-16 DeepSeek candidate calls in one request, six Quyet judgments per candidate, a DeepSeek move that batches independent safe calls, and a seven-question Quyet form check with up to two DeepSeek fixes (`kairyu_verification` reports it); requirements removed; `kairyu/` unchanged. GPU 2026-10-11: 8 of 9 gates pass (serving p50 c1/c16 74.0/131.1 s vs Winnow 137.2/208.6 s, every reply a structured call, 0 Quyet states cut); `verified-tool-route` fails 2/12 and the real-turn route replay misroutes 11/64 (Winnow 1/64), final submit turns routed THINK; TOOL floor 0.1 (owner decision) and a strict form check (review) added 2026-10-11: all nine GPU gates pass (verified-tool-route 12/12, serving p50 c1/c16 74.8/134.2 s, every reply a structured call); real-turn route replay 5/64 misrouted (pass line 6). The Winnow layout's last gates (2026-10-09, all nine pass) stay in its `MEASUREMENTS.md`; its DeepSWE r2 stopped at 24 passes of 42 scored (`deepswe-verified-tool-4w-20261009-r2`).
- GGUF via llama.cpp (LCP-D1..D6, 2026-10-04): `upstream: llamacpp` on the `openai` backend; CPU contract gate green on stock b11391 (its Winnow GPU examples were removed 2026-10-18)
- Quyet-1.0-Large System One example `quyet-1.0-large-1gpu` (FN-D9 Quyet amendment, PR #642, 2026-10-10): one GPU, System One only (Jev-family decision model, no chat): Kairyu publishes `/v1/systemone`; an example-owned adapter keeps the `quyet` package's prompt and calibration and reads option-letter logprobs from an internal, batch-invariant vLLM v0.31.0. All nine gates pass: JevBench public 231 through Kairyu 211 correct vs 209 for the package's own run (Brier 0.141/0.142, ECE 0.038/0.042, p50 0.10 s), 308/309 confident answers keep the package's top option, TypeSafe's SDK works (`models.list()` lacks Kairyu's `release_date`). Evidence in its `MEASUREMENTS.md`.
- GLM-5.3-Flash six-GPU example `glm-5.3-flash-6gpu` (FN-D9 GLM amendment, PR #644, 2026-10-11): official FP8 on stock vLLM v0.31.0 as one TP2 x DP3 / EP6 replica with MTP 3 on GPUs 0-5 (four GPUs left no 1M KV pool), `deepseek-v4.1-flash-6gpu` L2/L3, `kairyu/` unchanged. All nine gates pass (run `20261011-r2`): 70.6 / 298.3 / 455.7 tok/s at c1 / c16 / c64, needle at 1,039,890 tokens, 80 max-effort answers complete. Evidence in its `MEASUREMENTS.md`. Native messages since PR #647 (`upstream_chat_models`): all nine gates and the pre-benchmark check pass on the PR head (2026-10-13). DeepSWE max-effort runs for both six-GPU examples: issue #649.
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
- Human sign-off pending on M2–M4 design reviews
- Caller-API forwarding / example UIs on Responses (2026-10-17, PR #656): GPU final test complete 2026-10-19 — the 7 remaining examples with an Open WebUI pass every gate, `api-equivalence`, the Chat UI pass and the cancellation probe (each `MEASUREMENTS.md`); `glm-5.3-flash-quyet-8gpu` deleted (owner decision); remaining: merge after CI
- Messages thinking round trip and thinking off (m11 D4 amendment and review amendment 2026-10-19, PR #660): owner review passed; GPU smokes (dsh effort max and off) pass on `0513750a`; DeepSWE with DeepSeek Harness recorded in the kairyu-bench wiki (1017 six-GPU 76/113, eight-GPU 83/113); remaining: merge after CI. Open (recorded only): `count_tokens` 404 on tool-call history (#659 item 5)
- DeepSeek-V4.1-Flash-1017 examples (FN-D9 1017 amendment, PR #658): `deepseek-v4.1-flash-1017-6gpu` (DP6/EP6) and `-8gpu` (TP8/EP8) on vLLM v0.32.0 + own overlay (1017 encoder rule, top_p 1.0) pass every GPU gate (2026-10-19); DeepSWE v1.1 on the six-GPU one (113 tasks, 8 workers, max, 2026-10-19): 79/113 (69.9 %; 0909 six-GPU max 85/113), 6 h 04 min, 10,062 API calls, 89 per task (0909: 194.5); on the eight-GPU one (mini-swe-agent, 2026-10-20): 81/113, 3 h 59 min; kairyu-bench wiki DeepSWE-Comparison updated; the eight-GPU one's Chat UI also serves Japanese knowledge bases (Docling OCR, bge-m3, multilingual reranker; GPU-verified 2026-10-21, PR #661) and Word / Excel / PowerPoint read and edit through its default-on browser code interpreter (GPU-verified 2026-10-22, PR #662), delivered as download links in the reply (PR #664)

- Gemma 4 31B + EmbeddingGemma 2 example `gemma-4-31b-8gpu` (FN-D9 Gemma amendment, PR #667): GPU-verified 2026-11-04. The replica search chose 4 x TP2 chat replicas with MTP 4 (one GPU cannot hold a 262,144-token context) and 2 EmbeddingGemma 2 replicas sharing GPUs 1 and 7; serving c1/c8/c32/c64 68 / 385 / 589 / 651 tok/s; all 17 gates pass under the owner's gate amendments (official sampling only, model misses recorded); EmbeddingGemma 2 runs on the example's one-edit SM120 overlay; remaining: merge after CI

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-11-04 — [progress] Gemma 4 31B example GPU-verified; gate amendments
- What: replica search: TP1 excluded (26.5 of 33.3 GiB KV for one 262,144-token request); of eight L1 candidates only MTP 4 clears 5 %; under ingestion 4 x TP2 with the embedding replicas sharing GPUs 1 and 7 beats 3 x TP2 with dedicated GPUs by 29-32 % at c32/c64. EmbeddingGemma 2 needs a one-edit SM120 overlay (Triton encoder tiles); chat runs with custom all-reduce off. All 17 gates pass (serving 68 / 385 / 503 / 589 / 651 tok/s at c1-c64, 16 per replica at c8+).
- Why: owner decisions during the gates: official sampling only; a gate failing on the model's own behaviour is excessive, so the ~250K answer and file replies that deliver nothing are recorded, not judged; a file the tool attached counts as delivered.
- Refs: PR #667; FN-D9 Gemma amendment (GPU finding, gate amendments); `examples/gemma-4-31b-8gpu/MEASUREMENTS.md`

### 2026-11-03 — [amendment] Review fixes to the deterministic CI gates
- What: scheduler priority work also counts elements the priority list shifts or copies; sampler work also counts active-index upkeep (tensor sorts, copies, allocations; position/ref-count map writes); the nightly series explicitly starts a version-2 chain when only older-schema history exists, and warm-ups are retained under their own artifact name; F1a takes one final resource capture after traffic and requires it as the last row (m10 A23 amendment).
- Why: owner review of PR #667: a sorted-list priority queue and a per-step index rebuild passed the gate, the nightly could stay in warm-up forever after the schema change, and a truncated resource sidecar was judged complete.
- Refs: PR #667; `docs/design/m10-fleet-cpu.md` A23; `docs/design/issue-377-nightly-cpu-perf.md`

### 2026-11-03 — [design] CI verdicts depend only on the code
- What: every CI pass/fail is decided by the code alone: timing became diagnostics (cpu-microbench and nightly perf judge operation counts; kind F1a/F1b/F1c, async smoke and F2b judge order, fencing tokens or a virtual clock; ~190 pytest waits rewritten to events, fake clocks or hang bounds); runner inputs pinned (umask 022, uv 0.11.31, single-platform postgres digest, images loaded from archives); networked setup separated from tests; the CPU suite runs on four workers to meet the five-minute target. CLAUDE.md test policy rule 8.
- Why: owner principle after PR #667's CI failed on a wall-clock ratio and a runner image-store change ("not caused by this PR" is not a resolution).
- Refs: PR #668; plan `docs/superpowers/plans/2026-11-03-ci-deterministic.md`; m10 A31 amendment; `docs/design/issue-377-nightly-cpu-perf.md`
### 2026-11-03 — [amendment] Responses accepts reasoning.effort none (thinking off)
- What: `/v1/responses` turns an engine or pool model's thinking off on `reasoning.effort: "none"` (the internal `none` of the 2026-10-19 Messages amendment, sent on the model's wire); AUTO models keep their stage policy; the public Chat parser still rejects `none`.
- Why: the 2026-10-05 rejection predates any thinking-off path; since 2026-10-19 the L1 renders thinking off from its own template. The Gemma 4 example's Chat UI needs it for its thinking-off choice (owner decision).
- Refs: m11 D4 Responses thinking-off amendment; `test_responses_effort_none_turns_a_direct_models_thinking_off`

### 2026-11-03 — [design] Gemma 4 31B + EmbeddingGemma 2 example; upstream embedding replica pools
- What: `/v1/embeddings` forwards to OpenAI-compatible embedding servers (`embeddings.<name>.backend: openai`, replicas with `health_url`); the replicas form one `ReplicaPool` (least-outstanding, ejection, probe restore, placement log), `/readyz` and pool metrics include it. New example `gemma-4-31b-8gpu` (fifth family): Gemma 4 31B chat and EmbeddingGemma 2 embedding replicas on stock vLLM, the 1017 Chat UI features in the example itself, knowledge bases embedded through Kairyu with EmbeddingGemma's prefixes, sign-in Chat UI; layout provisional until the replica search.
- Why: owner request (L2 balances chat and embedding replicas, L3 uses the new embedding automatically, best replica counts by measurement); L3 could only serve in-process CPU embeddings. Main only (old PRs closed, L4 not used).
- Refs: m5 D4 / m11 D4 A20 / m7 D4 amendments; FN-D9 Gemma amendment; plan `docs/superpowers/plans/2026-11-03-gemma-4-31b-8gpu-example.md`

### 2026-10-22 — [progress] Download links for files the 1017 eight-GPU Chat UI makes
- What: a `file_download` Open WebUI tool (installed by `run.sh up`, on by default) reads a file the browser code interpreter wrote, stores it in Open WebUI's file store, attaches it to the reply and returns a link; its description also has the model import the bundled Office libraries directly. GPU: `file-download` (owner's Excel request: link saves the file on desktop, after a reload and on an emulated phone), `office` (edits saved from the link), `chat-ui`, `knowledge` pass.
- Why: the owner could not download a file the Chat UI made on a phone: a dead `sandbox:` link, invented UI steps, and working files that exist only in the page.
- Refs: PR #664; plan `docs/superpowers/plans/2026-10-22-1017-8gpu-file-download.md`; the example's `MEASUREMENTS.md`

