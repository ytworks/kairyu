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

Snapshot date: 2026-08-08. Hardware context: all GPU evidence so far is on
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
| M8 Engine CPU core | Complete (amended 2026-08-04) |
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
- B7 (KV answer-equivalence, #373): operator implemented and portable-validated; additive over F2/F4
- G4 MoE: M-A1 formal FAIL retained; M-A2 complete; M-A3 scope-closed by owner deviation (perf gate stays FAIL)
- G4 E-KV: FP8-E4M3 KV **FAIL** on Qwen3-32B long-context; `fp8_e4m3` startup rejected, BF16 KV fail-closed
- G5: F1a–F1d, F2a–F2d, F4a, F4b all closed; F4c decided (keep per-replica RadixKV + F2 routing, thresholded revisit)
- F5a/b/c (priority, noisy-neighbor, SLO admission): closed
- G6: P-A, P-B1–P-B4, P-C2/C3/C4 green (incl. Open WebUI P-B3 browser gate); remaining P-C gates continue
- #150 LiveCodeBench TP8 gate: passed after deadlock fix; #364 `logits_dtype`: valid negative, withdrawn

### What works today

- `kairyu serve --tp N` on real hardware: Qwen3-32B TP8, Llama-3.1-8B, Llama-3.3-70B FP8, Qwen3-VL-32B (via vLLM replica)
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; capable CUDA models pre-capture decode graphs before readiness
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Fully device-side sampling, penalties, spec verification, page-table caching; n-gram/EAGLE-3/MTP greedy drafts; incremental detokenization and stop matching
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: `kairyu bench run` (Fugu + core suites), hash-chained quality history, config A/B and quant sweeps
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Issue #318 verdict: depth beyond the two-step admission horizon is not an A6 fix (`no_measured_benefit_depth_gt_2`)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- Learned-draft real-checkpoint acceptance/performance evidence remains open; FP8-E4M3 KV disabled pending offline calibration
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-08 — [amendment] Learned draft heads enter native serving
- What: public native config wires EAGLE-3/MTP checkpoints into stateful target-hidden capture, O(T+k) cached rollout, exact accepted-row commit, per-source acceptance evidence, and request-local speculative pipeline dependencies; missing Radix hidden history safely degrades to target decode.
- Why: the trained heads and batched teacher path existed, but public serving exposed only low-acceptance n-gram proposals and serialized unrelated pipeline work behind a global barrier.
- Refs: issue #330; m17 A30-A32; `kairyu/engine/core/{draft,model_runner,spec_runner,scheduler}.py`; `kairyu/models/{eagle,mtp,llama}.py`

### 2026-08-08 — [amendment] Mixed engine steps share one ragged model chain
- What: native-ragged backends append one-token decode rows after prefill rows and execute one flat model forward; device-owned feedback uses existing decode slots, while unsupported backends, pure decode, and attention-DP's per-forward distributed protocol retain their prior paths.
- Why: colocated mixed steps otherwise traverse every model layer twice even though ragged prefill already supports a causal one-token continuation.
- Refs: issue #317; m13 D1; A12 #360; `kairyu/engine/core/model_runner.py`; `tests/{unit,gpu}/test_batched_prefill*.py`

### 2026-08-07 — [amendment] Exact KV scoring leaves routing locks
- What: exact hash-set identities and lifecycle revisions are captured under their respective locks, scored outside, then revalidated before a vector is accepted; approximate candidates do not prune the independent exact score space.
- Why: an 8K-token prompt across a large fleet performed O(replicas x prompt blocks) membership work while blocking event ingestion and replica lifecycle changes.
- Refs: issue #348; m10 A31; `kairyu/orchestration/{kv_index,kv_routing}.py`; `tests/unit/test_{kv_event_recovery,kv_routing_adapter}.py`

### 2026-08-07 — [amendment] Serving micro-overheads stay bounded
- What: metrics path templates use a bounded LRU; SSE separator escaping uses one ASCII fast-path scan; AUTO direct streaming drops its duplicate prefix scan; chat body limiting forwards validated chunks without replay buffering; unset AUTO usage fields are excluded at serialization call sites.
- Why: these repeated regex/string/Pydantic operations and duplicate request-body storage add small but compounding latency or peak memory on the public serving path.
- Refs: issue #349; m7 D4; m11 accounting/trace; `kairyu/entrypoints/server/{middleware,app,protocol}.py`; `kairyu/{sse,orchestration/orchestrator}.py`

### 2026-08-07 — [amendment] Native admission is bounded by the RoPE table
- What: real-model builders default an omitted `max_model_len` to `max_position_embeddings` and reject a larger override before loading or serving, across single-rank, TP, EP, P-D, and process-split paths.
- Why: a fixed resident RoPE table must fail cleanly at admission rather than let an out-of-range gather raise on CPU or poison a CUDA context.
- Refs: issue #332; m12 D2; `kairyu/engine/kairyu_backend.py`; `tests/unit/test_model_loader_backend.py`

### 2026-08-07 — [amendment] Model and stream hot paths avoid repeated setup
- What: rotary cos/sin is precomputed to the model position limit and gathered per step; scaled dense RoPE uses the fused kernel; invariant Q/K guards and kernel imports leave layer loops; tokenless nonterminal updates and wire materialization are skipped; sampler conversion makes one mutable copy.
- Why: decode repeatedly launched position/trig kernels, evaluated invariant guards/imports, and constructed cumulative output work even when no request produced a token.
- Refs: issue #332; m8 D1/D2/D6; m12 D2; `kairyu/models/{layers,attention}.py`; `kairyu/engine/{engine_loop,zmq_backend}.py`

### 2026-08-07 — [amendment] Grammar-free CUDA sampling is batched
- What: mixed decode rows now batch grammar-free temperature/min-p/top-k/top-p filtering and stateless Gumbel draws, use the maximum bounded top-k prefix for nucleus filtering, preserve exact full-vocabulary top-p when no finite top-k exists, and leave only grammar rows on the scalar CPU matcher path.
- Why: one non-greedy row previously forced per-row fp32 copies, kernel launches, and full-vocabulary sorts across the complete decode batch; a Qwen3-size 151,936-token vocabulary at B=16 measured 1.13 ms batched versus 14.21 ms scalar on the same GPU.
- Refs: issue #326; m8 D2; `kairyu/engine/core/{model_runner,sampler}.py`; `kairyu/kernels/sampling_gpu.py`; `tests/gpu/test_batched_sampler_gpu.py`

### 2026-08-07 — [amendment] Eager FlashInfer decode reuses the no-D2H planner
- What: ordinary tensor eager decode and graph-shape eager fallback now pass authoritative scheduler lengths into the existing fixed-shape metadata pack and FlashInfer fast planner after one stock initialization per shape.
- Why: eager and out-of-coverage graph steps rebuilt dynamic indices and copied schedule inputs to the host every decode step even though graph replay already had a validated no-sync path.
- Refs: issue #329; m17 A29; `kairyu/engine/core/{attention/{flashinfer_gpu,flashattention_gpu},model_runner,step_executor}.py`; `kairyu/models/attention.py`
