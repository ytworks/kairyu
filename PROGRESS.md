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

Snapshot date: 2026-08-14. Hardware context: all GPU evidence so far is on
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
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; assistant history round-trips typed `reasoning_content` while assistant-only LiteLLM provider objects and nullable legacy function calls are ignored before rendering and other extras remain fail-closed; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: 12-slot Accuracy plus Core/Quantization/Structured/Long Context suites, eight-model sourced Accuracy comparison with cell-level provenance, target-only streamed TTFT/TPS including exact public-vs-internal orchestration token rates, hash-chained quality history, config A/B and quant sweeps; SWE-bench Verified uses mini-SWE-agent's official 250-step `verified` flow plus the official harness with fail-closed denominators; Terminal-Bench keeps resumable raw Harbor jobs and bounds every agent phase to two effective hours
- The tiered RTX PRO example has one public chat model, one public pinned offline embedding model, and one orchestration YAML: Open WebUI lists only the chat model, L2 borrows the Qwen/DeepSeek L1 pools directly, and launcher readiness proves a two-input 384-dimensional embedding response. Both this path and the single-Qwen example accept image chat and complete the fail-closed 10-item CharXiv smoke; its composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes
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

### 2026-08-14 — [amendment] Exact LiteLLM assistant history is reusable
- What: Chat Completions now drops object-valued `provider_specific_fields` and nullable legacy `function_call` metadata from assistant history while retaining typed reasoning, content, and tool calls; invalid roles, value kinds, and unrelated extras remain fail-closed.
- Why: LiteLLM 1.96.2 emits both compatibility fields in a normal assistant `model_dump()`, so the nullable-only #480 policy still rejected an unmodified second-turn request.
- Refs: issue #482; `kairyu/entrypoints/server/chat_service.py`; `tests/server/test_{chat_template_policy,openai_api,orchestration_usage_trace}.py`

### 2026-08-14 — [amendment] Assistant reasoning history round-trips
- What: Chat Completions now accepts its typed `reasoning_content` response field in assistant history and drops only nullable LiteLLM `provider_specific_fields` metadata before rendering; non-null and unknown extras remain fail-closed.
- Why: The tiered product emitted visible intermediate work that normal LiteLLM serialization returned on the next agent turn, but the input schema rejected both its own field and nullable client metadata before dispatch.
- Refs: issue #480; `kairyu/entrypoints/server/{protocol,chat_service}.py`; `tests/server/test_{chat_template_policy,openai_api,orchestration_usage_trace,prompt_offload}.py`

### 2026-08-14 — [amendment] Tiered frontier API gains offline embeddings
- What: The eight-GPU example now builds the pinned FastEmbed MiniLM bundle, publishes truthful `embed-small` beside its chat product, keeps routing and Chat UI chat-only, and fails readiness unless a two-input 384-dimensional embedding smoke returns ordered finite vectors with exact usage.
- Why: tau2 banking requires embeddings before task 1, but the example exposed no embedding route; a truthful local ID avoids claiming that MiniLM is OpenAI's `text-embedding-3-large`.
- Refs: issue #479; kairyu-bench issue #5; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,kairyu.yaml,control.py,example.json,README.md}`

### 2026-08-13 — [amendment] Tiered L3 endpoints stay locally and externally reachable
- What: The example now binds both its Chat UI and public L3 API to all host interfaces by default while advertising the real host address, so both endpoints remain reachable through localhost and the external address; explicit bind overrides remain available.
- Why: Binding the API only to the external interface made its normal host-local URL unavailable, while publishing `0.0.0.0` as a client URL confused the listen address with the address clients should use.
- Refs: PR #478; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,control.py,README.md}`

### 2026-08-13 — [progress] Tiered Chat UI response repair closes GPU gates
- What: Default Qwen L1 chat now returns non-empty public content with no hidden reasoning, L3 returns separate non-empty final and attributed reasoning output, the pinned tiered browser smoke passes, and the deterministic CharXiv rerun completes 10/10 scored requests with zero errors or unmeasured requests.
- Refs: PR #478; commit `56c640a`; `/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-results/20260813T055825Z/`

### 2026-08-13 — [progress] Tiered Chat UI publication defaults to visible content
- What: The tiered example's Qwen replicas now default upstream HF chat templating to nonthinking while preserving request-level overrides and vision handling, so proposal and publisher roles return public content instead of exhausting their budget in reasoning-only output; GPU validation is in progress.
- Refs: PR #478; `examples/qwen3.6-deepseek-v4-8gpu/{compose.yaml,README.md}`; `tests/unit/test_tiered_frontier_examplectl.py`

### 2026-08-13 — [progress] Example vision CharXiv validation closes
- What: Both the single-Qwen and tiered Qwen/DeepSeek examples completed their deterministic 10-item CharXiv image runs with all 10 target requests measured and scored and zero target failures; the tiered DAG confines private DeepSeek reasoning to planning/synthesis/verification and uses its image-capable non-thinking Qwen pool for proposals and final publication.
- Refs: PR #478; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/`; `bench/results/examples/qwen3.6-27b-1gpu/charxiv-10-gpu-validation-6/`; `/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-results/charxiv-10-gpu-validation-4/`

### 2026-08-13 — [verified] Single-Qwen CharXiv vision closes 10/10
- What: The 1-GPU Qwen example completed its deterministic CharXiv run with `n=10`, `n_scored=10`, `requests=10`, `errors=0`, and `unmeasured_requests=0`; score was 60%. Tiered orchestration validation remains in progress.
- Refs: `examples/qwen3.6-27b-1gpu/`; `bench/results/examples/qwen3.6-27b-1gpu/charxiv-10-gpu-validation-6/`

### 2026-08-13 — [progress] Example vision orchestration enters GPU validation
- What: L3 now preserves validated image inputs for L2, role DAGs pass media only to capable workers, both Qwen examples enable the pinned checkpoint's vision encoder, and each owns a fail-closed 10-item CharXiv command; real-GPU closure is in progress.
- Refs: `kairyu/orchestration/`; `examples/qwen3.6-{27b-1gpu,deepseek-v4-8gpu}/`

### 2026-08-13 — [amendment] Kind CI tool setup is shared and verified
- What: All four kind gates now use one pinned installer with bounded nested retries, HTTPS-only downloads, SHA-256 verification, and executable version checks; a policy test bans the flaky action path from every workflow.
- Why: Per-workflow fixes left the same dependency-acquisition failure in other gates, incorrectly surfacing CI infrastructure failures as product-quality failures.
- Refs: PR #476; `scripts/install_kind_tools.sh`; `.github/workflows/{ci,f1a-churn,f1b-rollout,f1c-gateway}.yml`; `tests/unit/test_ci_workflow_policy.py`
