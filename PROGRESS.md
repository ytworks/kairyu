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

Snapshot date: 2026-08-13. Hardware context: all GPU evidence so far is on
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
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; MoA keeps the original response contract distinct from untrusted candidate drafts, with configured completion delimiters and the multi-stage boundary withholding private synthesis reasoning; prefix-aware replica placement obeys the configured queue-depth overload valve; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: 12-slot Accuracy plus Core/Quantization/Structured/Long Context suites, eight-model sourced Accuracy comparison with cell-level provenance, target-only streamed TTFT/TPS including exact public-vs-internal orchestration token rates, hash-chained quality history, config A/B and quant sweeps; SWE-bench Verified uses mini-SWE-agent's official 250-step `verified` flow plus the official harness with fail-closed denominators; Terminal-Bench keeps resumable raw Harbor jobs and bounds every agent phase to two effective hours
- The tiered RTX PRO example now has one public model and one orchestration YAML: Open WebUI calls Kairyu L3 once, L2 borrows the Qwen/DeepSeek L1 pools directly, runs the bounded planner/proposal/synthesis/verifier/publisher DAG, and returns model-attributed intermediate work in the same answer's expandable reasoning item while keeping publisher content separate. Its composed L1 workers remain vLLM-backed until the native full-checkpoint gate closes; prior MoA-3 measurements remain historical evidence only
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

### 2026-08-13 — [progress] Tiered example collapses to one layered product path
- What: The example now keeps one `auto-max.yaml` policy and one public Chat UI model; L2 borrows deployment-owned L1 pools, runs the bounded seven-role DAG, and streams attributed intermediate output separately from publisher content in the same response. Obsolete auto/MoA candidate YAMLs and commands were removed.
- Refs: PR #476; `examples/qwen3.6-deepseek-v4-8gpu/`; `scripts/webui_browser_smoke.mjs`

### 2026-08-13 — [design] Tiered UI owns one layered orchestration path
- What: The tiered example will expose one product model, borrow deployment L1 pools directly from L2, run a bounded verifier-gated DAG, and show policy-enabled model-attributed intermediate output separately from the final answer.
- Why: The prior loopback L3 workers and single-pass MoA policy did not exercise the intended L3→L2→L1 boundary, while the product UI now requires inspectable intermediate work without mixing it into the committed answer.
- Refs: EO-D1..EO-D5; `docs/design/example-layered-orchestration.md`; `docs/superpowers/plans/2026-08-13-example-layered-orchestration-correction.md`

### 2026-08-12 — [amendment] SWE-bench fixes selection before generation
- What: SWE-bench now persists the official ordered selection before generation, evaluates that full set even when a prediction is omitted, accepts the official v4.1 completed/error report overlap with error precedence, and retains redacted stage logs plus raw upstream evidence; failed runs resume while explicit reruns use isolated artifacts.
- Why: mini-SWE-agent can exit successfully after a worker omits its prediction, so deriving `--instance_ids` from `preds.json` could shrink the denominator and deleting work directories could discard the only audit and resume evidence.
- Refs: issue #472; `kairyu/bench/adapters/swebench.py`; `kairyu/bench/adapters/base.py`; `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`

### 2026-08-12 — [progress] SWE-bench Verified launches through Accuracy
- What: Accuracy now runs the 500-task Verified test split through mini-SWE-agent's standard 250-step configuration and the official SWE-bench evaluator, preserves every selected task in the resolved-rate denominator, retains auditable commands and raw evidence, and extends the sourced comparison report with Fable 5's published result without claiming one-trial/five-trial parity.
- Refs: issue #472; `kairyu/bench/adapters/swebench*.py`; `docs/benchmarks.md`; `tests/bench/test_bench_agentic*.py`; `tests/bench/test_bench_compare.py`

### 2026-08-12 — [progress] Tiered PR CI regressions reproduced and closed
- What: The Open WebUI initial/outage/recovery browser phases pass after keeping the deterministic AUTO SSE marker inside the mock's prompt-boundary echo window; the tiered control test now stubs storage path preparation instead of attempting `/mnt/nvme` writes on hosted runners. Production NVMe enforcement and L2/L3 behavior are unchanged.
- Refs: `scripts/webui_browser_smoke.mjs`; `tests/unit/test_tiered_frontier_examplectl.py`; PR #471

### 2026-08-12 — [design] SWE-bench Verified joins the Accuracy suite
- What: The Accuracy suite will run SWE-bench Verified through mini-SWE-agent's standard 250-step `verified` flow and the official SWE-bench harness, preserve exact selected-instance denominators and failure classes, and extend the sourced frontier comparison without inventing missing exact-model scores.
- Why: Issue #472 requires the benchmark to launch like existing suites and follow their score-comparison reporting, while upstream methodology differences and OpenAI's retirement warning must remain explicit and auditable.
- Refs: issue #472; `docs/superpowers/specs/2026-08-12-swe-bench-verified-design.md`

### 2026-08-12 — [progress] Tiered RTX PRO example closes full task-set scoring
- What: The selected private-thinking MoA-3 policy launched all 89 Terminal-Bench 2.1 tasks and closed at 60/89 (67.42%) under official task verifiers and Harbor's zero-inclusive Mean: 86 verifier rewards (60 one, 26 zero), two rewardless infrastructure errors, and one rewardless operator-terminated CPU outlier. The score does not substitute diagnostic retries.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/{MEASUREMENTS.md,terminalbench-result.json}`; run ID `terminalbench-2.1-auto-max-thinking2048-full-v1-20260812`; PR #471

### 2026-08-12 — [amendment] Agentic evidence survives interruption with bounded outliers
- What: Terminal-Bench now retains its raw Harbor job directory for authoritative per-task evidence/resume, and sets agent `max_timeout_sec=900` before the existing 8x multiplier so every task has a 7200-second effective agent cap. Official Harbor reward/Mean semantics remain unchanged.
- Why: A 3600-second task declaration previously expanded to eight hours and interrupt cleanup deleted the partial raw job, making one pathological task both operationally unbounded and non-resumable.
- Refs: `kairyu/bench/adapters/terminal_bench.py`; `tests/bench/test_bench_agentic*.py`; PR #471

### 2026-08-12 — [progress] Thinking MoA-3 clears the quality floor
- What: On the same fixed four Terminal-Bench 2.1 tasks, the 2048-token private-thinking MoA-3 policy scored 3/4 versus direct DeepSeek's 2/4. Both completed 4/4 items with zero failed/unjudged/skipped trials and null item errors on a clean source tree; `kairyu-auto-max` is selected for the full 89-task run.
- Refs: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`; run ID `terminalbench-selection-thinking2048-vs-deepseek-20260812`; PR #471
