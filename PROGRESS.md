# Progress

Cross-session memory of design changes and project progress.
Maintained per the rules in `.claude/rules/progress-log.md`.

## Current Status

_Last updated: 2026-07-02_

| Milestone | Status |
|-----------|--------|
| M1 — Orchestration (L2) + Interface (L3) | Complete and merged. Router / Conductor / MoA, vLLM-compatible `LLM` + `AsyncLLMEngine`, OpenAI-compatible server, YAML/decorator DSL. |
| M2 — Core engine (overlap scheduler + Radix-Paged KV) | CPU half done: scheduler, KV manager, EngineCore step loop, overlap pipeline, pre-GPU robustness (EOS, preemption, abort, pin TTL). Paged-KV attention validated with real tensors on CPU (greedy-equivalence). **Blocked on GPU hardware** for the GPU phase. |
| M3 — Spec decode / CUDA graphs / P-D separation | n-gram draft spec-decode policy and xgrammar structured output implemented CPU-side. CUDA graphs and the rest gated on M2 GPU phase. |
| M4 — Router learning pipeline | Implemented CPU-only (logs → distilled classifier → contextual bandit). Design reviewed. |
| M5 — Intra-node multi-GPU (TP, DP replicas, P-D intra-node) | Goal defined (`docs/goals/g2-multi-gpu.md`); design doc pending. Prereq: M2 GPU phase Gates 1–3. |
| M6 — Inter-node multi-GPU (2-node DP, KV transfer plane, P-D inter-node, PP) | Goal defined (`docs/goals/g2-multi-gpu.md`); design doc pending. Prereq: all M5 gates. |

What works today: full stack on CPU — `kairyu` EngineBackend wired through the
OpenAI-compatible server with the mock/CPU runner; serving/router/multiturn benchmarks
in `bench/`.

Active blockers: GPU (H100/A100) required for M2 GPU phase; execution plan is
`docs/gpu-runbook.md`. Human sign-off pending on M2–M4 design reviews.

## Change Log

### 2026-07-02 — [design] Multi-GPU goal (G2) defined — drives M5/M6
- What: Wrote `docs/goals/g2-multi-gpu.md`, the acceptance contract for intra-node
  (M5: TP, DP replicas via L2 Router, P-D intra-node) and inter-node (M6: 2-node DP,
  page-granular KV transfer plane, P-D inter-node, PP=2) multi-GPU serving.
  Targets: Llama-3.3-70B FP8 on 8×H100 + 2 nodes (IB/RoCE); vLLM-parity-or-better plus
  absolute scaling-efficiency gates (A1–A10, B1–B5); TP=2 is the scaling base (70B FP8
  cannot run TP=1). MoE/expert parallelism is an explicit non-goal.
- Why: Multi-GPU support existed only as a no-op `tensor_parallel_size` arg and the
  P-D admission-policy half; M5/M6 need a G1-style evidence-first goal to drive
  autonomous development. `docs/goals/` created since the original G1 goal was never
  filed as a document.
- Refs: `docs/goals/g2-multi-gpu.md`; seams: `kairyu/engine/core/engine_core.py`
  (ModelRunner), `scheduler.py` (`pd_separation`), `kairyu/orchestration/router.py`

### 2026-07-02 — [progress] Design-change memory harness added
- What: Added PROGRESS.md, `.claude/rules/progress-log.md`, CLAUDE.md, and AGENTS.md so
  Claude Code and Codex sessions share the same record of design changes and progress.
- Why: Design decisions were scattered across design docs, review-amendment commits, and
  session context; new sessions had no single place to recover project state.
- Refs: `.claude/rules/progress-log.md`

### 2026-07-02 — [progress] README enriched; GPU-day runbook added
- What: README expanded with architecture, roadmap, usage guides, and open-model setup
  (Kimi, Qwen). GPU-day runbook consolidates all remaining GPU-gated work into ordered,
  gated execution steps.
- Refs: commits 9d35360, cc45b08; `docs/gpu-runbook.md`

### 2026-07-02 — [progress] xgrammar structured output integrated (M3)
- What: Token-bitmask enforcer and `response_format` plumbing through the engine and
  OpenAI server.
- Refs: commits ad4e18c, a0851f9; `docs/design/m3-spec-decode-and-graphs.md`

### 2026-07-02 — [progress] Engine core validated end-to-end on CPU (M2 CPU half)
- What: `kairyu` EngineBackend exposed and wired through the OpenAI server; paged-KV
  attention proven greedy-equivalent with real torch tensors on CPU; pre-GPU robustness
  items landed (EOS under overlap, preemption, watermark, abort, pin TTL).
- Refs: commits 991832b, aa382f8, e977e1c; `docs/design/m2-engine.md`

### 2026-07-02 — [amendment] Design-review amendments applied (M2/M4)
- What: Compute-skip, computed gating, output caching, and bandit-router fixes applied
  across M2 KV/scheduler and M4 learning pipeline, per the agent design-review panel.
- Why: Review found gaps in the original D-decisions; docs updated with APPROVE-WITH-
  AMENDMENTS status and amendment sections (§5/§6).
- Refs: commits c14f035, 22b0b53; `docs/design/m2-engine.md` §6,
  `docs/design/m4-router-learning.md` §5

### 2026-07-02 — [design] M2–M4 designs written and reviewed; M4 pulled forward
- What: Design docs for M2 (overlap scheduler + Radix-Paged KV), M3 (spec decode, CUDA
  graphs, P-D separation), M4 (router learning) written and agent-review-approved with
  amendments. M4 was pulled ahead of schedule because it is GPU-independent.
- Why: M2's remaining half needs GPU; GPU-independent work (M3 CPU-side, M4) proceeds
  first to keep momentum.
- Refs: `docs/design/m2-engine.md`, `docs/design/m3-spec-decode-and-graphs.md`,
  `docs/design/m4-router-learning.md`; commits d2675a7, c976c64, 4d83229

### 2026-07-02 — [progress] M1 complete: orchestration + vLLM-compatible interface
- What: L2 orchestration (rule-based Router with JSONL decision log, Conductor role-DAG
  with verifier-gated refinement, MoA, immutable budget) and L3 interface
  (vLLM-signature `LLM`, `AsyncLLMEngine`, OpenAI-compatible server with SSE streaming
  and tool calls, YAML/decorator DSL) built on the `EngineBackend` protocol
  (mock / vLLM / external-OpenAI backends).
- Refs: commit 633fa37; `docs/design/m1-orchestration-and-interface.md`
