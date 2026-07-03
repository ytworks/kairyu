# Progress

Cross-session memory of design changes and project progress.
Maintained per the rules in `.claude/rules/progress-log.md`.

## Current Status

_Last updated: 2026-07-03_

Master roadmap: `docs/roadmap.md` (2026-07-03) — dual hardware profiles (NVLink-HBM
A100/H100/B200 nodes AND the PCIe-only RTX PRO 6000 fleet, A100 and later all
supported), three tracks (E: L1 engine → SOTA incl. MoE, F: fleet-scale control
plane, G6/P: product surface). Next actions: **E1** (single-GPU real engine — RTX
6000 Pro units are available now) + **P-A** (truthful API core, CPU).

| Milestone | Status |
|-----------|--------|
| M1 — Orchestration (L2) + Interface (L3) | Complete and merged. Router / Conductor / MoA, vLLM-compatible `LLM` + `AsyncLLMEngine`, OpenAI-compatible server, YAML/decorator DSL. |
| M2 — Core engine (overlap scheduler + Radix-Paged KV) | CPU half done: scheduler, KV manager, EngineCore step loop, overlap pipeline, pre-GPU robustness (EOS, preemption, abort, pin TTL). Paged-KV attention validated with real tensors on CPU (greedy-equivalence). **Blocked on GPU hardware** for the GPU phase. |
| M3 — Spec decode / CUDA graphs / P-D separation | n-gram draft spec-decode policy and xgrammar structured output implemented CPU-side. CUDA graphs and the rest gated on M2 GPU phase. |
| M4 — Router learning pipeline | Implemented CPU-only (logs → distilled classifier → contextual bandit). Design reviewed. |
| M5 — Intra-node multi-GPU (TP, DP replicas, P-D intra-node) | Design reviewed; **CPU half done** (Communicator/StepInput/TPModelRunner, TP plumbing live, ReplicaPool + affinity, PDCoordinator + `resume_with_kv`). GPU phase: `docs/gpu-runbook.md` §6, prereq M2 Gates 1–3. |
| M6 — Inter-node multi-GPU (2-node DP, KV transfer plane, P-D inter-node, PP) | Design reviewed; **CPU half done** (ClusterSpec, KVTransport + loopback + `bench/kv_transfer_bench.py`, openai_backend replica fixes, async runner contract + PipelinedEngineCore). GPU phase: runbook §7, prereq all M5 gates. |
| M7 — Productionization (serve CLI, gateway wiring, batch, observability) | **CPU half done** (design m7 D1–D8, goal G3): health/readyz/metrics/auth/concurrency guard, `kairyu serve` + DeploymentSpec, ReplicaPool gateway wiring + prober, HTTP session affinity, batch API, Dockerfile + compose + CI smoke drill, `docs/deployment.md`. GPU bring-up: runbook §9. |
| G4 — MoE engine (fused experts, EP, MTP, NVFP4, MLA) | Goal defined (`docs/goals/g4-moe-engine.md`); lifts the G2 MoE non-goal. Design doc + review required before implementation. |
| G5 — Fleet scale (elasticity, KV-aware routing, P/D pools, tiering, tenancy) | Goal defined (`docs/goals/g5-fleet-scale.md`); amends m7 D2 (k8s as machine layer), m5 D4/m7 D6 (prefix-aware placement), m6 D1 staticness, ClusterSpec cap, m7 D8 (OTel). F1/F2 are CPU-mock-testable now. |
| G6 — Product surface (truthful API, Fugu-class product, frontier scoreboard) | Goal defined (`docs/goals/g6-product-surface.md`). P-A (usage truth, HF chat templates, logprobs, structured outputs) is CPU work, start now. |

What works today: full stack on CPU — `kairyu` EngineBackend wired through the
OpenAI-compatible server with the mock/CPU runner; serving/router/multiturn benchmarks
in `bench/`; `kairyu serve <deployment.yaml>` runs a hardened gateway (pool of remote
replicas, auth, metrics, batch) or a replica node, and the compose topology
(1 gateway + 3 mock replicas) passes the CI smoke drill incl. kill/recover.

Active blockers: RTX 6000 Pro units are now partially available — M2/E1 GPU phase is
unblocked on the PCIe profile (H100 boxes still wanted for NVLink-profile gates);
execution plan is `docs/gpu-runbook.md` + `docs/roadmap.md` §4. Hardware procurement
(PCIe-switch chassis, ≥400 Gb/s RDMA NICs) gates E4/E5 and is decided during E3 from
E1's measured P2P matrix. Human sign-off pending on M2–M4 design reviews.

## Change Log

### 2026-07-03 — [amendment] G2 hardware contract widened to capability profiles (A100+); fleet-scale decisions amended
- What: G2 §7 gains 2026-07-03 amendments: the goal now spans capability profiles
  covering all NVIDIA GPUs from A100 (SM80) onward — original NVLink arithmetic and
  gates A1–A10 stand on NVLink-HBM profiles; the PCIe-GDDR profile (RTX PRO 6000,
  96 GB, no NVLink) uses TP=1/DP as the 70B scaling base and replaces A3–A5 with a
  placement-crossover report; B2/A10 fabric budgets restated against measured link
  rates; the §6 MoE, autoscaling, and H100-only non-goals are lifted. Related
  amendments: m7 D2 no-k8s → k8s as machine layer only (its own revisit triggers fire
  at thousands of GPUs); m5 D4/m7 D6 session-hash affinity → two-step prefix-aware
  placement then KV tiering; m6 D1 static-only topology relaxed (no-Ray stands);
  ClusterSpec coherence-domain cap 2 → 8; m7 D8 no-OTel flipped. Status notes added
  to m5/m6/m7 design docs and `docs/gpu-runbook.md` (§ header note, §6.1 NVLS scoped
  to NVLink profile).
- Why: The product target is an on-prem DC of thousands of GPUs across BOTH fleet
  shapes (8×H100-class NVLink nodes remain possible; the volume fleet is PCIe-only
  RTX PRO 6000, where 96 GB flips the 70B memory arithmetic and PCIe all-reduce
  latency makes TP-first the wrong default), serving all four model classes
  including MoE — the single-hardware, dense-only, static-fleet assumptions no
  longer hold. Original entries are preserved per progress-log rules.
- Refs: `docs/goals/g2-multi-gpu.md` §7, `docs/roadmap.md` §2/§5,
  `docs/design/m5-*.md` / `m6-*.md` / `m7-*.md` status notes, `docs/gpu-runbook.md`

### 2026-07-03 — [design] Master roadmap + goals G4/G5/G6 defined (gap analysis vs frontier serving)
- What: `docs/roadmap.md` — three-track improvement roadmap (E: own-L1 engine to
  SOTA — real runner/sampling/quant per SM, CUDA graphs + EAGLE-3/MTP via a scheduler
  multi-token commit, profile-aware multi-GPU, MoE/EP, frontier MoE over RDMA;
  F: fleet control plane — dynamic ReplicaPool + registry + k8s machine layer,
  prefix/KV-aware routing fed by RadixKV events, NIXL-candidate KV transport + P/D
  pools, DRAM/NVMe KV tiering, tenancy/SLO admission/autoscaling; P: product
  surface — tokenizer-true usage + cached_tokens, HF Jinja chat templates, streaming
  `kairyu-auto` with orchestration-usage/trace disclosure, Open WebUI integration,
  Responses API/embeddings, nightly frontier-API scoreboard). New goal docs:
  `docs/goals/g4-moe-engine.md`, `g5-fleet-scale.md`, `g6-product-surface.md`.
  Grounding research recorded in roadmap §3/§7: Sakana Fugu product facts (GA
  2026-06, orchestration-as-a-model, Responses API, orchestration token accounting,
  no latency win — Kairyu's wedge is orchestration quality at direct-call latency
  plus trace transparency), SM120 kernel-support gotcha list, and the fleet
  control-plane convergence (Dynamo/llm-d/SGLang gateway/Mooncake/AIBrix:
  prefix-cache-aware routing is the top lever; Kairyu's own RadixKV enables native
  KV-event routing; learned multi-model routing is uncovered white space).
- Why: The product goal (Fugu-class orchestration API + chat UI on an on-prem
  multi-thousand-GPU DC, beating Claude/GPT on TTFT/TPOT/goodput) needed a
  comprehensive gap analysis: the engine compute is placeholder, MoE/quant/spec-decode
  paths are absent, the control plane is static, and the API surface cannot yet
  support billing or honest benchmarks. The roadmap sequences the gaps by impact
  (E1+P-A first) while preserving every existing protocol seam.
- Refs: `docs/roadmap.md`, `docs/goals/g4-moe-engine.md`,
  `docs/goals/g5-fleet-scale.md`, `docs/goals/g6-product-surface.md`

### 2026-07-02 — [progress] M7 Phase 5: deployment guide, runbook §9, README — M7 CPU half complete
- What: `docs/deployment.md` (DC topology, security duty split with the managed cloud
  edge, systemd + compose node setup with documented k8s revisit triggers, config
  walkthrough, rolling model-update drill, observability, interconnect sizing, untested
  k8s appendix); `docs/gpu-runbook.md` §9 (production bring-up on real GPUs: real-engine
  compose smoke, affinity/radix hit-rate measurement through the gateway, rolling-update
  and batch-under-load drills on hardware); README M7 row + serving quickstart. With
  this, every CPU-verifiable G3 gate is implemented and tested (328 tests, 95% cov).
- Refs: `docs/deployment.md`, `docs/gpu-runbook.md` §9, README.md;
  m7 status line updated to Implemented — CPU half

### 2026-07-02 — [progress] M7 Phase 4: OpenAI-compatible batch API
- What: `/v1/files` (multipart upload/metadata/content) and `/v1/batches`
  (create/get/list/cancel) backed by a filesystem `BatchStore` (atomic JSON job state,
  JSONL input/output/error files) and an in-gateway `BatchWorker` lifespan task that
  drains jobs through the same served engines/pools under its own semaphore — strictly
  below the server's global concurrency guard, so interactive traffic stays admitted
  (gate C4, pinned by test). Cancel skips remaining lines; restart recovery marks
  in-flight jobs failed with an explicit resubmit message (single-gateway scope, m7 D7).
  Server helpers `sampling_params_from` / `completion_response` made public for reuse.
  New dep: python-multipart (FastAPI form uploads).
- Refs: m7 D7, G3 gate C4; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/batch_routes.py`, `kairyu/deploy/builder.py`;
  tests `tests/server/test_batches.py`

### 2026-07-02 — [progress] M7 Phase 3: container image, compose topology, CI smoke drill
- What: Multi-stage uv `Dockerfile` (one image for every role; the mounted
  DeploymentSpec decides gateway vs replica), `deploy/compose/` (1 gateway + 3 mock
  replicas with healthchecks, gateway/replica YAML configs), `scripts/compose_smoke.sh`
  (readiness → completion → SSE → affinity-by-metrics → replica kill/eject/zero-5xx →
  prober recovery), and a `compose-smoke` CI job separate from the coverage-gated
  pytest job. The full drill was verified end-to-end with the same configs as local
  processes (kill/eject: 10/10 subsequent 200s; prober auto-restore observed in the
  gateway JSON log); the container build itself runs in CI — this dev environment's
  network policy blocks registry CDNs.
- Refs: m7 D1/D2, G3 gates C1–C3; `Dockerfile`, `deploy/compose/`,
  `scripts/compose_smoke.sh`, `.github/workflows/ci.yml`

### 2026-07-02 — [progress] M7 Phase 2: `kairyu serve` CLI, DeploymentSpec, pool wiring, prober, HTTP affinity
- What: `kairyu serve <deployment.yaml>` console entrypoint builds gateway or replica
  from one YAML: `DeploymentSpec` (new, composes with — does not extend — ClusterSpec,
  m7 D3) declares engines, pools (N remote `openai` members, keyless node-to-node),
  server settings, optional DSL orchestrator, batch section. Builder wraps pool members
  in `ReplicaPool` and passes it into `create_app` unchanged (the pool IS an
  EngineBackend); lifespan starts a `HealthProber` per pool (GETs ejected replicas'
  `/health`, restores via existing `probe()`) and shuts engines down gracefully.
  HTTP affinity gap closed: OpenAI `user` field / `X-Session-ID` header now map to
  `CacheHint(session_id=...)`, so external multi-turn traffic reaches the radix-KV
  warm replica (previously cache_hint was never set on the HTTP path).
- Refs: m7 D3/D4/D6; `kairyu/deploy/{spec,builder,prober}.py`,
  `kairyu/entrypoints/cli.py`, `kairyu/entrypoints/server/{app,protocol}.py`;
  tests `tests/unit/test_{deployment_spec,prober,cli}.py`,
  `tests/server/test_serve_builder.py`

### 2026-07-02 — [progress] M7 Phase 1: server hardening landed
- What: `/health`, `/readyz` (pool-aware: 503 unless every ReplicaPool has ≥1 healthy
  replica), `/metrics` (per-app Prometheus registry; request counts/latency histograms,
  scrape-time pool collector for outstanding/health/decision counts), optional static
  API-key auth (env-sourced, constant-time, health endpoints exempt), global concurrency
  guard (429 + Retry-After on /v1/*), JSON access log with X-Request-ID — all pure-ASGI
  middleware so SSE streams hold their concurrency slot to the last byte. `create_app`
  gains an optional `ServerSettings`; defaults preserve pre-M7 behavior. `ReplicaPool`
  gains read-only `healthy`/`replica_count`/`decision_counts` accessors (still no
  background tasks, m5 D4). New dep: prometheus-client.
- Refs: m7 D4/D5/D8; `kairyu/entrypoints/server/{health,metrics,middleware,settings}.py`,
  `kairyu/orchestration/replica.py`; tests `tests/server/test_{health_metrics,auth,limits}.py`

### 2026-07-02 — [design] M7 productionization designed (G3 goal, D1–D8); G2 2-node scope clarified
- What: Wrote `docs/goals/g3-production-deployment.md` (gates C1–C7) and
  `docs/design/m7-productionization.md`. Decisions: D1 on-prem-DC topology (managed
  cloud WAF/LB front → private interconnect → stateless CPU gateway tier running
  `create_app` + Orchestrator + ReplicaPool of remote `openai`-backend replicas → N GPU
  replica nodes running the same artifact); D2 no Kubernetes — systemd + docker compose
  with everything containerized and documented k3s revisit triggers; D3 new
  `DeploymentSpec` (ClusterSpec untouched — it binds the TP/PP coherence domain, not
  fleet size); D4 health/readyz/metrics + serve-layer background prober (pool stays
  passive); D5 edge-owned WAF/TLS, gateway static API keys + concurrency guard, keyless
  node-to-node; D6 cache layer = per-replica radix KV + pool session affinity, no Redis
  (revisit trigger recorded) — includes fixing the gap that the HTTP path never set
  `cache_hint`; D7 minimal filesystem-backed `/v1/files` + `/v1/batches`; D8
  prometheus-client + stdlib JSON logs, no OTel.
- Why: The product-infrastructure review (LB/scaling, WAF, k8s, GPU pool + API layer,
  cache layer, batch orchestrator, DC–cloud interconnect) found all deployment
  machinery absent: components exist in-process (ReplicaPool, remote-replica backend)
  but nothing wires, launches, secures, observes, or packages them.
- Refs: `docs/goals/g3-production-deployment.md`, `docs/design/m7-productionization.md`;
  amendment: g2 §6 "exactly 2 nodes" clarified as TP/PP coherence-domain cap, not a
  ReplicaPool fleet-size cap (`docs/goals/g2-multi-gpu.md` §6, G3 §5).

### 2026-07-02 — [progress] Repo renamed to `ytworks/kairyu`; README refreshed for M5/M6
- What: GitHub repository renamed from `ytworks/rLLM` to `ytworks/kairyu` (local origin
  updated). README brought up to date: M5/M6 rows in the roadmap, TP / P-D / KV-transport /
  PP components in architecture, engine-core and project-layout sections, test count 290+,
  clone URL. Remaining `rLLM` references in CLAUDE.md / AGENTS.md / gpu-runbook fixed.
- Refs: README.md, CLAUDE.md, AGENTS.md, docs/gpu-runbook.md

### 2026-07-02 — [progress] M5/M6 GPU-independent halves implemented (177 → 289 tests)
- What: All CPU-testable pieces of both designs landed with tests (95% coverage):
  M5 — `Communicator`/`FakeCommunicator`, typed immutable `StepInput`, `TPModelRunner`
  (divergence-checked driver protocol; TP=2 greedy-equivalent to TP=1 through
  KairyuBackend), `tensor_parallel_size` plumbed end-to-end (no-op resolved),
  `ReplicaPool` (rendezvous-hash affinity, queue-depth valve, health ejection,
  `record_replica` JSONL), `PDCoordinator` + `LocalKVHandoff` + `Scheduler.resume_with_kv`
  (copy-before-commit ordering, preemption shield, P-D greedy-equivalence).
  M6 — `ClusterSpec` (topology validation), `KVTransport` protocol + `LocalFabric` +
  TCP-loopback transport, `bench/kv_transfer_bench.py` (CPU-runnable, real fragment
  layout), `openai_backend` replica fixes (real SSE streaming, pooled client, optional
  auth, token counts), async submit/handle runner contract + `PipelinedModelRunner`/
  `PipelinedEngineCore` (inter-step pipelining, bubble accounting: depth-2 <0.2 vs
  depth-1 ≈0.5 pinned by test). GPU-runbook §6/§7 added for the GPU days.
- Refs: `kairyu/engine/core/{comm,step_input,tp_runner,pd,kv_transport,pipeline}.py`,
  `kairyu/orchestration/{replica,cluster}.py`, `docs/gpu-runbook.md` §6–7

### 2026-07-02 — [design] M5/M6 designs written and reviewed (APPROVE-WITH-AMENDMENTS)
- What: `docs/design/m5-intra-node-parallelism.md` (TP runner with non-rank driver +
  typed StepInput prerequisite, ReplicaPool with session affinity, P-D copy-on-handoff
  with copy-before-commit protocol and `resume_with_kv`) and
  `docs/design/m6-inter-node-parallelism.md` (static ClusterSpec — no Ray, KVTransport
  with fragment aggregation, streamed P-D with layer-group final chunk, PP=2 via
  inter-step pipelining on an async ModelRunner handle). Three-reviewer agent panel
  fixed 6 blockers: zero-copy P-D donation withdrawn (dual-tree pool accounting unsound
  + incompatible with disjoint-GPU roles); PP intra-step micro-batching withdrawn
  (bounded at ~1.33× vs B4's 1.6×); `openai_backend` "no change" claim corrected (fake
  streaming, per-request client, mandatory auth, empty token_ids all block B1).
- Why: G2 requires reviewed design docs before implementing each milestone; review
  against the real scheduler/radix code caught mechanisms that could not work as drafted.
- Refs: `docs/design/m5-intra-node-parallelism.md` §7, `m6-inter-node-parallelism.md`
  §7, `docs/goals/g2-multi-gpu.md` §7 Amendments

### 2026-07-02 — [amendment] m1 "Ray arrives with multi-node" superseded
- What: M6 D1 uses a static ClusterSpec + torchrun-style rendezvous for the 2-node
  topology; Ray is not adopted.
- Why: G2 excludes elasticity; a dynamic-placement framework for two static nodes fails
  YAGNI. m1 §3/D4's note was forward-looking, not a binding decision.
- Refs: `docs/design/m6-inter-node-parallelism.md` D1;
  `docs/design/m1-orchestration-and-interface.md` §3

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
