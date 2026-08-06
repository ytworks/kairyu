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

Snapshot date: 2026-08-06. Hardware context: all GPU evidence so far is on
8× RTX PRO 6000 Blackwell (SM120), PCIe-only interconnect (P2P 30–37 GB/s);
NVLink-HBM (H100-class) formal gates still need hardware. Evidence lives in
`bench/results/` (see `index.json`); decisions and rationale in `docs/design/`.

### Milestones

| Milestone | Status |
|---|---|
| M1 Orchestration+Interface | Complete: Router/Conductor/MoA, vLLM-compat API, OpenAI server, DSL |
| M2 Core engine | CPU half done; unified EngineLoop + device sampling GPU-validated; NVLink perf gates pending |
| M3 Spec/graphs/P-D | n-gram spec, CUDA-graph serving, intra-node P-D production GPU-validated; EAGLE runtime deferred to G4 |
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
- Attention backends: `auto`/torch/FlashInfer/FA3/FA4 with `/backends` reporting; CUDA-graph decode opt-in
- Quantized serving: FP8/INT8/AWQ/GPTQ/NVFP4 without full dequantization; opt-in FP8 EAGLE/MTP draft loading
- Fully device-side sampling, penalties, spec verification, page-table caching; incremental detokenization and stop matching
- Hardened gateway: auth, tenancy metering/invoicing, priority + SLO admission, batch API, embeddings/RAG, Responses API
- Orchestration (Conductor/MoA) with streaming, usage accounting, trace v2; Codex CLI and IDE tool-calling work end-to-end
- Fleet: 3-gateway HA with PostgreSQL BatchStore, KV-aware prefix routing, DRAM KV tiering, Helm chart + kind CI drill
- Benchmark/eval tooling: `kairyu bench run` (Fugu + core suites), hash-chained quality history, config A/B and quant sweeps
- Process-split backend (`kairyu-proc`) with delta wire, TP group attestation, graceful lifecycle
- CPU suite green (thousands of tests, no selected skips); CPU microbenchmark smoke + nightly regression series in CI

### Open items / blockers

- G2 A6 performance gap vs vLLM is the open hard gate; full TP4/8 HTTP matrix deferred until closed
- Issue #333 verdict: process-split is not the A6 cause (`no_material_reduction`, ratio 0.92 vs ≤0.90 line)
- Production stage-sharded pipeline parallelism is a separate roadmap dependency (current PP report is not it)
- EAGLE-3 runtime integration remains a G4 follow-up; FP8-E4M3 KV disabled pending offline calibration
- NVLink-profile gates blocked on H100/A100-class hardware; PCIe-switch chassis and ≥400 Gb/s RDMA NICs gate E4/E5
- G6 remaining P-C gates still in progress
- Human sign-off pending on M2–M4 design reviews

## Change Log

Newest first; only the most recent entries are kept here (see the size budget
in `.claude/rules/progress-log.md`).

### 2026-08-06 — [design] PROGRESS.md restructured for minimal session-start tokens
- What: archived the verbose Current Status to `docs/progress/archive/status-2026-08-06.md` and all older Change Log entries to `docs/progress/archive/change-log.md` (both verbatim); PROGRESS.md now holds a compact Product/Current Status/Change Log. Size budgets and archiving live in `.claude/rules/progress-log.md` + `docs/progress/archiving.md`, enforced by a SessionStart hook running `scripts/check_progress_size.py`.
- Why: PROGRESS.md and the progress rules are loaded every session; at ~514 KB they burned the context they were meant to save.
- Refs: `docs/progress/archive/`; `.claude/settings.json`; `scripts/check_progress_size.py`

### 2026-08-06 — [progress] Repeated token SSE shapes bypass per-token model construction
- What: added stream-owned byte encoders for repeated role-less Chat content, unfinished legacy Completion text, and Responses output-text deltas. Constant JSON envelope fragments and per-index prefixes are retained per stream, only the dynamic scalar is serialized per chunk, and Starlette receives bytes directly. Strict predicates keep first-role, non-string/non-integer, logprob, tool, finish, usage, trace, error, and terminal shapes on their existing serializers. Hoisted fallback exclusion sets and byte-for-byte golden tests bind field order, usage omission/null, multiple indices, empty/control/Unicode content, SSE line separators in both constants and deltas, malformed boundaries, and operation counts. Same-process 100,000-iteration diagnostics measured 7.804→0.570 µs (13.69x) for Chat, 4.625→0.569 µs (8.13x) for Completions, and 3.847→0.753 µs (5.11x) for Responses.
- Why: constructing and validating nested Pydantic models plus serializing the complete roughly 200-byte envelope for every small token delta imposed avoidable CPU work and allocations on all dominant streaming paths.
- Refs: issue #334; `kairyu/entrypoints/server/{sse_encode,app,responses_service}.py`; `tests/server/test_sse_encode.py`

### 2026-08-06 — [progress] Retained benchmark evidence gains a strict tracked catalog
- What: added `bench/results/index.json` with one path-sorted record for all 63 Git-tracked top-level artifacts, including nullable recorded gate/date/source-commit/hardware/verdict metadata and authoritative bundle summaries. A package-owned strict parser and checkout validator reject duplicate/non-finite/non-canonical JSON, schema or slug drift, unsafe or symlinked paths, untracked summaries, and missing/stale Git coverage while excluding ignored and untracked runtime output. The read-only verifier now runs in portable CI alongside the entrypoint and wheel-boundary checks.
- Why: retained flat files and bundles were discoverable only through narrative documentation, so trend tooling and evidence audits could silently miss new roots or invent provenance for heterogeneous historical artifacts.
- Refs: issue #383; `bench/results/index.json`; `kairyu/bench/results_index.py`; `scripts/verify_bench_results_index.py`; `tests/bench/test_bench_results_index.py`; `bench/README.md`; `docs/benchmarks.md`

### 2026-08-06 — [progress] Benchmark profiling gains shared, bound trace artifacts
- What: added a lazy `kairyu.bench.profiling` context that preserves caller-owned warm-up, synchronization, and measured scopes while mapping explicit CPU/CUDA activities without fallback. Migrated the two checkout benchmark users and all nine GPU-test users from direct profiler construction. `serving_bench.py --profile` now records one CPU-only local-client range and publishes a private, strict-JSON, 64 MiB-bounded `*.client.pt.trace.json` sidecar through same-filesystem temporary export and exclusive hard-link publication; the paired UTC-microsecond result binds its relative name, format, size, SHA-256, activity, local scope, target exclusion, and diagnostic-only status. Both members publish without overwriting a concurrent winner, and result-publication failure rolls back only its exact untampered trace. The example serving reporter rejects profiled diagnostics rather than mixing them into timing comparisons. Disabled/help/core-wheel paths do not import torch, and missing torch, unavailable CUDA, unsafe or colliding paths, invalid exports, and publish races fail closed.
- Why: ad hoc profiler setup had divergent options and no common artifact identity or overwrite policy, while a serving-client trace could otherwise be misrepresented as remote server or GPU evidence.
- Refs: issue #380; `kairyu/bench/profiling.py`; `bench/{serving_bench,future_token_bench,batched_prefill_qwen}.py`; `tests/{unit/test_bench_profiling.py,gpu}`; `docs/benchmarks.md`; `bench/README.md`; `docs/gpu-runbook.md`

### 2026-08-06 — [design] Structured output gains paired schema-conformance evidence
- What: added the dedicated `structured` benchmark suite with a package-owned, SHA-256-addressed Draft 2020-12 corpus spanning nested, recursive local-reference, enum, regex-pattern, and union schemas. Each selected item and sampling seed retains one counterbalanced constrained/control pair whose wire bodies differ only by `response_format`; strict outer-response and completion JSON parsing, an independently pinned `jsonschema` evaluator, and type-sensitive exact-answer scoring separate request acceptance, JSON validity, schema conformance, task success, malformed output, endpoint-reported token coverage/deltas, and diagnostic latency. Explicit structured-schema 400/422 responses remain measured conformance failures, while unrelated HTTP/transport/malformed-envelope failures withhold the complete claim. Valid HTTP 200 refusals, content-filtered empty messages, and unexpected call payloads remain accepted non-JSON/task-failure evidence instead of becoming environment faults. Raw arms and all derived metrics are cross-validated; exact config-selected corpus IDs, rejected control constraints, fixed cache/source identities, installed evaluator distributions, and subset incomparability are rebound at runner, report, and fresh-history boundaries. Fresh scoreboards recompute the detailed claim, schema-1 history proves then strips it, and stored detached claims are rejected. The suite, wheel fixture, cache manifest, evaluator resources/dependencies, docs, and example config are isolated from the frozen Core/quantization matrices.
- Why: the previous one-object smoke assertion did not exercise common schema features, compare enforced generation with a matched control, quantify malformed output or token cost, or retain replayable evidence capable of detecting semantic-quality and reporting regressions.
- Refs: issue #375; `kairyu/bench/{structured,types,aggregate,history}.py`; `kairyu/bench/adapters/structured_output.py`; `kairyu/bench/fixtures/structured-output.jsonl`; `examples/bench_structured.yaml`; `tests/bench/test_bench_structured_output.py`; `docs/benchmarks.md`

### 2026-08-06 — [design] Chat evaluations gain grouped sampling-sensitivity evidence
- What: extended generative benchmark adapters so `attempts > 1` runs one ordered consecutive-seed sweep per source item, retaining strict child outcomes beneath that item rather than flattening correlated repeats. Complete rows recompute seed means, sample SD, range, and unbiased `1-C(n-c,k)/C(n,k)` pass@k for declared binary adapters; incomplete or tampered matrices withhold the sensitivity summary, and ordinary Wilson intervals plus configuration A/B are disabled for multi-attempt evidence. Raw source-item identities, canonical parent/pair status and reason, denominators, realizable binary pass@k margins, and fresh history labels are cross-validated. Runner, report, and history boundaries also bind target seed/mode/temperature plus the attempt budget to retained methodology; schema-1 history omits the derived label after proving it against the raw pair, while a new protocol marker preserves old agentic multi-trial records. SciCode independently reruns each seed's whole sequential problem chain. Targets can select an explicit temperature or `sampling_mode: recommended`, which omits temperature/top-p/top-k/min-p/repetition-penalty so endpoint generation defaults may apply while explicitly declining remote attestation. External harnesses fail closed when those chat-only policies cannot be forwarded, and SWE-Bench Pro requires one attempt. The default single-attempt target serialization, target-config fingerprint input, temperature-zero request bytes, and absent seed remain byte-shape compatible.
- Why: one deterministic chat call cannot quantify sampling noise or pass@k, but flattening repeated seeds would understate uncertainty and corrupt paired statistics. Explicit omission is also required to measure the model-owned generation defaults added by issue #351 without falsely claiming that a remote server applied them.
- Refs: issue #371; `docs/benchmarks.md`; `examples/bench_{fugu,core}.yaml`; `kairyu/bench/{sampling,types,aggregate,config,cli,config_ab,history}.py`; `kairyu/bench/adapters/{base,ifeval,scicode,swebench_pro,terminal_bench,tau_bench}.py`; `tests/bench/test_bench_{sampling_sensitivity,config,config_ab,history,agentic_conditions,scicode_sequential,runner}.py`

