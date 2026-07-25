# Progress

Cross-session memory of design changes and project progress.
Maintained per the rules in `.claude/rules/progress-log.md`.

## Current Status

**GPU bring-up in progress (2026-07-23): the CPU-safe full suite is green
(1409 passed, 12 skipped, 92% cov), the single-GPU kernel suite is green
(10 passed), and the explicit 2-GPU NCCL EP gate is green on the 8× RTX PRO
6000 Blackwell host. The three gate-integrity problems found during the first
hardware run (Issues #102–#104) are fixed and GPU-verified in PR #105. Real-model parity,
performance, and production/fabric drills remain. The all-visible-GPU Qwen3-32B
Compose and benchmark/report workflow are implemented but still need validation
on the GPU host.**

_Last updated: 2026-07-23_

Master roadmap: `docs/roadmap.md` (2026-07-03) — dual hardware profiles (NVLink-HBM
A100/H100/B200 nodes AND the PCIe-only RTX PRO 6000 fleet, A100 and later all
supported), three tracks (E: L1 engine → SOTA incl. MoE, F: fleet-scale control
plane, G6/P: product surface). Next actions: **E1** (single-GPU real engine — RTX
6000 Pro units are available now) + **P-A** (truthful API core, CPU).

| Milestone | Status |
|-----------|--------|
| M1 — Orchestration (L2) + Interface (L3) | Complete and merged. Router / Conductor / MoA, vLLM-compatible `LLM` + `AsyncLLMEngine`, OpenAI-compatible server, YAML/decorator DSL. Atomic pre-dispatch reservations enforce strict step admission and serialize result-priced work under configured cost caps without hiding a single admitted generation's actual-cost overrun. |
| M2 — Core engine (overlap scheduler + Radix-Paged KV) | CPU half done: scheduler, KV manager, EngineCore step loop, overlap pipeline, pre-GPU robustness (EOS, preemption, abort, pin TTL). Paged-KV attention validated with real tensors on CPU (greedy-equivalence). **Blocked on GPU hardware** for the GPU phase. |
| M3 — Spec decode / CUDA graphs / P-D separation | n-gram draft spec-decode policy and xgrammar structured output implemented CPU-side. CUDA graphs and the rest gated on M2 GPU phase. |
| M4 — Router learning pipeline | Implemented CPU-only (logs → distilled classifier → contextual bandit). Design reviewed. |
| M5 — Intra-node multi-GPU (TP, DP replicas, P-D intra-node) | Design reviewed; **CPU half done** (Communicator/StepInput/TPModelRunner, TP plumbing live, ReplicaPool + affinity, PDCoordinator + `resume_with_kv`). GPU phase: `docs/gpu-runbook.md` §6, prereq M2 Gates 1–3. |
| M6 — Inter-node multi-GPU (2-node DP, KV transfer plane, P-D inter-node, PP) | Design reviewed; **CPU half done** (ClusterSpec, KVTransport + loopback + `bench/kv_transfer_bench.py`, openai_backend replica fixes, async runner contract + PipelinedEngineCore). GPU phase: runbook §7, prereq all M5 gates. |
| M7 — Productionization (serve CLI, gateway wiring, batch, observability) | **CPU half done** (design m7 D1–D8, goal G3): health/readyz/metrics/auth/concurrency guard, `kairyu serve` + DeploymentSpec, ReplicaPool gateway wiring + prober, HTTP session affinity, batch API, Dockerfile + compose + CI smoke drill, `docs/deployment.md`. GPU bring-up: runbook §9. |
| M8 — Engine CPU core (real tokens/sampling/multi-token commit/spec decode/quant基盤/process split) | **Complete** (2026-07-03, `docs/design/m8-engine-cpu.md`): HF tokenizer seam + SSE-safe stop strings, full sampler + xgrammar in-path, scheduler spec reservation, n-gram SpeculativeRunner (spec ≡ greedy pinned), NVFP4/HardwareProfile/safetensors reader, ZMQ `kairyu-proc` process split. 437 tests, 95% cov. |
| M9 — Truthful API (usage/templates/logprobs/completions/n>1) | **Complete** (2026-07-03, `docs/design/m9-truthful-api.md`): G6 P-A gates CPU-green — real usage + cached_tokens + include_usage, HF Jinja templates (transformers byte-match), logprobs + /v1/completions, n>1 fan-out, response_format validation, bench token-TPOT. 471 tests. |
| M12 — Real model zoo dense (Llama/Qwen, PagedKVPool, PagedModelRunner) | **Complete** (2026-07-03, `docs/design/m12-model-zoo.md`): full-engine greedy == transformers generate (3 archs); loader + model_path wiring; pytest gpu/hf_hub/dist markers. 501 tests. |
| M13 — AttentionBackend seam (torch/MLA reference/FlashInfer adapter/selector) | **Complete** (2026-07-03, `docs/design/m13-attention-backend.md`): fake-pinned FlashInfer contract + tests/gpu mirror; MLA two-form equivalence oracle. 514 tests. |
| M14 — Quant compute (fp8/int8/awq/gptq/nvfp4 CPU references + Triton stubs) | **Complete** (2026-07-03, `docs/design/m14-quant-compute.md`): all 5 schemes load + run through the full engine on CPU; formats pinned vs live Hub checkpoints. 530 tests. |
| M15 — MoE + MLA archs (Qwen3-MoE, DeepSeek-V3 incl. yarn) | **Complete** (2026-07-03, `docs/design/m15-moe-mla.md`): full-engine greedy == hf.generate; latent MLA pool (M18-ready). 547 tests. |
| M16 — Distributed execution (gloo-tested TP/EP/PP; NCCL by constructor) | **Complete** (2026-07-03, `docs/design/m16-distributed.md`): TP=2/EP=2/PP=2 spawn parity gates green in the default suite. 553 tests. |
| M17 — StepExecutor (CUDA-graph seam) + EAGLE-3/MTP drafts | **Complete** (2026-07-03, `docs/design/m17-graphs-drafts.md`): fake-graph lifecycle suite; perfect-draft e2e ≡ greedy; corrected EAGLE-3/MTP formats. 571 tests. |
| M18 — KV transport (serde/remote handoff/NIXL adapter) + 2-process P-D | **Complete** (2026-07-03, `docs/design/m18-kv-transport.md`): TCP byte-parity E2E green. 584 tests. |
| G4 — MoE engine (fused experts, EP, MTP, NVFP4, MLA) | Goal defined (`docs/goals/g4-moe-engine.md`); lifts the G2 MoE non-goal. Design doc + review required before implementation. |
| M10a — Elastic fleet base (dynamic pool/registry/tracing/Helm) | **Complete** (2026-07-03, `docs/design/m10-fleet-cpu.md`). 594 tests. |
| M10b — KV-aware routing (prefix trie / KV events / offline tuning) | **Complete** (2026-07-03). 610 tests. |
| G5 — Fleet scale (elasticity, KV-aware routing, P/D pools, tiering, tenancy) | Goal defined (`docs/goals/g5-fleet-scale.md`); amends m7 D2 (k8s as machine layer), m5 D4/m7 D6 (prefix-aware placement), m6 D1 staticness, ClusterSpec cap, m7 D8 (OTel). F1/F2 are CPU-mock-testable now. |
| M11 — Product surface + tenancy (streaming auto/tenancy/responses/embeddings/F5) | **Complete** (2026-07-03, `docs/design/m11-product.md`). 627 tests. |
| G6 — Product surface (truthful API, Fugu-class product, frontier scoreboard) | Goal defined (`docs/goals/g6-product-surface.md`). P-A (usage truth, HF chat templates, logprobs, structured outputs) is CPU work, start now. |

What works today: full stack on CPU — `kairyu` EngineBackend wired through the
OpenAI-compatible server with the mock/CPU runner; serving/router/multiturn benchmarks
in `bench/`; `kairyu serve <deployment.yaml>` runs a hardened gateway (pool of remote
replicas, auth, metrics, batch) or a replica node, and the compose topology
(1 gateway + 3 mock replicas) passes the CI smoke drill incl. kill/recover.
Router inspection now has a non-dispatching, non-mutating `/v1/route` preview that shares
chat prompt rendering, an authenticated `/routing` descriptor, and opt-in structured actual
decisions; the GPU compose mounts an explicit routing spec for `kairyu-auto`.
Unary orchestrated responses can now opt into versioned `kairyu_trace_v2` events with
route/role status, attempts, timing, resolved engine/model, token usage, budget deltas, and
sanitized failures while retaining the legacy string trace and excluding prompt/output text.
The vLLM-compatible `AsyncLLMEngine` now owns an explicit registry of active
request IDs: inactive aborts are stateless, while an active abort interrupts and
closes its backend stream without poisoning later reuse of the same ID.
`OpenAICompatBackend` SSE preserves every observed choice index, including empty single
choices and mixed empty/non-empty `n > 1` results, while rejecting streams with no choices.
`BatchStore` exposes owner-scoped lazy binary-line iteration, metadata-last streaming
upload transactions, and transactional lazy JSONL writers. The files route reads fixed-size
chunks, applies its byte limit incrementally, and removes partial uploads on rejection,
cancellation, or storage failure. The batch worker streams input through a bounded queue and
fixed consumer pool, spools results incrementally, and persists controlled terminal failure
while rolling back partial result publications after ordinary processing or storage exceptions.
Each batch row now validates a typed method/URL/custom-ID envelope and enters the same
chat validation plus buffered-dispatch service as regular HTTP requests; invalid rows never
reach an engine, and backend error records reveal only the exception class.
Embedding backends are configured and discovered as explicit, non-colliding model IDs;
requests resolve that bounded registry before work, unknown IDs return `model_not_found`,
response, metric, and ledger identities use the resolved key, and limiter charging occurs
only after resolution.
Required and named tool choice is enforced independently for every returned choice after
filtering; mixed or empty results are rejected before response or buffered stream emission,
without regeneration, and the consumed generation remains metered exactly once.
Tenant usage accounting now covers synchronous and streaming generation, Responses,
embeddings, and successful batch lines with authenticated ownership and backend-or-derived
wire-count parity; each dispatched execution records exactly once even when a stream closes
early or a completed batch line is later rolled back by cancellation or spool failure.
The app lifespan owns each persistent usage-ledger append handle and closes it only after
inner worker/backend cleanup, including exceptional shutdowns. Aggregation skips and logs
truncated tails or complete malformed records while preserving every valid usage total.

The Open WebUI Compose topology is clean-checkout runnable with a standalone
`default` mock DeploymentSpec; CI validates its binds/rendered internal endpoint and
smokes only Kairyu readiness, exact model discovery, and completion without pulling the
mutable UI image.

The Helm chart has CPU-safe defaults plus a GPU overlay that requests one NVIDIA
GPU, selects the configured runtime/node profile, mounts an existing host path or
PVC read-only, and starts the real Kairyu engine from `/models/checkpoint`.
The checked-in SM120/`pcie-gddr` profile pins the torch attention fallback while
the strict chart value also permits FlashInfer on supported hardware.
CI now schema-lints and template-renders both the CPU defaults and GPU overlay
before the kind CPU deployment/HTTP drill; it does not schedule the GPU pod.
`kairyu bench run` executes the 11-slot Fugu-release quality suite against any
deployed gateway (single models and named orchestrations as scoreboard columns)
with dataset downloaders, LLM-judge/vision/docker degradation, and a dated
footnoted scoreboard (G6 P-C1).

Active blockers: RTX 6000 Pro units are now partially available — M2/E1 GPU phase is
unblocked on the PCIe profile (H100 boxes still wanted for NVLink-profile gates);
execution plan is `docs/gpu-runbook.md` + `docs/roadmap.md` §4. Hardware procurement
(PCIe-switch chassis, ≥400 Gb/s RDMA NICs) gates E4/E5 and is decided during E3 from
E1's measured P2P matrix. Human sign-off pending on M2–M4 design reviews.

## Change Log

### 2026-07-26 — [design] GraphDecodeBackend: the CUDA-graph decode capability contract
- What: a backend is capture-eligible only if it DECLARES it, and the decode step
  boundary now reaches it. Defined once in `kairyu/engine/core/attention/__init__.py`
  as the `GraphDecodeBackend` protocol plus its single enforcement point
  `graph_capture_gap()`: `supports_graph_capture` (declared, not inferred),
  `plan_decode(kv_pool, page_tables, seq_lens, *, num_qo_heads, q_dtype)` (the
  step-boundary HOST phase, a no-op where there is none), and the capture-safe
  `attend_decode`. `GraphStepExecutor` gained an optional `plan_fn`, called before
  each capture and after every `_copy_in` BEFORE `replay()`; `PagedModelRunner`
  supplies it via the new `DenseDecoder.plan_decode_tensors()`, which plans once per
  backend INSTANCE per step (not per layer). `_tensor_decode_gap()` now asks
  `graph_capture_gap()` instead of `hasattr(backend, "attend_decode")`.
- Why: method presence is not capture-safety. FlashInfer (PR #141) owns
  `attend_decode` and would have passed the old check, but its `plan()` copies
  `indptr` to the host and cannot run under capture at all — so the runner would
  construct successfully and the FIRST capture would die with a D2H `RuntimeError`,
  defeating the fail-fast the gate exists for. And planning outside capture is
  useless if nothing calls it: `forward_decode_tensors` reaches `attend_decode` per
  layer with no seam for a host phase, and only the executor knows which static
  buffers the next replay will read — the plan must therefore be taken after
  copy-in, or the replay attends over the previous step's pages.
- Refs: m17 D1/D2, PR #138 review [P1], PR #141 review [P1];
  `kairyu/engine/core/attention/{__init__,torch_backend}.py`,
  `kairyu/engine/core/{model_runner,step_executor}.py`,
  `kairyu/models/{attention,llama}.py`, `tests/unit/test_step_executor.py`,
  `tests/unit/test_graph_decode_wiring.py`.

### 2026-07-26 — [amendment] CUDA-graph decode: reserved scratch page + backend fail-fast
- What: `PagedModelRunner(graph_backend=...)` now wires the m17 D1 capture seam into
  batched decode (PR #138), with two review blockers fixed before merge. [P1] the
  scratch page the graph's padding rows write KV to is now RESERVED out of the
  allocator via the new `RadixKVCache.reserve_scratch_page()`, and a graph backend
  without a cache is rejected; `GraphStepExecutor`/`build_decode_batch` no longer
  default `scratch_page` to 0. [P2] the runner now checks at construction that every
  layer's attention implements the tensor decode contract
  (`forward_decode_tensors` + `backend.attend_decode`) and raises `ValueError`
  naming the gap. The seam stays OFF unless a backend is passed.
- Why: the scratch page defaulted to 0, which `PagePool` hands out as the FIRST
  ordinary page — so the capture warmup and every partial-bucket replay wrote K/V
  into a live request's page 0 slot 0, silently corrupting its cache whenever the
  damage did not cross an argmax boundary. And a FlashInfer or MLA model constructed
  fine and then died with `AttributeError` on the first batched decode, arbitrarily
  deep into a run, rather than at build time. Capacity now drops by exactly one page
  for the graph's lifetime — the documented cost of the reservation.
- Refs: m17 D1/D2/A5, `docs/gpu-runbook.md` §6.3, PR #138 review [P1]/[P2];
  `kairyu/engine/core/{model_runner,step_executor,radix_kv}.py`,
  `tests/unit/test_graph_decode_wiring.py`, `tests/gpu/test_cuda_graph_decode_gpu.py`.

### 2026-07-26 — [amendment] FlashInfer decode is split into a host plan and a capture-safe run
- What: `FlashInferBackend.attend_decode` no longer derives-and-plans inline. The adapter
  now owns `plan_decode()` (the HOST phase: device-derived indptr/indices/last_page_len
  handed to a `use_cuda_graph=True` wrapper whose paged buffers are persistent, one
  wrapper per (batch, max_pages) shape and never replaced) and `attend_decode()` (a bare
  `run()` over those buffers — no `.tolist()`, no `.cpu()`, no `plan()`). Inside a capture
  the adapter refuses to plan and refuses to run unplanned; eagerly it still plans lazily,
  now once per step instead of once per layer. Gated by a real `torch.cuda.CUDAGraph`
  capture on the 8× RTX PRO 6000 host whose replay reflects an in-place page-table/seq-len
  change after the step is re-planned, plus a CPU gate that forbids host synchronization
  inside the capture region.
- Why: the first cut of PR #141 claimed the m17 D1 tensor contract but converted the page
  table and lengths with `.tolist()`/`.cpu()` on every layer, so capture died with
  `cudaErrorStreamCaptureInvalidated` (reproduced on hardware) and the path was eager-only.
  FlashInfer's `plan()` "cannot be used in Cuda Graph" by its own documentation — it builds
  the split-KV schedule on the CPU — so the honest decomposition is plan-outside /
  run-inside, which is what the wrapper's cudagraph buffers exist for.
- Refs: PR #141 review [P1]; m17 D1, m13 D4; `kairyu/engine/core/attention/flashinfer_gpu.py`,
  `tests/unit/test_attention_backend.py`, `tests/gpu/test_flashinfer_tensor_decode.py`.

### 2026-07-25 — [amendment] Review remediation across the Fugu bench alignment PRs
- What: Addressed the review findings on the nine bench PRs. Highlights: pinned revisions
  are now passed to every fetch (they were recorded but unused, so the cache and run
  fingerprint attested unpinned bytes as pinned); LiveCodeBench Pro fails closed on any
  partial fetch instead of caching a shrunken denominator; MRCRv2 selects the official
  `o200k_base` prompt+answer bins (exactly 500 rows) rather than a chars/4 approximation;
  SciCode scores the official 288 sub-steps, supplies the three the evaluator skips, and
  pins `test_data.h5` by content hash; Harbor/τ commands match the real 0.17/tau2
  contracts (`trial_results` + `verifier_result.rewards`, imported `DATA_DIR`, unique
  `--save-to`); `extra_body` can no longer override built request or sampling fields; the
  progress reporter can no longer abort a run and heartbeats agentic slots; comparability
  is decided per cell so subset/fixture/dynamic-substitution runs withhold their deltas;
  and the Qwen example declares its text-only target so vision slots skip honestly.
- Why: Each finding let a number look like something it was not — a false provenance
  attestation, a shrunken denominator, a different population, or an unmarked delta
  against a published full-suite score.
- Refs: PRs #115–#123 review comments; `kairyu/bench/**`, `tests/bench/**`,
  `tests/unit/test_qwen_fugu_example.py`, `examples/qwen3-32b-multi-gpu/**`,
  `docs/benchmarks.md`.

### 2026-07-25 — [progress] One-command Fugu quality benchmark for the Qwen3-32B example
- What: Added `examples/qwen3-32b-multi-gpu/{run-,}fugu-benchmark.sh`. The first starts the
  all-GPU service (reusing one already running), waits for readiness, and chains into the
  second, which preflights the served model by exact id, declares the text-only target
  with `--no-vision`, runs `kairyu bench run` from the repository with the `bench` extra,
  and points the operator at both `scoreboard.md` and `comparison.md`. Every Fugu
  condition is reachable by environment variable (`REASONING_EFFORT`,
  `JUDGE_REASONING_EFFORT`, `EXTRA_BODY`, `ATTEMPTS`, `BENCH_ONLY`); `BENCH_LIMIT` defaults
  to 20 items per slot and announces itself as a subset run, with `BENCH_LIMIT=0` for the
  full suite. `PORT` reaches the Compose mapping. A static test pins the properties that
  would otherwise silently produce a misleading number.
- Why: The existing example only measured throughput. The quality suite runs on the host
  rather than in the serving image because the image carries no dataset dependencies —
  documented rather than worked around.
- Refs: `examples/qwen3-32b-multi-gpu/{fugu-benchmark.sh,run-fugu-benchmark.sh,README.md,
  compose.yaml,run.sh}`, `kairyu/bench/{cli,config}.py`,
  `tests/unit/test_qwen_fugu_example.py`. Verified end-to-end against a mock gateway
  (all 11 slots, progress, vision skips, subset banner, accuracy report); the GPU host run
  is still pending.

### 2026-07-25 — [amendment] MRCRv2 scores Fugu's 8-needle / 128K slice
- What: The MRCR adapter averaged all 2,400 published rows, which mix 2-, 4- and
  8-needle items across context lengths up to 1M tokens. It now selects only
  `n_needles == 8` rows whose estimated prompt tokens are <= 131,072 — Fugu's reported
  conditions — preferring the dataset's own `n_chars` over the chars/4 heuristic,
  recording the per-row estimate, printing how many rows each filter excluded, and
  failing closed if the slice is empty. The `n_needles` field was already carried in the
  payload but never used.
- Why: The scoreboard cell claimed comparability with Fugu's MRCRv2 number while
  measuring an easier, shorter population; needle count and context length are the two
  variables this benchmark exists to vary.
- Refs: `kairyu/bench/adapters/mrcr.py`, `kairyu/bench/fixtures/mrcr-v2.jsonl`,
  `tests/bench/test_bench_mcq_adapters.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] Agentic bench harnesses use real flags and Fugu's turn/trial limits
- What: All three agentic wrappers built commands the installed harnesses reject, so those
  cells could only report `failed`. `harbor run` has no `--output-dir` (that belongs to
  `harbor jobs download`) and selects datasets as `name@version`, not
  `terminal-bench/terminal-bench-2-1`; the τ harness has no `--output` (results are
  addressed by `--save-to <name>` under its `TAU2_DATA_DIR`) and has no `banking` domain
  (only `banking_knowledge`). Fixed all of those, then pinned Fugu's conditions:
  `agent.step_limit=1000` for SWE-Bench Pro (harness default 250, restating
  `swebench.yaml` because `-c` discards the default config), `--ak max_turns=500` for
  terminus-2, and `--retrieval-config alltools` plus `--user-llm-args` carrying the
  judge's reasoning effort for τ³. New `--attempts` (default 1) drives Harbor `-k` and τ
  `--num-trials`, with annotations recording that Fugu reports τ³ as pass@4 and the
  Terminal-Bench leaderboard wants ≥5.
- Why: A `failed` cell was indistinguishable from a genuinely bad score, and the
  published turn budgets are the difference between a truncated trace and the reported
  condition.
- Refs: `kairyu/bench/adapters/{swebench_pro,terminal_bench,tau_bench}.py`,
  `kairyu/bench/{types,cli,config,runner}.py`, `kairyu/bench/adapters/base.py`,
  `tests/bench/{test_bench_agentic_conditions,test_bench_tau,test_bench_agentic}.py`,
  `docs/benchmarks.md`.

### 2026-07-25 — [amendment] Bench targets own a sampling policy (reasoning effort)
- What: `BenchTarget` and `JudgeConfig` now share a `SamplingOptions` base carrying
  `reasoning_effort`, `top_p`, `seed`, and `extra_body_json`; `call_chat` merges those
  into every request body, and the judge client forwards its own. New flags:
  `--reasoning-effort`, `--top-p`, `--sampling-seed`, `--extra-body`,
  `--judge-reasoning-effort`, `--judge-extra-body`, all of which also override
  YAML-declared targets. `extra_body_json` is validated at load time (JSON object; may
  not override `model`/`messages`/`stream`) and stays a string so the frozen models
  remain hashable. Every field is part of the run fingerprint.
- Why: Fugu reports its table at each model's maximum reasoning effort and ran the τ³
  user simulator at `low`. The request body previously carried only
  model/messages/temperature/stream/max_tokens, so neither condition could reach the
  wire — and for Qwen3 the thinking toggle lives in `chat_template_kwargs`, which needs
  the same escape hatch.
- Refs: `kairyu/bench/{types,config,cli,judge}.py`, `kairyu/bench/adapters/base.py`,
  `tests/bench/test_bench_sampling.py`, `docs/benchmarks.md`, `examples/bench_fugu.yaml`.

### 2026-07-25 — [amendment] Bench dataset revisions are pinned in one registry
- What: Added `kairyu/bench/pins.py` mapping adapter name → (dataset id, commit sha) for
  HLE, GPQA Diamond, CharXiv, SciCode, MRCRv2, LongBench-v2, and LiveCodeBench Pro;
  `all_adapters()` fills each adapter's unset `hf_revision` from it per instance. A pin
  applies only when the recorded dataset id still matches, and an adapter that declares
  its own revision keeps it. Agentic slots stay unpinned because their harnesses fetch
  their own data with no revision knob — recorded as a limitation in `docs/benchmarks.md`.
- Why: `openai/mrcr` was corrected in December 2025 and HLE's item count has shifted since
  release, so unpinned cells were comparable to neither Fugu's numbers nor to earlier
  kairyu runs. Revisions are already part of the run fingerprint, so repinning now refuses
  resume instead of reinterpreting stored evidence.
- Refs: `kairyu/bench/pins.py`, `kairyu/bench/adapters/__init__.py`,
  `tests/bench/{test_bench_pins,test_bench_runner}.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] LiveCodeBench and LiveCodeBench Pro datasets are actually reachable
- What: Both code slots could never load their data, so their scoreboard cells were
  permanently blank. LiveCodeBench passed the *config name* `release_v6` as a git
  revision to a repo that has only `main` and no tags, and its script-loader path also
  needs `trust_remote_code` (removed in `datasets` 4.x); it now reads the
  `test.jsonl`…`test6.jsonl` shards directly at a pinned commit and fails closed unless
  `release_v6` yields exactly 1,055 problems. LiveCodeBench Pro asked for a `train`
  split that does not exist and expected a tabular testcase repo; it now pins Fugu's
  2025 Q2 slice (`quater_2025_4_6`, 167 problems), is declared `gated` (it needs
  HF_TOKEN), and joins each `problem_id` to a `<problem_id>.zip` of
  `testdata/<n>.in`/`.ans`. Stdin grading became per-line whitespace-normalized so a
  correct solution emitting trailing spaces or CRLF is no longer a false negative.
- Why: A permanently blank cell is worse than a documented approximation — the suite
  claimed to cover 11 Fugu slots while silently covering 9. The Pro archives ship a
  per-problem testlib `checker.cpp` that kairyu does not compile, so that cell is now
  annotated as a LOWER BOUND rather than presented as the official Accepted rate.
- Refs: `kairyu/bench/hub.py` (`load_jsonl_files`, `revision` on `download_file`),
  `kairyu/bench/adapters/{livecodebench,livecodebench_pro}.py`,
  `tests/bench/test_bench_lcb_datasets.py`, `docs/benchmarks.md`.

### 2026-07-25 — [amendment] SciCode runs sequentially and can reach its golden data
- What: The SciCode slot could not produce a meaningful number. `SciCode1/SciCode` ships
  no reference code at all (every `ground_truth_code` and `general_solution` is null), so
  the adapter's "gold prior-step code" was always the empty string and any sub-step
  calling an earlier step's helper could only raise `NameError`. Separately, 288 of the
  291 test-split sub-steps compare against `target` golden data from `test_data.h5`,
  which the HF export does not contain, so those items were all `unjudged` — leaving
  three scoreable sub-steps. Sub-steps now run SEQUENTIALLY per problem with the model's
  own earlier code carried into both the prompt and the executed program; `--limit`
  selects whole problems; the golden data is fetched from upstream first and otherwise
  from a pinned public mirror, accepted only on HDF5 magic bytes; and prompts now include
  problem-level and step-level background (Fugu's with-background condition). Extracted
  `attempt_item()` in `adapters/base.py` so the sequential loop and the shared generative
  loop classify request failures identically.
- Why: 288 is exactly the denominator Fugu reports, and the sequential setting is
  SciCode's main one — evaluating steps in isolation without any prior implementation
  measures nothing the benchmark is about.
- Refs: `kairyu/bench/adapters/{scicode,base}.py`, `kairyu/bench/fixtures/scicode.jsonl`,
  `tests/bench/test_bench_scicode_sequential.py`, `docs/benchmarks.md`.

### 2026-07-25 — [progress] Live progress display for bench runs
- What: Added `kairyu/bench/progress.py` with three reporters behind one protocol —
  `TqdmProgress` (suite bar + per-pair item bar on a TTY), `LineProgress` (one
  self-contained, throttled line per event for logs), and `NullProgress` — selected by
  `make_reporter()` from TTY-ness and the new `--no-progress` flag. `RunContext.progress`
  defaults to silence, the shared generative loop advances it for every item including
  skips and failures, and the runner announces each benchmark×target (labelling agentic
  slots, which have no item count until their harness returns). `progress` is excluded
  from the run fingerprint. `tqdm` joins the `bench` extra and its absence degrades to
  lines instead of raising.
- Why: A full Fugu run is thousands of judged items over hours; with no output, a working
  run and a hung one looked identical, and a redrawing bar is unusable in
  `docker compose logs`.
- Refs: `kairyu/bench/{progress,runner,types,cli,config}.py`,
  `kairyu/bench/adapters/base.py`, `pyproject.toml`,
  `tests/bench/{test_bench_progress,test_bench_runner}.py`, `docs/benchmarks.md`.

### 2026-07-25 — [progress] Accuracy report compares a run against the published Fugu scores
- What: Added `kairyu/bench/reference.py` (the sakana.ai/fugu-release table and
  per-benchmark figure as committed constants, with source URLs, both asset paths, the
  2026-07-25 retrieval date, the page's own footnotes, and the HLE text-only variant) and
  `kairyu/bench/compare.py`, which renders measured-vs-published with `Δ` against
  published `Fugu`. Every run now writes and prints `comparison.md`/`comparison.json`
  alongside the scoreboard; `kairyu bench report` rebuilds it (`--no-comparison` to skip).
- Why: G6's frontier scoreboard needs the published numbers next to ours, but a bare delta
  invites an apples-to-apples reading the data does not support. The report therefore
  refuses to print 0 for an unmeasured cell, marks partial denominators, withholds the
  delta entirely for the substituted Long Context Reasoning row, and repeats the release
  page's own statement that all non-Fugu scores are provider-reported. The page renders
  its table as a PNG, so the values are transcribed, not fetched — recorded as such.
- Refs: `kairyu/bench/{reference,compare,store,runner,cli}.py`,
  `tests/bench/test_bench_compare.py`, `docs/benchmarks.md`.

### 2026-07-23 — [progress] Versioned structured orchestration trace for evaluation tooling
- What: Added additive, opt-in `kairyu_trace_v2` on unary orchestrated chat responses while
  preserving `kairyu_trace`. Direct, Conductor, and MoA paths emit a common versioned envelope
  with sequenced route/role events, status/attempt, timestamps, resolved worker/engine/model,
  backend token usage, budget deltas, and exception class only. The Pydantic response schema
  declares both structured route and trace for capability discovery while unrequested extension
  fields remain absent. MoA preserves proposal/synthesizer resolution even when both fall back
  to one engine. A contract document pins compatibility, timing, privacy, and extension rules.
- Why: The verification app needs to overlay the actual execution path on the configured role
  DAG and diagnose latency/refinement/budget behavior without parsing strings or collecting
  prompts and generated text.
- Refs: `docs/design/observability-trace-contract.md`,
  `kairyu/orchestration/{trace,conductor,orchestrator}.py`,
  `kairyu/entrypoints/server/{protocol,app}.py`.

### 2026-07-23 — [progress] All-GPU Qwen3-32B TP serve and benchmark workflow implemented
- What: Replaced the single-GPU Qwen3-32B example with a Compose workflow that
  reserves every visible NVIDIA GPU, derives `tensor_parallel_size` at startup,
  and rejects GPU counts that cannot evenly shard Qwen3-32B. Added a one-command
  start-and-benchmark script targeting host port 8001; it records timestamped
  JSON and regenerates a Markdown summary of all saved runs.
- Refs: `examples/qwen3-32b-multi-gpu/`.

### 2026-07-23 — [progress] GPU gate-integrity issues #102–#104 fixed and hardware-verified
- What: Pinned the CPU-only synthetic model fixture to torch attention so the
  CUDA-visible default suite is green (1409 passed, 12 skipped, 92% coverage);
  removed the no-op `KAIRYU_DIST_BACKEND=nccl` rerun from the multi-GPU gate
  while retaining the explicit 2-GPU NCCL EP test; and added a real §0 recorder
  that writes the dated `EnvRecord` with driver, CUDA, library versions, GPU
  topology, MIG/vBIOS inventory, and explicit null/unmeasured P2P fields. The
  explicit NCCL test and real SM120 environment-record generation both pass.
- Why: Gate 0 must be host-independent, and hardware gates must produce the
  evidence they claim rather than returning a false green result.
- Refs: PR #105; Issues #102, #103, #104; `scripts/gpu_gates/{00_env,06_multigpu,record_env.py}`;
  `tests/unit/{test_model_loader_backend,test_gpu_gate_regressions}.py`.

### 2026-07-23 — [progress] First RTX PRO 6000 GPU validation exposed three gate-integrity defects
- What: Provisioned the locked dev/GPU/engine environment on an 8× RTX PRO 6000
  Blackwell Server Edition host (driver 595.71.05, CUDA 13.2). Ruff passed; the
  CUDA-hidden full suite passed with 1407 tests, 12 skips, and 92% coverage; the
  single-GPU marked suite passed 10 tests; the explicit 2-GPU NCCL EP test
  passed; and the 8-test distributed suite passed on gloo. With CUDA visible,
  two ordinary model-loader/backend tests failed because the SM120 profile
  auto-selected FlashInfer for an unsupported tiny fixture. Audit also found
  that the §6 script’s `KAIRYU_DIST_BACKEND=nccl` rerun remains on gloo and the
  §0 script never writes its required environment JSON.
- Why: These failures make Gate 0 host-dependent and allow the environment and
  multi-GPU gate scripts to report success without producing the evidence they
  claim.
- Refs: Issues #102, #103, #104; `tests/unit/test_model_loader_backend.py`;
  `kairyu/engine/kairyu_backend.py`; `scripts/gpu_gates/{00_env,06_multigpu}.sh`;
  `tests/{dist,gpu}/`; `bench/results/env-2026-07-23.json`.

### 2026-07-19 — [amendment] Routing verification is non-mutating and deployment-visible
- What: Added built-in Router `preview`/safe descriptor contracts, including RNG-cloned bandit
  preview; exposed rendered-prompt `/v1/route`, authenticated `/routing`, and opt-in structured
  actual decisions. GPU compose now mounts a routing spec consumed by inventory-driven gateway
  orchestration, while the verification BFF/UI separates preview from actual distribution/history.
- Why: A direct `route()` dry-run advanced bandit RNG, raw-query preview diverged from templated
  chat execution, and deployed GPU gateways did not serve any orchestrator even though local mock did.
- Refs: M1 D3/D6, M4 router learning; `kairyu/orchestration/`,
  `kairyu/entrypoints/server/app.py`, `deploy/compose/`.

### 2026-07-16 — [progress] GET /backends introspection endpoint (m13)
- What: Added an open `GET /backends` reporting the resolved attention backend
  (torch/flashinfer), library versions, and the per-engine backend map. New pure
  `select_backend_name(profile)` names the backend without importing flashinfer
  (`select_backend` delegates to it). In the gateway+replica topology the gateway
  runs no local attention, so for `ReplicaPool` engines it aggregates one replica's
  `/backends` (L1 `OpenAICompatBackend.fetch_backends` → L2 `ReplicaPool.probe_backends`,
  cached + best-effort) and surfaces the replica's kernel under `via_replica`; `role`
  distinguishes gateway vs engine-host.
- Why: tooling (the verify web app) must show "what am I running on" without
  deep-walking private engine internals. The aggregation was needed because the
  proxy gateway — the process callers actually reach — otherwise only ever reports
  its own CPU/torch, hiding the replicas' flashinfer.
- Refs: engine/core/attention/selector.py, entrypoints/server/health.py,
  entrypoints/server/middleware.py (`_OPEN_PATHS`), engine/openai_backend.py,
  orchestration/replica.py, tests/server/test_backends.py; PR #100.

### 2026-07-16 — [progress] FlashInfer AOT covers head_dim 128; multi-model serving is inventory-driven
- What: Extended the FlashInfer AOT jit-cache build in `Dockerfile.cuda` from head_dim 64 only to
  `fa2_head_dim=[(64,64),(128,128)]`, so head_dim-128 models (Llama-3.x) run on the SM120 FlashInfer
  kernels alongside head_dim-64 (Qwen2.5). Previously a 128 model hit a request-time JIT that fails on the
  slim `-runtime` image (no nvcc; `ninja` exit 127). Kept the stock `gpu-replica.yaml`/`gateway-gpu.yaml`
  single-model and refreshed `gpu-replica.multimodel.example.yaml`; the deployment (kairyu-iac) renders the
  multi-model compose from its `kairyu_models` inventory, so no deployment-specific model set is hardcoded here.
- Why: Serve several eval models (Qwen2.5-0.5B + Llama-3.2-3B + Llama-3.1-8B) on one RTX PRO 6000 (SM120)
  replica on FlashInfer, without baking a model list into the OSS compose.
- Refs: PR #99; `Dockerfile.cuda`, `deploy/compose/gpu-replica.multimodel.example.yaml`; kairyu-iac develop
  (compose templating from `kairyu_models`). Adding a model with a new head_dim needs the `fa2_head_dim`
  scope extended + an image rebuild (the one dependency that stays in this repo).

### 2026-07-14 — [amendment] Usage ledger shutdown and recovery are app-owned (m11 D3/A7)
- What: `create_app` now wraps its optional caller lifespan and flushes/closes the
  app-created usage ledger after all inner worker/backend cleanup, even when that cleanup
  raises. Ledger scans validate records independently, ignore whitespace, warn on a
  non-newline malformed tail, error with line numbers for complete malformed records,
  retain valid totals, and expose the latest malformed-record count.
- Why: Issue #90 showed that the persistent O_APPEND handle was never closed and that one
  truncated or schema-skewed JSONL line made the entire administrator usage endpoint fail.
- Refs: Issue #90; m11 D3/A7; `kairyu/entrypoints/server/{app,tenancy}.py`;
  `kairyu/deploy/builder.py`; `tests/server/{test_m11_product,test_serve_builder}.py`.

### 2026-07-14 — [amendment] Embeddings use explicit served model IDs (m11 D4)
- What: `create_app` and `DeploymentSpec` now accept model-ID-to-embedding-backend
  registries, reject IDs that collide with engines, pools, or orchestrators, and include
  configured embedding IDs in `/v1/models`. The embeddings route resolves before validation
  or execution, shares the 404 `model_not_found` response, routes multiple backends, and
  records response, metric, and ledger identity only from the resolved key while charging
  the limiter only after resolution.
- Why: Issue #89 showed that the anonymous global backend accepted and echoed arbitrary IDs,
  omitted its model from discovery, and admitted attacker-controlled metric and metering
  identities despite executing the same backend.
- Refs: Issue #89; m11 D4/A12; `kairyu/entrypoints/server/{app,extra_routes}.py`;
  `kairyu/deploy/{spec,builder}.py`; `tests/server/test_embeddings_models.py`.

### 2026-07-14 — [amendment] Required tool choice is enforced per response choice (m1 D6)
- What: required and named tool choice now succeeds only when every returned choice retains
  a permitted tool call after per-choice filtering. Mixed and empty results use the existing
  controlled 502 before buffered SSE emission, without regeneration, and their consumed
  generation is recorded exactly once.
- Why: Issue #88 showed that the existential satisfaction check accepted multi-choice
  responses when only one choice complied, violating the request contract for the remaining
  choices.
- Refs: Issue #88; m1 D6; `kairyu/entrypoints/server/chat_service.py`;
  `tests/server/{test_openai_api,test_m11_product}.py`.

### 2026-07-14 — [amendment] OpenAI-compatible streams preserve empty choices (m1 D1)
- What: streamed choice state is now initialized whenever an upstream index is observed,
  independently of text content, and partial/final completions are built from the union of
  text and finish indexes with empty defaults. Empty single choices and mixed `n > 1`
  results retain their indexes and finish reasons; streams with no choices still fail.
- Why: Issue #87 showed that truthy-content-only tracking converted valid immediate-EOS or
  content-filtered empty responses into upstream errors and silently dropped empty siblings
  from multi-choice results, contradicting non-streaming behavior.
- Refs: Issue #87; m1 D1; `kairyu/engine/openai_backend.py`;
  `tests/unit/test_openai_backend.py`.

### 2026-07-14 — [amendment] Batch and HTTP share the chat request boundary (m7 D7)
- What: batch JSONL rows now require a frozen envelope with non-blank, per-job-unique
  `custom_id`, `POST`, the owning job endpoint, and an object body. A new transport-neutral
  chat service owns tool, stream/logprob, response-format, image, model, `supports_n`,
  sampling, and backend preflight validation plus buffered dispatch and tool-choice
  satisfaction for both regular HTTP and batch. Controlled failures retain the same
  message/type/code without dispatch; unexpected backend errors expose only the exception
  class in both transports while their full tracebacks remain server-side.
- Why: Issue #86 showed that batch ignored its method and URL, accepted missing or duplicate
  IDs, skipped the public request checks, executed invalid work, and persisted arbitrary
  backend exception strings containing internal topology or secrets.
- Refs: Issue #86; m7 D7; `kairyu/batch/{envelope,worker}.py`;
  `kairyu/entrypoints/server/{chat_service,errors,app}.py`;
  `tests/{unit/test_batch_envelope,server/test_chat_parity,server/test_batches}.py`.

### 2026-07-14 — [amendment] Batch uploads use bounded storage transactions (m10a D3/A8)
- What: `BatchStoreProtocol` expands from ten to eleven methods with
  `save_file_streaming`. `/v1/files` now supplies fixed 1 MiB chunks, enforces the existing
  512 MiB limit incrementally before writing an over-limit chunk, and publishes content plus
  owner metadata only after the stream completes. Rejection, iterator failure, cancellation,
  close failure, and metadata failure remove partial artifacts.
- Why: Issue #85 showed that the route read up to the complete 512 MiB allowance into memory
  per request on top of Starlette's multipart spool, so concurrent accepted uploads could
  exhaust the gateway even though each individual request obeyed the size cap.
- Refs: Issue #85; m10a D3/A8; `kairyu/batch/store.py`;
  `kairyu/entrypoints/server/batch_routes.py`; `tests/unit/test_batch_store_streaming.py`;
  `tests/server/test_batches.py`.

### 2026-07-14 — [amendment] Async abort owns active request lifecycles
- What: `AsyncLLMEngine` replaced persistent abort markers with a registry of active
  request-local events. Generation now races backend progress against abort, rejects only
  concurrently active duplicate IDs, and centralizes deregistration, pending-task
  cancellation, and backend iterator closure across completion, abort, and consumer close.
- Why: unknown aborts previously retained attacker-controlled IDs and suppressed a future
  request reusing that ID, while an active stream blocked on its backend could not observe
  abort until another partial arrived.
- Refs: Issue #84; `kairyu/entrypoints/async_engine.py`;
  `tests/{compat/test_async_engine_compat,unit/test_async_engine_abort}.py`.

### 2026-07-14 — [progress] FlashInfer SM120 (Blackwell) attention enabled: GPU device placement + AOT image
- What: The single-process GPU serve path had no device placement — `build_engine_loop` loaded the
  model and `PagedKVPool` in fp32 on CPU, so a "GPU" deployment ran inference on CPU via the
  device-agnostic torch backend, and FlashInfer (the sm_120 default) failed with
  `KeyError: torch.float32` (its prefill `plan()` has no fp32 kernel). Fixes:
  - `build_engine_loop`: probe-driven device/dtype (cuda → bf16 on-device, cpu → fp32 on host;
    guarded so every CPU path is byte-identical). `load_model(dtype=…)` + `model.to(device)` +
    `PagedKVPool.for_cache(dtype, device)`.
  - `PagedModelRunner` / `PagedKVPool`: device-correct input and index tensors.
  - `Sampler`: sample on CPU (seeded RNG + penalties/enforcer) so the m8-D2 determinism pins hold
    while logits arrive on cuda.
  - `FlashInferBackend.attend`: pass the `[1,H,D]` decode query (0.6.x `decode.run` rejects a 2D
    `[H,D]` slice); the batched `attend_batched` path already passes 3D.
  - `app.py`: backend-exception handlers log the full traceback (client still gets only the class name).
  - `Dockerfile.cuda` → multi-stage AOT: a CUDA 13.0 `-devel` build stage compiles
    `flashinfer-jit-cache` (`FLASHINFER_CUDA_ARCH_LIST=12.0f`, bf16/head_dim-64/FA2); slim
    `13.0.1-runtime`, no runtime nvcc, no first-request JIT.
- Why: run the FlashInfer paged-attention kernels on RTX PRO 6000 Blackwell (sm_120). The torch
  backend was a correct-but-slow interim and — as found — CPU-bound. AOT removes the ~20s
  first-request JIT so a cold replica is not ejected by the gateway readiness probe.
- Refs: docs/design/flashinfer-sm120-aot.md; kairyu/engine/kairyu_backend.py,
  engine/core/{model_runner,kv_pool,sampler}.py, engine/core/attention/flashinfer_gpu.py,
  entrypoints/server/app.py, Dockerfile.cuda.

### 2026-07-14 — [progress] Responses, embeddings, and batches complete usage accounting
- What: successful `/v1/responses` and `/v1/embeddings` calls now record one usage
  event for the authenticated tenant using the same counts returned on the wire.
  The bounded batch worker receives optional ledger/limiter sinks from the deployment
  builder and records each successful backend line for its persisted owner immediately
  after execution, independently of later spool, finalization, or cancellation outcomes.
  Backend usage wins when present; Responses and batch fall back to the shared derived
  approximation, while embeddings charge input tokens with zero output tokens.
- Why: these three surfaces bypassed the shared tenant recorder, leaving successful work
  unaccounted; delaying batch accounting until file publication would also lose already
  executed calls whenever transactional output was rolled back.
- Refs: Issue #45 Task 3; `kairyu/{batch/worker,deploy/builder}.py`;
  `kairyu/entrypoints/server/{extra_routes,tenancy}.py`;
  `tests/server/{test_batches,test_m11_product,test_serve_builder}.py`.

### 2026-07-14 — [progress] Stream metering survives completion, failure, and disconnect
- What: engine chat, orchestrated chat, and legacy completions streams now share one
  idempotent finalization owner. Each dispatched stream records exactly one tenant usage
  event on normal completion, missing backend usage, partial client disconnect, or a
  partial upstream error; backend counts win when present and otherwise the existing wire
  approximation is derived from the rendered prompt and latest cumulative completions.
- Why: accounting after the normal stream loop skipped orchestrated/completions streams
  entirely and bypassed billing whenever a client disconnected or an upstream failed
  after doing partial backend work.
- Refs: Issue #45 Task 2; `kairyu/entrypoints/server/{app,metering}.py`;
  `tests/server/{test_openai_api,test_m11_product}.py`.

### 2026-07-13 — [amendment] Batch execution is bounded and failure-terminal (m7 D7)
- What: batch execution now uses one streaming producer, a bounded input queue, and a
  fixed consumer pool instead of whole-file materialization and one task per line.
  Output/error rows spool incrementally; unexpected line, append, or finalization
  failures roll back every partial file and persist a controlled `failed` state, while
  explicit cancellation wins and task cancellation still propagates.
- Why: Issue #44 demonstrated that valid large uploads could multiply into unbounded
  memory/tasks and that post-admission storage errors could leave jobs `in_progress` or
  expose half-published result files.
- Refs: Issue #44 Tasks 2–3; `kairyu/batch/worker.py`; `kairyu/batch/store.py`;
  `tests/server/test_batches.py`.

### 2026-07-13 — [amendment] Batch storage adds streaming transaction seams (m10a D3/A8)
- What: `BatchStoreProtocol` expands from eight to ten methods with owner-scoped
  `iter_file_lines` and `create_jsonl_writer`. The store now supports lazy binary-line
  input and a lazy JSONL transaction that writes one flushed line at a time, publishes
  owner-scoped metadata only on commit, and removes partial data on abort.
- Why: the existing batch worker materializes the full accepted upload and all output
  rows, so Issue #44 needs bounded storage primitives before Task 2 can replace that
  worker path without exposing partial result files or weakening tenant isolation.
- Refs: Issue #44 Task 1; m10a D3/A8; `kairyu/batch/store.py`;
  `tests/unit/test_batch_store_tenancy.py`.

### 2026-07-13 — [amendment] Deployment auth shares one preflight key snapshot
- What: the deployment builder now resolves both data-plane and administrator key
  sets before constructing owned backends, then passes those immutable snapshots
  through `create_app`; tenant validation and authentication therefore consume the
  same data-plane snapshot. Direct programmatic `create_app` calls retain their
  existing settings-based resolution behavior.
- Why: the initial Issue #46 Task 3 integration re-read data-plane keys during app
  construction and deferred administrator-key resolution until after backend
  ownership, allowing environment changes to desynchronize tenant mapping from
  authentication or to fail after resources had been created.
- Refs: Issue #46 Task 3 review; `af6e2fa`; `kairyu/deploy/builder.py`;
  `kairyu/entrypoints/server/app.py`; `tests/server/test_serve_builder.py`.

### 2026-07-13 — [progress] Deployment YAML tenants wired into runtime isolation
- What: `build_app_from_spec` now preflights the optional deployment `tenants:`
  section before constructing owned backends, converts its limit profiles into
  runtime `TenantLimits`, and passes one validated `TenantConfig` into the server.
  Two-key end-to-end coverage pins independent request buckets, role-auth scoped
  `/admin/usage`, and tenant-named ledger records while tenant-less deployments
  retain their legacy app state.
- Why: The typed schema and mapping validation from Issue #46 Tasks 1–2 were not
  yet connected to `kairyu serve`, so deployment files could describe tenants
  without activating runtime isolation or per-tenant accounting.
- Refs: Issue #46 Task 3; `kairyu/deploy/builder.py`;
  `tests/server/test_serve_builder.py`.

### 2026-07-13 — [design] D4 defines atomic orchestration budget admission
- What: D4 now requires every Conductor generation to reserve its step synchronously
  before dispatch, gives result-priced work one exclusive unknown-cost admission slot,
  and releases complete reservations on failure or cancellation. MoA reserves all
  proposal plus synthesis steps as one operation. An admitted generation's eventual
  actual cost remains fully accounted and queryable even when it crosses the cap.
- Why: Parallel DAG roots and MoA previously admitted work from stale completed-only
  state, multiplying strict step/cost limits. Pre-dispatch reservations close that race
  while preserving the truthful result-priced behavior: an unknowable actual charge is
  never clamped or hidden after admission.
- Refs: issue #43; D4 in `docs/design/m1-orchestration-and-interface.md`;
  `kairyu/orchestration/{budget,conductor,orchestrator}.py`

### 2026-07-13 — [amendment] Remote readiness requires generation-safe probes
- What: Remote replicas with declared readiness URLs now start unknown and stay out of placement and `/readyz` until a successful `/readyz` probe. The serve prober runs immediately at startup, validates unknown/ejected replicas with bounded concurrency, isolates failures, and binds results to the entry generation; URL-less local/programmatic pools remain trusted.
- Why: Treating a newly constructed remote backend as healthy allowed readiness and request placement before any live endpoint check, while serial or ID-only recovery could multiply startup latency or validate a replacement from a stale response.
- Refs: issue #52; `docs/design/m7-productionization.md` D4; `docs/design/m10-fleet-cpu.md` D1/D2/A15; `kairyu/orchestration/replica.py`, `kairyu/deploy/prober.py`, `kairyu/deploy/builder.py`

### 2026-07-13 — [amendment] Elastic ownership follows same-ID entry generations
- What: `ReplicaPool` now gives every backend entry an opaque generation token, and `PoolReconciler` binds applied identities and drain leases to that generation. An external remove/re-add of the same ID discards old tracking; desired absence acquires a fresh lease on the new entry, while desired presence baselines it without factory, replacement, or shutdown side effects. Fresh-entry manual drains remain authoritative.
- Why: Comparing only replica ID sets missed a complete entry replacement between reconciliation ticks, so an old lease could suppress draining of the fresh entry and prevent removal from converging.
- Refs: issue #41; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Manual drains follow same-ID backend replacement
- What: A successful identity replacement now carries the manual drain owner from the old pool entry to the new backend entry while discarding the reconciler lease. The pool exposes a manual-only drain query; the replacement backend starts with fresh health and outstanding state but remains non-eligible until manual undrain.
- Why: Manual ownership was stored on the backend entry, so deleting the old entry and adding the replacement silently made an operator-drained logical replica eligible.
- Refs: issue #41; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Elastic drains preserve overlapping owners
- What: Corrected the preceding reversible-drain amendment by splitting each pool entry's manual drain owner from opaque drain leases. Reconciliation now records and releases only its own lease, so a manual drain asserted before or after reconciliation remains active, and manual undrain cannot cancel reconciliation work.
- Why: ID-only ownership over one boolean could not represent overlapping owners and allowed a desired-state revert or retry factory failure to undrain a replica that an operator had drained afterward.
- Refs: issues #41 and #42; `docs/design/m10-fleet-cpu.md` D1/D2 and A14; `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`

### 2026-07-13 — [amendment] Elastic reconciliation owns reversible drains
- What: Discovery and applied state now carry complete typed replica identities; same-ID changes replace backends construct-before-drain with async ownership cleanup. The reconciler separately tracks drains it initiated and cancels them when replacement/removal intent reverts or a retry factory fails, without overriding manual drains.
- Why: Address/model/auth changes were previously invisible, while an in-flight replacement or removal could leave the old replica permanently non-eligible after desired state returned to the applied identity.
- Refs: issues #41 and #42; `docs/design/m10-fleet-cpu.md` D1/D2, A6, A14; `kairyu/deploy/registry.py`, `kairyu/orchestration/replica.py`

### 2026-07-13 — [amendment] Open WebUI Compose demo + Kairyu-only CI smoke (m11 D7)
- What: The checked-in WebUI topology now mounts a standalone valid
  `deploy/compose/config.yaml` serving keyless mock model `default`; all literal
  Compose binds and mounted DeploymentSpecs are validated before startup. The new
  `scripts/webui_smoke.sh` also pins the rendered internal WebUI endpoint, starts only
  Kairyu, and gates bounded readiness, exact `/v1/models`, and one non-streaming
  completion after the existing default Compose drill.
- Why: m11 D7 previously claimed only that the container config rendered while the
  checked-in bind target did not exist, so a clean checkout could not start the demo.
  Keeping the smoke Kairyu-only proves the broken startup and API contract without
  pulling or browser-testing the large mutable third-party Open WebUI image.
- Refs: m11 D7; `deploy/compose/{docker-compose.webui.yaml,config.yaml}`;
  `scripts/{validate_compose_binds.py,webui_smoke.sh}`; `.github/workflows/ci.yml`;
  `tests/unit/test_compose_configs.py`.

### 2026-07-13 — [amendment] Preflight the production benchmark model
- What: Amended m19 D3 so gate 09 checks `/v1/models` after `readyz` and before
  `serving_bench.py`, requires the requested model ID by exact equality, and uses
  the same `KAIRYU_BENCH_MODEL` value for both steps. Added safe failure handling
  for absent IDs, malformed responses, and non-2xx responses, plus source and
  default/override dry-run pins for ordering and propagation.
- Why: A healthy gateway can pass `readyz` while not serving the model selected
  for the production benchmark, which otherwise makes the benchmark fail late or
  exercise the wrong deployment contract.
- Refs: m19 D3; `scripts/gpu_gates/{09_production.sh,check_served_model.py}`;
  `tests/unit/test_gpu_gates_scripts.py`.

### 2026-07-13 — [amendment] Blackwell Helm profile pins the supported attention backend
- What: Added a strict Helm `attentionBackend` seam that renders
  `KAIRYU_ATTENTION_BACKEND`; CPU defaults omit it, the checked-in
  `pcie-gddr`/SM120 overlay pins `torch`, and operators can select `flashinfer`
  on supported hardware. Extended static and render contracts plus chart docs.
- Why: The automatic SM120 `fa2` tier selects FlashInfer, but the current build
  has no Blackwell kernels. Without an environment seam, the documented overlay
  could render successfully yet fail when starting the real backend.
- Refs: Issue #49 final independent review; `deploy/helm/kairyu/{values.yaml,
  values-gpu.yaml,values.schema.json,templates/deployment.yaml,README.md}`;
  `docs/design/m19-deploy-packaging.md` D2 clarification.

### 2026-07-13 — [amendment] GPU Helm overlay becomes a mandatory CI render gate
- What: `scripts/kind_smoke.sh` now runs fail-fast default/GPU `helm lint` and
  `helm template` gates before cluster creation, with a `--helm-check` mode used
  by an explicit CI schema/GPU-template step. The script remains the single
  command source; CI does not duplicate the four Helm invocations. Appended an
  M19 D2/D3 amendment recording the placement/runtime/storage/real-backend gate
  and its template-only, no-GPU execution boundary.
- Why: The GPU overlay was statically covered but not a mandatory CI input, so
  schema or rendering regressions could merge while the CPU kind smoke remained
  green. Fail-fast rendering makes both chart profiles release-gating without
  pretending ordinary CI can run a GPU workload.
- Refs: Issue #49 Task 3; `scripts/kind_smoke.sh`, `.github/workflows/ci.yml`,
  `tests/unit/test_fleet_elastic.py`, `docs/design/m19-deploy-packaging.md` D2/D3
  amendment.

### 2026-07-13 — [progress] Helm GPU overlay wires real model storage and engine
- What: Added strict chart values schema and a conditional, read-only model volume
  backed by exactly one absolute host path or existing PVC. The checked-in GPU
  values request one NVIDIA GPU, preserve runtime/node placement, mount `/models`,
  and replace the mock DeploymentSpec with backend `kairyu` at
  `/models/checkpoint`; CPU defaults keep model storage disabled. Added semantic
  render/schema regressions and operator documentation for hostPath/PVC use.
- Refs: Issue #49 Task 2; `deploy/helm/kairyu/values.yaml`,
  `values-gpu.yaml`, `values.schema.json`, `README.md`, `templates/deployment.yaml`,
  `tests/unit/test_fleet_elastic.py`. Helm-backed render/lint execution remains
  pending on a Helm-enabled host; local pure/static gates pass.

### 2026-07-13 — [progress] Backend ownership closes across replica and app lifecycles
- What: Replica removal is now an async ownership boundary that closes the removed backend exactly once. Shared shutdown aggregation attempts every unique backend, and orchestrator/application lifespan teardown cascades through separately owned workers even when another shutdown fails.
- Why: Removed/replaced replicas and DSL-built orchestrators leaked clients and worker tasks; one shutdown exception also skipped every later resource.
- Refs: issue #42; `kairyu/engine/backend.py`, `kairyu/orchestration/{replica,orchestrator}.py`, `kairyu/deploy/{registry,builder}.py`

### 2026-07-09 — [progress] Single-node GPU compose: dedicated gateway config + attention-backend env
- What: `docker-compose.gpu.yaml` now mounts a new `deploy/compose/gateway-gpu.yaml`
  (single `replica` upstream, forwards `model: default`) instead of the shared
  `gateway.yaml`, and passes `KAIRYU_ATTENTION_BACKEND` through to the replica
  (empty → auto-select; set `torch` to bypass FlashInfer). `gateway.yaml` is left
  in its CPU-smoke form (three `replica-1/2/3` upstreams, `model: llama`).
- Why: `gateway.yaml` was mounted by BOTH `docker-compose.yaml` (CPU smoke: three
  `replica-N` services serving engine id `llama`) and `docker-compose.gpu.yaml`
  (single `replica` service serving engine id `default`). The two topologies have
  different service names and engine ids, so one file cannot serve both — pointing
  it at the single GPU replica breaks the CPU compose and the CI `compose_smoke.sh`
  drill. Splitting into `gateway-gpu.yaml` lets each topology stand alone. The
  attention env exists because FlashInfer has no Blackwell/sm_120 kernels yet, so a
  Blackwell/RTX PRO 6000 replica can pin `torch` (selector: `KAIRYU_ATTENTION_BACKEND`,
  honored on the single-process `model_path` engine path).
- Refs: `deploy/compose/{gateway-gpu.yaml,docker-compose.gpu.yaml,gateway.yaml}`,
  `kairyu/engine/core/attention/selector.py`, `scripts/compose_smoke.sh` (m19 D2).

### 2026-07-09 — [progress] GPU image base bumped to CUDA 12.8.1 / Ubuntu 24.04
- What: `Dockerfile.cuda` base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  → `nvidia/cuda:12.8.1-runtime-ubuntu24.04`. Ubuntu 24.04 ships `python3.12`
  in its default repos, so the existing `apt-get install python3.12` line now
  resolves natively; nothing else in the image changed.
- Why: The GPU execution host runs Ubuntu 24.04, so the deployment image should
  match the host OS. The CUDA 12.8 runtime also adds SM120 (Blackwell /
  RTX PRO 6000) support that the roadmap's PCIe fleet targets and that the 12.4
  runtime lacked.
- Refs: `Dockerfile.cuda`; supersedes the base image recorded in the
  2026-07-03 M19 deploy-packaging entry (m19 D1).

### 2026-07-04 — [design] Review remediation Phase 6: GPU-day seam changes (CPU design + C5 contract test)
- What: Captured the five GPU-day seam changes from the full-repo review in
  `docs/design/gpu-day-seams.md` (C5 CUDA-graph static buffers, C4 batched
  execution, E3 engine-loop unification, TP delta-broadcast + sampling ownership,
  KVTransport region ownership), and landed the **C5 contract test**: a faithful
  `SnapshotGraphBackend` that freezes page_tables/seq_lens at capture (as a real
  CUDA graph does), plus `test_graph_replay_reflects_current_page_tables`
  (`xfail(strict=True)`) that concretely proves `GraphStepExecutor` currently
  rebinds page tables as Python attributes a real graph never sees. The test
  flips to pass when the static-device-buffer fix lands.
- Why: These CPU-pinned abstractions silently break (C5) or make a perf gate
  unreachable (C4) when real kernels/NCCL/FlashInfer replace the CPU references;
  they must be designed + contract-tested on CPU before GPU time, but can only be
  fully validated on hardware. The full implementations are a scheduled GPU-day
  design milestone (land before the runbook perf gates), not a same-session edit.
- Refs: review report; `docs/design/gpu-day-seams.md`,
  `kairyu/engine/core/step_executor.py` (`SnapshotGraphBackend`),
  `tests/unit/test_step_executor.py`.
### 2026-07-05 — [progress] Real multi-process TP wired into `kairyu serve --tp N`
- What: `build_engine_loop(model_path=…, tensor_parallel_size>1)` no longer
  raises "not yet wired" — it spawns a `DistTPLauncher` group (rank 0 in the
  serve process, ranks 1.. as workers running `worker_step_loop`) and drives it
  through `DistTPModelRunner`. The loop carries a `.tp_launcher` handle that
  `KairyuBackend.shutdown()` calls to stop the workers and destroy the group.
  Added `load_generation_defaults` (public eos/stop loader for the sharded path).
- Why: M16's distributed TP was spawn-tested only in `tests/dist` and unreachable
  from the serve entrypoint — so real tensor-parallel models could not be
  deployed. Now `kairyu serve --tp 2` runs end to end.
- Refs: `kairyu/engine/kairyu_backend.py` (`_build_dist_tp_loop`),
  `kairyu/engine/core/worker.py` (`DistTPLauncher`, `_tp_worker_entry`),
  `kairyu/models/loader.py`, test
  `tests/dist/test_distributed.py::test_dist_tp_launcher_serve_path_matches_single_process`.

### 2026-07-04 — [progress] Review remediation Phase 8: packaging + doc accuracy
- What: Fixed the cross-cutting packaging/doc defects from the full-repo review.
  Added an **`[engine]` extra** (torch + xgrammar + tokenizers + safetensors) so
  real models run WITHOUT the dev group, and pointed **`Dockerfile.cuda`** at it
  (`--extra engine` replaces `--extra hf`) so the production GPU image ships
  xgrammar and can serve `response_format: json_schema` (was missing). Fixed the
  misplaced comment above the `otel` extra (it described the fleet transports).
  `build_engine_loop`'s TP>1 error/docstring now state the truth — the
  multi-process `DistTPModelRunner` exists (m16, tests/dist) but is not yet wired
  into the single-process serve path — instead of "arrives in M16". Refreshed
  `docs/gpu-runbook.md` §0/§1: corrected the stale "177 tests" count, the
  `--group gpu`/`uv sync --dev` command errors (now `--extra gpu`/`--group dev`),
  and the "replace KairyuBackend._tokenize / TorchPagedRunner" instructions that
  M8/M12/M13 already delivered, with a note that the seams exist and GPU-day is
  enabling/tuning them.
- Why: The GPU image couldn't serve structured outputs, there was no non-dev
  install path for real models, and the runbook (the artifact GPU day executes
  from) contradicted the codebase.
- Refs: review report; `pyproject.toml`, `uv.lock`, `Dockerfile.cuda`,
  `kairyu/engine/kairyu_backend.py`, `docs/gpu-runbook.md`.
  **Deferred follow-up:** `kairyu validate` cross-artifact command, typed
  `GenerationRequest.prompt` (token-ids/multimodal), `deploy/spec.py`
  ServerSection compose-not-inherit, and the `kairyu/bench/` package boundary.
### 2026-07-04 — [progress] Review remediation Phase 7: host-path performance (safe subset)
- What: Fixed the provably-safe, output-preserving host-path hot spots from the
  full-repo review. **P5**: `prompt_chunks` re-hashed the whole prompt prefix per
  256-char chunk (O(L²) sha256 on the placement path, event-loop-blocking) and
  the pool called `overlap()` twice per replica; replaced with ONE streaming
  sha256 chain (byte-identical keys, proven equivalent over random trials) and a
  single `overlap()` per replica. **P-perf (completions)**: `/v1/completions`
  ran a prompt array serially (`await` per prompt = sum of latencies); now
  `asyncio.gather` runs them concurrently with order restored by index (response
  byte-unchanged).
- Why: Both are event-loop-blocking / latency costs on the request path that
  survive the GPU swap; both are output-identical so they carry no correctness
  risk.
- Refs: review report; `kairyu/orchestration/{prefix_index,replica}.py`,
  `kairyu/entrypoints/server/app.py`; tests `tests/unit/test_kv_routing.py`.
  **Deferred (risk/complexity, need care or their own change):** P1 incremental
  detokenization (correctness-sensitive output path — a subtle detok bug corrupts
  generation, and CPU tests can't cover every tokenizer edge, so not worth a
  perf-only rewrite), P3 (process-split delta wire), P4 (async ledger/router I/O
  — file-handle lifecycle), P6 (eviction leaf heap), P7 (batched spec verify),
  and the MEDIUM-perf items (sampler penalty state, stop-string offset, queue
  coalescing, scheduler deque, KV-event hash chain, page-table cache).
### 2026-07-04 — [progress] Review remediation Phase 5: bench scoring correctness + security
- What: Fixed the scoring-integrity and security defects in the Fugu bench suite.
  **B1**: the MCQ answer-extraction regex matched "answer" + the first letter of
  the following word (so "Answer: B, because the answer depends…" extracted D)
  and the fallback picked lone lowercase articles/pronouns — tightened to a
  bounded letter after the marker and an uppercase-only fallback. **B2**: an
  un-typed `normalize()` error (schema drift KeyError, image/codec, unpickling)
  crashed the whole suite run; it now degrades THAT dataset to `unavailable`
  ("degradation is data, not control flow"). **M6**: the dataset cache is now
  invalidated when the pinned dataset/revision changes, so bumping `hf_revision`
  re-downloads instead of scoring stale rows. **M7**: private-test blobs unpickle
  through a `_RestrictedUnpickler` that blocks class/global loading (was a
  download-time arbitrary-code vector); the judge response fed into the prompt is
  length-capped. **M8**: LCB solutions that start with `from __future__ import`
  no longer become a SyntaxError when the import header is prepended (the future
  import is hoisted). **M10**: the judge verdict regex accepts markdown-emphasized
  labels (`**correct:** yes`) and the judge token budget was raised so a reasoning
  judge is not truncated before its verdict.
- Why: Each silently corrupts the scoreboard (wrong scores, crashed runs, stale
  data) or is a security hole (ACE at download time).
- Refs: review report; `kairyu/bench/{adapters/base,adapters/livecodebench,cache,judge}.py`;
  tests under `tests/bench/`. **Deferred follow-up (design/policy):** B3 (resume
  per-pair config hash), B4 + denominator policy (skipped/unjudged as 0 or n/a,
  show per-target n_scored), LCB per-line/tolerant scoring, sandbox NPROC/session
  hardening, self-judge (judge==target) scoreboard flag, judge prompt delimiters.
### 2026-07-04 — [progress] Review remediation Phase 4: model + quant parity
- What: Fixed the parity-affecting model/quant defects from the full-repo review.
  **M3 (rope)**: unsupported `rope_scaling` kinds (linear/dynamic/longrope) now
  raise instead of silently dropping to None — a silent parity break vs
  hf.generate. **M4 (fp8 load)**: `Fp8Linear` adopts the checkpoint's
  weight_scale shape, so static per-tensor `(1,)` FP8 (and modelopt FP8) load
  instead of a size-mismatch crash. **M1 (nvfp4 oracle)**: the RNE tie table
  applied LUT *values* as indices and dropped two boundaries, corrupting the
  GPU-kernel packing oracle by up to 60%; replaced with the correct even-index
  table for all seven boundaries (and the test that pinned the wrong behavior).
  MEDIUM: DeepSeek MoE config now falls back to HF defaults
  (norm_topk_prob=True, routed_scaling=2.5, first_k_dense=3, n_group/topk_group
  8/4) for trimmed configs, and a missing expert count raises clearly instead of
  `int(None)`; GPTQ/AWQ `group_size=-1` (single whole-input group) normalizes to
  in_features instead of a negative buffer count; `tp_view` fails fast on MoE
  (no dense down_proj to row-parallelize) like it already does for MLA; bare
  `quant_method: "fp8"` rejects block-wise FP8 (weight_block_size) loudly
  instead of mis-routing DeepSeek block-FP8 to the per-channel path.
- Why: Each is a silent wrong-output or load-time failure on real checkpoints;
  all are CPU-validatable and covered by new tests.
- Refs: review report; `kairyu/models/{config,parallel}.py`,
  `kairyu/quant/{linear,nvfp4}.py`, `kairyu/engine/core/quant_config.py`;
  `tests/unit/test_config_and_fp8_load.py`, `test_quant_compute.py`.
  **Deferred (needs GPU + SpecForge reference to validate):** EAGLE-3 midlayer
  RoPE (H1) and KV-cached rollout feedback (H2) — both affect draft ACCEPTANCE
  RATE only, not output correctness (verification is by the target), so no CPU
  test can validate a fix; plus the design items (linear_factory context,
  forward_fused wiring, HF-name-preserving TP/EP wrappers, draft-head quant).
### 2026-07-04 — [progress] Review remediation Phase 3: orchestration + fleet reliability
- What: Fixed the L2 fleet/orchestration HIGH defects from the full-repo review.
  **O1**: request errors were all counted as replica failures — a new
  `UpstreamClientError` (4xx) is raised by the openai backend and excluded from
  `consecutive_failures`, so one misbehaving client can no longer cascade-eject
  the pool. **O2**: the HealthProber was ordinal-keyed against a dynamic
  id-keyed pool (wrong-replica restore / IndexError / silent prober death);
  it is now id-keyed, resolves URLs per id, and `run()` swallows a bad tick.
  **O3**: the prober now probes `/readyz` (readiness) not `/health` (liveness),
  so a drained/wedged node stays ejected — O1+O3 together kill the flap loop.
  **O4**: the Conductor wraps each unit so a transient backend error records a
  trace event and returns best-so-far instead of raising and discarding every
  completed output. MEDIUM: **M2** orchestrator direct calls no longer mint a
  random per-request session_id (which defeated prefix + least-outstanding
  routing); **M4** KvEventIndex stamps freshness only after a valid apply,
  handles vLLM `AllBlocksCleared`, and the ZMQ drain drops malformed frames
  instead of aborting; **M5** `remove_replica` calls `prefix_index.forget_replica`
  so a re-added id can't inherit phantom prefixes; **M7** lifespan shutdown
  isolates a crashed background task and shuts every engine down independently.
- Why: These are DoS / flap-loop / cost-and-routing-correctness defects the
  single-node CPU tests could not exercise.
- Refs: review report; `kairyu/orchestration/{replica,conductor,orchestrator,kv_index}.py`,
  `kairyu/engine/{backend,openai_backend}.py`, `kairyu/deploy/{prober,registry,spec,builder}.py`;
  tests under `tests/unit/`. Deferred follow-up: M1 (verifier non-target deps +
  _SafeDict masking), M3 (MoA path Budget/cost wiring), M8 (run_chat periodic
  keep-alive), and the KvEventIndex↔ReplicaPool integration (design item).
### 2026-07-04 — [progress] Review remediation Phase 2: API security + tenant isolation
- What: Fixed the CRITICAL/HIGH L3-server defects from the full-repo review.
  **C3 (CRITICAL) batch/file tenant isolation**: File/Batch objects gained an
  `owner`; the store scopes every get/read/list/cancel and cross-tenant access
  reads as not-found — a tenant can no longer enumerate or read another's batch
  prompts/outputs (worker output/error files inherit the batch owner). **S1**:
  a non-object JSONL line becomes a per-line error instead of wedging the job
  in_progress forever. **S2**: invalid sampling params (top_p=0, n=0,
  temperature<0) return 400, not a 500/mislabeled-502. **S3**: streaming chat
  and /v1/completions are now metered (were a billing bypass) — usage flows to
  the ledger; orchestrator-stream/responses/embeddings metering still TODO.
  **S4**: `tokens_per_minute` is enforced via a per-tenant token bucket charged
  post-response. **S5**: `/admin/drain` requires an admin key when configured
  (was any data-plane key = one-request DoS) and gains `/admin/undrain`.
  **S6**: streamed `delta.tool_calls[]` carry the required `index` (SDK
  accumulation). **S7**: `/v1/files` upload is size-capped (413) to prevent
  gateway OOM.
- Why: These are cross-tenant disclosure, billing bypass, and DoS holes that
  the single-tenant CPU test suite could not see.
- Refs: review report; `kairyu/batch/{store,worker}.py`,
  `kairyu/entrypoints/server/{batch_routes,app,health,settings,tenancy,protocol}.py`;
  tests under `tests/server/` + `tests/unit/test_batch_store_tenancy.py`.
  Deferred to a Phase 2 follow-up: MEDIUM items (Prometheus label cardinality,
  /v1/responses store bounds + tenant scope, error-body leak scrub, AUTO-model
  param handling, non-ASCII bearer 401, embeddings validation) and full S3
  metering coverage.

### 2026-07-04 — [progress] Repo-wide review remediation Phase 1: engine-core correctness
- What: Fixed the CRITICAL/HIGH engine-core defects found in the 2026-07-04
  full-repo review (report in job scratch). **C1 radix cache poisoning**:
  `commit_and_release` folded the final sampled token's page as computed even
  though the decode loop never writes that token's KV — a page-boundary
  completion poisoned the next multi-turn prefix (silent wrong output ~1/16 of
  requests). Now caps committable length below the unwritten final token
  (`radix_kv.py`). **C2 oversized-prompt permanent death**: a prompt larger
  than the whole KV cache blocked the head of line forever, turning every empty
  schedule into a fatal engine stall that killed all concurrent requests. The
  scheduler now rejects unadmittable prompts at admission (finish_reason
  "length", drained via `drain_rejected`), and all four engine cores
  (EngineCore/OverlapEngineCore/PipelinedEngineCore/EngineLoop) replace the
  fatal stall with `reject_waiting_head`. **E1 ZMQ receiver death**: a dead
  receiver left every subsequent request hanging; `_ensure_started` now respawns
  a fresh child over a crashed one and per-frame errors no longer kill the loop.
  **E2 state leaks**: `Scheduler.forget` + runner `release` reclaim finished
  per-request state (output lists, sampler seeds, grammar enforcers) — wired
  into `EngineLoop`. MEDIUM: engine_service per-message fault isolation,
  `resume_with_kv` honors ignore_eos/min_tokens/stop_token_ids/finish_reason,
  `RemoteKVReceiver.adopt` frees the allocation on failure, `zmq generate()`
  aborts on cancel, NIXL send yields instead of busy-spinning. LOW: PagePool
  rejects duplicate free ids, torch attention builds indices on the query
  device.
- Why: The CPU test suite was single-turn/single-tenant and could not see these
  multi-turn / long-running / crash-path failures; each is output-corrupting,
  a DoS, or an unbounded leak on the deploy-day paths.
- Refs: review report; `kairyu/engine/core/{radix_kv,scheduler,engine_core,
  overlap,pipeline,pd_remote,pages,model_runner,spec_runner,engine_service}.py`,
  `kairyu/engine/{engine_loop,zmq_backend}.py`; tests under `tests/unit/`

### 2026-07-03 — [progress] Fugu benchmark suite: one-command quality scoreboard (G6 P-C1)
- What: 646 → 730+ tests. New `kairyu/bench/` package + `kairyu bench
  run/download/report/list` CLI. All 11 rows of the Fugu release table
  (sakana.ai/fugu-release) implemented as adapters: GPQA Diamond, HLE,
  LiveCodeBench(+Pro community mirror), SciCode, CharXiv Reasoning, MRCRv2,
  LongBench-v2 (annotated substitute for the unpublished "Long Context
  Reasoning"), τ³-Bench Banking / SWE-Bench Pro (mini-swe-agent scaffold) /
  Terminal-Bench 2.1 (Harbor) as official-harness wrappers. One command
  downloads missing datasets (normalized JSONL cache under
  ~/.cache/kairyu/benchmarks, $KAIRYU_BENCH_CACHE), runs every benchmark ×
  every target, and writes bench/results/fugu/<run_id>/ with per-item
  evidence, methodology (dataset revisions, judge model, truncation policy)
  and a footnoted Fugu-layout scoreboard (JSON+MD). Degradation is data:
  docker/gated-dataset/judge/vision/context-length preconditions produce
  skipped/partial cells with reasons — exit 1 only on hard failures; same
  --run-id resumes. Configurable LLM judge endpoint (HLE free-form, CharXiv,
  τ user-simulator; unjudgeable items recorded, never guessed). Execution
  scoring in an rlimit subprocess sandbox (documented as not a security
  boundary). Orchestration measured as plain model names (kairyu-auto,
  kairyu-auto-max) via the new `orchestrators:` DeploymentSpec map. New
  extras: [bench] (datasets/hub/pillow/h5py), [bench-agentic]
  (mini-swe-agent/swebench/harbor; tau3 documented as git install). Offline
  fixtures keep the default CPU suite and --offline-fixtures runs hermetic;
  networked download tests are hf_hub-marked.
- Refs: goal G6 P-C1/P-B4, roadmap §6 evidence rules; `kairyu/bench/`,
  `kairyu/entrypoints/cli.py`, `docs/benchmarks.md`,
  `examples/{deploy_multi_orchestrator,bench_fugu,agent_pool_max}.yaml`,
  `tests/bench/`

### 2026-07-03 — [design] DeploymentSpec gains named `orchestrators:` (m7 D3 / m11 D2 amendment)
- What: `DeploymentSpec.orchestrators: dict[name, OrchestratorSection]` serves any
  number of named orchestrations (e.g. `kairyu-auto` + `kairyu-auto-max`) from one
  YAML; the legacy single `orchestrator:` key stays and is still served as
  `kairyu-auto`. Validators: name collisions with engines/pools rejected at spec
  load; `orchestrator:` + `orchestrators["kairyu-auto"]` double-declaration
  rejected. Builder passes the named map to `create_app(orchestrators=)` — the
  m11 tiered-auto path was already server-side, just not YAML-expressible.
- Why: The Fugu-suite benchmark work (G6 P-C1) needs "orchestration with an
  arbitrary model composition" to be deployable, then benchmarked as just another
  model name on the same endpoint. Previously `kairyu-auto-max` was reachable
  only via the `create_app` kwarg in tests, never from `kairyu serve`.
- Refs: `kairyu/deploy/{spec,builder}.py`,
  `tests/unit/test_deployment_spec.py`, `tests/server/test_serve_builder.py`

### 2026-07-03 — [progress] M19 complete: deploy-ready — the local-complete plan is DONE
- What: 627 → 646 tests. Dockerfile.cuda (nvidia/cuda 12.4 + gpu/hf/fleet
  extras), GPU compose (device reservations, model volume), Helm
  values-gpu.yaml (per-profile nodeSelector: pcie-gddr / nvlink-hbm),
  scripts/gpu_gates/ covering runbook §0/1/2/3/6/7/9 + G4/G5 — every script
  --dry-run capable, with a CPU suite pinning that dry-runs emit command
  plans AND every referenced path exists today. [gpu] extra with
  sys_platform=='linux' markers (macOS uv sync clean — verified).
  **All 13 milestones of the local-complete plan (M8–M19 + M10a/b) are
  implemented.** Remaining work is strictly the hardware list: performance
  gates, kernel selection/tuning, fabric bring-up, and `pytest -m gpu` /
  scripts/gpu_gates execution.
- Refs: `docs/design/m19-deploy-packaging.md`, `Dockerfile.cuda`,
  `scripts/gpu_gates/`, `deploy/helm/kairyu/values-gpu.yaml`

### 2026-07-03 — [progress] M11 complete: Fugu-class product surface + tenancy
- What: 610 → 627 tests. Usage threaded through MoA/Conductor/Orchestrator
  (was dropped at three layers) — the AUTO path now returns REAL summed
  usage (the m9 usage=None fallback removed at that call site only).
  run_chat streaming: direct route streams live token deltas; multi-stage
  routes emit SSE COMMENT keep-alives (data: lines would break the OpenAI
  SDK) then a buffered final; X-Kairyu-Trace: 1 opts into a kairyu_trace
  field. Tiered auto models (orchestrators dict; kairyu-auto-max routes
  multi_agent through MoA — previously dead code). Tenancy v1: auth stores
  the matched key in scope state, TenantLimitMiddleware runs INSIDE auth
  (401 wins; unauthenticated never drains buckets), per-tenant token
  buckets, O_APPEND JSONL usage ledger written from handlers,
  /admin/usage; isolation + exact reconciliation gates. /v1/responses
  (reviewed subset: exact output-item shapes, input/output_tokens usage
  names, instructions, previous_response_id store; stream descoped) and
  /v1/embeddings (base64 = the SDK default) — both OpenAI SDK round-trip
  tested (openai>=1.66). Vision wire format (content-parts flattening
  everywhere incl. batch-worker path; image parts 400 on non-vision
  engines). F5 CPU: priority admission with aging (injectable clock,
  effective priority at sort time, head still blocks on KVCacheFull),
  AdmissionController (gateway-observable TTFT EMA; admit/defer/shed),
  autoscale_decision hysteresis. Open WebUI compose + frontier_compare
  bench harness (scoreboard schema pinned).
- Refs: `docs/design/m11-product.md` (Status: Implemented);
  `kairyu/entrypoints/server/{tenancy,extra_routes,slo}.py`,
  `kairyu/orchestration/{orchestrator,conductor,moa}.py`,
  `bench/frontier_compare.py`, `deploy/compose/docker-compose.webui.yaml`

### 2026-07-03 — [progress] M10b complete: KV-aware routing
- What: 594 → 610 tests. PrefixIndex (text-chunk approximate trie — the
  gateway has no token ids, review A12; prefix-chained keys, LRU-capped);
  ReplicaPool opt-in prefix scoring (α·overlap − β·outstanding, session
  affinity still first, prefix_match decision reason; disabled default keeps
  m5 behavior byte-identical). RadixKVCache(event_sink=): BlockStored on the
  computed False→True transition + decode-extension nodes (allocate never
  emits; _release double-fire guarded), BlockRemoved ONLY from eviction —
  removed hashes proven identical to stored hashes. KvEventIndex (precise
  per-replica block hashes, staleness > 500 ms → None = fall back to the
  trie) + ZMQ PUB/SUB transport with a chaos gate (publisher killed →
  staleness fallback). Offline (α, β) grid tuner over PlacementRecords.
  Security-review hardening: /admin/drain pinned auth-protected when keys
  are configured (keyless = trusted-mesh mode by explicit m7 D5 choice).
- Refs: `docs/design/m10-fleet-cpu.md` (M10a+M10b Implemented);
  `kairyu/orchestration/{prefix_index,kv_index}.py`,
  `kairyu/engine/core/radix_kv.py`, `kairyu/orchestration/learning/dataset.py`

### 2026-07-03 — [progress] M10a complete: elastic fleet base
- What: 584 → 594 tests. ReplicaPool reworked to id-keyed dynamic membership
  (legacy sequences auto-id "0".."N-1" so HRW mappings AND Prometheus labels
  are unchanged — zero existing-test edits beyond one error message).
  add/drain/remove lifecycle: drain stops NEW placements (HRW runs over
  eligible = healthy ∧ not-draining), remove refuses in-flight unless forced,
  late completion on removed ids is a no-op. HRW remap property gates:
  removal moves ONLY the departed replica's sessions; addition moves ~1/N.
  deploy/registry.py: TTL-heartbeat ReplicaRegistry (injected clock),
  DiscoverySource protocol (static + registry; k8s-endpoints is a deploy-day
  adapter), PoolReconciler (drain-then-remove, tolerates in-flight refusal
  across ticks). POST /admin/drain flips /readyz 503 (node role).
  kairyu/telemetry.py traced_span (L2-safe, no-op without the otel extra) +
  pure-ASGI TracingMiddleware + pool-placement span; opentelemetry-sdk in
  dev group + otel extra. BatchStoreProtocol (full 8-method surface). Helm
  chart (readiness /readyz, config at /etc/kairyu/config.yaml) +
  scripts/kind_smoke.sh + CI kind-smoke job + helm-template render test.
- Refs: `docs/design/m10-fleet-cpu.md` (M10a Implemented);
  `kairyu/orchestration/replica.py`, `kairyu/deploy/registry.py`,
  `kairyu/telemetry.py`, `deploy/helm/kairyu/`

### 2026-07-03 — [progress] M18 complete: real-byte KV transfer + two-process P-D
- What: 571 → 584 tests. kv_serde (PagedKVPool ⇄ PageFrame, layer-major
  fragments, MLA empty-v contract, loud mismatch errors, pool_fingerprint
  handshake). KVHandoff seam widened to carry source page ids (a
  byte-extracting handoff cannot recover the tail page from tokens — the
  freed tail gets reallocated). RemoteKVHandoff/RemoteKVReceiver over the m6
  transport protocol: copy-before-commit ordering, receiver-side dedup skips
  injection of radix-cached pages, sender page ids remapped to
  new_full_pages+(tail). StreamCopyKVHandoff (side-stream copy window;
  synchronize even on failure). NIXL adapter (deferred import;
  registration-once + descriptor math pinned via fake module). FLAGSHIP:
  two REAL processes over TCP — prefill extracts page bytes between
  execute() and update(), decode adopts via resume_with_kv and decodes;
  outputs == single-engine greedy AND per-page sha256 byte parity.
- Refs: `docs/design/m18-kv-transport.md` (Status: Implemented);
  `kairyu/engine/core/{kv_serde,pd_remote,handoff_stream,kv_transport_nixl_gpu}.py`,
  `tests/dist/test_pd_two_process.py`

### 2026-07-03 — [progress] M17 complete: graph-capture seam + EAGLE-3/MTP draft heads
- What: 553 → 571 tests. StepExecutor seam: decode_buckets policy
  (vLLM-style sizes), GraphStepExecutor (capture-once-per-bucket, static
  buffer copy-in, padding to scratch page with outputs dropped, invalidate(),
  oversize→eager) fully pinned against FakeGraphBackend; cuda_graph_gpu.py
  holds the only CUDA lines (side-stream warmup, shared pool). DraftSource
  protocol: n-gram default byte-identical; ModelDraftSource e2e gate — a
  perfect draft through the FULL spec pipeline == plain greedy with >0.9
  acceptance. EAGLE-3 head per corrected review pins (2H midlayer, pre-norm
  residual, fc [H,3H] once per cycle, TRAINED reduced-vocab lm_head + d2t
  offset map, target-aliased embeddings) + SpecForge loader with format-drift
  guards. DeepSeek MTP head (embedding-first eh_proj, separate physical
  head/embed tensors, MoE decoder block at layer_index=num_hidden_layers) +
  extra-layer checkpoint loader. Scope honesty: batched decode capture rides
  FlashInfer's decode wrapper on deploy day (A1); grammar-rollback spec
  stays deferred.
- Refs: `docs/design/m17-graphs-drafts.md` (Status: Implemented);
  `kairyu/engine/core/{step_executor,graph_buckets,draft,cuda_graph_gpu}.py`,
  `kairyu/models/{eagle,mtp}.py`

### 2026-07-03 — [progress] M16 complete: TP/EP/PP run over real multi-process collectives
- What: 547 → 553 tests (incl. 5 gloo spawn gates that run in the default
  suite). TorchDistCommunicator (m5 protocol + tensor extension; NCCL is a
  constructor argument on deploy day). TP: pre-sharded-config scheme
  (tp_view divides heads/kv/intermediate — modules and kv pools come out
  rank-local automatically), get_slice per-rank loading with FULL-config
  bounds, RowParallelLinear (bias once, after the all_reduce), embed/lm_head
  replicated (every rank holds full logits → every rank samples identically,
  m5 D1 kept). TP=2 spawn gate: EngineCore on rank 0 via DistTPModelRunner
  (snapshot broadcast + A11 handshake), worker_step_loop on rank 1 — greedy
  output IDENTICAL to single-process for llama AND qwen2 (bias) tinies.
  EP: EpMoeBlock over uneven all_to_all_single (counts exchange first);
  EP=2 ≡ single-block to 1e-5. PP: PpStageModel stage seam (embed/mid/final,
  rebased per-stage pools) + hidden send/recv; PP=2 greedy ≡ single-process.
  RequestSnapshot finally extended per the m12 mandate (outputs/sampling/
  num_cached_tokens + allocation aliases). Quantized × TP rejected loudly.
- Refs: `docs/design/m16-distributed.md` (Status: Implemented);
  `kairyu/engine/core/{dist_comm,worker,pp_worker,step_input}.py`,
  `kairyu/models/{parallel,moe_parallel}.py`, `tests/dist/`

### 2026-07-03 — [progress] M15 complete: Qwen3-MoE and DeepSeek-V3 with full parity
- What: 530 → 547 tests. Sparse MoE blocks (Qwen3 softmax top-k with fp32
  routing; DeepSeek sigmoid + correction-bias grouped top-k matched exactly —
  bias affects selection only, top-2 group scores, +1e-20 renorm eps,
  routed_scaling on routed only, shared experts, first_k_dense_replace).
  MlaAttention over the latent pool (post-kv_a_layernorm c_kv ‖ roped k_pe as
  ONE kv head, v width 0 — M18 serde contract), q-LoRA and plain-q paths,
  INTERLEAVED rope (DeepSeek default; half-split is wrong), decompress form
  for prefill / absorbed for decode, HF's hardcoded 1e-6 MLA norm eps. yarn
  rope (inv_freq ramp + attention factor + mscale_all_dim² softmax scale).
  Config: dual-alias expert counts, MLA head_dim pinned to qk dims (never
  hidden//heads), kv-pool props (1 head, r+d_rope wide, v=0). Flagship gates:
  logits < 1e-4 AND full-engine greedy == hf.generate for Qwen3-MoE and
  DeepSeek-V3 (q_lora int/None, yarn on/off). Fixture note: random tiny gates
  produce near-tied routing that fp32 noise flips (block itself matches to
  1e-9 on identical inputs) — gates scaled for decisive margins.
- Refs: `docs/design/m15-moe-mla.md` (Status: Implemented);
  `kairyu/models/{moe,mla}.py`, `kairyu/models/{config,layers,llama}.py`,
  `kairyu/engine/core/kv_pool.py`

### 2026-07-03 — [progress] M14 complete: quantized checkpoints load and RUN on CPU
- What: 514 → 530 tests. kairyu/quant/ reference implementations with formats
  verified against AutoAWQ/AutoGPTQ/vLLM/compressed-tensors source and LIVE
  Hub safetensors headers: FP8-E4M3 (clamp-before-cast — torch CPU cast is
  non-saturating), INT8 W8A8 (exact int32 accumulation — the GPU kernels'
  bit-exact oracle), AWQ (out-axis nibble ORDER [0,2,4,6,1,3,5,7], no +1),
  GPTQ (sequential in-axis packing, z-1 storage offset, g_idx always), NVFP4
  (low-nibble-even packing, bit-3 sign, fp8 block scales × fp32 global, RNE
  boundaries). QuantizedLinear modules hold packed buffers under checkpoint
  names; forward_fused is the Triton seam (kairyu/kernels/ stubs, gpu-marked).
  Loader: linear_factory hook live, state_dict-based iteration (non-persistent
  buffers excluded), quantized payloads verbatim + assign=True + lm_head
  re-tie. Guards: AWQ non-gemm, GPTQ v2/non-4bit, compressed-tensors FP4
  (different names + inverted scale) all rejected loudly. Flagship gate: all
  five schemes quantize the tiny llama, write HF-format checkpoints, load,
  and generate through the FULL engine on CPU (8-bit ≥50% greedy agreement;
  4-bit non-degenerate at hidden-64).
- Refs: `docs/design/m14-quant-compute.md` (Status: Implemented);
  `kairyu/quant/{fp8,int8,awq,gptq,nvfp4,linear}.py`, `kairyu/kernels/`,
  `tests/gpu/test_quant_kernels.py`

### 2026-07-03 — [progress] M13 complete: AttentionBackend seam + FlashInfer adapter + MLA reference
- What: attention extracted into a swappable seam (501 → 514 tests, all M12
  parity suites unchanged — the extraction is behavior-free). Backends are
  plain objects (never nn.Module; state_dict safety), ONE instance shared
  across layers (FlashInfer workspace/plan-cache is per-instance).
  FlashInfer adapter written locally with the reviewed API pins (head_dim_qk
  spelling, workspace buffers, explicit q/kv dtypes, int32 host/device index
  arrays, bottom-right causal assertion, per-chunk plan cache) — logic
  CPU-pinned against an injected fake module, kernels mirrored in tests/gpu/
  (7 deselected until deploy day). MLA reference math (decompress ≡ absorbed
  ≡ naive oracle at the pinned (d_nope+d_rope)^-0.5 scale; shared single-head
  k_pe; post-RoPE cache layout) — M15's trusted oracle for the highest-risk
  kernel work. Selector: env override + hw-profile kernel tier;
  build_engine_loop(model_path=) picks the backend from probe() — deploy day
  is config-free.
- Refs: `docs/design/m13-attention-backend.md` (Status: Implemented);
  `kairyu/engine/core/attention/{__init__,torch_backend,mla_torch,flashinfer_gpu,selector}.py`,
  `tests/gpu/test_flashinfer_gpu.py`

### 2026-07-03 — [progress] M12 complete: real dense models with transformers parity
- What: all five m12 phases landed (471 → 501 tests, 95% cov). ModelConfig
  parses both config.json generations; DenseDecoder (HF-exact module tree)
  covers Llama-3.x / Qwen2 / Qwen3 with verified numerics (rotate_half RoPE,
  llama3 scaling, Qwen3 per-head qk-norm, rectangular chunk masks — SDPA
  is_causal measured wrong over cached prefixes); layer-major PagedKVPool;
  PagedModelRunner behind the m8 ModelRunner protocol with the canonical
  state-access contract, KV-write skip below num_cached_tokens, and
  SpeculativeRunner-compatible decode reads. Flagship gates: fp32 logits
  < 1e-4 vs transformers AND full-engine greedy == hf.generate through
  chunked prefill / radix reuse / page-crossing decode / EOS, per arch.
  Loader (tied embeddings mandatory — safetensors omits lm_head; eos LISTS
  from generation_config; quantized checkpoints fail fast until M14);
  KairyuBackend(model_path=) + kairyu-proc model_path (port reported before
  model load). pytest markers gpu/hf_hub/dist (+strict, default-deselected);
  scripts/parity_real_model.py is the opt-in pre-deploy real-model gate.
- Refs: `docs/design/m12-model-zoo.md` (Status: Implemented);
  `kairyu/models/{config,layers,attention,llama,loader}.py`,
  `kairyu/engine/core/{kv_pool,model_runner}.py`, `scripts/parity_real_model.py`

### 2026-07-03 — [progress] M9 complete: the API is truthful (usage, templates, logprobs, n>1)
- What: all five m9 phases landed (437 → 471 tests, 94% cov; goal G6 gates
  P-A1..P-A5 CPU-green). D1 usage truth — GenerationUsage reported by every
  backend (kairyu/proc/mock/openai-passthrough incl. cached_tokens), OpenAI
  include_usage chunk contract exact, batch JSONL outputs truthful. D2 HF Jinja
  chat templates with transformers byte-match parity (trim/lstrip blocks,
  loopcontrols, HF tojson), per-model DeploymentSpec.chat_templates threaded to
  HTTP AND batch identically. D3 logprobs surfaced (TokenLogprob with bytes,
  chunk-choice placement), /v1/completions (legacy four-array logprobs), real
  n>1 via engine sub-request fan-out (seed identity at i=0, cumulative merged
  streams, sibling aborts, prompt counted once). D4 response_format validated
  (400 not crash) + server-level schema-valid-JSON gate with grammar-stop.
  D5 serving_bench: bearer auth, token-granularity TPOT via include_usage with
  labeled chunk fallback, timestamped results JSON.
- Refs: `docs/design/m9-truthful-api.md` (Status: Implemented);
  `kairyu/entrypoints/server/{app,protocol}.py`, `kairyu/entrypoints/chat_template.py`,
  `kairyu/outputs.py`, `kairyu/engine/kairyu_backend.py`, `bench/serving_bench.py`

### 2026-07-03 — [progress] M8 complete: engine CPU core is real (tokens, sampling, spec decode, process split)
- What: all six m8 phases landed (328 → 437 tests, 95% cov). D1 tokenizer seam
  (HF `tokenizers` + incremental detokenizer, SSE-safe stop-string holdback,
  `finish_early` radix-commit path, finish_reason). D2 real sampling
  (SampledToken/StepOutput protocol ripple across every runner and bench;
  grammar-mask-first with xgrammar stop-token termination; raw-logits logprobs;
  sha256+splitmix64 seeding). D3 scheduler multi-token commit (capped spec
  reservation via chunk.num_tokens, capacity degrade-to-1, budget-accurate,
  exact shortfall release via recorded reservation). D4 n-gram
  SpeculativeRunner (overlay-state scoring, spec ≡ greedy pinned with measured
  acceptance > 0, per-request bypass gating). D5 NVFP4/modelopt/INT8 detection,
  HardwareProfile capability matrix + env-record writer, safetensors
  CheckpointReader with get_slice (M16 seam). D6 process split: shared
  `EngineLoop` extracted; ZMQ ROUTER `engine_service` child process (msgpack,
  ephemeral-port pipe handshake) + `kairyu-proc` backend (lazy zmq.asyncio,
  death detection, shutdown escalation, atexit); parity/stop/abort/usage-fields
  pinned across the process boundary. New deps: tokenizers/safetensors ([hf]
  extra), pyzmq/msgpack ([fleet] extra); coverage configured for the spawned
  service.
- Refs: `docs/design/m8-engine-cpu.md` (Status: Implemented);
  `kairyu/engine/{tokenizer,engine_loop,zmq_backend}.py`,
  `kairyu/engine/core/{sampler,sampling_types,spec_runner,hw_profile,weights,engine_service}.py`

### 2026-07-03 — [design] M8 engine-CPU-core designed and reviewed (local-complete program begins)
- What: `docs/design/m8-engine-cpu.md` — real tokenizer/incremental detokenizer
  (toy stays default), real sampling (SampledToken, StepOutput protocol ripple,
  grammar-mask-first, raw-logits logprobs, sha256 seeds), scheduler multi-token
  commit (capped reservation, degrade-not-stall, scheduler-enforced spec
  precondition), n-gram SpeculativeRunner (overlay-state scoring, per-request
  gating), quant/NVFP4 detection + HardwareProfile + safetensors reader, and the
  ZMQ/msgpack API↔engine process split. 3-reviewer panel APPROVE-WITH-AMENDMENTS;
  amendments applied inline (§6): stop-string SSE holdback + `finish_early`
  radix-commit path, step-thread op discipline (fixes a pre-existing add/abort
  race), budget/watermark accounting for spec chunks, loud update() validation.
- Why: The local-complete mandate (implement everything before GPU hardware;
  only measurement/tuning waits) starts with the engine core. Implementation
  milestones M8–M19 continue the m1..m7 numbering and map to roadmap tracks:
  M8/M9→E1-E2/P-A, M10→F1-F2, M11→P-B/P-C/F5, M12–M18→E-track local halves
  (model zoo, attention backends, quant compute, MoE/MLA, gloo/NCCL distributed,
  CUDA-graph/EAGLE seams, KV transport), M19→deploy packaging.
- Refs: `docs/design/m8-engine-cpu.md`; roadmap §4 Track E/P

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
