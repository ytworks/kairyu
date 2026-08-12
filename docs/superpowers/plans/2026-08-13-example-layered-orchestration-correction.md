# Example Layered Orchestration Correction Plan

> **Status:** In progress. Do not mark the example corrected until the end-to-end
> product-path and native-L1 gates in this plan pass.

**Goal:** Make the tiered Chat UI example exercise the intended product path:
one Chat UI request enters Kairyu at L3, crosses L2 exactly once, invokes L1
workers repeatedly through a bounded propose/synthesize/verify/refine graph,
and returns the committed L3 response to the UI with model-attributed completed
intermediate work in a separate expandable reasoning surface.

**Architecture:** Separate the public product model registry from internal L1
worker pools, inject those pools into L2 by object reference instead of calling
Kairyu's own L3 HTTP endpoint, express the quality policy as an explicit
Conductor DAG with a verifier-gated synthesis loop, and keep direct/candidate
models in an opt-in loopback benchmark profile. The measured vLLM workers may
be used while the structural work lands, but the example is not finally
compliant with the product roadmap until the same path passes on native Kairyu
L1 workers.

**Tech Stack:** Python 3.12, Pydantic, FastAPI, asyncio, Kairyu
`EngineBackend`/`ReplicaPool`/`Conductor`, YAML, Docker Compose, Open WebUI,
pytest, uv, GPU benchmark harnesses.

## 1. Problem statement and target invariant

The current example is not the requested layered product path:

1. `auto*.yaml` constructs `OpenAICompatBackend` workers whose `base_url` is
   `http://kairyu:8000/v1`. The real path is therefore
   `L3 -> L2 -> L3 -> ReplicaPool -> vLLM`, not `L3 -> L2 -> L1`.
2. `/v1/models` publishes Qwen, DeepSeek, all diagnostic MoA candidates, and
   the product orchestrators. Open WebUI can therefore select a direct model
   and bypass the intended L2 product policy.
3. `kairyu-auto-max` uses the `moa_samples` shortcut. It performs one proposal
   fan-out and one synthesis call. Every checked-in tiered policy sets
   `max_refine_depth: 0`, so synthesis/verification does not repeat.
4. The Compose workers are vLLM services. This is valid as an M1 backend seam,
   but it does not close the current roadmap statement that production L1 is
   Kairyu's own engine. Native Qwen3.6/DeepSeek V4 full-checkpoint gates are
   still open and must remain visible.

The corrected product request must have this observable shape:

```text
Open WebUI
  -> L3 chat ingress (one public model; validate messages/tools/sampling)
  -> L2 router + Conductor
       -> L1 planner
       -> L1 proposal A/B/C in parallel
       -> L1 draft synthesis
       -> L1 verifier
       -> [FAIL] revised synthesis -> verifier, bounded to two refinements
  -> L1 final publisher (answer stage)
  -> L3 usage/trace/OpenAI response adapter
  -> Open WebUI
```

Hard invariants:

- A product request crosses the L3 request adapter once and the L3 response
  adapter once. No L2 worker has a URL pointing at the public Kairyu gateway.
- The Chat UI key/model inventory can address only the product orchestrator.
  Internal engines and diagnostic policies are not callable through that
  surface, even when their names are guessed.
- L2 calls the already-built L1 `EngineBackend`/`ReplicaPool` objects directly.
  Shared resources are started and shut down exactly once.
- Intermediate generated output is hidden by default. The example policy opts
  in to model-attributed planner, proposal, synthesis, verifier, and refinement
  output via `reasoning_content`; the publisher answer remains separate
  `content`.
- Trace v2 remains metadata-only. Intermediate visibility never includes API
  keys, backend URLs, system prompts, or raw exception details.
- Refinement is bounded by both `max_refine_depth` and `max_steps`; budget
  exhaustion returns the best available draft with an explicit trace verdict.
- Cumulative orchestration usage covers every L1 call; public completion usage
  continues to describe only the L3 answer contract.
- vLLM-backed structural evidence and native-Kairyu production evidence are
  labeled separately. A vLLM pass cannot close the native-L1 gate.

## 2. Proposed product policy

Use an explicit role DAG for the one UI-visible quality model. Do not use the
single-pass `moa_samples` shortcut for this model.

| Role | Worker | Dependencies | Purpose |
|---|---|---|---|
| `planner` | Tier2 | none | Produce a compact private plan |
| `proposal_a` | Tier1 | `planner` | Independent solution draft |
| `proposal_b` | Tier1 | `planner` | Independent adversarial/edge-case draft |
| `proposal_c` | Tier1 | `planner` | Independent verification-oriented draft |
| `draft_synthesis` | Tier2 | all proposals | Synthesize the current best answer |
| `verifier` | Tier2 | `draft_synthesis` | Return `PASS` or `FAIL` plus actionable defects; verifies `draft_synthesis` |
| `publisher` | Tier2 | `draft_synthesis`, `verifier` | Emit only the final user-facing answer |

Set `max_refine_depth: 2`, `max_steps: 11`, and `moa_samples: 0`:

- planner: 1 step;
- three proposals: 3 steps;
- initial synthesis + verifier: 2 steps;
- at most two synthesis/verifier refinements: 4 steps;
- final publisher: 1 step.

The normal PASS-on-first-check path uses seven calls. The worst allowed path
uses eleven. Only `publisher` receives the caller's public tools, schema,
`n`/logprobs, and output-token allowance. All earlier stages use the bounded
private-stage policy.

## 3. Work plan

### Task 1: Amend the design contract before changing runtime behavior

**Files:**

- Modify: `docs/design/m1-orchestration-and-interface.md`
- Modify: `docs/design/frontier-native-runtime.md`
- Modify: `docs/goals/g6-product-surface.md`
- Modify: `PROGRESS.md` only when the amendment is accepted/implemented, per
  `.claude/rules/progress-log.md`

- [ ] Record that a deployment-owned orchestrator may borrow existing engine
  and pool objects and that borrowed resources are not owned by the
  orchestrator lifecycle.
- [ ] Amend the UI gate: the product profile exposes only orchestration models;
  direct and candidate models belong to a separate loopback diagnostic profile.
  Preserve generic Kairyu's ability to expose direct models outside this
  profile.
- [ ] Supersede the single-pass `auto-max` example decision with the explicit
  verifier-gated DAG above.
- [ ] State the two evidence phases: structural vLLM-backed closure, followed by
  native-Kairyu L1 production closure. Do not rewrite historical measurements.
- [ ] Define opt-in intermediate visibility, model attribution, the
  `reasoning_content`/`content` separation, and metadata-only trace boundary.
- [ ] Add the required English amendment entry and Current Status correction to
  `PROGRESS.md` in the implementation change that lands the accepted design.

**Acceptance:** The documents agree on the layer boundary, public model
visibility, loop bound, intermediate-output contract, and native-L1 completion
criterion.

---

### Task 2: Add deployment-engine references to the L2 DSL

**Files:**

- Modify: `kairyu/dsl/spec.py`
- Modify: `kairyu/dsl/loader.py`
- Modify: `kairyu/orchestration/orchestrator.py`
- Modify: `kairyu/orchestration/conductor.py`
- Modify: `kairyu/deploy/builder.py`
- Modify: `kairyu/deploy/validation.py`
- Modify: `tests/unit/test_dsl.py`
- Modify: `tests/server/test_serve_builder.py`
- Modify: `tests/unit/test_deployment_validation.py`

- [ ] Extend `WorkerSpec` with an explicit `engine_ref` field. A referenced
  worker has `name` plus `engine_ref` and cannot also declare `backend`,
  `base_url`, `api_key_env`, or factory `options`.
- [ ] Change `build_orchestrator` to accept a mapping of deployment-owned
  engines. Resolve `engine_ref` only from `engines:`/`pools:`; reject missing
  names, orchestrator-to-orchestrator references, and accidental recursive
  public-model references before resource startup.
- [ ] Keep standalone DSL behavior backward-compatible: factory-backed workers
  still construct and own their backends when no `engine_ref` is present.
- [ ] Add explicit resource ownership to `Orchestrator`. `startup()` and
  `shutdown()` operate only on factory-created owned workers; borrowed
  deployment engines are lifecycle no-ops from the orchestrator's perspective.
- [ ] Build pools before orchestrators in `deploy/builder.py`, then inject the
  exact pool objects into every named orchestrator. Deduplicate shared aliases
  by object identity.
- [ ] Make static validation resolve referenced names without importing GPU
  backends or allocating resources.
- [ ] Add tests proving object identity, no `OpenAICompatBackend` is created for
  a ref, and a pool shared by two orchestrators receives one startup and one
  shutdown.

**Acceptance:** A YAML worker such as
`{name: tier1, engine_ref: qwen3.6-27b}` dispatches directly to the already-built
`ReplicaPool`; no worker request reaches `/v1/chat/completions` on the same app.

---

### Task 3: Separate public product models from internal runtime models

**Files:**

- Modify: `kairyu/deploy/spec.py`
- Modify: `kairyu/deploy/builder.py`
- Modify: `kairyu/entrypoints/server/app.py`
- Modify: `kairyu/entrypoints/server/extra_routes.py`
- Modify: `kairyu/entrypoints/server/responses_service.py`
- Modify: `tests/server/test_serve_builder.py`
- Modify: `tests/server/test_openai_api.py`
- Modify: `tests/server/test_responses_api.py`

- [ ] Add an optional deployment-level `public_models` allowlist. Omission
  preserves today's all-model behavior; an explicit list is fail-closed and is
  validated against engines, pools, orchestrators, and embeddings.
- [ ] Partition runtime resources from public dispatch maps in the builder.
  Health, readiness, metrics, and shutdown continue to cover internal L1
  pools; `/v1/models`, Chat Completions, Completions, and Responses use only the
  public maps.
- [ ] Return the normal model-not-found response when a caller guesses an
  internal model or diagnostic orchestration name.
- [ ] Ensure `/routing` and trace payloads do not turn backend URLs, keys, or
  private prompts into public data. Operator diagnostics may retain sanitized
  engine identities.
- [ ] Add tests for explicit-one-model inventory, guessed-name rejection on all
  generation APIs, internal readiness participation, and backward-compatible
  behavior when `public_models` is omitted.
- [ ] Add an immutable `expose_intermediate_outputs` orchestration policy flag,
  default false, and expose it in sanitized routing diagnostics.

**Acceptance:** The product gateway can own Qwen/DeepSeek pools and diagnostic
objects while exposing exactly one model to Open WebUI and accepting exactly
that model on public generation routes.

---

### Task 4: Encode and verify the iterative synthesis DAG

**Files:**

- Modify: `examples/qwen3.6-deepseek-v4-8gpu/auto-max.yaml`
- Add or modify: a benchmark-only orchestration policy under
  `examples/qwen3.6-deepseek-v4-8gpu/`
- Modify: `tests/unit/test_tiered_frontier_examplectl.py`
- Modify: `tests/unit/test_conductor.py`
- Modify: `tests/unit/test_orchestrator.py`
- Modify: `tests/server/test_orchestration_usage_trace.py`

- [ ] Replace the product policy's URL-backed workers with `engine_ref` workers.
- [ ] Declare the seven roles from section 2, with three proposal roles sharing
  the Tier1 pool and the synthesis/verifier/publisher roles using Tier2.
- [ ] Set `moa_samples: 0`, `max_refine_depth: 2`, and `max_steps: 11`.
- [ ] Keep the final `publisher` unverified so its stream is irrevocable.
  Verify `draft_synthesis` instead; Conductor must finish its bounded loop
  before the first answer delta.
- [ ] Add scripted tests for PASS on the first synthesis, FAIL-then-PASS, and
  FAIL through the maximum depth. Assert exact call/step counts and that each
  retry includes the previous draft and verifier critique.
- [ ] Assert the final trace records attempts, verifier verdicts,
  `refinement_exhausted`, resolved L1 worker/model identities, and cumulative
  internal token usage.
- [ ] Assert hidden-by-default policies omit intermediate text, while the
  example policy emits every completed attempt in chronological,
  model-attributed `reasoning_content` and never mixes it into final `content`.

**Acceptance:** A failing verifier causes actual re-synthesis and re-verification
at L1; the loop stops on PASS or the declared bounds, then one final publisher
stage supplies the L3 answer.

---

### Task 5: Rebuild the example product and diagnostic surfaces

**Files:**

- Modify: `examples/qwen3.6-deepseek-v4-8gpu/kairyu.yaml`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/compose.yaml`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/control.py`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/benchmark.py`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/example.json`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/README.md`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`
- Modify: `tests/unit/test_tiered_frontier_examplectl.py`

- [ ] Configure the product gateway with
  `public_models: [kairyu-auto-max]`. Keep Open WebUI's sole API base at the
  product gateway and keep `DEFAULT_MODELS: kairyu-auto-max`.
- [ ] Set `expose_intermediate_outputs: true` only on the UI-visible product
  policy. Keep diagnostic policies hidden unless separately requested.
- [ ] Remove direct Qwen/DeepSeek and `auto-max-moa1..4` from the product model
  inventory and readiness assertion. Keep their pools as internal resources.
- [ ] Add an opt-in Compose `benchmark` profile with a second loopback-only
  gateway/config for direct baselines and policy candidates. It may connect to
  the same L1 services, but it must not share the Chat UI network alias or bind
  a non-loopback host port.
- [ ] Point all orchestration workers in both profiles directly at their L1
  pool refs. No example YAML may contain `base_url: http://kairyu:8000/v1`.
- [ ] Make `control.py` validate the product inventory is exactly the declared
  UI-visible set, not merely a superset. Validate the `/routing` DAG, refinement
  bounds, and absence of self-referential worker URLs.
- [ ] Move direct/policy-selection benchmark commands to the benchmark gateway.
  Keep the product-path serving and browser checks against the product gateway.
- [ ] Preserve historical vLLM measurements as historical records; add new
  rows instead of rewriting results produced by the single-pass policy.

**Acceptance:** Starting the default profile gives the browser one model and
one route. Starting the benchmark profile adds only loopback diagnostic access
and does not change the browser inventory.

---

### Task 6: Add end-to-end layer-boundary evidence

**Files:**

- Add: `scripts/tiered_orchestration_browser_smoke.mjs`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/compose.yaml`
- Modify: `.github/workflows/ci.yml` only for CPU/mock browser coverage
- Modify: `tests/unit/test_tiered_frontier_examplectl.py`
- Add or modify: focused server integration tests under `tests/server/`

- [ ] Add a recording `EngineBackend` integration test that tags every boundary
  and proves the sequence `L3-in -> L2 -> L1... -> L2 -> L3-out` with no second
  L3-in event.
- [ ] Test one complete OpenAI unary request and one SSE request. Require exact
  final text, one public response, cumulative internal usage, and a trace whose
  L1 call count matches the scripted verifier path.
- [ ] Add a browser smoke that asserts Open WebUI lists only
  `kairyu-auto-max`, sends one message, renders the publisher answer separately,
  opens the reasoning disclosure, and finds role/attempt/engine/model labels
  plus intermediate text there. Repeat after a product-gateway restart.
- [ ] Add a negative browser/API assertion for a guessed direct model name.
- [ ] Keep CPU CI on mock L1 backends. Run the real Compose/GPU version through
  the GPU runbook rather than pretending CI exercised model inference.

**Acceptance:** Automated evidence demonstrates both topology and behavior;
README arrows alone are not accepted as proof.

---

### Task 7: Move the final example to native Kairyu L1

**Files:**

- Add: native L1 deployment configs under
  `examples/qwen3.6-deepseek-v4-8gpu/`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/compose.yaml`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/control.py`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/example.json`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/README.md`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/MEASUREMENTS.md`
- Modify: `docs/gpu-runbook.md`
- Add or modify: GPU contract tests for the selected Qwen3.6 and DeepSeek V4
  native configs

- [ ] Define fail-closed native Qwen3.6 Tier1 replicas with
  `execution_mode: native`, exact checkpoint/revision attestation, TP1, and
  checkpoint-owned frontier cache settings.
- [ ] Define the native DeepSeek V4 Tier2 topology using the implemented
  Attention-DP/EP path. Select EP4 or EP8 only from same-checkpoint measured
  evidence; do not inherit the vLLM TP4/EP4 label.
- [ ] Disable CUDA Graphs, drafts, FP8 KV, or other features whose native
  frontier gates remain open. Enable each only through its existing separate
  correctness and goodput gate.
- [ ] Run full-checkpoint startup/readiness, exact tokenizer usage, unary and
  streaming parity, structured/tool behavior, 262K Qwen and 1M DeepSeek
  context gates, 30-minute soak, cancellation, OOM/worker-failure recovery,
  and clean teardown.
- [ ] Re-run the iterative product policy quality and serving matrices. Compare
  against the retained vLLM configuration on the same prompts, hardware, and
  checkpoint revisions; do not publish performance if quality is inferior.
- [ ] Switch the default Compose services from vLLM to native Kairyu only after
  every required gate passes. Until then, label the example transitional and
  leave the native profile opt-in.

**Acceptance:** The default product path is
`Open WebUI -> Kairyu L3 -> Kairyu L2 -> Kairyu L1 -> Kairyu L2 -> Kairyu L3`,
with full-checkpoint evidence and no vLLM process in the default profile.

---

### Task 8: Final verification and progress handoff

- [ ] Run focused CPU tests:

  ```bash
  uv run pytest \
    tests/unit/test_dsl.py \
    tests/unit/test_conductor.py \
    tests/unit/test_orchestrator.py \
    tests/unit/test_deployment_validation.py \
    tests/unit/test_tiered_frontier_examplectl.py \
    tests/server/test_serve_builder.py \
    tests/server/test_openai_api.py \
    tests/server/test_responses_api.py \
    tests/server/test_orchestration_usage_trace.py -q
  ```

- [ ] Run `uv run ruff check .` and `git diff --check`.
- [ ] Run Compose config expansion and the CPU/mock browser smoke.
- [ ] Execute the GPU gates from Task 7 and retain dated artifacts under
  `bench/results/examples/qwen3.6-deepseek-v4-8gpu/`.
- [ ] Update `PROGRESS.md` Current Status and prepend one concise English Change
  Log entry. Archive first if the size hook requires it.
- [ ] Verify docs, example manifest, Compose services, `/v1/models`, `/routing`,
  trace evidence, and retained measurements all describe the same topology.

## 4. Completion criteria

The work is complete only when all of the following are true:

1. Open WebUI can call exactly one product model and guessed internal model
   names fail.
2. No L2 worker calls the public L3 gateway or owns a duplicate copy of a
   deployment pool.
3. A scripted verifier failure produces at least one real synthesis retry and
   the trace/usage accounts for it.
4. Completed intermediate work is visible only in the separate expandable,
   model-attributed reasoning surface; only the publisher is final `content`.
5. Direct baselines and candidate policies remain reproducible through an
   opt-in loopback benchmark profile.
6. The default example uses native Kairyu L1 and passes its full-checkpoint GPU
   gates. Before that switch, structural work may be merged but must be labeled
   transitional rather than complete.

## 5. Explicit non-goals

- Do not remove generic direct-model serving from Kairyu; restrict only the
  product example through explicit deployment configuration.
- Do not expose hidden provider state, system prompts, secrets, or raw
  exceptions. The example deliberately exposes completed stage outputs, but
  only through its policy-owned, model-attributed reasoning surface; trace v2
  remains metadata-only.
- Do not replace Conductor with a second orchestration implementation. Extend
  the existing DSL, lifecycle, trace, and budget seams.
- Do not rewrite historical benchmark artifacts or compare new iterative-policy
  results as if they were produced by the old single-pass MoA-3 policy.
