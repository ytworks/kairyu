# M1 Design: Orchestration Layer (L2) + Interface Layer (L3) on a vLLM Backend

Status: Draft for review (implementation proceeds in parallel per autonomous-goal mode; revisit on review feedback)
Milestone: M1
Date: 2026-07-02

## 1. Goal

Ship a usable Kairyu package before the custom engine (M2) exists:

- `from kairyu import LLM, SamplingParams, Orchestrator` — drop-in signature compatibility
  with vLLM's offline `LLM` API so vLLM examples run with only an import rewrite.
- L2 orchestration natively built in: pluggable **Router** (tier1 / tier2 / multi-agent),
  **Conductor** (Planner/Worker/Verifier/Synthesizer role DAG, async dispatch, budget-bounded
  recursion), and **MoA** (parallel sampling + synthesis).
- OpenAI-compatible HTTP server (`/v1/chat/completions`, SSE streaming, tool calling).
- YAML / decorator DSL for declaring agent pools, role DAGs, and budgets.

## 2. Key design decisions and rationale

### D1. Engine access goes through a small async `EngineBackend` protocol

All of L2/L3 talks to a `EngineBackend` protocol (`generate(request) -> GenerationResult`,
plus a streaming variant). Backends in M1:

| Backend | Purpose |
|---|---|
| `MockBackend` | Deterministic, dependency-free. All unit/CI tests run against it. |
| `VLLMBackend` | Wraps `vllm.AsyncLLMEngine` behind an import guard. Only loads when vLLM is installed (Linux+GPU). |
| `OpenAICompatBackend` | External API workers (OpenAI/Anthropic/Gemini via their OpenAI-compat endpoints) over httpx. Used by the Conductor for frontier-tier roles. |

Rationale: this repo is developed on macOS without CUDA; vLLM cannot even be imported here.
The protocol boundary is also exactly the seam where the M2 custom engine plugs in — M2 is
"add a fourth backend", not a rewrite. This mirrors how SGLang and vLLM both isolate their
schedulers behind an engine-client interface.

### D2. vLLM compatibility is signature-level, verified by contract tests

`kairyu.SamplingParams`, `kairyu.LLM`, `kairyu.RequestOutput`, `kairyu.CompletionOutput`
replicate vLLM's public constructor/attribute surface (the subset exercised by vLLM's
official `examples/offline_inference/basic.py` and the OpenAI server examples).
A contract test suite pins the surface (`tests/compat/`); when vLLM is installed the same
suite additionally cross-checks against the real vLLM classes (skipped otherwise).
We deliberately do NOT subclass or re-export vLLM types: Kairyu must work without vLLM
installed, and M2 replaces the backend entirely.

### D3. Router is a protocol with a rule-based first implementation

`Router.route(query, context) -> RouteDecision` where `RouteDecision.target` is
`tier1 | tier2 | multi_agent` plus a confidence and the extracted features (for logging /
M4 training data). First implementation `RuleRouter` uses pure-Python feature extraction
(length, code-fence presence, math/reasoning keywords, multi-step markers) — no model in
the hot path, so the <10ms latency budget is trivially met and enforced by a test.
The protocol seam is where the M4 learned classifier / contextual bandit slots in.
Every decision is emitted to a `RouterLog` (JSONL) — this is the M4 training corpus.

### D4. Conductor is an explicit role DAG executed with asyncio

Roles (`planner`, `worker`, `verifier`, `synthesizer`, or custom) are nodes of a declared
DAG; edges are data dependencies (a node's prompt template can reference upstream outputs).
Execution is a topological wave schedule with `asyncio.gather` per wave — no threads, no Ray
in M1 (YAGNI; Ray arrives with multi-node). Recursive self-correction is modeled as a
verifier-gated retry loop with depth bounded by `Budget.max_refine_depth`; spend is bounded
by `Budget.max_cost_usd`, charged per generation via a pluggable `CostModel` (default
zero-cost; `chars_cost_model` estimates from prompt+completion volume, and the DSL exposes
`budget.cost_per_1k_chars_usd`). Step admission is strict and happens synchronously before
dispatch: a generation reserves its step before any `await`, and an operation that cannot
reserve its complete step requirement is skipped. Result-priced work also claims one
exclusive unknown-cost admission slot when a cost cap is configured, so parallel waves
cannot all dispatch against the same stale pre-charge balance. Success reconciles the
reservation with actual cost exactly once; failure and cancellation release the complete
reservation before the exception propagates. MoA is one atomic operation and must reserve
all proposal plus synthesis steps (`moa_samples + 1`) and its cost slot before any proposal
dispatches.

The dispatch limits are strict, but an admitted generation's exact cost is unknowable until
its result exists. That one generation may therefore cross `max_cost_usd`; accounting keeps
the full actual cost (never clamps or hides it) and reports the exhausted/overrun state for
querying while refusing later work. A wall-clock deadline bound is deferred to M2, where
the engine can enforce it per-step. Exceeding a budget is a normal, reported outcome (best
result so far is returned), not an exception, matching Fugu's "recursion depth as
inference-time compute axis" framing.

### D5. KV-affinity is designed in now, exploited in M2

The differentiation core (multi-step orchestration hitting shared-prefix KV cache) needs the
M2 Radix-Paged KV manager for full effect. In M1 we (a) keep every Conductor/MoA step's
prompt as `shared_prefix + role_suffix` (structural invariant, tested), and (b) when the
vLLM backend is active, enable `enable_prefix_caching=True` and route all steps of one
orchestration to the same engine so vLLM's block-hash prefix cache already gets hits.
The `GenerationRequest.cache_hint` field (session id + prefix fingerprint) is plumbed
through now so M2 can consume it without interface changes.

### D6. Server is FastAPI + SSE, one process, engine-agnostic

`kairyu.entrypoints.server` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
Tool calling passes `tools`/`tool_choice` through to the backend and parses tool-call
output into OpenAI's `tool_calls` schema. An `x-kairyu-orchestrate` request field (or
model name `kairyu-auto`) routes a request through the Orchestrator instead of a raw engine
— one endpoint, Fugu-style.

### D7. DSL: YAML is the source of truth; decorators build the same objects

YAML loader produces pydantic-validated `OrchestratorSpec` (agent pool, role DAG, budgets).
The `@role` decorator API constructs identical spec objects in Python. One schema, two
front-ends; the Conductor consumes only the spec.

## 3. Out of scope for M1 (deferred with reasons)

- Custom scheduler / KV manager / CUDA graphs / spec decode / quantized load — M2/M3.
- Learned router training pipeline — M4 (M1 emits the logs it will train on).
- Multi-node (Ray), P-D disaggregation — M3+.
- xgrammar structured output — arrives with the custom engine (vLLM backend already
  supports `guided_json` passthrough, exposed but not wrapped).

## 4. Testing strategy

- Unit tests for every module against `MockBackend` (no network, no GPU), pytest-asyncio.
- Contract tests pin the vLLM-compatible API surface; cross-check vs real vLLM when present.
- Server tests via `httpx.ASGITransport` (no socket).
- Router latency test asserts p99 < 10ms over 1000 routes.
- Coverage gate ≥ 80% in CI (GitHub Actions, `uv` + Python 3.11/3.12 matrix).

## 5. Package layout

```
kairyu/
  __init__.py               # LLM, SamplingParams, Orchestrator, RequestOutput, ...
  sampling_params.py        # vLLM-compatible SamplingParams
  outputs.py                # CompletionOutput / RequestOutput
  engine/
    backend.py              # EngineBackend protocol, GenerationRequest/Result, cache_hint
    mock.py                 # MockBackend
    registry.py             # backend factory/registry
    vllm_backend.py         # import-guarded vLLM adapter
    openai_backend.py       # external OpenAI-compatible API worker
  orchestration/
    features.py             # query feature extraction (pure functions)
    router.py               # Router protocol, RuleRouter, RouteDecision, RouterLog
    budget.py               # Budget, BudgetTracker
    conductor.py            # RoleSpec DAG + async executor
    moa.py                  # Mixture-of-Agents
    orchestrator.py         # Orchestrator facade (route -> engine | conductor | moa)
  dsl/
    spec.py                 # pydantic OrchestratorSpec schema
    loader.py               # YAML front-end
    decorators.py           # @role / @agent_pool front-end
  entrypoints/
    llm.py                  # vLLM-compatible LLM class
    server/
      protocol.py           # OpenAI request/response pydantic models
      app.py                # FastAPI app, SSE streaming, tool calls
bench/                      # reproduction scripts (M1: harness skeleton + mock run)
tests/{unit,compat,server}/
docs/design/
```
