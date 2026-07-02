# Kairyu M1 Implementation Plan — L2 Orchestration + L3 Interface

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `kairyu` v0.1: vLLM-signature-compatible `LLM`/`SamplingParams`, pluggable engine backends (mock / vLLM / external OpenAI-compat), Router+Conductor+MoA orchestration, OpenAI-compatible FastAPI server with SSE + tool calls, YAML/decorator DSL.

**Architecture:** All layers talk to an async `EngineBackend` protocol. Tests run against `MockBackend` (no GPU/network). vLLM adapter is import-guarded. See `docs/design/m1-orchestration-and-interface.md` for rationale (D1–D7).

**Tech Stack:** Python 3.11+, uv, pydantic v2, FastAPI, httpx, PyYAML, pytest + pytest-asyncio + pytest-cov, ruff.

**Conventions:** TDD per task (test → fail → implement → pass → commit). Frozen dataclasses / tuples everywhere (immutability). Conventional commits, no attribution footer.

---

### Task 1: Project scaffold

**Files:** Create `pyproject.toml`, `kairyu/__init__.py`, `tests/unit/test_package.py`, `.python-version` (3.12), update `README.md`.

- [x] Step 1: `pyproject.toml` with `[project] name="kairyu" version="0.1.0" requires-python=">=3.11"`, deps: `pydantic>=2`, `fastapi>=0.115`, `httpx>=0.27`, `pyyaml>=6`, `uvicorn>=0.30`; dev group: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`. pytest config: `asyncio_mode="auto"`, `addopts="--cov=kairyu --cov-report=term-missing"`.
- [x] Step 2: failing test `test_package_exposes_version` (`kairyu.__version__ == "0.1.0"`).
- [x] Step 3: implement `kairyu/__init__.py`; `uv sync && uv run pytest` → PASS.
- [x] Step 4: commit `chore: scaffold kairyu package with uv/pytest/ruff`.

### Task 2: `SamplingParams` (vLLM-compatible)

**Files:** Create `kairyu/sampling_params.py`, `tests/unit/test_sampling_params.py`.

- [x] Tests: default construction (`temperature==1.0, top_p==1.0, max_tokens==16`), vLLM-style kwargs (`n, best_of, presence_penalty, frequency_penalty, repetition_penalty, temperature, top_p, top_k, min_p, seed, stop, stop_token_ids, max_tokens, min_tokens, logprobs, ignore_eos, skip_special_tokens`), validation errors (negative temperature, top_p out of (0,1], n<1, max_tokens<1, stop as str normalized to tuple), `clone(**overrides)` returns new object, original unchanged.
- [x] Implement as frozen dataclass with `__post_init__` validation (raise `ValueError` with field name) and `clone()` via `dataclasses.replace`.
- [x] Commit `feat: add vLLM-compatible SamplingParams`.

### Task 3: Output types

**Files:** Create `kairyu/outputs.py`, `tests/unit/test_outputs.py`.

- [x] `CompletionOutput(index, text, token_ids: tuple, cumulative_logprob, logprobs=None, finish_reason=None, stop_reason=None)`; `RequestOutput(request_id, prompt, prompt_token_ids, outputs: tuple[CompletionOutput,...], finished=True, metrics=None)`. Both frozen. Test attribute surface matches vLLM names.
- [x] Commit `feat: add RequestOutput/CompletionOutput types`.

### Task 4: Engine backend protocol + MockBackend

**Files:** Create `kairyu/engine/backend.py`, `kairyu/engine/mock.py`, `kairyu/engine/__init__.py`, `tests/unit/test_mock_backend.py`.

- [x] `backend.py`: frozen `CacheHint(session_id, prefix_fingerprint)`, `GenerationRequest(request_id, prompt, sampling_params, cache_hint=None)`, `GenerationResult(request_id, prompt, completions, finished=True)`; `EngineBackend(Protocol)`: `async generate(req) -> GenerationResult`, `stream(req) -> AsyncIterator[GenerationResult]`, `async shutdown()`.
- [x] `MockBackend(responses: Mapping[str,str] | None, latency_s=0.0)`: substring-matched canned responses, fallback `f"mock:{prompt[-48:]}"`; honors `sampling_params.n` (suffix `" #i"` for i>0); `stream` yields cumulative chunks, final has `finished=True`.
- [x] Tests: deterministic echo, canned substring match, n>1, streaming chunks concatenate to full text, latency honored (~0).
- [x] Commit `feat: add EngineBackend protocol and MockBackend`.

### Task 5: Backend registry

**Files:** Create `kairyu/engine/registry.py`, `tests/unit/test_registry.py`.

- [x] `register_backend(name, factory)`, `create_backend(name, **kwargs)`, unknown name → `ValueError` listing known names. `mock` pre-registered; `vllm`/`openai` registered lazily in their modules.
- [x] Commit `feat: add engine backend registry`.

### Task 6: Query features + Router

**Files:** Create `kairyu/orchestration/features.py`, `kairyu/orchestration/router.py`, `tests/unit/test_router.py`.

- [x] `QueryFeatures` frozen: `char_len, word_count, has_code_fence, math_symbol_count, reasoning_keyword_count, multi_step_marker_count, question_count`; `extract_features(query)` pure.
- [x] `RouteTarget = Literal["tier1","tier2","multi_agent"]`; frozen `RouteDecision(target, confidence, features, reason)`; `Router(Protocol).route(query, context=None)`; `RuleRouter(thresholds=DEFAULT_THRESHOLDS)`: short/simple→tier1, code/math/reasoning-heavy→tier2, multi-step markers or very long→multi_agent. `JsonlRouterLog(path).record(query_hash, decision)` for M4 corpus.
- [x] Tests: trivial query→tier1; "prove…step by step" style→tier2; "first…then…finally"+long→multi_agent; p99 latency of 1000 routes < 10ms; log line schema.
- [x] Commit `feat: add rule-based Router with <10ms routing and JSONL decision log`.

### Task 7: Budget

**Files:** Create `kairyu/orchestration/budget.py`, `tests/unit/test_budget.py`.

- [x] Frozen `Budget(max_steps=16, max_refine_depth=2, max_cost_usd=None)`; frozen `BudgetState(budget, steps_used=0, cost_used=0.0)` with `charge(steps=1, cost=0.0) -> BudgetState` (new object), `is_exhausted`, `can_refine(depth)`. Charging never raises; exhaustion is a queryable state.
- [x] Commit `feat: add immutable orchestration budget`.

### Task 8: Conductor (role DAG)

**Files:** Create `kairyu/orchestration/conductor.py`, `tests/unit/test_conductor.py`.

- [x] Frozen `RoleSpec(name, worker, prompt, role_type="worker", depends_on=(), verifies=None)`; `Conductor(roles, workers: Mapping[str, EngineBackend], shared_prefix="")`. Validation at init: unknown deps/workers, cycles (Kahn), verifier must depend on its target. `run(query, budget) -> ConductorResult(final_text, outputs, budget_state, trace)`; wave-parallel `asyncio.gather`; prompts rendered from `{query}` + upstream `{node_name}` placeholders, prefixed by `shared_prefix` (KV-affinity invariant, D5); verifier verdict = first line startswith PASS/FAIL; on FAIL re-run target with feedback while `can_refine(depth)`.
- [x] Tests: linear DAG data flow; diamond DAG wave parallelism (both mid nodes see root output); cycle rejected; verifier FAIL→refine→PASS with depth bound; budget exhaustion returns best-so-far with `is_exhausted`; all prompts start with shared_prefix.
- [x] Commit `feat: add Conductor role-DAG executor with verifier-gated refinement`.

### Task 9: MoA

**Files:** Create `kairyu/orchestration/moa.py`, `tests/unit/test_moa.py`.

- [x] `run_moa(backend, query, n_samples=3, synthesizer=None, sampling_params=None, shared_prefix="") -> MoAResult(final_text, proposals, budget…)`: parallel `asyncio.gather` proposals (temperature bumped, distinct seeds), synthesis call over numbered proposals; synthesizer backend defaults to proposer backend.
- [x] Commit `feat: add Mixture-of-Agents parallel sampling + synthesis`.

### Task 10: Orchestrator facade

**Files:** Create `kairyu/orchestration/orchestrator.py`, `tests/unit/test_orchestrator.py`.

- [x] `Orchestrator(engines: Mapping[str, EngineBackend], router=None(RuleRouter), roles=None, budget=None, shared_prefix="")`; `async run(query) -> OrchestratorResult(text, route, trace)`; tier1/tier2 → direct engine call; multi_agent → Conductor over `roles` (default Planner→Worker→Verifier→Synthesizer DAG built from available engines) ; `run_sync` wrapper. Missing tier engine falls back to any available engine with a trace note (never crash on config gaps).
- [x] Commit `feat: add Orchestrator facade wiring router, conductor and engines`.

### Task 11: DSL (YAML + decorators)

**Files:** Create `kairyu/dsl/spec.py`, `kairyu/dsl/loader.py`, `kairyu/dsl/decorators.py`, `tests/unit/test_dsl.py`.

- [x] pydantic `WorkerSpec(name, backend="mock", model=None, base_url=None, api_key_env=None, options={})`, `RoleNodeSpec`, `BudgetSpec`, `OrchestratorSpec(workers, roles=[], budget=BudgetSpec(), shared_prefix="")` with cross-validation (role.worker must exist). `load_spec(path|str)`; `build_orchestrator(spec) -> Orchestrator` via registry. Decorators: `pool = AgentPool(); @pool.role(name=…, worker=…, depends_on=…)` on functions returning prompt templates; `pool.to_spec()` produces the same `OrchestratorSpec`.
- [x] Tests: YAML round-trip; invalid worker ref rejected; decorator pool ≡ YAML spec; `build_orchestrator` runs end-to-end on mock.
- [x] Commit `feat: add YAML/decorator DSL producing one OrchestratorSpec schema`.

### Task 12: `LLM` entrypoint (vLLM-compatible)

**Files:** Create `kairyu/entrypoints/llm.py`, `tests/compat/test_llm_compat.py`, export from `kairyu/__init__.py`.

- [x] `LLM(model, tokenizer=None, tensor_parallel_size=1, dtype="auto", seed=0, gpu_memory_utilization=0.9, enable_prefix_caching=None, trust_remote_code=False, backend=None, **engine_kwargs)` — unknown kwargs stored, not fatal (vLLM forward-compat). Default backend: `vllm` if importable else `mock`. `generate(prompts, sampling_params=None, use_tqdm=True)` accepts str|list, single or per-prompt params, returns `list[RequestOutput]` in input order; `chat(messages, …)` renders a minimal chat template. Sync via `asyncio.run` (error if called inside a running loop with guidance to use engines directly).
- [x] Contract tests pin: import surface `from kairyu import LLM, SamplingParams`, vLLM basic example shape (`outputs[0].outputs[0].text`, `.prompt`), param passthrough; cross-check block `pytest.importorskip("vllm")` compares constructor signatures.
- [x] Commit `feat: add vLLM-signature-compatible LLM entrypoint`.

### Task 13: External OpenAI-compat worker backend

**Files:** Create `kairyu/engine/openai_backend.py`, `tests/unit/test_openai_backend.py`.

- [x] `OpenAICompatBackend(base_url, model, api_key_env="OPENAI_API_KEY", timeout_s=60, transport=None)` → POST `/chat/completions`, maps to `GenerationResult`; explicit error surfaces (HTTP status, missing key). Tested with `httpx.MockTransport` (no network).
- [x] Commit `feat: add external OpenAI-compatible API worker backend`.

### Task 14: OpenAI-compatible server

**Files:** Create `kairyu/entrypoints/server/protocol.py`, `kairyu/entrypoints/server/app.py`, `tests/server/test_openai_api.py`.

- [x] `protocol.py`: pydantic `ChatCompletionRequest` (model, messages, temperature, top_p, n, stream, max_tokens, stop, tools, tool_choice, extra ignored), response/chunk/tool_call models matching OpenAI schema.
- [x] `app.py`: `create_app(engines, orchestrator=None, default_model=…)`; `GET /v1/models`; `POST /v1/chat/completions`: non-stream + SSE stream (`data:` chunks, terminal `data: [DONE]`); model `kairyu-auto` (or `x_kairyu_orchestrate: true` extra field) → Orchestrator; tool calling: when `tools` present and completion text contains `<tool_call>{json}</tool_call>` → `tool_calls` + `finish_reason="tool_calls"`; 404 unknown model, 422 invalid body.
- [x] Tests via `httpx.ASGITransport`: happy path, streaming chunk reassembly, tool call parse, orchestrated model, unknown model.
- [x] Commit `feat: add OpenAI-compatible server with SSE streaming and tool calls`.

### Task 15: vLLM backend adapter (guarded)

**Files:** Create `kairyu/engine/vllm_backend.py`, `tests/compat/test_vllm_backend.py`.

- [x] Import inside methods; `VLLMBackend(model, **engine_args)` builds `AsyncEngineArgs(enable_prefix_caching=True, …)`; maps kairyu `SamplingParams`→vLLM, vLLM outputs→`GenerationResult`. Module import must succeed without vLLM; instantiation raises clear `RuntimeError` if missing. Tests: import-without-vllm, param mapping (pure function, tested without vLLM), engine roundtrip under `importorskip("vllm")`.
- [x] Commit `feat: add import-guarded vLLM backend adapter`.

### Task 16: CI, coverage gate, bench skeleton, README

**Files:** Create `.github/workflows/ci.yml`, `bench/router_latency.py`, `bench/orchestration_mock_bench.py`, `examples/` (basic offline inference, YAML DSL, server), update `README.md`.

- [x] CI: uv, Python 3.11+3.12 matrix, `ruff check`, `pytest --cov --cov-fail-under=80`.
- [x] Bench skeletons print real measured numbers (router p50/p99, mock orchestration wall time) — no fabricated results; real engine benches land in M2.
- [x] Commit `chore: add CI, coverage gate, bench harness skeleton and examples`.

---

## Self-review notes

- Spec coverage: L2 Router(D3)/Conductor(D4)/MoA(9)/KV-affinity plumbing(D5, Task 4 `cache_hint` + Task 8 shared_prefix) ✔; L3 Python API(12)/server(14)/DSL(11) ✔; external workers(13) ✔; vLLM backend(15) ✔; router log for M4(6) ✔; xgrammar & learned router explicitly deferred (design doc §3).
- Type consistency: `GenerationRequest/GenerationResult` defined once (Task 4) and consumed by 8,9,10,13,14,15; `OrchestratorSpec` defined in 11, consumed by 11 only (Orchestrator ctor takes plain objects).
- No placeholders: each task lists concrete behaviors and error cases; code detail lives in implementation with tests as the specification.
