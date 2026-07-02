# Kairyu

**vLLM-compatible LLM inference framework with native orchestration.**

Kairyu (海流, "ocean current") combines a vLLM drop-in inference API with a first-class
orchestration layer: a learned-router-ready **Router**, a Planner/Worker/Verifier/Synthesizer
**Conductor** (role DAG), and **Mixture-of-Agents** — all behind one Python API and one
OpenAI-compatible endpoint. Under the hood, a custom engine core (Radix-Paged KV cache,
chunked-prefill scheduler, speculative decoding, xgrammar structured output) is being built
against the same pluggable backend seam.

- **Python**: 3.11+ &nbsp;|&nbsp; **License**: MIT &nbsp;|&nbsp; **Tests**: 170+ (coverage gate 80%)

## Why Kairyu

Most serving stacks treat orchestration (routing, multi-agent pipelines, budgets) as an
application-side afterthought bolted onto a raw completion endpoint. Kairyu makes it native:

- **One import away from vLLM** — `from kairyu import LLM, SamplingParams` runs existing
  vLLM offline examples unchanged, verified by contract tests.
- **Orchestration below the API line** — the Router sees engine-level signals and the
  Conductor's steps hit warm KV prefixes (`cache_hint` plumbing), which pure API-level
  frameworks cannot do.
- **Pluggable backends** — every layer talks to a small async `EngineBackend` protocol, so
  mock (CI), vLLM (local GPU), OpenAI-compatible (external APIs), and the custom `kairyu`
  engine core are interchangeable per worker.
- **Routers that learn** — serving logs feed a distillation + contextual-bandit pipeline
  that upgrades the rule router into a `LearnedRouter` without an API change.

## Architecture

```
L3  Interface       kairyu.entrypoints   LLM / AsyncLLMEngine (vLLM drop-in),
                                         OpenAI-compatible FastAPI server (SSE, tools)
L2  Orchestration   kairyu.orchestration Router → Conductor (role DAG) / MoA,
                                         Budget, JSONL decision logs, learning pipeline
L1  Engines         kairyu.engine        EngineBackend protocol:
                                         mock | vllm | openai | kairyu (custom core)
                    kairyu.engine.core   Radix-Paged KV, chunked-prefill scheduler,
                                         EngineCore step loop, n-gram spec decode,
                                         xgrammar structured output, FP8 quant config
```

Everything above L1 is engine-agnostic: the custom M2 engine is "a fourth backend", not a
rewrite.

## Status & roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1** | L2 orchestration + L3 interface on pluggable backends, vLLM drop-in `LLM` / `AsyncLLMEngine`, OpenAI-compatible server, YAML/decorator DSL | ✅ Complete |
| **M2** | Custom engine core: Radix-Paged KV manager, chunked-prefill scheduler, EngineCore step loop, torch CPU runner with greedy-equivalence proof | ✅ CPU half done — FlashInfer runner / FP8 / overlap pipelining need H100/A100 |
| **M3** | Speculative decoding: n-gram draft policy with tested greedy-equivalence invariant | ✅ CPU half done — EAGLE / CUDA graphs / P-D separation are GPU-phase |
| **M4** | Router learning: JSONL serving logs → labeled dataset → distilled classifier (`LearnedRouter`) → contextual-bandit online refinement | ✅ Complete (CPU-only) |

Per-milestone design docs (goals, decisions, review amendments) live in
[`docs/design/`](docs/design/); the GPU-phase execution plan is in
[`docs/gpu-runbook.md`](docs/gpu-runbook.md).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone <this-repo> && cd rLLM
uv sync
```

Core dependencies are lightweight (pydantic, fastapi, httpx, pyyaml, uvicorn). torch and
xgrammar are dev-group extras used by the engine-core tests; vLLM is only needed for the
`vllm` backend on a Linux GPU host.

## Quick start

```bash
uv run pytest                                        # full suite, coverage gate 80%
uv run python examples/basic_offline_inference.py    # LLM API on the mock backend
uv run python examples/run_yaml_pool.py              # declarative agent pool
uv run python examples/serve.py                      # OpenAI-compatible server on :8000
```

### vLLM drop-in

```python
from kairyu import LLM, SamplingParams   # was: from vllm import ...

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
print(outputs[0].outputs[0].text)
```

`SamplingParams`, `RequestOutput`, `CompletionOutput`, and `AsyncLLMEngine` replicate vLLM's
public surface (the subset exercised by vLLM's own examples), verified by the contract tests
in `tests/compat/`.

### Orchestration (programmatic)

```python
from kairyu import Orchestrator
from kairyu.engine.mock import MockBackend

orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
result = orchestrator.run_sync("First, plan X. Then do Y. Finally, verify.")
print(result.route.target, result.text)
```

The Router picks a target (`tier1` / `tier2` / multi-agent) from query features; multi-agent
routes dispatch to the Conductor (async role DAG with budget-bounded refinement) or MoA
(parallel sampling + synthesis).

### Declarative agent pools

Workers, a role DAG, and a budget in YAML — see
[`examples/agent_pool.yaml`](examples/agent_pool.yaml) for the full file:

```yaml
workers:
  - name: tier1
    backend: mock            # swap for: vllm (local GPU) or openai (external API)
  - name: tier2
    backend: mock

roles:
  - name: planner
    worker: tier2
    role_type: planner
    prompt: "[planner] Break the task into a short plan.\nTask: {query}"
  - name: worker
    worker: tier1
    prompt: "[worker] Execute the plan.\nPlan: {planner}\nTask: {query}"
    depends_on: [planner]

budget:
  max_steps: 12
  max_refine_depth: 2
```

The same spec is available as a decorator front-end via `kairyu.dsl.decorators.AgentPool`.

### OpenAI-compatible server

```bash
uv run python examples/serve.py
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "kairyu-auto", "messages": [{"role": "user", "content": "hi"}]}'
```

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` — with SSE streaming
(`"stream": true`), tool calling, and JSON-schema `response_format` (structured output
enforced by an xgrammar token bitmask on the `kairyu` backend). The reserved model name
`kairyu-auto` routes through the Orchestrator; concrete engine names bypass L2.

## Using open models (Kimi, Qwen, Llama, …)

Open-weight models plug in through two backends, chosen per worker:

- **`vllm`** — weights run on a local GPU via `vllm.AsyncLLMEngine` (prefix caching on by
  default).
- **`openai`** — any OpenAI-compatible endpoint: hosted APIs (Moonshot for Kimi, Together,
  Fireworks, Groq, OpenRouter, …) or a server you run yourself (`vllm serve`, SGLang,
  Ollama).

Model names below are illustrative — check your provider's docs for current identifiers.

### Single model

**Local GPU (vLLM backend).** With vLLM installed, `LLM` auto-selects it; construct
`VLLMBackend` explicitly to pass extra `AsyncEngineArgs` (tensor parallelism etc.):

```python
from kairyu import LLM, SamplingParams
from kairyu.engine.vllm_backend import VLLMBackend

# simplest — vLLM auto-detected when installed
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# explicit backend for engine args (e.g. a large MoE across GPUs)
backend = VLLMBackend(model="moonshotai/Kimi-K2-Instruct",
                      tensor_parallel_size=8, trust_remote_code=True)
llm = LLM(model="moonshotai/Kimi-K2-Instruct", backend=backend)
```

**Hosted API (OpenAI-compatible backend).** Kimi K2 is a ~1T-parameter MoE, so most setups
use Moonshot's API (or another provider) instead of local weights. The API key is read
from the environment variable named by `api_key_env` — never hardcoded:

```bash
export MOONSHOT_API_KEY=sk-...
```

```python
from kairyu import LLM, SamplingParams
from kairyu.engine.openai_backend import OpenAICompatBackend

backend = OpenAICompatBackend(
    base_url="https://api.moonshot.ai/v1",
    model="kimi-k2-0905-preview",
    api_key_env="MOONSHOT_API_KEY",
)
llm = LLM(model="kimi-k2", backend=backend)
outputs = llm.generate(["Explain paged KV caching in two sentences."],
                       SamplingParams(max_tokens=128))
print(outputs[0].outputs[0].text)
```

**Your own server.** The same backend points at any self-hosted OpenAI-compatible server —
useful when Kairyu runs on a laptop and the GPU box lives elsewhere:

```bash
# on the GPU box
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
# on the Kairyu side — the env var must exist even if the server ignores auth
export LOCAL_LLM_API_KEY=unused
```

```python
backend = OpenAICompatBackend(
    base_url="http://gpu-box:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key_env="LOCAL_LLM_API_KEY",
)
```

### Multi-model orchestration

The Router's targets are the worker names `tier1` (light/cheap) and `tier2`
(frontier/strong), plus `multi_agent` for role-DAG dispatch — so a typical mixed pool puts
a small local open model on `tier1` and Kimi on `tier2`. Declaratively:

```yaml
# pool.yaml
workers:
  - name: tier1                      # easy queries: local open model on your GPU
    backend: vllm
    model: Qwen/Qwen2.5-7B-Instruct
    options:                         # extra kwargs forwarded to the backend constructor
      gpu_memory_utilization: 0.85
  - name: tier2                      # hard queries + planner/verifier roles: Kimi K2
    backend: openai
    model: kimi-k2-0905-preview
    base_url: https://api.moonshot.ai/v1
    api_key_env: MOONSHOT_API_KEY

roles:
  - name: planner
    worker: tier2
    role_type: planner
    prompt: "[planner] Break the task into a short plan.\nTask: {query}"
  - name: worker
    worker: tier1
    prompt: "[worker] Execute the plan.\nPlan: {planner}\nTask: {query}"
    depends_on: [planner]
  - name: synthesizer
    worker: tier2
    role_type: synthesizer
    prompt: "[synthesizer] Final answer.\nDraft: {worker}\nTask: {query}"
    depends_on: [worker]

budget:
  max_steps: 12
  max_cost_usd: 0.50                 # hard cap for one orchestrated request
  cost_per_1k_chars_usd: 0.002
```

```python
from kairyu.dsl.loader import build_orchestrator, load_spec

orchestrator = build_orchestrator(load_spec("pool.yaml"))
result = orchestrator.run_sync("Compare radix-tree and hash-based KV prefix sharing.")
print(result.route.target, result.text)
```

Or programmatically, mixing backends freely:

```python
from kairyu import Orchestrator
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.vllm_backend import VLLMBackend

orchestrator = Orchestrator(engines={
    "tier1": VLLMBackend(model="Qwen/Qwen2.5-7B-Instruct"),
    "tier2": OpenAICompatBackend(
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k2-0905-preview",
        api_key_env="MOONSHOT_API_KEY",
    ),
})
```

Short queries route to the local Qwen worker; long or multi-step queries escalate to Kimi
or the role DAG (thresholds are configurable via `RouteThresholds` in
`kairyu/orchestration/router.py`). To serve the pool over HTTP, pass the orchestrator to
`create_app` as in [`examples/serve.py`](examples/serve.py) and call model `kairyu-auto`.

## Engine core (`kairyu.engine.core`)

The custom engine behind backend name `kairyu`, developed CPU-first so every component is
unit-tested before GPU time is spent:

- **Radix-Paged KV manager** (`radix_kv.py`, `pages.py`) — radix-tree prefix sharing over
  paged KV blocks; targets >80% cache hit rate on shared-prefix workloads.
- **Chunked-prefill scheduler** (`scheduler.py`) — token-budget scheduling with
  robustness tests for preemption and capacity edges.
- **EngineCore step loop** (`engine_core.py`, `overlap.py`) — vLLM-V1-style API/core split;
  overlap pipelining lands with the GPU runner.
- **Speculative decoding policy** (`spec_decode.py`) — n-gram prompt-lookup drafting +
  greedy verification, with a tested invariant: identical output to plain autoregressive
  greedy decoding given identical target logits.
- **Structured output** (`structured.py`) — xgrammar-compiled JSON-schema grammars applied
  as per-step token bitmasks.
- **Torch CPU runner** (`torch_runner.py`) — proves engine greedy-equivalence with real
  tensors, paged-KV attention included.

GPU acceptance criteria (vs vLLM / SGLang on identical hardware) and the step-by-step
execution plan are in [`docs/gpu-runbook.md`](docs/gpu-runbook.md).

## Router learning (M4)

`JsonlRouterLog` records routing decisions (`query_sha256`, target, features, confidence);
outcome records join on the query hash — raw text is never stored. From there:

1. `build_dataset` labels queries with the highest mean-utility target
   (`utility = quality − cost_weight · cost_usd`).
2. A distilled classifier warm-starts `LearnedRouter` on the same pluggable `Router`
   protocol as the rule router.
3. A contextual bandit refines the policy online (distillation alone inherits
   logging-policy bias — see `docs/design/m4-router-learning.md` §2.2).

Acceptance target: ≥97% of tier2-only quality at ≥40% lower inference cost.

## Project layout

```
kairyu/
  engine/            EngineBackend protocol, registry, mock/vllm/openai backends
    core/            custom engine: radix KV, scheduler, spec decode, structured output
  orchestration/     Router, Conductor, MoA, Budget, Orchestrator, feature extraction
    learning/        dataset builder, distilled classifier, contextual bandit
  dsl/               YAML / decorator agent-pool spec + loader
  entrypoints/       LLM, AsyncLLMEngine, chat templating
    server/          OpenAI-compatible FastAPI app + protocol models
examples/            offline inference, YAML pool, serving
bench/               router latency, orchestration overhead, serving, multi-turn prefix
tests/               unit / compat (vLLM surface) / server suites
docs/design/         one reviewed design doc per milestone (M1–M4)
docs/gpu-runbook.md  consolidated GPU-day execution plan
```

## Development

```bash
uv run pytest                        # tests + coverage (gate: 80%, enforced via addopts)
uv run ruff check .                  # lint (E, F, I, UP, B; line length 100)
uv run python bench/router_latency.py
uv run python bench/orchestration_mock_bench.py
```

Conventions: all CI-facing tests run against `MockBackend` (deterministic,
dependency-free); GPU-dependent claims are never reported without a `bench/` reproduction
script.

## License

MIT — see [LICENSE](LICENSE).
