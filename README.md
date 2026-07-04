# Kairyu

**vLLM-compatible LLM inference framework with native orchestration.**

Kairyu (海流, "ocean current") combines a vLLM drop-in inference API with a first-class
orchestration layer: a learned-router-ready **Router**, a Planner/Worker/Verifier/Synthesizer
**Conductor** (role DAG), and **Mixture-of-Agents** — all behind one Python API and one
OpenAI-compatible endpoint. Under the hood, a custom engine core (Radix-Paged KV cache,
chunked-prefill scheduler, speculative decoding, xgrammar structured output) is being built
against the same pluggable backend seam, along with multi-GPU serving — tensor parallelism,
DP replicas, prefill-decode separation, inter-node KV transfer, pipeline parallelism —
developed CPU-first against the same protocols.

- **Python**: 3.11+ &nbsp;|&nbsp; **License**: MIT &nbsp;|&nbsp; **Tests**: 640+ (coverage gate 80%, currently 92%)

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
                                         Budget, JSONL decision logs, learning pipeline,
                                         ReplicaPool (DP replicas), ClusterSpec (2-node)
L1  Engines         kairyu.engine        EngineBackend protocol:
                                         mock | vllm | openai | kairyu (custom core)
                    kairyu.engine.core   Radix-Paged KV, chunked-prefill scheduler,
                                         EngineCore step loop, n-gram spec decode,
                                         xgrammar structured output, FP8 quant config,
                                         TP runner, P-D coordinator, KV transport,
                                         PP inter-step pipelining
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
| **M5** | Intra-node multi-GPU: tensor parallelism (Communicator + TPModelRunner, TP=2 greedy-equivalent to TP=1 on CPU), DP replicas (ReplicaPool with session affinity), intra-node P-D separation (PDCoordinator + `resume_with_kv`) | ✅ CPU half done — GPU phase is runbook §6, gated on M2 |
| **M6** | Inter-node multi-GPU: static 2-node ClusterSpec, page-granular KV transfer plane (KVTransport, TCP loopback), inter-node P-D, PP=2 via inter-step pipelining | ✅ CPU half done — GPU phase is runbook §7, gated on M5 |
| **M7** | Productionization: `kairyu serve` CLI + DeploymentSpec, health/metrics/auth/concurrency guard, ReplicaPool gateway wiring + background health prober, HTTP session affinity, OpenAI-compatible batch API, Docker/compose topology with CI smoke drill | ✅ CPU half done — GPU bring-up is runbook §9 |
| **M8** | Engine core on real tokens: HF tokenizer seam, full sampler (temp/top-k/top-p/min-p/penalties/seed/logprobs), multi-token scheduler commits, n-gram speculative pipeline, ZMQ process-split engine (`kairyu-proc`) | ✅ Complete |
| **M9** | Truthful API: token-accurate usage + `cached_tokens`, HF Jinja chat templates (byte-matched), logprobs, `/v1/completions`, `n>1`, `json_schema` structured output | ✅ Complete |
| **M12** | Real model zoo: Llama-3.x / Qwen2 / Qwen3 from safetensors; full-engine greedy == `transformers.generate`; fp32 logits < 1e-4 | ✅ Complete |
| **M13** | AttentionBackend seam: torch backend, FlashInfer adapter (contract-pinned), MLA reference math (≡ DeepseekV3Attention to 3.7e-9) | ✅ Complete — kernels verified on deploy day |
| **M14** | Quantization compute: FP8/INT8/AWQ/GPTQ/NVFP4 — all five load and RUN through the full engine on CPU; Triton kernel seams | ✅ Complete |
| **M15** | MoE + MLA architectures: Qwen3-MoE and DeepSeek-V3 (yarn incl.) with full `hf.generate` parity; latent MLA KV pool | ✅ Complete |
| **M16** | Distributed execution over real collectives (gloo): TP=2/EP=2/PP=2 spawn parity gates in the default suite; NCCL is a constructor arg | ✅ Complete — NCCL run on deploy day |
| **M17** | StepExecutor (CUDA-graph seam, fake-graph pinned) + DraftSource + EAGLE-3/MTP heads with checkpoint loaders | ✅ Complete — capture on deploy day |
| **M18** | KV transport: serde, remote P-D handoff, NIXL adapter; two-REAL-process P-D over TCP with byte-parity gates | ✅ Complete — RDMA on deploy day |
| **M10a/b** | Fleet: dynamic ReplicaPool membership, registry/reconciler, OTel tracing, Helm+kind; KV-aware routing (prefix trie + radix KV events + staleness fallback) | ✅ Complete |
| **M11** | Product surface: streaming `kairyu-auto` tiers with honest usage, tenancy v1 (limits/ledger), `/v1/responses`, `/v1/embeddings`, vision wire format, F5 SLO/priority/autoscale logic | ✅ Complete |
| **M19** | Deploy packaging: `Dockerfile.cuda`, GPU compose/Helm, `scripts/gpu_gates/` (runbook §0–§9 + G4/G5, all `--dry-run` pinned) | ✅ Complete — **deploy-ready** |

**Current state: every planned milestone is implemented and CPU-verified. The remaining work is strictly GPU execution — performance gates, kernel tuning, fabric bring-up, `pytest -m gpu`, `scripts/gpu_gates/` — with no code left to write first.** See [`PROGRESS.md`](PROGRESS.md).

Per-milestone design docs (goals, decisions, review amendments) live in
[`docs/design/`](docs/design/); the multi-GPU acceptance contract is
[`docs/goals/g2-multi-gpu.md`](docs/goals/g2-multi-gpu.md); the GPU-phase execution plan is
in [`docs/gpu-runbook.md`](docs/gpu-runbook.md); production deployment (DC topology,
cloud front, rolling updates) is [`docs/deployment.md`](docs/deployment.md).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ytworks/kairyu.git && cd kairyu
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

### Serving (production)

One config file declares the node's role — a gateway (ReplicaPool over remote
replicas, auth, metrics, batch API) or a replica (local engine):

```bash
uv run kairyu serve deploy/compose/gateway.yaml      # or your own DeploymentSpec
./scripts/compose_smoke.sh                           # 1 gateway + 3 replicas via Docker
```

Endpoints: `/v1/chat/completions` (SSE, tools), `/v1/models`, `/v1/files` +
`/v1/batches`, `/health`, `/readyz`, `/metrics` (Prometheus). Requests carrying
the OpenAI `user` field (or `X-Session-ID`) stick to the replica holding their
warm radix-KV prefix. See [`docs/deployment.md`](docs/deployment.md).

### Benchmarks (Fugu suite)

One command runs every benchmark from the
[Fugu release table](https://sakana.ai/fugu-release/) against a deployed
gateway — single models and orchestration tiers as scoreboard columns — and
prints a dated, footnoted scoreboard (goal G6 P-C1):

```bash
kairyu serve examples/deploy_multi_orchestrator.yaml &
kairyu bench run --base-url http://localhost:8000/v1 \
    --model m1 --model kairyu-auto --model kairyu-auto-max
```

Datasets are downloaded to `~/.cache/kairyu/benchmarks` (never committed);
unmet preconditions (no docker, gated dataset, no judge) become annotated
`skipped` cells, so the run always completes. See
[`docs/benchmarks.md`](docs/benchmarks.md).

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
- **Tensor parallelism** (`comm.py`, `step_input.py`, `tp_runner.py`) — `Communicator`
  protocol (with a `FakeCommunicator` for tests) and a divergence-checked TP driver;
  TP=2 is proven greedy-equivalent to TP=1 on CPU (`bench/parity_tp.py`).
- **P-D separation** (`pd.py`) — `PDCoordinator` + `LocalKVHandoff` with copy-before-commit
  KV handoff into `Scheduler.resume_with_kv`; mixed prefill/decode harness in
  `bench/pd_mixed.py`.
- **KV transfer plane** (`kv_transport.py`) — `KVTransport` protocol with in-process and
  TCP-loopback transports for inter-node P-D; benched by `bench/kv_transfer_bench.py`.
- **Pipeline parallelism** (`pipeline.py`) — async submit/handle runner contract and
  `PipelinedEngineCore` inter-step pipelining, with bubble accounting pinned by tests.

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
    core/            custom engine: radix KV, scheduler, spec decode, structured output,
                     TP runner, P-D coordinator, KV transport, PP pipelining
  orchestration/     Router, Conductor, MoA, Budget, Orchestrator, feature extraction,
                     ReplicaPool (DP replicas), ClusterSpec (2-node topology)
    learning/        dataset builder, distilled classifier, contextual bandit
  dsl/               YAML / decorator agent-pool spec + loader
  entrypoints/       LLM, AsyncLLMEngine, chat templating
    server/          OpenAI-compatible FastAPI app + protocol models
examples/            offline inference, YAML pool, serving
bench/               router latency, orchestration overhead, serving, multi-turn prefix,
                     TP parity, mixed P-D, KV transfer
tests/               unit / compat (vLLM surface) / server suites
docs/design/         one reviewed design doc per milestone (M1–M6)
docs/goals/          evidence-first goal contracts (G2 multi-GPU)
docs/gpu-runbook.md  consolidated GPU-day execution plan
```


## Configuration reference

Everything a deployment can set, in one place. The single source of truth is the
**DeploymentSpec YAML** passed to `kairyu serve <config.yaml>` (also mounted at
`/etc/kairyu/config.yaml` in the Docker/Helm images).

### DeploymentSpec (YAML)

```yaml
server:                        # ServerSection = bind address + ServerSettings
  host: 0.0.0.0
  port: 8000
  api_keys_env: KAIRYU_KEYS    # env var with comma-separated keys; null = keyless
                               #   (keyless = trusted node-to-node mesh mode)
  max_concurrency: 256         # global in-flight cap on /v1/*; null disables
  metrics: true                # expose /metrics (Prometheus)
  protect_metrics: false       # require an API key for /metrics too
  access_log: true             # one JSON line per request (X-Request-ID echoed)
  tracing: false               # OTel spans (needs the otel extra; no-op without)
  usage_ledger_path: null      # JSONL usage ledger; enables GET /admin/usage

engines:                       # served model name -> one backend
  qwen:
    backend: kairyu            # mock | kairyu | kairyu-proc | openai | vllm
    options:                   # factory kwargs (see backend options below)
      model_path: /models/qwen2.5-0.5b
  remote-a:
    backend: openai
    options: {base_url: "http://replica-a:8000/v1"}
    health_url: null           # default: <base_url minus /v1>/health

pools:                         # served model name -> ReplicaPool of N replicas
  fleet:
    replicas:
      - {backend: openai, options: {base_url: "http://replica-a:8000/v1"}}
      - {backend: openai, options: {base_url: "http://replica-b:8000/v1"}}
    unhealthy_after: 3         # consecutive failures before leaving the ring
    queue_depth_threshold: 8   # session-affinity load valve
    probe_interval_s: 5.0      # background health prober

chat_templates:                # served model -> HF Jinja template (text or *.jinja path)
  qwen: templates/qwen.jinja

orchestrator:                  # optional kairyu-auto (OrchestratorSpec YAML)
  spec: orchestrator.yaml

orchestrators:                 # optional NAMED auto models (any number; each an
  kairyu-auto-max:             #   arbitrary worker/role DAG — arbitrary composition)
    spec: agent_pool_max.yaml

batch:                         # optional OpenAI-compatible /v1/files + /v1/batches
  data_dir: /var/kairyu/batches
  max_concurrency: 4
```

### Backend options (`engines.*.options`)

| backend | option | default | meaning |
|---|---|---|---|
| `kairyu` | `model_path` | — | safetensors checkpoint dir (Llama-3.x / Qwen2 / Qwen3 / Qwen3-MoE / DeepSeek-V3; FP8/INT8/AWQ/GPTQ/NVFP4 quantized checkpoints auto-detected) |
| | `tokenizer` | model dir | HF tokenizer dir override (`tokenizer.json`) |
| | `num_pages` | 4096 | KV pool pages |
| | `page_size` | 16 | tokens per KV page |
| | `max_num_batched_tokens` | 2048 | chunked-prefill budget per step |
| | `speculative` | null | `"ngram"` enables speculative decoding |
| | `speculative_tokens` | 4 | draft length k |
| | `tensor_parallel_size` | 1 | TP degree (CPU toy path; real-model TP runs through `kairyu/engine/core/worker.py`) |
| `kairyu-proc` | same as `kairyu` | — | runs the engine in a separate process over ZMQ/msgpack (crash isolation) |
| `openai` | `base_url`, `api_key`, `model` | — | any OpenAI-compatible endpoint |
| `vllm` | vLLM engine kwargs | — | needs a Linux GPU host with vllm installed |
| `mock` | — | — | deterministic CI backend |

### Environment variables

| variable | effect |
|---|---|
| `KAIRYU_ATTENTION_BACKEND` | `torch` \| `flashinfer` — overrides the hw-profile kernel selection (invalid values fail loudly) |
| value of `server.api_keys_env` | comma-separated API keys |
| `KAIRYU_MODEL_DIR` | model volume for `docker-compose.gpu.yaml` |
| `GLOO_SOCKET_IFNAME` | set `lo0` on macOS if gloo rendezvous fails (dist tests) |

### HTTP surface

`/v1/chat/completions` (SSE, tools, logprobs, n>1, `response_format: json_schema`,
vision content-parts wire format), `/v1/completions`, `/v1/embeddings`
(float + base64), `/v1/responses` (subset: `input`, `instructions`,
`previous_response_id`), `/v1/models`, `/v1/files` + `/v1/batches`, `/health`,
`/readyz`, `/metrics`, `POST /admin/drain` (auth-protected; flips readyz to 503),
`GET /admin/usage?tenant=` (when the ledger is enabled).

Request extras: `X-Session-ID` (or the OpenAI `user` field) pins a session to the
replica holding its warm KV prefix; `X-Kairyu-Trace: 1` adds a `kairyu_trace`
block to `kairyu-auto` responses; `stream_options: {include_usage: true}` appends
the final usage chunk.

### Multi-tenancy (programmatic)

`create_app(..., tenant_config=TenantConfig(key_tenants={...}, limits={"team-a":
TenantLimits(requests_per_minute=600, tokens_per_minute=200_000)}))` — per-tenant
token buckets run inside auth (401 wins over 429); usage lands in the JSONL
ledger (`server.usage_ledger_path`).

### Tiered auto models

`create_app(..., orchestrators={"kairyu-auto": Orchestrator(...), "kairyu-auto-max":
Orchestrator(..., moa_samples=4)})` — the max tier routes heavy queries through
Mixture-of-Agents. Streaming emits SSE comment keep-alives between stages, so any
OpenAI SDK client works unchanged. The same tiers are declarable in a
DeploymentSpec via the `orchestrators:` map (see the YAML reference above).

### Distributed serving

- **TP** (tensor parallel): `kairyu/engine/core/worker.py` — rank 0 drives the
  scheduler and broadcasts frozen step snapshots; per-rank sharded weights load
  via safetensors slicing. gloo on CPU, `backend="nccl"` on GPUs (constructor arg).
- **EP** (expert parallel): `EpMoeBlock` over `all_to_all` — wraps the MoE blocks.
- **PP** (pipeline parallel): `PpStageModel` stage slices over send/recv.
- **P-D separation**: same-process `PDCoordinator`, or two processes with real
  KV byte transfer over TCP (`RemoteKVHandoff`/`RemoteKVReceiver`; NIXL/RDMA
  adapter ready for deploy day).

### Installation extras & test markers

| extra / marker | contents |
|---|---|
| `uv sync` | core (pydantic, fastapi, httpx, pyyaml, uvicorn, jinja2) |
| `--extra hf` | tokenizers, safetensors (real checkpoints) |
| `--extra fleet` | pyzmq, msgpack (process-split engine, KV events) |
| `--extra otel` | opentelemetry-sdk (tracing) |
| `--extra gpu` | flashinfer / triton / nixl (linux-only markers; macOS ignores) |
| `--extra bench` | datasets / huggingface_hub / pillow / h5py (`kairyu bench download`) |
| `--extra bench-agentic` | mini-swe-agent / swebench / harbor (docker-based benchmarks) |
| `pytest` (default) | everything except `gpu` and `hf_hub` |
| `pytest -m gpu` | deploy-day kernel/graph tests (`tests/gpu/`) |
| `pytest -m hf_hub` | opt-in real-checkpoint downloads |
| `pytest -m dist` | multi-process gloo tests (included in the default run) |

### Deployment artifacts

| artifact | purpose |
|---|---|
| `Dockerfile` / `Dockerfile.cuda` | CPU / CUDA images (one image per role; the mounted spec decides) |
| `deploy/compose/docker-compose.yaml` | gateway + 3 CPU replicas smoke topology |
| `deploy/compose/docker-compose.gpu.yaml` | gateway + GPU replica (nvidia device reservation) |
| `deploy/compose/docker-compose.webui.yaml` | Open WebUI chat surface on the gateway |
| `deploy/helm/kairyu/` (+ `values-gpu.yaml`) | k8s chart; readiness `/readyz`, per-GPU-profile nodeSelector |
| `scripts/kind_smoke.sh` | end-to-end kind cluster smoke (CI job) |
| `scripts/gpu_gates/*.sh` | GPU-day gate scripts (runbook §0–§9 + G4/G5); all support `--dry-run` |
| `bench/serving_bench.py`, `bench/frontier_compare.py`, `bench/kv_transfer_bench.py` | latency/goodput/transfer benches |

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
