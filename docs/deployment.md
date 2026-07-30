# Deployment Guide — On-Prem DC with a Managed Cloud Front

How to run Kairyu as a product: one container image, one `kairyu serve`
entrypoint, an on-prem GPU fleet behind a stateless gateway tier, and a thin
managed cloud edge. Design rationale: `docs/design/m7-productionization.md`;
acceptance gates: `docs/goals/g3-production-deployment.md`.

## 1. Topology

```
Internet
   │
Cloud edge (managed; no Kairyu code here)
   ├─ WAF            — DDoS, per-client rate limits, bot filtering
   ├─ L7 LB          — TLS termination, health-checked routing to gateways
   │
Private interconnect (Direct Connect / ExpressRoute class; IPsec VPN fallback)
   │
DC gateway tier — 2× CPU nodes, stateless          kairyu serve gateway.yaml
   ├─ OpenAI-compatible API + orchestration (Router/Conductor/MoA)
   ├─ ReplicaPool per served model → remote GPU replicas
   ├─ /health /readyz /metrics, API keys, concurrency guard, batch worker
   │
DC GPU replica tier — N nodes                      kairyu serve replica.yaml
   └─ each node: same image, local engine (kairyu / vllm backend),
      internal TP / P-D / PP layout per docs/gpu-runbook.md §6–7
```

Traffic crossing the interconnect is completions requests/responses only.
KV pages, TP collectives, and P-D transfers never leave the DC fabric
(IB/RoCE stays node-to-node inside the DC).

## 2. Division of security duties

| Concern | Owner | Kairyu's part |
|---|---|---|
| DDoS, bot filtering, per-client rate limits | Cloud WAF | — |
| TLS termination, certificates | Cloud LB (or DC reverse proxy) | serves plain HTTP behind it |
| Client authentication | Gateway | `server.api_keys_env` (static keys, constant-time compare) |
| Process overload | Gateway | `server.max_concurrency` → 429 + Retry-After |
| Routing inspection | Gateway | `/v1/route` uses data-plane auth; `/routing` remains inside the configured API-key boundary |
| Node-to-node auth inside the DC | Deployment choice | keyless (`api_key_env: null`) or a shared key env var |
| Audit trail | Gateway | JSON access log with `X-Request-ID`, JSONL router decision log |

## 3. Node setup (systemd + docker compose — design m7 D2)

Kubernetes is deliberately not required: the fleet is small and static, GPU
nodes are pinned to hardware, and the design lineage excludes elasticity
(g2 §6). Everything is containerized, so adopting k3s/RKE2 later is
manifest-writing, not re-architecture. Revisit k8s when any of these hold:
the fleet grows past ~5–8 nodes, multiple teams deploy onto the cluster, or
rolling deploys become weekly toil.

Per node: install Docker, drop the compose file + config, and wrap it in a
systemd unit so the stack survives reboots:

```ini
# /etc/systemd/system/kairyu.service
[Unit]
Description=Kairyu node (gateway or replica; config decides)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/kairyu
ExecStart=/usr/bin/docker compose up -d --wait
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

GPU replica nodes additionally need the NVIDIA container toolkit and a
`deploy.resources.reservations.devices` (or `gpus: all`) stanza in their
compose service.

## 4. Configuration walkthrough

Both roles run the same image (`Dockerfile` at the repo root); the mounted
DeploymentSpec decides what the process is. Working CPU examples live in
`deploy/compose/` and are exercised by `scripts/compose_smoke.sh` in CI.
The flat `server:` mapping is a versioned deployment schema and is translated
explicitly to runtime server settings. Runtime-only settings do not
automatically become accepted YAML keys.

Validate a deployment graph before rollout without starting a server:

```bash
kairyu validate /etc/kairyu/deployment.yaml
```

Validation is always deterministic and offline. It checks the DeploymentSpec
schema and declared local filesystem references: orchestrator specs, `.jinja`
chat templates, and local `kairyu`/`kairyu-proc` model and tokenizer artifacts.
Orchestrator and `.jinja` paths resolve from the deployment file's directory;
native model and tokenizer paths retain the serve process's working-directory
semantics.
It does not read credential environment variables, expose secret values,
construct a backend, materialize model tensors, contact an endpoint, or probe
a GPU. Model compatibility is derived from metadata-only `meta` shapes, and
checkpoint tensor data is never read. Network
checks are therefore skipped and hardware checks remain indeterminate; those
belong to the readiness and deployment acceptance gates. A valid graph exits
`0`, a validation failure exits `1`, and argparse usage errors exit `2`.

The current DeploymentSpec has no standalone adapter, grammar, or benchmark
artifact references, so `validate` does not invent checks for them. Their
request-, benchmark-, and runtime-owned validation remains unchanged.

Replica node (GPU):

```yaml
server: { host: 0.0.0.0, port: 8000 }
engines:
  llama-70b:
    backend: kairyu            # or vllm; mock for CPU smoke
    options:
      model_path: "meta-llama/Llama-3.3-70B-Instruct"
      tensor_parallel_size: 4
      max_num_seqs: 16       # bound active sequences
      priority_age_s: 60.0   # null = FIFO; 0 = strict priority; >0 = aging
      pipeline_depth: 2        # unified schedule/device overlap; default 1
      decode_mode: cuda_graph   # explicit opt-in; eager is the safe default
      cuda_graph_max_batch: 8
      cuda_graph_max_pages: 64
```

`pipeline_depth: 1` reproduces the synchronous serving path. Depth 2 or greater
submits immutable request snapshots ahead of the oldest commit while preserving
the same streaming, stop, grammar, speculative, preemption, and chunked-prefill
commit path. CUDA-graph decode reserves one KV page for padding rows and captures buckets up
to the configured batch and page-table limits. Oversized decode steps fall back
to eager execution. Invalid modes, CPU placement, unsupported attention/model
paths, or a page limit that leaves no scratch page fail during startup rather
than after traffic arrives.

Gateway:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  api_keys_env: KAIRYU_API_KEYS     # comma-separated client keys
  max_concurrency: 256
pools:
  llama-70b:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "llama-70b", api_key_env: null, upstream: kairyu }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "llama-70b", api_key_env: null, upstream: kairyu }
    unhealthy_after: 3
    queue_depth_threshold: 8
    probe_interval_s: 5.0
orchestrator: { spec: agent_pool.yaml }          # optional: kairyu-auto routing
embeddings:
  embed-test:
    backend: mock                                # deterministic built-in CPU backend
    dimensions: 384
batch: { data_dir: /var/lib/kairyu/batch, max_concurrency: 8 }  # single gateway
tenants:
  default_tenant: default
  limits:
    default:
      request_burst: 8        # initial/refill request burst, not one full minute
      token_burst: 20000      # token-bucket capacity
      max_in_flight: 8        # lease held through final streamed body byte
      interactive_priority: 0  # smaller values run first
      batch_priority: 1        # Batch API overrides client-supplied priority
```

Every tenant profile must keep `interactive_priority < batch_priority`.
For noisy-neighbor isolation, configure `request_burst`, `token_burst`, and
`max_in_flight` explicitly. The tenant lease is acquired outside the global
concurrency guard, released exactly once after unary/SSE completion or failure,
and also enforced per Batch API line. After validation, every generation
surface reserves a worst-case compute ceiling before shared replica placement;
`n`/`best_of`, prompt arrays, and bounded AUTO fan-out are included. Only exact
single-candidate terminal usage refunds unused capacity. Failure, disconnect,
missing usage, and multi-candidate work consume the full reservation.
`kairyu_tenant_in_flight_requests` and `kairyu_tenant_reserved_tokens` must both
return to zero after work drains.

For two or more gateways, select the shared PostgreSQL batch backend instead of
mounting the filesystem store over NFS:

```yaml
batch:
  store: postgres
  dsn_env: KAIRYU_BATCH_POSTGRES_DSN
  store_id: production
  max_concurrency: 8
  poll_interval_s: 0.5
  lease_seconds: 30
```

Inject the DSN from a Secret and set `KAIRYU_BATCH_WORKER_ID` to the immutable
gateway Pod UID. PostgreSQL owns file chunks, jobs, claims, leases and fencing
tokens; a process-local submit signal only wakes the local poller. After an
owner crash, inference may run again, but only the current fenced claimant may
publish the terminal job and output.

The formal F5b GPU check is `bench/noisy_neighbor_gpu_bench.py --assert-gate`.
It compares 10x offered noisy traffic against bracketed compliant-neighbor
controls with matched accepted work; good-only latency is retained as a
secondary reference. The pinned Qwen3-32B TP8 result is
`bench/results/f5b-noisy-neighbor-qwen3-32b-tp8-2026-07-28.json`.

### OpenAI-compatible upstream capabilities

Set `options.upstream` explicitly whenever the provider is known. Kairyu
validates non-default request intent before opening the HTTP client, so a
compatibility endpoint cannot return 200 while silently discarding a field.
The `generic` default preserves older deployments but should be treated as a
migration profile.

Every profile currently declares `prompt_kinds={"text"}` because
`OpenAICompatBackend` targets Chat Completions. This is independent from the
local native and vLLM adapters, which accept `TokensPrompt` directly, and from
Kairyu's `/v1/completions` token-array input. A remote Chat-compatible worker
therefore rejects token and multimodal prompts before creating its HTTP client;
it never stringifies IDs or flattens media. Prompt kinds participate in the
same immutable capability key as sampling/tool policy.

Prompt content must never be placed in `extra_args`. Kairyu reserves alternate
text/token/embed/multimodal carrier names and `cache_salt` at the common
`SamplingParams` boundary, independently of a provider allowlist. Unknown
top-level or nested Chat fields are also rejected before dispatch. Use
`PromptInput` for content and `CacheHint` for native affinity; a legacy backend
without `validate_request` remains compatible only with plain string prompts.

| `upstream` | Portable request controls | Provider-specific notes |
|---|---|---|
| `openai` | OpenAI Chat Completions sampling, logprobs, structured output, tools | Emits the canonical `max_completion_tokens`; allows `reasoning_effort`, `service_tier`, and `parallel_tool_calls` in `extra_args`. |
| `anthropic` | temperature (0–1), top-p, one completion, max tokens, stop, non-strict tools | Rejects penalties, seed, logprobs, `response_format`, and strict tool schemas because the [Anthropic compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) documents them as ignored. Anthropic recommends its native API for production features. |
| `gemini` | max tokens, structured output, and non-strict tools | Allows `reasoning_effort` and the documented `extra_body.google` extension object. Sampling controls vary across Gemini model families, so fields not guaranteed by the [Gemini OpenAI compatibility contract](https://ai.google.dev/gemini-api/docs/openai) fail closed unless a pinned deployment declares a verified custom contract. |
| `kairyu` | OpenAI controls plus `top_k`, `min_p`, `repetition_penalty`, `stop_token_ids`, `min_tokens`, `ignore_eos`, and signed-int64 `priority` | Use for gateway-to-Kairyu replica traffic. These extensions and the bounded interactive/batch class hint are typed and preserved through the receiving HTTP boundary into native scheduler admission. |
| `vllm` | OpenAI controls plus result-preserving [vLLM Chat extensions](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and `priority` | Includes `skip_special_tokens` in addition to the Kairyu extension set. Smaller priority values run first. Kairyu's local vLLM adapter requires `scheduling_policy=priority` so the field cannot be silently ignored; a separately operated remote vLLM server must enable the same policy. `prompt_logprobs` fails closed until Kairyu's result/API types can return the upstream prompt distribution. |

Example provider configurations:

```yaml
engines:
  hosted-openai:
    backend: openai
    options:
      base_url: https://api.openai.com/v1
      model: gpt-4o
      api_key_env: OPENAI_API_KEY
      upstream: openai
  hosted-anthropic:
    backend: openai
    options:
      base_url: https://api.anthropic.com/v1
      model: claude-sonnet-4-5
      api_key_env: ANTHROPIC_API_KEY
      upstream: anthropic
  hosted-gemini:
    backend: openai
    options:
      base_url: https://generativelanguage.googleapis.com/v1beta/openai
      model: gemini-2.5-flash
      api_key_env: GEMINI_API_KEY
      upstream: gemini
```

Vendor extensions use `SamplingParams(extra_args=...)`. Each key must be in
the selected preset's allowlist or added by deployment configuration:
for example, Gemini thinking configuration is
`extra_args={"extra_body": {"google": {"thinking_config": {...}}}}`.

```yaml
options:
  upstream: generic
  capabilities:
    allow_extra_args: [vendor_cache]
    allow_sampling_fields: [best_of]  # only after verifying this endpoint executes it
```

Core keys such as `model`, `messages`, `temperature`, `tools`, and `stream`
cannot be allowlisted or overwritten through vendor extensions.
`response_format` remains a reserved canonical intent in the existing
`SamplingParams.extra_args` representation and cannot be reclassified by the
vendor allowlist. Capability overrides are validated while loading deployment
and orchestrator specs; unsupported non-neutral request values return a
pre-dispatch HTTP 400.
Additive sampling overrides are limited to the `generic` custom-provider
profile. Named provider presets may be narrowed but not broadened, preventing
configuration from re-enabling fields that the provider documents as ignored.

### Responses API and Codex

`POST /v1/responses` supports unary and canonical typed SSE responses for text
and function calls. Flat OpenAI function definitions and Codex namespace
definitions are accepted; `function_call_output` and tenant-scoped
`previous_response_id` continue the tool loop without resending earlier items.
Only successful responses are stored. `store: false` disables continuation.
The in-process store is bounded and intentionally not shared across gateways,
so route a continued response to the same gateway or omit
`previous_response_id` and send the full input history in an HA deployment.

Codex can use Kairyu as a custom Responses provider without source changes:

```bash
KAIRYU_BASE_URL=http://127.0.0.1:8000/v1 \
KAIRYU_MODEL=qwen3-32b \
KAIRYU_API_KEY=local \
scripts/codex_responses_smoke.sh
```

The script creates an ephemeral Codex run with `wire_api="responses"` and a
read-only sandbox. Its default `KAIRYU_SMOKE_MODE=tool` requires a real `pwd`
command event, its tool result, and a final message containing `PASS`;
`KAIRYU_SMOKE_MODE=text` selects a text-only wire smoke. Kairyu accepts Codex
function namespaces and its disabled web-search declaration; enabled hosted
web search remains unsupported.
`background`, Conversations API objects, hosted prompt templates, moderation,
automatic truncation, context management, `max_tool_calls`, and response
top-logprobs fail before model dispatch. This explicit rejection boundary keeps
the accepted compatibility surface truthful. `service_tier` supports only the
neutral `auto` selection; explicit paid/priority tiers fail rather than being
echoed as executed. Codex reasoning/include metadata is accepted for wire
compatibility but Kairyu emits no reasoning or encrypted-reasoning output item.
`text.verbosity` is applied as a model instruction, not claimed as a provider
quality-of-service tier.

Operational notes:

- **Session affinity is the cache layer.** Clients that send the OpenAI
  `user` field (or an `X-Session-ID` header) keep a conversation on the
  replica holding its warm radix-KV prefix. Watch
  `kairyu_pool_decisions_total{reason="session_affinity"}` — a low share on
  multi-turn traffic means clients aren't sending session identity.
- **Gateway HA**: run gateways behind an L7 load balancer that consistently
  hashes `X-Session-ID`. Use `batch.store: postgres` for cross-gateway files and
  jobs. The filesystem store remains single-gateway only and must not be shared
  over NFS.
- **Two GPU nodes acting as one model** (TP/PP/P-D across nodes) is an
  engine-layer concern configured by `ClusterSpec` per `docs/gpu-runbook.md`
  §7; the gateway still sees one OpenAI endpoint per coherence domain.
- **Embedding model IDs are explicit.** Each `embeddings:` key is listed by
  `/v1/models`, routes only to its configured backend, and must not collide
  with an engine, pool, or orchestrator name. Unknown IDs return
  `model_not_found` without execution or usage accounting.

## 5. Rolling model update (gate C7)

Weights update = rolling replica restart; there is no hot swap (m7 §3).
Do not stop a still-eligible replica and rely on a failed request to eject it.
Use a drain-first, partitioned StatefulSet rollout:

1. Set `updateStrategy.type=RollingUpdate` and hold
   `rollingUpdate.partition` at the current replica count before staging the
   new image or weights. For the F1b restart drill, invoke
   `kubectl rollout restart` once while that partition still holds every Pod.
2. Starting at the highest ordinal, call the replica's `/admin/drain` through
   the authenticated Kubernetes Pod proxy. Wait for both the ready,
   non-terminating EndpointSlice UID to disappear and the gateway membership
   record to mark that UID ineligible. Also wait for already placed work to
   complete; do not drain the remaining replicas.
3. Lower the partition by one. The StatefulSet may now replace only that
   ordinal. Wait for a different Pod UID at the update revision, the replica's
   `/readyz` to return 200, and the gateway to report the new UID eligible.
4. Repeat the same bounded step for the next ordinal. On timeout, stop without
   advancing the partition; do not repair the rollout with direct Pod deletes.

The no-new-work boundary is the later of gateway ineligibility and
EndpointSlice withdrawal. A local `/admin/drain` acknowledgement alone is not
proof that distributed routing has converged. Requests placed before that
boundary may finish normally, but no placement after it may select the old UID,
and the partition is not released until its outstanding count is zero.

The automated F1b rehearsal freezes this sequence in a checked-in driver:
exactly 100 old UIDs become 100 disjoint new UIDs, retry count is zero, every
offered request returns 2xx, and raw request, placement, membership, rollout,
readiness, Pod, and EndpointSlice evidence is independently replayed. Once the
job starts, no human or cluster-installed rollout Operator chooses or repairs a
step. Pull-request smoke validates the state machine at reduced scale; only the
clean exact-head 100-replica formal kind artifact satisfies F1b.

`scripts/compose_smoke.sh` remains the C1–C3 kill/eject/recover prerequisite.
Its intentional fault-trigger request is not a drain-first rolling update and
is not C7/F1b acceptance evidence.

## 6. Observability

- `/metrics` (Prometheus): request counts by model/status, latency, and
  ledger-reconcilable usage executions plus prompt/completion/cached/uncached
  tokens by tenant,
  histograms, per-replica outstanding/health, pool decision counts, batch
  job states. Priority-enabled replicas also expose bounded
  `kairyu_scheduler_priority_events_total`,
  `kairyu_scheduler_queue_depth`, and
  `kairyu_scheduler_queue_high_watermark` series by model and
  interactive/batch class. Tenant-enabled gateways additionally expose
  `kairyu_tenant_admission_total` by bounded tenant/source/decision/reason and
  `kairyu_tenant_in_flight_requests`. Scrape every gateway and replica.
- With a versioned `pricing:` section, `/admin/usage.csv` snapshots the local
  immutable ledger and exports tenant charges for a `[start_ts,end_ts)` period.
  The CSV carries source SHA-256, price-sheet version, Decimal unit rates,
  cached/uncached/output components, tenant discount, and total. Corrupt or
  truncated input fails closed with `invoice_ledger_invalid`.
- JSON logs on stdout (one access line per request, prober/batch events);
  ship with the log collector of your choice.
- `/readyz` is the LB health check for gateways; `/health` is the container
  healthcheck on every node.

## 7. DC–cloud interconnect

Only the gateway tier needs to be reachable from the cloud edge; size the
link for request/response payloads (tokens, not tensors — KV never crosses
it). A Direct Connect / ExpressRoute class link gives predictable latency
for TTFT-sensitive SLOs; keep an IPsec VPN as the fallback path. Restrict
the edge→DC path to the gateway port; replica ports stay DC-internal.

## 8. Appendix: Kubernetes (untested reference)

If a revisit trigger from §3 fires, the migration is: one Deployment per
role, the DeploymentSpec as a ConfigMap, `/health`→livenessProbe,
`/readyz`→readinessProbe, a Service in front of gateway pods, and the NVIDIA
GPU operator on replica nodes (pin one replica pod per node with
`nodeSelector` + `resources.limits.nvidia.com/gpu`). These manifests are
deliberately not shipped as tested artifacts — the compose topology is the
supported path until the triggers fire (m7 D2).
