# M11 Design: Fugu-Class Product Surface + Tenancy — CPU Complete

Status: **Implemented** (2026-07-03; D3/D6 amended 2026-07-27). Reviewed
(1-reviewer panel with file/line evidence + OpenAI SDK verification,
2026-07-03; §5 binding).
Milestone: M11 (roadmap P-B/P-C + F5 CPU halves; goal G6)
Date: 2026-07-03
Depends on: m7 Orchestrator/Conductor/MoA/Budget, m9 server surface, M10a
(BatchStoreProtocol, telemetry), m8 scheduler. Consumed by: production
launch (G6 gates).

## 1. Goal

Close the product gaps that make Kairyu a Fugu-class service rather than a
bare inference server: streaming orchestration with honest usage, tiered
auto models, multi-tenant limits+metering, Responses/Embeddings APIs,
vision wire format, and the F5 latency-protection logic (priority admission,
SLO shedding, autoscaler decisions) — all CPU-tested.

## 2. Key decisions

### D1 — Streaming orchestrator with token accounting

`Orchestrator.run_chat(messages, stream=False)` joining the m7 `run()`
path: (a) token accounting — every internal `GenerationResult.usage` is
summed into `OrchestratorResult.usage` (`orchestration_input/output_tokens`)
and surfaced through the API as REAL usage (removing the m9 "usage=None
until M11" fallback); (b) streaming — the FINAL unit (direct route,
conductor final unit, or MoA synthesis) streams token deltas; pre-final
stages emit typed `status` keep-alive events (`OrchestratorEvent =
status|delta|result`). `X-Kairyu-Trace: 1` request header opts into a
trace block (stage timings + decisions) in the final event. app.py's AUTO
path switches to `run_chat` for both stream and non-stream.

### D2 — Tiered auto models

`create_app(orchestrators: dict[str, Orchestrator])` (back-compat shim for
the single `orchestrator=` kwarg): `kairyu-auto` (default tier) and
`kairyu-auto-max` (deep tier: bigger budget, MoA enabled) are just two
configured Orchestrator instances listed in /v1/models.

### D3 — Tenancy v1

`tenancy.py`: `TenantConfig` (key→tenant map, per-tenant rate + token
budgets), `TenantLimitMiddleware` (pure-ASGI, token-bucket per tenant on
/v1/*; 429 with retry-after), usage ledger — persistent O_APPEND JSONL,
one record per execution (tenant, model, prompt/completion/cached tokens, ts),
and row-isolated aggregation that preserves valid totals around malformed
records — plus `GET /admin/usage?tenant=`. A bounded lifecycle-owned writer
moves append/flush calls to one background thread, batches up to 128 records,
and drains every accepted record at app shutdown. Aggregation enters an ordered
flush barrier and performs the subsequent file scan through
`asyncio.to_thread`, preserving read-after-write without blocking the event
loop. A barrier flushes language/runtime buffers to the OS for reader
visibility; it deliberately does not call `fsync`, so no power-loss durability
is claimed.
Isolation gate: tenant A at its limit 429s while tenant B proceeds; ledger
totals reconcile with returned usage to <0.1%. Dedicated Prometheus counters
mirror only accepted usage rows (not generic HTTP requests): executions and
prompt/completion/cached tokens are labeled by the bounded tenant identity. On
single-gateway restart those counters are restored from ledger totals before
serving, so ledger-versus-Prometheus reconciliation remains exact across a
clean process restart. A truncated crash tail is preserved for diagnostics and
terminated before the next complete row is appended.

Fleet reconciliation does not move shared state into the request path. Each
gateway remains the sole writer of its local append-only ledger; an offline
reconciler independently aggregates request audit logs and immutable gateway
ledgers by tenant. The fixed F5e replay covers all public endpoint families,
post-dispatch disconnect/failure, pre-result failure, batch-output rollback,
three gateways, and cached-token export. A same-host shared SQLite candidate
with no network latency and durability disabled still measured worse than the
selected local asynchronous boundary: the explicit-uncached rerun's
median-of-five producer p99 was 28.026 → 19.696 us and throughput
72,374 → 95,029 rows/s over 30,000 rows.

Pricing also remains outside the request path. New ledger rows freeze both
cached and uncached input counts; legacy rows derive uncached input as
prompt-minus-cached. A validated, versioned blended `PriceSheet` applies one
cached-input discount rule and optional tenant discounts with Decimal
arithmetic. `/admin/usage.csv` reads a size-bounded ledger snapshot and exports
`[start_ts,end_ts)` invoice rows with separated input/output quantities, unit
rates, component charges, discount, total, source SHA-256, and deterministic
invoice ID. Any malformed/truncated record fails the export closed.

### D4 — `/v1/responses` (subset) + `/v1/embeddings`

Responses: `POST /v1/responses` accepting `model`, `input` (string or
message array), `previous_response_id`, `stream`; a `ResponseStore`
(protocol + in-memory impl) persists response items so
`previous_response_id` reconstructs context; OpenAI SDK
`client.responses.create` round-trip test. Embeddings: `EmbeddingBackend`
protocol (`embed(texts) -> list[vector]`), `MockEmbeddingBackend`
(deterministic hash-based vectors), `POST /v1/embeddings` with usage. Embedding
backends are registered under explicit, non-colliding served model IDs; the
handler resolves that bounded registry before work, returns `model_not_found`
for misses, exposes every configured ID through `/v1/models`, and uses only the
resolved ID for the response, request metrics, and usage accounting.

### D5 — Vision wire format

protocol.py accepts OpenAI content-parts (`type: text|image_url`) in chat
messages; `ChatTemplate` renders text parts and passes image references to
a `VisionAdapter` seam that M-class VLM engines implement later (GPU);
non-vision engines get a clean 400. Wire-format tests only — no VLM
inference locally.

### D6 — F5 CPU: priority admission + SLO shed + autoscaler logic

(a) `EngineRequest.priority` already exists — `Scheduler._admit_waiting`
orders by (priority, arrival); starvation guard: priority ages up after
`age_s`. (b) SLO early rejection: `slo.py` `AdmissionController` — a TTFT
predictor from queue depth + running EMA of step time; over-SLO requests
are shed (429 `slo_shed`) or deferred to batch (`defer` decision recorded).
(c) `autoscale.py`: pure decision function `(metrics window) →
scale_up/down/hold + reason` with hysteresis; logged, not executed (the
executor is a deploy-day k8s HPA/keda adapter).

**Indexed waiting queue amendment (2026-07-27, issue #219).** FIFO admission
uses an `OrderedDict` so append, head pop, and cancellation by request ID are
O(1). Priority admission uses a stable sequence-numbered heap plus an ID index;
removal leaves lazily reclaimed tombstones with amortized compaction. The aging
rank is factored as `priority - arrival/age` because the omitted `now/age` term
is common to every waiting request. This preserves stable ties and starvation
prevention without re-sorting the full queue on every schedule. Recompute
preemption retains front-of-tie placement, and KV allocation still blocks at
the selected head without skip-ahead. Reproduce the 100k A/B measurements with
`uv run python bench/scheduler_queue_bench.py --requests 100000 --repeats 5`.

### D7 — Open WebUI + frontier bench

`deploy/compose/docker-compose.webui.yaml` points Open WebUI at the internal
Kairyu endpoint `http://kairyu:8000/v1` and mounts a standalone
`deploy/compose/config.yaml` that serves the keyless CPU-safe mock model
`default`. `scripts/webui_smoke.sh` first validates literal bind sources and the
rendered internal endpoint, then starts only the `kairyu` service and asserts
bounded readiness, exact `/v1/models == ["default"]`, and one non-streaming
completion. CI runs this after the existing Compose drill and deliberately does
not pull or browser-test the large mutable Open WebUI image.
`bench/frontier_compare.py`: multi-target harness (kairyu vs OpenAI vs
Anthropic endpoints), method block (same prompts, N trials, TTFT/TPOT/
quality-proxy), scoreboard JSON+md; offline unit test with mock targets.

## 3. Non-goals

- Real VLM inference (GPU); online bandit for tiers; billing/invoicing.
- Cross-node tenant state (single-gateway token buckets; the distributed
  limiter is a G6 note).
- Autoscaler EXECUTION (decision logic only).

## 4. Verification

- Streaming orchestrator: SSE event sequence (status* delta+ result) with
  usage totals == sum of stage usages; trace opt-in only with the header.
- Tenancy isolation + ledger reconciliation gates (D3).
- OpenAI SDK round-trips: responses.create (+previous_response_id chain),
  embeddings.create.
- Vision: content-parts accepted, image parts rejected cleanly on
  non-vision engines.
- F5: priority ordering + aging; SLO shed under synthetic overload
  (deterministic fake clock); autoscaler hysteresis table test.
- Bench: mock-target run produces the scoreboard schema.

## 5. Review record (binding amendments)

- **A1 (D1)**: usage is currently DROPPED at three layers — thread it: usage
  fields on ConductorResult/OrchestratorResult/MoAResult, accumulate in
  Conductor._generate; the stream contract is "usage read from the LAST
  partial" (MockBackend final-only; KairyuBackend every-partial — both fit).
- **A2 (D1, CRITICAL)**: status keep-alives must NOT be data: lines (the
  OpenAI SDK parses every data: payload as a chunk) — SSE comment lines;
  trace rides an explicit optional ``kairyu_trace`` field (no extra=allow).
- **A3 (D1)**: only the AUTO call-site usage=None fallback is removed;
  _wire_usage's approximation branch stays for third-party backends.
- **A4 (D1/D2)**: MoA is currently unreachable from Orchestrator.run() —
  wire a ``moa`` route (tier option); run_chat receives the PRE-RENDERED
  prompt string (app.py renders; orchestrator engines have no template
  knowledge) plus messages only for future vision routing.
- **A5 (D1)**: Conductor final unit streams only when verifier-free (refine
  regeneration would invalidate streamed deltas); else buffered.
- **A6 (D3)**: AuthMiddleware stores the matched key hash in scope state;
  TenantLimitMiddleware runs INSIDE auth (added before it) so 401 wins over
  429 and unauthenticated requests never drain buckets; keyless mode →
  tenant "default".
- **A7 (D3, amended by Issues #90 and #213)**: ledger = O_APPEND single-writer JSONL
  (atomic-rename doesn't fit appends); writes happen in handlers, stream
  generators, and successful batch consumers (middleware can't see usage).
  `record` performs only bounded `put_nowait` admission (4,096 records by
  default); one writer thread owns the append handle and flushes each batch of
  at most 128 records. Queue saturation raises `AuditQueueFull` immediately
  rather than blocking a request thread or silently dropping billing data.
  Append/flush/close failures become sticky `AuditWriteError`s and surface on
  the next admission or ordered barrier; no failed write is reported durable.
  `create_app` closes the writer in an outer lifespan after caller/builder
  cleanup finishes or raises, and normal close drains all accepted records
  before closing the handle. `/admin/usage` waits for the ordered flush barrier
  on a worker thread before scanning, so it observes every prior accepted row
  while other requests continue. Flush means runtime buffers reach the OS and
  readers; it is not an `fsync` or a power-loss guarantee. Aggregation validates
  each non-whitespace record independently: a malformed final line without a
  newline is a truncated tail (skip + warning), complete malformed lines are
  corruption (skip + per-line error), and all valid rows still count.
- **A13 (D3, Issue #199)**: generic HTTP request counters cannot reconcile
  usage because they intentionally include rejected/unknown requests.
  `kairyu_usage_requests_total{tenant}` and
  `kairyu_usage_tokens_total{tenant,type}` therefore increment at the same
  exactly-once metering seam as ledger admission across sync/stream chat,
  completions, AUTO, Responses, embeddings, and batch lines. Startup restores
  their tenant totals from the app-owned ledger; model is deliberately not a
  label because the aggregate ledger API has no model dimension and tenant
  reconciliation must survive restart without inventing attribution.
- **A14 (D3, Issue #194)**: gateway ledgers remain independently owned and
  fleet aggregation is an offline boundary. Cached prompt tokens flow from
  every generation path (including Conductor/MoA and batch) into ledger,
  `/admin/usage`, and `kairyu_usage_tokens_total{type="cached"}`. The committed
  fixed replay reconciles independent request logs and three gateway ledgers
  with 0.0% maximum error; the committed A/B selects local async writes by
  measured request-path p99 and throughput against a favorable same-host
  synchronous shared-store candidate.
- **A15 (D3, Issue #204)**: every new usage row stores
  `cached_tokens + uncached_tokens == prompt_tokens`; old rows derive the
  missing uncached count. DeploymentSpec validates a versioned blended price
  sheet, cached-input and tenant discount fractions, known tenants, and the
  required ledger. Invoice export uses Decimal half-even rounding at six
  currency decimal places and a reader-stable ledger snapshot; streaming,
  Responses, embeddings, and batch executions reconcile into the CSV while
  corrupt records produce no partial invoice.
- **A8 (D4)**: Responses usage names are input_tokens/output_tokens/
  total_tokens; output item = {type: message, role: assistant, status,
  content: [{type: output_text, text, annotations: []}]}; instructions
  supported; STREAM DESCOPED (typed response.* event protocol is its own
  milestone — recorded).
- **A9 (D4)**: /v1/embeddings must support encoding_format=base64 (the SDK
  DEFAULT); dev dep bumped to openai>=1.66 (client.responses exists from
  1.66).
- **A10 (D5)**: content-parts touch ChatMessage.content typing, render_chat,
  _normalize_message flattening, and the shared render_prompt (batch worker
  parity).
- **A11 (D6)**: injectable clock + arrival timestamps; EFFECTIVE priority
  computed at sort time (EngineRequest frozen); fairness restated (highest
  priority at head blocks on KVCacheFull; no skip-ahead); priority plumbing
  descoped to engine-level (HTTP→priority mapping is tenant config, G6);
  TTFT predictor uses GATEWAY-observable signals (in-flight count + observed
  TTFT EMA — engine internals invisible through ZMQ/vLLM backends).
- **A12 (D4)**: embeddings use a model-ID → backend registry, not one anonymous
  global backend. IDs are discoverable, resolve before validation/execution,
  cannot collide with chat or orchestration IDs, and are the only identities
  admitted to response, metric, and ledger labels; limiter charging occurs
  only after resolution. Misses use the shared 404 `model_not_found` response
  and record no usage.
