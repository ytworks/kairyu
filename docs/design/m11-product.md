# M11 Design: Fugu-Class Product Surface + Tenancy — CPU Complete

Status: **Implemented** (2026-07-03). Reviewed (1-reviewer panel with
file/line evidence + OpenAI SDK verification, 2026-07-03; §5 binding).
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
/v1/*; 429 with retry-after), usage ledger — JSONL append via the
BatchStore atomic-rename pattern, one record per request (tenant, model,
prompt/completion tokens, ts) — and `GET /admin/usage?tenant=` aggregation.
Isolation gate: tenant A at its limit 429s while tenant B proceeds; ledger
totals reconcile with returned usage to <0.1%.

### D4 — `/v1/responses` (subset) + `/v1/embeddings`

Responses: `POST /v1/responses` accepting `model`, `input` (string or
message array), `previous_response_id`, `stream`; a `ResponseStore`
(protocol + in-memory impl) persists response items so
`previous_response_id` reconstructs context; OpenAI SDK
`client.responses.create` round-trip test. Embeddings: `EmbeddingBackend`
protocol (`embed(texts) -> list[vector]`), `MockEmbeddingBackend`
(deterministic hash-based vectors), `POST /v1/embeddings` with usage.

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

### D7 — Open WebUI + frontier bench

`deploy/compose/docker-compose.webui.yaml` (Open WebUI pointed at the
gateway; smoke asserts the container config renders — no image pull in CI).
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
- **A7 (D3)**: ledger = O_APPEND single-writer JSONL (atomic-rename doesn't
  fit appends); writes happen in handlers/stream generators (middleware
  can't see usage); batch-worker executions are NOT metered in v1 (recorded).
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
