# M20 Design: OpenAI Responses API Full Compatibility — Proposed

Status: **Proposed** (2026-10-05). Phase 0 (WP-00–05) implemented (parts of D1,
D3/D7/D9, D10, D22), Phase 1 refactors 06/06b/12a and WP-07; WP-53 closes M20.
Milestone: M20 (roadmap Track P, P-C2 reopened; goal G6)
Date: 2026-10-05
Depends on: m11 D4 (superseded progressively, §10), m7 D5, m9 D1–D6,
m10 D6/A37/A40, frontier-native-runtime FN-D3/FN-D9, ECO-D1 (unchanged).
Sources: read-only audit workflow wf_3e95ac80-c0d (80 verified gaps plus 31
verifier-added gaps), design workflow wf_ac47de7b-27a (final plan with the
completeness, boundary-architecture and feasibility-risk reviews applied), and
the owner decisions of 2026-10-05.

## 1. Context

`POST /v1/responses` exists (`kairyu/entrypoints/server/responses_service.py`,
m11 D4 with the #201/#530 amendments) and works for Codex 0.147 and
openai-python 2.44 only in narrow configurations. A 2026-10-04 audit of
main@e60404af against the pinned OpenAPI document, openai-python v3.24, codex-rs
rust-v0.160.0 (0.162-alpha advisory), Open Responses 2026-04-24, and
vLLM/SGLang/Ollama/llama.cpp found six P0 gaps that break Codex today:

1. an omitted `max_output_tokens` capped output at 1024 tokens (Codex never
   sends it and retries `incomplete` five times);
2. keep-alives were SSE comments, which never reset Codex's 300 s idle timer,
   and the live and AUTO relay paths sent no heartbeat at all;
3. no `context_length_exceeded`, so Codex cannot auto-compact;
4. live/indexed `web_search`, which Codex declares under full access, got a
   terminal 400;
5. `input_image` and `view_image` results got a 400 that poisons every later
   turn;
6. requests without an effort (the Codex default) leak `<think>` into text and
   tool parsing.

Structural gaps: reasoning is discarded in both directions; tool streams are
fully buffered; the store is process-local and keeps only items; retrieve,
delete, cancel, `input_items`, `input_tokens`, `compact`, background mode,
Conversations and WebSocket mode are missing; hosted tools are rejected; error
bodies are not OpenAI-shaped (`{detail}`, no `param`); the legacy flatten
collapses a conversation into one user message; there is no schema gate and the
Codex smoke is stale (0.145/0.147). Externally, Codex moved to 0.160 (since
0.147 it no longer calls `POST /responses/compact`, compacting remotely only
through `compaction_trigger`, and always sends `store:false`) and openai-python
v3.0 is a breaking release (httpx2).

## 2. Goal and owner decisions

Goal: Codex CLI (latest stable and a version matrix), openai-python and
openai-node, and the OpenAI Agents SDK use every Responses feature against
Kairyu **without modification**, proven by the pinned schema gate, the Open
Responses suite and real-client gates.

| # | Decision (2026-10-05, binding) |
|---|---|
| 1 | **Scope is everything:** the P0 fixes; every endpoint (create, retrieve incl. stream resume, delete, cancel, `input_items`, `input_tokens`, `compact`); streaming, error and usage conformance; **plus background mode, the Conversations API, WebSocket mode and hosted-tool executors**. |
| 2 | **Reasoning:** raw reasoning becomes `reasoning` items (`content[reasoning_text]`, `response.reasoning_text.*`); a summary is mirrored only when requested; `encrypted_content` is sealed (AES-GCM, tenant-bound) and restored on replay into the template's `reasoning_content`; AUTO exposure stays DSL policy. |
| 3 | **Hosted tools:** default accept-and-drop (echoed, hidden from the model); a deployment can switch to reject (typed 400, `param:"tools[i]"`); with a configured executor the tool runs server-side and emits the spec's items and events. |
| 4 | **Framework authorization:** (a) a structured chat prompt kind; (b) the lossless effort vocabulary including `none`, plus `GenerationUsage.reasoning_tokens`; (c) a `ResponseStore` protocol with Postgres and a generalized background worker. Each still carries an admission row (§8). |
| O-1 | Every further shared-contract change in the admission table (§8) is **approved**. |
| O-2 | Transient overload and transient tenant quota on `/v1/responses*` answer **503 `slow_down` + `Retry-After`**; a refusal mid-response is in-band `rate_limit_exceeded`; only a reservation that can never fit is 429. Chat and Messages keep 429. |
| O-3 | The OpenAI stateful probe (SP-5) is run by the owner with their own key; implementation supplies the probe script. Until then the documented defaults stand as open points (§12). |
| O-4 | Tenants without an allowance for `priority`/`fast`/`ultrafast` are **downgraded to `default` and the executed tier is echoed** (no 400). |
| O-5 | **In scope:** webhooks (WP-55), URL input fetch (WP-56), `input_file` document extraction through a `DocumentExtractor` port (new WP-57). **Won't do:** example reference hosted services (former WP-52), a `programmatic_tool_calling` executor, Lark grammar enforcement, a served Codex `/models` route, `/alpha/search`, served-model aliases. |

## 3. Reference contract

| Artifact | Pin | Use |
|---|---|---|
| OpenAPI | `openai/openai-openapi@13fa6e7ab9` (full `13fa6e7ab9301b00c03af2a5d2f584e7a9b84391`; openapi 3.1.0, info 2.3.0) | `$ref` closure of the Responses and Conversations roots, vendored by `scripts/vendor_openai_schema.py` into `tests/contracts/openai/responses-schema@13fa6e7ab9.json` |
| openai-python | v3.24.0 | dev dependency `openai[realtime]>=3.24,<4` with `_strict_response_validation=True` (WP-13; WP-01 makes the 2.44 harness strict) |
| Agents SDK | openai-agents 0.23.x | HTTP and WebSocket smokes (WP-53) |
| openai-node | latest at WP-53 | nightly smoke |
| Codex | rust-v0.160.0 blocking; 0.147.0 and 0.153.4 only for shapes whose wire differs; latest stable and alpha advisory | fixtures and the extension inventory generated from codex-rs serde types per pinned tag (WP-02) |
| Open Responses | spec 2026-04-24, suite `92c12d96d7` | applies where it does not contradict OpenAI; divergences recorded in `tests/contracts/divergences.toml` |

## 4. Principles

1. **Strict and complete.** Request bodies use `extra="forbid"`; a truly
   unknown field is 400 `unknown_parameter` with `param`. Accepted: the pinned
   surface, the generated Codex extension inventory (each entry with defined
   semantics), the Kairyu extension `priority`, and the v3.24 additive fields
   (preserved verbatim in echo, store and replay). Operator escape hatch
   `ResponsesConfig.unknown_nested_fields: reject|drop_and_log` (default
   `reject`, never top-level, dropped fields logged and counted). An unbuilt
   feature returns a typed 400 with `param` until its WP lands.
2. **Truthfulness.** Never emit what did not execute: hosted events only when
   an executor ran; usage is real on every terminal. **Call finality:** once a
   client-executed call's `output_item.done` is sent, nothing turns the
   response `incomplete` or `failed` (a later length cut, generation error or
   persistence failure ends `completed`), so Codex never runs a tool twice.
3. **Error envelope.** Every `/v1/responses*` and `/v1/conversations*` error is
   `{error:{message,type,param,code}}`, including 422→400, 404, 405, malformed
   JSON, 413 and middleware errors. `/v1/messages` keeps the Anthropic shape.
4. **Error placement** (`ClassifiedError.placement`): schema/surface errors are
   400 before the stream. Client-derived prompt overflow is unary 400
   `context_length_exceeded` `param:"input"`; on a stream (every path, incl.
   AUTO pre-dispatch and ZMQ preflight) it is `created` → `in_progress` →
   `error` → `response.failed{context_length_exceeded}` with no dispatch,
   metering or storage. Internal-stage overflow is `server_error`. Unknown
   `previous_response_id` is 400 `previous_response_not_found`.
5. **Heartbeat.** Every Responses SSE path repeats `response.in_progress` (a
   data event) after 15 s without data; compaction sends
   `response.compaction.compacting` at most every 30 s; a queued background
   job sends `response.queued`; WebSocket uses the same data heartbeat.
   `data: [DONE]` follows the terminal event (SP-4 confirms client tolerance).
6. **Defaults.** An omitted `max_output_tokens` (and compaction) means the
   remaining context (Chat #496 semantics). Echoes are `null` when omitted and
   the applied value when explicit. Responses bodies cap at 64 MiB, and
   decompression has its own 64 MiB cap.
7. **Framework boundary.** Every change to `kairyu/` outside the Responses L3
   contract has an admission row (§8). Example-owned: templates and their
   declared reasoning/developer capabilities, effort maps, vLLM flags, search
   providers and sidecars, hosted services, MCP allowlists, tool-description
   text, compaction-prompt overrides.
8. **Repository rules.** New modules 200–400 lines (≤800); no net growth of
   any touched file over 800 lines (`wc -l` per WP). Proposed exception,
   pending owner acknowledgement (§12 open points): a design record over 800
   lines may grow by the dated amendments this plan requires, since the
   never-rewrite rule forbids offsetting them; the WP reports that growth.
   Cross-boundary records are frozen (documented single-owner state machines
   allowed); the CLAUDE.md test policy (necessary and sufficient, one schema
   gate, A→C, counts reported per WP); PROGRESS and D-ID rules.

## 5. Deliberate deviations

- **D-a** `max_output_tokens` echoes `null` when omitted, the applied cap when
  set (OpenAI echoes `null`; AUTO cannot know the effective value early; a
  number invites clients to replay it as an explicit cap).
- **D-b** Stream failure order is `in_progress` → `error` → `response.failed`
  (Codex ignores `error`; Open Responses expects it).
- **D-c** Obfuscation is opt-in (`stream_options.include_obfuscation:true`).
- **D-d** `previous_response_not_found` is 400 with `param` (VS Code BYOK
  recovers only on 400; Open Responses uses 400; SP-5 confirms).
- **D-e** WebSocket ships disabled until WP-47: uvicorn runs `ws="none"`, so an
  upgrade is plain HTTP and `GET /v1/responses` answers 426; once enabled, an
  at-capacity upgrade is denied with 426 and Codex falls back to HTTPS.
- **D-f** `truncation:"auto"` needs exact token counting; AUTO models and
  backends that cannot count get 400 `param:"truncation"`.
- **D-g** Backpressure on `/v1/responses*` is 503
  `service_unavailable_error`/`slow_down` + `Retry-After`/`retry-after-ms`
  (O-2), not the spec's `server_is_overloaded` (Codex: not retried) or 429
  (Codex: terminal).
- **D-h** Call finality wins over the Open Responses rule "an incomplete item
  must be last": Codex dispatches at `output_item.done` and retries
  `incomplete`/`failed` turns.
- **D-i** `incomplete_details.reason:"interrupted"` is a Codex extension
  (WebSocket `response.interrupt`).
- **D-j** Compaction rendering on every path: for the latest Kairyu compaction
  item at input position k, items before k render only if they are user or
  developer messages, every other earlier item is dropped (the seal's summary
  covers it), and the compaction item renders in place as summary plus bridge.
  This matches `/compact` output, Codex v2 retained messages and stateless
  `context_management` replays without re-growing the prompt.

## 6. Decisions

Each D-ID is Proposed until its last WP lands; partial implementation is noted.

**D1 — Reference contract and conformance infrastructure** (WP-01, 02, 02b,
05, 08a, 13, 14, 53; *WP-01 in progress; WP-02, 02b, 05 implemented*). The §3
pins are the contract. WP-01 vendors the OpenAPI closure and gates every test:
an ASGI recorder wraps the app of every in-process test transport,
`ContractValidator` (Draft 2020-12)
validates each `/v1/responses*` and `/v1/conversations*` body, each SSE event by
its `type` (unknown types fail) and each error body against `ErrorResponse`,
and an unmapped route under those prefixes fails. `tests/contracts/divergences.toml`
allowlists current behavior, one entry per gap with its owner WP;
`CONTRACT_STRICT=1` fails a full run on stale entries or on zero validated
items. The same recorder captures every route's wire bytes when
`KAIRYU_WIRE_CAPTURE=<dir>` is set (DoD #9). Its comparison
(`python -m tests.contracts.wire_capture diff`) masks only run-to-run volatile
values (ids keep their prefix and are numbered per test), skips the tests it
lists as nondeterministic, and pairs tests moved between modules with
`--key test` (WP-06). Later WPs add the Codex extension inventory and fixture
replay (02), the ScenarioBackend test engines (02b), the Codex catalog and
nightly matrix with a 7-day fixture refresh procedure (05), the strict v3.24
SDK harness (13), Open Responses CI (14) and client smokes (53).
*2026-10-05, WP-02:* `tests/fixtures/codex/rust-v<ver>/` holds one fixture per
Codex wire contract (request, scripted turns, expected outcome, gap IDs,
provenance), recorded via `scripts/codex_gate/record_proxy.py` or derived from a
recording by data edits cited from codex-rs, and replayed by
`test_codex_fixture_replay`; a later WP's behavior is `xfail(strict=True)`.
`extension_inventory.py` writes each tag's `extensions.json`, the list WP-08a
accepts; 0.147/0.153.4 cover only §7 wire differences. **Refresh** (7 days after
a Codex stable): re-record, `promote` (keeps the replay, re-applies derived
edits), regenerate the inventory; multi-turn shapes wait for WP-05's launcher.

**D2 — Package architecture and single pipeline** (WP-06, 12a, 12b, 17a, 18).
`responses_service.py` becomes `kairyu/entrypoints/server/responses/` (routes,
frozen `ResponsesDeps`, schema, resolve → frozen `EffectiveRequest`, envelope,
canonical/to_chat, emitter, engine/auto paths, `ResponsesExecution.run(prepared,
scope, sink)`, `RoundDriver`). Shared L3 modules: `stream_util`,
`error_classifier`, `engine_admission`, `segment_stream` (one splitter →
`ToolStreamScanner` → gates → outcome pipeline for Messages and Responses),
`tenant_admission`, `request_scope`, `auto_dispatch`, one frozen
`ResponsesConfig`. Refactor-only WPs pass the wire-capture gate.

**D3 — Emitter contract** (WP-03, 17a–c, 18, 38b; *heartbeat interim part
implemented in WP-03*). One state machine owns `sequence_number` and
`output_index`; foreground start is `created` → `in_progress`; every item
carries `status`; reasoning, text and tool segments follow the plan's event
order; tool calls stream progressively; call finality; terminal rule
(`incomplete{max_output_tokens}`, `content_filter` → `incomplete`, required tool
choice with no call → `failed{server_error}`, error → `error` + `failed` with real
usage, else `completed` with `completed_at`), then `[DONE]`; data heartbeats on
idle; obfuscation opt-in. WP-03 already repeats `response.in_progress` on every
path; WP-18 makes the emitter the only heartbeat owner.

**D4 — Progressive tools and in-stream gates** (WP-17b, 12b, 29). Function
calls emit `added` + `function_call_arguments.delta*` as generated;
`parallel_tool_calls:false` and `tool_choice` gate inside the stream; preamble
text closes with `phase:"commentary"`; the AUTO raw tool stream becomes an
explicit `raw_tool_stream` parameter (12b); vLLM deltas relay progressively (29).

**D5 — Reasoning output and replay** (WP-16, 21). Reasoning spans come only
from declared configuration (template/deploy `reasoning:{open, close,
starts_open, replay_field, disable_kwarg}`, validated at load), split by one
shared splitter independent of effort. Items carry `content[reasoning_text]`;
summaries are mirrored only on request. Replay precedence: stored item → kst2
(same tenant and model) → plaintext content only without a Kairyu seal or with a
same-model seal → dropped. `x-reasoning-included: true`.

**D6 — Sealed items v2** (WP-20, 21, 36). `kst2.` tokens: header
`ver ‖ purpose ‖ kid ‖ issued_at ‖ salt`, per-token HKDF subkey, AAD binds owner
and header, plaintext binds the served model, 4 MiB cap and optional max age;
`kcp1.` stays decode-only. A secret is required at any replica count (release N
warns, N+1 fails) unless `sealing.ephemeral: true`; Helm generates and persists
one; rotation is two-phase. Compaction fails closed (400
`invalid_encrypted_content`); foreign reasoning blobs are ignored.

**D7 — Request surface and defaults** (WP-03, 08a–c, 11; *output-cap default
implemented in WP-03*). Typed schema with extensions from the inventory,
v3.24 additive fields, nullable defaults, metadata bounds, typed 400s for
unbuilt features, hosted-only params (`prompt`/`moderation`/`multi_agent` 400,
`access_programs` echoed), unknown model 404 `param:"model"`. Full echo envelope
with Open Responses requirements and verbosity per C26. Off-loop parse and a 64
MiB body cap. Service tiers: `flex` → batch class, `scale` → default,
unallowed priority tiers downgraded and echoed (O-4). WP-03 removed the 1024 and
4096 caps (remaining-context semantics, echo `null`).

**D8 — Input items and client tools** (WP-09, 22, 24, 25, 27). Each
(reasoning?, message?, calls*) run becomes one assistant turn; instructions
become one leading system message; developer content merges into it unless the
template declares `developer_role`; `configuration_update`, `agent_message`,
refusals and call linkage are canonicalized; D-j applies. Custom/freeform tools
(grammar verbatim in the description), shims for `local_shell`, `shell`,
`apply_patch`, `computer`, client `tool_search` and `additional_tools`.
`input_image` (data URL, `file_id` via the `FileReader` port) and `input_file`
(`text/*`; other types through the `DocumentExtractor` port of WP-57).

**D9 — Hosted-tool policy** (WP-03, 23; *drop default for `web_search`
implemented in WP-03*). Modes drop (default) / reject / execute;
`HostedToolPolicy.public_view` redacts MCP secrets and is the only source for
echo and persistence; `defer_loading` and `allowed_callers` rules; hosted
`tool_choice` routing; hosted history items dropped.

**D10 — Error classification** (WP-04, 07 *implemented*; pending: WebSocket
frames, Responses SLO shed 17c, follow-up refusals 18). L1 overflow error and
`resolve_output_budget`; fixed-message upstream classification; L3
`error_classifier` (`surface`, `placement`, `param` always); scoped envelopes
incl. middleware, framework 422/404/405; O-2 table by path (C25, m7 D5 amendment).

**D11 — Usage details** (WP-08b, 19, 31). `reasoning_tokens` (native, ZMQ,
upstream `completion_tokens_details`), AUTO public cached/reasoning tokens,
`cache_write_tokens:0`, exact counts before word-count fallbacks, logprobs
partitioned away from reasoning.

**D12 — Response store contract** (WP-32, 33). `kairyu/response_store`
protocol with a memory backend (incremental storage, `chain_root`, immutable
bytes, byte LRU, TTL, tenant cap); persistence matrix with call finality
(`store:false` stores nothing, finalize commits before the terminal frame,
chainable statuses completed/incomplete); `public_view` projection; retrieve,
delete, cancel, `input_items`, `item_reference`.

**D13 — PostgreSQL backend** (WP-34a, 34b). Shared
`kairyu/storage/postgres_common.py` (also adopted by the batch and async
stores), versioned migrations, pool justification against m10 A37 with SP-7
data, Helm `responsesPostgres`, retention script, three-gateway kind gate.

**D14 — Token counting and truncation** (WP-35). One `token_count.py` shared
with Messages `count_tokens`; `POST /v1/responses/input_tokens`;
`truncation:"auto"` fitter per D-f.

**D15 — Compaction family** (WP-18, 36). Shared summarizer with a
deployment-overridable prompt; `POST /v1/responses/compact` returns the input's
user/developer messages in clear plus one sealed summary item; `compaction_trigger`
and `context_management` as round-0 strategies with follow-up admission;
failure codes per C4 (input overflow or starved budget →
`context_length_exceeded`; empty or malformed summary → `server_error`; an
emitted compaction item is never followed by `incomplete`, amending #531).
Expected Codex behavior (remote v2 retries at most twice) is recorded with the
live run. *2026-10-05, WP-05 (live run):* remote v2 trims history to the catalog
window before sending and never retries an in-band `context_length_exceeded`
(`compact_remote_v2*.rs@rust-v0.160.0`): an overstated window fails it (WP-36).

**D16 — Background mode** (WP-37, 38a–c). One `EndpointExecutor` registry for
async and batch; per-job scheduling class; queued start order; the event log as
the single sequence authority (CAS append) with resume via
`GET ?stream=true&starting_after`; fenced writes, terminal + finalize in one
transaction, drain, read-repair; `store:false` jobs kept 600 s.

**D17 — Conversations API** (WP-39). Eight routes, memory and Postgres stores,
`conversation` exclusive with `previous_response_id`, append inside finalize,
D-j rendering, a documented delete-semantics deviation.

**D18 — Batch `/v1/responses` lines** (WP-40) through the shared executor
registry; lines reject `stream`, `background` and `conversation`.

**D19 — WebSocket transport** (WP-15, 44–47, 54). `ws="none"` while disabled
(D-e); then authenticated ingress, per-create admission, connection cache,
lanes/`stream_id`, 60-minute lifetime, interrupt (D-i) and a steering stub;
default on after acceptance (WP-47); steering and prewarm only with measured
benefit (WP-54).

**D20 — Hosted mechanism** (WP-48–51, 56). `HostedToolExecutor` protocol and
registry (DI and config), `RoundDriver` strategy with budgets, `EgressPolicy`
(DNS pinning, private ranges blocked, redirect re-check, byte caps) before any
executor, per-tenant budgets and metering, spec builders for every hosted kind,
reference `web_search` and MCP executors, one generic service adapter
(`POST {base_url}/v1/hosted/{kind}:execute`). No provider bindings; ECO-D1 is
not widened.

**D21 — Transport hardening** (WP-17c, 41–43). SLO admission on Responses,
request-id always set, gzip core and zstd extra (415 without it), CORS optional,
disconnect cancellation for Chat unary/Messages/Responses, affinity precedence
`prompt_cache_key` > `session-id` > `X-Session-ID` > conversation > chain root >
`user` (tenant-bound hash) plus prefix fingerprint.

**D22 — Codex catalog and provider docs** (WP-05, 53; *WP-05 implemented*).
`scripts/codex_model_catalog.py` emits `model_catalog_json` (context window,
auto-compact limit ≈0.9·ctx, modalities, `apply_patch_tool_type`, honest
reasoning levels); provider TOML, `stream_idle_timeout_ms`, `web_search` modes,
compaction gating, auxiliary slugs, sealing secret and tenant sizing are
documented. *2026-10-05, WP-05:* matrix passes on macOS; first Linux CI run pending.

### Design conflict resolutions referenced above

| # | Choice |
|---|---|
| C1 | One sealed format `kst2.` with per-token HKDF subkeys and model binding; `kcp1.` decode-only |
| C2 | `encrypted_content` always on create and WebSocket `response.create`; include-gated on GET, `input_items` and conversation items |
| C3 | Unknown `previous_response_id` → 400 + param (D-d) |
| C4 | Compaction failures: overflow/starved budget → `context_length_exceeded`; empty/malformed summary → `server_error`; no `incomplete` after an emitted compaction item |
| C5 | One `error_classifier.py` with `surface` and `placement` |
| C6/C7/C8 | One shared splitter in `kairyu/reasoning/span.py` fed only by declared spans; the `<think>` heuristic stays only for undeclared models with reasoning intent |
| C9 | Effort `"disabled"` is a Codex alias of `none` at the Responses boundary; numeric/custom → 400 |
| C10 | `x-reasoning-included` always `true` |
| C11/C12 | One `RoundDriver` (compaction and hosted are strategies); WP-18 uses the existing `chat_dispatch`, WP-12b swaps in `RequestScope` |
| C13 | One `json_field_stream.py` decoder |
| C14 | `ci_` prefix belongs to `code_interpreter_call` |
| C15 | `generate:false` is state only; prefill warm-up deferred to WP-54 |
| C16/C17/C18 | One catalog generator; one `test_codex_fixture_replay`; one `test_hosted_tool_policy` |
| C19 | `truncation:auto` is owned by the state fitter (D-f) |
| C20 | gzip core; zstd and `websockets` as extras until enabled; uvicorn `ws="none"` while WebSocket mode is off |
| C21 | `created_by` on compaction items is not emitted |
| C22 | Sealing secret required at any replica count (staged), Helm-generated, two-phase rotation; direct `create_app` keeps the ephemeral key |
| C23 | One frozen `ResponsesConfig`; the existing `responses_compaction_secret_env` name is kept |
| C24 | Affinity precedence as in D21, `"aff-"+sha256(owner‖key)` |
| C25 | Backpressure rendered by path dialect in middleware |
| C26 | Verbosity: template kwarg → upstream field → deployment-configurable hint (default = today's strings) |
| C27 | Hosted executors live under `responses/hosted/`; no top-level hosted package, no provider bindings |

## 7. Work packages

Sizes: S < 1 day, M 2–4 days, L 1–2 weeks. Lanes: A Responses core, B shared
L1/L2, C state, D conformance/transport, E hosted. Each WP is one PR meeting the
DoD below. Merge order for hot files: `app.py` 12a → 07 → 08c → 16 → 19 → 12b
→ 41 → 42; `messages_service.py` 03 → 04 → 12a → 17a → 16 → 10 → 35;
`openai_backend.py` 04 → 06b → 16 → 10 → 19 → 26a → 28 → 29.

**Definition of done (every WP).** (1) Regression tests first, seen failing
(at gate level: delete `divergences.toml` entries and flip `xfail(strict=True)`
fixtures in the WP). (2) Base/head collected counts with the same command and a
rationale per test area. (3) `CONTRACT_STRICT=1` schema gate green with no stale
entries. (4) New modules ≤800 lines; `wc -l` before/after and zero net growth
for every touched file over 800 lines (the proposed design-record exception of
Principle 8 aside). (5) Cross-boundary records
immutable. (6) m20 D-ID status and supersession rows updated; PROGRESS only for
D-ID, milestone or blocker changes. (7) Enabled capabilities flipped in the Codex
catalog. (8) `ruff check .`. (9) Refactor-only WPs (06, 06b, 12a, 12b, 37):
Chat, Messages and Responses wire captures identical at base and head.

| Phase | WP | Size/lane | Scope | Depends | Status |
|---|---|---|---|---|---|
| 0 | 00 | S docs | PROGRESS archiving (Current Status ≤80 lines) | – | Done |
| 0 | 03 | S A | Codex P0 hotfix: no 1024/4096 caps, `stream_util`, data heartbeats on every path, `web_search` accept-and-drop, tenant-budget warning | 00 | Done |
| 0 | 01 | S/M D | Schema gate (vendored closure, `ContractValidator`, ASGI recorder + wire capture, `divergences.toml`, strict SDK on 2.44); this record | – | Done |
| 0 | 02b | M D | `tests/support/scenario_backend.py`, `fake_vllm_upstream.py` | – | Done |
| 0 | 02 | M D | Codex fixture corpus, extension inventory, one replay test | 01, 02b | Done |
| 0 | 04 | M B | Overflow classification (L1 typed error, upstream classifier, L3 classifier with placement, AUTO in-band) | – | Done |
| 0 | 05 | M D | Codex catalog generator, nightly matrix + `--live`, provider docs | 02, 03, 04 | Done |
| 1 | 06 | M A refactor | Package split, tests moved to `tests/server/responses/` | 03, 04 | Done |
| 1 | 06b | M B refactor | Extract `engine/admission.py`, `engine/openai_payload.py`, `chat_render.py` | 01, 04 | Done |
| 1 | 12a | M B refactor | `engine_admission.py` for Messages and the Chat engine path | 01 | Done |
| 1 | 15 | S D | uvicorn `ws="none"`, reliable 426 | 06 | Planned |
| 1 | 07 | M A | Error contract: `param` everywhere, scoped handlers, middleware via classifier, O-2 table | 06 | Done |
| 1 | 17a | L A | Shared `segment_stream`; Responses emitter and engine path | 06, 07, 12a | Planned |
| 1 | 17b | M A | Progressive calls, in-stream gates, terminal rule, call finality, `phase` | 17a, SP-4 | Planned |
| 1 | 17c | S/M A | SLO admission, obfuscation opt-in, `content_filter` → incomplete | 17a, 07 | Planned |
| 1 | 18 | L A | AUTO `auto_path`, `RoundDriver`, `reserve_followup`, compaction round-0 strategy; legacy paths deleted | 17b | Planned |
| 2 | 08a | M/L A | Request schema, typed rejections, `ResponsesConfig` | 18, 02 | Planned |
| 2 | 08b | M A | Full echo envelope, verbosity (C26), `include` | 08a | Planned |
| 2 | 08c | S A | Off-loop parse, 64 MiB body cap | 08a | Planned |
| 2 | 09 | M A | Canonical items, chat translation, developer fallback, D-j | 08a | Planned |
| 2 | 10 | M B | Lossless effort with default legacy fold | 08a, 16 | Planned |
| 2 | 11 | S B | Service tiers (O-4), Chat echo | 08a | Planned |
| 2 | 13 | M D | SDK v3.24 live-server harness, strict validation | 01, 15 | Planned |
| 2 | 14 | S D | Open Responses CI | 01, 02b | Planned |
| 2 | 16 | M B | Declared `ReasoningSpan`, unified splitter | SP-3, 17a | Planned |
| 2 | 19 | M B | `reasoning_tokens`, AUTO public usage | 16 | Planned |
| 2 | 20 | M C | kst2 codec, key ring, staged secret preflight, Helm | 06 | Planned |
| 2 | 21 | L A | Reasoning items/events, sealed replay | 10, 18, 19, 20 | Planned |
| 3 | 22 | L A | Tool registry, custom tools, `allowed_tools` | 17b | Planned |
| 3 | 23 | S/M E | Hosted policy, `public_view`, dependent-tool rules | 22, 08b | Planned |
| 3 | 24 | L A | Client shims, `tool_search`, `additional_tools` | 22, 23 | Planned |
| 3 | 25 | M A | `input_image`, `input_file` (`text/*`), `FileReader` | 09 | Planned |
| 3 | 26a | M/L B | L1 `ChatPrompt` kind and capabilities | SP-11, 16, 06b, SP-1 | Planned |
| 3 | 26b | M B | ChatPrompt routing text, metering, admission | 26a | Planned |
| 3 | 27 | M B | `ChatRenderPolicy`, `upstream_chat_models`, tool-output images (P0 #5 closed) | 26b, 25 | Planned |
| 3 | 28 | M/L B | `tool_grammar.py`, vLLM strict fallback | 16, 22, SP-2 | Planned |
| 3 | 29 | M B | vLLM progressive tool-delta relay | 17b, 26a | Planned |
| 3 | 30 | S B | In-process vLLM fail-closed, exact usage | – | Planned |
| 3 | 31 | S/M A+B | Logprobs partition, `output_text.logprobs` | 16, 17b | Planned |
| 4 | 12b | L A refactor | `request_scope.py`, `auto_dispatch.py` | 18 | Planned |
| 4 | 32 | L C | Response store core, persistence matrix | 07, 18, 23 | Planned |
| 4 | 33 | M C | Retrieve/delete/cancel/`input_items`, `item_reference` | 32, 13 | Planned |
| 4 | 34a | M/L C | `postgres_common.py`, Postgres store | 32, 20 | Planned |
| 4 | 34b | M C | HA wiring, Helm, retention, kind gate | 34a | Planned |
| 4 | 35 | M C | `token_count.py`, `input_tokens`, truncation fitter | 32 | Planned |
| 4 | 36 | L C | Compaction family (`/compact`, `context_management`) | 18, 20, 35 | Planned |
| 4 | 37 | M C | `EndpointExecutor` registry | 12a, 12b | Planned |
| 4 | 38a | M C | Background submit/poll/cancel (memory) | 33, 37 | Planned |
| 4 | 38b | M C | Event log, stream resume | 38a | Planned |
| 4 | 38c | M/L C | Background HA: fencing, drain, Postgres | 38b, 34b | Planned |
| 4 | 39 | L C | Conversations API | 33, 34a | Planned |
| 4 | 40 | M C | Batch `/v1/responses` lines | 12b, 32, 37 | Planned |
| 4 | 55 | M C | Webhooks (HMAC sink, at-least-once, `EgressPolicy`, default off) | 38c, 48 | Planned |
| 5 | 41 | S/M D | Request-id, decompression caps, optional CORS | 07 | Planned |
| 5 | 42 | S B | Disconnect cancellation | 12a | Planned |
| 5 | 43 | S/M B | Affinity and prefix fingerprint | 26b, 32 | Planned |
| 5 | 44 | M D | WebSocket ingress (auth, tenancy, `ConcurrencyGate`) | 12b, 15, SP-8a | Planned |
| 5 | 45a | M D | WebSocket single-lane create | 44, 18, 32 | Planned |
| 5 | 45b | M D | WebSocket connection cache | 45a | Planned |
| 5 | 45c | M D | Lanes, `stream_id`, lifetime | 45b | Planned |
| 5 | 46 | S D | Interrupt, steer stub | 45c | Planned |
| 5 | 47 | S/M D | WebSocket acceptance, default on | 46, 05, 13, 14 | Planned |
| 6 | 48 | L E | Hosted mechanism, `EgressPolicy`, server `tool_search` | 18, 23, 24, SP-10 | Planned |
| 6 | 49 | M E | `web_search` reference executor | 48 | Planned |
| 6 | 50 | M E | MCP reference executor | 48, SP-8 | Planned |
| 6 | 51 | M E | Generic hosted service adapter | 48 | Planned |
| 6 | 56 | S/M A | URL inputs through `EgressPolicy` (default off) | 48, 25 | Planned |
| 6 | 57 | M A | `DocumentExtractor` port for `input_file` (implementations example/deployment-owned; unset → 400 `unsupported_file_type`) | 25 | Planned |
| 7 | 53 | S/M | Agents/node smokes, final Codex matrix, docs; close g6 P-C2; m20 Implemented | all | Planned |
| 7 | 54 | M | WebSocket steering and prewarm (spike-gated) | 47 | Planned |
| 7 | E-1..n | example | Declared reasoning/developer capabilities, effort opt-in, `upstream_chat_models` (A7-gated), vLLM flags | 10, 16, 27 | Planned |

Former WP-52 (example reference hosted services) is won't-do (O-5).

| Spike | Purpose | Gates |
|---|---|---|
| SP-1 | vLLM capability matrix (GPU) | 04, 26a, 27, 28 |
| SP-2 | xgrammar `pattern` support, Lark subset | 28 |
| SP-3 | Tag atomicity, example template scan | 16, 19 |
| SP-4 | `[DONE]`, repeated `in_progress`, `phase`, `x-reasoning-included` tolerance (node, Agents, Vercel, Codex) | 17a/b, 21 |
| SP-5 | OpenAI stateful probe (owner key, O-3) | 32, 33, 36, 38b, 39 |
| SP-6 | v3.24 accumulator with early `added` and `compacting` | 18 |
| SP-7/7b | Postgres latency, event-chunk batching | 34a, 38b |
| SP-8/8a | Codex WebSocket semantics, MCP lock; uvicorn denial with `websockets-sansio` | 44, 45, 50 |
| SP-9 | 50-turn body growth | 08c, 21, 31 |
| SP-10 | Open-model shim and citation use (GPU): Phase 6 go/no-go | 48–51 |
| SP-11 | `prompt_kind` call-site enumeration | 26a |

**Phase 0 exit:** on ScenarioBackend in CI and one live GPU run, Codex 0.160
default and full-access runs, long outputs, AUTO turns over 300 s, overflow →
compaction (engine and AUTO) and local compaction pass on text-only models.

## 8. Framework admission

Every `kairyu/` change outside the Responses L3 contract. Auth: a/b/c =
decision 4; O-1 = approved by the owner on 2026-10-05.

| WP | Change | Shared contract / code path | Why extension points do not suffice | Independent regression | Smallest mechanism; example-owned part | Auth |
|---|---|---|---|---|---|---|
| 04 | Typed `ContextLengthExceededError`, `resolve_output_budget`, upstream classification, `placement` | `context_length_exceeded` on all surfaces (`engine_loop.py`, `zmq_backend.py`, `openai_backend.py`) | Errors are untyped `ValueError`s; upstream text hidden behind 502 | Chat on vLLM overflow → 502; ZMQ preflight assumes 16 tokens | `code` attribute + classifier table | O-1 |
| 07 | Middleware `send_error` via the classifier; Responses-dialect 503 `slow_down`; scoped 422/404/405; `TenantAdmission.exceeds_token_capacity` + refill-based `retry_after_s` | Shared ingress errors (m7 D5) | Middleware builds its own envelope without `param`; a refused reservation cannot tell never-fits from transient | Chat errors omit the spec-required `param` (M-ST-6) | Dialect switch + classifier; limits deployment-owned | O-1 |
| 08c | Off-loop parse set + Responses body default | `app.py` parse set; middleware body limits | Parse set and body paths are fixed | `max_chat_body_bytes` defaults to None; unbounded bodies | Set entry + `ResponsesConfig.max_body_bytes` | O-1 |
| 10 | Lossless effort + default fold map | Shared validator and L1/L2 effort vocabulary | Folding happens in the shared validator | Chat `none` → 400; Chat `medium` reaches vLLM as `high` | Enum, `reasoning_level()`, per-model map defaulting to the legacy fold; opt-in and budgets example-owned | b |
| 11 | Tenant service tiers | OpenAI `service_tier` on Chat and Responses | No field exists | Chat `service_tier` → 400 | `service_tier.py`; tiers deployment-owned | O-1 |
| 16 | Declared `ReasoningSpan`; `GenerationRequest.reasoning_span` | Private reasoning never enters `content` or tool parsing | The split is effort-gated; a third tag source would conflict | Chat/Messages leak `<think>` for models that think by default | Span from declared config + load-time validation; tags, `starts_open`, `disable_kwarg`, `developer_role` example-owned | D2 + b |
| 18 | `TenantAdmission` → `tenant_admission.py` + `reserve_followup`/settle | Multi-round dispatch in one request | `reserve_tokens` raises on a second reservation | AUTO two-round dispatch raises or bypasses quota | One method + settle semantics | O-1 |
| 19 | `GenerationUsage.reasoning_tokens`; AUTO `public_cached_tokens`/`public_reasoning_tokens` | `completion_tokens_details`, `prompt_tokens_details` | No field exists | Chat drops vLLM `reasoning_tokens`; AUTO cached always 0 | Trailing optional fields | b; cached O-1 |
| 25 | `FileReader` port over the batch Files store | OpenAI `file_id` inputs | `/v1/files` exists only with batch | – (first consumer Responses) | Read-only port; `kairyu/batch` unchanged | O-1 |
| 26a/b, 27 | ChatPrompt kind, capabilities, routing text, `upstream_chat_models` | Role- and part-preserving chat to OpenAI-compatible upstreams | `MultimodalPrompt` is lossy | Chat tool history flattened; Chat image + transcript → 400 | One prompt kind with capability gating; opt-in models example-owned | a |
| 28 | `tool_grammar.py` gating; vLLM `strict_tools_fallback` | `tool_choice`, `parallel_tool_calls`, `strict` | Grammar built only for `strict` | Chat `required` → post-hoc 502; `strict` → 400 on vLLM | xgrammar normalization port | O-1 |
| 29 | Progressive vLLM tool-delta relay | Streaming tool deltas | Arguments are accumulated | Chat streams from vLLM deliver calls in one chunk | `openai_stream_relay.py` | O-1 |
| 30 | In-process vLLM fail-closed + exact usage | m9 D6 | Intents silently dropped | Chat `logprobs`/`response_format` ignored | `validate_request` + `_to_result` | O-1 |
| 31 | Chat logprob partition | Logprobs exclude private reasoning | – | Chat logprobs include reasoning tokens | `partition_logprobs` | O-1 |
| 32/34a/b | `kairyu/response_store`, `storage/postgres_common.py`, `psycopg-pool` (fleet extra) | Stored responses and conversations | Non-injectable in-route store | Multi-gateway continuations fail | Protocols + two backends; DSN/retention deployment-owned | c |
| 36 | Deployment compaction-prompt override | Compaction wire contract | A default prompt is needed unconfigured | – | `ResponsesConfig.compaction.{summary_prompt,bridge_prefix}` (values deployment-owned) | O-1 |
| 37/38a–c | `EndpointExecutor` registry; per-job scheduling class; `convert_to_async_submission`; fenced event-log writes | Durable leased execution (m10 A38/A40) | Worker hard-codes chat and the batch class | AUTO not runnable async; non-flex jobs yield under SLO pressure | Registry + per-job fields | c |
| 40 | Batch `/v1/responses` lines | OpenAI Batch | Single endpoint constant | Batch with `/v1/responses` → 400 | Register an executor | O-1 |
| 41 | Decompression, request-id, optional CORS | Shared ingress | – | `Content-Encoding` POST → `{detail}` 400; SDK `_request_id` None | Middleware with caps; CORS off | O-1 |
| 42 | Disconnect cancellation | Generation stops when the client leaves | – | Chat unary keeps generating after disconnect | `disconnect.py` (documented behavior change) | O-1 |
| 43 | Affinity + routing-text fingerprint | Chat `prompt_cache_key`, prefix placement | Session-hinted traffic skips prefix tracking | Lost system-prompt KV reuse | `affinity.py`; `prefix_index` example-owned | O-1 |
| 15, 44 | uvicorn `ws` selection; WebSocket auth, tenancy, `ConcurrencyGate`; `websockets` extra; uvicorn pin | Shared ingress | `AuthMiddleware` and tenancy skip non-http scopes | Any WebSocket route would be unauthenticated as owner `"default"` | Branch + gate extraction | O-1 |
| 20/34b/44/48 | `deploy/responses_{spec,wiring,validation}.py`, Helm guard and generated secret | Deployment schema for sealing, store, WebSocket, hosted | Only the existing compaction-secret field exists | Rolling restarts break compacted sessions | Sections + preflight; values operator-owned | sealing O-1; store c |
| 48 | Hosted mechanism, `EgressPolicy`, budgets, `hosted_tool` ledger record, spec builders | Server-executed tools bound to the spec wire | Nothing exists | – (Responses-only consumer, under `responses/hosted/`) | Mechanism only; services, sidecars, allowlists, descriptions example-owned | O-1 |
| 49 | `web_search` reference executor over a minimal search-service contract | Codex live/indexed `web_search` | ECO-D1-style `base_url` is the precedent | – | One executor; SearXNG sidecar example-owned | O-1 |
| 50 | MCP reference executor (`mcp` extra) | Public MCP protocol | – | – | One executor; allowlists deployment-owned | O-1 |
| 51 | Generic hosted service adapter | file_search, image_generation, code_interpreter, container shell | No public protocol; DI alone leaves DeploymentSpec users without configuration | – | One adapter; no provider bindings; ECO-D1 `ExecutionRequest` unchanged | O-1 |
| 55, 56 | Webhook sink; URL-input fetch | Background notifications; http(s) inputs | – | – | Default off | O-1 + O-5 |
| 57 | `DocumentExtractor` port in `ResponsesDeps` | OpenAI `input_file` for non-text types | No extraction hook exists | – (first consumer Responses) | Port, mime detection, size caps, typed errors; extractor implementations example/deployment-owned | O-1 + O-5 |

Refactors without behavior change need no row: WP-06, 06b, 12a, 12b, 17a
(Messages side), 35 (Messages `count_tokens`) and 37 (chat side); each passes
the wire-capture gate.

## 9. Gap coverage matrix

Severity is verdict-corrected; "→" is delivery order; *hosted* = handled under
decision 3; WND = won't do. Every one of the 80 audit gaps and 31
verifier-added gaps is mapped.

| Gap | Sev | WP | Resolution |
|---|---|---|---|
| G-request-fields-1 | P0 | 03 → 08b | No cap (compaction included); echo per D-a |
| G-request-fields-2 | P2 | 11 | Executed tier echoed; `flex` → batch class; `scale` → default |
| G-request-fields-3 | P2 | 08a → 21, 31, 33, 48–51 | Every include value accepted; implemented by its feature WP |
| G-request-fields-4 | P2 | 31 | Logprobs partitioned away from reasoning |
| G-request-fields-5 | P2 | 35 | Exact-count fitter; D-f |
| G-request-fields-6 | P2 | 08a, 09, 23 | v3.24 fields preserved verbatim; `program` items dropped |
| G-request-fields-7 | P3 | 08a → 17c, 38b | Obfuscation opt-in (D-c) |
| G-request-fields-8 | P2 | 08b, 09 | Verbosity per C26; one merged system message |
| G-request-fields-9 | P3 | 08a → 48 | Echo now; enforced for hosted tools |
| G-request-fields-10 | P3 hosted | 08a | `prompt`/`moderation`/`multi_agent` 400; `access_programs` echoed |
| G-request-fields-11 | P3 | 08a | Spec bounds; no `max_output_tokens ≥16` rule |
| G-streaming-1 | P0 | 03 → 17a, 18 | Data heartbeats everywhere |
| G-streaming-2 | P1 | 16, 17a | Splitter + `segment_stream` |
| G-streaming-3 | P2 | 21 | Reasoning events |
| G-streaming-4 | P2 | 17b, 18, 29 | Engine, AUTO, vLLM |
| G-streaming-5 | P3 | 17b (no change) | `name` kept on `function_call_arguments.done` (harmless superset) |
| G-errors-1 | P0 | 04 (+18) | Typed error, placement, AUTO in-band |
| G-errors-2 | P3 | 07 → 32 | 400 + param (D-d) |
| G-errors-3 | P2 | 07, 18 | Re-render + dispatch-first |
| G-errors-4 | P2 | 07 | Scoped envelopes, middleware included |
| G-errors-5 | P2 | 07, 17c (18 for compaction codes) | Mapping table (O-2); compaction codes per C4 |
| G-tools-1 | P0 hosted | 03 → 23 → 49 | Drop → reject override → execute |
| G-tools-2 | P1 | 22, 28 | Shim + events; regex per SP-2; Lark WND |
| G-tools-3 | P1 | 28 | vLLM best-effort fallback |
| G-tools-4 | P2 | 17b, 28 | In-stream gates + constrained decoding |
| G-tools-5 | P2 | 22, 28 | L3 + L1 `allowed_tools` |
| G-tools-6 | P2 | 24, 48 | Client / server `tool_search` |
| G-tools-7 | P2 | 24 | Shell shims + `tool_choice` routing |
| G-tools-8 | P2 | 22, 24 | Freeform + typed `apply_patch` |
| G-tools-9 | P3 | 24, 27 | Screenshots need ChatPrompt |
| G-tools-10 | P2 | 23 → 50 | Policy + redaction → MCP executor |
| G-tools-11 | P3 hosted | 23 → 48 → 49, 51 | Builders for all kinds; `programmatic_tool_calling` dropped, rejected under execute |
| G-tools-12 | P3 | 08a | `parameters:null` → `{}` |
| G-input-items-1 | P0 | 05 → 25 → 27 | Text-only catalog (done in WP-05) → message images → tool-output images |
| G-input-items-2 | P2 | 24 | `additional_tools` |
| G-input-items-3 | P2 | 33 | `item_reference` |
| G-input-items-4 | P2 | 25, 56, 57 | `text/*` files; URL fetch (56); document extraction port (57) |
| G-input-items-5 | P3 hosted | 23, 48 | Drop / replay |
| G-input-items-6 | P3 | 09 | Refusal → assistant text |
| G-input-items-7 | P3 | 09 | `configuration_update` |
| G-input-items-8 | P3 | 08a, 09 | `agent_message`, encrypted parts; `input_audio` → typed 400 |
| G-reasoning-1 | P1 | 16 | Declared spans |
| G-reasoning-2 | P1 | 21 | Items + mirror |
| G-reasoning-3 | P1 | 20, 21, 27 | kst2 + replay precedence + upstream carrier |
| G-reasoning-4 | P1 | 10 | Lossless (b) with default fold |
| G-output-object-1 | P2 | 08b | Full envelope + Open Responses requirements |
| G-output-object-2 | P1 | 17b | Preamble kept |
| G-output-object-3 | P3 | 17b | `phase` (SP-4) |
| G-output-object-4 | P3 | 17c | `incomplete{content_filter}` |
| G-endpoints-1 | P1 | 33, 38b | Unary GET, then resume |
| G-endpoints-2 | P2 | 33 | DELETE |
| G-endpoints-3 | P2 | 33 | `input_items` |
| G-endpoints-4 | P2 | 35 | `input_tokens` |
| G-endpoints-5 | P2 | 39 | Conversations |
| G-endpoints-6 | P2 | 38a–c | Background lifecycle, HA |
| G-state-store-1 | P1 | 32, 34a/b | Protocol + Postgres (c) |
| G-state-store-2 | P2 | 32 | Incremental storage, immutable bytes |
| G-other-1 | P1 | 05 (+flags per WP) | Catalog generator; done in WP-05 |
| G-other-2 | P2 | 08c | Body limit + off-loop parse |
| G-other-3 | P2 | 17c | `engine_admission` |
| G-other-4 | P2 | 26a/b, 27 | ChatPrompt (a) |
| G-other-5 | P3 | 32, 43 | Affinity |
| G-other-6 | P3 | 42 | Shared disconnect |
| G-other-7 | P3 | 05, 34b, 53 | Docs; Codex part done in WP-05 |
| G-other-8 | P3 hosted | 55 | Optional HMAC webhook sink, default off (O-5 approved) |
| G-conformance-1 | P1 | 01 | Schema gate |
| G-conformance-2 | P1 | 02, 05, 53 | Fixtures + matrix; fixtures done in WP-02, matrix in WP-05 |
| G-conformance-3 | P2 | 14 | Open Responses |
| G-conformance-4 | P2 | 13, 53 | v3.24 + node/Agents |
| G-compaction-1 | P2 | 36 | `context_management` |
| G-compaction-2 | P2 | 36 | `/compact` (retained messages in clear + summary seal) |
| G-compaction-3 | P3 | 20 | Key ring, kid, max age, per-token subkeys; `created_by` WND (C21) |
| G-compaction-4 | P3 | 05 | Gating docs; done in WP-05 |
| G-transport-1 | P2 | 44–47, 54 | WebSocket mode |
| G-transport-2 | P3 | 15 | `ws="none"` while disabled |
| G-transport-3 | P3 | 41 | gzip core, zstd extra |
| G-usage-1 | P2 | 19, 21 | `reasoning_tokens` |
| G-usage-2 | P3 | 08b | `cache_write_tokens:0`; radix cache-write metric WND |
| G-usage-3 | P3 | 19, E-* | AUTO cached; vLLM flag example-owned |
| G-usage-4 | P3 | 19, 30 | Exact-count fallback; in-process vLLM |
| M-W-1 effort `none` | P2 | 10 | – |
| M-W-2 explicit nulls | P3 | 08a | – |
| M-W-3 `context_management` 400 | P2 | 08a (typed) → 36 | – |
| M-W-4 developer role | P2 | 09, 16, 27, E-* | L3 fallback merge; templates declare support |
| M-W-5 two assistant turns | P3 | 09 | Merged turn |
| M-W-6 hosted/MCP `tool_choice` | P3 | 23, 48 | Computer/shell/apply_patch routed to shims |
| M-W-7 pre-stream code shape | P2 | 04, 07 | – |
| M-W-8 zero usage on `failed` | P3 | 17a | – |
| M-W-9 X-Request-ID | P3 | 41 | – |
| M-W-10 local compaction cap | P2 | 03 | Done in WP-03 |
| M-SR-1 1024 cap | P0 | 03 | Done in WP-03 |
| M-SR-2 compaction events / early `added` | P1 | 18 | – |
| M-SR-3 ephemeral key | P2 | 20 | Secret required (staged) + Helm generated |
| M-SR-4 `x-reasoning-included` | P2 | 21 | Always true (C10) |
| M-SR-5 `reasoning_tokens` = 0 | P3 | 19, 21 | – |
| M-SR-6 `stream_options` | P3 | 08a | – |
| M-SR-7 `reasoning` validation | P3 | 08a, 21 | – |
| M-SR-8 #530 comment claim | – | 03 | Done in WP-03 (code + m11 amendment) |
| M-ST-1 not-found contract | P1 | 07, 32 | D-d |
| M-ST-2 `/compact` | P2 | 36 | – |
| M-ST-3 HA ephemeral key | P2 | 20 | – |
| M-ST-4 body guard | P2 | 08c (+33, 35, 36, 39 predicates) | Default 64 MiB |
| M-ST-5 framework envelopes | P2 | 07 | – |
| M-ST-6 `param` missing | P2 | 01 (detection), 07 | Middleware included |
| M-ST-7 hand-written Codex test | P2 | 02 | Done in WP-02 (AUTO `namespace-tool-loop` fixture) |
| M-ST-8 lenient SDK | P2 | 01 (2.44 strict), 13 | – |
| M-ST-9 beta surface | P3 | 08a | – |
| M-ST-10 CORS | P3 | 41 | – |
| M-ST-11 Codex `/models` | P3 | 05 | Generator (done in WP-05); served route WND (O-5) |
| M-ST-12 `/alpha/search` | P3 | 07 (envelope 404) | WND (O-5) |
| M-ST-13 `context_length_exceeded` never emitted | – | 04 | – |

Review-derived requirements without a gap ID:

| R | Requirement | WP |
|---|---|---|
| R-1 | Codex extension inventory from codex-rs serde types; per-entry semantics; `priority` | 02, 08a |
| R-2 | Call-finality invariant | 17b, 24, 32 |
| R-3 | Overflow placement incl. AUTO; public vs internal-stage overflow | 04, 18 |
| R-4 | Error/backpressure mapping incl. middleware and WebSocket | 07, 17c, 18, 45a |
| R-5 | Resume contract: full log, CAS sequence, start order, typed errors | 38b |
| R-6 | Sealing secret at any replica count; Helm secret; two-phase rotation | 20 |
| R-7 | Compaction shape + D-j | 09, 32, 36, 39 |
| R-8 | Open Responses envelope requirements | 08b |
| R-9 | Dependent-tool drop semantics | 23 |
| R-10 | Developer-role fallback | 09, 16 |
| R-11 | Responses body default + independent decompression cap | 08c, 41 |
| R-12 | `interrupted` divergence; summary-part `incomplete` | 21, 46 |
| R-13 | MCP secret redaction before any echo or persistence | 23 |
| R-14 | ScenarioBackend + FakeVLLMUpstream | 02b |
| R-15 | Default effort fold for existing models | 10 |
| R-16 | Background fencing, per-job scheduling, drain | 37, 38a, 38c |
| R-17 | Per-token seal subkeys | 20 |
| R-18 | Tenant budget vs remaining-context reservation | 03, 07 |
| R-19 | Unknown-nested-field escape hatch; Codex release procedure; drift issue | 05, 08a |
| R-20 | Follow-up admission with `RoundDriver` | 18 |
| R-21 | Shared `segment_stream` | 17a |
| R-22 | Declared reasoning spans | 16 |
| R-23 | WebSocket error/limit/field rules | 45a–c |
| R-24 | Replay precedence by seal/model | 21 |
| R-25 | v3.24 additive surface; `input_tokens` model/conversation rules | 08a, 35 |
| R-26 | `tool_choice`/shim linkage; no undeclared item types | 23, 24, 28 |
| R-27 | `FileReader` port | 25 |
| R-28 | Additional Codex fixtures + unknown-model contract | 02, 08a |

**Won't do (O-5):** `programmatic_tool_calling` executor (drop/reject only);
Lark grammar enforcement (grammar verbatim in the description, regex per SP-2);
a served Codex `/models` route (the generator covers it); `/alpha/search`;
served-model aliases for Codex auxiliary slugs (docs only); `created_by` on
compaction items; example reference hosted services (former WP-52).

**Not done in M20:** refusal events and annotation events other than
`url_citation`; numeric or custom efforts; default-on obfuscation; a default
sealed-item TTL; `prompt_cache_retention` → radix pins; `x-codex-turn-state`;
KV-event routing for chat prompts; logprobs or structured output inside the
in-process VLLMBackend (fails closed); `service_tier` passthrough to an OpenAI
upstream; MCP `connector_id`/`tunnel_id`; Vector Stores, Files and Containers
APIs; provider-bound executors and ECO-D1 widening; hosted result retention;
`before` pagination; LISTEN/NOTIFY; re-executing streamed background jobs; an
in-memory background queue in a DeploymentSpec (A36); hierarchical
summarization; request-id-idempotent metering.

## 10. m11 supersession table

The single list of m11 D4 statements M20 supersedes. m11 text is never
rewritten; each row flips to Done in the WP that supersedes it. Locations refer
to `docs/design/m11-product.md` D4 and its amendments.

| m11 D4 statement | Superseded by | Status |
|---|---|---|
| #530: "the 1024-token default output cap applies to AUTO responses as it does to engines" (omitted `max_output_tokens` → 1024) | WP-03 (remaining context, echo `null`) | Done 2026-10-05 |
| #530/#531: compaction summaries capped at 4096 tokens | WP-03 | Done 2026-10-05 |
| #530: buffered streams emit `: keep-alive` comments; AUTO status keep-alives stay comments | WP-03 (data `response.in_progress` heartbeats) → WP-18 (emitter-owned) | Done 2026-10-05 (interim) |
| #530: search configuration accepted only on the disabled `web_search` tool (live/indexed → 400) | WP-03 (accept-and-drop) → WP-23 (reject override) → WP-49 (execute) | Done 2026-10-05 (default policy) |
| #201: "tool streams buffer until parsed calls satisfy the requested policy" | WP-17b (progressive calls, in-stream gates) | Planned |
| #530: `reasoning_content` dropped, "matching the documented no-reasoning-output stance"; echoed `reasoning` input items dropped | WP-21 (reasoning items, sealed replay) | Planned |
| D4: in-memory `ResponseStore`; #201: bounded, deep-copy isolated, process-local `previous_response_id` storage (single gateway) | WP-32 (store protocol) → WP-34a/b (Postgres, HA) | Planned |
| #201: "only successful stored responses are continuable" | WP-32 (chainable statuses completed/incomplete; SP-5) | Planned |
| #531: a length-terminated summary returns `response.incomplete`; an empty summary fails as 502 `compaction_failed` | WP-18 (C4 failure codes) → WP-36 | Planned |
| #530: a successful compaction stores only the compaction item (pre-compaction history replaced) | WP-09 / WP-36 (D-j keeps user and developer messages) | Planned |
| #201: unsupported fields fail before dispatch — `background` | WP-38a | Planned |
| #201: unsupported fields fail before dispatch — `conversation` | WP-39 | Planned |
| #201: unsupported fields fail before dispatch — `truncation:"auto"` | WP-35 | Planned |
| #201: unsupported fields fail before dispatch — `context_management` | WP-36 | Planned |
| #201: unknown or unsupported fields rejected (shape of the rejection) | WP-08a (400 `unknown_parameter` with `param`, extension inventory) | Planned |
| #530: without a configured secret, a process-local key limits tokens to one gateway lifetime | WP-20 (secret required at any replica count, Helm-generated, kst2) | Planned |
| #530: `GET /v1/responses` answers 426 because no WebSocket library is installed | WP-15 (explicit `ws="none"`) → WP-47 (WebSocket mode; 426 only when disabled or at capacity) | Planned |
| #530: "tenant 429s are not retried by Codex (bench deployments should size admission accordingly)" | WP-07 (503 `slow_down` + `Retry-After`, O-2) | Done (2026-10-05) |
| D4 behavior pinned by `test_unknown_previous_id_404`: unknown `previous_response_id` → 404 | WP-07 (400 `previous_response_not_found`, D-d) | Done (2026-10-05) |
| #530: "Acceptance: … unmodified codex-cli 0.147.0 runs" | WP-02 (fixtures) → WP-05 (0.160 matrix) → WP-53 | In progress (WP-02 fixtures, WP-05 matrix 2026-10-05) |
| D4/#201/A16: official SDK round-trips (lenient openai 2.44 tests) as binding coverage | WP-01 (strict 2.44 harness + schema gate) → WP-13 (v3.24 live server) → WP-53 (Agents, node) | In progress (WP-01) |

m11 D3 (tenant tiers) and D6 (`flex` → batch class) receive their own dated
amendments in WP-11.

## 11. Records, amendments and docs

All amendments are dated new entries; old text is never rewritten.

- m11 D4: WP-03 Codex P0 amendment (done); WP-01 pointer to §10 (done); rows
  above flip per WP. m11 D3/D6: WP-11.
- m7 D5: Responses-dialect backpressure (WP-07; done 2026-10-05).
- m9: D1 usage (19); D2 ChatPrompt, `upstream_chat_models`, declared
  reasoning/developer template capabilities (16, 26b, 27); D3 logprob
  partition and in-process fail-closed (30, 31); D4 tool grammar, vLLM strict
  fallback, custom tools, no Lark (28); D6 lossless effort with default fold,
  `context_length_exceeded`, `param`, Chat `prompt_cache_key`/`service_tier`
  (04, 07, 10, 11, 43).
- frontier-native-runtime: FN-D3 lossless effort opt-in; FN-D9 migration note.
- m10: D6 prefix fingerprint (43); A37 pool justification (34a); A40 executor
  registry shared with batch, A36 stays non-deployable (37, 38c).
- ECO-D1 unchanged; D20 records that it is not widened.
- Goals and roadmap: g6 P-C2 reopened in WP-01, closed in WP-53;
  `docs/roadmap.md` note added in WP-01.
- PROGRESS: WP-01 adds the M20 milestone row and a `[design]` entry; later
  entries only for D-ID, milestone or blocker changes; pure refactors get none.
- Docs: `docs/ide-clients.md` (Codex incl. auxiliary slugs, Agents, node,
  WebSocket), `docs/deployment.md` Responses section (store/HA, sealing and
  rotation, hosted and egress, WebSocket LB timeouts, body/decompression caps,
  backpressure, tenant sizing, compaction gating; the live gate replaces
  `scripts/codex_responses_smoke.sh`), `docs/gpu-runbook.md`.

## 12. Verification

- **Per WP:** base/head collected counts with
  `pytest --collect-only -q -p no:cacheprovider --no-cov tests`; deletion-heavy
  WPs (03, 07, 17a, 18, 32) show a net decrease in their areas.
- **CPU gates:** (1) schema gate over every existing test, `CONTRACT_STRICT=1`,
  wire capture for refactors; (2) `test_codex_fixture_replay` (PR-blocking) and
  the nightly Codex matrix; (3) Open Responses (blocking after 08b and 18);
  (4) strict v3.24 SDK on a live server incl. `responses.stream(response_id=…)`
  resume, Agents and node smokes; (5) Postgres marker suite and the kind state
  gate; (6) performance: emitter and `segment_stream` per-delta overhead in
  `test_serving_micro_overheads.py`, a 2 MB Codex body parsed off-loop.
- **No tests for:** construction-time invariants, static preset/registry
  contents, the catalog key list, refactors (wire capture), per-field schema
  checks (schema gate).
- **GPU/live** (recorded in `bench/results/responses-*.json`): SP-1 and SP-10 on
  the first GPU day; declared-span split and `reasoning_tokens ≤
  completion_tokens` on native Qwen3 thinking; Codex 0.160 against a real
  reasoning model (replay, long outputs, compaction quality, `view_image`,
  gateway hop, shed → retry); A7 before/after WP-43 and each E-PR; WebSocket TTFT
  vs HTTP; hosted `web_search` with a real sidecar; Open WebUI P-B3; example
  `verify.sh`.

**Open points.** SP-5 results (O-3) decide: 400 vs 404 for unknown ids (D-d),
chaining from failed/in-progress responses, the lock code, the cancelled-stream
end, DELETE while in progress, the `input_items` shape, `encrypted_content` on
retrieve, resume error codes, and the roles `/compact` retains. Until the owner
runs the probe, the defaults in D12, D15, D16 and D17 stand.

**Owner acknowledgement needed (WP-01).** `docs/design/m11-product.md` grows
from 869 to 876 lines (+7) with the m11 D4 pointer amendment and its Status
note. The plan's size rule has no exception for append-only design records,
and offsetting the growth would rewrite past text. Principle 8 proposes the
exception; until the owner acknowledges it, this growth is a recorded
deviation from DoD (4).

## 13. Risks

| Risk | Mitigation |
|---|---|
| Chat/Messages behavior changes (declared-span split, effort, grammar `required`, disconnect cancellation) | One WP per change with its amendment and admission row; default effort fold; A7 and Open WebUI gates |
| Oversized files across parallel lanes | No net growth with `wc -l`; fixed merge order; refactor-first WPs (06b, 12a, 26a) |
| Duplicate tool execution | Call finality (D-h); "call then cut/error" fixtures |
| Clients rejecting `[DONE]`, repeated `in_progress` or `phase` | SP-4 before 17a/b; one-constant fallback |
| Codex adds wire fields ahead of the pin | Extension inventory, `drop_and_log` hatch, weekly drift job, 7-day fixture refresh |
| Tenant budget below `max_model_len` | WP-03 startup/validate warning; never-fits → 429 with explanation |
| Sealing key management | Staged secret requirement, Helm generation, two-phase rotation |
| ChatPrompt blast radius | SP-11, `assert_never`, 26a/26b split, A7 gate per E-PR |
| Postgres hot-path latency | SP-7 (p99 < 5 ms at 200 rps), incremental storage |
| Background zombie writers | Fencing in the same database, drain, kind gate |
| WebSocket maturity and LB timeouts | `ws="none"` default, SP-8a, connection caps, default off until WP-47 |
| SSRF, prompt injection, secret leakage via hosted tools | `EgressPolicy` before any executor (48), `public_view` (23), byte caps, budgets |
| Open models cannot use hosted tools | SP-10 go/no-go before Phase 6 |
