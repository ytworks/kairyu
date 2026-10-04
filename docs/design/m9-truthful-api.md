# M9 Design: Truthful API — Usage, Chat Templates, Logprobs, Structured Outputs

Status: **Implemented** (2026-07-03; D2 amended 2026-08-04; D3 amended
2026-08-05; D6 amended 2026-10-05 by M20 WP-04, WP-07 and WP-30). Reviewed —
APPROVE-WITH-AMENDMENTS (2-reviewer agent panel, 2026-07-03; amendments
applied inline, see §6).
All five phases (D1–D5) landed with tests: 437 → 471 tests, 94% coverage.
Milestone: M9 (realizes roadmap Track P-A, goal G6 gates P-A1..P-A5)
Date: 2026-07-03
Depends on: M8 (tokenizer seam, StreamUpdate usage fields,
`Scheduler.num_cached_tokens`, engine-side `response_format` enforcement).

## 1. Goal

Make every number and string the API returns true: token counts from the real
tokenizer (with cached-token detail), chat prompts from real HF Jinja
templates, logprobs surfaced, `/v1/completions` served, `n > 1` real, and
`response_format` enforced end-to-end. Bench gains the honesty fixes (auth,
token-granularity TPOT, results files).

## 2. Key design decisions and rationale

### D1 — Usage truth: backend-reported counts, `cached_tokens`, `include_usage`

- `GenerationResult` gains `usage: GenerationUsage | None = None` (frozen:
  `prompt_tokens`, `completion_tokens`, `cached_tokens`). Producers:
  - `KairyuBackend` fills from `StreamUpdate` (`num_prompt_tokens`,
    `len(outputs)`, `num_cached_tokens` — all landed in m8 D6).
  - `ZmqEngineBackend` fills from the wire event (fields already present).
  - `MockBackend` fills deterministic counts (its token_ids lengths).
  - `openai_backend` parses upstream `usage` incl.
    `prompt_tokens_details.cached_tokens` (gateway pools stay truthful).
- `protocol.py`: `PromptTokensDetails(cached_tokens: int = 0)`;
  `Usage.prompt_tokens_details: PromptTokensDetails | None = None`;
  `ChatCompletionRequest.stream_options: StreamOptions | None`
  (`include_usage: bool = False`).
- `app.py`: `completion_response` signature changes from
  `texts: list[tuple[str, str|None]]` to `completions:
  Sequence[CompletionOutput]` + `usage: GenerationUsage | None`. The word-split
  `_approx_tokens` fallback applies to ANY `usage=None` result — the
  orchestrator path (which synthesizes `CompletionOutput(index=0,
  text=result.text, token_ids=())` until M11) and third-party backends
  (vllm_backend et al.) alike; recorded limitation. Call sites updated in the
  same commit: app.py ×3 and **`kairyu/batch/worker.py`** (the actual second
  consumer — batch JSONL output embeds usage and silently becomes truthful).
- **Chunk-level usage contract (amended)**: with `include_usage`, every
  non-final chunk carries `"usage": null` and one final extra chunk before
  `[DONE]` carries populated usage with `"choices": []`; without
  `stream_options`, the `usage` key is OMITTED from chunks entirely
  (serialization must exclude the field, not emit null). 400 when
  `stream_options` is present with `stream: false`. The tools+streaming path
  (`_stream_choices`) emits the usage chunk too.
- **n>1 aggregation (amended)**: prompt is counted ONCE (`prompt_tokens` and
  `cached_tokens` from sub-request 0); `completion_tokens` sums across
  choices; naive summation would report n× the prompt.
- `openai_backend` (amended): the streaming path must REQUEST
  `stream_options: {"include_usage": true}` from upstreams and parse the
  empty-choices usage chunk (it iterates `choices` only today and would drop
  it); a config flag tolerates upstreams that 400 on `stream_options`.

### D2 — HF Jinja chat templates; role concatenation is explicit-only

- `chat_template.py` rewritten around `ChatTemplate`, matching HF's
  `_compile_jinja_template` exactly (amended — anything less breaks
  byte-match): `jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True,
  lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])`, globals
  `raise_exception`/`strftime_now`, and **HF's `tojson`**
  (`json.dumps(..., ensure_ascii=False)` — Jinja's builtin html-escapes
  `<>&'`). Render context: `messages`, `tools` (**None, not [], when absent**
  — templates gate on `is not none`), `add_generation_prompt=True`, and the
  tokenizer's named string-valued special-token variables. Assistant
  `tool_calls.arguments`
  arriving as JSON strings are parsed to dicts before rendering (HF
  convention; Qwen templates `| tojson` them). The templated prompt carries a
  tokenizer-owned typed marker. HFTokenizer already encodes with
  `add_special_tokens=False`; the direct vLLM adapter sets that option only for
  the marker, while ordinary completion strings retain vLLM's completion
  default. Templates emit `bos_token` themselves, so this prevents double-BOS
  without changing completion semantics or truthful prompt counts.
  `ChatCompletionRequest.chat_template_kwargs` carries model-specific JSON
  variables (for example Qwen3's `enable_thinking`) into that Jinja context.
  It is rejected when the resolved model has no Kairyu template, and cannot
  replace trusted `messages`, `tools`, `add_generation_prompt`, or configured
  special-token variables.
  `render_chat(messages)` (legacy concatenator) keeps its signature, but as of
  the 2026-08-04 issue #350 amendment it is no longer the implicit production
  default. Production DeploymentSpecs select it only for served model names
  listed in `DeploymentSpec.legacy_chat_models`, with a startup warning.
  Lower-level app, prompt-validation, Responses, and batch-worker entrypoints
  enforce the same template-or-model-membership contract; construction logs an
  explicit warning when both policies are absent and chat then fails before
  dispatch. Completion-only programmatic apps remain usable, and none of these
  paths has an unscoped role-concatenation fallback.
- Per-model config (amended — per-MODEL, not per-replica):
  `DeploymentSpec.chat_templates: dict[str, str]` (model name →
  inline template or `*.jinja` path) — a single map avoids
  BackendSpec-vs-PoolSpec ambiguity and stays out of `options` (factory
  kwargs). An explicit entry has highest precedence. Otherwise `builder.py`
  auto-loads the effective local tokenizer (`tokenizer` override before the
  native `model_path` or direct-vLLM `model`). Root `chat_template.jinja` and
  `additional_chat_templates/*.jinja` files override
  `tokenizer_config.json`'s `chat_template`, matching Transformers; named
  special tokens are loaded from tokenizer metadata and injected into the
  render context. Named template selection follows HF: `tool_use` for tool
  requests when present, otherwise `default`. A non-materialized tokenizer may
  use only a self-contained explicit template: tokenizer-owned variables such
  as `bos_token` fail preflight when their metadata cannot be verified for
  every concrete replica instead of rendering as empty strings.
- **Fail-closed amendment (2026-08-04, issue #350):** template resolution runs
  before backend/GPU construction. Unresolved real local text models fail
  startup unless they have an explicit template or per-model legacy opt-in.
  Static local pools must resolve one identical tokenizer/template policy
  across all replicas. Current OpenAI-compatible remote backends,
  discovery-backed pools, and orchestrators cannot preserve a pre-rendered
  prompt through an upstream template or derived planner/worker prompts, so
  they accept only the explicit legacy policy and reject `chat_templates` at
  startup. Only the DeploymentSpec builder gives deterministic mock backends an
  isolated test-double exception. VLMs have no preflight exception: the
  checked-in remote VLM overlay selects legacy for its text-only path, while
  image-bearing requests bypass that renderer and remain
  structured/upstream-owned.
- `builder.py` threads the same templates and legacy policy into BOTH
  `create_app` and `BatchWorker` (batch and HTTP must render identical prompts);
  `app.py` renders AFTER model resolution. Tool schemas render in-template;
  the `<tool_call>` output-side parse stays.
- Goldens: Llama-3.x and Qwen2.5 chat-template `.jinja` files committed under
  `tests/fixtures/templates/` with fixed message/tool transcripts; expected
  outputs generated once via `transformers` `apply_chat_template` and
  **byte-match committed strings**; a live cross-check against transformers
  runs when the dev dep is present (it is — added to the dev group with a
  pinned minor band, needed by M12 anyway).
- Dep: `jinja2` becomes a core dependency (tiny; the template path is the
  production default).

### D3 — Logprobs surface, `/v1/completions`, real `n > 1`

- **Token strings**: OpenAI logprobs carry token *strings* and `bytes`; the
  engine carries ids. `TokenLogprob` lives in `kairyu/outputs.py` (stdlib-only
  module, cycle-free): `token: str`, `token_id: int`, `logprob: float`,
  `bytes_: tuple[int, ...] | None`, `top: tuple[TokenLogprob, ...]`.
  `CompletionOutput.logprob_content: tuple[TokenLogprob, ...] | None`, built
  in `EngineLoop` (the token-string and byte rules are pinned below).
  Id-keyed `logprobs` dicts stay for vLLM compat. Wire: msgpack nested lists
  in `_event_from_update`, tuples rebuilt client-side; new `StreamUpdate`
  fields keyword-defaulted (positional construction exists at
  `kairyu_backend._pump`). Chunk placement: `logprobs` sits on the CHUNK
  CHOICE (sibling of `delta`), never inside delta; `top_logprobs: []` (empty,
  not null) when not requested; 400 when `top_logprobs` set without
  `logprobs: true`.
- `protocol.py`: request `logprobs: bool = False`, `top_logprobs: int | None`
  (0–20); response `Choice.logprobs: ChoiceLogprobs | None`
  (`content: list[LogprobEntry]`), chunk deltas likewise.
  `sampling_params_from` maps `logprobs=True` →
  `SamplingParams(logprobs=top_logprobs or 0)`.
- **Raw-vocabulary token amendment (2026-08-05, issue #362):** native
  `EngineLoop` selected and `top` `TokenLogprob.token` values use the exact raw
  piece at `vocab[token_id]` from one lazily cached immutable vocabulary
  snapshot. HF snapshots span `max(token_id) + 1`, rather than the vocabulary
  entry count, and retain empty holes so sparse valid IDs remain addressable.
  Before allocating that dense table, a count-relative multiplier plus bounded
  slack rejects pathologically sparse IDs fail-loud, preventing a tiny
  vocabulary map from causing unbounded amplification in Kairyu's own
  ID-indexed table. Loading checkpoint JSON remains the upstream tokenizer's
  trusted-local-model boundary.
  Remote adapters continue to preserve the upstream provider's token
  representation because they do not own a local vocabulary or stable token
  ID. The native representation is independent of visible detokenization and
  `skip_special_tokens`: marker text such as a registered special or a
  byte-level fragment remains inspectable even when its decoded contribution
  is empty or U+FFFD. Raw lookup requires a true integer ID with
  `0 <= token_id < len(vocab)`; negative/non-integer IDs must never wrap into
  the table, and their decoder validation still propagates as an internal-
  contract failure. Valid non-negative padded LM-head IDs outside the tokenizer
  table fall back to the prior flag-aware single-ID decoded string instead of
  failing raw lookup.
  `bytes_` deliberately remains the UTF-8 bytes of that request's
  `skip_special_tokens`-sensitive single-ID decode, not an encoding of the raw
  vocabulary piece. It therefore describes the API's decoded token bytes but
  does not claim sequence-level losslessness for incomplete byte-level pieces.
  The selected entry and every top-logprob entry use the same rule, and the
  numeric ID-keyed compatibility surface is unchanged.
- `/v1/completions` (legacy text): `CompletionRequest` (`prompt: str |
  list[str] | list[int] | list[list[int]]`, `max_tokens`, sampling fields,
  `logprobs: int | None` — legacy
  top-k int, capped at 5 with 400, `stream`, `stream_options`). No chat
  template applied; ids prefixed `cmpl-`; `object: "text_completion"` for
  responses AND stream chunks (not delta-shaped). Legacy logprobs is the
  four-parallel-array shape built from the same TokenLogprob tuples:
  `tokens[]`, `token_logprobs[]`, `top_logprobs[] | null`, `text_offset[]`
  (starts at 0 — echo is rejected, origin documented).
  **Issue #362 amendment:** raw vocabulary notation must not move the visible
  text cursor by marker length. Each offset therefore advances by the decoded
  UTF-8 contribution retained in `bytes_`; a skipped special contributes zero,
  while malformed third-party bytes fall back to the returned token-string
  length. This preserves the pre-amendment Kairyu offset basis. Incomplete
  byte-fragment single-ID decodes retain the existing sequence-level caveat.
  `echo`, `suffix`, `best_of` → 400 with a clear message.
  **Issue #227 amendment:** one `list[int]` is a single pretokenized prompt,
  while `list[list[int]]` is a batch. Empty, mixed, boolean, negative, and
  out-of-u64 shapes fail before dispatch. A single token prompt may stream;
  batches may not. The server constructs `TokensPrompt` and never converts IDs
  to text. Missing-backend-usage fallback counts IDs exactly rather than
  splitting display text.
- **`n > 1`**: `KairyuBackend` implements it as n engine sub-requests
  (`{rid}#c{i}`, sibling params cloned with `n=1`). **Amended (review): the
  sub-requests do NOT share prefill via radix** — siblings admitted in the
  same schedule() hit the uncomputed-node insertion collision and prefill
  privately; M9 accepts n independent prefills and n× prompt page pressure
  (documented, with the KVCacheFull risk noted); in-flight prefix sharing is
  an M11+ optimization. Seeds: completion 0 uses the user seed IDENTICALLY
  (reproducibility parity with direct engine use); completions i>0 use
  splitmix(seed, i); unseeded sub-ids get sha256-derived engine seeds
  (already process-stable) — distinct completions at temperature>0, matching
  OpenAI. **Merged-stream contract**: every partial carries the cumulative
  snapshot of ALL n completions (MockBackend semantics — `_stream_engine`
  emits finish chunks from `last.completions` only). Failure/abandonment
  aborts sibling sub-requests. `ZmqEngineBackend` keeps `n = 1`; the SERVER
  validates `n > 1` per backend capability and returns 400 (not a 502 via the
  exception path).

### D4 — `response_format` end-to-end through the server

No new mechanism (m8 D2 built it): `sampling_params_from` already passes
`response_format` via `extra_args` and the engine enforces it. M9 adds the
missing server-level proof: an API test with a char-vocab tokenizer +
Sampler-equipped `TorchPagedRunner` backend asserting schema-valid JSON and
`finish_reason="stop"` via grammar termination, plus a request-validation
error for malformed `response_format` payloads (400, not engine crash).

### D5 — Bench honesty

`verification/l1/performance/serving_bench.py`: `--api-key` (Authorization header); token-granularity
TPOT — the bench SENDS `stream_options: {"include_usage": true}` and parses
the empty-choices usage chunk (its current loop would drop it), falling back
to chunk counts when the target 400s on `stream_options` — **the method is
labeled in the results JSON**, not just stdout; per-run JSON written to
`bench/results/<date>T<time>-serving.json` (timestamped — same-day runs must
not overwrite).

### D6 — Request-surface truthfulness (amended additions)

- `max_completion_tokens` accepted as an alias of `max_tokens` (the modern
  SDK default — silently ignoring it runs 16-token generations).
- `presence_penalty`/`frequency_penalty`/`logit_bias`-absent: the two
  penalties are added to `ChatCompletionRequest` and mapped through
  `sampling_params_from` (they already exist end-to-end below the API).
- finish_reason wire domain: only {stop, length, tool_calls} leaves the
  server; internal reasons (abort) map to "stop"; the `or "stop"` terminal
  fallback stays (OpenAI requires non-null on final chunks).

**Amendment 2026-10-05 (M20 WP-04) — prompt overflow is a typed, classified
error.** Before this, a prompt that did not fit was an untyped `ValueError`
(400 with `code: null`) on the native engine, a 502 `backend_error` behind a
vLLM upstream, and an AUTO preflight message naming internal engine keys; the
`kairyu-proc` parent preflight also assumed 16 output tokens when `max_tokens`
was omitted, rejecting prompts its child accepts (#496 remaining-context
semantics). Codex compacts only on an in-band `response.failed` with
`context_length_exceeded`; a pre-stream 400 is terminal for it.

- L1: `kairyu/engine/request_errors.py` owns `ContextLengthExceededError`
  (a `ValueError` with `code`, `prompt_tokens`, `max_tokens`,
  `max_model_len`) and `resolve_output_budget`, the one prompt + output vs
  `max_model_len` check used by `EngineLoop` and the `kairyu-proc` parent
  preflight; that parent derives an unset limit from the model config exactly
  as its child engine does (`kairyu/engine/model_limits.py`, moved from
  `kairyu_backend.py`). `kairyu/engine/openai_errors.py` owns the upstream
  4xx mapping (`raise_for_status`, moved from `openai_backend.py`): a 400
  whose OpenAI, vLLM nested/flat, or Kairyu body reports an overflow (by
  `code: context_length_exceeded` or the "maximum context length" / "longer
  than the maximum model length" / legacy Kairyu texts) becomes
  `UpstreamClientError(code="context_length_exceeded")` with a fixed public
  message; upstream text never crosses the tenant boundary. Other 4xx keep
  the sanitized 502.
- L2: only the client-derived public prompt overflows publicly. Every AUTO
  preflight prompt is built from the client prompt, so `Orchestrator.
  prepare_request` re-raises the typed error. An `OrchestratorExecutionError`
  carries `context_overflow_reason`: a MoA or conductor stage prompt is
  orchestration-built, so by default an overflow beneath it is a server error,
  logged with reason `internal_stage_context_overflow`. The direct route (and
  its tier2 failover) dispatches the client prompt itself and passes `None`,
  so an upstream that rejects it yields `context_length_exceeded`.
- L3: `error_classifier.py` maps a failure to a frozen `ClassifiedError
  (status, type, code, message, param, retryable, retry_after_s, placement)`
  per surface. Chat: 400 `invalid_request_error` / `context_length_exceeded`
  / `param: "messages"` with OpenAI's wording (counts when Kairyu tokenized),
  before SSE headers when detected before dispatch. Messages: Anthropic 400
  `invalid_request_error`, "prompt is too long: N tokens > M maximum" or
  "input length and `max_tokens` exceed context limit: N + M > L, …", plain
  "prompt is too long" without counts. Responses (`placement:
  in_band_on_stream`): unary 400 with `param: "input"`; a streaming request
  whose overflow is found before dispatch (native, `kairyu-proc`, or AUTO
  preflight) gets HTTP 200 `response.created` → `response.in_progress` →
  `error` → `response.failed{code: context_length_exceeded}` with no
  dispatch, metering, or storage; the delegated AUTO Chat error is
  re-rendered this way instead of an HTTP error before the stream. After
  dispatch, the buffered tool, compaction, and AUTO buffered-tool paths (the
  Codex shape, AUTO direct route included) end with the same `error` →
  `response.failed{context_length_exceeded}` pair.
- Admission (framework boundary): the shared contract is the overflow signal
  from L1 to every surface. It had no typed or coded carrier (a bare
  `ValueError`, and an `UpstreamClientError` whose body hides behind a 502),
  so no existing extension point could expose `context_length_exceeded`; the
  mechanism is a `code` attribute plus the L3 classifier table, and nothing
  in the change is example-owned. The m20 D10 record of this WP is pending
  the m20 design doc.
- Deviation: over an OpenAI-compatible upstream, Chat and Messages streams
  learn of the overflow only after Kairyu committed HTTP 200 SSE headers, so
  they render it as the stream's in-band error (Chat error frame, Messages
  `error` event) where OpenAI, vLLM, and Anthropic answer 400. Fixing it needs
  the upstream status before the headers commit.
- Known limits: (a) AUTO streams through `_stream_orchestrator` (Chat AUTO
  stream; the Responses AUTO relay without tools) see only the error type of
  a post-dispatch failure, so a direct-route overflow is not reported as
  `context_length_exceeded` and an internal-stage overflow is not logged with
  its reason; deferred to WP-18 (AUTO unification). (b) On the Responses live
  text path an upstream-reported overflow emits `response.output_item.added`
  and `response.content_part.added` before `error` (fixed by WP-17a's lazy
  item open). (c) A stage failure the conductor swallows loses its overflow
  cause and stays a generic 502.

**Amendment 2026-10-05 (M20 WP-07) — nullable `param` on Chat errors.**
OpenAI's `ErrorResponse` requires `param` (null when no single parameter is
at fault); Kairyu's Chat bodies had it only when one was. Every error rendered
through `error_classifier` now carries it: `ChatRequestError.payload()`,
`sanitize_backend_error` (the 502 `backend_error`),
`invalid_request_payload`, the engine admission 429s (`ChatAdmissionErrors`)
and the middleware's 401, 403, 413 and 429 (`send_error`, tenant limits
included). Status, type, code, message and `Retry-After: 1` are unchanged;
Messages keeps the Anthropic envelope and Jev its own. Hand-built Chat-family
bodies (embeddings, async requests, batches, admin routes, Chat stream error
frames) still omit it; they are outside M20's Responses scope (m7 D5
amendment, known limit b).

**Amendment 2026-10-05 (M20 WP-30) — the in-process vLLM adapter fails closed
and reports exact usage.** `VLLMBackend` mapped only vLLM's sampling fields and
validated only `forced_token_ids`, `chat_template_kwargs`, `assistant_prefill`
and the prompt kind. Chat `logprobs`, `prompt_logprobs`, `best_of`, every
`extra_args` key (so `response_format` and Responses `text.format`) and strict
tools were dropped and the request ran unconstrained; every result had
`usage=None`, so the D1 word-split fallback (its recorded limitation) billed
and settled tenant budgets from an approximation.

- `validate_request`, and `stream` for direct `EngineBackend` callers, raise
  one `ValueError` naming each unhonored field (`best_of`, `logprobs`,
  `prompt_logprobs`, `forced_token_ids`, `extra_args.<key>`,
  `chat_template_kwargs`, `assistant_prefill`, `tools[i].function.strict`)
  before dispatch, so the server answers 400 `invalid_request` as for the
  native engine's surface check. `to_vllm_sampling_kwargs` rejects its own
  unmapped sampling fields.
- Usage follows D1 from each cumulative `RequestOutput`: `prompt_tokens` is
  `len(prompt_token_ids)` once (also for `n > 1`), `completion_tokens` sums the
  completions' token IDs, and `cached_tokens` is `num_cached_tokens` (0 when
  vLLM reports none).
- Not covered: D3 logprobs and D4 structured outputs stay unsupported inside
  this adapter; a deployment that needs them serves vLLM through the `openai`
  backend (`upstream: vllm`), which forwards both. `reasoning_effort`,
  `tool_choice` and `parallel_tool_calls` keep the native engine's semantics
  (Kairyu chat template, public-boundary tool gate); M20 WP-10 owns effort.
- Admission (framework boundary): the shared contract is this section's
  request-surface truthfulness for an `EngineBackend`. Only the adapter sees
  the dropped intents, so no other extension point can reject them. The
  regression (Chat `logprobs`/`response_format` silently ignored, approximate
  usage) is independent of any example. The mechanism is `validate_request`
  plus `_to_result`, and nothing in it is example-owned (m20 admission row 30).

## 3. Non-goals

- Orchestrator (`kairyu-auto`) real usage accounting and streaming (M11).
- `/v1/responses`, `/v1/embeddings`, vision content-parts (M11).
- Prompt-caching *pricing* signals (M11 tenancy/ledger).
- `best_of`, beam search.

## 4. Phasing (each green: pytest + ruff, cov ≥ 80%)

1. D1 usage truth (+ batch_routes/openai_backend updates).
2. D2 chat templates (+ goldens).
3. D3 logprobs + /v1/completions + n>1.
4. D4 server-level structured-output proof.
5. D5 bench fixes.

## 5. Verification

- Usage matches the tokenizer exactly (kairyu backend); `cached_tokens > 0` on
  a repeated ≥1-page prefix; `include_usage` final-chunk shape.
- Template goldens and rendered token IDs byte-match live Transformers;
  unresolved production models fail closed and legacy rendering requires an
  explicit per-model opt-in.
- OpenAI SDK round-trips chat + completions + logprobs against the ASGI app;
  `n=3` streaming interleaves correct indices; seeded n>1 reproducible.
- 400 (not 500) on echo/suffix/best_of/malformed response_format.
- serving_bench smoke vs the mock server produces a results JSON with the
  TPOT method labeled.

## 6. Review record

2-reviewer agent panel, 2026-07-03 — both APPROVE-WITH-AMENDMENTS; applied
inline above:

- **OpenAI-compat reviewer**: full include_usage chunk contract (null on
  non-final chunks, field omitted without stream_options, 400 on
  stream_options without stream); n>1 usage aggregation rule; HF Jinja
  environment exactness (trim_blocks/lstrip_blocks/loopcontrols/HF tojson —
  goldens would not byte-match otherwise); double-BOS guard + special-tokens
  map + tools=None; tool_calls.arguments string→dict before render; `bytes`
  in logprob entries; chunk logprobs on the choice, not delta; legacy
  completions four-array logprobs shape + caps; seeded n>1 identity at i=0;
  finish_reason wire domain; max_completion_tokens + penalties accepted.
- **Integration reviewer**: the "radix makes n>1 prefill nearly free" claim
  is FALSE against radix_kv insertion-collision semantics — replaced with
  documented independent prefills; completion_response's second consumer is
  batch/worker.py (not batch_routes.py); openai_backend must request
  include_usage upstream and parse the empty-choices chunk; usage=None
  fallback covers all backends (vllm et al.), not only the orchestrator;
  merged n>1 stream carries cumulative snapshots of all n completions +
  sibling aborts + server-side 400 for unsupported n; chat_template becomes a
  per-model DeploymentSpec map threaded to BOTH create_app and BatchWorker,
  rendered after model resolution; TokenLogprob lives in outputs.py
  (cycle-free) with msgpack list encoding; bench must send include_usage and
  timestamp its results filename.

Issue #350 amendment, 2026-08-04 — the old implicit role-concatenation default
was removed. Local tokenizer templates and named special tokens are now loaded
with Transformers-compatible precedence. Every production and lower-level chat
boundary requires a template or model-scoped legacy membership; only the
DeploymentSpec builder auto-selects legacy for deterministic mocks, and VLM
deployments have no preflight exception.
