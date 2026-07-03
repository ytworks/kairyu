# M9 Design: Truthful API — Usage, Chat Templates, Logprobs, Structured Outputs

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (2-reviewer agent panel,
2026-07-03; all amendments applied inline, see §6).
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

### D2 — HF Jinja chat templates; legacy concatenator stays the default

- `chat_template.py` rewritten around `ChatTemplate`, matching HF's
  `_compile_jinja_template` exactly (amended — anything less breaks
  byte-match): `jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True,
  lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])`, globals
  `raise_exception`/`strftime_now`, and **HF's `tojson`**
  (`json.dumps(..., ensure_ascii=False)` — Jinja's builtin html-escapes
  `<>&'`). Render context: `messages`, `tools` (**None, not [], when absent**
  — templates gate on `is not none`), `add_generation_prompt=True`, and the
  tokenizer's full special-tokens map. Assistant `tool_calls.arguments`
  arriving as JSON strings are parsed to dicts before rendering (HF
  convention; Qwen templates `| tojson` them). The templated prompt is
  encoded with `add_special_tokens=False` (HFTokenizer already does) —
  templates emit `bos_token` themselves; double-BOS would corrupt both
  generation and the "truthful" prompt count.
  `render_chat(messages)` (legacy concatenator) keeps its signature and
  remains the default when no template is configured.
- Per-model config (amended — per-MODEL, not per-replica):
  `DeploymentSpec.chat_templates: dict[str, str] | None` (model name →
  inline template or `*.jinja` path) — a single map avoids
  BackendSpec-vs-PoolSpec ambiguity and stays out of `options` (factory
  kwargs). `builder.py` threads the same mapping into BOTH `create_app` and
  `BatchWorker` (batch and HTTP must render identical prompts); `app.py`
  renders AFTER model resolution. Tool schemas render in-template; the
  `<tool_call>` output-side parse stays.
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
  in `EngineLoop` (per-id decode; byte-level BPE fragments may render U+FFFD —
  `bytes_` from `token.encode()` is the lossless fallback, caveat recorded).
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
- `/v1/completions` (legacy text): `CompletionRequest` (`prompt: str |
  list[str]`, `max_tokens`, sampling fields, `logprobs: int | None` — legacy
  top-k int, capped at 5 with 400, `stream`, `stream_options`). No chat
  template applied; ids prefixed `cmpl-`; `object: "text_completion"` for
  responses AND stream chunks (not delta-shaped). Legacy logprobs is the
  four-parallel-array shape built from the same TokenLogprob tuples:
  `tokens[]`, `token_logprobs[]`, `top_logprobs[] | null`, `text_offset[]`
  (offsets from 0 within `text` — echo is rejected, origin documented).
  `echo`, `suffix`, `best_of` → 400 with a clear message.
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

`bench/serving_bench.py`: `--api-key` (Authorization header); token-granularity
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
- Template goldens byte-match; live transformers cross-check; legacy default
  unchanged.
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
