# M20 Ingress Parity (WP-41) — companion to m20 D21

Status: **Implemented — CPU** (2026-10-05, WP-41). Records the WP-41 part of
m20 D21 (transport hardening); closes M-W-9, G-transport-3 and M-ST-10.
Date: 2026-10-05
Depends on: m20 D21 and C20 (`docs/design/m20-responses-compat.md`), m7 D5/D8
(amended 2026-10-05, WP-41), m20 §8 admission row 41 (O-1).

## 1. Context

Three shared-ingress gaps break OpenAI clients on every surface:

- `X-Request-ID` was set only by `AccessLogMiddleware`, so with
  `access_log: false` the OpenAI SDKs' `_request_id` was `None`, and error
  answers never carried an id for correlation (M-W-9).
- A `Content-Encoding` body reached the JSON parser undecoded: 400. Codex sends
  zstd bodies when it uses ChatGPT auth against an OpenAI base URL
  (`codex-rs/core/src/client.rs@rust-v0.160.0:1617-1626`; fixture
  `harbor-zstd`) (G-transport-3).
- No CORS: a browser client could not call the API, and a preflight met auth
  as an unauthenticated request (M-ST-10).

## 2. Decisions

**I1 — The request id is always set.** `RequestIngressMiddleware`, always the
outermost middleware, assigns a 16-hex id to every HTTP request, stores it in
the request state and appends `X-Request-ID` to every response, errors
included, whether or not access logging is on. The access log, tracing
(`kairyu.request_id`), the Anthropic error body and the engine request id read
the same value. A client-sent `X-Request-ID` is never adopted: the id keys
`GenerationRequest.request_id`, which must be unique in flight, so a chosen id
could collide with another tenant's request. Clients correlate with their own
header (Codex sends `x-client-request-id`). An exception no handler classified
is answered by this middleware, before the response starts, with a 500
`server_error` in the route's dialect and the id, then re-raised so Starlette's
`ServerErrorMiddleware` (outside all app middleware) still logs it.

**I2 — Request bodies are decoded under their own cap.**
`kairyu/entrypoints/server/decompression.py` decodes gzip (`x-gzip` alias,
concatenated members) with the stdlib and zstd (concatenated frames) with the
optional `kairyu[zstd]` extra (`zstandard`, also in the dev group so CI replays
`harbor-zstd`). gzip is in core and zstd is an extra (C20).

- Any other coding, a coding list, or zstd without the extra: 415
  `unsupported_content_encoding`, with `Accept-Encoding` naming what is
  accepted (RFC 7694). `identity` is ignored.
- The decoded size is capped by `ServerSettings.max_decompressed_bytes`
  (deployment `server.max_decompressed_bytes`; the plan's
  `ingress.max_decompressed_bytes`), 64 MiB by default: 413
  `request_too_large`. Every decoding step is bounded (zlib by a 1 MiB output
  step, zstd by 64-byte input slices, at most 2 MiB each), so a bomb is refused
  after at most one step past the cap, never inflated in full. The same cap
  bounds the compressed bytes received: padding that decodes to nothing (empty
  gzip members, zstd skippable frames) cannot stream past the body limits,
  which count decoded bytes.
- zstd frames may need at most an 8 MiB window (RFC 9659); a larger window,
  corrupt data or a truncated member or frame is 400
  `invalid_content_encoding`.
- Chunks are decoded on the request-body worker (`run_request_body_work`), so
  a large body never stalls the event loop.
- Errors render in the route's dialect: OpenAI envelope (Responses rules on
  `/v1/responses*`), Anthropic on `/v1/messages`, Jev on `/v1/systemone`.

**I3 — CORS is optional and off by default.** `cors_allowed_origins`
(`ServerSettings` and deployment `server`; each entry `*` or
`scheme://host[:port]`, checked at load) installs Starlette's
`CORSMiddleware` just outside auth. A preflight is then answered without
credentials; actual requests still authenticate, and every answer, errors
included, carries the CORS headers and exposes `x-request-id`, `retry-after`,
`retry-after-ms` and `x-should-retry` to the browser. Credentials mode is off
(bearer keys, no cookies). With the setting empty, OPTIONS meets auth as
before (401).

## 3. Ingress order

Outer to inner: request id and timing → access log → tracing → CORS → auth →
tenant limits → concurrency → metrics → decompression → body limits → app.

- Decoding sits inside auth, so 401 wins over 415 and nothing is decoded for an
  unauthenticated request; it sits inside metrics, so its 413/415/400 answers
  are counted with their status.
- Decoding sits outside every body limit, and the app sees no
  `content-encoding` or `content-length`, so `max_chat_body_bytes` (and the
  WP-08c Responses body default) bound the decoded JSON.
- The plan's data-flow line lists "decompress → body-limit → auth → tenant →
  request-id"; the implemented nesting puts the request id outermost so every
  error carries it.

## 4. Framework admission (m20 §8, row 41)

1. Shared contract: HTTP ingress for every surface (`middleware.py`, the
   middleware stack in `app.py`).
2. No existing extension point decodes request bodies, answers preflights
   before auth, or sets the id without the access log.
3. Example-independent regressions: a gzip-compressing client got 400; the SDK
   `_request_id` was `None` with the access log off; browsers were blocked.
4. Smallest mechanism: one capped middleware, the id moved into the existing
   outermost middleware, and Starlette's `CORSMiddleware` behind an allowlist.
   Deployment-owned: the cap and the origins. Container images are unchanged:
   operators who need zstd install `kairyu[zstd]`. A Codex custom provider with
   an API key sends plain JSON.

## 5. Verification

- `tests/server/test_decompression.py`: gzip accepted; gzip and zstd bombs
  capped at the default 64 MiB (413); empty-member padding past the cap (413);
  zstd without the extra (415, import failure monkeypatched).
- `tests/server/test_codex_fixture_replay.py`: `harbor-zstd` passes (strict
  xfail removed); zstd acceptance is covered there.
- `tests/server/test_health_metrics.py`: every response, a 401 and an
  unhandled exception's 500 included, echoes a server-generated id with the
  access log off.
- `tests/server/test_auth.py`: a preflight skips auth only when CORS is
  configured; the 401 is readable cross-origin.
