# Structured trace contract

Status: implemented for unary and streaming AUTO and direct native-engine responses

## Goal

Expose enough orchestration state for evaluation tools to render the selected
route, actual DAG path, attempts, timings, usage, budget consumption, and safe
failures. For direct native generation, expose request-observed engine and SSE
stage durations so client latency regressions can be attributed. The trace must
not retain prompts or generated text and must not break clients that use the
existing string trace.

## Compatibility

- Clients opt in with `X-Kairyu-Trace: 1`.
- Existing AUTO `kairyu_trace: string[]` values remain unchanged. Direct native
  responses use the same field for bounded human-readable stage summaries.
- `kairyu_trace_v2` is an additive response field.
- `kairyu_route` and both trace fields are declared in the OpenAPI response
  model for generated-client and capability discovery. `kairyu_route` remains
  AUTO-only; direct responses contain the two trace fields without a route.
- The schema is versioned independently with `trace_version: "2.0"`.
- Without the header, route and trace extension fields are omitted rather than
  serialized as null.
- Streaming returns the opted-in trace fields together on one terminal
  `choices: []` metadata chunk. AUTO also includes `kairyu_route` and preserves
  its legacy SSE trace comment. Events are not emitted live, so clients see one
  stable trace envelope after execution rather than having to assemble mutable
  fragments.
- `verification/l1/performance/serving_bench.py --stage-trace` sends the opt-in header, records
  valid/partial/missing/invalid coverage, reports an observed/missing request
  denominator for every native stage, and computes nearest-rank p50/p99 stage
  durations. Producer labels that are not safe artifact identifiers cause only
  that event to be omitted; they do not invalidate the rest of the envelope.
  The default benchmark path sends no trace header.

## Envelope

```json
{
  "trace_version": "2.0",
  "request_id": "chatcmpl-...",
  "started_at": "2026-07-23T10:00:00.000Z",
  "completed_at": "2026-07-23T10:00:00.500Z",
  "events": []
}
```

The HTTP response ID is used as `request_id`. Event `seq` values start at one
and reflect the stable response order. Concurrent role events may overlap in
time; consumers must use their timestamps rather than infer serialization from
`seq`.

## Orchestration event

Every orchestration event includes:

- `seq`, `node`, `role`
- `kind`: `routing`, `generation`, `verification`, or `synthesis`
- `status`: `success`, `skipped`, or `failed`
- zero-based `attempt`
- logical `worker`, resolved `engine`, and configured `model`
- `timing`: queued, started, optional first-token, completed timestamps
- backend-reported prompt, completion, and cached token usage
- orchestration step and cost budget delta
- a bounded, typed `detail` map
- failure class and retryability, without exception messages

Direct routes produce a router event and one generation event. Conductor routes
produce router plus per-role generation/verification events. MoA routes produce
a router event and one aggregate synthesis event because the current MoA result
does not expose per-proposal timing. The aggregate event resolves the proposal
and synthesizer engines separately; `engine` / `model` contain the distinct
actual identifiers and `detail` preserves each logical role-to-engine mapping.

## Direct native stage event

A direct native-engine trace uses one cumulative event per observed stage. It
keeps trace version 2 and uses the existing additive event surface:

```json
{
  "seq": 1,
  "node": "tokenize",
  "role": "engine",
  "kind": "stage",
  "status": "success",
  "attempt": 0,
  "timing": null,
  "detail": {
    "stage": "tokenize",
    "duration_ns": 125000,
    "occurrences": 1,
    "aggregation": "sum",
    "scope": "request-observed"
  }
}
```

`duration_ns` is a cumulative monotonic-clock duration and `occurrences` is the
number of observations in that sum. Stage names are unique within an envelope.
The currently produced stages are:

- `tokenize`: native prompt token-ID resolution and context validation. For
  `kairyu-proc` with parent preflight enabled, the sum includes both the real
  parent tokenizer encode/validation and the child's token-ID validation.
- `queue_wait`: native request submission until the start of the first
  scheduler call that admits work for the request.
- `schedule`: cumulative wall duration of scheduler calls whose returned plan
  contains the request.
- `prefill`: cumulative submit-to-result-observation duration of native step
  handles containing the request's prefill work.
- `decode_step`: the equivalent cumulative duration for decode step handles.
- `detokenize`: cumulative incremental-detokenizer push and finalize duration.
- `sse_write`: cumulative application-side encode/yield-to-ASGI-resume duration
  for content, finish, and optional usage chunks. It excludes the terminal
  trace chunk, `[DONE]`, network RTT, and client-side parsing. A gateway sums
  its own observation with any propagated Kairyu-replica observation.

These are request-observed wall durations, not exclusive CPU/GPU ownership
times. Batched requests can each observe the same schedule or device interval;
pipeline overlap means the stage values can overlap and must not be added to
reconstruct end-to-end latency.

## Data minimization

The structured trace never contains:

- the raw prompt or chat messages
- generated answers, proposals, or verifier text
- authorization headers or API keys
- raw backend exception messages
- user or session identifiers

Router feature counts and the existing bounded route reason are allowed.
Failures expose only the exception class. Output capture, if needed by an
evaluation application, is a separate opt-in responsibility outside Kairyu.

## Timing semantics

Timestamps are UTC RFC 3339 strings with millisecond precision.

- `queued_at`: immediately before budget reservation / dispatch preparation
- `started_at`: immediately before the backend call
- `first_token_at`: null for unary backend calls
- `completed_at`: after the backend result or controlled failure is observed

For a streamed final worker/synthesizer, `first_token_at` is the time Kairyu
observes its first non-empty cumulative backend result. Unary calls leave it
null.

Direct native stage events do not use RFC 3339 event timestamps: `timing` is
null and their bounded monotonic measurements live in `detail.duration_ns`.
The envelope timestamps still delimit the HTTP request observation.

Stage measurement is diagnostic and strictly opt-in. With no trace header, the
native tokenizer, scheduler, runner, and detokenizer timing branches and their
per-request accumulators remain inactive. Kairyu-to-Kairyu OpenAI-compatible
replica calls propagate the opt-in and validate the returned scalar metrics.
An older replica or a selected external vLLM/SGLang/OpenAI-compatible backend
may return no engine-owned stages. A Kairyu gateway can still publish its local
`sse_write`, making the envelope partial; consumers report every absent native
stage as missing coverage, never as zero latency. A target that returns no
trace envelope at all is overall missing.

## Usage and failure finalization

AUTO usage adds `orchestration_input_tokens` and
`orchestration_output_tokens`. They are the cumulative backend-reported totals
for every internal call, including retries, verifier calls, fallback engines,
proposals, and synthesis. Non-stream responses always carry them. Streaming
follows the existing OpenAI contract: the usage-bearing terminal metadata
chunk is present when `stream_options.include_usage` is true; internal
accounting still finalizes on every dispatched stream, including disconnects.

Direct, Conductor, and MoA publish cumulative values through a synchronous
accounting observer between the execution and HTTP layers. It is not an SSE
event or a task/queue bridge: each backend-reported usage update replaces the
request owner's latest total immediately. This lets the owner commit completed
pre-final and partial-final usage even if a client disconnects between
user-visible events. On a backend failure, Kairyu emits known partial usage and
the opt-in structured trace before the sanitized SSE error. Failure events
expose only the exception class; arbitrary exception messages remain
server-side.

## Extension rules

- Existing fields keep their meaning within trace version 2.
- New optional event fields may be added without changing the major version.
- Renaming, removing, or changing field semantics requires a new major version.
- Unknown `kind`, `status`, and detail keys must be ignored by consumers.
- UI-specific layout coordinates and persisted prompts do not belong in this
  server contract.
