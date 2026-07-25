# Structured orchestration trace contract

Status: implemented on the observability fork branch (unary responses)

## Goal

Expose enough orchestration state for evaluation tools to render the selected
route, actual DAG path, attempts, timings, usage, budget consumption, and safe
failures. The trace must not retain prompts or generated text and must not
break clients that use the existing string trace.

## Compatibility

- Clients opt in with `X-Kairyu-Trace: 1`.
- `kairyu_trace: string[]` remains unchanged.
- `kairyu_trace_v2` is an additive response field.
- `kairyu_route` and both trace fields are declared in the OpenAPI response
  model for generated-client and capability discovery.
- The schema is versioned independently with `trace_version: "2.0"`.
- Without the header, route and trace extension fields are omitted rather than
  serialized as null.
- This first contract covers completed unary responses. Streaming continues to
  expose the legacy SSE trace comment; live structured events are deferred.

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

## Event

Every event includes:

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

First-token timing becomes available when structured live streaming is added.

## Extension rules

- Existing fields keep their meaning within trace version 2.
- New optional event fields may be added without changing the major version.
- Renaming, removing, or changing field semantics requires a new major version.
- Unknown `kind`, `status`, and detail keys must be ignored by consumers.
- UI-specific layout coordinates and persisted prompts do not belong in this
  server contract.
