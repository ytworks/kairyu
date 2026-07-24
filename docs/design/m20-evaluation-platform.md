# M20 Design: Reproducible Evaluation Platform

Status: **Foundation and GPQA Diamond implemented** (2026-07-24; the remaining
benchmark adapters and the legacy cutover are pending).
Milestone: M20
Date: 2026-07-24
Depends on: M7 production lifecycle patterns, M9/M11 OpenAI-compatible wire
semantics, G6 competitive-proof requirements.

## 1. Goal

Replace the foreground-only `kairyu bench` quality suite with a reproducible
evaluation platform that can run each supported benchmark independently, keep
durable partial evidence, compare only compatible protocols, and produce a report
after success, cancellation, partial completion, or failure.

M20 covers exactly these quality benchmarks:

1. SWE-Bench Pro
2. Terminal-Bench 2.1
3. LiveCodeBench v6
4. LiveCodeBench Pro
5. Humanity's Last Exam
6. CharXiv Reasoning
7. GPQA Diamond
8. SciCode
9. τ³ Banking
10. Artificial Analysis Long Context Reasoning
11. MRCR v2

The performance and operational tools under top-level `bench/` are not part of
this catalog and remain supported.

## 2. Decisions

### D1 — Replace the quality suite instead of preserving compatibility

The new package is `kairyu.evaluation`, exposed as `kairyu benchmark`. The old
`kairyu.bench` package and `kairyu bench` entry point remain only while the
benchmark-by-benchmark implementation series is in flight. The final cutover
removes them; no permanent compatibility adapter is retained.

This cutover is limited to the accuracy-oriented quality suite. Top-level `bench/`
performance, capacity, GPU, and operational tooling remains supported and is not
renamed, archived, or deleted by M20. Legacy quality-suite results are not silently
deleted or promoted into comparable native runs. A one-time importer records them
as immutable legacy artifacts with checksums and incompatible/unknown protocol
provenance.

### D2 — Profiles and upstreams are immutable inputs

`fugu-2026`, `official-latest`, and `smoke` are versioned profiles. A lock records
repository commits, dataset revisions, dependency locks, patches, and container
digests. Updating `official-latest` creates a reviewed snapshot; it never mutates
`fugu-2026`.

The Fugu report does not publish every commit, seed, patch, judge snapshot, or
simulator snapshot needed for byte-exact reproduction. Unknown fields are recorded
as unresolved rather than guessed. A run may use a fixed best-effort implementation,
but it cannot claim exact comparability while critical fields are unresolved.

### D3 — Canonical protocol identity controls comparison and resume

Each run has a canonical `ProtocolSignature`. Canonical JSON is UTF-8, key-sorted,
compact, and restricted to validated protocol fields. Nested JSON inputs are deeply
immutable after validation, and non-finite numbers are rejected before hashing.
SHA-256 is the protocol identity.

Comparison has three outcomes:

- `exact`: hashes match; a future reviewed equivalence record may explicitly promote
  another pair, but the foundation never infers that promotion;
- `near`: critical fields match and only the reviewed non-critical
  `schema_version` or `harness_version` differs;
- `incompatible`: a critical field differs or evidence needed to compare it is
  missing.

Resume creates a new immutable successor run and requires both the same protocol hash
and the same SHA-256 of the ordered item-ID/input-hash manifest. Missing identity
evidence rejects resume. The successor starts in `pending` and replays preparation;
each adapter may reuse a checkpoint only after its item hashes match. The predecessor
run ID, terminal state, events, and artifacts are never reopened or overwritten. A
reference cannot claim `exact` or `near` comparability without a protocol hash. A
smoke, sample, or partial result is never ranked against a published full result, even
if protocol fields otherwise match.

### D4 — Full execution fails closed

Full execution requires all of:

- mode `full`;
- an explicit confirmation flag;
- `BENCHMARK_ALLOW_FULL_RUN=1`.

`CI=true` rejects full execution regardless of the other inputs. The guard belongs
in the shared orchestration boundary and is also called by every external entry
point, so CLI, API, or direct adapter use cannot bypass it.

Smoke execution is offline by default and uses synthetic fixtures plus a fake
OpenAI-compatible connector. Sample execution requires an explicit limit or sample
IDs and is always reported as unofficial.

### D5 — Durable control metadata and immutable artifacts are separate

The first implementation is a single-controller `ControlStore` backed by SQLite in
WAL mode. It owns run state, ordered events, worker claims, leases, and artifact
metadata; adapter PRs add item/checkpoint persistence behind the same interface. The
interface permits a future PostgreSQL implementation without changing adapters.

Large evidence stays in an artifact tree:

```text
benchmark_runs/<run-id>/
  manifest.json
  protocol.json
  events.jsonl
  predictions.jsonl
  item_results.jsonl
  metrics.json
  errors.jsonl
  usage.json
  references.json
  report.json
  report.md
  report.html
  logs/
  upstream/
```

Artifact publication validates containment, pins every path component with no-follow
directory descriptors, rejects symlink traversal, enforces a configurable per-artifact
byte limit (64 MiB by default), and holds a store-owned publication transaction guard
across the publish boundary. Active workers use their lease generation. A terminal,
non-leased run normally receives a version-fenced finalization token solely for `report.json`, `report.md`, and
`report.html`. A queued run can be cancelled before any worker lease exists, so a
cancelled-run token has one narrower exceptional capability: it may create the exact
standard aggregate paths listed above from the already-frozen run/job snapshot. That
token still cannot create checkpoints, logs, arbitrary paths, or upstream outputs. All
other execution evidence requires an active worker lease. The store writes and fsyncs a
same-directory temporary file, then creates the destination atomically without replacement through the pinned
parent descriptor. Reads, idempotence hashing, and parent-directory fsync use the
same bounded descriptor-relative path. A byte-identical retry is idempotent; different
existing bytes are a conflict and are never overwritten. Artifact metadata is registered under the
same publication fence. Report regeneration reads artifacts and local reference
snapshots only. Aggregate paths are published once per run; resumable checkpoints use
item-scoped paths under `upstream/checkpoints/`.

Lease expiry is evaluated from a store-owned clock only after SQLite acquires its
write lock; callers cannot supply authorization timestamps. Jobs persist cancellation
intent. A controller can cancel queued or leased work; cancelled and failed status
retains an independent partial-evidence flag when item counts show saved work.
Resuming a cancelled, partial, failed, blocked, or needs-user-action attempt
atomically creates one new run ID and job with explicit lineage, while the source run
and its artifacts remain immutable. A
completed attempt cannot be resumed. Every worker mutation and filesystem publish is
fenced by job, run, worker, and lease-attempt generation, or by the terminal run
version during final report publication.

### D6 — Evaluation workers are separate from inference gateways

The CLI and optional benchmark control API submit durable work. Dedicated evaluation
workers claim it with leases and heartbeat. The inference gateway never receives a
Docker socket or writable evaluation volume.

Benchmark adapters run trusted orchestration and communicate with pinned upstream
harnesses through a versioned JSON/JSONL protocol. Generated code and third-party
repositories run only in isolated executors. Code-only sandboxes have no network.
Agentic tasks receive no provider API key; an allowlisted credential broker injects
credentials outside the sandbox.

Harnesses that require privileged nested containers, notably LiveCodeBench Pro,
require a disposable VM or an equivalent strong isolation backend. Unsupported
isolation produces `blocked`, not a host execution fallback.

### D7 — Target, judge, and simulator are independent connectors

The evaluation connector preserves multi-turn messages, image parts, tool calls,
streaming, usage, latency, finish reasons, request identifiers, and structured
errors. Target, judge, and user simulator are separate roles with separate endpoints,
model IDs, capability checks, and secret references.

Secrets are never fields in protocol, manifest, report, or artifacts. The persistence
boundary scans structured payloads, persisted identifiers, artifact names, and common
text encodings against a run-scoped digest registry without retaining plaintext.
Workers pass an allowlist of environment variables instead of copying the ambient
environment.

### D8 — Reports and references are evidence, not score decoration

Every completed, cancelled, partial, or failed run produces JSON, Markdown, and
self-contained HTML reports. The partial-evidence flag is independent of terminal
status, so a cancelled or failed run can truthfully retain completed item evidence.
Benchmark adapters define their real metric names; the platform does not relabel
every score as accuracy.

Smoke, sample, and partial reports state that they are unofficial and not full-suite
accuracy. They do not extrapolate, calculate deltas against published full scores,
or produce ranks.

Reference results are append-only versioned data with source, locator, date,
protocol, and evidence hashes. Refresh creates a reviewable proposal and never
silently overwrites an existing record. Report generation performs no web scraping.

## 3. Foundation boundaries

The foundation PR establishes:

- schemas and the exact eleven-entry catalog;
- canonical protocol hashing and comparison;
- smoke/sample/full guards;
- immutable, create-once, publication-fenced artifacts;
- SQLite migrations, events, state compare-and-swap, store-clocked worker leases,
  controller cancellation, report-only terminal finalization, and identity-fenced
  immutable successor creation;
- the non-destructive `kairyu benchmark list` entry point.

It deliberately does not:

- download an official dataset;
- run an upstream harness or container;
- call a model, judge, simulator, or paid API;
- claim that any benchmark is runnable;
- reuse benchmark-specific item checkpoints before an adapter verifies their hashes;
- remove the legacy command before replacement coverage exists.

Each following PR adds one benchmark adapter, its pinned runtime/profile metadata,
and four-layer tests. The final PR archives legacy results and removes the legacy
package, dependencies, configuration, documentation, and command.

### GPQA Diamond implementation

The first adapter pins the GPQA behavior used by EvalScope 1.8.1: zero-shot
single-answer prompting, one sequential Python-MT choice shuffle with seed 42, the
upstream answer extraction behavior, and mean exact-match Accuracy on the 198-item
Diamond split. Kairyu also records its explicit bounded request policy: one sequential
chat-completion call per item, temperature 0, top_p 1, at most 1,024 output tokens, a
120-second per-attempt timeout, and the connector retry limit. This is a reviewed
compatibility policy, not a claim of byte-identical provider behavior. The protocol
also pins the local compatibility module by SHA-256 alongside the dependency lock. The
implementation is local and offline; ordinary CI uses only a two-item synthetic fixture and a fixed fake
connector.

GPQA never downloads gated data. Non-smoke preparation requires a manually approved
local CSV or JSONL snapshot, an explicit access acknowledgement, and the caller-provided
SHA-256. Parsing and hashing use the same bounded, no-symlink byte snapshot. Public item
IDs are hashes, and official questions and choices are excluded from control metadata
and reports.

`official-latest` and `fugu-2026` remain fail-closed for official comparison until a
reviewed dataset-byte lock resolves their dataset provenance. Smoke, sample, partial,
cancelled, and provider-model-mismatched results are always unofficial. Stored Fugu
table values remain incompatible reference evidence and therefore produce no delta or
rank. Provider API version, runtime attestation, source retrieval date, worker/provider
hardware, and monetary cost remain explicit unknowns with reasons rather than guessed
values.

## 4. Implementation order

1. Foundation.
2. GPQA Diamond.
3. Humanity's Last Exam.
4. CharXiv Reasoning.
5. MRCR v2.
6. Artificial Analysis Long Context Reasoning.
7. LiveCodeBench v6.
8. LiveCodeBench Pro.
9. SciCode.
10. SWE-Bench Pro.
11. Terminal-Bench 2.1.
12. tau3 Banking.
13. Legacy archive and atomic cutover.

The order establishes the shared MCQ path first, then multimodal and long-context
connectors, then hostile-code isolation, then agentic/multi-role orchestration.

## 5. Verification boundary

Ordinary CI is offline and must not start a full benchmark, call a paid API, or
download an official dataset. It verifies schemas, protocol identity, full-run
rejection, storage recovery, adapter contracts, fake connector behavior, synthetic
end-to-end execution, cancellation, partial reports, reference comparison, and
secret redaction.

Optional upstream startup smoke tests may read one official sample, use a fake model,
limit turns and calls, cancel intentionally, and require a partial or cancelled
report. They are not evidence that a full benchmark completes or reproduces a
published score.
