# M10 Design: Fleet Elasticity (M10a) + KV-Aware Routing (M10b) — CPU Halves

Status: **M10a + M10b Implemented** (2026-07-03; D7/A13 amended
2026-07-27; D2/D5/A16–A19 amended 2026-07-28). Reviewed (1-reviewer panel
with repo-line evidence; §6 binding; covers M10a+M10b).
Milestone: M10a/M10b (roadmap Track F1/F2; goal G5 base)
Date: 2026-07-03
Depends on: m7 ReplicaPool/JsonlRouterLog, m7 deploy (spec/builder/prober),
m9 server surface, M18 (KV events source shape). Consumed by: G5 fleet
gates, M11 product surface.

## 1. Goal

Thousands-of-GPUs operation needs (a) replicas that come and go without
restarts, (b) a discovery/reconciliation loop, (c) traces for debugging a
distributed request path, and (d) routing that knows where the KV prefix
lives. All logic CPU-testable; k8s manifests exercised by a kind smoke.

## 2. M10a decisions

### D1 — Dynamic ReplicaPool membership (`orchestration/replica.py` rework)

Index-keyed lists → id-keyed `_ReplicaEntry` (backend, outstanding,
consecutive_failures, manual drain owner, drain-lease set). API:
`add_replica(replica_id, backend)`,
async `remove_replica(replica_id)` (refuses while outstanding>0 unless force,
otherwise removes ownership and awaits shared `shutdown_all` exactly once),
`drain(replica_id)` / `cancel_drain(replica_id)` (acquire/release only the
manual owner), `acquire_drain(replica_id) -> DrainLease` /
`release_drain(replica_id, lease)` (opaque independent owners),
`is_manually_draining(replica_id)` (manual owner only),
`entry_generation(replica_id)` (opaque identity for one entry lifetime),
`require_probe(replica_id)`, and `probe(replica_id)`. A remote entry with a
declared health URL starts unvalidated; only a successful readiness probe makes
it healthy and eligible. Entries without a readiness URL remain locally trusted.
A replica is draining while the manual owner or any lease
is active. Removing and re-adding the same ID creates a new entry generation.
Releasing one owner does not alter other owners, health, or outstanding state.
HRW hashing keys on `replica_id` STRINGS (not indices)
— adding/removing a replica remaps only ~1/N sessions (property test).
Constructor accepts `dict[str, EngineBackend]` or the legacy sequence
(auto-ids `"0".."N-1"`, existing tests unchanged). Selection precedence:
healthy ∧ not-draining → session HRW → queue-depth fallback →
least-outstanding.

### D2 — Registry + reconciler + discovery (`deploy/registry.py`)

Discovery carries frozen `ReplicaConfig(address, model?, api_key_env)` values;
the reconciler resolves each one to a complete frozen
`ReplicaIdentity(address, model, api_key_env)` before any pool mutation.
`ReplicaRegistry` stores typed TTL-heartbeat membership
(`register(id, address, ttl_s, model?, api_key_env)`, `heartbeat(id)`,
`alive()` — monotonic-clock injected for tests). `DiscoverySource` protocol:
`poll() -> dict[id, ReplicaConfig]`; `StaticDiscovery` snapshots either string
addresses or typed configs and `RegistryDiscovery` exposes registry snapshots.
The Kubernetes source is a lifecycle-owned in-cluster EndpointSlice REST
adapter behind that protocol. It reads the projected service-account token on
every poll, verifies the API server with the mounted CA, selects one named or
numeric TCP port, and admits only `ready=true`, non-terminating IPv4/IPv6
endpoints. Pod UID is the stable identity when present, with name/address
fallback; dual-stack rows for the same UID are aggregated once using an
explicit address-family preference and deterministic address selection.

`PoolReconciler(pool, source, factory, default_model?)` passes only complete
identities to `Callable[[ReplicaIdentity], (EngineBackend, health_url)]`.
Discovery's non-empty model wins over the default and its auth value is complete,
including explicit `None`; a missing model fails before mutation. Reconciliation
tracks applied identities together with each pool entry's opaque generation and
runs deterministic phases: add missing IDs in
desired order, construct-before-drain replacement of changed identities in
desired order, then drain/remove absent IDs in pool order. Removal is awaited and
closes the removed backend through `shutdown_all`; unused candidates are also
closed through that helper, with cleanup failures propagated. In-flight refusal
keeps the applied identity for retry. The reconciler keeps the opaque lease returned
for each replacement/removal it initiates: if intent reverts to the applied identity,
or a retry factory fails, it releases only that lease. Manual/admin drains remain
active whether acquired before or after the reconciler lease, and manual undrain
cannot release a reconciler lease. Successful same-ID replacement transfers only
the manual owner to the new backend entry; the reconciler lease is not transferred,
and a new remote entry starts unvalidated with fresh failure and outstanding
state. Lease tracking
is cleared when a replica disappears or a new backend is successfully installed.
If an external caller removes and re-adds the same ID between ticks, the changed
generation invalidates every applied identity and lease owned for the old entry.
An absent desired ID then receives a fresh lease on the new entry; a present desired
ID baselines the externally installed entry at the current desired identity without
factory, replacement, or shutdown side effects. A fresh entry's manual owner is
never changed by generation reconciliation.
Each entry generation is a serializable random token. Effective membership
transitions (add/remove, drain/undrain, first successful probe, health ejection,
and restore) enter a reconciler-owned ordered outbox with a source ID,
monotonic sequence, and stable event ID. A sink failure retains the head event
for identical retry before further discovery; already-applied partial
mutations are therefore not lost. Placement rows carry the same generation,
the server-assigned request ID, pool/eligible sizes, and process-ingress to
selection latency.
Server: `POST /admin/drain` marks the pool replica draining and flips `/readyz`
to 503 (existing prober contract).

### D3 — `BatchStoreProtocol` and streaming file storage

The file-backed batch store's surface (`create/get/update/list`) is extracted
to a Protocol so M11 tenancy ledgers and tests can fake it. The surface also
includes owner-scoped lazy input-line iteration, an async streaming upload
transaction, and a transactional JSONL writer. Streaming uploads enforce their
byte limit incrementally and remove partial state on cancellation or failure;
both writers publish metadata only after temporary content is closed and
atomically renamed.

### D4 — OTel tracing (`entrypoints/server/tracing.py`)

Deferred `import opentelemetry`; `ServerSettings.tracing=False` default.
`traced_span(name, attrs)` context manager: no-op when disabled or OTel
missing (server runs without the dependency). Spans: gateway request →
pool placement (replica_id, reason) → backend generate; Conductor stages.
Tests use OTel's InMemorySpanExporter (dev dependency) and assert the span
tree + attributes.

### D5 — Helm chart + kind gates

`deploy/helm/kairyu/`: Deployment (readiness=/readyz, liveness=/healthz),
Service, ConfigMap (DeploymentSpec JSON), values.yaml (replicas, image,
resources; `values-gpu.yaml` arrives in M19). `scripts/kind_smoke.sh`:
kind create → build image → load → helm install → wait ready → curl
/v1/models + a completion → teardown. CI job (`kind-smoke`) runs it on
ubuntu-latest; locally optional. A CPU test pins that the chart renders
(`helm template` golden) so drift fails fast without kind.

G5 F1a adds a separate resource-bounded kind topology: one dynamic gateway,
one headless Service, and a 200-member parallel StatefulSet using a static
standard-library-only mock. Its fixed-seed driver applies ten disjoint
20-pod churn batches at absolute one-minute boundaries under retry-free
open-loop traffic. It joins the echoed server request ID to the raw gateway
placement row, joins the selected replica UID to the mock response, and records
pod/EndpointSlice/resource snapshots plus the exact kill/join schedule.
The verifier rehashes and replays every JSONL sidecar. Pull requests run the
same protocol at smoke scale; the frozen 200-replica profile is scheduled and
manually dispatchable. Images and manifests are built from a clean Git archive,
carry the source revision, and run on digest-pinned kind/Kubernetes inputs.
Gate traffic reaches a fixed NodePort through kind's localhost port mapping;
`kubectl port-forward` is excluded because its process-local file-descriptor
and tunnel lifecycle are not part of the gateway-under-test.

## 3. M10b decisions (implemented after M10a in the same doc's scope)

### D6 — Prefix index + KV-aware selection (`orchestration/prefix_index.py`)

Block-granular approximate trie: `observe(replica_id, token_blocks)`,
`overlap(replica_id, token_blocks) -> int`. `ReplicaPool` gains optional
`prefix_index=` + score `α·overlap − β·outstanding` over power-of-two
random candidates; session affinity remains the tiebreak; `enabled=False`
default. Decision reason `prefix_match` in the router log.

### D7 — RadixKV events → gateway index (`radix_kv.py` event_sink + `kv_index.py`)

`RadixKVCache(event_sink=...)`: emits BlockStored/BlockRemoved
(vLLM-compatible schema) from allocate/commit/evict. ZMQ PUB publisher +
gateway subscriber updating the trie; staleness > 500 ms → graceful
fallback to the approximate trie (chaos test kills the publisher).

**Incremental hash-chain amendment (2026-07-27, issue #220).** Event-enabled
radix nodes retain the SHA-256 continuation state for the canonical
`repr(tuple(prefix))` token stream, the prefix token count, and their local
block-boundary digests. A child copies its parent's continuation once and
feeds only its own token suffix. Tuple closing punctuation is appended only to
a digest snapshot, including the singleton trailing comma, so emitted hashes
remain byte-identical to the original full-prefix protocol. Splits reuse the
existing digest suffix and derive only the new upper node; stored and removed
events read node-local hashes directly. Caches without an event sink allocate
no hashers and perform no SHA work. Reproduce the long-prefix comparison with
`uv run python bench/kv_event_hash_bench.py --tokens 32768 --repeats 5`.

### D8 — Learning placement

Placement decisions + TTFT into `learning/dataset.py`; offline bandit grid
over (α, β) (pure function over the dataset; no online learning).

## 4. Non-goals

- Cross-cluster federation; autoscaler execution (M11 F5 logic only).
- KV-event compression/batching tuning (deploy-day).

## 5. Verification

- HRW remap property: removing 1 of N replicas remaps only sessions that
  lived on it; adding remaps ≤ ceil(S/N)+slack.
- Drain: no new placements, in-flight completes, then removable; /readyz
  503 while draining.
- Reconciler diff: add/remove/no-op paths with a fake source; TTL expiry
  drops replicas.
- EndpointSlice discovery: readiness/termination filtering, named and numeric
  ports, token rotation, deterministic dual-stack UID aggregation, lifecycle
  close, and failure-resilient polling.
- F1a: one gateway plus 200 mock replicas, ten minutes of 10%-per-minute churn,
  zero failed requests, and placement p99 below 10 ms. Raw request,
  membership, pod, EndpointSlice, resource, and kill/join sidecars must pass
  the independent artifact replay.
- Tracing: span tree with InMemorySpanExporter; disabled → zero overhead
  (no otel import).
- Helm: `helm template` golden test; kind smoke in CI.
- M10b: prefix routing beats least-outstanding on a synthetic
  shared-prefix workload (decision counts); staleness fallback chaos test.

## 6. Review record (binding amendments)

- **A1**: auto-ids are "0".."N-1" (Prometheus labels stay stable);
  ``probe()`` accepts int (ordinal) or str id; the router log keeps
  ``replica`` as ordinal and ADDS ``replica_id``; ``outstanding``/``healthy``
  stay insertion-order tuples (+ ``*_by_id`` variants).
- **A2**: in-flight completion on a removed id is a no-op (guarded
  decrement); streams count as in-flight until generator close.
- **A3**: HRW runs over ELIGIBLE entries (healthy ∧ not draining) — draining
  remaps its sessions immediately; all-draining raises like all-unhealthy.
- **A4**: health URLs live with the pool entries (dict[id, url]); the prober
  keys by id; probe() resets failures but NEVER clears draining.
- **A5**: /admin/drain semantics split by node role — replica node: sets
  app.state.draining → /readyz 503 (the prober sees it); gateway: drains a
  pool member; only zero-ELIGIBLE replicas 503 the gateway readyz.
- **A6**: reconciler factory =
  `Callable[[ReplicaIdentity], (EngineBackend, health_url)]`; the default closes
  over `create_backend("openai")` with the identity's address, resolved model,
  and auth environment plus the readiness URL `/v1`-strip rule. Reconcile
  tolerates in-flight remove refusal, closes each unused candidate, and retries
  next tick without changing the applied identity.
- **A7**: registry takes ``now: Callable[[], float] = time.monotonic``.
- **A8**: BatchStoreProtocol is the FULL 11-method surface (save_file,
  save_file_streaming, get_file, read_file_content, iter_file_lines,
  create_jsonl_writer, create_batch, get_batch, list_batches, update_batch,
  recover_orphans) + FileObject/BatchJob/JsonlFileWriter models.
- **A9**: traced_span lives in ``kairyu/telemetry.py`` (L2 must not import
  L3); the gateway request span is a middleware; ServerSection threading
  already copies model_fields — only the field addition is needed.
- **A10**: opentelemetry-sdk added to the dev group (and an ``otel`` extra).
- **A11**: chart mounts the DeploymentSpec YAML at exactly
  /etc/kairyu/config.yaml (the Dockerfile CMD); kind-smoke is a third CI
  job; the gateway image needs the ``fleet`` extra for the D7 subscriber.
- **A12 (M10b)**: the gateway has NO token ids — the approximate trie keys
  on fixed-size TEXT chunks of the prompt (documented approximation); the
  KV-event index is a SEPARATE per-replica block-hash structure with
  staleness tracking; key unification via gateway tokenization is a
  deploy-time option (install tokenizers in the gateway image).
- **A13 (M10b)**: BlockStored emits on the computed False→True TRANSITION
  (mark_computed + commit_and_release decode-extension nodes; never
  allocate; guard the _release double-fire); BlockRemoved only from
  _ensure_free eviction; _split emits nothing; release_preempted emits
  nothing (never stored); vLLM schema fields block_hashes/parent_block_hash/
  token_ids/block_size + ts; replay endpoint out of scope (recorded).
  Block hashes use the node-cached incremental SHA continuation while remaining
  exactly compatible with the original `sha256(repr(prefix))[:16]` values.
- **A14**: drain cancellation is ownership-scoped. `_ReplicaEntry` separately
  stores the manual owner and opaque `DrainLease` owners; effective draining is
  their OR. `PoolReconciler` records the exact lease it acquired, and a desired
  identity reverting to the applied identity or a retry factory failure releases
  only that lease. Manual/admin drains remain authoritative even when asserted
  after a reconciler lease; manual undrain likewise cannot release that lease.
  A successful same-ID replacement carries the manual owner to the new entry but
  not the reconciler lease; health and outstanding state reset with the backend
  entry. Removal, disappearance, and successful installation clear reconciler
  tracking. Applied identities and leases are bound to the pool's opaque entry
  generation, so an external same-ID remove/add also clears old tracking: desired
  absence acquires a fresh lease, while desired presence baselines the fresh entry
  without replacing it. This never clears a fresh entry's manual owner. Releasing
  any owner never changes health or outstanding state.
- **A15**: readiness validation is generation-scoped. A declared remote health
  URL makes a new entry unknown and ineligible until `probe()` succeeds; no URL
  retains locally trusted compatibility. The serve-layer prober performs its
  first tick immediately, snapshots unknown/ejected entries by ID, entry
  generation, and URL, probes them with bounded concurrency, isolates per-entry
  failures, and applies a 200 response only to the same generation. `/readyz`
  and placement use the same validated-and-not-ejected predicate, while
  `probe()` resets failures without clearing any drain owner.
- **A16**: the deploy-day Kubernetes adapter is now production code rather than
  a non-goal. EndpointSlice `targetRef.uid` is the replica identity and the
  source rejects not-ready or terminating endpoints before reconciliation.
  Reconciliation and health changes publish post-transition snapshots through an
  ordered retryable outbox; placement and membership rows share serializable
  entry generations. F1a evidence is fail-closed: no request retry, 429/5xx,
  transport errors, unsent arrivals, missing request-to-placement joins,
  incomplete final gateway membership, missing raw sidecars, or any placement
  p99 window at or above 10 ms may pass.
- **A17 (superseded by A18)**: one dynamic `ReplicaPool` owns one outbound
  `AsyncClient`, shared by every discovered OpenAI backend and its readiness
  prober. Replica removal and replacement never close that transport;
  application teardown stops reconcilers/probers/backends first and then closes
  the sole owner exactly once, cancellation-safely. Data calls retain their
  backend timeout and probes retain their shorter timeout per request. Active
  and idle connection-count limits are open because trusted EndpointSlice
  membership has no declared fleet ceiling; gateway admission and the probe
  semaphore bound active use, while a 30-second idle expiry bounds sockets for
  churned-out Pod IPs. This reduces client and TLS setup from O(replicas) to
  O(dynamic pools), matching the shared-client design used by current vLLM
  routers.
- **A18**: one dynamic `ReplicaPool` eagerly creates one immutable TLS context,
  then each discovered OpenAI backend lazily creates and owns an origin-local
  `AsyncClient` from it. Removing or replacing a replica closes only that
  replica's client, cancellation-safely through the existing backend lifecycle.
  Active data connections remain bounded by gateway admission rather than a
  transport queue; each origin retains at most one idle connection for 30
  seconds. Readiness uses a separately owned client capped at the prober's 16
  concurrent connections. Both paths ignore proxy environment variables because
  EndpointSlice addresses are cluster-internal. Sharing the TLS context removes
  synchronous per-replica CA loading while separate transports avoid httpcore
  1.0's fleet-wide flat connection scan. vLLM's shared aiohttp/reqwest clients
  are not a counterexample: both index reusable connections by origin, whereas
  httpcore 1.0 scans one cross-origin list and performs quadratic idle cleanup.
  A18 therefore supersedes A17's shared-httpx-transport choice while retaining
  its pool-scope TLS setup objective and per-operation timeout contract.
- **A19**: request validation preserves the intersection of all eligible
  replica contracts without repeating an identical immutable contract for
  every replica. A backend may explicitly publish a hashable
  `request_validation_key`; `ReplicaPool` validates one representative only
  when both the concrete backend type and key match. Missing, `None`, or
  unhashable keys remain per-replica, and every distinct type/key remains in
  the intersection. `OpenAICompatBackend` uses its frozen resolved capability
  contract as that key because its validator depends on no address, client, or
  model-instance state; subclasses fall back to per-replica validation unless
  they explicitly publish their complete contract. At 200 equivalent replicas
  this reduces measured validation from 1.785 ms to 0.143 ms median without
  changing placement or rejection semantics. Router and membership records
  still snapshot and enter the bounded queue synchronously, preserving
  event-loop/outbox order and immediate overflow behavior, but JSON encoding
  now executes on the existing lifecycle-owned writer thread. Writer batches
  are bounded by both 128 records and 64 KiB of encoded JSON, preventing a
  burst of large membership rows from retaining the Python GIL across an
  event-loop scheduling quantum while preserving batching for small placement
  rows. Encoding and filesystem failures remain sticky and surface through
  append, flush, or close; no generic executor task or second unordered outbox
  is introduced.
