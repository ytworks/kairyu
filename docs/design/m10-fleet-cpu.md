# M10 Design: Fleet Elasticity (M10a) + KV-Aware Routing (M10b) — CPU Halves

Status: **M10a + M10b Implemented** (2026-07-03). Reviewed (1-reviewer
panel with repo-line evidence; §6 binding; covers M10a+M10b).
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
consecutive_failures, draining). API: `add_replica(replica_id, backend)`,
async `remove_replica(replica_id)` (refuses while outstanding>0 unless force,
otherwise removes ownership and awaits shared `shutdown_all` exactly once),
`drain(replica_id)` (stops NEW placements; in-flight completes),
`cancel_drain(replica_id)` (clears only the draining flag), and
`probe(replica_id)`. Canceling a drain does not alter health or outstanding
state. HRW hashing keys on `replica_id` STRINGS (not indices)
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
The k8s-endpoints source remains a THIN deploy-day adapter behind that protocol.

`PoolReconciler(pool, source, factory, default_model?)` passes only complete
identities to `Callable[[ReplicaIdentity], (EngineBackend, health_url)]`.
Discovery's non-empty model wins over the default and its auth value is complete,
including explicit `None`; a missing model fails before mutation. Reconciliation
tracks applied identities and runs deterministic phases: add missing IDs in
desired order, construct-before-drain replacement of changed identities in
desired order, then drain/remove absent IDs in pool order. Removal is awaited and
closes the removed backend through `shutdown_all`; unused candidates are also
closed through that helper, with cleanup failures propagated. In-flight refusal
keeps the applied identity for retry. The reconciler separately tracks only drains
it initiated: if replacement/removal intent reverts to the applied identity, or a
retry factory fails, it uses `cancel_drain` to restore eligibility. External drains
are never canceled, and ownership tracking is cleared when a replica disappears
or a new backend is successfully installed. Server: `POST /admin/drain` marks the
pool replica draining and flips `/readyz` to 503 (existing prober contract).

### D3 — `BatchStoreProtocol` (pure refactor)

The file-backed batch store's surface (`create/get/update/list`) extracted
to a Protocol so M11 tenancy ledgers and tests can fake it. No behavior
change; existing tests must pass unmodified.

### D4 — OTel tracing (`entrypoints/server/tracing.py`)

Deferred `import opentelemetry`; `ServerSettings.tracing=False` default.
`traced_span(name, attrs)` context manager: no-op when disabled or OTel
missing (server runs without the dependency). Spans: gateway request →
pool placement (replica_id, reason) → backend generate; Conductor stages.
Tests use OTel's InMemorySpanExporter (dev dependency) and assert the span
tree + attributes.

### D5 — Helm chart + kind smoke

`deploy/helm/kairyu/`: Deployment (readiness=/readyz, liveness=/healthz),
Service, ConfigMap (DeploymentSpec JSON), values.yaml (replicas, image,
resources; `values-gpu.yaml` arrives in M19). `scripts/kind_smoke.sh`:
kind create → build image → load → helm install → wait ready → curl
/v1/models + a completion → teardown. CI job (`kind-smoke`) runs it on
ubuntu-latest; locally optional. A CPU test pins that the chart renders
(`helm template` golden) so drift fails fast without kind.

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

### D8 — Learning placement

Placement decisions + TTFT into `learning/dataset.py`; offline bandit grid
over (α, β) (pure function over the dataset; no online learning).

## 4. Non-goals

- Real k8s API client (fake-source contract; the adapter is deploy-day).
- Cross-cluster federation; autoscaler execution (M11 F5 logic only).
- KV-event compression/batching tuning (deploy-day).

## 5. Verification

- HRW remap property: removing 1 of N replicas remaps only sessions that
  lived on it; adding remaps ≤ ceil(S/N)+slack.
- Drain: no new placements, in-flight completes, then removable; /readyz
  503 while draining.
- Reconciler diff: add/remove/no-op paths with a fake source; TTL expiry
  drops replicas.
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
- **A8**: BatchStoreProtocol is the FULL 8-method surface (save_file,
  get_file, read_file_content, create_batch, get_batch, list_batches,
  update_batch, recover_orphans) + FileObject/BatchJob models.
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
- **A14**: drain cancellation is ownership-scoped. `PoolReconciler` records only
  drains it initiated; a desired identity reverting to the applied identity, or
  a factory failure on a previously drained retry, cancels only that owned drain.
  Manual/admin drains remain authoritative. Removal, disappearance, and successful
  installation clear the ownership record. `ReplicaPool.cancel_drain()` changes
  only the draining flag, never health or outstanding state.
