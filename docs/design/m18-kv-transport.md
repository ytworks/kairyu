# M18 Design: KV Serde + Remote P-D Handoff + NIXL Adapter

Status: **Implemented** (2026-07-03). Reviewed — REVISE applied (1-reviewer
panel with empirical repo verification; §6 binding).
Milestone: M18 (roadmap Track E7/G2 B-series local half)
Date: 2026-07-03
Depends on: M12 (PagedKVPool layer-major layout — chosen FOR this milestone),
M15 (MLA latent pool: v width 0), m6 seams (KVTransport/PageFrame protocol,
KVHandoff, PDCoordinator — all unchanged). Consumed by: deploy day (NIXL/RDMA
links), G2 B-gates.

## 1. Goal

Disaggregated prefill→decode over REAL byte transfer, proven end-to-end on
CPU: two engine processes, TCP transport, greedy outputs identical to a
single engine AND the transferred KV bytes identical to locally-computed KV.
The NIXL adapter is written now (deferred import + fake-module contract
tests); deploy day swaps the transport constructor.

## 2. Key design decisions

### D1 — `kv_serde.py`: PagedKVPool ⇄ PageFrame

`extract_page(pool, page_id) -> PageFrame`: fragments are the layer-major
slices — `2 * num_layers` fragments per page (`k[layer, page]` then
`v[layer, page]`, contiguous `.numpy().tobytes()`), matching the m6 comment
that fragments are per-layer × per-shard. MLA pools have v width 0 → v
fragments are empty bytes (the m15 contract). `inject_page(pool, page_id,
frame)` reverses it with dtype/shape derived from the pool; fragment-count
and byte-length mismatches raise `KVTransportError` (loud). `pool_meta`
fingerprint (layers/page_size/heads/head_dim/v_head_dim/dtype) rides in
`SequenceMeta` extension? No — SequenceMeta is frozen protocol; a separate
handshake message at connection setup validates pool compatibility.

### D2 — `pd_remote.py`: `RemoteKVHandoff` (sync facade over async transport)

Implements the m6 `KVHandoff` protocol so `PDCoordinator` drops it in
unchanged. Prefill side: after prefill completes, `transfer(tokens,
first_token)` extracts the prompt's pages from the PREFILL pool, sends
frames + SequenceMeta over the transport, then waits for the decode side's
ACK carrying the decode-side allocation — **copy-before-commit**: the decode
allocation is returned only after bytes landed (the m6 D4 ordering). Decode
side: `RemoteKVReceiver.serve_one()` — recv frames, allocate in the decode
cache (receiver-side dedup: pages already cached are skipped WITHOUT
injecting — radix reuse wins), inject non-cached pages into the decode pool,
mark computed, ACK with allocation info. Failure paths: transport error →
`KVHandoffError` (PDCoordinator's retry contract).

### D3 — `handoff_stream.py`: `StreamCopyKVHandoff` + StreamProvider

The GPU-phase copy pipeline shape: extraction happens on a side stream.
`StreamProvider` protocol (`begin()`, `synchronize()`): `CpuNoopStream` (tests)
and `CudaStreamProvider`. `StreamCopyKVHandoff` wraps any KVHandoff:
enter-stream → inner.transfer (extract+copy) → synchronize — ordering pinned by
a recording fake.

> **Amended 2026-07-25.** `CudaStreamProvider` is implemented; the original
> "decode compute overlaps the copy" and the "deferred" marker on the class are
> both replaced by this paragraph rather than left standing beside it.
>
> `kv_serde._to_bytes` now copies to host before `.numpy()`. Without that the
> real handoff could not read a CUDA `PagedKVPool` at all (`TypeError: can't
> convert cuda:0 device type tensor to numpy`), so this seam had never run
> against a device pool.
>
> What landed: the extraction copy runs on its own stream, which waits on the
> caller's stream FOR THE DEVICE THE PROVIDER WAS BUILT FOR. Work the caller
> queues after that point is independent of the copy.
>
> What did NOT land, and remains open: decode compute does not overlap the copy.
> `StreamCopyKVHandoff` blocks the host before returning and `PDCoordinator`
> commits before stepping decode, so nothing is queued alongside it. That needs
> the consumer to take a completion EVENT in place of the host-wide wait.
>
> **Production wiring landed 2026-07-26.** `PDCoordinator` previously had no
> production constructor — it existed only in `tests/unit/test_pd.py`, so no
> deployment could reach a `KVHandoff`, which is why the provider had no caller.
> `engine/core/pd_factory.py` now supplies both halves:
> `build_pd_coordinator()` assembles a prefill/decode pair from a checkpoint with
> both engines placed by the same probe the single-process path uses (overridable
> per component, so tests inject a placement instead of probing), and
> `build_kv_handoff()` picks the handoff from where the KV lives — plain for a
> host pool, `StreamCopyKVHandoff` over a `CudaStreamProvider` bound to the
> pool's own device for a device pool.
>
> **Amended 2026-07-26 (review [P1]).** The first version of that constructor
> wrapped `LocalKVHandoff`, which allocates in the destination cache and marks it
> computed without touching either pool. The two halves own separate pools, so it
> published the decode pool's untouched pages as computed and decode continued
> from KV that was never written — silently, since nothing raises and the token
> counts are unchanged. `pd.LocalCopyKVHandoff` is the corrected inner handoff:
> destination allocate → skip the leading `len(cached_pages)` source pages
> (receiver-side dedup skips the COPY, not the pages) → direct pool-to-pool page
> copy → mark computed, releasing the allocation instead of publishing a
> half-written prefix if the copy raises. It is a direct tensor copy rather than
> `RemoteKVHandoff`'s serde round-trip because in-process there is no wire: a
> device pair stays D2D instead of paying D2H + H2D. `LocalKVHandoff` is now
> documented as a test double, not a deployment option.
>
> **Overlap landed 2026-07-26.** `StreamCopyKVHandoff(..., defer=True)` records a
> completion event instead of blocking. The producer can queue its next step
> while the copy runs — measured on device: the deferred form returns strictly
> faster than the blocking one on the same work, the copied values are correct
> once the event is settled, and on the coordinator the copy's device-time
> interval overlaps the next prefill forward's
> (`test_the_copy_overlaps_the_next_prefill_forward_on_the_coordinator`).
>
> Deferring is a CONSUMER CONTRACT with two halves, not one. Nothing may read the
> destination pages before the event — and nothing may reuse the SOURCE pages
> before it either. `PDCoordinator`'s m6 D4 rule releases the prefill-side
> allocation the moment the transfer returns (`update()` → `commit_and_release`
> pool-frees the tail page, `abort()` → `release_preempted` frees all of it), so
> a deferred copy would be reading pages the next prefill step can allocate and
> overwrite on the caller's stream. Waiting on the destination side does not stop
> that; it is a source-side read/write race.
>
> The coordinator therefore owns the lease. `PDCoordinator._settle_handover` is
> the single point where a transfer is completed — the prefill-side commit, the
> abort on a failed copy, and the decode-side `resume_with_kv` all go through it
> — and it calls `gate_pending()` first. The gate is stream-ordered, not a host
> block (`event.wait(current_stream)`). A deferring handoff that exposes no
> `gate_pending()` is refused at construction rather than releasing pages under a
> copy the coordinator cannot order against. Events ACCUMULATE: a prefill step
> transfers every prompt that completed in it, and a single-slot `pending_event`
> would silently drop the ordering for all but the last copy.
>
> **Amended 2026-07-26 (review [P1]): where the gate sits is the whole point.**
> The first version settled at the end of the very step that recorded the copy.
> Correct, and no host block — but the next GPU work of any kind (the decode step,
> then the next prefill forward) was queued *after* the gate, so the device
> timeline stayed exactly as serial as the blocking form. Measured on device:
> copy `[11.651, 12.342] ms`, next prefill forward `[12.898, 13.369] ms`.
>
> The settlement is therefore PIPELINED one producer step. `_step_prefill` plans,
> runs its forward, and only THEN settles the previous step's transfers, so the
> gate lands with that forward — and the decode step queued before it — already
> in front of it. `_Handover` is what carries a step's `commit` / `adopt` /
> `retries` across that boundary. Adoption has to travel with the release: it is
> what puts the destination pages in front of the decode runner, which is the
> other half of the m6 D4 rule. The source pages stay leased for the extra step,
> which costs prefill KV capacity — never correctness, because a page the prefill
> scheduler still owns cannot be handed to anyone else. When a step has nothing
> to schedule the settlement happens up front instead: there is no overlap left to
> win, and the leased pages may be exactly what the next prompt needs.
>
> This applies to the deferring path ONLY. A blocking handoff has already finished
> its copy inside `transfer()`, so `_step_prefill` settles it in its own step and
> the request starts decoding immediately, as before.
>
> Production wiring: `build_pd_coordinator(defer_handoff=True)` — the default,
> and the only caller that turns it on, because it is the only one that settles
> it. `build_kv_handoff(..., defer=...)` defaults off for everyone else.
>
> **Serving exposure landed 2026-07-26.** `backend: kairyu` accepts
> `pd_separation: true`, so a deployment YAML can serve through a prefill/decode
> pair. m2 §2.4 reserved that config surface and never wired it. Combinations
> the coordinator does not implement — TP > 1, speculative decoding — are
> rejected rather than silently serving a different topology.
>
> Three things that wiring needs, stated because the first cut of it had none:
>
> - `engine/core/pd_loop.py::PDLoopAdapter`. `EngineLoop` owns ONE scheduler and
>   ONE runner, and `_drain_ops` adds submissions straight to the scheduler it
>   was given. Handing it the coordinator's decode scheduler admits requests
>   into DECODE with no prompt KV and never calls `PDCoordinator.add_request`,
>   and the coordinator has no `execute` for the loop to call. The adapter is
>   both surfaces: submissions enter at prefill, `schedule()` runs one prefill
>   step (prefill → transfer → commit) before planning decode, `execute()` runs
>   the decode runner, and abort/forget/release reach both halves.
> - A handoff that moves BYTES. The two halves own two POOLS, so the
>   accounting-only `LocalKVHandoff` leaves the destination zero-initialised and
>   decode attends over empty KV. `pd.LocalCopyKVHandoff` (amended in above) is
>   the ONE copying handoff in this stack, and the serving path inherits it from
>   `build_pd_coordinator` rather than introducing its own. Greedy equivalence
>   against the single-engine path is the test that holds this down — now through
>   the serving loop as well as through the coordinator directly.
> - Copy ordering. The serving path inherits `build_pd_coordinator`'s
>   `defer_handoff=True`, which is safe for exactly the reason recorded above:
>   `PDCoordinator` holds the prefill-side lease and settles it only behind the
>   copy's completion event. `PDLoopAdapter.schedule()` drives
>   `PDCoordinator.step_prefill`, so the serving loop goes through that same
>   `_settle_handover` gate rather than around it — and inherits its
>   pipelining, so a deferred handoff starts decoding one step later than a
>   blocking one.
>
> **Amended 2026-07-26 (review [P1]): token 0 keeps its sampling identity.**
> The prefill clone ran under an INTERNAL id (`r#p0`) while decode ran under the
> public one, on a different runner with a different `Sampler`. Three things
> broke, none of them visible to a greedy parity test:
>
> - `Sampler._state_for` derives the base seed from the request id when
>   `sampling.seed is None`, so with the default stochastic params token 0 and
>   token 1 onward came off DIFFERENT RNG streams. `EngineRequest.sampling_id`
>   is the fix: the clone keeps its own `request_id` for scheduler bookkeeping
>   and carries the public id as its `sampling_identity`, which is what every
>   runner now keys the sampler under.
> - The grammar enforcer was rebuilt on the decode side from its INITIAL state,
>   having never accepted token 0, so every later mask was computed against the
>   wrong grammar position. `Sampler.hand_over()` moves the state object itself
>   at adoption — seed and matcher together — because a matcher cannot be
>   reconstructed from an id.
> - Token 0's `SampledToken` was reduced to its `token_id`, so its `logprob`,
>   `top_logprobs` and `grammar_terminated` never reached the driver.
>   `resume_with_kv` commits it directly into the decode outputs, so no
>   `execute()` ever reports it — and at `max_tokens=1` the request completes
>   during the handoff with no decode step at all. `PDCoordinator` keeps the
>   whole `SampledToken` and hands it over through `drain_carried_tokens()`,
>   which `EngineLoop` drains and prepends to the request's metadata; grammar
>   termination at token 0 finishes the request at adoption, before decode is
>   planned, which is where the ordinary path's `finish_early` sits too.

### D4 — `kv_transport_nixl_gpu.py`: NIXL adapter

Deferred `import nixl`; constructor takes agent name + peer metadata;
`register()` pins the pool tensors (nixl agent register_memory), `send()`
builds descriptor lists from PageFrame fragments, `recv()` posts/waits.
Contract tests with a fake `nixl` module pin: register-before-send, one
registration (m6 contract), descriptor construction (page→address math),
completion polling. Coverage-omitted; logic CPU-pinned via the fake.

### D5 — Two-process P-D E2E (the flagship)

`tests/unit/test_pd_two_process.py` (spawn, reuses the m16 harness pattern):
process A = prefill engine (tiny llama, real PagedModelRunner) + TCP
transport server; process B = decode engine + receiver. A prefills the
prompt, transfers KV; B decodes to completion. Gates: (1) outputs ==
single-engine greedy; (2) decode-side pool page BYTES == prefill-side bytes
for the transferred pages (torch.equal); (3) receiver-side dedup: second
request sharing a prefix transfers only the non-cached suffix pages' bytes.
Plus the m6 contract suite parametrized over LocalFabric / TcpLoopback
(existing) with serde now carrying REAL pool bytes.

## 3. Non-goals

- RDMA/NCCL-p2p performance, staging-ring sizing (deploy day, B-gates).
- KV-cache quantization for transfer (G4 E-KV); compression.
- Cross-TP resharding on transfer (fragments are per-shard; reshard is a
  G4-era extension recorded in m6).

## 4. Phasing

1. kv_serde + round-trip/mismatch tests (incl. MLA pool).
2. RemoteKVHandoff/Receiver over LocalFabric (single-process async tests).
3. StreamCopyKVHandoff + provider fakes.
4. NIXL adapter + fake contract tests; two-process TCP E2E.

## 5. Verification

- Serde round-trip: extract→inject equality per layer (GQA + MLA pools);
  fragment-count/length mismatch errors.
- Remote handoff over LocalFabric: allocation returned only after inject
  (copy-before-commit ordering observable via a recording pool).
- Two-process E2E gates (D5) — output parity + byte parity + dedup.
- NIXL fake contract: registration-once, descriptor math, poll-until-done.

## 6. Review record (binding amendments, applied)

- **A1 (BLOCKING)**: D2 split in two — RemoteKVHandoff is a SINGLE-process
  KVHandoff (KVAllocation carries a live radix node; it cannot cross
  processes), and the two-process E2E is an EXPLICIT protocol (prefill:
  extract between execute() and update(); decode: recv → allocate → inject →
  mark_computed → resume_with_kv → engine loop). The KVHandoff seam was
  widened to ``transfer(tokens, first_token, pages)`` — a byte-extracting
  handoff cannot recover the prefill tail page from tokens (re-allocate
  returns a FRESH empty tail); PDCoordinator passes the prefill allocation's
  pages.
- **A2 (BLOCKING)**: extraction happens strictly before update() commits —
  commit_and_release frees the TAIL page to the pool where it can be
  reallocated and overwritten.
- **A3**: PageFrame.page_id is sender-local; frames travel in prompt-page
  order and the receiver skips the first len(cached_pages) (radix matches are
  prefix-only) then zips against new_full_pages + (tail_page,).
- **A4**: receiver-side dedup skips INJECTION, not wire bytes (gate weakened
  accordingly; a token-first two-phase protocol that saves wire bytes is
  future work).
- **A5**: no in-band ACK in the two-process E2E (TCP connections are
  unidirectional; empty frames are rejected) — byte parity is asserted via
  per-side sha256 in the m16 JSON-result pattern; copy-before-commit stays
  pinned by the single-process LocalFabric test.
- **A6**: E2E lives in tests/dist (spawn2 harness); the rendezvous file
  carries "host:port|pool_fingerprint" — the transport has no handshake hook,
  so pool compatibility validates at connect time.
- **A7 (verified)**: pool[layer, page] slices are contiguous;
  torch.frombuffer rejects b"" — MLA v fragments assert-empty and skip; serde
  goes through numpy (fp32/fp16/int8), bf16 rides a uint8 view on deploy day
  (recorded).
- **A8**: request params are shared constants in the E2E (production
  metadata sidecar in non-goals); receiver allocate raises → KVHandoffError;
  mark_computed publication is conditional on no uncomputed-sibling
  collision (noted).
