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

The GPU-phase copy pipeline shape: extraction happens on a side stream so
decode compute overlaps the copy. `StreamProvider` protocol (`stream()`,
`synchronize()`): `CpuNoopStream` (tests) and `CudaStreamProvider`
(`*_gpu`-style deferred, tiny). `StreamCopyKVHandoff` wraps any KVHandoff:
extract-on-stream → synchronize → inner.transfer — ordering pinned by a
recording fake.

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
