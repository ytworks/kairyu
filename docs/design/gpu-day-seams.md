# GPU-Day Seam Changes (repo-review Phase 6)

Status: **design + contract tests on CPU; implementation is GPU-day work.**

The 2026-07-04 full-repo review found five CPU-pinned abstractions that will
break — or silently produce wrong results — when real CUDA/NCCL/FlashInfer land.
They are grouped here because each is a *protocol* change that must be designed
and contract-tested on CPU before hardware time, but cannot be fully *validated*
without a GPU (that is the point: the seam breaks exactly when kernels replace
the CPU references). Land these before the `docs/gpu-runbook.md` perf gates.

Priority order: C5 (silent corruption, contract test exists) → C4 (unreachable
perf gate) → E3 (loop unification) → TP + KVTransport (widen before fabric).

---

## C5 — CUDA-graph static-buffer contract (CRITICAL, **IMPLEMENTED**)

**Bug (fixed).** `GraphStepExecutor._copy_in` wrote `token_ids`/`positions` in
place but rebound `page_tables`/`seq_lens` via `object.__setattr__`. A real CUDA
graph replays fixed kernels over fixed memory and can NEVER see a post-capture
Python-attribute change, so every replay attended over the capture-time scratch
page → silent wrong logits. `FakeGraphBackend` masked it by re-invoking
`fn(static_batch)` and reading the new attributes.

**Fix (landed).** `DecodeBatch.page_tables` is now a `[B, max_pages] int32`
device tensor (padded with the scratch page) and `seq_lens` a `[B] int32`
tensor; `GraphStepExecutor` pre-allocates the static buffers per bucket and
`_copy_in` writes ALL four inputs in place. A page table wider than the captured
`max_pages` falls back to eager (never silently truncated). `build_decode_batch`
pads ragged page lists. `SnapshotGraphBackend` (a faithful graph that sees
in-place writes but not attribute rebinds) plus
`test_graph_replay_reflects_current_page_tables` now pass — the contract is met.
Remaining GPU-day wiring: the `PagedModelRunner` decode path consumes the padded
tensor when the real graph backend is enabled.

## C4 — batched cross-request execution (CRITICAL, **IMPLEMENTED**)

**Gap (fixed for decode).** `PagedModelRunner.execute` ran sequences
sequentially; `AttentionBackend.attend(...)` was one-sequence-per-call. N
concurrent decodes = N kernel-launch chains per layer per step.

**Fix (landed).** `PagedModelRunner.execute` runs all single-token decodes in a
step as one batched forward when ≥2 are present. Supported eager and captured
execution share `DenseDecoder.forward_decode_tensors`: token ids, positions,
ragged padded page tables, sequence lengths, and the cached-KV `write_from` mask
stay on device. Torch uses tensorized paged attention; FlashInfer performs one
step-boundary plan and one batched kernel per layer. The old
`forward_decode_batch` list path remains only as compatibility fallback for a
model/attention backend that does not declare the tensor contract.
Byte/token parity, cached/shared-page preservation, and KV-write equality are
pinned on CPU and GPU. A torch-profiler gate measures zero
`aten::_local_scalar_dense` events at B=1 and B=8 for both the tensor path and
the host-metadata compatibility fallback; only the audited pre-fix path grew
with B.

**Prefill completion (#224).** Compatible prefill chunks are represented by
one validated ragged `PrefillBatch`: flat tokens/positions plus request-local
query indptr, page tables, sequence lengths, cached write offsets, and row
ownership. Dense projection/RoPE/MLP run once over the flat token axis;
`PagedKVPool.write_ragged` maps each token to its owner's physical slot without
rewriting shared cached pages; FlashInfer builds one CSR prefill plan and runs
it once per layer. Writable-page overlap or page-table aliasing fails before
execution. B=1, MLA, Torch, and custom backends without an explicit native
capability retain sequential behavior. Real SM120 profiling binds structural
launch reduction rather than a jitter-sensitive latency threshold, and the
formal Qwen3-32B TP8 gate gathers the exact model/plan/run counts from all
ranks. On-device sampling/future-token fill was completed separately in #206.

## E3 — one engine loop with pluggable pipeline depth (HIGH, **IMPLEMENTED**)

**Fix (landed).** Production `EngineLoop` owns schedule-ahead, immutable
`StepInput` submission, oldest-step commit, streaming and late-result cleanup.
`pipeline_depth=1` reproduces synchronous serving; depth 2+ overlaps CPU
scheduling with the serial device lane or a native async/PP runner. Stop-string
holdback, grammar termination, speculation (with variable-result commit
barriers), preemption, chunked prefill and P-D carried tokens all use that same
path. Finished requests retain scheduler/runner state until every scheduled-ahead
surplus result is trimmed. `OverlapEngineCore` and `PipelinedEngineCore` are
explicitly compatibility-only; the production acceptance suite enters through
`EngineLoop`.

## TP — delta broadcast + sampling ownership (HIGH, **IMPLEMENTED**)

**Gap (fixed for the broadcast).** `DistTPModelRunner.execute` broadcast a full
pickled snapshot of every active request's (growing) prompt+outputs every step.

**Fix (landed).** `StateSync` (step_input.py) diffs live scheduler state into a
`StepDelta` — full snapshots only for first-seen or re-allocated (preempted)
requests, small field deltas + the committed-token tail for the rest, dropped
ids to evict finished ones. Both the driver and every worker apply the same
delta to reconstruct snapshot_step()'s exact `RequestSnapshot`s, so
`DistTPModelRunner`/`worker_step_loop` now broadcast O(chunks + committed tokens)
per step instead of O(all active requests' full state). Byte-identical output:
the `tests/dist` TP=2/EP=2/PP=2 spawn parity gates (TP=2 == TP=1) pass unchanged,
plus `test_state_sync_delta_reconstructs_full_snapshot_each_step`.

**Sampling ownership (landed, issue #225).** Rank 0 is the sole owner of RNG,
penalty, grammar, and logprob state. Non-zero ranks run a passive model/KV path:
eager followers skip `lm_head`, every follower skips sampling and public D2H.
Rank 0 places one int64 token in each emitting `ScheduledChunk` slot (partial
prefill slots are `-1`) and broadcasts that fixed-layout device packet on the
bounded model communicator. Every rank adopts the packet-backed device scalar
before the next step. There is no sampled-result object traffic and no rank can
advance from an independently selected token. Missing/extra owner output,
malformed packet layout, a follower entering a sampling path, or collective
failure is fatal with a protocol-specific error.

The choice is intentionally different from official vLLM
`bb3b61f2fd2333ab165ebaba13f133db4210b9f2` (audited 2026-07-28).
`v1/executor/multiproc_executor.py` dispatches sampling to every TP rank and
returns only `output_rank`; its TP path neither broadcasts the selected token
nor checks equality, so RNG/state skew can remain local.
All-rank sampling in Kairyu would duplicate sampler/logprob/D2H work and still
need a token collective to make divergence blocking. The selected protocol uses
one tiny broadcast and centralizes every stateful sampling responsibility.

**Blocking gates.** Fixed-layout, seeded/filter/penalty/logprob, partial-prefill,
mixed-state, structured, release, speculative-overlay, and injected stale-token
tests run on CPU/gloo. Runbook §6 adds real NCCL canonical adoption, an 8-rank
communication gate, and a production Qwen3-32B TP1/TP8 distribution-compatibility
gate. The real TP2 injected production test binds exact packet adoption and
requires the next decode to consume it. The TP8 communication artifact binds
the NCCL overwrite primitive, while the Qwen artifact separately binds complete
rank topology and rank-0-only ownership; neither is mislabeled as a TP8
per-rank adoption digest.

Free-running TP1/TP8 continuation equality is diagnostic only: a valid change in
TP reduction order can move one near-tie and give every later position a
different prefix. The cross-degree gate instead binds complete, finite raw
per-position evidence and the actual rank topology/ownership metadata. It
compares distributions only while the two runs have an identical input prefix.
Before the first divergence, the common selected token's logprob must remain
within the declared tolerance. At the first divergence, the harness extracts
both selected-token values directly from each run's full raw log-softmax and
both reciprocal logprob deltas must remain within that tolerance. This avoids
treating the pre-penalty public top-N set as post-penalty support. Later
positions are retained but
cannot be a binding distribution comparison because their prefixes differ.

The communication verdict binds to exact logical traffic and a worst-rank
CUDA-event steady-state p95 ceiling. The complete p99, maximum, and tail-count
evidence remains diagnostic: a late host launch makes every rank's tiny
collective event observe OS jitter, which must not decide protocol correctness.

**Recorded evidence (2026-07-30).** On 8× NVIDIA RTX PRO 6000 Blackwell Server
Edition with NCCL 2.29.7, the clean-commit protocol run retained 256 worst-rank
samples per B=1/8/16 cell and measured rank-0 broadcast p95 of
0.078400/0.070816/0.075648 ms. Exact overwrite/equality/divergence checks and
stored replay all pass. The clean-commit Qwen3-32B production run retained 43
tokens per TP degree over six mixed requests and proves TP1 local ownership
plus TP8 gloo-control/NCCL-model topology with only rank 0 holding a sampler.
The 41 common-prefix selected-token logprobs differ by at most 0.148189; the
direct reciprocal cross-selected values at the first divergence differ by at
most 0.101015, below the binding 0.25 tolerance. Free-running equality is
41/43 and remains diagnostic. Artifacts:
`bench/results/issue-225-tp-sampling-comm-rtxpro6000-2026-07-30.json` and
`bench/results/issue-225-tp-sampling-qwen3-32b-rtxpro6000-2026-07-30.json`.

## KVTransport — region ownership + source-addressed recv (HIGH)

**Gap.** `PageFrame.fragments: tuple[bytes,...]` and `register(num_pages)` carry
no memory region, so an RDMA transport can't pin the pool through the seam (the
NIXL adapter reaches around it); `kv_serde.extract_page` does per-layer D2H+copy;
`TcpLoopbackTransport.recv(src)` ignores `src`; bf16 serde is unimplemented. G2
B2 (≥70% NIC line rate) is unreachable through bytes-copy semantics.

**Design (CPU-testable now).** Widen `register(pool_descriptor)` with region
info; allow frames to carry `(page_id, region_offset)` alternatives to bytes; add
a `recv(src)` conformance test. **bf16 serde: IMPLEMENTED** — `kv_serde` now
serializes every fragment through a dtype-agnostic uint8 view, so bfloat16 pools
(which numpy cannot represent) round-trip byte-exact like fp32/fp16 (test:
`test_round_trip_bfloat16`). The region-ownership `register(descriptor)` and
source-addressed `recv(src)` widenings remain for the RDMA bring-up.

---

Refs: repo-review report; `engine/core/{step_executor,model_runner,attention/,
worker,kv_transport,kv_serde}.py`, `engine/{engine_loop,kairyu_backend}.py`.
