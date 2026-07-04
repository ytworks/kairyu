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

## C4 — batched cross-request execution (CRITICAL, **IMPLEMENTED (decode)**)

**Gap (fixed for decode).** `PagedModelRunner.execute` ran sequences
sequentially; `AttentionBackend.attend(...)` was one-sequence-per-call. N
concurrent decodes = N kernel-launch chains per layer per step.

**Fix (landed).** `AttentionBackend.attend_batched(...)` (per-sequence contexts,
one call per layer per step); `DenseDecoder.forward_decode_batch` (row-batched
projections/RoPE/MLP, per-sequence KV write to private decode pages, batched
attention); `PagedModelRunner.execute` now runs all single-token decodes in a
step as ONE batched forward when ≥2 are present. The torch backend loops
internally (CPU reference); the GPU FlashInfer backend replaces the loop with one
batched kernel over indptr/indices behind the same signature. Byte-identical to
sequential decode — `test_batched_decode.py` pins `forward_decode_batch[i] ==
forward_tokens(seq_i)` and KV-write equality, and the full parity suite is
unchanged. Remaining GPU-day work: batched PREFILL plan, and on-device batched
sampling with a single async D2H (the CPU sampler already samples per row).

## E3 — one engine loop with pluggable pipeline depth (HIGH)

**Gap.** The production path is the synchronous `EngineLoop.step()`;
`OverlapEngineCore`/`PipelinedEngineCore` are imported only by tests. Overlap +
streaming + stop-string holdback + spec decode + grammar termination have never
coexisted in one loop, and the overlap cores pass live mutable `_RequestState`
across the execution seam (chunked-prefill range drift, decode-ahead IndexError,
preemption races).

**Design.** Converge on one `EngineLoop` with a pipeline-depth knob (depth=1
reproduces today). Make `StepInput`/`snapshot_step` mandatory at submit/execute
so the runner never reads live scheduler state. Run the whole CPU suite through
the unified loop; delete/demote the two unused cores.

## TP — delta broadcast + sampling ownership (HIGH)

**Gap.** `DistTPModelRunner.execute` broadcasts a full pickled `StepInput`
snapshot every step (`dist.broadcast_object_list` — the vLLM-V0 control-plane
mistake), and every rank samples independently from full logits (rank agreement
proven only on gloo/CPU; one non-deterministic GPU kernel forks rank KV state).

**Design.** Delta-broadcast only new/finished requests + committed tokens (the
snapshot machinery already isolates the delta). Promote the rank-divergence
check to a blocking runbook gate. Decide rank-0-samples-and-broadcasts vs
all-ranks-sample now — it changes the worker step protocol. Also wire the runner
into `build_engine_loop`/serve (today real-model TP exists only in `tests/dist`).

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
