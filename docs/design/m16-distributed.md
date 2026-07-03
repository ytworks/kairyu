# M16 Design: Distributed Execution — gloo-Tested TP/PP/EP, NCCL by Constructor

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (1-reviewer panel with gloo
spawn verification incl. uneven all_to_all splits, 2026-07-03; §6 binding).
Milestone: M16 (roadmap Track E3 local half; G2-as-amended multi-GPU gates'
code, NCCL swapped in on deploy day)
Date: 2026-07-03
Depends on: M12/M15 (models to shard), M8 (CheckpointReader.get_slice),
M13 (attention backend per rank). Empirical basis: torch.distributed gloo
verified working on this dev machine (all_reduce fp32/bf16, all_to_all,
all_to_all_single, send/recv, barrier — 2-process spawn).

## 1. Goal

All multi-GPU execution code written and CPU-tested with real multi-process
collectives (gloo); on deploy day NCCL is a CONSTRUCTOR ARGUMENT, not new
code. Deliverables: `TorchDistCommunicator` (satisfies the m5 `Communicator`
protocol + a tensor extension), real column/row-parallel sharding with TP=2
greedy parity vs TP=1, EP dispatch/combine over all_to_all with EP=2 parity,
PP stage execution over send/recv behind the existing `StageWorker` protocol,
per-rank sharded safetensors loading, and the SPMD worker main.

## 2. Key design decisions

### D1 — `TorchDistCommunicator(backend="gloo"|"nccl")` (`engine/core/dist_comm.py`)

Implements the existing object-level `Communicator` protocol
(broadcast/all_reduce-on-float-tuples/all_gather/barrier/send/recv via
`broadcast_object_list`/tensorized reduce/`all_gather_object`/
`send_object_list`) PLUS a `TensorCommunicator` extension protocol:
`tensor_all_reduce(t)`, `tensor_all_to_all_single(out, in, out_splits,
in_splits)`, `tensor_send/recv(t, peer)` — thin over `torch.distributed`
with an optional group arg. gloo gap (verified): **no reduce_scatter** — all
call sites use all_reduce (+ local slice); NCCL's reduce_scatter is a
same-call-site optimization recorded for deploy day.

### D2 — Tensor-parallel sharding (`models/parallel.py`)

`ColumnParallelLinear` (shards out_features; optional gather),
`RowParallelLinear` (shards in_features; all_reduce output),
`VocabParallelEmbedding` (shards vocab rows; masked lookup + all_reduce).
**Shard math lives in pure functions** (`shard_bounds(total, world, rank)`,
`shard_qkv_heads(config, tp, rank)` — GQA constraint: kv heads divide
evenly, reusing `validate_tp_degree`) so coverage is in-process; the modules
are thin wrappers. A `TpDenseDecoder` builder maps the M12 tree: q/k/v/gate/up
column-parallel, o/down row-parallel, embed/lm_head vocab-parallel, norms
replicated. Per-rank weights load via `CheckpointReader.get_slice` along the
module's shard dim (the m8 seam, unused until now).

### D3 — EP dispatch/combine (`models/moe_parallel.py`)

`EpMoeBlock` wraps the m15 blocks: routing runs replicated (fp32, identical
on every rank — cheap at gate sizes); tokens permute to expert-owning ranks
via `tensor_all_to_all_single` (counts exchange first, then payload), local
experts compute, reverse all_to_all, weighted combine locally. Expert→rank
assignment is contiguous blocks (`num_experts // ep_size` each). The math is
IDENTICAL to the m15 token-loop (pinned by EP=2 ≡ EP=1 parity); gloo and
NCCL share the code path.

### D4 — SPMD worker + PP stage (`engine/core/worker.py`, `pp_worker.py`)

`worker.py`: `run_tp_worker(rank, world, init_method, model_dir, ...)` —
init_process_group, build the rank-sharded model + pool + runner, loop:
rank 0 broadcasts `StepInput` (already-broadcastable m5 snapshot), all ranks
execute, rank 0 samples (rank agreement is a debug flag now — the real
invariant is identical logits via deterministic collectives). A
`DistTPModelRunner` driver-side class implements `ModelRunner`, so it drops
in where `TPModelRunner` sits. `RequestSnapshot` (m12 review A2) already
carries outputs/sampling/num_cached_tokens — extended there.
`pp_worker.py`: a real `StageWorker` (m6 `pipeline.py` protocol untouched):
non-final stages run their layer slice and `tensor_send` hidden states +
positions; the final stage recvs, finishes, samples.

### D5 — tests/dist harness

`tests/dist/conftest.py`: `torch.multiprocessing.spawn` with **file://
init_method in tmp_path** (no port races), 120 s timeouts, child results via
`mp.SimpleQueue`, child exceptions re-raised with rank tags; `@dist` marker
(runs in CI — ubuntu gloo works). Kept small (~6 spawn tests: TP=2 parity,
EP=2 parity, PP=2 parity, communicator contract): spawned-process lines
don't count toward coverage, so all decision logic stays in pure in-proc
functions.

## 3. Non-goals

- NCCL execution/tuning, reduce_scatter/symmetric-memory optimizations, P2P
  matrices (deploy day; constructor arg + env scripts).
- Cross-node rendezvous (`kairyu.launch` — G5 F3); DeepEP/UCCL adapters
  (deploy-day EP fast path; the all_to_all path is the portable baseline).
- Overlap of comm/compute streams (GPU-only).
- TP for MLA (attention-DP is the DeepSeek strategy; recorded).

## 4. Phasing

1. dist_comm + communicator contract tests (in-proc fake vs gloo param).
2. parallel.py shard math + modules + sharded loading; TP=2 spawn parity.
3. moe_parallel + EP=2 spawn parity.
4. pp_worker + PP=2 spawn parity; DistTPModelRunner wiring.

## 5. Verification

- `pytest -m dist`: TP=2 greedy ≡ TP=1 ≡ transformers (tiny llama); EP=2 ≡
  EP=1 (tiny qwen3-moe); PP=2 ≡ single-process; communicator contract suite
  passes on gloo exactly as on FakeCommunicator.
- In-proc: shard bounds/QKV head math, vocab-parallel masking, get_slice
  loading equals full-load-then-slice.
- Full suite green; dist tests excluded from cov accounting by design.

## 6. Review record (binding amendments)

- **A1 (BLOCKING)**: the m12-mandated RequestSnapshot extension never landed —
  M16 adds it: `outputs: tuple[int,...]` (output_len becomes a property),
  `sampling`, `num_cached_tokens`, plus `allocation -> self` / `pages` /
  `decode_pages` aliases so PagedModelRunner's canonical contract works on
  snapshots. This also closes the decode-token loop (workers read rank-0's
  committed token from the NEXT broadcast snapshot's outputs).
- **A2 (BLOCKING)**: shard ownership = PRE-SHARDED CONFIG (`tp_view(config,
  tp, rank)` divides heads/kv-heads/intermediate); modules come out rank-local
  for free (Attention's view() and the kv pool sizing are automatic); parallel
  Linears only ADD communication — the builder swaps o_proj/down_proj for
  RowParallel(all_reduce) since linear_factory can't tell call sites apart;
  shard-loading bounds computed from the FULL config; validate_tp_degree with
  the config's real kv heads.
- **A3 (BLOCKING)**: sampling needs FULL logits — vocab-parallel lm_head
  all_gathers logits shards (gloo rejects unequal shapes: fail-fast
  `vocab_size % tp == 0`); EVERY rank samples identically (keeps the m5 D1
  agreement invariant; rank-0-only sampling buys nothing at TP=2); sharded
  loader re-ties lm_head to the LOCAL embed shard after assign-load.
- **A4**: column-parallel slices bias with the same bounds; row-parallel adds
  its replicated bias ONCE, after the all_reduce; TP parity includes a Qwen2
  fixture (biases).
- **A5**: repo coverage config MEASURES spawned children — dist tests run in
  the default suite; worker mains stay in measured code; pure-function
  discipline retained on its own merits.
- **A6 harness pins**: init_process_group(timeout=120s) — gloo's 30-min
  default turns deadlocks into CI killers; start_processes(join=False) +
  polled join; torch.set_num_threads(1) in children; module-level spawn
  targets; one rendezvous file per group; GLOO_SOCKET_IFNAME=lo0 fallback
  recorded.
- **A7**: EP=2 ≡ EP=1 gate is greedy-token equality / allclose (index_add_
  accumulation order differs — algebraically identical, not bitwise).
- **A8**: replicated routing divergence guard (hash of topk_indices under a
  debug flag) recorded for deploy day.
- **A9**: PP needs a stage seam — stage forward (embed on stage 0, hidden
  input mid, norm+logits final), per-stage pools with rebased layer indices
  (runner sizing check accommodates), final stage samples and returns
  StepOutput to the driver.
- **A10**: quantized checkpoints × TP rejected loudly in M16 (group-crossing
  slices are real machinery with no G2 payoff).
- **A11**: worker startup handshake (rank 0 broadcasts num_pages/page_size/
  num_layers/config hash; workers validate) — workers have no cache object so
  the m12 sizing check doesn't fire; shutdown = broadcast None sentinel;
  per-rank KV bytes are 1/tp (sizing note).
