# M16 Design: Distributed Execution — gloo-Tested TP/PP/EP, NCCL by Constructor

Status: **Implemented** (2026-07-03). Reviewed (1-reviewer panel with gloo
spawn verification incl. uneven all_to_all splits, 2026-07-03; §6 binding).
Amended 2026-07-25 (D1: `tensor_reduce_scatter`, measured), 2026-07-26
(**D4**: separate TP control/model serving groups; **D6**: opt-in sequence
parallelism, lifting the §3 call-site non-goal), and 2026-07-27
(**D2/A10**: compressed-FP8 dense TP), and 2026-07-30
(**D2/A12**: canonical TP/EP parameter names), and 2026-08-08
(**D4**: Gloo tensor headers with fixed-layout NCCL step payloads).
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
call sites use all_reduce (+ local slice).

> **Amended 2026-07-25 (measured).** `tensor_reduce_scatter` is now implemented
> (NCCL's real collective; gloo keeps all_reduce + local slice, identical
> result). It is **not** a same-call-site optimization, and the earlier note
> saying so was wrong.
>
> Method (`verification/l1/performance/reduce_scatter_bench.py`): per-trial **worst-rank** elapsed via
> CUDA events, barrier-bounded so ranks start together, MAX-reduced across ranks
> — a collective finishes when its slowest participant does, so a per-rank
> minimum measures nothing about it. Buffers are prepared outside the timed
> region, and the path order is ROTATED per round so each occupies every position
> equally. 6 rounds x 20 trials = 120 samples per path, kept as raw per-trial
> records.
>
> 8x RTX PRO 6000 Blackwell, 8192x5120 bf16, torch 2.12.1+cu130 / NCCL 2.29.7,
> driver 595.84, PCIe (topology recorded in the result). 6 rounds x 20 trials,
> path order rotated per round, raw per-trial records retained
> (`bench/results/reduce-scatter-2026-07-25.json`):
>
> | | median ms | min | p95 | vs all_reduce |
> |---|---|---|---|---|
> | `all_reduce` | 3.784 | 3.760 | 3.910 | 1.00x |
> | `reduce_scatter` + `all_gather` | 3.944 | 3.926 | 3.960 | **0.96x — slower** |
> | `reduce_scatter` alone | 1.988 | 1.979 | 2.008 | **1.90x** |
>
> The ~4% loss is not straggler noise: all_reduce's p95 sits below rs+ag's
> MINIMUM. The full distributions do overlap — all_reduce has a handful of
> samples above rs+ag's floor — so the claim is about the bulk of the
> distributions, not their supports. Swapping one for the other at the
> `RowParallelLinear` call site moves the same bytes and adds a launch.
>
> The 1.90x is real but only available if the consumer accepts a **shard** —
> i.e. sequence parallelism, where the shard survives the norm and the next
> column-parallel matmul re-gathers it. That design change is now specified in
> **D6** (added 2026-07-26) — and, as this note predicted, it is justified by
> activation memory, NOT by comm time.

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

> **Amended 2026-07-27 (G2 A2).** Dense compressed-tensors FP8 is the first
> quantized TP contract. Projection weights follow D2's existing column/row
> shard axis. Per-output-channel `weight_scale` shards with a column-parallel
> output axis and remains replicated for row-parallel projections. Serialized
> BF16 scales widen value-preservingly to the FP32 fused scaled-MM ABI.
>
> Dynamic activation scaling is also topology-aware. Column-parallel ranks see
> the same full activation and independently obtain the same scale. A
> row-parallel rank sees only one input-feature shard, so a rank-local amax
> silently changes W8A8 quantization as TP changes. `RowParallelLinear`
> therefore MAX-reduces one FP32 amax per token, then every rank quantizes its
> local feature shard with that global per-token scale before the ordinary
> partial GEMM and output sum. No full FP32 activation or dequantized weight is
> materialized. Other quantization formats remain rejected until their packed
> shard/scale contracts are specified.

> **Amended 2026-07-30 (issue #233).** Parallel execution is bound to the
> parameter-owning module at its canonical HF/checkpoint leaf. TP row reduction
> and SP scatter/gather behavior are attached to that same module object; they
> do not register synthetic `local`, `norm`, or `embedding` children. The
> canonical leaf therefore remains the single identity used by `state_dict`,
> `named_parameters`, tied-weight identity, adapter lookup, quantization
> metadata, and checkpoint loading after binding.
>
> EP preserves the global expert namespace in a global-length
> `ModuleList`. An owned expert remains registered at
> `experts.<global-index>`; a remotely owned slot is a `None` hole and therefore
> owns no tensor and emits no state entry on this rank. Gate and shared-expert
> modules likewise remain at their canonical paths. A local lookup of a remote
> expert reports its deterministic owner-rank error instead of renumbering it
> into a rank-local namespace.
>
> Canonical state from one parallel rank is explicitly **rank-local**. Its keys
> are native checkpoint names, but its tensors contain only that rank's TP/EP
> ownership and are loadable only into the same rank and topology. Requesting a
> complete `full-hf` serialization from one sharded rank fails deterministically
> with the recorded shard topology; a future collective exporter must gather TP
> slices, union EP holes, validate replicated values, and restore ties before it
> may claim a complete HF checkpoint.
>
> TP checkpoint member specifications are derived from the contextual linear
> leaf itself: its canonical qualified name supplies the source prefix, its
> typed TP placement supplies logical ownership, and each dense or packed
> format supplies exact physical axes for every persistent member. Loading
> binds the canonical module tree first and then slices into that tree. There
> is no suffix rewrite table between checkpoint loading, adapter targeting, and
> quantization policy; an unknown persistent member or a context/path
> disagreement fails closed. These naming and tooling changes do not alter any
> collective: column/row TP still uses the D2 shard and reduction rules
> (including an unbiased local GEMM followed by one canonical bias add), SP
> still uses D6 scatter/gather/reduce-scatter, and EP still uses D3's all-to-all
> dispatch/combine.

### D3 — EP dispatch/combine (`models/moe_parallel.py`)

`EpMoeBlock` wraps the m15 blocks: routing runs replicated (fp32, identical
on every rank — cheap at gate sizes); tokens permute to expert-owning ranks
via `tensor_all_to_all_single` (counts exchange first, then payload), local
experts compute, reverse all_to_all, weighted combine locally. Expert→rank
assignment is contiguous blocks (`num_experts // ep_size` each). The math is
IDENTICAL to the m15 token-loop (pinned by EP=2 ≡ EP=1 parity); gloo and
NCCL share the code path.

> **Amended 2026-08-08 (issue #359).** The generic all-to-all path accumulates
> each rank's weighted expert contribution in FP32, all-reduces those FP32
> partials, and casts once to the model dtype after the collective. This avoids
> a model-dtype rounding boundary per EP rank and pins the same combine result
> for the exercised EP2/4/8 ownership partitions. The homogeneous NVFP4 fused
> correctness path retains G4 D3's explicitly measured BF16 finalize contract.

> **Corrected 2026-08-08 (PR #448 Fable 5 review).** Reverse all-to-all already
> returns the complete computed expert rows to each replicated source rank.
> Generic EP therefore combines every slot locally in a fixed FP32 order and
> casts once, without a redundant combine all-reduce. The result and its
> addition grouping are independent of EP2/4/8 ownership; the NVFP4 carve-out
> above remains unchanged.

> **Amended 2026-08-08 (issue #331).** Dense BF16 EP uses fixed
> `[ep_size, tokens * top_k, hidden]` peer-capacity buffers. Device-side sorted
> positions fill each destination chunk, padding targets that destination's
> first local expert, and the forward/reverse `all_to_all_single` split lists
> are host constants. Local rows use M15 A10's grouped GEMMs. This removes the
> counts `.item()`/`.tolist()` and data-dependent expert loop from the CUDA hot
> path and makes a static decode shape graph-capturable. Quantized/custom
> compatibility paths and the fused NVFP4 path retain their established
> transports.

> **Corrected 2026-08-08 (PR #453 Fable 5 review).** Fixed-capacity EP is
> limited to at most 8,192 total grouped rows after EP padding; larger prefills
> retain the exact-split compatibility transport. Grouped row counts are
> power-of-two bucketed and every permitted plan is warmed before decode graph
> capture, reserving FlashInfer's grouped workspace high-water mark. Dense BF16
> EP1/EP2 parity is tolerance-pinned because different cuDNN expert-group
> shapes may choose different numerically valid GEMM plans; FP32 slot combine
> order remains identical.

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

> **Amended 2026-07-29 (issue #225).** “Rank 0 samples” is now an enforced
> ownership protocol rather than a return-value convention. Followers execute
> `PagedModelRunner.execute_passive`, then receive one fixed-layout int64 token
> packet on the model communicator. All ranks adopt the packet into their
> future-token device state before returning to the control receive. Followers
> are constructed with `sampling_owner=False`; entering any sampling path is a
> fatal error. The in-process `TPModelRunner` remains a compatibility/test
> facade rather than the production SPMD transport, but enforces the same
> rank-0 execute / follower-passive / canonical-packet-adoption contract.
>
> The production closure gate separates two invariants. A real TP2 injection
> gate requires exact rank-0 packet adoption and verifies that the next decode
> consumes it; TP8 evidence separately binds the NCCL overwrite primitive and
> full rank-0-owner/passive-follower topology. Across TP1 and TP8, exact
> free-running continuation equality is diagnostic:
> a BF16 reduction-order near-tie can choose a different valid token and thereby
> change every later autoregressive prefix. Binding cross-degree evidence must
> instead contain finite raw records plus the actual world size, complete
> rank set, rank-0 owner/sampler identity, and passive follower identities. It
> compares distributions only at positions reached from a common prefix. Before
> divergence, the common selected token is checked within the declared logprob
> tolerance; at the first divergence, both selected tokens are directly scored
> under each run's full raw distribution and both reciprocal deltas must remain
> within that tolerance. No later
> different-prefix position enters the compatibility verdict.

> **Amended 2026-07-26 (issue #148, corrected after the #150 hardware
> rerun; amended 2026-08-08 by issue #323).** CUDA TP serving has two
> operational groups, created in the same order on every rank after the startup
> handshake:
>
> - **control:** gloo only, with an effectively process-lifetime idle timeout —
>   a fixed two-word tensor header for every transaction, followed by an object
>   body only for rare controls such as probes, mode changes, and shutdown;
> - **model:** the placement backend (NCCL on CUDA), with the 120 s fail-fast
>   timeout — a versioned, lossless int64 `StepDelta` payload plus model tensor
>   collectives.
>
> CPU/Gloo TP retains the object-control fallback. The long-timeout startup
> group remains separate. Python object broadcasts must
> not share the model NCCL group: `broadcast_object_list` uses metadata and
> payload collectives plus receiver-side host deserialization, so rank 0 can
> enqueue the next model all-reduce while peers are still completing the control
> transfer. On Qwen3-32B TP=8 this produced rank 1–7 at BROADCAST sequence 4163
> while rank 0 had entered ALLREDUCE sequence 4165, followed by the 120 s NCCL
> watchdog. A blocking gloo hand-off orders every rank before it enters the
> independent model group.
>
> The timeout asymmetry is load-bearing. Non-zero ranks wait *inside* the next
> Gloo control-header broadcast while the server is idle. Giving that receive the model
> group's 120 s timeout killed a healthy deployment after 120 idle seconds; its
> workers then entered graph teardown and produced a misleading NCCL barrier
> watchdog. The model group has no collective pending while idle, so only it can
> safely retain the short operational bound. Any TP step exception marks the
> group fatal: the backend stops retrying the permanently divergent sequence,
> readiness requests process replacement, and teardown takes the abort/reap path.

### D5 — tests/dist harness

`tests/dist/conftest.py`: `torch.multiprocessing.spawn` with **file://
init_method in tmp_path** (no port races), 120 s timeouts, child results via
`mp.SimpleQueue`, child exceptions re-raised with rank tags; `@dist` marker
(runs in CI — ubuntu gloo works). Kept small (~6 spawn tests: TP=2 parity,
EP=2 parity, PP=2 parity, communicator contract): spawned-process lines
don't count toward coverage, so all decision logic stays in pure in-proc
functions.

### D6 — Sequence parallelism (Megatron TP+SP), opt-in (`models/parallel.py`)

> **Added 2026-07-26.** This lifts the §3 non-goal on using `reduce_scatter` at
> the `RowParallelLinear` call site. It is a design change to D2, recorded here
> rather than in a new milestone doc because it only re-plumbs D2's own modules.

`build_tp_model(..., sequence_parallel=True)` shards the residual stream
BETWEEN blocks along **tokens**, so the norms and the inter-block residual hold
S/tp rows instead of S. Off by default; `tp >= 2` required (fail-fast
`ValueError`), and D2's plain-TP path is byte-for-byte the old one when the flag
is off — the wrappers are only installed when a context exists.

`SequenceParallelContext` owns the region and is shared by every wrapper of one
model. Its contract:

- `scatter(x)` — entry: full `[S, H]` -> this rank's `[ceil(S/tp), H]`.
- `gather(x)` — shard -> the REAL sequence (`tensor_all_gather`, padding
  trimmed).
- `reduce_scatter(x)` — full `[S, H]` partial sums -> this rank's reduced shard.

Placement (D2's tree, wrappers only): `ScatterAfterEmbedding` on
`embed_tokens` enters the region; `SequenceParallelNorm` on
`input_layernorm`/`post_attention_layernorm` runs the norm ON THE SHARD then
all_gathers into the TP region — RMSNorm/LayerNorm are per-token, so this is the
identical arithmetic, which is what makes a sharded residual stream valid at all;
`RowParallelLinear(o_proj/down_proj)` takes the context and exits with
`tensor_reduce_scatter` instead of `tensor_all_reduce` (it both sums across ranks
AND re-shards); `GatherBeforeNorm` on the final norm exits, so the lm_head and
A3's vocab-parallel gather still see the whole sequence.

**Padding lives at the shard boundary ONLY** — added on the way in, removed on
the way out (`context.padding`). It must NEVER be visible inside the TP region:
attention builds its mask from the real sequence length, so a padded gather
produces a residual add between a 12-row and an 11-row tensor. `reduce_scatter`
re-pads immediately before the collective (which needs a divisible count) and
the next `gather` trims.

**What this does NOT buy: comm time.** all_gather + reduce_scatter moves what one
all_reduce moves — measured slightly WORSE (D1 amendment, ~0.96x). The gain is
ACTIVATION MEMORY. Enabling it for speed would be a mistake, so the module
docstring says so too.

Scope: dense decoders built by `build_tp_model` (D2). NOT wired into EP (D3),
PP (D4), or the SPMD worker/`DistTPModelRunner` (D4) — those keep plain TP.

## 3. Non-goals

- symmetric-memory optimizations (deploy day). NCCL execution, the P2P matrix
  and `reduce_scatter` itself landed 2026-07-25 (D1 amendment), and using it at
  the `RowParallelLinear` call site landed 2026-07-26 as opt-in sequence
  parallelism (**D6**) — that non-goal is lifted. What remains non-goal is
  sequence parallelism for EP/PP/the SPMD worker, and making it the default.
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
- D6: SP ≡ plain TP on gloo (max error AND argmax), and again with a **ragged**
  11 tokens across 2 ranks asserting an 11-row output — the gate that pins
  padding to the shard boundary (`tests/dist/test_distributed.py`); the same
  parity over real NCCL in bf16 (`tests/gpu/test_sequence_parallel_nccl.py`).
- D4 control/model split: 512 production `DistTPModelRunner` steps on two real
  GPUs prove that Gloo carries only tensor headers plus a mode-toggle/shutdown
  object body while `StepDelta` payloads use NCCL (`tests/gpu/test_tp_control_plane_nccl.py`);
  a two-rank gloo gate
  holds a worker receive beyond the model timeout before delivering the next
  step. Hardware validation also uses the exact
  Qwen3-32B TP=8 / Accuracy LiveCodeBench 20-item / concurrency-8 workload. The
  latter ran 654.53 s without a watchdog, kept `/readyz` at 200, and shut down
  with all eight GPUs returning to 0 MiB / 0% without a reset.
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
- **A3 (BLOCKING, amended by issue #225)**: row-parallel reductions leave a
  replicated hidden state and the current TP builder keeps a replicated
  `lm_head`, so rank 0 owns sampling from the full logits. Non-zero ranks run
  the same attention/MLP/KV work but do not own RNG, penalties, grammar,
  logprobs, or a public StepOutput; eager followers also skip `lm_head`.
  Rank 0 broadcasts a fixed `ScheduledChunk`-ordered int64 device-token packet
  on the model communicator and every rank adopts it before the next forward.
  This replaces the earlier all-rank-sampling amendment: that scheme duplicated
  stateful work and still required result comparison to prevent divergence.
  Exact adopted-token equality and next-decode use are binding in the real TP2
  injection gate. TP8 separately binds communicator overwrite and complete
  ownership topology. TP1/TP8 free-running sequence equality is diagnostic
  only; cross-degree compatibility is evaluated from complete finite raw
  evidence on common-prefix positions, including direct cross-selected logprob
  tolerance at the first divergence, and must be accompanied by verified
  rank/ownership topology.
  Recorded 2026-07-30 on 8× RTX PRO 6000: the TP8 NCCL broadcast p95 is
  0.078400/0.070816/0.075648 ms for B=1/8/16 over 256 samples per cell; the
  Qwen3-32B TP1/TP8 gate retains 43 tokens per degree, with binding aligned and
  first-divergence cross-selected maxima of 0.148189 and 0.101015 versus the
  0.25 limit. Both artifacts and their stored replay pass.
  A future vocab-sharded head may gather non-greedy logits to rank 0 (or reduce
  greedy value/index pairs), but must preserve this single-owner token contract.
- **A4**: column-parallel slices bias with the same bounds; row-parallel adds
  its replicated bias ONCE, after the all_reduce; TP parity includes a Qwen2
  fixture (biases).
- **A5**: repo coverage config MEASURES spawned children — dist tests run in
  the default suite; worker mains stay in measured code; pure-function
  discipline retained on its own merits.
- **A6 harness pins**: test init_process_group(timeout=120s) — gloo's 30-min
  default turns test deadlocks into CI killers. Production TP control receives
  are intentionally exempt because idle workers wait inside that collective;
  production Gloo header receives retain the idle allowance while model
  collectives retain 120 s. start_processes(join=False) +
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
- **A10 (amended 2026-07-27)**: dense compressed-FP8 TP is supported under
  D2's explicit weight/scale/global-activation-scale contract. Other quantized
  checkpoints × TP are still rejected loudly; packed formats require their own
  group-crossing shard specification.
- **A11**: worker startup handshake (rank 0 broadcasts num_pages/page_size/
  num_layers/config hash; workers validate) — workers have no cache object so
  the m12 sizing check doesn't fire; shutdown = broadcast None sentinel;
  per-rank KV bytes are 1/tp (sizing note).
- **A12 (issue #233)**: TP/SP execution behavior binds to the canonical
  parameter-owning leaf without wrapper-owned parameter prefixes. EP retains a
  global-index `ModuleList` with `None` holes for remote experts. Native-name
  state is explicitly rank-local; one sharded rank cannot claim a full HF
  export. Context-derived checkpoint slicing, adapter targeting, quantization
  metadata, and post-bind enumeration therefore share one name and ownership
  contract while D2/D3/D6 collective behavior remains unchanged.
