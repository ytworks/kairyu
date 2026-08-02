# G4.1 Design: Mid-tier MoE — Real NVFP4 Loading and Correctness

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (2026-08-03; the M-A1 best
implementation is retained with its formal FAIL, M-A2 production integration
and formal hardware evidence are complete, and M-A3 retains its first formal
performance FAIL while an optimized production candidate awaits remeasurement).
Milestone: G4.1 M-A1/M-A2/M-A3.
Depends on: M12 (Qwen model), M13 (paged attention), M14 (native NVFP4
projection kernels), M15 (Qwen3-MoE math), M16 (NCCL communicator and EP
dispatch), M8/M9 (paged runner and scheduler).
Consumed by: G4 M-A2 (EP + radix KV) and M-A3 (mid-tier throughput).

## 1. Goal and boundary

Load and execute the pinned NVIDIA Qwen3-235B-A22B NVFP4 checkpoint with
expert parallel degree 2 and 4 on one node, then produce replayable
correctness evidence against an immutable reference runtime using the same
checkpoint and the same GPU count.

M-A1 is a correctness anchor. It deliberately replicates QKV projection,
attention score/value computation, and KV state on every EP rank so the
existing single-owner SPMD protocol can be reused without changing request
ownership. The attention output projection alone is row-parallel on the EP
communicator to preserve the reference's BF16 rank-partial reduction boundary.
This is not the attention-DP or overlap-capable implementation required to
make a throughput claim in M-A3. The M-A1 formal verdict therefore reports
correctness only; timing and memory fields are diagnostic and cannot close
M-A3.

## 2. Pinned production contract

### D1 — Checkpoint identity and quantization metadata

The model is:

```
nvidia/Qwen3-235B-A22B-NVFP4
revision 21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2
```

The checkpoint contains `Qwen3MoeForCausalLM`, 94 layers, 128 routed experts
per sparse layer, top-8 routing, hidden size 4096, and ModelOpt group-16
NVFP4 weights. The public checkpoint keeps its weight/KV quantization
declaration in `hf_quant_config.json`, not `config.json`.

All runtime checkpoint entry points resolve both metadata locations. An
external declaration requires a non-empty ModelOpt producer version, NVFP4
group size exactly 16, a valid exclusion list, and an optional KV algorithm
of FP8 only. Embedded and external weight declarations must agree when both
exist. A missing, malformed, conflicting, or unsupported declaration fails
before tensor loading.

The external FP8 KV declaration is calibration metadata, not permission to
enable FP8 KV on this hardware profile. G4 E-KV failed its quality gate, so
M-A1 uses BF16 KV. The 94 `k_proj.k_scale` and 94 `v_proj.v_scale` scalar
tensors are required, shape/value validated, recorded as unused auxiliary
metadata, and never silently applied to the BF16 cache.

Every other checkpoint member must match the model-derived global name and
shape contract. Unknown members, missing remote-expert members, wrong packed
dtypes, and shape mismatches fail closed on every rank.

### D2 — Meta-first, expert-owned checkpoint construction

The complete 134 GB model must never be materialized before sharding.
Construction occurs on the meta device with canonical HF module names.
For a contiguous expert assignment, rank `r` owns:

```
[r * (128 / ep_size), (r + 1) * (128 / ep_size))
```

Remote routed projections are allocation-free construction placeholders.
After each sparse block is bound to `EpMoeBlock`, remote slots become `None`
holes in the global-length `ModuleList`, preserving the M16 canonical-name
contract. QKV projections, attention/KV state, embeddings, norms, routers,
and the output head are replicated. Each NVFP4 attention `o_proj` retains its
canonical checkpoint path but owns only the rank's contiguous packed K-axis
weight and block-scale shard; its global activation and weight scales remain
replicated.

Each rank reconstructs and validates the full global checkpoint contract and
loads only rank-owned routed experts. Selected tensor names are grouped by
safetensors shard so each shard is opened at most once per loading pass.
The validated `o_proj` path currently materializes its 188 small dense members
through the normal sequential mmap loader, transfers them with the rest of the
rank, then replaces each GPU buffer one layer at a time with its contiguous K
shard and immediately releases the full buffer. Earlier full-run CPU-slice
attempts ended before model startup, but a later isolated probe constructed all
188 EP2 slices in 4.105 seconds; that evidence does not isolate inner-axis
slicing as the cause, so no unsupported stall claim is made. The retained path
is the one that completed the real EP2 diagnostic and EP4 smoke. Its maximum
temporary `o_proj` allocation above final residency is 854 MiB at EP2 and
1,273 MiB at EP4. Source-at-a-time GPU staging can reduce that follow-up P2 to
one full member (at most 16 MiB) without changing the numerical path. The
complete 134 GB model is never materialized on one rank, and the steady-state
module owns only the local `o_proj` shard. Dense floating tensors are converted
to the requested compute dtype. NVFP4 packed tensors retain their exact ABI:

- `weight`: `uint8`
- `weight_scale`: `float8_e4m3fn`
- `weight_scale_2`: `float32`
- `input_scale`: `float32`

An aligned, contiguous packed weight is passed directly to FlashInfer.
Creating an equal-sized zero-width padding copy is forbidden. Only the scale
layout required by the native kernel may be cached separately.

### D3 — Correctness-mode EP execution

M-A1 launches one process per GPU with separate long-lived gloo control and
NCCL model groups, following M16 D4. Rank 0 owns scheduling, sampling,
logprobs, and public output. Followers execute passive forwards and adopt the
rank-0 token packet before the next step.

Every rank receives the same scheduled tokens and owns the same QKV,
attention-core, and KV state. The complete attention context is split on its
innermost K axis; each rank's NVFP4 `o_proj` produces a BF16 partial and NCCL
all-reduce reconstructs the replicated hidden output.

For the pinned homogeneous NVFP4 experts, `EpMoeBlock` packs each rank's
`[up, gate]` FC1 and FC2 weights into FlashInfer's public CUTLASS fused-MoE
ABI. Routing stays replicated. Each rank evaluates only its contiguous local
expert range, finalizes that rank's weighted partial in BF16, and NCCL
all-reduces the partials. `enable_alltoall=False`; no token or expert row is
duplicated through an all-to-all. The generic mixed/non-NVFP4 compatibility
path retains the M16 all-to-all contract.

Packed weights share storage with their canonical per-expert buffers.
FlashInfer's 128x4 scale order cannot be represented as an inverse 2-D strided
view, so M-A1 retains both canonical checkpoint scales and one derived
swizzled scale set. The exact additional steady CUDA allocation is
6.609375 GiB per EP2 rank and 3.3046875 GiB per EP4 rank; both completed real
diagnostics include this allocation. Removing it requires a separate
block-owned lowering plus inverse serialization/load hooks and remains a P2,
not a correctness shortcut in this implementation.

The fused operand object is derived runtime state rather than checkpoint
state. Any module `_apply` operation (`to`/`cuda` included) or
`load_state_dict` invalidates it before registered buffers can change. A stale
device or scale generation can therefore never execute; the fused runtime
probe disappears until the caller explicitly repacks. The normal construction
path moves the model first and packs afterward. A forward is marked successful
only after the fused kernel, rank all-reduce, and shared-expert contribution
all complete.

The M-A1 runner accepts only:

- one node and EP degree 2 or 4;
- CUDA + NCCL model collectives;
- BF16 compute and BF16 KV;
- eager execution;
- pipeline depth 1;
- no public tensor/pipeline parallel composition (the internal `o_proj`
  row-parallel reduction is part of this EP correctness operator);
- no CUDA graph, P-D handoff, DRAM KV tier, FP8 KV, or attention-DP.
- the M-A1 correctness capture alone disables radix/block reuse between formal
  requests; M-A2 deliberately enables one persistent production radix cache.

Unsupported combinations fail during construction. Backend/status reporting
uses `expert_parallel_size`; it must not label EP as tensor parallelism.
M-A1 used this runner as an L1 formal correctness operator. M-A2 wires the
same bounded EP2/EP4 execution envelope into the L3 production
serving/status surface; this later integration does not change or retroactively
pass the M-A1 correctness verdict.

### D4 — Immutable reference

The formal oracle is:

```
nvcr.io/nvidia/tensorrt-llm/release
@sha256:cb4d8af81c586a90235ae3739b6d4ddc5d8336f2174c8a1c6b573d2e13faf5d7
TensorRT-LLM 1.2.1, PyTorch backend, CUTLASS MoE, BF16 KV
```

Reference-2 uses TP2/EP2 and reference-4 uses TP4/EP4. Attention-DP,
overlap scheduling, CUDA graphs, and block reuse are disabled. The four
cells run sequentially rather than competing for GPUs:

1. reference-2;
2. Kairyu EP2;
3. reference-4;
4. Kairyu EP4.

Both stacks expose a fixed 4,096-token BF16 KV capacity. Kairyu uses 256
16-token pages. TensorRT-LLM uses `max_tokens=4096` plus a 5% free-memory
ceiling; the measured EP2 post-weight free memory makes that ceiling larger
than the approximately 376 MiB TP2 pool, while leaving allocator/workspace
headroom. The complete formal batch needs only 1,689 unrounded tokens, so the
fixed capacity is sufficient at both degrees without a topology-dependent
percentage-sized pool.

TensorRT-LLM 1.2.1's HTTP completion postprocessor does not expose the
engine logprobs. Formal evidence therefore uses the official Python `LLM`
API in the same immutable image and stores token IDs and token-ID-keyed
top-64 logprobs directly. HTTP may be used only for a health smoke and cannot
satisfy the correctness gate.

vLLM 0.26 is diagnostic only: its SM120 `flashinfer_b12x` MoE backend is
explicit rather than automatic and does not support expert parallelism. A
TP-only vLLM run may be retained as a cross-check, but it is not the M-A1
oracle and cannot replace either reference cell.

## 3. Formal correctness method

### D5 — Fixed inputs and common-prefix comparison

The input set is the exact 64-text `_TEXT_PROMPTS` tuple from
`bench/parity_tp.py`. Evidence stores the text, tokenizer-produced IDs, and
canonical hashes. Generation is 16 greedy tokens with seed `20260702`,
temperature 0, top-p 1, no top-k truncation, and EOS ignored until token 16.

Each GPU count has its own reference teacher rollout. At position `i`, the
reference first evaluates a fresh full-prefix request on:

```
prompt_ids + reference_W_teacher_tokens[:i]
```

The selected token becomes position `i` of that immutable teacher rollout.
Kairyu-EPW is then evaluated on the exact same frozen prefix. Repeating this
for 16 positions provides 1,024 same-prefix comparisons per GPU count and
prevents a numerically valid near-tie at one position from turning every later
different-prefix token into a false disagreement.

The ordinary 16-token retained-KV continuations from both stacks are retained
as diagnostics but never supply the binding prefix. A pre-formal EP2
diagnostic demonstrated why: TensorRT-LLM's retained-KV decode and fresh
full-prefix path agreed on only 933/1,024 positions even though every returned
prompt token sequence was exact. The largest fresh-path deficit for the
retained-decode token was 6.125 natural-log units, so those 91 differences
cannot be relabelled as near-tie noise. The binding claim is therefore
fresh-prefill next-token parity on identical prefixes; M-A1 does not claim
retained-KV decode parity between the stacks.

Replay requires the complete reference rollout chain: every reference
teacher row's selected and canonical token must be identical, and every
prefix count/hash must equal the prompt plus all earlier reference teacher
tokens. Each Kairyu cell passes only if all evidence is present and finite and
all of the following hold:

- token agreement is at least `ceil(0.99 * 1024) = 1014`;
- substantive disagreement count is zero;
- at matching positions, selected-token logprob absolute delta is at most
  0.25 natural-log units;
- at a differing position, both selected tokens occur in both token-ID
  top-64 sets and both reciprocal selected-token logprob deltas are at most
  0.25;
- a differing position is non-substantive only when the measured reference
  alternatives are within the fixed `0.125` natural-log-unit near-tie bound.

The 99% floor is fixed and is never lowered to fit observed reference noise.
OS scheduling jitter is irrelevant to this correctness verdict; it applies
only to non-binding timing fields.

### D6 — Evidence and replay

`bench/g4_ma1_qwen3_235b_nvfp4_capture.py` produces the source-bound arm
fragments and canonical JSONL; `bench/g4_ma1_qwen3_235b_nvfp4_bench.py`
performs run, verify, and raw-only replay. The raw JSONL is the authority. It
records run/cell/rank lifecycle,
configuration, complete environment and topology, checkpoint members and
digests, projection/kernel inventory, prompts, free-running outputs, and all
teacher-forced position records. A manifest is derived only from raw rows.

`verify` recomputes the manifest and requires exact equality. `replay`
ignores a stored manifest and independently derives the verdict from raw
rows. Missing or duplicate positions, unknown row types, unconsumed rows,
unknown ranks, ownership gaps/overlap, non-finite values, retry/failure rows,
source drift, checkpoint drift, reference image drift, fallback/oracle
kernel use, or incomplete run termination all fail closed.

Kairyu retains an all-rank runtime probe for the actual gloo control group,
NCCL model group, rank-0-only sampler, device placement, loaded expert
partition, fused local-expert execution, attention-output K-shard geometry
and successful BF16 partial forwards, native NVFP4 projection inventory, and
unresolved meta counts.
Rank-local probe failures are transported as bounded error envelopes so every
rank enters the same gloo gather before the driver fails; a diagnostic error
cannot strand another rank in the long-lived control collective.

GitHub-hosted CI does not have this 8-GPU checkpoint environment. CI runs
the deterministic replay/tamper suite over committed evidence; it does not
attempt or pretend to rerun the 235B measurement. The hardware run is
performed on the declared 8×RTX PRO 6000 host and its complete raw evidence
is committed.

## 4. M-A2/M-A3 seams and explicit non-claims

### D7 — Rank-invariant radix truth for M-A2

M-A2 reuses the same model loader, native kernels, and replicated-attention
EP runner. The L2 `Scheduler` and `RadixKVCache` remain single logical owners
on rank 0. Each inference step already broadcasts the immutable `StepDelta`
containing the logical allocation, page table, and `num_cached_tokens`; every
replicated KV rank applies that same snapshot before executing. Therefore the
cache rate is counted once from terminal `StreamUpdate` engine usage, never
once per rank. Four rank receipts are equality witnesses for the control-plane
allocation and page-table view; they are not four rate samples and do not
claim a bytewise readback of physical KV contents.

The formal operator,
`bench/g4_ma2_qwen3_235b_ep_kv_bench.py`, arms one opt-in, bounded recorder on
all EP ranks before the trace and gathers it once afterward over the existing
gloo control group. It records only the first prefill view of each request:
prompt/cache token counts, the actual prefill start (including the scheduler's
full-hit final-token recompute rule), prompt digest, and logical page-table
digest. Capture is limited to 4,096 observations and 256-byte request IDs.
Overflow and rank-local capture errors are retained as explicit failure
evidence; a diagnostic error never interrupts the inference collective order.
Outside an armed EP capture the recorder retains no request history, and TP
does not construct it.

The fixed real-model cell is Qwen3-235B NVFP4 EP4 with BF16 KV, eager decode,
pipeline depth 1, page size 16, and one persistent cache/scheduler/engine loop.
It serializes the A7-lineage 64-session × 8-turn trace: a 512-token shared
prefix and 128 appended tokens per turn, for 512 one-token requests. The
derived cache capacity is 4,129 pages: 4,128 retained pages plus one
active-request guard. Binding success requires all 512 terminal engine usages,
the strict logical `sum(cached) / sum(prompt) > 0.80`, exact raw
`BlockStored` events, four complete rank receipts with no accounting/page
drift, and complete source/checkpoint/container/GPU/topology/kernel evidence.
`verify` must reproduce the manifest and raw-only `replay` must derive the
same verdict. Timing, OS jitter, output equality against another engine, and
rank-multiplied token counts are non-binding.

Production configuration exposes `expert_parallel_size` 2 or 4 through
`KairyuBackend`; TP and EP are mutually exclusive and the existing EP
correctness envelope remains fail-closed. Readiness/shutdown use one
topology-neutral launcher handle, while `/backends` reports complete EP
metadata locally and through a `ReplicaPool` without relabelling EP as TP.
The formal GPU operator drives the same production L1 `DistEPLauncher`
directly so it can arm the all-rank recorder and attach the raw radix event
sink; CPU/server regression tests separately bind the L3 construction and
reporting path.

The clean-commit real run at `d2d33e0472fb3101b680f4085d22e80a4ac7ceca`
passed all 12 binding checks on 4× RTX PRO 6000. All 512 requests completed;
the one logical engine rate was 491,008 / 557,056 = 0.8814338235, all four
ranks reported those exact totals and identical page identities, and the raw
trace retained 512 `BlockStored` events and 4,128 unique blocks. Manifest
verification and raw-only replay both pass. The complete evidence is retained
under
`bench/results/g4-ma2-ep-kv-qwen3-235b-rtxpro6000-2026-08-02/`.

M-A3 replaces correctness-mode attention duplication with request-owned
attention-DP and reuses the grouped/fused local-expert ABI introduced here.
The implementation diagnostic selected pipeline depth 5 over depth 1 for the
production candidate; this choice is fixed before the formal run and the
diagnostic timing is not M-A3 evidence. The binding comparison runs Kairyu and
SGLang sequentially at equal checkpoint/config and publishes steady-state
token throughput and TTFT statistics. M-A1 timing cannot be reused as M-A3
evidence.

### D8 — Request-owned attention-DP and the M-A3 comparison boundary

The M-A3 Kairyu path is an opt-in, fail-closed production envelope for the
pinned Qwen3-235B NVFP4 checkpoint: one node, four ranks, TP1, EP4, BF16
compute/KV, FCFS scheduling, request-owned attention-DP, and either eager or
CUDA-graph decode. A stable round-robin assignment owns each request's
attention, KV pages, sampling state, and selected-token packet on one rank for
its lifetime. Q/K/V, attention output, embeddings, norms, routers, and the
output head are replicated; each rank retains only its contiguous expert
partition. Unlike M-A1, `o_proj` is therefore a complete replicated
projection, produces a complete local hidden row, and has no attention-output
all-reduce. Every rank still participates in the ordered control and MoE
collectives, including when it has no real row; bounded scratch requests/pages
make that participation explicit rather than borrowing another request's KV.

The MoE boundary communicates the smallest native operands. Each owner routes
and NVFP4-quantizes its local BF16 hidden rows before communication. The four
packed activation, input-scale, selected-expert, and routing-weight tensors
are issued as one grouped direct-NCCL operation. Eager unequal owner layouts
use rank-ragged grouped broadcasts and reductions without zero-row padding;
equal layouts, including graph buckets, use fixed all-gather and
reduce-scatter shapes. Every expert rank evaluates only its local fused
experts into a grow-only stream-local BF16 partial workspace, and
reduce-scatter returns only each attention owner's completed rows. There is no
post-MoE all-reduce. The direct communicator is a deliberately small ctypes
binding over `libnccl.so.2`; the existing Gloo/NCCL process groups retain
control-plane and lifecycle ownership. An all-rank initialization failure may
select the declared `torch.distributed:nccl` compatibility transport, but a
runtime direct-NCCL failure aborts instead of changing algorithms in flight.
M-A3 formal evidence requires direct NCCL to be active, so the compatibility
transport cannot satisfy the performance gate.

Compatible Q/K/V NVFP4 projections share one activation quantization and one
concatenated FlashInfer W4A4 GEMM through `NvFp4LinearPack`. The three
canonical checkpoint module names and state-dict keys remain intact, and their
weights/scales become views of one packed owner so the optimization does not
retain a second raw QKV copy. Any incompatible scale, layout, hook, placement,
or lifecycle mutation declines or invalidates the pack and executes the
existing separate projections. The pinned checkpoint's 94 attention layers
were compatible in the real load smoke; that smoke and its CUDA-graph replay
proof establish implementation readiness but are not M-A3 performance
evidence.

CUDA-graph policy is coordinated before any model collective. The driver
gathers each rank's prospective owner-row and page-table width, chooses one
global eager/capture/replay decision, and arms that decision exactly once on
every rank. Captures use unique scratch pages and enough dummy rows to make
the selected local bucket identical across ranks; mixed prefill/decode steps
retain separate layouts. The production comparison pins local buckets
`1,2,4,8,16,24,32`, maximum page-table width 128, and three backend warmup
forwards. Before every measured scenario, arm-neutral, trace-disjoint global
bursts of `4,8,16,32,64,96,128` requests, each retained for 16 completion
tokens, populate all seven local buckets.
The `/backends` witness taken after warmup must show all seven captures,
direct-NCCL ownership, and zero eager fallback. A second witness after traffic
must keep every structural field and counter fixed except for a strictly
increased replay count. Lazy capture or fallback during measurement is a
formal failure, not benchmark overhead to subtract.

Configured pipeline depth remains 5, but admission-sensitive work has a
smaller unresolved horizon. While the scheduler has a waiter or unfinished
prompt, or an already-submitted snapshot contains prefill, at most the
previous and current forwards may remain unresolved. A producer add/abort that
arrives during fill stops further schedule-ahead at the next loop observation.
Once all scheduler and pending work is pure decode, the configured depth 5 is
restored. This matches the bounded previous/current structure used by pinned
SGLang and vLLM async scheduling without reducing steady decode overlap.
Adapters that cannot prove a read-only prefill state fail safe at depth 2.

Pure-greedy attention-DP sampling also has one fewer host-control transaction.
The fixed NCCL token packet begins with one status slot per gathered rank,
followed by the existing owner-token slots. Every rank validates the same
status matrix; one execution or packet-encoding failure aborts the step on all
ranks. Status columns are copied asynchronously to pinned host memory on the
same copy stream/event as deferred public tokens rather than being converted
on the current CUDA stream. Driver and passive ranks retain the preceding
sidecar and, after the next common control broadcast, all resolve exactly one
FIFO entry without branching on rank-local event readiness; shutdown drains
the remaining tail. The driver and passive ranks therefore skip the final
Gloo reply gather only on this fast path without introducing either an eager
CUDA-to-host synchronization or divergent failure participation. Logprobs,
grammar, and every other non-fast sampling path retain the former all-rank
Gloo reply and validation.

The comparison uses the same four physical GPUs, immutable checkpoint,
BF16 KV, four request owners, EP4 ownership, FCFS policy, prompt/completion
trace, aggregate 65,536-token cache, and disabled speculative decoding. The
implementations' internal topology remains visible: Kairyu is TP1 plus
attention-DP4/EP4 with replicated attention projections, while SGLang's
v0.5.16 CLI records TP4/DP4/EP4 plus `--enable-dp-attention`. SGLang receives
16,384 cache tokens per owner so its aggregate capacity is 65,536 rather than
four times Kairyu's global pool. Kairyu configures an 8,192-token global
batched-prefill limit. SGLang configures an 8,192-token chunked-prefill limit
that v0.5.16 divides across DP4 and an explicit 2,048-token per-owner
`max-prefill-tokens`; both therefore resolve to 2,048 tokens per request owner.
This is a matched request-owner/EP comparison, not a claim that the two
runtimes use identical internal projection sharding. SGLang also fixes
`--log-level-http warning`, caps decode CUDA-graph batch size with
`--cuda-graph-max-bs-decode 32`, and disables prefill CUDA graph.

`bench/g4_ma3_kairyu_server.py` is the dedicated production-server launcher,
and `bench/g4_ma3_sglang_bench.py` prepares one immutable seed-0 ShareGPT trace.
The complete matrix contains exactly ten fresh, sequential generations: one
retained preflight for each already-fixed production arm, then four formal
pairs in K/S, S/K, S/K, K/S order. The preflights freeze their selection before
formal traffic. Each formal cell synchronously releases 128 requests at
concurrency 128 and requires exactly 128 streamed completion tokens per
request. Completion throughput is successful completion tokens divided by the
first-start-to-last-terminal span and four GPUs; TTFT p99 is nearest-rank over
all 128 requests. The gate uses the exact median of the four per-pair K/S
ratios: throughput must be at least 1 and TTFT p99 at most 1. It performs no
round-before-gate, outlier removal, retry, or failure exclusion. No additional
measurement generation belongs to the formal artifact.

The operator hashes every byte of the exact 27-shard checkpoint once before
the matrix and once after it. Assembly requires identical boundary captures
and binds every shard to the start descriptor. Every shard also binds one
fresh sequential server generation to the clean source commit, image
RepoDigest/platform/config identities, container and read-only model volume
with no read-write consumer, physical GPU inventory,
driver/CUDA/NCCL/Torch/FlashInfer versions, resolved runtime argv, and the live
`/backends` or `/server_info` response. The shard end independently
re-observes the same running container, source, runtime, GPU-process ownership,
and volume consumers. All operator commands execute the detached clean
`SOURCE_ROOT`; `capture-provenance` requires
`--checkpoint-start`, and assembly requires both boundary captures. Assembly
rejects a reused/overlapping server, changed provenance, changed
trace/selection, incomplete SSE/usage, retry, fallback, or unknown raw field.
Raw JSONL is authoritative; `verify` compares the derived manifest with an
independent replay, and `replay` ignores the stored manifest. SGLang's SM120
limitations are always disclosed beside the result but never modify the gate:
it uses FlashInfer CUTLASS rather than the SM100-only TRTLLM-gen MoE path,
prefill CUDA graph is disabled while decode graph remains enabled, and
MTP/speculative decoding is deferred to M-A4.

The first clean-commit matrix at
`55f3a8ca4513e158182d4b9b4a818c24f5ae7b34` completed all ten fresh
generations, 4,630 raw rows, and every non-performance binding check. Its four
throughput ratios were 0.741839/0.798127/0.829296/0.769510 and its exact median
was 0.783818; the exact median TTFT-p99 ratio was 1.352633. Both performance
checks therefore failed and retained verification/raw replay correctly reject
`--assert-gate`. The optimized horizon/status-packet candidate described above
is CPU-validated but not yet GPU-measured. The failed verdict remains
authoritative until a new complete clean-commit matrix independently passes;
no diagnostic or mixed-commit result can close M-A3.

## 5. Verification

- Unit: external metadata success/conflict/malformed cases, exact NVFP4 ABI,
  global checkpoint names/shapes, remote expert omission, selected shard-open
  count, packed-weight storage alias, prepared-operand invalidation after
  device/state changes, complete-forward success markers, and every
  construction refusal.
- Distributed CPU: EP2 parity, rank ownership, single-sampler packet
  adoption, failure propagation, and teardown.
- GPU: native FlashInfer NVFP4 projection oracle, NCCL EP2/EP4 tiny-model
  parity, then official-checkpoint load smoke on both degrees.
- Formal: both 1,024-position cells pass raw replay and manifest verification;
  all 64 free-running outputs, provenance, and topology are retained.
- M-A2 formal: the fixed 512-request EP4 trace passes the strict logical
  cache-rate gate, all four rank receipts are invariant, raw radix events are
  exact, and both manifest verification and raw-only replay pass.
- M-A3 formal: the fixed production arms complete one preflight each, the
  frozen selection precedes eight fresh sequential paired cells, the full
  ten-generation window is enclosed by identical checkpoint hashes, all
  Kairyu cells prove direct-NCCL plus graph replay without capture/fallback
  drift, and the exact paired medians satisfy both throughput and TTFT
  thresholds. All declared SGLang limitations remain visible without
  affecting the binding verdict.
- Repository: targeted tests, full applicable CPU/dist/GPU suites, ruff, and
  all required GitHub checks are green before merge.
