# G4.1 Design: Mid-tier MoE — Real NVFP4 Loading and Correctness

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (2026-08-01; the M-A1 best
implementation is retained with its formal FAIL, while M-A2 production
integration and formal hardware evidence are in progress).
Milestone: G4.1 M-A1, with explicit seams for M-A2 and M-A3.
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

M-A3 must replace correctness-mode attention duplication with request-owned
attention-DP and add an overlap strategy chosen from measured throughput. It
may reuse the grouped/fused local-expert ABI introduced here. It must compare
Kairyu and SGLang sequentially at equal checkpoint/config and publish
steady-state token throughput and TTFT statistics. M-A1 timing cannot be
reused as M-A3 evidence.

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
- Repository: targeted tests, full applicable CPU/dist/GPU suites, ruff, and
  all required GitHub checks are green before merge.
