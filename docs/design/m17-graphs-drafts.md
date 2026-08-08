# M17 Design: StepExecutor (CUDA-Graph Seam) + EAGLE/MTP Draft Models

Status: **Implemented and production-enabled** (2026-07-26). Reviewed — REVISE
applied (1-reviewer panel, web-verified vs vLLM/SpecForge/SGLang + live
SpecForge and DeepSeek-V3 safetensors headers; §6 binding).
Milestone: M17 (roadmap E5/E6 local halves; G2 A-series latency levers)
Date: 2026-07-03
Depends on: M8 (SpeculativeRunner/propose_ngram/verify), M12/M15 (models,
runner), M13 (backend seam). Consumed by: deploy day (real CUDAGraph capture,
EAGLE checkpoints).

## 1. Goal

Two decode-latency levers, implemented so the GPU day is configuration:

1. **StepExecutor seam** — the capture/replay lifecycle around decode-step
   execution, with ALL policy (bucket sizes, capture eligibility, cache
   invalidation) CPU-tested against a fake graph; `cuda_graph_gpu.py` holds
   the only CUDA-touching lines.
2. **Draft models** — `DraftSource` protocol generalizing M8's n-gram
   proposer; `EagleDraftHead` (fusion + one decoder layer) and
   `MtpDraftHead` (DeepSeek MTP layer) as CPU-runnable modules with
   random-weight invariant tests + SpecForge/DeepSeek checkpoint loaders.

## 2. Key design decisions

### D1 — `StepExecutor` protocol (`engine/core/step_executor.py`)

`execute_decode(batch: DecodeBatch) -> Tensor` where `DecodeBatch` is the
frozen decode-shaped input (token ids [B], positions [B], page tables,
seq_lens). Implementations:
- `EagerStepExecutor` — call the model directly (today's behavior, default).
- `GraphStepExecutor` — pads B up to the nearest bucket, replays a captured
  graph per bucket, captures on first use; holds STATIC input buffers per
  bucket and copies inputs in (the CUDA-graph contract); invalidates all
  captures on `invalidate()` (weight swap / pool resize).
The graph OBJECT is behind a `GraphBackend` protocol (`capture(fn, inputs)
-> Replayable`): `FakeGraphBackend` (CPU tests: records capture count,
replays by re-running fn with the static buffers, asserts no re-capture,
detects shape drift) and `cuda_graph_gpu.CudaGraphBackend` (@gpu,
torch.cuda.CUDAGraph + side-stream warmup + graph pool).

### D2 — `graph_buckets.py` (pure policy)

`decode_buckets(max_batch) -> tuple[int, ...]`: [1, 2, 4, 8, 16, 24, 32,
then +8 steps] capped at max_batch (vLLM-style cudagraph_capture_sizes
convention). `bucket_for(batch, buckets)` = smallest bucket ≥ B, None → eager
fallback (never crash). Padding rows replay with page_table pointing at a
dedicated scratch page and are dropped from outputs (correctness pinned on
CPU via the fake backend).

### D3 — `DraftSource` protocol (`engine/core/draft.py`)

`propose(request_state, k) -> list[int]`. `NGramDraftSource` wraps M8's
`propose_ngram` (params: n, window). `ModelDraftSource` runs a draft head
autoregressively for k tokens off the target model's last hidden state +
sampled token. `SpeculativeRunner` gains a `draft_source` arg (default
n-gram — existing behavior byte-identical); verify path unchanged (M8's
verify_greedy is draft-agnostic).

### D4 — EAGLE-3 head (`models/eagle.py`) + SpecForge loader

`EagleDraftHead`: `fc` fusion projecting concatenated low/mid/high target
hidden states [3H] → H, one llama-style decoder layer (reusing M12
`DecoderLayer` machinery where possible), `norm` + target-tied lm_head.
Draft step input = fused hidden ‖ embedding of the last sampled token.
CPU invariant tests: shape contract, deterministic greedy rollout,
autoregressive state advance. `eagle_loader.py` maps SpecForge checkpoint
names → module tree (fail loudly on unknown tensors).

### D5 — MTP head (`models/mtp.py`)

DeepSeek-V3 MTP layer: `enorm`/`hnorm` RMSNorms, `eh_proj` ([2H] → H)
fusing normed embedding ‖ normed hidden, one full DeepSeek decoder layer
(MLA + MoE — reuses M15 modules), `shared_head.norm` + target lm_head.
Loader maps the `model.layers.{N}` MTP-extra layer from DeepSeek checkpoints
(`num_nextn_predict_layers`). k>1 MTP = reapplying the head on its own
output (DeepSeek convention).

## 3. Non-goals

- Real CUDAGraph capture/tuning, graph memory pools sizing (deploy day).
- EAGLE tree attention (top-k branching) — linear-chain drafts only (the
  M8 verify contract); tree is a G4-era extension.
- Draft-model training; piecewise graphs (attention outside graph).

## 4. Phasing

1. graph_buckets + StepExecutor + FakeGraphBackend suite; PagedModelRunner
   opt-in wiring (`step_executor=` kwarg, default eager, decode-only).
2. DraftSource + SpeculativeRunner integration (n-gram default pinned).
3. EagleDraftHead + loader + ModelDraftSource e2e (draft==target tiny →
   100% acceptance ≡ greedy).
4. MtpDraftHead + loader; tests/gpu mirror for CudaGraphBackend.

## 5. Verification

- Fake-graph suite: capture-once-per-bucket, replay N times, padding rows
  dropped, shape-drift assertion, invalidate() forces re-capture, oversize
  batch falls back to eager, captured outputs == eager outputs (CPU).
- Draft e2e: tiny target as its own draft → acceptance 1.0 and output ==
  plain greedy through the FULL engine; n-gram path regression-pinned.
- EAGLE/MTP: random-weight forward shape/determinism invariants; loader
  round-trips a synthetic SpecForge/MTP checkpoint written by the tests.

## 6. Review record (binding amendments, applied)

- **A1 (BLOCKING, scope honesty)**: no batched decode path exists in the model
  stack — M17 delivers the EXECUTOR/POLICY layer (bucketing, capture-once,
  padding, invalidation, eager fallback) fully CPU-pinned against
  FakeGraphBackend with a synthetic decode_fn. The real batched capture rides
  FlashInfer's decode wrapper (use_cuda_graph + fixed-size index buffers) on
  deploy day; the eager torch backend is NON-capturable (per-step shapes,
  Python page-list metadata) — recorded, not hidden. The production runner now
  passes the host-owned contiguous chunk boundary and writable suffix once to
  sequential dense and MLA attention; those paths neither extract device
  scalars per layer nor use CUDA boolean indexing for KV writes. The original
  arbitrary-position model call retains its mask-based compatibility path.
  This removes eager prefill host drains without claiming that list metadata is
  graph-capturable.
- **A2 (BLOCKING, EAGLE-3 corrected)**: lm_head is TRAINED over a reduced
  draft vocab with a ``d2t`` int64 OFFSET map (target_id = draft_id + d2t);
  midlayer q/k/v in_features = 2H over cat([input_layernorm(embeds),
  hidden_norm(hidden)]), residual = pre-norm hidden; fc [H, 3H] applied ONCE
  per verify cycle; embed_tokens absent (target-aliased); checkpoint names
  ``midlayer.*``/``fc``/``norm``/``lm_head``/``d2t``/``t2d``.
- **A3 (BLOCKING)**: draft-head KV — the CPU reference recomputes densely per
  proposal (sidesteps rejection bookkeeping); paged draft-KV is a deploy-day
  optimization behind the same rollout contract.
- **A4 (BLOCKING)**: the target hidden seam — forward_tokens already returns
  post-norm hidden ("M17's tap"); EAGLE-3 needs PRE-norm residual-added aux
  hiddens from 3 layers — the fuse() input contract; engine integration of
  aux capture lands with the GPU EAGLE path (G4), the head itself is fully
  pinned now.
- **A5-A6**: pad rows: seq_len 1, position 0, token 0, scratch page (writes
  land in one scratch slot — benign, outputs dropped); invalidate() on weight
  swap AND pool reallocation (graphs capture raw pointers); page-table
  GROWTH does not invalidate (max-width static buffers); sampling stays
  outside the graph.
- **A7**: graphs capture B×1 decode only; multi-token verify capture is
  future work. Grammar-rollback speculation stays deferred (re-deferred from
  m8 — recorded).
- **A8 (MTP corrected)**: layer id = num_hidden_layers; eh_proj concat is
  EMBEDDING-FIRST (implementations win over the paper's equation);
  shared_head.head and embed_tokens are SEPARATE physical tensors (never
  assume tying); decoder block built with layer_index = num_hidden_layers so
  it correctly comes out MoE.
- **A9**: propose_ngram params are (max_draft, max_ngram, min_ngram);
  DraftSource may return fewer than k tokens (scheduler degrade paths
  already honor it).

### 2026-07-26 production amendment

- **A10 (explicit serving policy):** `KairyuBackend` and DeploymentSpec expose
  `decode_mode: eager|cuda_graph`; eager remains the default. Graph mode
  constructs `CudaGraphBackend` only for a real CUDA model whose attention path
  declares the graph contract. Invalid dimensions, CPU placement, P-D
  separation, and unsupported model/attention paths fail at startup. Batches
  or page tables outside capture limits retain D2's eager fallback.
- **A11 (distributed ownership):** the scheduler reserves the scratch page
  before TP workers are spawned and passes its exact id to every rank. Captured
  NCCL graphs are explicitly invalidated before shutdown. Replayables must not
  retain `CUDAGraph` through a class closure; serving subgroups are drained,
  synchronized, and destroyed in identical order on every rank before the
  default group.
- **A12 (capture width is workload capacity):** a deployment's
  `cuda_graph_max_pages * page_size` must cover the sequence length it intends
  to accelerate. The Qwen3-32B quality example uses 512 pages (8,192 tokens)
  instead of 64; wider requests still retain D2's explicit eager fallback.
  FlashInfer plans only live page indices, so this fixed graph-input width does
  not make attention traverse padding. The same example leaves the attention
  override empty so hardware policy selects FlashInfer rather than forcing the
  torch reference path.

### 2026-07-30 batched-verification amendment

- **A13 (explicit runner capability):** `supports_batched_verification is
  True` opts a `ModelRunner` into non-prefill chunks wider than one target
  position. Undeclared custom runners retain A7's one-position calls, selected
  before execution; a malformed opted-in result fails without retrying a model
  or KV side effect.
- **A14 (flattened target geometry):** the native runner flattens
  `[previous, draft[0], ..., draft[k-1]]` across compatible requests. Every row
  has its own absolute position and causal length, duplicate page-table view,
  and unique writable physical slot. All rows write KV before each layer's
  attention, so earlier rows mask later draft positions while later rows can
  attend to the earlier draft KV exactly as in sequential target scoring.
  Rejected suffix KV remains stale and is overwritten from the first
  correction position; no clearing pass is required.
- **A15 (graph capacity is rows):** speculative graph capacity is
  `min(token_budget, request_capacity * (k + 1))`. The graph bucket is selected
  from the total target positions, padding still writes only the scheduler's
  reserved scratch page, and an in-range formal cell must report one graph
  dispatch with zero eager fallback.
- **A16 (TP target packet):** the rank-0 packet owns one fixed int64 slot per
  target position (partial prefill retains one sentinel). Offsets are derived
  only from the broadcast `ScheduledChunk` tuple. Every follower executes the
  same model/KV rows and adopts every authoritative target token device-side.
- **A17 (eager policy by measurement):** the selected eager implementation is
  flattened tensor decode, shared with the graph path. On the pinned full
  Qwen3-32B checkpoint and the same 8-request/32-position/page/write geometry,
  warmed alternating measurements gave median CUDA time 59.004 ms versus
  75.561 ms for native ragged prefill (`flattened/native = 0.7809`), with all
  32 selected tokens equal. Cross-kernel BF16 KV numerical distance is retained
  as a diagnostic rather than an equality gate; production correctness remains
  bound directly against sequential target scoring. Timing never decides the
  issue's pass/fail result.

### 2026-07-30 decode page-table cache amendment

- **A18 (bounded storage, exact rollback):** each `PagedModelRunner` owns one
  geometrically grown int32 page-table tensor per device, rather than a
  request-keyed tensor cache. Growth copies the old compatible rectangle D2D,
  preserves row signatures, and never changes a captured graph's separate
  static buffers. The runtime OFF mode passes neither the cache nor ownership
  metadata, so it remains the former allocate/populate plus full graph-copy
  path and is a valid matched baseline. A narrow view over wider reserved
  capacity may be strided; the internal tensor contract does not promise
  contiguity, and the supported Torch gather/index plus graph copy paths accept
  that layout. A future custom tensor backend must validate any stronger layout
  requirement explicitly.
- **A19 (ownership and staleness):** a row signature binds the request ID,
  request-local lane, owned page-ID tuple, visible width, and padding page.
  Ordinary decode uses lane 0; flattened speculative positions use lanes
  1..k. Stable rows are hits, sequence growth or page reallocation uploads only
  the smallest safe changed range, row movement/owner replacement rewrites the
  complete row, and width regrowth treats previously hidden columns as
  unknown. Request release forgets all lanes in both the reusable tensor and
  every graph bucket before an ID can be reused; storage may remain allocated
  but retains no request ownership.
- **A20 (static graph copy and evidence):** graph page-table addresses stay
  fixed. Trusted host signatures allow only changed dynamic ranges to be copied
  into each bucket; metadata-free callers keep the unconditional rectangular
  copy. Structural allocation/upload/copy/hit/release counters are gathered
  from every TP rank over the bounded model communicator. Qwen3-32B TP8 output
  parity and deterministic counter reductions decide retention; wall/CUDA
  timing is diagnostic because OS jitter cannot decide correctness.

### 2026-07-31 quantized draft-head amendment

- **A21 (format ownership):** an external EAGLE checkpoint owns its standard
  `quantization_config` and never inherits target-model packing. An MTP layer
  embedded in the target checkpoint inherits target quantization unless
  `draft_quantization_config` is present; `null` or `{}` is an explicit dense
  override. Both loaders require `config.json` and compare the complete
  checkpoint semantics with the caller configuration before constructing the
  head.
- **A22 (initial supported dialect):** the only admitted quantized draft format
  is compressed-tensors dynamic FP8 with one Linear group, per-channel FP8
  weights, and dynamic per-token FP8 activations. Unsupported methods,
  strategies, layouts, or ambiguous metadata fail before tensor loading.
  Eligible projections are built through the contextual draft factory while
  retaining checkpoint-canonical names. MTP router and MLA `kv_b_proj`
  projections remain dense and must be covered by canonical checkpoint ignore
  rules.
- **A23 (packed-only serving boundary):** loading never quantizes a dense draft
  online, dequantizes a packed draft, or substitutes another execution format.
  Packed weight dtype/shape is exact; FP8 scale tensors may be losslessly
  normalized to the registered FP32 ABI only after finite and positive checks.
  `pack_dynamic_fp8_draft_state` is an explicit offline conversion helper and
  preserves the source dtype of dense members.
- **A24 (public EAGLE geometry and evidence):** EAGLE parsing retains explicit
  query-head, KV-head, and even rotary head-width geometry; Qwen3-32B therefore
  executes 64-query/8-KV-head GQA instead of deriving a false MHA shape.
  Default auxiliary captures translate SGLang's before-layer taps to Kairyu
  after-layer outputs: `(1, N/2-1, N-4)`, or `(1, 31, 60)` for 64 layers.
  Draft input follows the trained EAGLE target-root contract: auxiliary row
  `t` pairs with the target embedding for token `t+1`, so the shifted embedding
  sequence ends with the target-produced root. Verification evaluates
  `[root, *proposals]`; proposal decisions use all but the final target-logit
  row, and the final row supplies the all-accepted bonus/correction.
  Accepted proposal prefixes remain exact against the sequential teacher.
  Because sequential and multi-token target shapes can cross a BF16 near tie,
  correction parity binds both selected tokens under both distributions to the
  established 0.25-nat reciprocal log-probability limit; exact correction
  equality remains a reported diagnostic rather than an impossible bitwise
  cross-shape requirement.
  The fixed public trained head is compared dense versus offline-packed FP8 on
  one real Qwen3-32B target with identical teacher traces, exact greedy target
  correction, acceptance, latency, memory, and committed-token goodput.
- **A25 (scope boundary):** issue #234 owns draft construction, checkpoint
  compatibility, fused-kernel execution, and trained-head verification. Native
  serving still accepts only the established n-gram speculative source;
  wiring EAGLE/MTP proposal state into `ModelDraftSource` remains the existing
  G4 runtime milestone and is not implied by the standalone trained-head gate.

### 2026-08-06 startup-capture amendment

- **A26 (resolved default and coverage):** omission of `decode_mode` selects
  CUDA graphs for a graph-capable real CUDA model in single-rank, TP, and
  request-owned attention-DP EP serving. CPU, model-less/custom, P-D,
  explicitly force-CPU TP, replicated-attention EP, and current MLA paths
  resolve to eager; an explicit `cuda_graph` request remains fail-closed. An
  omitted graph batch limit tracks
  `max_num_seqs`. An omitted page limit covers
  `ceil(max_model_len / page_size)`, capped by scheduler-visible KV capacity;
  when no context limit is declared it covers every non-scratch KV page.
- **A27 (readiness and distributed convergence):** every configured bucket is
  captured after weights and the final serving communicator are resident but
  before the engine builder or distributed launcher can publish readiness.
  TP/EP first compare bucket, pending-state, page-width, scratch-page, and
  capture-forward identities on a dedicated bounded host control group (the
  long-idle serving control group is not used). Attention-DP also constructs
  direct NCCL and performs both startup probes and live ragged-layout agreement
  on that bounded group; no readiness-critical or live per-step collective is
  allowed onto the group whose receive may legitimately idle for the server
  lifetime. It then arms one uniform row layout per warmup/capture forward and
  gathers preparation ACKs before any rank enters model collectives. Partial
  capture invalidates the complete local graph set; all ranks synchronize CUDA
  and gather final ACKs. The first live request therefore replays an existing
  bucket rather than recording it inline. A checkpoint-independent EP4 GPU gate
  executes this exact readiness transaction with real CUDA graphs and real
  direct-NCCL mixed-dtype gather/reduce-scatter, then proves the first decode
  replays without another Python forward, capture, layout step, or fallback.
- **A28 (fallback observability):** the existing structural eager-fallback
  count is exported at Prometheus scrape time as
  `kairyu_cuda_graph_eager_fallbacks_total{model=...}`. The counter increments
  only when a live decode batch or page table exceeds configured graph shape;
  startup capture itself does not increment replay or fallback counters. A
  local replica pool exports the model-level sum; replica generations and raw
  child-process resets are accumulated so removal, replacement, restart, or a
  temporarily failed scrape can never decrease the Prometheus counter.

- **A29 (eager FlashInfer fast plan):** scheduler-owned host sequence lengths
  now accompany both ordinary tensor eager decode and graph-shape eager
  fallback. Each `(batch, page width)` wrapper uses stock FlashInfer `plan()`
  once for initialization, then shares the existing fixed-shape Triton metadata
  pack plus `fast_decode_plan` path with graph replay; steady eager planning has
  neither boolean-mask `nonzero` nor a device-to-host schedule copy. The earlier
  generation-key proposal is superseded by A17/A18: production CUDA eager
  decode no longer enters the list `_plan`/`attend_batched` cache, its wrapper
  key is O(1) shape/dtype metadata, and the bounded page-table cache already
  owns host-side change detection. Legacy list compatibility paths remain
  unchanged rather than adding an unused generation contract.

### 2026-08-08 learned-draft serving amendment

- **A30 (stateful learned-draft contract):** native configuration admits
  `speculative: eagle|mtp` only with an explicit target checkpoint and
  `draft_model_path`. Learned drafting uses eager target execution so each
  authoritative target input row can retain either the EAGLE auxiliary fusion
  taps or the post-final-norm MTP hidden. Token inputs are shifted by one while
  absolute positions stay unchanged: hidden row `t` pairs with token `t+1`,
  including the target-produced root. A RadixKV prefix hit without matching
  hidden history fails safe to ordinary target decode for that request. This
  supersedes A25's native-serving scope boundary.
- **A31 (rollout, rollback, and ownership):** one proposal cycle runs the
  learned head over its context once, then appends one-token draft K/V for the
  remaining proposals (`O(T+k)`, not `O(k*T)`). Target verification stages all
  candidate input rows and retains exactly the accepted prefix plus one
  correction/bonus input row. Single-rank serving owns the head locally; TP
  rank 0 owns the learned head and capture history while all ranks still enter
  identical target-model collectives. Draft source, proposed/accepted counts,
  and mean acceptance are emitted together in benchmark evidence.
- **A32 (request-local commit dependency):** an outstanding variable-length
  speculative result blocks only its own next scheduler snapshot. Independent
  requests may fill later pipeline slots; the prompt-completing root remains a
  commit barrier before that request's first proposal.

- **Measurement:** Qwen3-32B on 8x RTX PRO 6000 Blackwell, TP8, 8 concurrent
  synthetic requests x 32 output tokens, torch attention: tensor eager wall
  8.844 s, TPOT 192.075 ms/token, 0.90 req/s; CUDA graph wall 7.196 s,
  TPOT 130.297 ms/token, 1.11 req/s. That is 18.6% lower wall time, 32.2%
  lower TPOT, and 23.3% higher throughput after separating the #207 tensor
  metadata improvement from graph capture. The graph run includes first-use
  warmup and capture. Evidence:
  `bench/results/decode-row-sync-qwen3-32b-tp8-2026-07-26.json`.
- **Issue #150 gate:** the Qwen3-32B TP8 LiveCodeBench 20-item subset at
  concurrency 8, 8,192 max output tokens, and a 600 s request timeout completed
  20/20 with zero inference failures. Maximum request latency was 460.681 s;
  total pair time was 1,049 s because twenty requests were served in waves of
  eight. Evidence:
  `bench/results/issue-150-qwen3-32b-tp8-livecodebench-2026-07-26.json`.
