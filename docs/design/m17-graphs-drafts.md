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
  ``positions[0].item()`` host syncs) — recorded, not hidden.
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
- **Measurement:** Qwen3-32B on 8x RTX PRO 6000 Blackwell, TP8, 8 concurrent
  synthetic requests x 32 output tokens, torch attention: eager wall 16.722 s,
  TPOT 441.831 ms/token, 0.48 req/s; CUDA graph wall 8.928 s, TPOT
  194.200 ms/token, 0.90 req/s. That is 46.6% lower wall time, 56.0% lower
  TPOT, and 87.5% higher throughput. The graph run includes first-use warmup
  and capture. Evidence:
  `bench/results/cuda-graph-qwen3-32b-tp8-2026-07-26.json`.
