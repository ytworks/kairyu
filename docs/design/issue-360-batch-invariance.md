# Issue #360 Design: A12 Batch-Invariance Determinism Gate

Status: **GPU-validated** (2026-08-05).

Related contracts: M2 radix-paged KV and chunked prefill, M13 FlashInfer
attention selection, M15 MLA/MoE execution, and M17 CUDA-graph decode buckets.
Issue #317 can reshape a decode row only in a mixed prefill/decode step; none of
the frozen A12 arms schedules such a step, so its retained claim is unchanged.
The `A12` name in this document is the 2026-08-03 accuracy-review identifier;
it is unrelated to the historical M16 decision with the same short label.

## 1. Goal and limits

One fixed factual prompt must produce the same complete native greedy answer
when its model work is reshaped by production batching and cache state.  The
formal gate compares four executions of the same retained 129-token prompt:

1. a cold target co-batched with 31 unrelated requests;
2. the cold target alone after its prefix has been proved evicted;
3. the target alone with a proved 128-token page-aligned native radix hit and
   the required one-token terminal recomputation; and
4. the cold target alone in a fresh runtime whose 32-token prefill budget
   forces the 129-token prompt into `32, 32, 32, 32, 1` chunks.

Every target request uses `temperature=0`, `seed=0`,
`min_tokens=max_tokens=32`, `ignore_eos=true`, and
`skip_special_tokens=false`.  All four responses must contain exactly the same
32 native token IDs, raw vocabulary pieces, and final text, and must finish by
the length bound.  There is no text-only comparison, prefix tolerance, or
post-hoc disagreement allowance.  Timing is not a verdict input.

This is a same-checkpoint, same-host, same-source shape-invariance claim.  It
does not claim cross-commit reproducibility (issue #369), agreement with
Transformers, or exact equality across different models or GPU generations.
Qwen3-32B is dense GQA, so the formal run covers the production FlashInfer,
radix, chunked-prefill, TP, and graph paths but does not claim real-model MLA
or MoE numerical coverage.  Portable MLA/MoE tests remain control-path
coverage only.

## 2. Frozen production geometry

The formal child is one clean-source Qwen3-32B run at TP8 on exactly eight
RTX PRO 6000 Blackwell Server Edition GPUs.  It binds the complete reviewed
17-shard checkpoint and tokenizer, FlashInfer selection, BF16 KV, one pipeline
stage, and 128 configured 16-token pages (127 allocatable after the permanent
graph scratch page), with `max_model_len=192` and `max_num_seqs=32`.  CUDA graph
decode is mandatory with maximum batch 32,
maximum page width 16, and three capture warmups.

The persistent full runtime has `max_num_batched_tokens=512`; the separate
fresh chunk runtime has the same identity and geometry except for the fixed
32-token budget.  Thirty-one distinct distractors are submitted before the
first scheduling step.  Each is forced to survive for 32 output tokens, so
the target's ordinary decode remains in a real 32-row cohort rather than only
sharing a prefill call.  The retained schedule proves the exact first prefill
cohort and every target-bearing decode cohort.  All eight ranks must report
FlashInfer, graph bucket 32, positive capture/replay, and zero eager fallback.

The target descriptor fixes the full UTF-8 text, exact 129 tokenizer IDs,
text SHA-256
`73c8c1bc844c21fc7a9c49b3e9dbcc43297e0144da435d12f8fe7dd4f09645a8`,
and canonical token-list SHA-256
`a47440ea80a9e87ac8f30202651b582812e520ea4f1789044b843ef635e7823f`.
Native tokenization must reproduce the retained IDs before GPU execution.
All prompt and output IDs are bounded by Qwen3's tokenizer-owned domain
`0 <= id < 151669`; the padded 151936-row logit width is not a tokenizer
vocabulary boundary.

## 3. Causal cache and schedule proof

The full runtime executes phases in the immutable order
`batch32_cold`, `pressure`, `alone_cold`, `alone_warm`.  Pressure consists of
the six fixed 129-token lowercase `a` through `f` prompts and is not a
comparison output. Their first-page roots are disjoint from all 31
distractors, so cold-pressure proof cannot depend on generated continuations.
With the frozen capacity, five such requests leave the target resident; the
sixth is the necessary and sufficient deterministic eviction pressure.
Immediately after pressure, the
non-mutating radix probe must report zero cached target tokens.  The following
cold request must independently report zero cached prompt tokens and execute a
complete prefill.  Before the warm request the same probe must report the
maximum reusable 128-token page-aligned prefix, native usage must independently
report 128 cached tokens, and the schedule must retain the required one-token
terminal prefill.  RadixKV intentionally recomputes the last prompt token even
when every preceding page is reusable.  These separate observations prevent a
scenario label or copied usage value from fabricating a cache transition.

The one-token terminal prefill is also bound at the attention seam.  It counts
as one scheduler/runner prefill call but, by the production `chunk_len == 1`
dispatch, performs one stock FlashInfer decode plan/run.  The warm arm therefore
retains zero prefill-backend plans and one decode-backend plan/64 layer runs;
the chunk arm retains four prefill-backend plans plus that terminal decode call
before its new graph bucket is captured.  Treating every scheduler prefill row
as a prefill-backend invocation would make the formal gate reject the exact
shape-dependent path it exists to exercise.

The chunk runtime has a different nonce and a fresh empty cache.  Its target
pre-probe and native cached usage must both be zero.  The retained schedule and
runner counters must prove five sequential prefill calls with the exact chunk
lengths `32, 32, 32, 32, 1`; merely recording the configured budget is not
enough.

The batch scenario must prove one 32-row native prefill group and 31 target-
bearing 32-row decode steps after the prefill-selected first token.  Alone
scenarios must prove one-row execution.  Rank topology and graph counters are
snapshotted around each phase so a missing follower, eager fallback, or
relabeled batch cannot satisfy output equality.

## 4. Ordinary MLA decode correction

The gate work exposed an independent deterministic failure in the ordinary
decode dispatcher.  Two or more DeepSeek/MLA rows were sent to
`DenseDecoder.forward_decode_batch` whenever tensor decode was unavailable,
even though `MlaAttention` has no list-batched implementation.  Construction
now computes one shared decode-batch capability gap.  Ordinary and
speculative decode enter a batched path only when either tensor decode or the
complete model/layer/backend list-batch contract is available; unsupported
MLA/custom stacks execute rows sequentially.

The check occurs before any model call.  Catching `AttributeError` and retrying
would be unsafe because preceding layers may already have written KV.  Tensor-
unsupported custom models with a valid list-batched contract remain batched,
so the correction does not serialize supported eager implementations.

## 5. Evidence and replay boundary

`bench/batch_invariance_bench.py` is a checkout-only, path-only formal
operator.  Evidence commands must begin as `python -I -B` through the direct
script path.  Before importing repository code, the wrapper proves the tracked
tree clean, rejects import-shadowing untracked/ignored code and symlinks, and
uses an unpredictable private bytecode-cache directory.  Loaded repository
modules must be tracked and byte-identical to `HEAD`.

`run-native` records one canonical JSONL raw stream and derives a manifest
through `kairyu.bench.batch_invariance`.  The source snapshot and complete
checkpoint are hashed before runtime construction and again after both
runtimes shut down.  Hardware, CUDA/NCCL/Torch, backend selection, runtime
nonces, request rows, schedules, cache probes, all-rank counters, native
outputs, and recomputed checks are retained.  Output creation is exclusive;
existing paths are never overwritten, and written bytes are immediately
re-read and rehashed.

`verify` hashes and strictly parses one immutable raw-byte snapshot, checks the
derived manifest, then independently recomputes every verdict.  `replay`
derives the result from raw evidence without trusting stored `checks` or
`passed`.  Duplicate JSON keys, non-finite numbers, bool/integer coercion,
unknown or missing rows, row reordering, hash drift, source/checkpoint drift,
rank loss, graph fallback, cache-proof failure, and any output difference fail
closed.  Portable fake-runtime and tamper tests validate this operator and
contract; they make no real-GPU execution claim.

The formal Qwen3-32B TP8 run at source commit `d5044c2` produced 38 canonical
raw rows and passed all 28 derived checks.  Direct-path retained verification
and raw replay both passed with raw SHA-256
`c42797f18b8db7b9c87ab9203a3abc2bf0b26aaa2264d9ce42024fbcc5bf8b88`.

## 6. Decisions

- **A12-D1:** exact 32-token native equality is binding; timing and a measured
  tie floor are diagnostic only.
- **A12-D2:** one retained run must causally prove all four shapes, not combine
  unrelated historical outputs.
- **A12-D3:** schedule, cache, and all-rank graph observations are prerequisites
  for comparing answers; scenario labels alone are not evidence.
- **A12-D4:** the chunk arm uses a fresh runtime because its scheduler budget is
  construction-time geometry, while the first three phases share one runtime
  to prove the cold/warm transition.
- **A12-D5:** unsupported ordinary decode batching falls back before model/KV
  mutation; valid list-only batching remains supported.
- **A12-D6:** formal evidence is retained as canonical raw plus a derived,
  independently replayable manifest under a fail-closed checkout boundary.
