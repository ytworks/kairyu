# M3 Design: Speculative Decoding, CUDA Graphs, P-D Separation

Status: Draft — n-gram draft policy implemented CPU-side now; everything else gated on the
M2 GPU phase completing first
Milestone: M3
Date: 2026-07-02

## 1. Scope and ordering

M3 adds to the M2 engine: speculative decoding (n-gram draft first, EAGLE-family second),
CUDA graph capture for decode steps, and single-node P-D separation. Only the **n-gram
draft/verify policy** is GPU-independent, so it is implemented and tested now
(`kairyu/engine/core/spec_decode.py`); it drops into the EngineCore step loop once the GPU
ModelRunner exists (the runner scores draft tokens in one batched forward pass).

## 2. N-gram draft (implemented)

- `propose_ngram(context, max_draft, max_ngram, min_ngram)`: find the longest suffix
  (n = max_ngram..min_ngram) of the context that re-occurs earlier, propose the tokens that
  followed that earlier occurrence. No model call — pure prompt-lookup drafting, the same
  family as vLLM's `[ngram]` speculator; strongest on code/structured text with repetition.
- `verify_greedy(draft, target_tokens)`: accept the longest prefix of the draft that
  matches the target model's greedy tokens, then take the target's bonus token. Invariant
  (tested): **output equals plain autoregressive greedy decoding exactly** — spec decode
  changes latency, never results. Sampling verification (rejection sampling) arrives with
  EAGLE in the GPU phase.

## 3. Deferred to GPU phase (design sketch only)

- **EAGLE-family draft**: draft head on hidden states; requires the M2 runner's hidden
  state plumbing. Same `propose/verify` seam.
- **CUDA graphs**: capture decode-step graphs per batch-size bucket (vLLM V1 style);
  invalidated on page-table growth — the M2 page-granular KV layout was chosen so graph
  capture only depends on max page count, not radix topology.
- **P-D separation (single node)**: prefill and decode replicas on separate CUDA streams or
  GPUs with page-granular KV handoff (contiguous pages, design m2 §2.4); admission policy
  reuses the M2 scheduler with a prefill-only budget on one side and decode-only on the
  other.

## 4. Acceptance

Per goal: acceptance-rate and TTFT/TPOT deltas measured in `bench/` on the M2 GPU rig only;
the CPU tests here pin correctness (greedy-equivalence), not speed.
