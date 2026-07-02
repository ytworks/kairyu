# M3 Design: Speculative Decoding, CUDA Graphs, P-D Separation

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (agent design-review panel, 2026-07-02;
see §5). n-gram draft policy implemented CPU-side; everything else gated on the M2 GPU
phase. Human sign-off pending.
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
  Differs from vLLM in preferring the most *recent* match (vLLM takes the earliest); the
  naive O(max_ngram·len) scan is acceptable CPU-side — switch to KMP if it runs per-step
  at long context.
- `verify_greedy(draft, target_tokens)`: accept the longest prefix of the draft that
  matches the target model's greedy tokens, then take the target's bonus token. Contract:
  `target_tokens[i]` must be the target's greedy token conditioned on `context + draft[:i]`
  (one batched forward over draft positions on GPU). Invariant (tested at policy level):
  output equals plain autoregressive greedy decoding **given identical target logits**; on
  real GPUs batch-shape reduction-order nondeterminism can flip argmax ties, so the bench
  measures output-match rate rather than asserting bit-exactness. Sampling verification
  (rejection sampling) arrives with EAGLE in the GPU phase.

### 2.1 Scheduler integration protocol (amendment — required before wiring in)

The M2 scheduler is single-token-per-decode-step today. Spec decode integration requires:
(a) decode chunks carrying draft length with `num_tokens = draft+1` charged against the
token budget; (b) KV capacity reservation of `draft+1` slots plus freeing rejected draft
positions; (c) `update()` accepting variable-length accepted sequences with `in_flight`
reconciliation; (d) truncation at `max_new_tokens` when acceptance overshoots. Under the
overlap loop, `position = outputs + in_flight` is stale when acceptance counts vary —
first integration runs with the overlap pipeline at depth 1 (serial) until the
reconciliation protocol is specified and tested.

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

## 5. Review record

Agent design-review panel, 2026-07-02. Verdict: **APPROVE-WITH-AMENDMENTS**. Disposition:
scheduler multi-token protocol added as §2.1 (design; implementation lands with GPU
phase); greedy-equivalence claim softened to logits-conditional with the target_tokens
conditioning contract stated; n-gram match-selection difference vs vLLM documented; §3
corrected — the P-D admission-policy half (independent prefill/decode budgets) is already
implemented in the M2 scheduler (`pd_separation`, `decode_token_budget`), only the
stream/replica/KV-handoff half is deferred. Human sign-off pending.
