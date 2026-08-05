# Issue #368 Design: Exact Continuation Log-Likelihood for Benchmarks

Status: **Implemented** (2026-08-05).

Related contracts: M8 D2 (raw log-probability reporting), M9 D3 (legacy
completions), issue #362 (raw vocabulary pieces), and issue #367 (core evals).

## 1. Goal and non-goals

The installed quality suite needs the conditional log-likelihood of an exact
candidate continuation, not the probability of whichever text the model chose
to generate.  For context token sequence `x` and continuation tokens
`c = (c_1, ..., c_n)`, the score is

```
L(c | x) = sum_j ln p(c_j | x, c_<j)
```

Every `c_j` is therefore teacher-forced and its own raw, pre-sampling-processor
log-probability is returned.  Generated top-k membership is diagnostic only;
it is not a substitute for this value.  A candidate outside top-k, a
multi-token candidate, or a duplicate decoded vocabulary piece must still be
scored exactly.

This change supplies that reusable request/scoring path and moves the core
MMLU row from generated-letter parsing to ordered A-D continuation ranking. It
does not add WikiText data/windowing or claim a perplexity result.  The same
multi-token primitive can support a later corpus adapter without changing its
mathematics.  MMLU remains the explicitly disclosed zero-shot Kairyu variant;
canonical five-shot prompt construction is a separate methodology change.

## 2. Wire contract

Kairyu extends non-streaming `POST /v1/completions` with one optional field:

```json
{
  "model": "m",
  "prompt": "...Answer:",
  "kairyu_continuation": " A",
  "max_tokens": null,
  "temperature": 0.0,
  "stream": false,
  "n": 1,
  "logprobs": 0
}
```

The extension deliberately does not overload OpenAI `echo`.  The response is
an ordinary `CompletionResponse` with `mode: "loglikelihood"`; its single
choice contains the continuation text, the standard selected-token logprob
arrays, and explicit `prompt_token_ids` / `continuation_token_ids` evidence.
The request is accepted only when all of the following hold:

- `prompt` and `kairyu_continuation` are non-empty strings;
- `max_tokens` is explicitly `null`, `n == 1`, `stream == false`,
  `temperature == 0`, and `logprobs == 0`;
- sampling filters, penalties, stop rules, and random seeds cannot change the
  scoring contract; non-null seeds and unknown request extensions are rejected,
  while EOS and special-token handling remain internally owned;
- the selected backend exposes the exact tokenizer and native forced-token
  path.

An ordinary completion request without `kairyu_continuation` retains its
previous body, validation, generation, and response behavior.  A remote or
otherwise incapable backend returns a stable 400 capability error before any
generation.  The benchmark turns that explicit first-item capability result
into a skipped pair rather than launching the full dataset or recording model
errors as wrong answers.

## 3. Token boundary and engine semantics

The server tokenizer encodes both `context` and `context + continuation`.  The
context IDs must be an exact prefix of the combined IDs, and the remaining
continuation ID sequence must be non-empty.  Otherwise the request fails with
an unaligned-boundary error.  Client-side independent tokenization, character
offset guessing, raw vocabulary strings, and decoded-text equality are never
used to infer the boundary; BPE can merge across the join.

Native generation receives the verified context IDs and the ordered forced
continuation IDs.  At output position `j`, the sampler:

1. computes `log_softmax` over the raw model logits, before grammar masks,
   penalties, temperature, top-k, top-p, or min-p;
2. reads the value for the exact forced token `c_j`;
3. commits `c_j` as the next model input so position `j + 1` is genuinely
   conditioned on the candidate prefix.

Forced requests leave the greedy/speculative fast paths and work in both CPU
and device sampling paths.  Token IDs and positions are range-checked.  EOS is
teacher-forced as data rather than terminating the request, and the server
requires the returned ID sequence to equal the requested continuation exactly.
The backend must also attest that it processed the exact requested context IDs;
missing, untyped, or mismatched prompt evidence fails the response closed.
Forced continuations cannot be combined with native structured-output grammar,
because a grammar mask or grammar termination would contradict exact token and
length ownership; that conflict is rejected when sampling parameters are built.
The existing in-process and process-isolated native layouts carry the forced
IDs through the same sampling wire.  OpenAI-compatible and direct-vLLM
adapters reject this native-only intent rather than silently dropping it.

## 4. Installed benchmark contract

`LogLikelihoodRequestSpec` is separate from `ChatRequestSpec`.  A sibling
`LogLikelihoodAdapter` reuses the common dataset, cache, precondition,
selection, methodology, and result aggregation machinery, but has no generated
text or empty-completion semantics.  It sends one completion request per
ordered candidate, with the normal target authentication and retry policy.

Every successful candidate must have:

- exactly one response choice in `loglikelihood` mode;
- choice index zero, `finish_reason: "length"`, and continuation text exactly
  equal to the requested candidate;
- non-empty, aligned token IDs and selected-token arrays;
- finite natural-log values no greater than zero;
- the same non-empty `prompt_token_ids` sequence across every ordered candidate.

Missing arrays, non-finite/positive values, reordered evidence, an HTTP error,
or malformed HTTP 200 data makes the item failed and unmeasured.  It is never
converted to `-inf`, zero probability, or a benchmark score of zero.  Candidate
scores use an explicitly fingerprinted reduction: raw sum by default, or
`mean_token` when an adapter declares it.  Ordered stable argmax resolves an
exact tie; the tie set remains in item evidence.

MMLU requests the ordered candidates `" A"`, `" B"`, `" C"`, and `" D"` and
requires each to be exactly one target token and the four token IDs to be
distinct.  It retains upstream choice order and item-micro aggregation.
Per-item details retain the prompt token IDs and every candidate's text, token
IDs, per-token logprobs, sum/reduced score, selected/gold labels, tie set, and
winner margin so the score can be audited without recontacting the model.  The
shared protocol/parser implementation and MMLU adapter source are both
content-bound into the run fingerprint.

The first runnable item is a serialized capability probe before dataset
fan-out.  An explicit 400/422 extension rejection, or an HTTP 200 response that
does not identify `mode: "loglikelihood"`, skips the pair as unsupported.
Authentication/authorization failures, any 404, or a response that claims the
mode but returns malformed scoring evidence fail the pair and prevent the
remaining items from being dispatched.  Exhausted transport, 429, and 5xx
retries do the same.  MMLU additionally treats a probe rejection of the fixed
`Answer:` / candidate token boundary, or failure of its one-token,
four-distinct-ID candidate structure, as a target-tokenizer incompatibility.
It does not dispatch the remaining items or spend another 56,164 calls proving
the same fixed boundary or candidate structure repeatedly.

## 5. Verification contract

The portable gates cover:

1. forced CPU and device selection against a direct `log_softmax` oracle,
   including multi-token conditioning, range failures, and unchanged ordinary
   sampling;
2. in-process and process-wire propagation of forced IDs;
3. completion-extension validation, unsupported backends, tokenizer-boundary
   rejection, exact returned IDs, and ordinary-completion regression;
4. client authentication/retry behavior and fail-closed malformed, missing,
   non-finite, positive, and reordered evidence;
5. sum versus mean-token reductions, stable ties, item/pair failure semantics,
   and source-identity invalidation;
6. MMLU ordered candidates, one-token requirement, known ranking, methodology,
   and an offline fixture end-to-end run through an exact synthetic endpoint.

No GPU performance threshold is introduced.  The scoring mode is intentionally
slower than free generation because exact candidate evidence is the acceptance
criterion; later batching can optimize it without changing the wire or score.
