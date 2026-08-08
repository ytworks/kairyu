# M8 Design: Engine CPU Core — Real Tokens, Real Sampling, Multi-Token Commit

Status: **Implemented** (2026-07-03; D1/D2/D6 amended 2026-08-08). Reviewed — APPROVE-WITH-AMENDMENTS
(3-reviewer agent panel, 2026-07-03; all amendments applied inline, see §6).
All six phases (D1–D6) landed with tests: 328 → 437 tests, 95% coverage.
The original M8 deliverables remain implemented and CPU-tested. Issue #333's
later D6 diagnostic is an explicitly separate real-TP4 hardware measurement.
Milestone: M8 (implementation milestone; realizes roadmap Track E1/E2 CPU halves —
the M8–M19 numbering continues docs/design/m1..m7 and maps to roadmap tracks:
M8/M9→E1-E2/P-A, M10→F1-F2, M11→P-B/P-C/F5, M12–M18→E-track local halves,
M19→deploy packaging. Recorded in PROGRESS.md.)
Date: 2026-07-03
Depends on: M2 engine (scheduler, radix KV, EngineCore, overlap); consumed by M9
(usage/tokenizer seams), M17/G4 (multi-token machinery for EAGLE/MTP), M18 (wire
schema).

## 1. Goal

Replace the last placeholder layers of the CPU engine with real implementations
behind the existing seams, so the GPU phase swaps kernels only:

1. Real tokenization/detokenization (HF `tokenizers`) with incremental streaming.
2. Real sampling — temperature/top-k/top-p/min-p/penalties/seed/logprobs — and
   xgrammar structured-output enforcement in the sampling path.
3. Scheduler multi-token commit (prerequisite for all speculative decoding).
4. N-gram speculative decoding wired end-to-end, greedy-equivalence pinned.
5. Quant/profile groundwork: NVFP4/modelopt/INT8 detection, `HardwareProfile`,
   safetensors reader.
6. API-server ↔ engine-core process split over ZMQ/msgpack.

## 2. Key design decisions and rationale

### D1 — Tokenizer seam: protocol + incremental detokenizer; toy stays the default

New `kairyu/engine/tokenizer.py`:

- `Tokenizer` protocol: `encode(text) -> tuple[int, ...]`, `decode(ids) -> str`,
  `vocab() -> list[str]`, `eos_token_id: int | None`.
- `IncrementalDetokenizer`: per-request; emits only text that can no longer change
  (holds back incomplete UTF-8 sequences / partial merges). Invariants pinned:
  every cumulative stable update is a prefix of `finalize()`, and finalization
  may append a held terminal suffix but can never rewrite text already exposed
  to a streaming client.
- `ToyTokenizer` (today's word-hash + `tok<N>`) **remains the default**;
  `HFTokenizer` wraps the `tokenizers` library (deferred import, `structured.py`
  pattern), loads `tokenizer.json`, exposes real `eos_token_id`.

**Incremental-performance amendment (2026-07-27, issue #211):**

- `TokenDecodeStream` is an optional per-request capability. `HFTokenizer` uses
  the Rust `tokenizers.decoders.DecodeStream` (including its incomplete-UTF-8
  buffer and special-token policy), and `ToyTokenizer` joins only the arriving
  token delta. Push work is O(delta).
- Production `HFTokenizer` requires `tokenizers>=0.21.1` and its
  `DecodeStream`; a mismatched installation fails fast instead of silently
  selecting O(length²) full-prefix work. Custom tokenizers without a stream,
  and subclasses overriding `decode()` without a matching stream, retain the
  exact full-prefix compatibility path. Capability is never inferred merely
  from inheritance.
- Operation-count coverage fixes 4,096 one-token pushes at N incremental work;
  finalization decodes only an un-emitted terminal window and never re-decodes
  published history. ByteLevel, WordPiece, Metaspace, special-token skipping,
  multi-token commits, and fallback parity are pinned. The earlier Qwen3-32B
  measurement, whose native number still included the former full final pass,
  measured 2.038 s full-prefix versus 0.0149 s native streaming (137.25×)
  across 4,096 random tokens while preserving final bytes.

**Terminal-consistency amendment (2026-08-05, issue #361):**

- Incremental output is authoritative once published. A native stream may
  expose an optional `finalize_suffix()` hook. The HF adapter excludes special
  IDs, retains only the un-emitted terminal window beginning at a
  replacement-bearing ID, and flushes it only when the reconstructed window and
  suffix end in U+FFFD. This preserves WordPiece/Metaspace boundary context,
  CTC separators, and incomplete ByteLevel/ByteFallback tails without
  duplicating ordinary CTC or other no-growth decoder output. Toy has no
  terminal delta. If Rust rejects a later token because malformed held bytes
  make its reconstructed window disagree with a character already published,
  the adapter appends only that held replacement suffix and retries the token
  in a fresh native stream. A legacy custom stream with only `push()` retains
  its historical final full decode only when that candidate extends stable
  output; an unrelated custom `finalize()` method is never invoked.
- The compatibility full-prefix path caches its latest decode during `push()`.
  Finalization does no additional decode: it accepts the cached candidate only
  when that candidate extends the already-published stable prefix, otherwise it
  preserves the stable prefix unchanged. Repeated finalization is idempotent.
- The engine terminal path independently enforces the same no-retraction rule
  before incremental stop matching. A decoder disagreement therefore cannot
  retract SSE text or turn into a stop-matcher request failure. Valid
  CJK/emoji/multibyte, special-token, and complete byte-fallback sequences retain
  full-decode parity whenever the full decode extends published text; malformed
  terminal or interior byte runs append only their held flush and continue.

**Per-request special-token policy amendment (2026-08-05, issue #362):**

- `SamplingParams.skip_special_tokens` is request-owned, defaults to `True`,
  and is captured by each `IncrementalDetokenizer`; it never mutates shared
  tokenizer state. HF full-prefix decoding and its per-request native
  `DecodeStream` receive the same value, so concurrent requests with opposite
  policies remain isolated. If the native stream capability is absent or
  unusable for the requested policy, the matching full-prefix path remains the
  correctness fallback.
- Existing custom tokenizers keep their historical `decode(ids)` and
  `new_decode_stream()` contracts. Flag support is detected once from the
  callable signature, including `**kwargs`; an operational `TypeError` is not
  reinterpreted as an old signature. If a custom `decode` accepts the flag but
  its stream factory does not, a `False` request uses the flag-aware full-prefix
  decoder instead of silently retaining that stream's default policy. A custom
  tokenizer with neither capability keeps its prior authoritative semantics.
- `False` exposes registered special-token text when that token is otherwise
  visible, including ignored EOS, an EOS/stop ID masked below `min_tokens`, and
  length termination; `True` retains the established skip behavior. The #352
  terminal rule has precedence: an ID that actually causes EOS or stop-token
  termination is never passed to visible detokenization under either policy,
  while its token ID, usage, logprobs, scheduler/KV state, and radix history are
  still retained.

**Config surface (amended)**: `KairyuBackend(tokenizer: str | Tokenizer = "toy")` —
`"toy"` → ToyTokenizer; any other string is a filesystem path (a `tokenizer.json`
file or a directory containing one) → HFTokenizer; a `Tokenizer` instance is
accepted programmatically. **Validation is fail-fast at construction** (bad path →
`ValueError` at `kairyu serve` startup — `build_app_from_spec` constructs engines
eagerly). YAML: `options: { tokenizer: /models/llama-3.1-8b }` (BackendSpec.options
already forwards as kwargs; verified builder.py → registry.py).

`_submit` sets `eos_token_id`, `stop_token_ids`, `min_tokens`, `ignore_eos` from
`SamplingParams`.

**Pretokenized-input amendment (2026-07-30, issue #227):**
`EngineLoop.resolve_prompt_token_ids` is the only public-to-core normalization
point. Text still calls the backend tokenizer. `TokensPrompt` bypasses
`encode()` exactly, validates every ID against the selected backend tokenizer
vocabulary before scheduler mutation, and forwards the immutable tuple to the
unchanged `EngineRequest`/Scheduler/RadixKV core. IDs are exact usage truth.
Optional display text does not participate in execution, admission, or cache
identity. A multimodal value is rejected until a backend supplies a processor
and exact post-processing token count; Kairyu never estimates that count from
media bytes.

**Stop-string handling (amended — SSE-safe, radix-safe):**
- **Hold-back**: while the pending detokenized tail is a prefix of any stop string,
  the backend withholds up to `max(len(stop)) - 1` trailing characters from the
  stream queue — a stop string spanning two deltas must never leak its prefix to
  an SSE client (deltas cannot be retracted).
- **Incremental scan (2026-07-27, issue #217)**: each request remembers the
  previously searched stable-text length and rescans only the newly exposed
  suffix plus `max_stop_length - 1` characters of overlap. The first observed
  minimum absolute match is cached. Finalization feeds any last UTF-8 tail
  through the same overlap, preserving cross-token, overlapping-pattern, and
  earliest-match behavior without rescanning the cumulative prefix.
- Scan/truncate happens **inside the step loop, before `queue.put_nowait`**; the
  queue payload becomes a small frozen `_StreamUpdate(outputs, text, finish_reason,
  error)` so `finish_reason="stop" | "length"` flows explicitly to
  `CompletionOutput` (no more hardcoded "length").
- **Termination uses a new `Scheduler.finish_early(request_id)`** — truncate then
  commit-and-release through the normal `_finish` path — NOT `abort`:
  `_release_without_commit` would skip the radix commit and silently regress
  multi-turn prefix reuse (the E1 radix-hit gate).
- **Threading discipline (amended, load-bearing for D3)**: all scheduler mutations
  (add_request, abort, finish_early — including stop-string finishes) are queued
  and drained **on the step thread between `update()` and the next `schedule()`**,
  in both the in-process backend and the D6 service. The existing
  `asyncio.to_thread(_step)` pump plus loop-thread `add_request` already violates
  this in spirit; M8 fixes it (submit enqueues an op; the step loop drains ops).
- **Operation batching (2026-07-27, issue #218)**: a producer lock makes
  duplicate-ID reservation and queue mutation atomic. Consecutive adds share
  request/track arrays; lifecycle-duplicate aborts collapse to one ID. The step
  thread swaps out one frozen batch snapshot, uses Scheduler's atomic bulk-add
  contract where available, and restores untouched suffixes ahead of concurrent
  producer work on a partial failure. Purge filters batches under the same lock,
  and close seals the queue before reclaiming active requests. Reproduce the
  burst throughput measurements with
  `uv run python bench/op_queue_bench.py --operations 100000 --repeats 5`.
- **Presentation-lane amendment (2026-08-07, issue #327)**: production drivers
  split one core advance from detokenization/output completion. One serial
  `kairyu-output` future overlaps with at most one raw next step, including at
  pipeline depth one; there is no unbounded output queue. Stop-string feedback
  and traced requests form a barrier, so `finish_early` and all scheduler/track
  reclamation still occur on the serialized step owner before another schedule.
  Direct `EngineLoop.step()` remains the synchronous compatibility path.
  `KairyuBackend` publishes from the output lane. The D6 ROUTER thread keeps
  polling while a serial core executor advances; the output lane detokenizes,
  builds wire events, and msgpacks an immutable owner snapshot, while only the
  ROUTER thread sends. The process parent resolves every text/templated prompt
  to `TokensPrompt` off its event loop even when `max_model_len` is omitted.
- **Cumulative-state amendment (2026-08-07, issue #324)**: append-only scheduler
  outputs cross overlapping step boundaries through immutable-length views, so
  freezing a step is O(1) without exposing later appends. `StateSync` uses the
  owner-declared output epoch plus retained lengths and sends only new output
  and decode-page tails; reallocation, speculative overlay, epoch change, or
  retraction still forces one full snapshot. The output lane likewise retains
  mutable cumulative storage behind immutable-length `StreamUpdate` views and
  appends only new logprob/content entries. Public `CompletionOutput` values,
  legacy wire frames, and the first v2 snapshot materialize tuples/lists at
  their compatibility boundaries; steady v2 frames remain deltas.
- **Empty-output amendment (2026-08-07, issue #332)**: a tracked request with
  no newly committed token and no terminal transition produces no
  `StreamUpdate`; both production drivers skip empty presentation jobs. A v2
  process client still applies sequenced metadata-only deltas to its cursor but
  does not join cumulative text or construct a public result for them. CPU and
  CUDA sampler compatibility paths likewise make exactly one mutable fp32 copy:
  an existing CPU fp32 tensor is cloned, while a device or dtype conversion is
  itself the copy.
- **Delta-result amendment (2026-08-08, issue #338)**: public streaming
  completions carry `text_delta` plus its cumulative `text_offset`. Native,
  process-split, and OpenAI-compatible adapters preserve those deltas through
  the API and orchestration layers; legacy cumulative-only backends fall back
  to suffix slicing. Cumulative `text` remains compatible and is backed by an
  append-only lazy snapshot where flattening would otherwise repeat per token.
  Offset validation replaces repeated full-prefix scans and fails closed if a
  backend changes text that has already crossed the streaming boundary; an
  exact terminal completion re-presented by orchestration is idempotent.
- **Stream-backpressure conflation (2026-08-04, issue #335)**: the in-process
  `KairyuBackend` publishes cumulative `StreamUpdate` values through one bounded,
  single-consumer mailbox per request. The first token-bearing state is retained
  for TTFT, and later non-terminal states replace the pending latest state. A
  cumulative successful terminal can replace that latest state. The payload-free
  error sentinel cannot, so the latest cumulative state is delivered once before
  the error. Publishing the
  first FIFO terminal seals the mailbox even after a waiting `get()` removes it,
  preventing a later unrelated pump failure from turning an already successful
  stream into an error. The steady queue is at most two snapshots; only the
  pre-consumer `first-token + latest + error` shape reaches three. Consumers also
  drain consecutive non-terminal states before the next public yield, with the
  same rule independently per `n > 1` choice. Tests pin bounded 128-update
  backlog, first-token cadence, full text/token/logprob/usage preservation,
  success/error ordering, and sibling error propagation. A direct ASGI gate
  blocks the first body send while all 128 tokens finish, then requires one
  queued terminal snapshot, two content chunks, exact usage/finish/DONE, and a
  constant body-send bound. This does not drop D6 ZMQ v2 wire events: those are
  sequenced deltas rather than cumulative mailbox snapshots and must all reach
  the client accumulator.
- **Terminal stop-token visibility (2026-08-04, issue #352)**: a token that
  actually causes EOS/stop-token termination remains in output token IDs,
  usage, raw-logprob metadata, scheduler/KV history, and the radix commit, but
  is never fed into visible detokenization. An equal token sampled before the
  minimum, an ignored EOS, or a length terminal is not suppressed by this rule
  and retains the tokenizer's ordinary semantics. The decision is made before
  an incremental decoder sees the
  terminal token, so native streams never emit text that must be retracted.

Tests: tiny BPE built programmatically (no committed blobs); Japanese multi-byte
boundaries; WordPiece and Metaspace decoder parity; linear operation count;
custom-tokenizer fallback; EOS/stop end-to-end; randomized incremental/full
stop-match equivalence; long-output/many-stop bounded work; concurrent
add/abort producer stress and partial-failure recovery; stop-string-across-deltas
holdback pinned.
Deps: `tokenizers>=0.21.1` (dev group +
`[project.optional-dependencies] hf` extra).

### D2 — Sampler: `SampledToken`, tuple-valued runner output, grammar-mask-first

New `kairyu/engine/core/sampler.py`, pure torch functions (device-agnostic):

- `EngineSampling` frozen dataclass (engine-side subset): temperature, top_k,
  top_p, min_p, presence/frequency/repetition penalties, seed, logprobs (top-k
  count), json_schema. Default = greedy.
- `EngineRequest` gains **keyword-only** fields appended after `eos_token_id`
  (kw_only — verified no positional construction beyond the first two args
  anywhere): `sampling: EngineSampling = EngineSampling()`,
  `stop_token_ids: tuple[int, ...] = ()`, `min_tokens: int = 0`,
  `ignore_eos: bool = False`, `priority: int = 0` (admission ordering lands in
  M11; field lands now to avoid a second frozen-dataclass ripple).
- `SampledToken` frozen dataclass: `token_id`, `logprob: float | None`,
  `top_logprobs: tuple[tuple[int, float], ...] | None`.
- **`ModelRunner.execute` returns `StepOutput = dict[str, tuple[SampledToken, ...]]`**
  (alias defined in `engine_core.py` next to a shared `token_ids(step_output)`
  helper — five consumer sites convert through the one helper, not hand-rolled).
  One-commit ripple (amended, full census): `engine_core.py`, `overlap.py`,
  `pipeline.py` (`StageWorker`), `pd.py`, `tp_runner.py`, `torch_runner.py`,
  `kairyu_backend.py`, **`bench/parity_tp.py`, `bench/pd_mixed.py`**, and the
  inline stub runners in tests. `pd.py`'s int-typed public seams
  (`KVHandoff.transfer(first_token: int)`, `resume_with_kv(first_token: int)`)
  keep their int signatures; the coordinator unwraps `[0].token_id` explicitly.
  **`tp_runner` rank agreement compares token_ids only** (not logprob floats —
  the m5 D1 invariant is about tokens; float equality would be brittle on GPU).
  **`Scheduler.update` validates every committed token with
  `isinstance(token, int)` and raises** — a `tuple[SampledToken, ...]` is a
  `Sequence` and would otherwise be silently iterated into `outputs`, and an
  unconverted token would silently defeat the EOS comparison.

**Sampling order (amended to the defensible convention):**

1. raw logits → capture `log_softmax` **for logprob reporting** (vLLM v1's
   default is `raw_logprobs`; the OpenAI convention is temperature-independent —
   the previous "post-penalty pre-mask" draft matched no convention;
   `processed_logprobs` is a future opt-in).
2. **xgrammar `mask_logits()` FIRST** (if enforcer) — matching vLLM, which masks
   raw logits before the sampler. Mask-last can leave zero grammar-legal tokens
   after top-k/top-p, triggering the degenerate argmax fallback and distorting
   the nucleus. Mask-first + `min_tokens_to_keep=1` semantics on top-p/min-p
   guarantees non-empty support; penalties cannot resurrect `-inf`.
3. While the logical output position is below `min_tokens`, mask model EOS and
   all request/model stop-token IDs to `-inf`.
4. Penalties — **repetition over prompt + committed outputs; presence/frequency
   over committed outputs only** (matches both vLLM and HF defaults; pinned to
   honor the vLLM-signature promise in `sampling_params.py`).
5. `temperature == 0` → argmax **on the masked logits**, done; else scale.
6. **min_p, then top-k, then top-p** (vLLM v1 order; HF differs — divergence
   recorded here deliberately, vLLM compat wins).
7. softmax → stateless seeded Gumbel-max sample.

**Minimum-token amendment (2026-08-04, issue #352):** CPU, CUDA, batched
greedy, overlap, P-D, TP/EP, and speculative target sampling all apply the same
stop mask for every logical position `p < min_tokens`; `p == min_tokens` is the
first eligible stop position. The scheduled logical position, not a potentially
lagging committed-history length, is authoritative. `ignore_eos` controls only
termination: EOS is still masked below the minimum, becomes eligible at/after
it, and then does not stop when ignored. Raw-logit logprob reporting is captured
before the processor and remains unchanged. `SamplingParams` rejects an
explicit `min_tokens > max_tokens`, matching the compatible request contract.

**Determinism (amended)**: per-request base seed = `sampling.seed` or
**sha256(request_id) → 63-bit int** (never Python `hash()` — randomized per
process, and D6 splits processes); the splitmix64-style mix of
(base_seed, position) retains its complete unsigned 64-bit output — plain
addition collides across adjacent user seeds. Scope of the claim: the sampler
*preserves* TP rank
agreement given bitwise-identical logits per rank (a collectives/runner
property); it cannot repair divergent logits. CPU, structured-output, and CUDA
paths use one stateless Gumbel-max algorithm:
each uniform is a pure function of `(base_seed, output_position, vocab_index)`.
The vocabulary index is first permuted by an odd 64-bit counter stride, XORed
with every per-position seed bit, and avalanched by a SplitMix64 finalizer.
PyTorch's signed-int64 wraparound plus explicitly masked logical shifts preserve
the unsigned word on CPU and CUDA. A 52-bit mantissa is then mapped by its
float64 midpoint to the strictly open interval `(0, 1)`, giving the flipped
Gumbel winning tail a minimum uniform of `2^-53` instead of `2^-25`.
Changing only the sampling execution path therefore does not change the random
stream, and representative fixed-logit CPU/CUDA parity is a binding regression
gate. Ordinary fp32 filtering and the float64 Gumbel transcendental operations
may differ by a few ulps across devices, so candidates at a min-p/top-p/top-k
support boundary or a near-tied final Gumbel score are not claimed to have a
cross-device token guarantee.

**GPU amendment (2026-07-27, issue #206; amended 2026-08-05, issues #353 and
#354):** grammar-free CUDA sampling keeps the reviewed processing order. CPU and
structured-output sampling use the same canonical tensor draw instead of
`torch.multinomial`. Issue #353 intentionally changed their prior seeded
stochastic sequence while retaining CUDA's then-current sequence; issue #354
then intentionally replaces that common CPU/CUDA stochastic sequence with the
full-width keyed counter and 52-bit open uniform described above. Same-version
replay, TP rank execution, batch reordering, and CPU reproduction need no
host-owned generator offset. Greedy sampling (and therefore spec ≡ greedy) is
unchanged. Device penalties include committed host history plus uncommitted
device scalars. Logprobs remain raw and temperature-independent. The CPU
implementation skips degenerate-fallback construction on positive finite
support and retains at most four immutable vocabulary-offset tensors up to
1,048,576 entries; concurrent cold misses may construct duplicate temporary
tensors. CUDA retains branchless device execution and returns the selected token
without a host-visible scalar read.

**Batched CUDA amendment (2026-08-07, issue #326):** a decode batch now groups
every grammar-free CUDA row even when a grammar-constrained row shares the
step. Row-specific temperature/min-p/top-k/top-p and stateless seed/position
inputs feed one batched softmax and one batched Gumbel draw. Bounded top-k +
top-p uses the single maximum top-k prefix for cumulative thresholding instead
of sorting each complete vocabulary row; top-p without a finite top-k retains
the full-vocabulary sort because truncating that support would change the
request contract. At an exact tie across the bounded top-k boundary, the
combined top-k + top-p path intentionally defines its nucleus over exactly the
library-selected top-k prefix; the former threshold path included every tied
token and could exceed k. This changes the draw only for that exact boundary
case, bounds the optimized work, and is shared by batched, scalar, and
structured sampling so execution-path determinism remains intact. Top-k
without top-p retains its previous tie-inclusive threshold behavior.
Minimum-token masks and incremental penalty state remain row-specific, while
only grammar rows retain the CPU matcher path. Small host parameter vectors
use pinned non-blocking copies, and each sampler invocation produces one
mutable fp32 working copy. Scaled rows are automatically chunked at 2^22
vocabulary elements to bound temporary float64 Gumbel storage at large batch
sizes. Row seed/position identity preserves batch-order invariance.

**Incremental penalty-state amendment (2026-07-27, issue #216):** a
penalty-active request lazily allocates one dense row for its logits
device/vocabulary: prompt membership, repetition membership, output counts,
and output membership. It retains its own append-only committed shadow, while
the scheduler supplies an output epoch that changes only when an existing
prefix is replaced. The normal path therefore neither copies nor compares the
retained history: it updates only newly committed tokens and the bounded
overlap-ahead suffix. A shorter history, changed epoch, or replaced pending
suffix rebuilds or subtracts only on that exceptional path, so rejected or
restarted positions never leak into later penalties. The exact device scalar
already counted as pending becomes the scheduler's committed result without a
second state mutation; pending device tokens otherwise reconcile by tensor
identity, with no scalar D2H comparison. CPU active IDs use growing buffers
plus position maps, with a one-time initial sort and O(1)-amortized normal
insert/remove instead of rebuilding a sorted tensor every token.
`Sampler.hand_over()` moves the request state, and the first sample on a
different P-D device rebuilds the row from authoritative prompt/output history
while releasing the old device row.

This responsibility boundary is Kairyu-native: request-owned persistent state
fits its overlap scheduler, sampler lifecycle, and P-D handoff. It is not copied
from current vLLM. vLLM main at
`5f89a03dcb52702a62644e15b93f766765d06b28` converts full per-request output
lists to a padded CPU tensor and transfers it to the logits device on each
penalty application; its source labels that implementation inefficient and
planned for rework. We therefore compared both the legacy rebuild and an
independently optimized committed-count plus transient-pending alternative.

Selection uses the complete normal step, not steady apply alone. At Qwen's
151,936-word vocabulary, 32,768 output tokens, and pending depth 2,
committed-plus-effective-pending counts measured 513.9 µs on CPU versus 672.8
µs for the optimized alternative and 9,265.6 µs for the former full-history
rebuild (1.31× and 18.03× faster). CUDA measured 190.7 µs versus 334.7 µs and
6,879.2 µs (1.76× and 36.07× faster). The comparison gives the alternative the
same single-token CPU direct update, reuses the prior pending CUDA scalar, and
applies total counts in one operation for exact floating-point order; it does
not charge avoidable tensor construction or rounding changes to that design.
The benchmark commits one token and shifts the pending window on every measured
iteration, uses the production mutable-list/epoch contract, and checks every
transition plus a duplicate-pending adversarial case bitwise against the legacy
oracle outside the timer. Repetition-only short history (32 tokens) also wins:
138.7 µs versus 176.0 µs alternative and 501.1 µs legacy. Steady apply retains
the win on both devices (1.10× CPU and 2.71× CUDA versus the alternative). The
dense tensors cost 7 bytes per vocabulary entry (1,063,552 bytes at 151,936);
CPU additionally owns sparse active-ID containers. Reproduce with
`uv run python bench/sampler_penalty_state_bench.py --device cpu` and the same
command with `--device cuda:0`.

XGrammar is intentionally excluded from the device path because its matcher is
a stateful CPU FSM. Structured requests retain the mask-first/accept-once path
below; this preserves grammar correctness at the cost of the documented host
compatibility boundary rather than advancing a stale mask.

**Tokenizer-metadata amendment (2026-07-28, issue #208):** grammar compilation
receives a serializable `GrammarVocabulary`, not an untyped token-string list.
It preserves xgrammar's RAW / BYTE_FALLBACK / BYTE_LEVEL interpretation,
`add_prefix_space`, the tokenizer's encoded vocabulary, and the model lm-head
width. The last value is deliberately distinct from tokenizer vocabulary
length (Qwen3-32B has 151,669 tokenizer ids but 151,936 logits); padded model
ids are represented through `TokenizerInfo.vocab_size`, not fabricated empty
tokens. The same object is passed to every spawned TP rank and both P-D
samplers.

This follows the correctness boundary used by vLLM's xgrammar backend
(`TokenizerInfo.from_huggingface(..., vocab_size=model_vocab_size)`) while
retaining Kairyu's lighter `tokenizers.Tokenizer` dependency and sampler
lifecycle. A real Qwen byte-level token such as `Ġ` must decode as a space:
treating it as RAW can produce a grammar state with zero legal tokens, turn an
all-`-inf` argmax into token 0, and fatally reject that token under the mask.
The minimal byte-level regression and Qwen3-32B TP8 `n=1` / `n=2` schema gates
pin the corrected behavior and post-request runner health.

**Grammar state (amended)**: `accept()` runs **exactly once per committed token,
backend/driver-side after rank agreement** — never inside the per-rank sample
path (the CPU TP path runs the sampler N times; `GrammarMatcher` is stateful and
would advance N× per token). After each commit the backend checks
`is_terminated()` → finish with `finish_reason="stop"` via the same
`finish_early` mechanism as stop strings. `accept()` returning False under
mask-first is an invariant violation → raise into the engine error path (the
pump already propagates). `response_format` mapping (P-A gate):
`{"type":"json_object"}` → builtin JSON grammar; `{"type":"json_schema",
"json_schema":{"schema":{...}}}` → `EngineSampling.json_schema`; enforcer built
per-request in `_submit` from the tokenizer's `GrammarVocabulary`.

Logprobs land in `CompletionOutput.logprobs`/`cumulative_logprob`, filled by the
backend from accumulated `SampledToken`s.

### D3 — Scheduler multi-token commit: capped reservation, degrade-not-stall

`Scheduler.update(sampled: Mapping[str, int | Sequence[int]])` — bare int kept as
sugar (all existing call sites valid); lists commit in order with per-token
EOS/stop_token_ids/max_new_tokens checks; tokens after a terminal are discarded.

Speculative reservation — all four review blockers folded in:

- `Scheduler(speculative_tokens: int = 0)` (k). A spec decode chunk is emitted
  **only when `state.in_flight == 0`** — enforced in the scheduler itself, not
  just the backend, so any composition (tests, PD, pipeline, service) is safe;
  otherwise that request gets a plain 1-token chunk. Under `pipeline_depth ≥ 2`
  this means spec chunks simply never fire (positions planned ahead assume full
  commit — the device-side "future token" patch that makes spec × deep overlap
  sound is a GPU-phase mechanism). `KairyuBackend` additionally rejects the
  combination at construction as the user-facing error.
- **Reservation is `min(k + 1, max_new_tokens - len(outputs))`, carried in
  `chunk.num_tokens`**; the runner must not write draft KV beyond it (the KV
  hazard is runner-side, before update() can check). `SpeculativeRunner`
  truncates its draft to `num_tokens - 1`.
- **Capacity degrade, never stall**: if `_ensure_decode_capacity` cannot reserve
  `num_tokens` slots (after the existing preemption attempt), the chunk degrades
  to a plain 1-token reservation — baseline progress is always preserved (k+1
  must not introduce stalls that k=0 doesn't have).
- **Budget**: a spec chunk consumes `num_tokens` from the decode/shared token
  budget (not 1), preserving `pd_separation`'s TPOT knob. Documented:
  `decode_watermark_pages` was sized for +1 growth and should scale with k.
- **Shortfall rule (amended)**: spec mode guarantees exactly one outstanding
  chunk per request; after committing a non-terminal list, `in_flight` is set
  to 0 with an assertion that it equaled the chunk's reservation on entry.
  Non-spec paths keep today's per-token decrement. On a terminal token mid-list
  with k > 0, **both `in_flight` and the would-be surplus are zeroed** (not
  transferred): rejected/beyond-terminal spec slots will never arrive, and a
  stale surplus would mask double-commit bugs behind the silent-trim path.
  (`surplus_in_flight` keeps its existing meaning — overlap late arrivals that
  WILL come — untouched for non-spec flows.)
- KV pages reserved for rejected positions stay with the request; verified
  against `commit_and_release`: excess decode pages land in `leftover` and are
  pool-freed, garbage slots beyond `prompt+outputs` are never folded into the
  radix tree, and the candidate ordering matches the runner's slot→page map.
- PD × spec is **unsupported in M8** (prefill cores are structurally safe —
  a >1-token return for a prefill-completing chunk fails loudly; a spec decode
  core with `resume_with_kv` adoption is untested and out of scope).

### D4 — `SpeculativeRunner`: overlay-state scoring, verify, return the list

New `kairyu/engine/core/spec_runner.py`, a `ModelRunner` wrapper:

- On a spec decode chunk for R: `draft = propose_ngram(prompt + outputs)`
  truncated to `chunk.num_tokens - 1`; empty draft → the chunk degenerates to a
  normal 1-token decode (shortfall accounting covers it).
- **Scoring mechanism (amended)**: the wrapped runner's decode path reads
  `state.outputs[p-1]` from scheduler state, and draft tokens are not in
  `outputs` — so the wrapper passes an **immutable overlay state view** whose
  `outputs` = committed outputs + draft prefix accepted so far, one scored
  position at a time (walked example verified against `torch_runner.py`:
  `target_tokens[0]` = the normal next-token sample given the committed prefix;
  `target_tokens[i]` = sample after writing KV of draft token i-1 — length
  `len(draft)+1`, each position conditioned on the DRAFT prefix, exactly
  `verify_greedy`'s contract).
- Rejected-slot correctness is load-bearing on three named conditions:
  (a) `in_flight == 0` at spec schedule time (D3, scheduler-enforced);
  (b) `seq_len` derived from committed outputs only; (c) stale slots are never
  radix-folded (`commit_and_release` keys pages by prompt+outputs; readers
  recompute beyond `num_cached_tokens`). The next step overwrites the first
  stale slot before any read.
- **Per-request gating (amended by #358)**: deterministic drafts support T=0
  greedy verification plus T>0 point-mass rejection verification, including
  filtered and penalized target distributions. Grammar and forced-token
  requests keep the one-token bypass because they require rollback beyond the
  current draft contract.
- Wired via `KairyuBackend(speculative="ngram", speculative_tokens=k)`; default
  off. Acceptance-length counters exposed (G4 M-A4 lineage).

Invariant pinned: spec ≡ non-spec greedy through the full engine, on repetitive
prompts (accepts > 0) and adversarial ones (accepts 0).

**2026-08-08 amendment (#358):** every serving draft source remains
deterministic (n-gram lookup or learned-head argmax), so its proposal
distribution is a point mass q(t)=1. For `temperature > 0`, one ordinary draw
X from the processed target distribution implements standard rejection
sampling exactly: X=t accepts the draft with probability p(t); conditioned on
X!=t, X is already distributed as the normalized residual `max(0, p-q)`.
`SpeculativeRunner` therefore reuses target samples as acceptance/correction
records without transferring vocabulary-sized distributions. Penalties are
supported by slicing each verification row's history at its logical position.
Grammar and forced-token requests retain the one-token bypass because matcher
or forced-continuation rollback is a separate contract.

### D5 — Quant detection, hardware profile, safetensors reader

- `quant_config.py`: add `NVFP4`/`INT8`; parse `quant_method: "modelopt"`
  (`quant_algo: NVFP4|FP8`) and compressed-tensors INT8 W8A8. Real-world
  config.json snippets as fixtures.
- New `kairyu/engine/core/hw_profile.py`: frozen `HardwareProfile` (arch/SM,
  memory, measured bandwidth, P2P matrix, formats, kernel tier); `probe()`
  returns a `cpu` profile without CUDA (the thin `torch.cuda` branch is
  acknowledged uncovered on CPU CI; decision logic lives in pure tested
  functions); `best_format(quant_config)` decision table (roadmap §2); writer
  for the `bench/results/env-<date>.json` schema.
- New `kairyu/engine/core/weights.py`: safetensors index/shard reader with
  `get_slice` hook (M16 per-rank loads). Tested against tiny generated
  checkpoints. Dep: `safetensors`.

### D6 — Process split: ZMQ ROUTER/DEALER, msgpack; engine owns token truth

New `kairyu/engine/core/engine_service.py` (child main) +
`kairyu/engine/zmq_backend.py` (`EngineBackend`, name `"kairyu-proc"` —
**registered via a `_LAZY_MODULES` entry in `registry.py`**, the only wiring that
makes it reachable from YAML).

- Service: the ROUTER thread drains ops and sends frames, one serial step
  executor owns schedule/execute/scheduler mutation, and the D1 output lane
  owns detokenization/event construction/msgpack. The bounded one-ahead
  protocol preserves the D1 discipline while the ROUTER can answer heartbeats
  during engine/output work. The configured death-detection timeout still must
  exceed worst-case step time (documented knob).
- Events: msgpack wire v2 is negotiated per `add` without a global handshake.
  The client sends `wire_version=2` plus a fresh `stream_id`; an old service
  ignores those fields and returns the legacy cumulative event, while a new
  service defaults a missing version to legacy for old clients. A v2 request
  receives one cumulative `snapshot` at sequence 0, then only sequenced deltas
  `{output_offset, new_token_ids, text_offset, text_delta, new_logprobs?,
  new_logprob_content?, finished, finish_reason, num_cached_tokens}`.
  **The snapshot also carries `num_prompt_tokens`** from the engine's validated
  token truth, so M9 usage does not depend on client estimates. Offsets and
  sequence are checked before reconstruction. Empty
  non-terminal updates are suppressed so unrelated scheduler steps do not
  inflate a request's wire volume. Normal visible text appends; terminal exact
  detokenization may replace one suffix using a single longest-common-prefix
  offset. Each public request is scheduled under a fresh internal wire request
  ID, and every result/error and abort is generation-bound by `stream_id`, so
  cancelling and immediately reusing a public request ID cannot cross streams
  even on the legacy response path. When the caller omits a sampling seed, the
  client sends the historical public-request-ID-derived seed explicitly; the
  internal ID therefore cannot change deterministic output.
- **Typed request envelope (2026-07-30, issue #227):** legacy string requests
  keep their original `prompt` field and response wire v1/v2 unchanged.
  Explicit text and token inputs add an independent
  `prompt_wire_version=1` tagged envelope decoded by one strict shared codec.
  Missing, unknown, or extra fields and invalid token IDs fail only that
  request. The service receives `TokensPrompt` and `EngineLoop` bypasses
  tokenization; it never stringifies IDs. Multimodal envelopes round-trip in
  codec tests but are rejected by the client preflight because the child has
  no media processor.
- **Parent preparation (2026-08-07, issue #327):** current clients resolve text
  and templated text to the tagged `TokensPrompt` envelope before send,
  independent of `max_model_len`; the child validates the exact IDs and never
  repeats text tokenization. Legacy raw strings remain accepted for rolling
  upgrades.
- Backend: `zmq.asyncio` DEALER with **lazy socket/receiver-task creation on
  first `_submit`** (amended — `build_app_from_spec` constructs backends before
  any event loop exists; same lazy pattern as today's `_pump_task`).
- Lifecycle: `multiprocessing.get_context("spawn")` with a top-level importable
  child entrypoint (spawn pickles it); ephemeral-port handshake via pipe.
  **`shutdown()` = shutdown op → `join(timeout)` → `terminate()` → `kill()`,
  plus an atexit guard** for non-lifespan construction. Coverage: the parity
  tests end via the clean shutdown op (a terminated child loses coverage data);
  `[tool.coverage.run] concurrency = ["multiprocessing"]` + `sigterm` is added
  (the section does not exist yet — created here); the test suite shares one
  service fixture (spawn re-imports kairyu per child).
- `KairyuBackend` keeps its bespoke inline step loop in M8 (refactoring it onto
  `EngineCore`/`OverlapEngineCore` — and per-step streaming out of
  `OverlapEngineCore`, the remaining m2 §5 item 3 entry — is **deferred to M12**
  where `PagedModelRunner` arrives; recorded so the runbook §3 list stays
  truthful). D3's spec constraint is scheduler-enforced, so this deferral is
  safe.

Deps: `pyzmq`, `msgpack` (dev group + `[fleet]` extra).

**Process-wire amendment (2026-07-29, issue #212):** the original D6 delta
schema above is now the production protocol rather than a documented intent.
`bench/proc_wire_bench.py` feeds cumulative production `StreamUpdate` values
through both msgpack encoders and retains every frame size. Its gate uses exact
serialized byte counts, not timing: protocol v2 must grow approximately 2x
when output length doubles, while the legacy cumulative control must expose
its approximately 4x growth. Long-generation process parity covers tokens,
text, usage, finish state, id-keyed logprobs, and rich logprob content. The
retained clean-source result is
`bench/results/proc-wire-delta-2026-07-29.json`: at 1,024 tokens it measures
31,012,271 legacy bytes versus 356,199 v2 bytes, with empirical growth
exponents 1.97–1.99 versus 1.01–1.02.

**Distributed process-isolation amendment (2026-08-04, issue #333):**
`kairyu-proc` now accepts real-model tensor parallelism instead of silently
remaining a TP1-only seam. The child returns its constructed TP degree only
after every rank is live; the API advertises no topology until that identity
matches the configured degree. TP2+ uses a non-daemon service in a private
POSIX session so the service can spawn workers. A one-way lifetime lease kills
that complete process group if the API parent disappears. On Linux the API is
a child subreaper and waits only adopted zombies from the private group, so
startup cancellation, rank-0 death, follower death, heartbeat timeout, normal
shutdown, and TERM/KILL escalation finish with every descendant reaped before
another generation may start. The service checks launcher failure/dead-rank
state on each ping; the parent sends pings between events and marks the node
fatally unready when the child reports failure or stays silent for the default
120-second worst-step allowance. Add, abort, heartbeat, and shutdown DEALER
writes are bounded so a wedged transport cannot block route release or process
cleanup. Because Linux subreaper status is process-wide, this backend assumes a
dedicated serving process; a generic embedding that also supervises unrelated
child trees needs an external supervisor boundary.

Issue #333's diagnostic is deliberately separate from the formal A6 verdict.
`bench/issue_333_proc_http_bench.py` replays the exact TP4 ShareGPT c128 trace
with four fresh servers in
`kairyu`, `kairyu-proc`, `kairyu-proc`, `kairyu` order, retaining raw strict-SSE
rows plus clean source, actual imported module, immutable image, full
checkpoint, GPU, config, `/backends`, and PID/PPID/PGID process-tree evidence.
Every cell begins and ends with selected-GPU compute-process absence, zero
utilization, and memory exactly restored to the stable per-GPU run-start idle
baseline. A completed server is stopped with a bounded graceful Docker stop,
its immutable launch identity and zero exit are re-attested before logs are
retained and it is removed without force; forced recovery invalidates the cell
even if it restores the hardware to idle.
The v2 evidence contract binds the existing strict A6 completion structure,
requires all 163 response IDs to be unique within each cell, and requires each
of the four serialized ShareGPT warm-up outputs to match across all four fresh
servers. Measurement-burst output equality is deliberately non-binding. The
first fail-closed trial was discarded after its same-arm repeats agreed on
only 29/128 and 41/128 output hashes, demonstrating that all-four equality was
not an arm-neutral integrity condition. Concurrent admission and batch
composition changing the greedy floating-point path is a plausible mechanism,
not a causal claim. The estimand ends at first-token arrival, whereas the old
hash condition covered the subsequent 128-token continuation and was itself
downstream of the backend treatment; conditioning the TTFT interpretation on
that continuation would be post-treatment selection. The raw rows retain
well-formed output and complete-stream digest metadata but not the decoded
bytes needed to recompute those digests;
all six measurement-cell agreement rates are therefore retained only as an
explicit diagnostic without serializing or otherwise changing the c128 load.
Before measurement, a paired-median process/in-process TTFT-p99 ratio at or
below 0.90 was declared a material report-only movement. This has no A6
acceptance threshold, and its causal scope is the net process split including
ZMQ/msgpack/delta/lifecycle overhead rather than pure GIL isolation.
The discarded v1 run remains an invalid observation rather than a result: its
paired-median TTFT-p99 ratio was 0.9454156693989547, but failed evidence means
no classification. Its raw and manifest SHA-256s are recorded in `PROGRESS.md`
against source commit `ad11a32`. Exactly one fresh full v2 ABBA run is allowed,
and it is the issue result regardless of its performance direction.
That sole v2 run is now complete on Qwen3-32B TP4 and 8× RTX PRO 6000. All 15
binding checks pass independent replay, all 512 measurement requests succeeded
without retry, and all four containers restored the selected GPUs to their
run-start idle baseline after graceful zero exit and non-forced removal. The
paired process/in-process TTFT-p99 ratios were 0.9189755344057482 and
0.9219867334510442, giving a 0.9204811339283963 median. Because this is above
the predeclared ≤0.90 material line, the report-only classification is
`no_material_reduction` and the dominant process/GIL-contention hypothesis is
`not_supported`. Median goodput and TTFT-p50 ratios were 1.086201492163829 and
0.9282808350389853. This remains a net-backend diagnostic rather than a pure
GIL attribution or formal A6 verdict. Complete evidence is retained under
`bench/results/issue-333-proc-http-qwen3-32b-rtxpro6000-2026-08-05/`.

## 3. What M8 does not include (explicit non-goals)

- `n > 1` parallel sampling in the kairyu backend (M9, rides D2's seams).
- Real model architectures / multi-layer KV pools (M12); `TinyAttentionLM`
  stays the oracle.
- The original M8 scope excluded EAGLE/MTP and sampled-mode verification;
  M17 added deterministic learned drafts and #358 added their T>0 point-mass
  rejection policy. Grammar-composed speculation remains excluded.
- The original M8 scope excluded TP multi-process SPMD and gave the in-process
  `TPModelRunner` only the D2 return-type ripple. M16 later delivered SPMD, and
  issue #333's D6 amendment above exposes that implementation through
  `kairyu-proc`.
- Beam search / `best_of` (fields stay accepted-and-ignored).
- Per-step streaming out of `OverlapEngineCore` / backend refactor onto the core
  classes (M12, see D6).

## 4. Phasing (each phase lands green: pytest + ruff, cov ≥ 80%)

1. D1 tokenizer seam + stop handling + step-thread op discipline.
2. D2 sampler + protocol ripple (largest commit; incl. bench/ and update()
   validation).
3. D3 scheduler multi-token commit (+ robustness tests).
4. D4 SpeculativeRunner (+ equivalence suite).
5. D5 quant/profile/weights (independent).
6. D6 process split (after D1/D2 — wire schema final).

## 5. Verification

- Full existing suite green at every phase (328 baseline).
- Pinned invariants: incremental detok ≡ full detok; temp=0 ≡ argmax ≡ today's
  outputs; same seed → same tokens (across the zmq process boundary too);
  stop-string holdback never leaks a partial stop across deltas; stop finishes
  commit to radix (hit-rate preserved next turn); spec ≡ non-spec greedy
  (accept>0 and accept=0); spec bypass for temp>0/penalties/schema; multi-token
  robustness (EOS mid-list, max_new_tokens cap → reservation cap, capacity
  degrade to 1-token, abort with reserved slots, budget consumption = num_tokens,
  preemption paths untouched); xgrammar 50-schema validity through the full
  engine incl. termination → finish_reason="stop"; update() rejects non-int
  tokens; zmq backend parity incl. abort + service-death error propagation.
- `bench/serving_bench.py` smoke against the zmq backend (manual, CPU).

## 6. Review record

3-reviewer agent panel, 2026-07-03 — all APPROVE-WITH-AMENDMENTS; amendments
applied inline above:

- **Scheduler/KV invariants**: shortfall rule re-specified as zero-the-sole-chunk
  with entry assertion (not arithmetic subtraction); capacity failure degrades to
  1-token chunk (k+1 must never stall where k=0 wouldn't); spec precondition
  `in_flight == 0` enforced in the scheduler, not only the backend; reservation
  capped to remaining tokens and bound via `chunk.num_tokens` (runner-side KV
  hazard); spec chunks consume `num_tokens` of budget; terminal-mid-list zeroes
  both counters; step-thread op discipline fixes a pre-existing add/abort race
  that D1's stop-abort would have widened; PD unwrap made explicit; PD × spec
  declared unsupported.
- **Sampling/spec correctness**: grammar mask moved to raw logits (mask-first,
  vLLM convention; mask-last can NaN-crash); logprobs from raw logits
  (`raw_logprobs` default — previous draft matched no convention); repetition
  penalty scope pinned to prompt+outputs; min_p before top-k/top-p (vLLM order,
  HF divergence recorded); `accept()` exactly once post-rank-agreement (per-rank
  accept corrupts the matcher N×); sha256-based seeds (Python hash() is
  process-randomized — fatal across the D6 boundary); overlay-state scoring
  mechanism specified with verified off-by-one walkthrough; per-request spec
  gating replaces unimplementable constructor enforcement; grammar termination
  and accept-failure paths specified.
- **Integration/back-compat**: ripple census extended to bench/ harnesses and
  pd.py's int-typed seams; `StepOutput` alias + shared `token_ids()` helper;
  loud `update()` int validation (silent-EOS-defeat hazard); rank agreement on
  token_ids only; tokenizer config surface pinned (str|Tokenizer, fail-fast);
  stop-string SSE holdback + `finish_early` commit path (abort would regress
  radix reuse); `_LAZY_MODULES` registration; `num_prompt_tokens` in the first
  event; lazy zmq socket creation; bounded shutdown escalation + coverage
  config for the spawned service; EngineRequest fields kw_only; milestone-ID
  mapping recorded; OverlapEngineCore streaming explicitly deferred to M12;
  `response_format` mapping specified.
