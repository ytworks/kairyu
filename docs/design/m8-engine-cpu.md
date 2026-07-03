# M8 Design: Engine CPU Core — Real Tokens, Real Sampling, Multi-Token Commit

Status: **Implemented** (2026-07-03). Reviewed — APPROVE-WITH-AMENDMENTS
(3-reviewer agent panel, 2026-07-03; all amendments applied inline, see §6).
All six phases (D1–D6) landed with tests: 328 → 437 tests, 95% coverage.
Local-complete mandate: everything here is implemented and tested on CPU; no
deliverable waits for hardware.
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
  (holds back incomplete UTF-8 sequences / partial merges). Invariant pinned:
  concatenated incremental output == full `decode()` at every step.
- `ToyTokenizer` (today's word-hash + `tok<N>`) **remains the default**;
  `HFTokenizer` wraps the `tokenizers` library (deferred import, `structured.py`
  pattern), loads `tokenizer.json`, exposes real `eos_token_id`.

**Config surface (amended)**: `KairyuBackend(tokenizer: str | Tokenizer = "toy")` —
`"toy"` → ToyTokenizer; any other string is a filesystem path (a `tokenizer.json`
file or a directory containing one) → HFTokenizer; a `Tokenizer` instance is
accepted programmatically. **Validation is fail-fast at construction** (bad path →
`ValueError` at `kairyu serve` startup — `build_app_from_spec` constructs engines
eagerly). YAML: `options: { tokenizer: /models/llama-3.1-8b }` (BackendSpec.options
already forwards as kwargs; verified builder.py → registry.py).

`_submit` sets `eos_token_id`, `stop_token_ids`, `min_tokens`, `ignore_eos` from
`SamplingParams`.

**Stop-string handling (amended — SSE-safe, radix-safe):**
- **Hold-back**: while the pending detokenized tail is a prefix of any stop string,
  the backend withholds up to `max(len(stop)) - 1` trailing characters from the
  stream queue — a stop string spanning two deltas must never leak its prefix to
  an SSE client (deltas cannot be retracted).
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

Tests: tiny BPE built programmatically (no committed blobs); Japanese multi-byte
boundaries; EOS/stop end-to-end; stop-string-across-deltas holdback pinned.
Deps: `tokenizers` (dev group + new `[project.optional-dependencies] hf` extra —
the section is created in this milestone).

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
   after top-k/top-p (NaN → multinomial crash) and distorts the nucleus.
   Mask-first + `min_tokens_to_keep=1` semantics on top-p/min-p guarantees
   non-empty support; penalties cannot resurrect `-inf`.
3. Penalties — **repetition over prompt + committed outputs; presence/frequency
   over committed outputs only** (matches both vLLM and HF defaults; pinned to
   honor the vLLM-signature promise in `sampling_params.py`).
4. `temperature == 0` → argmax **on the masked logits**, done; else scale.
5. **min_p, then top-k, then top-p** (vLLM v1 order; HF differs — divergence
   recorded here deliberately, vLLM compat wins).
6. softmax → seeded sample.

**Determinism (amended)**: per-request base seed = `sampling.seed` or
**sha256(request_id) → 63-bit int** (never Python `hash()` — randomized per
process, and D6 splits processes); per-position generator seed =
splitmix64-style mix of (base_seed, position) — plain addition collides across
adjacent user seeds. Scope of the claim: the sampler *preserves* TP rank
agreement given bitwise-identical logits per rank (a collectives/runner
property); it cannot repair divergent logits. torch CPU multinomial with a
seeded Generator is deterministic per build (stated assumption; cross-platform
bitwise identity is not claimed).

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
per-request in `_submit` from `tokenizer.vocab()`.

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
- **Per-request gating (amended)**: spec is bypassed per request (normal decode
  path) unless `temperature == 0` AND no penalties AND no `json_schema` —
  a constructor-time check cannot see per-request params, and penalties change
  the argmax so equivalence would not hold; grammar would need per-position
  `accept()`+rollback (xgrammar has `rollback`, deferred to M17). The
  equivalence suite asserts the bypass behavior.
- Wired via `KairyuBackend(speculative="ngram", speculative_tokens=k)`; default
  off. Acceptance-length counters exposed (G4 M-A4 lineage).

Invariant pinned: spec ≡ non-spec greedy through the full engine, on repetitive
prompts (accepts > 0) and adversarial ones (accepts 0).

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

### D6 — Process split: ZMQ ROUTER/DEALER, msgpack; service owns the tokenizer

New `kairyu/engine/core/engine_service.py` (child main) +
`kairyu/engine/zmq_backend.py` (`EngineBackend`, name `"kairyu-proc"` —
**registered via a `_LAZY_MODULES` entry in `registry.py`**, the only wiring that
makes it reachable from YAML).

- Service: single-threaded loop — **drain ZMQ ops → schedule → execute →
  update** (the D1 threading discipline holds by construction). Owns tokenizer +
  sampler + engine core. Heartbeats are answered between steps; the backend's
  death-detection timeout must exceed worst-case step time (documented knob).
- Events: msgpack `{request_id, new_token_ids, text_delta, logprobs?, finished,
  finish_reason, num_cached_tokens}`; **the first event for a request also
  carries `num_prompt_tokens`** (amended — the service owns the tokenizer, so
  the server side cannot count prompt tokens; M9's usage truth needs it).
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

## 3. What M8 does not include (explicit non-goals)

- `n > 1` parallel sampling in the kairyu backend (M9, rides D2's seams).
- Real model architectures / multi-layer KV pools (M12); `TinyAttentionLM`
  stays the oracle.
- EAGLE/MTP draft sources and sampled-mode (rejection-sampling) or
  grammar-composed speculation (M17/G4) — D3/D4 build the machinery.
- TP multi-process SPMD (M16); the in-process `TPModelRunner` only gets the D2
  return-type ripple.
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
