# V4.1 ensemble example

Status: GPU validation in progress (2026-09-13). The initial GPU-resident
Engram candidate failed startup; no six-GPU serving or quality claim yet.

## V41E-D1 — Separate six-plus-two deployment

Keep the original example unchanged. Deploy V4.1 Flash on GPUs 0–5 and two
Qwen3.8 27B FP8 TP1 replicas on GPUs 6 and 7. The static candidate is TP2 ×
Attention-DP3 / EP6, PP1, within one physical DeepSeek service. TP6 cannot
divide the model's attention heads. Shared cross-layer caches and uneven
Engram placement make PP3 a separate investigation, not the default.

Reuse the pinned V4.1 SM120 overlay. DSpark is off: the draft's 128 experts
do not divide EP6, and the TP8 result cannot establish draft support here.
The initial context/memory limits are unverified candidates. GPU tests must
establish startup, memory fit, kernel/collective correctness and performance.

2026-09-13 amendment: the initial 16K-batch, GPU-resident Engram candidate
loads 84.09 GiB of weights per GPU but fails sparse-indexer memory profiling
(512 MiB allocation with only 377 MiB free). Select the existing pinned-runtime
Engram CPU-offload option for the next trial, retaining TP2/DP3/EP6, 1M context,
16K batching and all other limits. This uses pinned host tables and UVA lookup;
the fixed source implements the same DP/TP gathers for resident and offloaded
tables. Actual startup and performance remain gates.

The offloaded runtime starts but exposes a masked sparse-KV failure: invalid
indices gather slot zero, whose nonfinite values contaminate the attention value
product even when its weight is zero. Eager/NCCL experiments reproduce it;
stage instrumentation finds the first NaN at layer 0 Attention, and an independent
GPU oracle reproduces all 12 poisoned masked cases. Add a source-hash-guarded
example-local FlashInfer overlay that gathers a zero row for invalid candidates
in decode and prefill. Preserve copy sizes/barrier accounting and all valid
addresses. Keep the sibling example unchanged and isolate generated kernel caches.
Both the numerical oracle and full serving gates must pass before selection.

## V41E-D2 — Two-policy ensemble with native images

Preserve the five-route judge, Qwen head/draft, DeepSeek critique/synthesis,
Qwen audit and two-refinement policy. One DeepSeek planner generates exactly
two distinct policies; two Qwen answers and the independently refined draft
form three peer candidates. All roles use original images directly. Remove
the image-description bridge and old DeepSeek text scaffolds; vLLM's V4.1
tokenizer/reasoning/tool parser owns native chat rendering.

## V41E-D3 — Fixed-high DeepSeek requirements

Port the example-level mechanism from PR #595 at `31f1adc`, without changing
the original example or importing its measurements. A requirements root
extracts stable minimum/optional criteria and passes them to the planner,
answers, critique, synthesis and audit. Head/draft remain parallel roots and
direct routes bypass requirements. The original conversation remains authoritative.

Requirement uses native DeepSeek high regardless of request effort. Other
DeepSeek thinking roles inherit effort, default high; direct is non-thinking.
Qwen keeps its existing medium alias and non-thinking roles. Example middleware
matches complete shipped role templates, including refinement forms, and sets
native thinking/effort, JSON checklist and audit serialization policies. It
reserves checklist/public output within existing total caps using vLLM's
thinking-budget processor. Native DeepSeek enforcement passes; composed
candidate completion is checked separately.

Amendment (2026-09-13): reserve half the Qwen draft/answer allowance for actual
candidate text, without changing effort or increasing total caps. The first
GPU primary request produced a truncated draft and one all-thinking, empty
policy answer. Require nonempty bodies and below-cap completions for draft
and all three synthesis peers in the GPU probe; API success alone is insufficient.

Existing publication behavior is preserved: the opening can precede audit,
and an exhausted failing final answer can still be published. The mechanism
does not guarantee minimum compliance or factual correctness for all requests.

## V41E-D4 — Transferable evidence

CPU fixtures establish orchestration/wire contracts, not model quality. GPU
verification was authorized on 2026-09-13. Runs must compare startup config
hashes and running image IDs against the checkout before recording measurements,
including under `--no-start`. Paired latency baselines must be measured on the
new configuration; no fallback to old-model measurements is allowed.

## V41E-D5 — Opt-in public paragraph separator

GPU replay still joins a complete head to `Facts:` without whitespace despite
both prompts requesting a separator. Add a head-only `continuation_separator`
option to the DSL/Conductor, default empty, and enable two newlines here. This
amends EO-D7 only when explicitly configured: exact head-prefix deduplication
and `NO_CONTINUATION` suppression happen first, then the separator is inserted
only between nonempty, non-whitespace adjacent text. Existing whitespace is
preserved. Whole, deferred verified and live-stream publication agree.

Head-only exact answers and headless JSON/tool responses gain no extra text;
raw candidate bodies and audit inputs remain unchanged. These presentation
bytes are not model-generated tokens and do not change usage or role budgets.
The new example's deployment hash includes the Conductor/DSL source files so
verification cannot silently use an older API image after this source change.

## V41E-D6 — Streaming iterator ownership on disconnect

A public client disconnect during the deferred Qwen audit leaves upstream
inference running for about 139 seconds. Native unary cancellation, including
the exact audit hook, clears promptly; a CPU ASGI reproduction identifies an
unclosed body iterator when a keepalive send is interrupted. The shared SSE
response must explicitly close its iterator under a cancellation shield. The
chat renderer and Conductor event adapter likewise close their owned sources,
draining keepalive cancellation into the outstanding backend generation.
ASGI 2.3 disconnect, 2.4 send failure and normal exhaustion are regression
cases. This is a shared lifecycle correction; existing model/routing policy
and generated response bytes are unchanged. GPU public cleanup is a separate
gate from the already-passing native cancellation tests.

References: `examples/qwen3.8-deepseek-v4.1-8gpu/README.md`, `L1-NOTES.md`,
`MEASUREMENTS.md`, and implementation plan `2026-09-12-v41-ensemble-example.md`.
