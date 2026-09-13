# V4.1 ensemble example

Status: GPU validation in progress (2026-09-13). Native high-floor, all four
composed effort contracts and the bounded V41E-D12 image integration pass.
The generic serving matrix completes all 128 measured requests; coding and its
paired native baselines are running. Final idle/readiness and metadata/CI remain.
V41E-D13 limits completion to implementation behavior; model-answer quality is
not a completion gate. Earlier evidence remains preserved in MEASUREMENTS.

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

## V41E-D3 — DeepSeek requirements (effort amended by D10)

Port the example-level mechanism from PR #595 at `31f1adc`, without changing
the original example or importing its measurements. A requirements root
extracts stable minimum/optional criteria and passes them to the planner,
answers, critique, synthesis and audit. Head/draft remain parallel roots and
direct routes bypass requirements. The original conversation remains authoritative.

The initial Requirement policy used native DeepSeek high regardless of request
effort; V41E-D10 supersedes that effort rule with a high floor. Other
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

## V41E-D7 — Representable JSON literals in Requirement generation

The pinned XGrammar 0.2.6 lowers string `minLength: 1` to a character class
that excludes every JSON escape. Actual GPU criteria therefore truncate at an
embedded quote; CPU character and native-token matchers reproduce rejection
of otherwise valid quote, backslash and newline escapes. Remove `minLength`
only from the three arbitrary-text fields in the example generation schema.
Keep array cardinality, object shape, field types, ID pattern and priority enum.
The unrestricted native JSON-string rule can represent complete user literals.

The smoke and quality validators still reject empty/whitespace checklist fields;
this is verification-time enforcement, not a new serving-time guarantee. The
runtime grammar now permits empty strings, an explicit tradeoff to avoid
silently excluding valid literal content. Original requests remain authoritative
for synthesis/audit. Keep failed prompt trials and the exact grammar/source
reproduction, and rerun actual fixed-high Requirement literals after deployment.

## V41E-D8 — Seeded top-p retains a forced thinking terminator

A fixed-high Requirement can exhaust 8192 tokens with an empty public body.
Native token-ID diagnostics isolate 16 ordinary tokens followed by 112 special
zero tokens at a 16-token thinking budget. Both structured and unstructured
requests fail with seeded positive-temperature top-p sampling; greedy and
`top_p=1` correctly emit the terminator. The active V2 budget kernel writes a
large forced logit (`1e9`). In the split top-p kernel, FP32 cutoff reconstruction
rounds up to that maximum, and the strict comparison removes every candidate.

Add the existing monolithic cutoff guard to the split kernel in this example's
pinned child runtime: if the cutoff is not below the true row maximum, use
negative infinity as the cutoff. Keep the budget kernel and forcing value
unchanged. This preserves the forced distribution without changing model effort,
total caps, or the sampler's ordinary path. Exact input/output source hashes
and idempotence fail closed. CPU/GPU oracle results and preserved failures are
recorded in MEASUREMENTS; stochastic forced-budget cases join the native probe.
The sibling/parent image remains unchanged. The corrected child is a new image
and requires attested native/composed replay before closing its gates.

## V41E-D9 — Candidate word-limit planning

The explicit-high primary replay completes publicly, but one Qwen policy answer
recounts individual words until its forced thinking boundary, continues that
deliberation in its exposed body, then truncates the actual memo at the total
4096-token cap. The policy does not request word enumeration. The final audit
checks synthesis, so its PASS does not prove that every peer was complete.

Add role-local guidance to the Qwen draft and two policy answers: for maximum
word limits leave a margin rather than enumerate/recount words, and emit the
complete requested answer after private reasoning. Preserve exact-length
requests, medium effort, sampling, total caps and thinking reservations. This
is a prompt correction requiring replay, not a guaranteed bound on model
deliberation or a serving-time repair mechanism. Retain the failing peer and
do not relax the candidate-completion check to hide it.

## V41E-D10 — Requirement effort has a high floor

The user amended the contract: when the request's effort exceeds high,
Requirement must use that higher effort. With the supported canonical modes,
omitted/low/high use native high and max uses native max. The public API already
normalizes aliases before orchestration (including xhigh to max).

Change the Requirement role from fixed high to inherit, preserving canonical
request effort until the example hook applies the floor. The resolved top-level
role effort is authoritative; nested chat-template kwargs are overwritten to
the same effective value and thinking remains enabled. Thus max/nested-low
becomes max, while low-or-omitted/nested-max becomes high. Nested template
metadata does not select orchestration effort. Keep the 8192 total and 4096
thinking reservation unchanged; native effort and the safety cap are separate.
Other DeepSeek and all Qwen effort settings remain unchanged. Preserve older
fixed-high measurements at their revisions and rerun native/conflicting-input
and public effort cases before claiming this amended behavior GPU-validated.

## V41E-D11 — Preserve supplied facts, assumptions and unknowns

The high-floor replay produces complete candidates and correct effort forwarding,
but LOW and omitted final memos label explicitly supplied hard constraints as
assumptions. LOW also strengthens "not supplied" into a categorical claim that
no other options or revisions exist. The checklist retains the original facts
and constraints; audit nevertheless accepts these statements because they sit
under an assumptions heading. This is a semantic synthesis/audit defect, not
loss of Requirement data. Preserve the successful protocol evidence separately
from these failed semantic checks and the intentional interruption of high.

Clarify both synthesis templates and audit generically: supplied facts and
requirements keep that status; missing evidence does not establish absence;
added assumptions must be necessary, conditional and consistent with the given
information. Do not invent premises to fill a section. Audit must assess the
actual statements rather than treating headings as evidence of compliance.
Keep native effort, reservations, caps, schemas, other roles and fixtures
unchanged. This prompt correction needs a new served configuration and fresh
composed replay; it does not guarantee arbitrary model correctness.

References: `examples/qwen3.8-deepseek-v4.1-8gpu/README.md`, `L1-NOTES.md`,
`MEASUREMENTS.md`, and implementation plan `2026-09-12-v41-ensemble-example.md`.

## V41E-D12 — Preserve text/image boundaries in orchestration context

The formal image quality case ends its actual user text with the required
literal `Health status: unknown.` and then attaches an image. L3's flattened
display representation appends `<image:0>` directly to that text in both
conversation JSON and the duplicate latest-user view. Requirement extracts the
combined string as the required ending; synthesis publishes it and audit
accepts it. Exact native message/trace correlation establishes that the extra
characters originate in rendering, not in the user's instruction.

Keep ordered text and image-reference parts separate in the L2 conversation
JSON. Preserve each text part verbatim and represent attachment indices as
non-text metadata, without copying image payloads into the role prompt. For a
latest user turn containing images, use that structured view rather than a
second flattened text view. Actual typed image forwarding, direct VLM input,
plain-text requests and literal marker-like user text retain their behavior.
Do not strip strings from generated answers or weaken exact-ending checks.

This changes shared L3 orchestration rendering for image requests. Native
models, Requirement effort, Qwen defaults, L2 roles, prompts and token caps
stay unchanged. Verify the original text boundary and image order in CPU
request tests, then confirm image forwarding and response completion with a
bounded integration request. Observations about the headed memo's assumptions
remain examples of model behavior, not additional implementation blockers.

## V41E-D13 — Verify implementation behavior without promising model quality

The owner explicitly rejects treating model-answer quality as an implementation
completion condition. Additional independent semantic reviews, negative/positive
audit controls and repeated prompt tuning exceeded the requested scope. Stop
those additions, preserve existing artifacts and do not require their completion
before serving measurements or PR handoff. An interrupted diagnostic is neither
a model failure nor an unfinished requirement of this example.

Verify the requested deployment and mechanisms: model/replica topology, effort
selection and propagation, Requirement generation and consumption, native image
forwarding, API completion, cancellation/resource release, startup and the
existing serving measurements. Use scoped regressions for reproducible code
defects, such as altered input text. Reuse applicable completed evidence when
the relevant implementation is unchanged; do not repeat broad matrices merely
because documentation or an unrelated input path changes.

Model-generated requirements, answers and audit judgments may be wrong. A
model's PASS, a fixture diagnostic or successful execution is not a guarantee
of factual correctness, requirement completeness or reliable self-correction.
Report observed behavior and its limits without turning content preferences or
unprovable quality claims into new gates. Retain this distinction in the PR's
remaining tasks and handoff.
