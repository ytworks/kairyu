# V4.1 ensemble example

Status: implemented and CPU-validated; GPU validation deferred by owner
(2026-09-12). No six-GPU serving or quality claim.

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
thinking-budget processor; hardware enforcement is still a pending gate.

Existing publication behavior is preserved: the opening can precede audit,
and an exhausted failing final answer can still be published. The mechanism
does not guarantee minimum compliance or factual correctness for all requests.

## V41E-D4 — Transferable evidence

CPU fixtures establish orchestration/wire contracts, not model quality. GPU
verification is explicitly deferred. Later runs must compare startup config
hashes and running image IDs against the checkout before recording measurements,
including under `--no-start`. Paired latency baselines must be measured on the
new configuration; no fallback to old-model measurements is allowed.

References: `examples/qwen3.8-deepseek-v4.1-8gpu/README.md`, `L1-NOTES.md`,
`MEASUREMENTS.md`, and implementation plan `2026-09-12-v41-ensemble-example.md`.
