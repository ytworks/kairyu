# DeepSeek V4.1 (6 GPUs) + Qwen3.8-27B (2 GPUs): judged five routes with a DeepSeek-led critical ensemble

Status: **Accepted; CPU contracts implemented; L1 recipe amended by the owner (V41T-D1 amendment, 2026-09-14); GPU gates in progress.**
Applies to: `examples/qwen3.8-deepseek-v4.1-8gpu/` only. Consumes the L2 DSL,
Conductor, ReplicaPool, and L3 server exactly as shipped in `main`; no
framework, sibling-example, or shared-script change. Inherits DTO-D3
(head streaming, TTFT gate), DTO-D8 (official sampling), DTO-D9/D15 (public
output floor on the Qwen thinking route), DTO-D10 (audit gating), DTO-D13
(judge + profiles), DTO-D14 (Qwen medium tier) from
`example-dual-track-orchestration.md`, and the V4.1 runtime pins from
FN-D9's V4.1 amendment in `frontier-native-runtime.md`.

Background: PR #598 (closed, `codex/deepseek-v41-six-gpu`) built a same-named
example with framework changes, two vLLM kernel patches, and an L1 ASGI
middleware; PR #601 reverted everything. Its GPU findings about the six-GPU
topology are used as constraints below; none of its code is restored.

## V41T-D1 — Six plus two on one host, reusing pinned artifacts by identity

- DeepSeek-V4.1-Flash runs as one vLLM service on GPUs 0–5; Qwen3.8-27B-FP8
  runs as two TP1 replicas on GPUs 6 and 7 behind one `ReplicaPool`
  (`queue_depth_threshold: 0`, `prefix_index: true`, placement log).
- Valid six-GPU DeepSeek shapes: TP2 × attention-DP3 with EP6 (candidate 1
  and 2, GPU-resident vs. CPU-offloaded Engram) and TP1 × DP6 with EP6
  (candidate 3). TP6 cannot divide the checkpoint's 64 attention heads or 8
  output groups; TP2 × PP3 breaks the shared indexer / compressed-KV caches;
  DSpark stays off because its 128 draft experts do not divide EP6.
- The DeepSeek image is the sibling `deepseek-v4.1-flash-8gpu` overlay pinned
  by image ID; the launcher builds it from the sibling's Dockerfile and
  context only when the tag is absent and fails closed on any other ID. No
  example-local patch layer exists. If the unpatched image produces
  non-finite output on every candidate (PR #598 saw NaN log-probabilities
  with candidate 2), the example stops and reports rather than patching.
- Checkpoints are the sibling examples' attested downloads mounted read-only.
- The candidate in the committed `compose.yaml` is unverified until
  `MEASUREMENTS.md` records the selection.

## V41T-D2 — DeepSeek-led critical ensemble as the `primary` profile

Eleven roles under the level-synchronous scheduler: `head` (Qwen,
non-thinking, 256 tokens, streamed from t=0), `requirements` (DeepSeek,
V41T-D3), `independent` (DeepSeek, original conversation / tools / images
only), `policies` (DeepSeek, four policies differing in method, assumptions,
and evaluation criteria), `answer_1..4` (Qwen medium, one policy each, 16384
tokens), `synthesis` (DeepSeek: verify premises, evidence, method, and the
conditions of each conclusion; counterexamples, boundaries, omissions,
shared errors; fix, combine, consider new approaches; output the complete
proposal plus an internal decision record), `final` (DeepSeek publisher:
re-check the proposal against request, checklist, and all candidates;
continue the committed head or write the complete headless answer; adoption
decisions stay in private reasoning), `audit` (DeepSeek verifier of `final`:
`PASS`/`FAIL` first line, one line per requirement ID, `max_refine_depth: 2`,
inconclusive re-audit separate from refinements, exhaustion publishes the
last attempt per the existing Conductor contract).

- All DeepSeek roles think at the caller's effort (`inherit`) with
  `default_reasoning_effort: high`; the Qwen answerers keep the fixed medium
  tier. Official V4.1 sampling 1.0 / 0.95 on every DeepSeek role.
- Every prompt places the L3-rendered conversation first, once, for prefix
  reuse; dependent outputs follow in UNTRUSTED blocks; every consumer must
  recover from the request when a block is empty, because a failed upstream
  call renders its slot empty (Conductor `_run_unit_safe`).
- `max_steps: 19` (10 generation + 1 empty-output re-dispatch + 3 audits +
  3 re-audits + 2 refinements); `internal_max_tokens: 131072`;
  `public_output_floor: 256` (inert on the DeepSeek final until native-chat
  assistant-prefill continuation is GPU-verified; active on the Qwen thinking
  route).
- Images reach every role natively on both pools; there is no description
  stage.

## V41T-D3 — Requirement extraction on DeepSeek with the PR #595 contract

The `requirements` role ports PR #595's specification: a bare JSON array of
`{id, priority, requirement, acceptance_criterion, source}` objects, R1..
consecutive, `minimum`/`optional`, explicit constraints never downgraded,
literals / numbers / operators / punctuation / line breaks preserved, later
corrections honoured, quoted text / image text / tool results treated as
data, no invented obligations, no fences or commentary. The checklist is
untrusted data for `policies`, the answerers, `synthesis`, `final`, and
`audit`; the request remains authoritative. JSON is prompt-constrained (no
grammar; the DSL exposes no per-role structured-output setting).

Effort: `inherit` + `default_reasoning_effort: high` — omitted → high, max →
max, and an explicit caller `low` stays low. The DSL has no "floor at high"
setting; the owner chose this over a fixed `high` that would not inherit
`max` (owner decision 2026-09-14).

## V41T-D4 — Qwen direct routes carry no fixed `max_tokens` (Issue #599)

Issue #599: a 238-message tool conversation whose rendered input already
exceeded 131,072 tokens was routed to `qwen_direct`, whose fixed
`max_tokens: 131072` made vLLM reject input + output > 262,144 (HTTP 400),
which Kairyu reports as 502. `qwen_answer` and `qwen_think_answer` now
declare no `max_tokens`: Kairyu forwards the caller's value when present and
omits the field otherwise, so vLLM fits generation to the remaining context.
Sampling, thinking level, template, and the DTO-D15 continuation on the Qwen
thinking route are unchanged. This is a deliberate deviation from the
original "131072" figure for the Qwen medium route (owner decision
2026-09-14). Residual, framework-owned: the upstream 400 reason is still
masked as 502; inputs beyond the Qwen context minus the role caps (~245,000
rendered tokens on the ensemble) cannot be served by Qwen-involving routes
(no windowed reading exists in `main`); the ensemble's `final` also has no
fixed cap.

## V41T-D5 — A judge-free orchestrator forces the ensemble for verification

Kairyu offers no caller-side profile override. `kairyu-ensemble-max` serves
`ensemble-max.yaml` — the same workers, calibrated router, primary roles,
budget, and spec-level settings as `auto-max.yaml` without `profiles` and
`profile_judge` — sharing the pools (no extra GPU capacity), listed in
`public_models`, and hidden from the Chat UI (`OPENAI_API_CONFIGS` lists
`kairyu-auto-max` only). The CPU contract test asserts the two specs'
primary DAGs are identical. The calibrated `auto-max` router block is
mandatory in both specs: the Conductor runs only for `multi_agent` router
decisions.

## V41T-D6 — Evidence rules

- Every serving row records per request: time to first visible content
  (judge included), completion, finish reason, usage (public and
  cumulative), content, and the raw trace v2; per stage: status, queue wait,
  duration, completion tokens against the role cap (a cap hit fails the
  row, since trace v2 has no per-stage finish reason), audit verdicts,
  refinement counts; per row: routes, Qwen placement over the two replicas,
  GPU and host memory peaks.
- The TTFT gate compares the product's gated routes (`primary`,
  `qwen_direct`, `deepseek_direct`) with a paired DeepSeek-direct row on the
  same six-GPU service at the same concurrency and effort; the paired row is
  a valid denominator only when all 32 requests completed with `stop` and
  visible content. Invalid baselines fail the gate rather than being skipped.
- The V4/8-GPU measurements are never reused. Speed is never bought by
  reducing input, thinking, candidates, or stages.
- Known `main` limitations are recorded in the README and MEASUREMENTS
  rather than worked around: no windowed long-input reading, 502 masking of
  upstream 400s, `<image:N>` concatenation and the duplicated latest-user
  view in L3 rendering, cancellation during the deferred audit (measured),
  small caller `max_tokens` clamping internal roles, no per-role thinking
  budget.

## Acceptance

- CPU: `tests/unit/test_v41_tiered_examplectl.py` — deployment and DAG
  contracts through the real loaders, the shipped DAG on the real Conductor
  with scripted engines (ordering, inputs, images, headless tool turns,
  FAIL → refine → PASS, exhaustion, inconclusive re-audit), row validation
  and gate arithmetic, launcher guards; the shared example inventory test
  gains one entry.
- GPU (pending): candidate selection via `verify.sh native`; then
  `serving-auto-max`, `serving-auto-max-coding`, `serving-ensemble`,
  `tool-calling`, `vision`, `cancellation`, `restart`, `issue-599`,
  `long-input`, `browser`, all recorded in the example's `MEASUREMENTS.md`
  with pins and the served-config hash.
