# DeepSeek V4.1 (6 GPUs) + Qwen3.8-27B (2 GPUs): judged five routes with a DeepSeek-led critical ensemble

Status: **Accepted; CPU contracts implemented; L1 recipe amended by the owner (V41T-D1 amendment, 2026-09-14); audit evidence rule amended after GPU diagnosis (V41T-D2 amendment, 2026-09-14); GPU gates in progress.**
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
re-check the proposal against request and all candidates;
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

### V41T-D2 amendment (2026-09-14) — the audit does not accept unbacked execution claims

- What: the first forced-ensemble serving row (generic, c1, 32 requests)
  exhausted the audit on 8 requests and needed refinement on 19. Reading
  the audit texts (three requests replayed with intermediate outputs
  exposed) showed the cause: the benchmark row label `Run <id>, case N`
  was extracted as a minimum requirement to execute a run, an answer that
  said plainly that no run was performed was FAILed three times, and
  answers that invented an execution result ("case run accepted as PASS,
  verified") were PASSed — 25 of the 32 published answers carried such a
  claim. The audit now treats a statement that something was run,
  executed, tested, measured, or verified as evidence only when the
  conversation contains the matching tool call and result, FAILs it as
  fabrication otherwise, and treats a requirement that demands an action
  the turn cannot perform as satisfied by an answer that says so plainly
  and delivers everything else; `synthesis` and `final` (streamed and
  headless) carry the same rule. The verification datasets label the row
  identity as an identifier, not an instruction.
- Why: the audit's purpose is to gate publication on real evidence; a
  protocol that rewards fabricated execution claims inverts it. The rule is
  example-owned prompt policy; Kairyu is unchanged. After the change the
  same three requests passed on the first audit with an explicit "cannot
  be executed in this turn" statement and no invented result.
- Consequence: every public gate is re-run on the revised served config;
  the earlier rows are kept in `MEASUREMENTS.md` as evidence of the defect.

### V41T-D2 amendment 2 (2026-09-14) — the head opens with the answer, the final honours the total length

- What: with the audit texts recorded, the remaining first-attempt FAILs
  (17/32 on the forced generic c8 row) were the head's own instruction
  "one sentence that states what is being answered" — every published
  answer opened with "The request asks for…" — and a combined length of
  ≈300–370 tokens against a requested ≈256. The head now begins with the
  answer itself, never restates, classifies, or comments on the request,
  and keeps under half of any stated length; the final counts the opening
  first so opening plus remainder meet the requested total and does not
  continue a framing opening. Owner decision; every public gate is re-run.
- Why: the preamble is a user-visible defect, not only an audit finding,
  and the refinement rounds it caused doubled ensemble completion time.

### V41T-D2 amendment 3 (2026-09-15) — the checklist is verification data; the Qwen answerers' input is bounded by the policies cap

- What (owner instruction): the requirements checklist is read by `audit`
  only. `policies`, `answer_1..4`, `synthesis`, and `final` (streamed and
  headless) no longer receive it or depend on it; `policies` now runs in the
  first wave next to `head`, `requirements`, and `independent`.
  `requirements` remains a scheduling dependency of `final` because the
  Conductor runs a verifier inline after its target and rejects a verifier
  input that the target does not depend on (`conductor.py` `_validate`);
  the final's prompts never render it. `budget.max_steps` is unchanged.
- Why: a role that writes the answer while reading the extracted criteria
  lets the extractor's interpretation leak into the answer, and the audit
  then checks an answer that was written to its own checklist; the request
  must remain the only source for the answer.
- Qwen input bound: superseded the same day by amendment 4 (the static
  8,192 cap left the conversation itself unbounded).

### V41T-D2 amendment 4 (2026-09-15) — the Qwen answerers read the policies output alone

- What (owner instruction): the roles that run before the Qwen answerers
  must adjust their output so that the answerers' input never exceeds
  Qwen's context, whatever the conversation's length. The answerers'
  prompts no longer contain `{query}`; their only input is the `policies`
  output, whose cap is the DSL ceiling 131,072 (`internal_max_tokens`
  admits no more; the plan's 245,000 was reduced for that reason):
  131,072 + ≈300 scaffold + 16,384 answer budget = 147,756 < 262,144.
  `policies` writes `=== REQUEST ===` (the request and the material the
  answer relies on, verbatim while it fits, otherwise selected with the
  omissions stated) and `=== POLICIES ===` (the four policies). The final
  writes the complete answer when the opening is empty. Images still reach
  the answerers natively (Conductor attaches them to every multimodal role).
- Why: a bound that includes the conversation cannot be guaranteed by
  configuration; a bound on a DeepSeek output can. DeepSeek reads up to
  1,048,576 tokens, so it can prepare the answerers' input for any
  conversation the product accepts.
- Not covered: the head reads the conversation and has no role before
  it; beyond Qwen's context it fails and the ensemble answers without a
  streamed opening (`conductor.py:1935-1970`). The judge reads only a
  4,000-character view of the latest user turn (`orchestrator.py`
  `_bounded_profile_judge_view`), so it cannot route by length and may
  choose a Qwen direct route for such a conversation, which then fails
  (recorded limit of the judged product). The long-input gate proves the
  forced ensemble completes on 300K- and 450K-token conversations (the
  L3 rendering carries the latest user turn twice, so DeepSeek reads about
  2× the conversation; 450K keeps 2× + the policies cap 131,072 under
  DeepSeek's 1,048,576).
- Cost: DeepSeek copies the request into its output (≈65 tokens/s
  single-stream), and the wave scheduler makes the answerers wait for it.

### V41T-D2 amendment 5 (2026-09-16) — synthesis cap and caller ceiling at the DSL maximum

- What (owner decision): `synthesis` gets 131,072 tokens at high and max
  effort (was 65,536 at high); the verification matrices and the Chat UI
  send `max_tokens 131072` instead of 65,536, because Kairyu clamps every
  internal role to the caller's limit.
- Why: on coding tasks in the forced ensemble, DeepSeek spent all of
  65,536 tokens thinking and emitted no text in 5 of 32 requests
  (`policies` twice, `synthesis` three times); the pipeline still
  published audited answers (four PASS, one exhausted), but each such stage
  cost about 950 s and lost its contribution. Thinking length cannot be
  bounded directly; doubling the room halves the chance of running out at
  the price of a longer worst case (about 30 minutes per run-out).

## V41T-D3 — Requirement extraction on DeepSeek with the PR #595 contract

The `requirements` role ports PR #595's specification: a bare JSON array of
`{id, priority, requirement, acceptance_criterion, source}` objects, R1..
consecutive, `minimum`/`optional`, explicit constraints never downgraded,
literals / numbers / operators / punctuation / line breaks preserved, later
corrections honoured, quoted text / image text / tool results treated as
data, no invented obligations, no fences or commentary. The checklist is
verification data read by `audit` alone (amendment 2026-09-15, below); the
request remains authoritative. JSON is prompt-constrained (no
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
masked as 502; the ensemble's Qwen answerers are bounded by construction
(V41T-D2 amendment 4), while the judge and the head still read the whole
conversation, so beyond 262,144 rendered tokens the product answers through
the ensemble without a streamed opening and the Qwen direct routes are
unreachable
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
- Amendment (2026-09-14, extended 2026-09-15): a candidate role
  (`answer_1..4`, `independent`) or the `policies` role that ends at its cap
  is counted per stage (`stages.json` cap hits) and recorded, not a failed
  row — each is one weak input to a synthesis that reviews all five
  candidates critically, and the published answer is still gated by the
  audit (since amendment 4 the policies output feeds the answerers only;
  its first cap hit was a 65,536-token thinking run on the generic prompt
  whose answer still passed the audit). Head, requirements, synthesis,
  final, and audit cap hits keep failing the row. `VERIFY_CONCURRENCY`
  re-runs selected rows of a matrix under a new run id.
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
