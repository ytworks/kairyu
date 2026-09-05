# Evidence-based repair route (DTO-D16)

The existing Example now automatically selects among **six routes** with the
same bounded Qwen judge. Start it with the existing `./run.sh`; use the same
`kairyu-auto-max` model. There is no separate deployment or manual route selector.
The original `primary` ensemble, four direct profiles, five existing criteria,
common budgets, model pools and hardware layout are preserved. No framework,
DSL, API, external agent or benchmark harness changes are required.

## Selection and composition

`SWE` selects `swe_evidence` for ambiguous repository failures, competing repair
hypotheses, cross-file/interface uncertainty, or visible test results that
contradict completion. Clear edits, routine reads/commands and general questions
remain covered by the existing routes. A hard problem that one DeepSeek call
can solve should still use `DEEPSEEK_THINK`; general ensemble work retains
`ENSEMBLE`. The criteria do not use benchmark names or instance IDs.

The judge still reads only the existing latest-user 4,000-character head/tail
view plus tool/image flags. It cannot reliably count failures hidden in prior
turns. The selected roles, unlike the judge, receive the conversation context.
The GPU check must measure actual route selection: mocked labels do not prove
that the small judge classifies difficult issues correctly.

| Stage | Worker / effort | Task | Output cap |
|---|---|---|---:|
| `swe_hypothesis` | Qwen medium | Root cause and minimal repair/investigation | 2,048 |
| `swe_alternative` | Qwen medium | Independent explanation and regression risk | 2,048 |
| `swe_plan` | DeepSeek caller effort | Compare evidence and plan the next action | 8,192 |
| `swe_critic` | Qwen medium | Check requirements, assumptions and verification | 2,048 |
| `swe_decision` | DeepSeek caller effort | Resolve critique and specify the exact action | 8,192 |
| `swe_final` | Qwen non-thinking | Encode the decision in the original response/tool contract | 8,192 |

The two hypotheses run in parallel, followed by plan, critique, decision and
publication. All Qwen thinking roles use the existing DTO-D14 **medium** tier:
DSL `reasoning_effort: high` maps to the Example's medium Qwen template, regardless
of the caller's effort. `swe_final` is non-thinking. DeepSeek inherits effort.

Only the final role receives structured tool intent. Qwen's existing chat
path/template renders tools; the DeepSeek role template is a scaffold
passthrough, so DeepSeek stages remain internal action advisers. Actual editing
and tests remain the external agent's responsibility. Missing proposals are
not treated as evidence and suggested tests are never treated as executed tests.

The existing conditional `image_description` role is copied into this profile
and feeds both DeepSeek stages. This matters because the existing judge's
capability filter can offer a Qwen-containing profile on image input.
There is no head or verifier in the new profile, so no semantic refinement loop.
The existing bounded empty-final-output retry remains; arbitrary malformed tool
syntax has no new repair loop. Intermediate output follows the unchanged
`expose_intermediate_outputs` reasoning/trace setting, not public answer content.

## Cost interpretation

Six ordinary text calls total at most **30,720 generated tokens**, including
thinking, before the caller's smaller per-call allowance. The judge adds at
most eight tokens. Images add up to 4,096 description tokens; an empty final
can consume one additional bounded generation. These are generation caps,
not measured utilization, latency or a problem-wide spending guarantee.
Input processing, cache hits, repeated API turns and failed calls must be counted.

The framework is unchanged, so this Example does not introduce a persistent
problem budget. The objective is a problem's total cost no greater than the
existing ensemble's reference budget, with mean routed cost no greater than
the original five-route policy. Do not infer those conditions from call counts.

## CPU checks and remaining GPU work

CPU tests compile the actual configuration and verify unchanged original policy
semantics, all labels/fallback, medium template rendering, dependencies, token
caps, real backend HTTP tool forwarding, streaming and required trace stages.
They use simulated model responses; they do not establish repair quality.

The following work requires the eight-GPU environment and remains open:

- [ ] Re-run the current baseline's pending DTO-D15 coding/generic gates and
  lock its digest. Preserve the pre-addition five-route configuration from
  commit `097affb1` for the comparison; keep engine/model/template settings equal.
- [ ] Run the updated Example's normal `./run.sh`, `./verify.sh coding` and
  `./verify.sh generic`; confirm `/routing` lists all six choices. Retain old
  TTFT gates and report the repair route's latency separately (it has no head).
- [ ] Verify real tool-call round trips and image-description behavior. Inspect
  new roles for thinking-only empty proposals and final formatting failures.
- [ ] Freeze 100 repository-stratified Public problems and model, image,
  dataset, harness, sampling and turn-limit revisions. Compare the original
  routing, forced original ensemble, and new automatic routing; use a
  no-critic ablation on this development set if needed. Do not expose gold
  patches or evaluator-only tests to generation.
- [ ] Measure model-specific input/cache/output costs on the same hardware;
  include all judge/internal/thinking/retry usage from traces, not just public
  tokens. Report missing usage instead of treating it as free work. Freeze
  the original ensemble's development p95 total problem cost as the reference
  threshold before the full evaluation; do not raise it after seeing results.
- [ ] Freeze the chosen configuration, then evaluate all **731** Public tasks
  with one final patch per problem using the unchanged external harness and
  official evaluator. Count every problem and report the 631 development-excluded
  tasks separately. **585/731 or more** is the 80% target, not a current result.
- [ ] Report resolution rate, per-problem cost/max/p95, routed mean cost,
  route mix, latency and failures against the original five-route run. Any
  threshold breach or unknown cost leaves the cost objective unverified.
- [ ] Re-pin the measured Example configuration and record results in
  `MEASUREMENTS.md`. Until then existing GPU measurements remain historical.
