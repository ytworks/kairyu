# Verified-tool route: DeepSeek draft, four Jev angles, DeepSeek repair

Owner-approved plan, 2026-10-05. Example: `examples/deepseek-v4.1-openjev-verified-8gpu`.

## Goal

VERIFIED_TOOL is one DeepSeek call at max effort, unverified (VCO-D17). It
becomes a loop:

1. DeepSeek V4.1 Flash drafts the reply at the caller's effort (default high)
   with the caller's tools. Only DeepSeek follows the effort.
2. OpenJev reads the draft's planned tool calls from four angles.
3. All pass: the reply is published. A failed angle: DeepSeek gets the draft
   and Jev's result and writes the reply again.
4. Back to 2 (at most two repairs).

The four angles' thresholds are calibrated on recorded DeepSWE turns.

## Approach: L2 DSL only, no framework change

The existing L2 already provides every piece: a checklist verifier with
`refine_prompt` and `max_refinements`; fixed questions without `foreach`
(`kairyu/orchestration/checklist.py`); the final unit carries the caller's
tools on every attempt, repairs included (`conductor.py` `_request_intent`);
`reasoning_effort: inherit` with `max_tokens_by_effort`. The longest path
(draft, three reads, two repairs = 6 steps) fits `max_steps: 15`.

## 1. Route (`verified.yaml`, `verified-always.yaml`, identical profile)

- `verified_tool_answer` (DeepSeek, final unit): `inherit` effort, 16,384 /
  32,768 / 65,536 tokens by effort (capped by the caller), prompt `{query}`;
  `refine_prompt` = conversation + draft (`{previous}`) + Jev's failed angles
  (`{feedback}`), returning tool calls through the tool interface.
- `verified_tool_check` (Jev verifier): state = whole conversation with tool
  results (`query`, 120,000 chars like the route judge), `tools`, the reply.
  Four questions: `first_call` (appropriate next move given the understanding
  and tool results), `order`, `progress` (toward the final goal), `runs`
  (no error, does what is intended). "Planned tool calls" = the calls in the
  reply plus later steps its text states. Thresholds 0.5 until calibrated;
  `steps: 4`; `max_refinements: 2`; `on_unavailable: publish_unverified`;
  no acceptance read; the last attempt is published on exhaustion.
- The route judge is unchanged.

## 2. CPU test

Replace the "one max-effort call, unverified" test with the new contract
(count unchanged): tool conversation at caller effort low -> draft at low with
tools -> Jev fails an angle -> the repair carries the draft, Jev's result and
the tools -> pass -> tool calls published with `kairyu_verification`.

## 3. Calibration (DeepSWE replay)

- Data: `~/kairyu-bench/results/deepswe-verified-tool-e-full-max-4w-20261004-r1`
  (VERIFIED_TOOL, max effort; 1,537 turns, 31 tasks, 13 solved), each turn
  with its actual execution result (returncode, output), the later trajectory
  and the task outcome; plus the 83 turns of the VCO-D16 replay. Requests are
  rebuilt from the relay records with the bash tool definition.
- Sample about 200 turns stratified by task x position (early/mid/late),
  oversampling failure signals (nonzero returncode, repeated command, a change
  later undone, unsolved task) with reweighted figures reported; split by
  task into calibration and held-out halves.
- Labels: each recorded reply, per angle OK/NG, in hindsight from the
  execution result (`runs`) and the later trajectory and outcome (the other
  three); two independent Claude labellers blind to Jev, a third settles
  disagreements; agreement reported. VCO-D16's A0 labels are not reused.
- Jev probabilities through the production checklist path with the same YAML
  (OpenJev only, no DeepSeek).
- Per angle: the smallest threshold whose accepted replies have a one-sided
  95 % Clopper-Pearson upper bound on the NG rate <= 0.10 on the calibration
  half. Reported per angle and for all four: held-out miss rate, rate of
  sound replies sent to repair, AUROC. With no threshold meeting alpha, the
  threshold minimising total errors is recommended; the owner decides. Too few
  NG labels or AUROC below about 0.7: a wording change is proposed to the
  owner (a design change).
- Files: `calibrate_tool.py` (example-owned), `datasets/deepswe-tool-turns.json`
  (turn ids, split, labels and reasons; no conversation text), gate
  `calibrate-tool`, results in `MEASUREMENTS.md`.

## 4. GPU gates

- `verified-tool-route`: 20 tool-requiring conversations x unary/stream x
  caller effort (none/low/high/max): route verified_tool, structured
  tool_calls, every DeepSeek generation at the caller's effort (none: high),
  `kairyu_verification` with the four angles; repairs, failed angles and time
  recorded. `chat()` treats the route as verified.
- Every gate in GATES runs on GPU. A replay of the new route over the 83
  turns and the held-out turns: routing, repair rate, changed calls mapped to
  failed angles, no jump to submission, no lost call, effort, p50/p90.

## 5. Records

VCO-D18 in `docs/design/example-verified-checklist-orchestration.md`
(supersedes VCO-D17's unverified max call), `PROGRESS.md`, the example's
`README.md` and `MEASUREMENTS.md`.

## Order

1. Branch from main, this plan, draft PR.
2. Sections 1, 2 and 5; ruff and the CPU suite.
3. Calibration (section 3) -> report -> owner decides thresholds.
4. Thresholds in the YAML -> all GPU gates and the replay -> `MEASUREMENTS.md`.

## Risks

- Jev reads tool calls as `<tool_call>` markup in the reply text (once read
  a real call at p 0.74). Calibration exposes it; passing them structured is a
  separate framework decision, not in this plan.
- Labels are Claude hindsight labels grounded in execution results and task
  outcomes, not human labels.
- Calibration uses recorded max-effort replies; serving drafts use the
  caller's effort. The replay checks the repair rate on serving drafts.
