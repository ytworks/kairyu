# Jev-verified DeepSeek answers: step criteria, sound repairs, minimal framework

Replaces PR #618 (closed). Owner-approved plan, 2026-10-03.

## Goal

The L2 mechanism in which OpenJev (Jev) guarantees DeepSeek-V4.1 answers must
work on agent turns (DeepSWE) as well as on single answers, and raise the
DeepSWE score: at least never below the think route (DeepSeek directly). The
framework keeps only what that mechanism needs; the example owns prompts,
thresholds, routing and workflow.

## Evidence (PR #618 runs r3, r4 and the full run; 83 verified turns)

- 73/83 turns ended `refinement_limit` (two repairs, not accepted); 4/83 guaranteed.
- 0/83 replies carried text: the Chat API nulls `content` when tool calls exist.
  The same draft read 0.05-0.47 at the acceptance read without its text,
  0.68-0.999 with it.
- 51/74 repaired turns changed the agent's commands; 9 lost calls. One repair
  replaced an exploring draft with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,
  which was published (reward 0).
- 6/41 turns of long conversations had an empty implicit list (unverified).
- Verified turn p50 153-240 s (think 17-22 s); the latency requirement is
  p50 <= 180 s (VCO-D12).

## Pillars

A. **Criteria for a step (central).** An agent turn is one step of solving a
   task, not a finished answer. Criterion (A0): "given the conversation and the
   latest tool results, is this reply a sound next step toward solving the
   task?" (uses the latest results correctly; moves the task forward; no wrong
   premise or destructive action; no completion claim or submission while the
   work is unfinished). Answers without tools keep today's criterion.
   Realised in the example only: two verified profiles (`verified_answer`,
   `verified_step`); the Jev route judge gains a STEP choice (tools offered and
   the reply is the next action of an ongoing task). `verified_step` extracts
   points for this step, judges coverage and acceptance as a step, and reads
   the request, the recent conversation verbatim (bounded `query`), the
   history summary and the answer. STEP thresholds come from labelled turns.
B. **No excessive rewriting.** A repair is the same message written again:
   the draft's frame (conversation first, every adopted point), then the draft
   and the clearly unmet points (p < 0.5), changing only what those points
   require. No repair without an unmet point (B3). Measured: a changed call
   must map to an unmet point; no lost calls; no submission introduced.
C. **Other failures.** The Chat API returns text with tool calls (L3 bug);
   long conversations never empty an extractor; latency p50 <= 180 s.
D. **Minimal framework** across #614, #616 and #618 (below).

## Framework: keep / add / remove

Keep or add: System One client, deploy `systemone:` and `systemone_ref`;
checklist verifier core; single-list curation; internal grammars;
`{conversation}` / `{response_format}` / `{tools}`; conversation bounds; the Jev
route judge; `{tools}` in admission bounds; Chat API text with tool calls;
no guarantee for an empty answer; `request` state source; the acceptance read
(with its accounting) and B3.

Remove or not carry over: rule-based checks; `seed_from` and its #617 seed
fixes (the generator becomes the final role); question batching changes
(`max_questions_per_call: 256` in the example); multi-list curation; empty
checklist rules (`on_empty`); judging a "published form" (tool_call_markup and
related); the node-cache umask fix (separate PR). The public System One API
and the OpenJev one-GPU example stay unchanged.

## Stages (one draft PR, pushed stage by stage)

- S0 branch from main, draft PR, close #618.
- S1 L3: Chat API returns text with tool calls (+ failing-first tests).
- S2 L2 framework minimisation (+ tests; removed features lose their tests).
- S3 example: profiles, STEP routing, step prompts and state, repair prompt,
  extractor limits.
- S4 docs; Codex review until no actionable finding.
- S5+ replay of the 83 recorded turns (all pillar measures) -> labels and
  STEP threshold (owner decides from the error/miss table) -> GPU gates ->
  DeepSWE subset, think vs verified (verified >= think, no early submission)
  -> full run on the owner's go.
