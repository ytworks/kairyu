# Qwen requirements reads the conversation without replayed reasoning

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641).
Status: approved by the owner (2026-10-07); supersedes the earlier unapproved
draft of this file.

## 1. Problem

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): on 87 of 160
VERIFIED turns `judgments` failed with `checklist_unavailable` and Winnow
received nothing. 85 of them: the Qwen `requirements` request was rejected
(HTTP 400, input 199,082-325,598 tokens + 65,536 output > 262,144).

Cause: mini-swe-agent replays each earlier assistant turn with its
`reasoning_content`, which is Kairyu's own stage report
(`expose_intermediate_outputs: true`). `validate_orchestration_chat_input`
copies it into the L2 `{conversation}` (`_message_wire_shape`), so the report
is ~80% of the conversation (14:42:49 UTC turn: 189,953 of 240,650 Qwen
tokens). Responses AUTO already drops replayed reasoning for this reason
(m11 D4, `responses_service.py` `replay_reasoning=not orchestrated`); Chat
Completions does not.

## 2. Change

The owner's correction (2026-10-07): only Qwen and Winnow do without the
replayed reasoning; DeepSeek needs it. A uniform drop at the Chat
Completions boundary (commit a3420252) was reverted.

- Framework (`kairyu/orchestration/request.py`, `conductor.py`): a role
  placeholder `{conversation_without_reasoning}` renders `{conversation}`
  without assistant `reasoning_content`. `{conversation}` is unchanged.
- Example (`verified.yaml`, `verified-always.yaml`): Qwen `requirements`
  reads `{conversation_without_reasoning}`. DeepSeek `drafts`
  (`{conversation}`) and `answer` (`{query}`) keep the reasoning. Winnow
  `judgments` reads `request` (system/developer + latest user), which holds
  no assistant turn.
- Docs: m1 D8 and VCO-D19 amendments; `PROGRESS.md` entry.

Framework admission: (1) a role cannot read the conversation without
replayed reasoning; `Conductor._render` fixes `{conversation}`; (2) no
extension point renders a role's conversation differently; (3) any DAG that
mixes a small-context worker with exposed intermediates and a multi-turn
client overflows the same way; (4) one placeholder; which role uses it
stays in the example.

## 3. Expected effect (measured)

Without `reasoning_content`, the 67 failed turns measured 30,330-96,837 Qwen
tokens; + 65,536 output ≤ ~162K < 262,144. Three such replays returned valid
11-13 point lists. DeepSeek prompts are unchanged.

## 4. Test

One conductor test (`tests/unit/test_conductor.py`): over a conversation
with a replayed `reasoning_content`, `{conversation}` keeps it and
`{conversation_without_reasoning}` drops it while keeping the messages.

## 5. Verification

1. Local: ruff and pytest for the changed paths.
2. GPU: deploy; replay r1's failed turns with the production request
   assembly — `requirements` succeeds and `judgments` reaches Winnow; re-run
   all 9 gates.

## 6. Out of scope

- DeepSeek reads the full conversation; ~290+ messages may exceed its
  1,048,576 tokens (estimate, ~3.4K tokens per message).
- Conversations whose body alone exceeds ~196K Qwen tokens still overflow
  `requirements` (publish unverified); fixing that needs compaction or a Qwen
  context change — a separate owner decision.
- The 2 short `requirements` outputs (not reproducible on replay).
- The Winnow route judge (`profile_judge`, `Orchestrator._systemone_judge_body`)
  still sends replayed `reasoning_content` uncut inside its 120,000-character
  bound; reported to the owner, not changed here.
