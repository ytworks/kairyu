# L2 drops replayed reasoning on Chat Completions

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

## 2. Change (one)

- `kairyu/entrypoints/server/chat_service.py`: the L2 conversation built by
  `validate_orchestration_chat_input` omits assistant `reasoning_content`.
  Direct-engine chat templates keep it. No example, DSL, checklist or L1
  change.
- `docs/design/m11-product.md`: amendment — the 2026-08-14 assistant-history
  field is preserved for direct engines only; L2 drops it as Responses AUTO
  does. `PROGRESS.md` entry.

Framework admission: (1) Chat Completions L2 conversation grows by every
replayed stage report; code path above; (2) no extension point — message
shaping is server code; (3) any orchestrated model with exposed intermediates
and a multi-turn client (e.g. the tiered Chat UI) regresses the same way;
(4) drop one field at the L2 boundary, matching the existing Responses rule.

## 3. Expected effect (measured)

Without `reasoning_content`, the 67 failed turns measured 30,330-96,837 Qwen
tokens; + 65,536 output ≤ ~162K < 262,144. Three such replays returned valid
11-13 point lists. DeepSeek prompts shrink alike (241,080 → 43,326 tokens on
the 14:42:49 turn).

## 4. Test

Extend the existing Chat round-trip test
(`tests/server/test_orchestration_usage_trace.py`) to run through the
orchestration chat path and assert the second request's L2 prompts carry the
earlier answer but not the replayed stage report. No new test.

## 5. Verification

1. Local: ruff and pytest for the changed paths.
2. GPU: deploy; replay r1's failed turns with the production request
   assembly — `requirements` succeeds and `judgments` reaches Winnow; re-run
   all 9 gates.

## 6. Out of scope

- Conversations whose body alone exceeds ~196K Qwen tokens still overflow
  `requirements` (publish unverified); fixing that needs compaction or a Qwen
  context change — a separate owner decision.
- The 2 short `requirements` outputs (not reproducible on replay).
