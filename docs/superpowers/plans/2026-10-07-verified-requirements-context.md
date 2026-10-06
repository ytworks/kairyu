# Verified route: keep `requirements` inside Qwen's context

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641).
Follows `2026-10-06-winnow-verified-three-wave.md` (VCO-D19), which is
GPU-verified on single-turn requests. Status: awaiting owner approval.

## 1. Goal

On multi-turn agent conversations, every VERIFIED turn must run all three
waves: Qwen must always receive a `requirements` request that fits its
context, so Winnow can judge the drafts. Today long conversations make that
request exceed Qwen's 262,144-token context, and the turn is answered without
requirements and judgments.

## 2. Owner requirements for this fix

- The cause is Qwen and its token limit.
- The fix is minimal and does not widen the problem.
- Kairyu keeps emitting each turn's stage report in `reasoning_content`, and
  the caller keeps sending earlier turns back; that is expected behaviour,
  not the defect.
- Stay on the `reasoning_content` mechanism that produces the overflow.
- Plan only; no benchmark work in this plan.

## 3. Facts (DeepSWE r1 `deepswe-verified-3wave-full-4w-20261006-r1`, recorded turns)

| Fact | Value | How measured |
|---|---|---|
| VERIFIED turns / with judgments / without | 132 / 61 / 71 | trace v2 of every recorded call |
| `requirements` failed (no list) | 67 turns + 2 empty lists | trace v2 |
| Qwen input on the 67 failed turns | 199,082-325,598 tokens (median 267,676) | Kairyu's exact rendered request, live Qwen `/tokenize` |
| Qwen's reply on an over-long turn | HTTP 400 "maximum context length is 262144 tokens … 65536 output tokens … at least 196609 input tokens" | live replay (Qwen's access log is off, so it never appeared in its log) |
| Composition of one L2 conversation (14:42 turn, 61 messages) | 240,650 tokens = earlier assistant `reasoning_content` 189,953 + message content 33,852 + tool calls 3,845 + JSON framing | `/tokenize` of each part |
| What that `reasoning_content` is | Kairyu's own stage report (five drafts, Qwen's reasoning, judgments, attribution), returned by mini-swe-agent in the history | recorded messages |
| Same 67 turns without earlier `reasoning_content` | 30,330-96,837 tokens (median 75,815); all fit with the 65,536 cap; Qwen returns valid lists (3 replayed: 11-13 points) | live `/tokenize` and replay |
| `requirements` output when it succeeds | max 3,486 tokens, median 1,899 | trace v2 (61 calls) |
| The two empty lists (148K / 181K inputs) | Qwen ended inside its reasoning after 188 / 342 tokens; the same inputs replayed 4 times all returned valid lists | live replay |
| Longest agent conversation seen on this host | 714 messages, 925,539 content characters (09/12 DeepSeek-only run); ~2.5 JSON characters per Qwen token → ~370K tokens of content alone | trajectory files |

Code path: `validate_orchestration_chat_input`
(`kairyu/entrypoints/server/chat_service.py`) copies every message field,
including assistant `reasoning_content`, into the L2 conversation JSON; every
role's `{conversation}` / `{query}` renders that JSON. Responses AUTO already
drops replayed reasoning for exactly this reason (m11 D4: "AUTO drops
replayed reasoning: stage output would grow every L2 prompt"); Chat
Completions does not (m11 assistant-history amendment, 2026-08-14, keeps it
"for key-sensitive model templates and L2 conversation history").

## 4. Causes

1. **Echoed stage reports in the L2 conversation.** Each turn's report
   (~6K tokens) comes back in the history and is re-read by every later
   turn; it is ~79 % of Qwen's input on long turns.
2. **Qwen's context.** Input + the 65,536 output cap exceeds 262,144 →
   vLLM 400 → no list → no judgments.
3. **Long conversations' own content.** Without (1), a very long agent run
   (hundreds of messages) can still exceed Qwen by content alone.
4. **Two empty lists.** Not reproducible on the same inputs; seen only on
   report-polluted 148K-181K inputs. Expected to disappear with (1); checked
   in verification, not fixed separately.

## 5. Options considered

| Option | Result against the requirements |
|---|---|
| Lower the `requirements` output cap | Fits only 29 of 67 even at 4,096; leaves the cause in place. Rejected |
| YaRN 1M for Qwen | Changes the model's behaviour for every input (static YaRN harms short texts per the model card) to carry text Qwen does not need. Rejected by owner |
| Bound what `requirements` reads, only | Cuts real earlier messages (the task's progress) while keeping the echoed reports. Rejected by owner |
| Stop exposing stage reports (`expose_intermediate_outputs: false`) | Removes output the product shows. Rejected (requirement) |
| Run `requirements` on DeepSeek | Changes the owner's design (Qwen lists requirements). Rejected |
| Drop echoed `reasoning_content` per role only | Leaves every other role (DeepSeek drafts and answer) re-reading 3-5x more text each turn, and needs a new per-role switch for what is one L2 rule. Rejected |
| **Drop echoed assistant `reasoning_content` from the L2 conversation (F2)** | Removes cause 1 at the point it enters; output to the caller unchanged; same rule as Responses AUTO. **Chosen** |
| **Bound `{conversation}` per role with the existing `bounded_conversation` (F3)** | Only for cause 3, only above a size Qwen cannot read at all; same mechanism as the route judge and checklist state. **Chosen, as a guard** |

## 6. Changes

| # | Layer | Owner | File | Change | Cause |
|---|---|---|---|---|---|
| F2 | L3→L2 | framework | `kairyu/entrypoints/server/chat_service.py` (`validate_orchestration_chat_input`) | the L2 conversation omits assistant `reasoning_content`; the field is still accepted and still reaches direct engines' chat templates | 1, 2, 4 |
| F3 | L2 | framework | `kairyu/orchestration/conductor.py` (`_render`), `kairyu/dsl/spec.py`, `kairyu/dsl/loader.py` | role option `max_conversation_chars`: that role's `{conversation}` is rendered with `bounded_conversation` (system and first messages incl. the task, then the newest that fit; omitted count shown) | 3 |
| E | L2 | example | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/verified.yaml`, `verified-always.yaml` | `requirements`: `max_conversation_chars: 400000` (~160K Qwen tokens at the worst measured 2.5 chars/token; + 1.5K template + 65,536 output < 262,144) | 3 |

Nothing else changes: prompts, caps, efforts, DAG, routing, Winnow, the
stage report Kairyu emits, L1.

## 7. Framework admission

F2:
1. Broken contract: an orchestrated Chat Completions conversation re-reads
   every earlier stage report; code path in §3.
2. No extension point: the L2 conversation is built before any role or
   example config applies.
3. Independent regression: any orchestrator with
   `expose_intermediate_outputs` called by a client that echoes
   `reasoning_content` (LiteLLM-based agents, mini-swe-agent) grows each
   role's input by the report of every earlier turn.
4. Smallest mechanism: omit one field at one place; Responses AUTO already
   applies the rule. Policy (whether to expose reports) stays in examples.

F3:
1. Missing contract: a role cannot bound the conversation it renders, while
   the route judge (`max_conversation_chars`) and checklist state
   (`max_total_chars`) can.
2. No extension point: `{conversation}` is rendered by the Conductor.
3. Independent regression: a role on a smaller-context worker in any
   orchestrator fails once the conversation outgrows that worker.
4. Smallest mechanism: reuse `bounded_conversation`; the limit value stays
   in the example.

## 8. Tests (test policy: A input → C observable result)

- F2: one test in the chat-input tests: a request whose assistant history
  carries `reasoning_content` produces an L2 prompt without it, with the
  assistant content and tool calls intact.
- F3: one conductor test: a role with `max_conversation_chars` receives the
  task and the newest messages and the omitted count; a role without it
  receives the whole conversation.
- Example: none added (the bound is configuration; its effect is checked on
  GPU).
- Report base/head collection counts.

## 9. Docs

- m11 assistant-history amendment: amended for the L2 path (F2).
- m1: the role option (F3).
- VCO-D19: note that `requirements` reads at most 400,000 characters.
- `PROGRESS.md`: one Change Log entry; Current Status line.

## 10. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: `uv run ruff check .`; changed-path tests | green | 10 min |
| 2 | Live replay, before redeploying: the 67 failed turns' `requirements` requests rendered by the new code, sent to Qwen (4 at a time) | all 67 accepted; 67 valid lists; 0 empty | 30 min |
| 3 | Live replay of the 09/12 longest conversation (925,539 characters) | rendered ≤ 400,000 characters, task message kept, accepted, valid list | 5 min |
| 4 | Redeploy with plain `./run.sh` (gateway restarts) | healthy, L1 probes pass | 15 min |
| 5 | All nine GPU gates in GATES order (l1, routing, think-route, effort, verified-route, fallback, serving, serving-routed, browser) | each gate's existing criteria | 2 h |
| 6 | New gate `long-conversation`: five r1 turns whose old Qwen input exceeded 262,144, through `kairyu-verified-always` | 200; requirements and judgments succeed on all five | 20 min |

Stop and report at the first failure.

## 11. Risks and limits

- Roles no longer see earlier turns' stage reports in their input; the
  earlier answers, tool calls and tool results stay. The reports are still
  emitted to the caller.
- F3 truncates the middle of a conversation only above 400,000 characters,
  only for `requirements`; DeepSeek roles and Winnow read as before.

## 12. Decision requested

Approve F2, F3 and E as above.
