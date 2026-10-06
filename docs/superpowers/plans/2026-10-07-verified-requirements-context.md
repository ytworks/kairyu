# Verified route: Winnow stops judging because `reasoning_content` is re-read

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19 (`2026-10-06-winnow-verified-three-wave.md`). Status:
awaiting owner approval.

## 1. Problem

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): Winnow judged 61
of 132 VERIFIED turns. On the other 71 no Winnow request was sent
(`judgments` → `checklist_unavailable` in 0.0 s) and the answer was written
without judgments. From about turn 20 of each problem Winnow stops.

## 2. Causal chain (each link measured)

1. `judgments` builds its questions from `requirements`' list; with no list
   the whole checklist is unavailable and no Winnow request is sent
   (`checklist._pending_questions` → `_source_items`).
2. `requirements` (Qwen) gave no list on 69 turns: 67 rejected by Qwen
   (vLLM 400: "maximum context length is 262144 tokens … at least 196609
   input tokens"; inputs 199,082-325,598 tokens + 65,536 output cap), 2
   ended empty inside Qwen's reasoning on 148K / 181K inputs (the same inputs
   replayed 4 times all returned valid lists).
3. Qwen's input is that large because of `reasoning_content` (section 3).

## 3. `reasoning_content`: where it comes from and where it goes

| Step | What happens | Evidence |
|---|---|---|
| Emitted | With `expose_intermediate_outputs: true` the Conductor returns each turn's stage report in the assistant message's `reasoning_content`: "Final answer attribution", then per stage (drafts, requirements, judgments, answer) the model's reasoning and the stage output. Median 11,815 Qwen tokens per turn (max 14,536) | recorded responses; `/tokenize` |
| Echoed | The agent (mini-swe-agent via LiteLLM) keeps each assistant message as returned, `reasoning_content` included, and sends it back in the next request | recorded requests: 18 of the 14:42 turn's assistant messages carry it |
| Accepted | Chat Completions accepts the field (m11 assistant-history amendment, 2026-08-14: kept "for key-sensitive model templates and L2 conversation history") | `protocol.ChatMessage` |
| Re-read | `validate_orchestration_chat_input` copies every message field into the L2 conversation JSON; every role's `{conversation}` / `{query}` renders it | `chat_service.py` |
| Size | 14:42 turn (61 messages): L2 conversation 907,608 characters = 240,650 Qwen / 241,080 DeepSeek tokens; without `reasoning_content` 117,040 characters = 46,339 Qwen / 43,326 DeepSeek tokens. The reports are 81 % of what every role reads | `/tokenize` on both L1s |
| Readers | `requirements` (Qwen, 262,144) overflows; `drafts` and `answer` (DeepSeek, 1,048,576) read 5.6x more than the conversation itself, every turn, and approach their own limit as turns grow; Winnow's state uses only the request (system + latest user message) and is unaffected | role prompts; checklist state config |
| Precedent | Responses AUTO already drops replayed reasoning: "stage output would grow every L2 prompt" (m11 D4); Chat Completions was never aligned | `docs/design/m11-product.md` |

The report is output for the caller (UIs show it). It is not an input any
role asks for: every earlier answer, tool call and tool result is already in
the conversation as content.

## 4. Fix

| # | Layer | Owner | File | Change |
|---|---|---|---|---|
| 1 | L3→L2 | framework | `kairyu/entrypoints/server/chat_service.py` (`validate_orchestration_chat_input`) | Build the L2 conversation without assistant `reasoning_content`. The field is still accepted on the wire, still reaches direct (non-orchestrated) engines' chat templates, and Kairyu still emits the stage report to the caller. |

Framework admission: (1) broken contract — an orchestrated conversation
re-reads every earlier stage report (code path above); (2) no extension
point — the conversation is built before any role or example config
applies; (3) independent regression — any orchestrator with
`expose_intermediate_outputs` behind a client that echoes
`reasoning_content` (LiteLLM-based agents) grows every role's input by ~12K
tokens per turn; (4) smallest mechanism — omit one field in one place, the
rule Responses AUTO already follows.

No example change: prompts, caps, efforts, DAG, routing, Winnow, L1 stay.

Effect (measured on the r1 turns that failed): Qwen's input 199,082-325,598
→ 30,330-96,837 tokens; all 67 fit with the 65,536 cap; the three replayed
returned valid lists (11-13 points); DeepSeek's drafts/answer input falls by
the same factor (241,080 → 43,326 on the 14:42 turn).

## 5. Considered, not included

| Option | Why not |
|---|---|
| Keep `reasoning_content`, enlarge Qwen (YaRN 1M) | Keeps feeding 81 % unused text to every role; DeepSeek still re-reads it |
| Keep it, lower Qwen's output cap | Input alone exceeds Qwen on 38 of 67 turns |
| Keep it, bound what Qwen reads | Cuts real messages while keeping the reports |
| Stop emitting the report | The caller-facing output is not the defect |
| Make `judgments` survive a missing list (split into two verifiers) | Changes the verified DAG; with the fix the list is produced on every recorded turn |

## 6. Tests

One chat-input test: a request whose assistant history carries
`reasoning_content` yields an L2 prompt without it, with the assistant
content and tool calls intact. Report base/head collection counts.

## 7. Docs

- `docs/design/m11-product.md`: amend the 2026-08-14 assistant-history
  amendment — the L2 conversation omits `reasoning_content`, as Responses
  AUTO does.
- VCO-D19: note the multi-turn finding and the fix.
- `PROGRESS.md`: Change Log entry and Current Status line.

## 8. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: `uv run ruff check .`; `tests/server/test_openai_api.py` and the new test's file | green | 10 min |
| 2 | Live replay of the 69 turns that lost judgments: Kairyu's new `requirements` request for each, sent to Qwen | all accepted; 69 valid lists; 0 empty | 40 min |
| 3 | Redeploy (`./run.sh`) | healthy | 15 min |
| 4 | All nine gates in GATES order | existing criteria | 2 h |
| 5 | New gate `long-conversation`: five r1 turns that lost judgments, through `kairyu-verified-always` | 200; requirements and judgments succeed on all five; the response still carries the stage report in `reasoning_content` | 20 min |

Stop and report at the first failure.

## 9. Limits

A conversation whose own content exceeds Qwen (~0.8K tokens per message:
about 250 messages) would again lose `requirements` on that turn.

## 10. Decision requested

Approve change 1 (framework) with this verification.
