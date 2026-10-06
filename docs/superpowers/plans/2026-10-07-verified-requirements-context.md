# Verified route: Winnow does not judge on long conversations

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19 (`2026-10-06-winnow-verified-three-wave.md`). Status:
awaiting owner approval.

## 1. Problem

In DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`) Winnow judged
61 of 132 VERIFIED turns. On the other 71 no Winnow request was sent at all
(`judgments` → `checklist_unavailable`, 0.0 s), so the answer was written
without any judgment. From about turn 20 of each problem, Winnow stops.

## 2. Causal chain (each link confirmed)

| Link | What happens | Evidence |
|---|---|---|
| L1 | `judgments` is one checklist: 5 adoption questions (need only the request and the drafts) + 5 x N coverage questions (one per item of `requirements`' list). Building the questions reads that list first; a missing or unparsable list raises `checklist_unavailable` for the whole checklist, so not even the 5 adoption questions reach Winnow. | `checklist._pending_questions` → `_source_items`; trace: failed judgments take 0.0 s, no Winnow read |
| L2 | `requirements` (Qwen) returned no list on 69 turns: 67 rejected by Qwen, 2 empty. | trace v2 |
| L3 | Rejected because Qwen's input (199,082-325,598 tokens on the 67 turns) + the 65,536 output cap exceeds Qwen's 262,144 context. | live replay: vLLM 400 "maximum context length is 262144 tokens … at least 196609 input tokens"; live `/tokenize` of Kairyu's exact request |
| L4 | The input is that large because the L2 conversation keeps every earlier assistant `reasoning_content`, which here is Kairyu's own stage report (five drafts, reasoning, judgments) echoed by the agent: 189,953 of 240,650 tokens on the 14:42 turn. Responses AUTO already drops replayed reasoning for this reason (m11 D4); Chat Completions keeps it (`validate_orchestration_chat_input`). | `/tokenize` of each part; code path |
| L5 | The 2 empty lists: Qwen ended inside its reasoning after 188 / 342 tokens on 148K / 181K report-filled inputs; the same inputs replayed 4 times all returned valid lists. | live replay |

Without L4 the 67 turns' Qwen input is 30,330-96,837 tokens (median 75,815):
all fit, and the three replayed returned valid lists (11-13 points).

## 3. Fix

| # | Link | Layer | Owner | Change |
|---|---|---|---|---|
| A | L1 | L2 | example (`verified.yaml`, `verified-always.yaml`) | Split `judgments` into two checklist verifiers: `adoption` verifies `drafts` (the 5 adoption questions; state: request + drafts) and `coverage` verifies `requirements` (the 5 x N questions; state: request + drafts, waiting for `drafts` as PR #641's F1 allows). `answer` reads `{adoption}` and `{coverage}`. Winnow then judges every draft's adoptability on every VERIFIED turn, whatever happens to Qwen. |
| B | L3, L4, L5 | L3→L2 | framework (`kairyu/entrypoints/server/chat_service.py`) | The orchestration Chat Completions conversation omits assistant `reasoning_content` (still accepted; direct engines still receive it), as Responses AUTO does. Every role's input loses the echoed reports; Qwen's input on the failed turns falls to ≤ 96,837 tokens. |

Framework admission for B: (1) broken contract — an orchestrated
conversation re-reads every earlier stage report, code path above; (2) no
extension point — the conversation is built before roles apply; (3)
independent regression — any orchestrator with
`expose_intermediate_outputs` behind an agent client that echoes
`reasoning_content` (LiteLLM, mini-swe-agent) grows every role's input each
turn; (4) smallest mechanism — omit one field in one place; Responses AUTO
already follows the rule.

Unchanged: Kairyu still emits the stage report to the caller; prompts,
caps, efforts, routing, L1.

Residual: a conversation whose own content exceeds Qwen (~196K tokens,
roughly 250 messages at the measured ~0.8K tokens per message without
reports) still loses `coverage` on that turn; `adoption` still runs (A).

## 4. Tests

- B: one chat-input test — assistant `reasoning_content` is absent from the
  L2 prompt; content and tool calls stay.
- A: the example test — with Qwen failing, Winnow still receives the 5
  adoption questions and `answer` reads them; with Qwen succeeding, both
  reads happen (replaces the one-read assertion).

## 5. Docs

m11 assistant-history amendment (B); VCO-D19 amendment (A); `PROGRESS.md`.

## 6. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: ruff; changed-path tests | green | 10 min |
| 2 | Live replay of the 69 failed turns' `requirements` requests rendered by the new code | all accepted; valid lists; 0 empty | 40 min |
| 3 | Redeploy (`./run.sh`) | healthy | 15 min |
| 4 | All nine gates (verified-route expects two Winnow reads: 5 adoption + 5 x N coverage) | existing criteria | 2 h |
| 5 | New gate `long-conversation`: five r1 turns that lost judgments, through `kairyu-verified-always` | 200; adoption and coverage both judged on all five | 20 min |
| 6 | Qwen stopped: one verified request | 200; adoption judged; coverage absent | 10 min |

Stop and report at the first failure.

## 7. Decision requested

Approve A (example) and B (framework).
