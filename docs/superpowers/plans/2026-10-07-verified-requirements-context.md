# Winnow judges every turn of a long agent conversation (generic fix)

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19. Status: awaiting owner approval.

## 1. Problem and goal

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): Winnow judged 61
of 132 VERIFIED turns; on 71 no Winnow request was sent. Goal: on every turn
of any conversation, every judgment that can be asked is asked — in this
example and in any orchestration built the same way.

## 2. Causal chain (measured) and the generic defect behind each link

| Link | Measured | Generic defect |
|---|---|---|
| L1 | No `requirements` list → `judgments` sends nothing, not even the 5 adoption questions that need only the request and the drafts (`checklist._pending_questions` raises for the whole checklist when one `foreach` source is missing) | A checklist is all-or-nothing on its item sources |
| L2 | Qwen gave no list on 69 turns: 67 rejected by vLLM (input 199,082-325,598 + 65,536 cap > 262,144), 2 empty on 148K / 181K inputs (valid on 4 replays each) | — (consequence of L3/L4) |
| L3 | 81 % of Qwen's input is earlier turns' `reasoning_content`: Kairyu's stage report, echoed by the agent and copied into the L2 conversation by `validate_orchestration_chat_input` (14:42 turn: 240,650 → 46,339 Qwen tokens without it; DeepSeek 241,080 → 43,326). Responses AUTO already drops replayed reasoning (m11 D4) | The orchestrated Chat Completions conversation re-reads every earlier stage report |
| L4 | Even without L3, conversations outgrow a smaller worker: ~0.77K tokens per message, Qwen holds ~255 messages; 09/12 DeepSWE conversations: median 298, max 714 messages (77/113 problems above 255) | A role's `{conversation}` is rendered whole even when its worker cannot hold it; the request then fails |
| L5 | Winnow's input on judged turns ≤ 15,115 tokens (context 65,536) | none |

## 3. Changes (all generic, framework; no example change)

| # | Fixes | Layer | File | Change |
|---|---|---|---|---|
| G1 | L3 | L3→L2 | `kairyu/entrypoints/server/chat_service.py` (`validate_orchestration_chat_input`) | The L2 conversation omits assistant `reasoning_content`. Still accepted; still passed to direct engines' templates; Kairyu still emits the report. Same rule as Responses AUTO |
| G2 | L1 | L2 | `kairyu/orchestration/checklist.py` (`_pending_questions`, `judge`), `conductor.py` (trace) | A `foreach` source without a usable list skips only the questions bound to it; every other question is still asked. The verdict and trace record the skipped questions (`unjudged: <source>`); with nothing left to ask, behaviour is as today |
| G3 | L4 | L2 | `kairyu/orchestration/conductor.py` (`_render`) | When a role's prompt would not fit its worker (`max_model_len` − the role's `max_tokens`), `{conversation}` / the `{query}` transcript is rendered with the existing `bounded_conversation` (system and first messages incl. the task, then the newest that fit; omitted count shown) at the size that fits; roles that fit are rendered whole, as today |

Framework admission:

- G1: (1) contract broken: orchestrated conversations re-read every earlier
  stage report; (2) no extension point: built before roles apply; (3) any
  orchestrator exposing intermediate outputs behind an echoing client
  (LiteLLM agents) grows every role's input ~12K tokens per turn; (4) one
  field omitted in one place, the Responses AUTO rule.
- G2: (1) one missing source discards independent questions; (2) the
  checklist is framework code; (3) any checklist mixing static and per-item
  questions loses all of them on one failed upstream role; (4) skip per
  question, report what was skipped.
- G3: (1) a role's request exceeds its worker and fails; (2) rendering is
  Conductor code; the route judge and checklist state already bound with the
  same function; (3) any orchestration mixing workers of different context
  sizes fails on its smallest worker as conversations grow; (4) reuse
  `bounded_conversation`, only when the request would not fit.

## 4. Coverage

| Case | Before | After |
|---|---|---|
| Echoed reports push Qwen over its context (67 turns) | no judgment | G1: 30,330-96,837 tokens; requirements and all judgments |
| Empty list on report-filled input (2 turns) | no judgment | G1 input (replays valid); if empty again, G2: adoption still judged |
| Conversation beyond Qwen's context (long problems) | requirements fails | G3: Qwen reads the task and the newest messages that fit; all judgments |
| Qwen down | no judgment | G2: adoption judged |
| Any other orchestration with the same shapes | same failures | same fixes |

## 5. Tests (one per generic contract)

- G1: chat input with assistant `reasoning_content` → L2 prompt without it;
  content and tool calls kept.
- G2: checklist with a static question and a `foreach` question whose source
  failed → the static question is asked, the other reported unjudged.
- G3: a role whose worker's `max_model_len` is smaller than the conversation
  → bounded conversation with the task, newest messages and omitted count;
  a role on a large worker → whole conversation.
- Example test: unchanged except the assertion that a failed `requirements`
  still yields the adoption read.

## 6. Docs

m11 (G1, amending the 2026-08-14 assistant-history amendment), m1 D8 (G2),
m1 (G3), VCO-D19 note, `PROGRESS.md`.

## 7. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: ruff; changed-path tests | green | 10 min |
| 2 | Live replay of the 69 turns that lost judgments, new code, live Qwen | 69 valid lists, 0 empty | 40 min |
| 3 | Live replay of the 714-message 09/12 conversation | Qwen request fits (G3), valid list | 10 min |
| 4 | Redeploy (`./run.sh`) | healthy | 15 min |
| 5 | All nine gates | existing criteria | 2 h |
| 6 | New gate `long-conversation`: five r1 turns that lost judgments + the 714-message conversation, through `kairyu-verified-always` | 200; requirements and all judgments on all six; the stage report still emitted | 40 min |
| 7 | Qwen stopped, one verified request | 200; adoption judged, coverage reported unjudged | 10 min |

Stop and report at the first failure.

## 8. Decision requested

Approve G1-G3.
