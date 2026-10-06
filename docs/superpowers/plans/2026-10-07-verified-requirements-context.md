# Exact multi-turn conversations beyond a worker's context (L2 only)

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19. Status: awaiting owner approval.

## 1. Problem and goal

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): Winnow judged 61
of 132 VERIFIED turns; on 71 no Winnow request was sent.

Goal, for any orchestration and any public API (Chat Completions, Responses,
Messages):

- every role works on every turn however long the conversation grows;
- every earlier turn's information, including `reasoning_content`, stays
  available to every role (exact multi-turn exchanges);
- no dependence on the client (Codex-style client compaction exists only on
  Responses; mini-swe-agent and other Chat Completions clients never compact).

## 2. Causal chain (measured)

| Link | Measured | Defect |
|---|---|---|
| K1 | No `requirements` list → `judgments` sends nothing, not even the 5 adoption questions that need only the request and the drafts (`checklist._pending_questions` raises for the whole checklist when one `foreach` source is missing) | a checklist is all-or-nothing on its item sources |
| K2 | Qwen gave no list on 69 turns: 67 rejected by vLLM (input 199,082-325,598 + 65,536 cap > 262,144), 2 empty on 148K / 181K inputs (valid on 4 replays each) | consequence of K3 |
| K3 | Every role renders the whole conversation. It grows ~3.4K tokens per message with each turn's stage report (240,650 tokens at 61 messages, 320,460 at 95): Qwen (262,144) overflows from ~60 messages, DeepSeek (1,048,576) from ~290; the 09/12 DeepSWE conversations reached median 298 and max 714 messages | a role's input is unbounded while its worker's context is fixed; Kairyu has no server-side way to carry a conversation beyond a worker's context, so the request fails |
| K4 | Winnow's input on judged turns ≤ 15,115 tokens (context 65,536) | none |

## 3. Changes

| # | Fixes | Layer | Owner | Change |
|---|---|---|---|---|
| C1 | K3 | L2 | framework (`kairyu/orchestration/compaction.py` new, `conductor.py`, `dsl/spec.py`, `dsl/loader.py`) | **Context-fit compaction.** Before a role is dispatched, if its rendered request would exceed its worker (`max_model_len` − the role's `max_tokens`), Kairyu replaces the oldest messages of that role's conversation with a summary and keeps the system/developer messages, the first task message and the newest messages verbatim. The summary is written by the orchestrator's `compaction` worker over the omitted messages (content, tool calls, tool results and `reasoning_content`); it is cached by the hash of the omitted messages and extended incrementally (previous summary + newly omitted messages), so later turns and other roles reuse it and no summarization call ever exceeds its own worker. Compaction leaves headroom (compacts down to half the budget) so it runs every few turns, not every turn. A role that fits reads the whole conversation as today. Trace v2 records per role: compacted, omitted messages, summary tokens, cache hit. Works on the L2 conversation, so Chat Completions, Responses and Messages behave the same |
| C3 | K1 | L2 | framework (`checklist.py`, `conductor.py` trace) | A `foreach` source without a usable list skips only the questions bound to it; every other question is asked; skipped questions are recorded as unjudged |
| E | — | L2 | example (`verified.yaml`, `verified-always.yaml`) | `compaction: {worker: deepseek, reasoning_effort: low, max_tokens: 32768, prompt: …}` — the policy: which worker summarizes and how |

No change to Qwen's L1, the roles' prompts, caps, efforts, DAG, routing,
Winnow or the emitted stage report.

## 4. Framework admission

- C1: (1) contract missing — a served conversation that outgrows a worker
  makes the request fail; code path: `Conductor._render` renders the whole
  conversation; (2) no extension point — rendering and dispatch are
  Conductor code, and client compaction exists only on Responses; (3)
  independent regression — any orchestration whose conversation outgrows
  its smallest worker (any agent client, any API) fails; (4) smallest
  mechanism — trigger only when a request would not fit, one summary per
  omitted prefix, reused; which worker and prompt stay in the example.
- C3: (1) one failed upstream role discards independent questions; (2)
  checklist code; (3) any checklist mixing static and per-item questions;
  (4) skip per question, report it.

## 5. Coverage

| Case | Before | After |
|---|---|---|
| Conversation beyond Qwen (67 turns; long problems) | no requirements, no judgment | Qwen reads summary + newest verbatim; requirements and all judgments |
| Conversation beyond DeepSeek (>~290 messages) | turn fails | DeepSeek reads summary + newest verbatim |
| Empty list (2 turns) | no judgment | adoption judged; the missing list recorded |
| Qwen down | no judgment | adoption judged |
| Chat Completions vs Responses vs Messages | only Responses clients can compact | the same L2 compaction behind every API |

## 6. Tests (one per contract)

- C1: a role whose worker is smaller than the conversation gets the summary
  plus the newest messages; a second turn reuses the cached summary and
  extends it; a role on a large worker gets the whole conversation; the
  same behaviour through Chat Completions and Responses input.
- C3: a static question is asked when the `foreach` source failed; the
  other is reported unjudged.
- Example test: compaction config loads; a failed `requirements` still
  yields the adoption read.

## 7. Docs

m1 (C1 and the C3 amendment to D8), VCO-D19 note, example README,
`PROGRESS.md`.

## 8. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: ruff; changed-path tests | green | 15 min |
| 2 | Live replay of the 69 turns that lost judgments (whole recorded conversations) | Qwen's request fits after compaction; 69 valid lists; DeepSeek roles uncompacted (they fit) | 60 min |
| 3 | Live replay of the 714-message 09/12 conversation | both Qwen and DeepSeek requests fit; compaction cache reused across roles | 20 min |
| 4 | Redeploy (`./run.sh`) | healthy | 15 min |
| 5 | All nine gates | existing criteria; no compaction on their short requests | 2 h |
| 6 | New gate `long-conversation`: five r1 turns that lost judgments and the 714-message conversation, via Chat Completions and via Responses | 200; requirements and all judgments on all; compaction recorded in trace; second pass hits the cache | 60 min |
| 7 | Qwen stopped, one verified request | 200; adoption judged, coverage recorded unjudged | 10 min |

Stop and report at the first failure.

## 9. Risks

- A summary is not the verbatim text; only the part a worker cannot hold
  is summarized, newest turns stay exact.
- The first compaction of a long conversation adds one summarization call
  (DeepSeek); later turns reuse it until new messages need compacting.
- The cache is per gateway process; another gateway recomputes it.

## 10. Decision requested

Approve C1, C3 and E. All changes are in L2; L1 and L3 are untouched.
