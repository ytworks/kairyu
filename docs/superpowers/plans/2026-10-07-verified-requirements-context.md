# Verified route: Winnow judges every turn of a long agent conversation

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19 (`2026-10-06-winnow-verified-three-wave.md`). Status:
awaiting owner approval.

## 1. Problem and goal

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): Winnow judged 61
of 132 VERIFIED turns; on the other 71 no Winnow request was sent and the
answer was written without judgments. From about turn 20 of each problem
Winnow stops.

Goal: on every VERIFIED turn of any conversation the route can serve,
Winnow judges every draft's adoptability and every draft against every
requirement.

## 2. Causal chain (each link measured)

| Link | What happens | Evidence |
|---|---|---|
| L1 | `judgments` builds all its questions from `requirements`' list; no list → the whole checklist is unavailable and no Winnow request is sent, not even the 5 adoption questions that need only the request and the drafts | `checklist._pending_questions` → `_source_items`; failed judgments take 0.0 s with no Winnow read |
| L2 | `requirements` (Qwen) gave no list on 69 turns: 67 rejected (vLLM 400 "maximum context length is 262144 tokens … at least 196609 input tokens"; inputs 199,082-325,598 + 65,536 output cap), 2 empty after 188 / 342 tokens on 148K / 181K inputs (the same inputs replayed 4 times all returned valid lists) | live replay; live `/tokenize` of Kairyu's exact request |
| L3 | 81 % of that input is earlier turns' `reasoning_content` (section 3) | `/tokenize` |
| L4 | Without L3 the conversation is still larger than Qwen on long problems: ~0.77K Qwen tokens per message (46,339 tokens at 61 messages); Qwen's 262,144 − 65,536 holds ~255 messages; the 09/12 DeepSWE run's conversations reached median 298, p90 437, max 714 messages (77 of 113 problems above 255) | `/tokenize`; trajectory files |
| L5 | Winnow itself is not the limit: its input on the 61 judged turns was at most 15,115 tokens (context 65,536); drafts output at most 20,804 tokens | trace v2 |

## 3. `reasoning_content`

| Step | What happens | Evidence |
|---|---|---|
| Emitted | With `expose_intermediate_outputs: true` each answer's `reasoning_content` is the turn's stage report: attribution, then per stage (drafts, requirements, judgments, answer) the model reasoning and the output. Median 11,815 Qwen tokens per turn, max 14,536 | recorded responses |
| Echoed | mini-swe-agent (LiteLLM) keeps each assistant message as returned and sends `reasoning_content` back next turn | recorded requests |
| Accepted | Chat Completions accepts the field (m11 assistant-history amendment, 2026-08-14: "for key-sensitive model templates and L2 conversation history") | `protocol.ChatMessage` |
| Re-read | `validate_orchestration_chat_input` copies every message field into the L2 conversation; every role's `{conversation}` / `{query}` renders it | `chat_service.py` |
| Size | 14:42 turn (61 messages): 907,608 characters = 240,650 Qwen / 241,080 DeepSeek tokens; without it 117,040 characters = 46,339 / 43,326 | `/tokenize` on both L1s |
| Readers | Qwen overflows (L2); DeepSeek drafts and answer read 5.6x more each turn and approach 1M as turns grow; Winnow reads only the request and is unaffected | prompts; checklist state |
| Precedent | Responses AUTO drops replayed reasoning: "stage output would grow every L2 prompt" (m11 D4); Chat Completions was not aligned | `docs/design/m11-product.md` |

The report is caller-facing output; no role asks for it as input. Every
earlier answer, tool call and tool result stays in the conversation.

## 4. Changes

| # | Removes | Layer | Owner | File | Change |
|---|---|---|---|---|---|
| 1 | L3 (and L2's empty lists) | L3→L2 | framework | `kairyu/entrypoints/server/chat_service.py` (`validate_orchestration_chat_input`) | Build the L2 conversation without assistant `reasoning_content`. Still accepted on the wire, still passed to direct engines' templates; Kairyu still emits the report |
| 2 | L4 | L1 | example | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/compose.yaml` (service `qwen`), `kairyu.yaml` (pool `qwen3.8-27b`) | Qwen at its documented 1M context: `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `--hf-overrides` with `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144` (other `rope_parameters` as in the checkpoint), `--max-model-len 1000000`; pool `max_model_len: 1000000`. Qwen's limit then equals DeepSeek's (1,048,576): `requirements` fails only where the whole turn would |
| 3 | L1 | L2 | example | `verified.yaml`, `verified-always.yaml` | Split `judgments` into `adoption` (verifies `drafts`; the 5 adoption questions) and `coverage` (verifies `requirements`; the 5 x N questions; waits for `drafts`, PR #641 F1). Winnow judges adoption on every turn even if Qwen fails; `answer` reads `{adoption}` and `{coverage}` |

Everything else stays: prompts' wording (except `answer` naming the two
judgment blocks), caps, efforts, routing, Winnow, DeepSeek, the emitted
report.

Framework admission (change 1): (1) broken contract — an orchestrated
conversation re-reads every earlier stage report (code path above); (2) no
extension point — the conversation is built before any role or example
config; (3) independent regression — any orchestrator exposing intermediate
outputs behind a client that echoes `reasoning_content` grows every role's
input ~12K tokens per turn; (4) smallest mechanism — omit one field in one
place, as Responses AUTO does.

## 5. Coverage of every failure

| Case | Before | After |
|---|---|---|
| Echoed reports push Qwen over 262,144 (67 turns) | no judgment | input 30,330-96,837 tokens: requirements and both judgments |
| Empty list on report-filled input (2 turns) | no judgment | input without reports (replays valid); if Qwen still fails, adoption is judged |
| Conversation content beyond 255 messages (77/113 problems in 09/12) | would lose requirements | Qwen 1M holds ~1,200 messages; both judgments |
| Qwen down or erroring | no judgment | adoption judged; coverage absent |
| Conversation beyond 1M | whole turn fails on DeepSeek too | unchanged (route limit) |

## 6. Tests

- Change 1: one chat-input test — assistant `reasoning_content` is absent
  from the L2 prompt; content and tool calls stay.
- Change 3: the example test — with Qwen failing, Winnow still receives the
  5 adoption questions and `answer` reads them; with Qwen succeeding, both
  reads happen (replaces the one-read assertion).
- Change 2: configuration, checked on GPU.

## 7. Docs

m11 (amend the 2026-08-14 assistant-history amendment for the L2 path);
VCO-D19 amendment (split judgments, Qwen 1M); example README (L1 row, DAG);
`PROGRESS.md`.

## 8. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: ruff; changed-path tests | green | 10 min |
| 2 | Redeploy (`./run.sh`; Qwen restarts with YaRN) | healthy; Qwen reports max model len 1,000,000 | 20 min |
| 3 | Live replay of the 69 turns that lost judgments through the new code's `requirements` | 69 accepted, 69 valid lists, 0 empty | 40 min |
| 4 | Live replay of the 09/12 longest conversation (714 messages) | accepted; valid list | 10 min |
| 5 | All nine gates in GATES order; verified-route expects two Winnow reads per turn (5 adoption + 5 x N coverage) | existing criteria | 2 h |
| 6 | New gate `long-conversation`: five r1 turns that lost judgments and the 714-message conversation, through `kairyu-verified-always` | 200; adoption and coverage judged on all six; the response still carries the report in `reasoning_content` | 40 min |
| 7 | Qwen stopped, one verified request | 200; adoption judged | 10 min |

Stop and report at the first failure.

## 9. Risks

- Static YaRN may affect Qwen on short inputs (model card); step 5 runs the
  short-request gates through it.
- Four long `requirements` calls may exceed Qwen's 1,791,840-token KV cache;
  vLLM queues them (slower, not failing).

## 10. Decision requested

Approve changes 1-3 with this verification.
