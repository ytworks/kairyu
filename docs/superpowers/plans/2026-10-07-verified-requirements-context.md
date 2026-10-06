# Winnow judges every turn of a long agent conversation

Plan, 2026-10-07. Branch `claude/winnow-verified-three-wave` (PR #641),
follows VCO-D19. Status: awaiting owner approval.

## 1. Problem and goal

DeepSWE r1 (`deepswe-verified-3wave-full-4w-20261006-r1`): Winnow judged 61
of 132 VERIFIED turns; on 71 no Winnow request was sent. Goal: Winnow judges
on every turn the route can serve, while every role keeps reading the whole
conversation, including each earlier turn's `reasoning_content`, so that
multi-turn exchanges stay exact.

## 2. Causal chain (measured)

| Link | Measured | Defect |
|---|---|---|
| L1 | No `requirements` list → `judgments` sends nothing, not even the 5 adoption questions that need only the request and the drafts (`checklist._pending_questions` raises for the whole checklist when one `foreach` source is missing) | generic: a checklist is all-or-nothing on its item sources |
| L2 | Qwen gave no list on 69 turns: 67 rejected by vLLM (input 199,082-325,598 + 65,536 cap > 262,144), 2 empty on 148K / 181K inputs (valid on 4 replays each) | consequence of L3 |
| L3 | The route's roles read the same conversation: DeepSeek (drafts, answer) holds 1,048,576 tokens, Qwen (requirements) 262,144. The conversation, with each turn's stage report in `reasoning_content`, grows ~3.4K Qwen tokens per message (240,650 at 61 messages, 320,460 at 95), so Qwen overflows from ~60 messages while DeepSeek still reads it | example: one worker of the route cannot hold what the route serves |
| L4 | Winnow's input on judged turns ≤ 15,115 tokens (context 65,536) | none |

## 3. Changes

| # | Fixes | Layer | Owner | File | Change |
|---|---|---|---|---|---|
| 1 | L3 | L1 | example | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu/compose.yaml` (service `qwen`), `kairyu.yaml` (pool `qwen3.8-27b`) | Qwen at its model card's 1M context: `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; `--hf-overrides` with `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144` (other `rope_parameters` as in the checkpoint); `--max-model-len 1000000`; pool `max_model_len: 1000000`. Qwen then reads exactly what DeepSeek reads, up to the same limit |
| 2 | L1 | L2 | framework | `kairyu/orchestration/checklist.py` (`_pending_questions`, `judge`), `kairyu/orchestration/conductor.py` (trace) | A `foreach` source without a usable list skips only the questions bound to it; every other question is still asked; skipped questions are recorded as unjudged with their source in the verdict and trace. With nothing left to ask, behaviour is as today |

Unchanged: the conversation every role reads (whole, with `reasoning_content`),
prompts, caps, efforts, DAG, routing, Winnow, the emitted report.

Framework admission (change 2): (1) contract broken — one failed upstream
role discards independent questions (code path above); (2) the checklist is
framework code, no extension point; (3) any checklist mixing static and
per-item questions loses all of them when its item role fails (overflow,
backend down, unparsable JSON), independent of this example; (4) skip per
question and report it; which questions exist stays in the example.

## 4. Coverage

| Case | Before | After |
|---|---|---|
| Conversation beyond Qwen's 262,144 (67 turns) | no judgment | Qwen 1M reads it whole: requirements and all judgments |
| Empty list (2 turns) | no judgment | adoption judged (change 2); the list's absence is recorded |
| Qwen down or erroring | no judgment | adoption judged |
| Conversation beyond 1M | DeepSeek fails too | unchanged: the route's own limit (~290 messages at the measured growth) |

## 5. Tests

- Change 2: one checklist test — a static question and a `foreach` question
  whose source failed: the static question is asked, the other is reported
  unjudged.
- Example test: `requirements` failing still yields the adoption read and
  `answer` reads it (replaces the case that expected no read).
- Change 1: configuration, checked on GPU.

## 6. Docs

m1 D8 amendment (change 2); VCO-D19 note (Qwen 1M, why); example README L1
row; `PROGRESS.md`.

## 7. Verification

| # | Step | Pass criteria | Budget |
|---|---|---|---|
| 1 | CPU: ruff; changed-path tests | green | 10 min |
| 2 | Redeploy (`./run.sh`; Qwen restarts with YaRN) | healthy; Qwen log shows max model len 1,000,000 | 20 min |
| 3 | Live replay of the 69 turns that lost judgments (exact requests, whole conversation) | 69 accepted, 69 valid lists | 40 min |
| 4 | All nine gates (short requests through YaRN Qwen) | existing criteria | 2 h |
| 5 | New gate `long-conversation`: five r1 turns that lost judgments, through `kairyu-verified-always` | 200; requirements and all judgments on all five | 30 min |
| 6 | Qwen stopped, one verified request | 200; adoption judged, coverage recorded unjudged | 10 min |

Stop and report at the first failure.

## 8. Risks

- Static YaRN may affect Qwen on short inputs (model card); step 4 covers
  short requests.
- Several long `requirements` calls together exceed Qwen's 1,791,840-token
  KV cache; vLLM queues them (slower, not failing).

## 9. Decision requested

Approve changes 1 and 2.
