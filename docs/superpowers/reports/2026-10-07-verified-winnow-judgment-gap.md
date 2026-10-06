# Defect report: Winnow judges only part of the VERIFIED turns in multi-turn agent conversations

Facts only, 2026-10-07 00:44 JST. No fix is proposed or approved here.

## 1. Symptom

On `kairyu-verified` driven by an agent client (DeepSWE via kairyu-bench),
Winnow's judgment (`judgments`) is not performed on a large share of
VERIFIED turns. At 00:44 JST: 160 VERIFIED turns, 73 judged, 87 not judged.
Every unjudged turn has `judgments` status `failed`, reason
`checklist_unavailable`, duration 0.0 s, and no request reached Winnow. The
turn still returns HTTP 200: `answer` is written without judgments.
Unjudged turns start at roughly the 20th agent turn of each problem and then
dominate.

## 2. Environment

| Item | Value |
|---|---|
| Host | 8 x RTX PRO 6000 Blackwell (SM120), PCIe |
| Repo / branch / head | `ytworks/kairyu`, `claude/winnow-verified-three-wave` (PR #641), deployed code = commit `239a1d97` + `1ee0cb28` (gate-script fix only); later commits are docs |
| Example | `examples/deepseek-v4.1-qwen3.8-winnow-8gpu`, started with plain `./run.sh` (2026-10-06 21:1x JST) |
| L1 | DeepSeek-V4.1-Flash DP6/EP6 GPU 0-5, pool `max_model_len` 1,048,576; Qwen3.8-27B FP8 GPU 6, vLLM v0.23.0, `--max-model-len 262144`, pool `max_model_len` 262,144, KV cache 1,791,840 tokens, uvicorn access log disabled (`--disable-uvicorn-access-log`); Winnow-12B Q8_0 GPU 7, System One `--decision-context 65536` |
| L2 (`verified.yaml` primary = `verified-always.yaml`) | `drafts` (DeepSeek, effort inherit, JSON D1..D5, cap 131,072) and `requirements` (Qwen, effort low, JSON `points` ≤ 16, cap 65,536) run in parallel; `judgments` = checklist verifier on `drafts`, depends on `drafts` and `requirements`: 5 static adoption questions + 5 per-draft `foreach: {role: requirements, path: points}` question groups, state = `request` (≤ 40,000 chars) + `drafts`, threshold 1.0, `max_refinements: 0`, `on_unavailable: publish_unverified`; `answer` (DeepSeek, effort inherit, cap 262,144) reads `{drafts}`, `{requirements}`, `{judgments}`; `expose_intermediate_outputs: true`; `internal_max_tokens: 131072` |
| GPU gates before this run | all nine passed on single-turn requests (example `MEASUREMENTS.md`, 2026-10-06 21:17-23:03 JST) |

## 3. Run that shows it

| Item | Value |
|---|---|
| Run | `kairyu-bench` DeepSWE v1.1, run id `deepswe-verified-3wave-full-4w-20261006-r1`, started 2026-10-06 23:09 JST, still running |
| Conditions | 113 problems, 1 attempt, 4 workers, no `reasoning_effort` sent (server default high), API timeout 3,600 s, model auto-detected `kairyu-verified` |
| Client | mini-swe-agent 2.4.6 via LiteLLM, `/v1/chat/completions`, tools (bash) |
| Relay | `/tmp/deepswe-verified-3wave-full-4w-20261006-r1-relay.py` records every call with `X-Kairyu-Trace: 1` |
| Launcher | `/tmp/deepswe-verified-3wave-full-4w-20261006-r1-launch.sh` |
| Progress at 00:44 JST | 2 problems scored, 1 solved; 4 running |

## 4. Observations

### 4.1 `judgments` failures

- 87 of 160 VERIFIED turns: `judgments` `failed` / `checklist_unavailable`,
  started and completed at the same timestamp (0.0 s); no Winnow request.
- On all 87, `requirements` produced no usable list:
  - 85: `requirements` generation `failed`, trace message
    `UpstreamClientError`.
  - 2 (turns starting 14:31:01 and 14:38:32 UTC): `requirements`
    `success` with 188 and 342 completion tokens; the stage report shows
    Qwen reasoning that stops mid-sentence and an empty stage output.

### 4.2 Why `requirements` fails (live replays)

- Replaying the 14:42:49 UTC turn's `requirements` request exactly as Kairyu
  builds it (script in §6) to Qwen returns HTTP 400:
  `"This model's maximum context length is 262144 tokens. However, you
  requested 65536 output tokens and your prompt contains at least 196609
  input tokens, for a total of at least 262145 tokens."` The 400 does not
  appear in Qwen's container log (access log disabled).
- Qwen token counts (`/tokenize` on the exact rendered request) of the 67
  turns that had failed by 00:20 JST: min 199,082, median 267,676, max
  325,598.
- Successful `requirements` outputs (61 calls by 00:20 JST): max 3,486
  completion tokens, p99 3,422, median 1,899.
- The two short outputs (§4.1) replayed: 14:31:01 → valid list (finish
  `stop`, 525 / 1,784 / 2,073 completion tokens on three replays);
  14:38:32 → valid list (1,949 tokens). Not reproduced.

### 4.3 What the input consists of

- The request body from the client carries, for earlier assistant turns,
  `reasoning_content` holding Kairyu's own stage report (headings
  "Final answer attribution", "drafts — attempt 1", per-stage
  "Model reasoning" / "Stage output"), as returned by Kairyu with
  `expose_intermediate_outputs: true`.
- Per-turn report size: median 11,815 Qwen tokens, max 14,536 (first 10
  reports of the 14:42:49 turn).
- 14:42:49 UTC turn, 61 messages, 18 carrying `reasoning_content`:
  - L2 `{conversation}`: 907,608 characters = 240,650 Qwen tokens =
    241,080 DeepSeek tokens.
  - Parts (Qwen tokens): `reasoning_content` 189,953; message content
    33,852; tool calls 3,845.
  - Same conversation with `reasoning_content` removed: 117,040 characters
    = 46,339 Qwen / 43,326 DeepSeek tokens.
- 15:17:08 UTC turn, 95 messages: L2 `{conversation}` 320,460 Qwen tokens
  (`reasoning_content` 236,197; content 54,575; tool calls 9,379).
- The 67 failed turns re-tokenized with `reasoning_content` removed:
  30,330-96,837 Qwen tokens (median 75,815). Three of them replayed that
  way returned valid lists (11-13 points).
- Character share of `reasoning_content` in the conversation: 14:31 turn
  389,264 of 536,279; 14:42 turn 776,783 of 887,879.

### 4.4 Other sizes

- `drafts` output (149 calls): max 20,804 tokens, median 4,248.
- Winnow input on judged turns (68 reads by 00:20 JST): max 15,115 tokens,
  median 12,720 (decision context 65,536).
- VERIFIED turn latency in r1: p50 ~107-116 s; Qwen `requirements` stage
  p50 67-98 s, the longest stage (DeepSeek drafts p50 32-40 s).
- Conversation lengths on this host's earlier DeepSeek-only DeepSWE run
  (`deepswe-full-8w-20260912-r1`, 113 trajectories): messages median 298,
  p90 437, max 714; max content 925,539 characters.

## 5. Code paths involved

- `kairyu/orchestration/checklist.py`: `_pending_questions` iterates every
  question; for a `foreach` question `_bindings` → `_source_items` raises
  `ChecklistUnavailable("checklist_unavailable")` when the source role has
  no output or no JSON list at the path. The exception aborts the whole
  `judge`, including questions without `foreach`.
- `kairyu/orchestration/conductor.py`: `_checklist_round` handles
  `ChecklistUnavailable` with `on_unavailable: publish_unverified` (trace
  `verified:unavailable`, output of the verifier not set; `{judgments}`
  renders empty in `answer`). `_render` fills `{conversation}` with
  `conversation_text(query)` — the whole transcript, no size bound.
- `kairyu/entrypoints/server/chat_service.py`:
  `validate_orchestration_chat_input` builds the L2 transcript from
  `_message_wire_shape`, which copies every set field of each message,
  including assistant `reasoning_content`.
- `kairyu/engine/openai_backend.py`: upstream 4xx → `UpstreamClientError`.
- Design docs: m11 assistant-history amendment (2026-08-14) keeps
  `reasoning_content` "for key-sensitive model templates and L2 conversation
  history"; m11 D4 (Responses) "AUTO drops replayed reasoning: stage output
  would grow every L2 prompt".

## 6. Evidence and reproduction

| What | Where |
|---|---|
| Every recorded call (request message ids, usage, full response incl. `kairyu_trace_v2` and `reasoning_content`) | `/home/y-takagi/kairyu-bench/results/deepswe-verified-3wave-full-4w-20261006-r1-telemetry/calls/*.json` |
| Deduplicated request messages | `…-telemetry/messages/*.json` |
| Benchmark progress / trajectories | `/home/y-takagi/kairyu-bench/results/deepswe-verified-3wave-full-4w-20261006-r1/raw/deepswe/` |
| Replay one turn's `requirements` against live Qwen (`<HH:MM:SS UTC start> [strip]`) | `…-r1-analysis/replay_requirements.py` |
| Token-count the failed turns' `requirements` requests (with / without `reasoning_content`) | `…-r1-analysis/count_requirements_tokens.py`, `count_requirements_tokens_noreasoning.py`; inputs `failed_turns.json`; outputs `failed_counts.json`, `failed_counts_noreasoning.json` |
| Progress report used during the run | `…-r1-analysis/report_full.py <run-id>` |

Run the scripts from `/home/y-takagi/kairyu` with
`CUDA_VISIBLE_DEVICES= uv run python <script> …` while the example is up
(Qwen on `127.0.0.1:8015`, DeepSeek on `127.0.0.1:8014`).

## 7. Not determined

- Why Qwen ended generation after 188 / 342 tokens on the two turns of
  §4.1 (not reproduced on replay).
- Whether DeepSeek roles would also exceed 1,048,576 later in long problems
  (not observed yet in r1).

## 8. Status

- No fix is approved. `docs/superpowers/plans/2026-10-07-verified-requirements-context.md`
  holds unapproved drafts; it is not part of this report.
- r1 is still running.
