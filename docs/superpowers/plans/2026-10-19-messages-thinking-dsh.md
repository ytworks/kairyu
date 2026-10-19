# Anthropic Messages thinking round trip for single-model pools (DeepSeek Harness)

Status: **Approved 2026-10-19 (owner); smoke at effort max passed on `423494b0`; review fixes approved 2026-10-19, implemented.**

Owner decision 2026-10-19: fix Kairyu, then run DeepSWE with DeepSeek Harness
(`dsh` 0.2.1rc2, `sdk-minimal`, `deepseek-official` provider; kairyu-bench PR
#31) through Kairyu's `/v1/messages` at effort max on
`deepseek-v4.1-flash-1017-6gpu`. Kairyu-side problems are logged in #659.

## Problems (#659)

1. `thinking: {"type": "enabled"}` without `budget_tokens` answers 400.
   dsh always sends it for efforts low/high/max (`serialize.ts` 147-156) with
   `output_config.effort`. Source: `messages_service._reasoning_effort_from`.
2. An assistant `thinking` block in the history answers 400
   (`_assistant_blocks` accepts only `text` and `tool_use`). dsh replays the
   previous turns' thinking blocks (`signature` optional).
3. Messages responses never carry thinking, also for single-model pools
   (`_content_blocks_from_message` and the SSE paths drop `reasoning_content`
   to keep AUTO orchestration reasoning private). A replaying harness loses
   the model's own reasoning at every tool call. Responses already returns and
   replays reasoning for non-orchestrated models (`replay_reasoning=not
   orchestrated`).

The upstream wire needs no change: the normalized chat history's
`reasoning_content` already becomes thinking blocks on vLLM's `/v1/messages`
(`upstream_wire.messages_body` / `anthropic_conversation(thinking=True)`).

## Changes (`kairyu/`, L3)

- A. `{"type": "enabled"}` without `budget_tokens` means thinking on, like
  `adaptive` (effort from `output_config.effort`); with `budget_tokens` it
  stays a 400.
- B. Assistant `thinking` blocks (`type`, `thinking`, optional `signature`)
  become `reasoning_content` for non-orchestrated models and are dropped for
  AUTO models (as Responses). Signatures are not verified (Kairyu does not
  sign).
- C. Structural: non-orchestrated models return `reasoning_content` as a
  `thinking` block (unary: first content block; SSE: `thinking`
  `content_block_start`, `thinking_delta`, `signature_delta`), with an empty
  `signature`. AUTO models keep returning none. Messages clients of pool
  models (Claude Code included) now see thinking, as Responses clients do.

## Tests

- `tests/server/test_messages_api.py`: `enabled` joins the accepted thinking
  modes; the `budget_tokens` 400 stays.
- `tests/server/test_upstream_api_forwarding.py`: a dsh-shaped request
  (`enabled`, effort max, thinking + tool_use history, tool_result) reaches
  vLLM's `/v1/messages` with the thinking block and effort, and the pool
  response returns a thinking block; an AUTO model returns none.

## Verification

1. ruff and the changed paths' tests.
2. GPU (six-GPU 1017, L1 unchanged): `run.sh up` replaces only Kairyu; a dsh
   smoke at effort max with `--limit 1`: every `/v1/messages` 200, responses
   carry thinking, later turns replay it, `stop_reason` tool_use/end_turn,
   committed patch and verifier reward, normalized agent `deepseek-harness`.
3. DeepSWE with dsh: 113 tasks, 8 workers, one attempt, effort max, context
   1,048,576, `max_tokens` 49152; kairyu-bench PR #31 `4b53d338`; only tasks
   lost to infrastructure failures are re-run; kairyu-bench wiki as
   "DeepSeek-1017 6GPU max (DeepSeek Harness)".
4. Merge after CI is green, by the owner's decision.

## Review fixes and issue #659 item 4 (approved 2026-10-19, owner)

GPU result on `423494b0` (15:13-15:25 JST): 66 `/v1/messages` calls all 200,
65 replies with thinking (199,605 characters), 65 requests replayed 2,085
thinking blocks, `stop_reason` tool_use 65 / end_turn 1, `model.patch` and
reward 1 (f2p 20/20, p2p 3/3), normalized agent `deepseek-harness`.

1. #659 item 4: `thinking: disabled` turns an engine or pool model's thinking
   off (internal effort `none`, set only by the Messages boundary; public Chat
   and Responses unchanged; AUTO unchanged). Wires: Messages
   `thinking: {"type": "disabled"}`, Responses `reasoning.effort: "none"`, chat
   `reasoning_effort: "none"`; Kairyu-rendered templates `thinking_mode: chat`,
   `enable_thinking: false`. Checked on the 1017 vLLM: all three wires render or
   generate without thinking.
2. Review: `thinking.display: "omitted"` returns thinking blocks without text;
   other values than `summarized`/`omitted` are 400.
3. Review: a Kairyu upstream receives replayed thinking blocks.
4. Review: the first thinking delta is the SLO first token (text and tool
   streams).

Tests: the forwarding round trip gains `upstream: kairyu` and `omitted` cases;
a thinking-off test over the three wires; an SLO test (thinking 0.5 s, answer
10 s, SLO 1 s); `disabled` leaves the no-op thinking test.

GPU verification (after the review has no open findings; six-GPU 1017, only
Kairyu replaced):
- Smoke A, effort max, one task (15-30 min): the criteria of step 2 above.
- Smoke B, effort off, one task (about 30 min): every call 200, no thinking
  block in replies, the first request's input is 1,171 tokens (rendered without
  thinking), a verifier reward.
- Stop and report at the first failure. DeepSWE with dsh starts on the owner's
  order.
