# Anthropic Messages thinking round trip for single-model pools (DeepSeek Harness)

Status: **Approved 2026-10-19 (owner); implementation in progress.**

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
