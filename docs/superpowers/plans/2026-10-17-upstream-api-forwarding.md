# Plan: every example web UI on the Responses API, and L1 in the same API as L3

Status: implemented in one PR; GPU verification of the 13 examples is the
remaining task (owner decision 2026-10-17: code review and fixes in this PR).

## Context

The non-Jev examples' Open WebUI talked to Kairyu over `/v1/chat/completions`,
and Kairyu sent every request (chat, Responses, Anthropic Messages) to L1 as
chat or completions, mixing three chat modes per model (Kairyu-rendered
`/completions`, legacy flattening into one user message, native messages).
Goals: (1) the UIs and the example gates use `/v1/responses`; (2) L1 receives
the same API as L3.

## Owner decisions (2026-10-17)

1. L3 serves chat and Responses. L1 receives the caller's API; an L1 without
   it gets chat (Kairyu converts).
2. Anthropic `/v1/messages` follows the same rule, preferring Messages, then
   Responses, then chat, for direct models and AUTO role calls alike.
3. L1 API support is declared per model in `kairyu.yaml`
   (`upstream_responses_models`, `upstream_messages_models`, like
   `upstream_chat_models`); startup fails when a declared API is missing.
4. Chat reaches L1 as native messages in every non-Jev example; Kairyu-rendered
   `chat_templates` stays in the framework (unused by examples) and legacy
   stays for Jev.
5. AUTO role calls follow the caller's API; Jev roles stay Jev.
6. The public Responses contract is unchanged (strict OpenAI, effort
   low/high/max, `none` rejected).
7. On the Responses wire `max` is sent as `xhigh`; L1 templates map it back.
8. Jev examples are untouched: `openjev-diffusiongemma-26b-1gpu`,
   `quyet-1.0-large-1gpu`, `winnow-12b-q8-1gpu`, `winnow-12b-q8-dp8-8gpu`,
   and the Quyet parts of the two Quyet 8-GPU examples.
9. Framework changes authorized: body limit on Responses paths, native
   prefix placement, `kairyu_*` extensions on Responses, caller-API
   forwarding.
10. All existing example gates move to `/v1/responses`; the shared
    `serving_bench.py` gains a Responses mode (formal gates stay on chat);
    `tool_result_turn` gets PR #647's wording.
11. Sign-in Open WebUI examples install the filter with
    `WEBUI_ADMIN_EMAIL`/`WEBUI_ADMIN_PASSWORD` when set, else warn.
12. `deepseek-v4-flash-0731-8gpu` takes the dp2 L1 settings;
    `qwen3.8-deepseek-v4-8gpu` rebuilds its L2 like DTO-D16.
13. Kairyu-to-Kairyu gateways follow the same rule.
14. One PR; GPU verification is a remaining task.

## Phase 0 findings (CPU, 2026-10-17)

The pinned Open WebUI image could not be pulled (its blob host is blocked by
this session's egress policy), so v0.11.0's own `convert_to_responses_payload`,
`convert_responses_result` and `handle_responses_streaming_event` were
extracted from the tagged source and run against a real Kairyu app:

| Case | Result |
|---|---|
| First turn, streaming | 200; reasoning and message items rebuilt |
| Second turn replaying Responses-born output items | 200 |
| Chat Controls `reasoning_effort`/`seed`/`top_k` | 400 `reasoning_effort` raw; 200 normalized |
| Native tool round trip | 200 |
| Chat-era saved output items (`attributes`, `started_at`...) | 400 raw; 200 normalized |
| Non-streaming result (`stream_response: false`) | reasoning dropped by the converter |
| `store: false` | 200 |

Filter normalization therefore: effort selector or top-level
`reasoning_effort` → `reasoning.effort`; drop non-Responses sampling keys;
keep only Responses fields on saved output items; `store: false`; stream when
the Thought section matters.

L1 effort mapping for `xhigh` (pinned sources): Qwen3.8-27B clamps every
effort to low and Qwen3.8-Flash-Next accepts `xhigh` natively (no change);
vLLM's DeepSeek-V4.1 encoder treats `xhigh` as 75 (≠ max 100) and the
DeepSeek-V4 encoder maps it to high, so their example overlays map `xhigh` to
`max`; GLM-5.3's template is checked on GPU.

## Framework (kairyu/)

- `kairyu/engine/upstream_wire.py`: wire preference, conversation/tools/format
  codecs for Responses and Messages, reply/stream readers.
- `GenerationRequest.caller_api`; `OpenAICompatBackend(upstream_apis=...)`,
  `_select_wire`, native generate/stream, `startup()` route check;
  `responses_api`/`messages_api` capability bits (`vllm`, `kairyu`; `kairyu`
  also gains `conversation_prefix`).
- `DeploymentSpec.upstream_responses_models`/`upstream_messages_models`
  (subsets of `upstream_chat_models`); the builder injects `upstream_apis`.
- L2: `OrchestrationRequest.caller_api` threaded to every role request.
- `ReplicaPool` prefix placement hashes native conversations
  (`placement_text`).
- Responses envelopes carry `kairyu_*` extensions; `X-Kairyu-Trace` honored.
- `ChatBodyLimitMiddleware` covers the Responses paths.
- Gateways in front of static Kairyu replicas declare the three lists;
  discovery pools (kind f1a/f1c) cannot and stay legacy.

Design records: m9 D2, m11 D4/D7 and m1 D8 amendments of 2026-10-17.

## Examples (13)

Open WebUI `api_type: responses` + normalization filter; pools on the three
upstream lists (Quyet pools stay legacy); `xhigh`→`max` in DeepSeek overlays;
0731-8gpu L1 = dp2 flags; qwen3.8-deepseek-v4 L2 rebuilt; gates on
`/v1/responses` with the `api-equivalence` gate
(`verification/product/correctness/upstream_api_equivalence.py`).

## Verification

CPU: `uv run pytest` (base/head counts), `uv run ruff check .`,
`uv run python -m verification check`. GPU (remaining): every example's
gates, `api-equivalence`, browser gates, and the L1 effort renders.
