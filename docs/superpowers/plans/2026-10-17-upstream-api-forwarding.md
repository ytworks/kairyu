# Plan: every example web UI on the Responses API, and L1 in the same API as L3

Status: implementation and CPU verification in one PR, up to receiving
review. Remaining tasks after that: review and fix rounds, then the final
test in a separate GPU environment (see "Remaining tasks").

Order (owner decision 2026-10-17): implement → run every check a CPU can run
→ review, fix, repeat → final test in a separate GPU environment. This plan
covers the work up to receiving review.

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

Second round (2026-10-17, after implementation started):

15. Readiness `17 * 19` probes through Kairyu move to `/v1/responses` and
    drop their log-probability check there (Responses has none); the L1
    gates keep it.
16. Qwen3.8-Flash-Next's Chat UI drops `presence_penalty` 1.5 (no Responses
    field; documented).
17. Sign-in examples rewrite a saved chat connection to `api_type: responses`
    with the admin credentials.
18. AUTO models are compared with L1 on status and usage keys in
    `api-equivalence`.
19. A field a native wire lacks is sent there only when its value is the one
    the upstream applies without it (vLLM Responses: `min_p` 0; vLLM
    Messages: priority 0; Kairyu Messages: priority 0); otherwise the next
    wire.
20. Locked measurement hashes keep the old hash as history and record the
    current one under "no measurement yet (GPU verification pending)".
21. Rebuilt L1 images have no recorded image ID until the GPU build;
    `run.sh up` stops unless the example's allow-unpinned variable is set.
22. DeepSeek V4 `xhigh`→max is an example-owned overlay image.

Review round 1 (owner review of `6c011416`, decisions 2026-10-18):

23. A request with an assistant prefill never takes Messages (vLLM opens a new
    turn there); it takes Responses or chat.
24. `parallel_tool_calls: false` is not sent on vLLM's Responses or Messages
    wire (neither keeps one call on the pinned versions); it takes chat.
25. MoA passes the caller's API with or without a prepared handle.
26. `api-equivalence` compares vLLM's `response.reasoning_part.*` as the
    `response.content_part.*` events Kairyu emits.
27. The compose demo ships its own Responses filter and a `webui-setup`
    service run once after `up` (`run --rm webui-setup`; an exited one-shot
    service fails `up --wait`), admin credentials → install, else warn; P-B3 runs the
    browser `signup` phase, then `webui-setup` as that account, then
    `initial` from sign-in.
28. Open WebUI v0.12.3 (linux/amd64 digest) for the demo and every example;
    the v0.11.0 replay is repeated on v0.12.3.
29. The standard image, not slim: v0.12's slim stores RAG vectors only in
    PostgreSQL/pgvector, and the standard image is v0.11.0-slim's size.
30. The Jev examples (OpenJev, both Winnow) take the same Open WebUI image;
    nothing else in them changes (decision 8 otherwise stands).

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

## Verification in this PR (CPU)

`uv run pytest` (base/head collection counts), `uv run ruff check .`,
`uv run python -m verification check`, each example's own tests, the
P-B3 compose/browser smoke harness tests, and the filter test over every
example's Chat UI filter. The Open WebUI image cannot be pulled in this
session, so the browser gates themselves run with the GPU tasks.

## Remaining tasks (after this PR's CPU work)

1. Review: receive review of the PR, fix, and repeat until approved.
2. Final test in a separate GPU environment, per example (13):
   - build the rebuilt L1 overlays and record their image IDs
     (`example.json` `image_id`, `kairyu.yaml` `container_image_digest`);
   - `run.sh up` (Kairyu's startup check of `/v1/responses` and
     `/v1/messages` on every declared L1);
   - every `verify.sh` gate on `/v1/responses`, `verify.sh api-equivalence`,
     the browser gates (repo P-B3 and the examples' own);
   - the L1 effort renders: `xhigh` as max on DeepSeek V4/V4.1 overlays and
     GLM-5.3's template, Anthropic `output_config.effort`;
   - manual Chat UI pass (multi-turn thinking, image, tools, effort), checking
     in Kairyu and L1 logs that requests travel on `/v1/responses`;
   - update each MEASUREMENTS.md; report any failing gate to the owner
     before changing a gate or a design.
3. In the same GPU environment, the 3 Jev examples' Chat UI on Open WebUI
   v0.12.3 (decision 30): `run.sh up` and their own UI checks.
