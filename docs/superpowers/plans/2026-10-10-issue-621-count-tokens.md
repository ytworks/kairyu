# Issue #621: token counts equal billed prompt tokens on vLLM upstreams

Status: **Approved 2026-10-10 (owner: exact count, request-based count
interface, image counts declined); implemented, GPU verification pending.**

## Problem

`/v1/messages/count_tokens` and `/v1/responses/input_tokens` must return the
prompt tokens generation bills (m9 D1 usage truth). For an `openai` engine with
`upstream: vllm` and no Kairyu chat template (`legacy_chat_models`, the shape of
every example that publishes a vLLM model directly), they do not:

- The routes render a string (`render_tool_intent(...)`) and the backend posts
  `{"model", "prompt"}` to vLLM's `/tokenize`, which tokenizes it as a raw
  completion prompt.
- Generation sends that text as a chat message (`messages=[{"role": "user", ...}]`)
  with tools as a separate `tools` field, and vLLM applies the chat template.
- No tools: the count misses the template's role delimiters and generation
  prompt (too low). Tools: the count includes Kairyu's tool-intent suffix, which
  the `openai` backend never sends, and misses the template's own tool rendering.
- Images: no change needed. The planning-time finding that
  `/v1/messages/count_tokens` counts an image request as an empty string was
  wrong: the Messages surface rejects image blocks with 400 before counting
  (correction found while writing the tests). `/v1/responses/input_tokens`
  already declines image input.

Cause: the count interface receives only a string; only the backend knows what
it sends.

## Decision

Count exactly what generation sends.

- The routes build the `GenerationRequest` with the function generation uses
  (`validate_chat_request_async`) and pass it to
  `backend_count_prompt_tokens_async(backend, request)`.
- Each backend counts its own wire input. Native/process/mock backends count the
  same tool-intent text as today (no count change). `ReplicaPool` delegates the
  request.
- `openai` + `upstream: vllm`: the `/tokenize` body is derived from the same
  `_payload()` generation sends. The `/completions` path (Kairyu-rendered
  template) keeps `{"model", "prompt"}`. The chat path sends `messages`, `tools`,
  `add_generation_prompt`, `continue_final_message` and `chat_template_kwargs`,
  adding `reasoning_effort` / `enable_thinking` the way vLLM's chat request
  merges them (`/tokenize` does not merge them itself; checked on vLLM
  `0.30.1rc1.dev396`).
- Image (multimodal) prompts stay declined: the Messages surface rejects image
  blocks (400) and Responses keeps its `unsupported_value` error; the vLLM
  count returns `None` for a multimodal prompt. llama.cpp stays declined (LCP-D3).
- Known limit: vLLM's Kimi K3 and Cohere renderers read `tool_choice` /
  `response_format`, which `/tokenize` cannot carry; no served example uses them.

Framework admission (`.claude/rules/framework-boundary.md`):

1. Broken shared contract: count = billed prompt tokens, broken on vLLM
   upstreams (`messages_service.py` count route, `responses_service.py`
   `input_tokens` route, `openai_backend.py` `count_prompt_tokens_async` vs
   `_payload`).
2. The string-only `count_prompt_tokens_async(prompt: str)` cannot carry the
   messages, tools and template kwargs that generation sends.
3. Independent use: any vLLM-backed public model used by Claude Code or the
   OpenAI SDK; observable regression: count != `usage` input tokens.
4. Smallest mechanism: the count interface takes the request; no example policy.

## Changes

| Layer | File | Change |
|---|---|---|
| L3 | `kairyu/entrypoints/server/messages_service.py` | count the generation request |
| L3 | `kairyu/entrypoints/server/responses_service.py` | count the generation request |
| L1 | `kairyu/engine/backend.py` | count helper takes the request |
| L1 | `kairyu/engine/openai_backend.py` | `/tokenize` body from `_payload()`; decline images |
| L1 | `kairyu/engine/{kairyu_backend,zmq_backend,mock}.py` | take the request; same count |
| L2 | `kairyu/orchestration/replica.py` | delegate the request |
| Docs | `docs/design/m9-truthful-api.md` (D1 amendment), `PROGRESS.md` | decision and GPU evidence |

## CPU tests

- Add one parametrized route test (`tests/server/test_messages_api.py`): a fake
  vLLM (`tests/server/_fake_vllm.py`, `httpx.MockTransport`) whose `/tokenize`
  and `/v1/chat/completions` derive counts from the template-relevant fields
  by vLLM's merge rules; count_tokens equals `/v1/messages`
  `usage.input_tokens` for no-tools and tools+thinking. Both fail on main.
- Replace `tests/server/test_responses_contract.py::test_input_tokens_counts_the_rendered_prompt`
  with the same fake-vLLM equality against `/v1/responses` usage.
- `tests/unit/test_openai_backend.py`: drop the `vllm-count` body-shape case
  (covered by the route test); keep the fail-soft cases (non-200, non-int,
  non-vLLM, transport error).
- Report base/head collection counts.

## GPU verification

Reproduce on the current Kairyu image, then swap only the Kairyu container to
the branch build (vLLM keeps running).

| Gate | Example | Covers | Budget |
|---|---|---|---|
| V1 | `qwen3.8-27b-1gpu` | Jinja template, `enable_thinking`, direct engine | start 15 min + 5 min |
| V2 | `deepseek-v4.1-flash-8gpu` | DeepSeek encoder, ReplicaPool | start 30 min + 10 min |

Cases per model (generation `max_tokens` 1): text only; system + three tools;
multi-turn tool_use/tool_result transcript + tools; thinking + tools. Cases 1-3
also through `/v1/responses/input_tokens`.

Pass: every count equals the generation's input tokens (difference 0); the
current image shows the mismatch. The probe script
stays in the session scratchpad; results go to the PR and the m9 amendment.
