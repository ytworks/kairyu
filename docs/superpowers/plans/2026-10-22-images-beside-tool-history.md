# Images beside tool-call history for upstream-rendered models

Status: **Approved 2026-10-22 (owner), in PR #662.** Owner direction: fix
this in Kairyu (option 2 on PR #662), so the Chat UI can keep its code
interpreter on by default; the fix ships in the same PR as the Chat UI change.

## Problem

A Chat Completions or Responses request that carries an image and tool-call
history (assistant tool calls, tool results) is rejected with 400, even for
models whose upstream (vLLM) renders the conversation itself. Reproduced on
`examples/deepseek-v4.1-flash-1017-8gpu` (issue #659 item 6): an image alone
is 200; the same request with a `function_call` and its output is 400
"images cannot be combined with tool-call history in one request". Open
WebUI passes built-in tools (knowledge, code interpreter) in every chat, so
any chat where the model used a tool can no longer take an image. With the
code interpreter on by default (PR #662) the model ran code on an image
question and the next turn failed.

## Facts (main `b9018877`)

1. Responses rejects up front for every model:
   `responses_items.reject_images_beside_tools`, called by
   `responses_service.create_response`.
2. Chat Completions rejects while building the image prompt:
   `chat_service._render_multimodal_prompt` raises on any message with
   tool calls, a tool call id or a name, and on assistant messages without
   content. Reasoning on an image chat is dropped without an error.
3. The image prompt type `MultimodalPrompt` (`kairyu/engine/prompt.py`)
   holds per message only a role and text / image parts; its wire codec
   (`prompt_to_wire` / `prompt_from_wire`, used by kairyu-proc) enforces
   exactly those fields.
4. `openai_backend` builds the upstream messages for an image prompt from
   those role / part messages only. The native wires already express tool
   calls, tool results and reasoning (`upstream_wire.responses_input`,
   `anthropic_conversation`) and images in user content (`input_image`).
5. "Upstream-rendered" is decided by `upstream_chat_models`: the build
   requires every replica of such a model to be a static `openai` backend
   whose profile renders native chat (vLLM). Kairyu's own engines
   (kairyu, engine loop, kairyu-proc, in-process vLLM, mock) reject image
   prompts outright and are unaffected.

## Framework admission

1. Broken shared contract: upstream-rendered models promise the caller's
   conversation reaches the upstream natively (m9 D2 amendment); with an
   image present the tool transcript cannot be expressed and the request is
   refused (`chat_service`, `responses_items`).
2. No extension point can carry a tool transcript inside an image prompt.
3. Independent uses: every example serving a VLM through
   `upstream_chat_models` (the V4.1, GLM and Qwen examples) and any client
   that sends an image after tool use (Open WebUI built-in tools, Codex or
   other agents attaching a screenshot in a later user turn).
4. Smallest mechanism: the image prompt's messages carry the caller's
   transcript fields; only upstream-rendered models accept them. Example
   policy (which tools, code interpreter defaults) stays in the example.

## Changes

1. `kairyu/engine/prompt.py`: `MultimodalMessage` gains optional transcript
   fields (`tool_calls`, `tool_call_id`, `name`, `reasoning_content`);
   content may be empty only for an assistant message with tool calls. The
   wire codec carries them exactly.
2. `kairyu/entrypoints/server/chat_service.py`: for a model in
   `upstream_chat_models`, `_render_multimodal_prompt` copies the validated
   transcript fields (reasoning included) instead of rejecting. Other
   models, AUTO / orchestrated models and Kairyu-template models keep
   today's rejection.
3. `kairyu/entrypoints/server/responses_service.py` /
   `responses_items.py`: the up-front rejection applies only to models not
   in `upstream_chat_models`.
4. `kairyu/engine/openai_backend.py`: an image prompt's upstream messages
   include the transcript fields, on the chat wire and through the existing
   native-wire converters for Responses and Messages callers.
5. Out of scope (unchanged, documented): images inside a tool result (Codex
   `view_image`), images on Anthropic Messages (rejected outright today),
   AUTO / orchestrated models, Kairyu's own engines.
6. Tests (test policy): cross-component contracts only. A Chat Completions
   and a Responses request with an image plus tool history reach an
   upstream-rendered model's upstream with image parts, tool calls, tool
   results and reasoning in order; the same request to a model outside
   `upstream_chat_models` is still a typed 400; the image prompt's wire
   codec round-trips the transcript. Existing tests asserting the rejection
   for upstream-rendered models are updated, not duplicated; base and head
   collection counts reported with a rationale per added test.
7. Docs: amendment in `docs/design/m9-truthful-api.md` (upstream chat:
   image chats carry the transcript) and the limit in
   `docs/design/m11-product.md`, `docs/design/m1-orchestration-and-interface.md`,
   `docs/deployment.md` (Codex: images beside tool history now work on
   upstream-rendered models; `view_image` output still rejected); one
   PROGRESS entry.

## Verification

Behavior only. Stop at the first failure, fix all known risks together,
re-run only the affected gates.

| # | Check | Pass |
|---|---|---|
| F0 | Local CPU | ruff; tests of the changed paths (server Responses / chat / upstream forwarding, openai backend, prompt); pytest collection base vs head reported |
| F1 | Restart | `deepseek-v4.1-flash-1017-8gpu` from this branch, `run.sh up` passes |
| F2 | Image beside tool history, GPU | `api-equivalence` gains cases on Responses and Chat Completions: a red image, an earlier tool call whose result holds a code word, a question for both; Kairyu answers 200 with the color and the code word, agrees with vLLM direct, and vLLM's counters show the caller's API |
| F3 | Regression, GPU | the example's `vision`, `tool-calling` and `api-equivalence` gates pass |
| F4 | Chat UI, GPU | `chat-ui` (323, 324, Red, knowledge answer) and `office` pass with the code interpreter on by default |

Afterwards: PR status, push, CI, report. Merging is the owner's decision.
