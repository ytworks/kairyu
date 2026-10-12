# Upstream-rendered chat: send a pool model's conversation as native messages

Status: **Approved 2026-10-12 (owner: fix Kairyu before the DeepSWE runs,
opt-in model list, max effort for the runs).** Amended 2026-10-12 (owner): the
DeepSWE runs and the wiki report moved to issue #649; this PR is verified by
the nine example gates and the pre-benchmark check on both examples and merges
when both pass and CI is green.

## Problem

A tool-using agent's multi-turn conversation reaches a vLLM-served model
through Kairyu as one flattened user message, so the model's own chat template
never sees the turns:

- `legacy_chat_models` text chats are rendered by `render_chat`
  (`kairyu/entrypoints/server/chat_service.py`, `_render_prepared_prompt`) into
  `system: ...\nuser: ...\nassistant: ...<tool_call>...\ntool: ...\nassistant:`.
- `OpenAICompatBackend._payload` (`kairyu/engine/openai_backend.py`) sends that
  text as `messages=[{"role": "user", "content": text}]` plus `tools`.
- This is the documented legacy contract (m9 D2; LCP-D6: "Text chats of
  `legacy_chat_models` reach every `openai` upstream as one rendered user
  message"). Image chats already go to vLLM as native role messages.

Observed (2026-10-12, DeepSWE on `glm-5.3-flash-6gpu`, run
`deepswe-glm-5.3-flash-6gpu-8w-20261012-r1`, stopped and unpublished): from
the second agent turn on, GLM answered with short text and no tool call; all
23 finished tasks ended in `RepeatedFormatError`. The same second-turn request
sent to the same vLLM directly returned a structured tool call (2 of 2); a
capture of Kairyu's upstream body showed the single flattened user message.
The DeepSeek reference run `deepswe-full-8w-20260912-r1` (67/113) went
through the same flattening (`deepseek-v4.1-flash-8gpu`, `legacy_chat_models`).

## Decision

Add a model-scoped chat policy `upstream_chat_models`: the upstream renders
the model's own chat template, so Kairyu sends a text chat's validated
messages natively.

- `DeploymentSpec.upstream_chat_models: frozenset[str]`. A name must be a
  served `engines:`/`pools:` model, not an orchestration, and cannot also be
  in `legacy_chat_models` or `chat_templates`.
- Startup (builder chat-policy resolution, before any backend is built):
  every replica of such a model must be a static `openai` backend whose
  upstream profile declares the `conversation_prefix` capability (today
  `upstream: vllm`); otherwise the build fails. Discovered pools fail.
- Text chats of these models (Chat Completions, Responses, Messages, batch and
  AsyncRequest all use `validate_chat_request(_async)`): the validated message
  list (the same list the legacy renderer receives, including the
  `parallel_tool_calls: false` constraint text) goes in
  `GenerationRequest.conversation_prefix` with an empty text prompt, so
  `_payload` sends exactly those messages. `tools`, `tool_choice`,
  `parallel_tool_calls` and `reasoning_effort` are forwarded as today.
- Unchanged: image chats (already native), `chat_template_kwargs` on text
  requests (still rejected for template-less models), `legacy_chat_models`
  behavior, every other example.
- Token counts follow automatically: the vLLM `/tokenize` body is built from
  the same `_payload` (m9 D1 amendment 2026-10-10).
- Usage fallback (owner review): when the upstream reports no usage, the
  response, ledger and metrics count the conversation's JSON with the prompt
  text (`metering.metered_prompt`), as the admission bound does.
- Known limit: KV-aware prefix placement reads the prompt text, which is
  empty for these requests, so a multi-replica pool of such a model places by
  load and session affinity only. Both target examples run one replica.

Rejected alternatives: changing `legacy_chat_models` for every vLLM upstream
(silently changes 11 GPU-verified examples); a one-role L2 DSL with
`native_conversation` per example (turns single-model examples into
orchestrated models and leaves the pool path broken).

Framework admission (`.claude/rules/framework-boundary.md`):

1. Broken shared contract: the pool chat path (no L2) cannot deliver a
   caller's conversation to an upstream that owns the chat template
   (`chat_service._render_prepared_prompt` -> `render_chat`;
   `openai_backend._payload`).
2. `native_conversation` (m1 D8 amendment 2026-10-09) exists only on L2
   publisher roles; a pool model would have to become an orchestration.
3. Independent use and observable regression: any vLLM-served model driven by
   a multi-turn tool agent; the upstream body is one user message. m1 D8
   measured the effect on DeepSeek-V4.1 (calls written as text 11/12 with a
   JSON transcript, 0/12 natively).
4. Smallest mechanism: one model list plus the existing
   `GenerationRequest.conversation_prefix`; which models opt in stays in the
   examples.

## Changes

| Layer | Owner | File | Change |
|---|---|---|---|
| L3 | framework | `kairyu/deploy/spec.py` | `upstream_chat_models` field and validation |
| L3 | framework | `kairyu/deploy/builder.py` | resolve the policy, startup capability check, pass the set to the app, batch and AsyncRequest workers |
| L3 | framework | `kairyu/deploy/validation.py` | map the new startup error to its field |
| L3 | framework | `kairyu/entrypoints/server/chat_service.py` | native messages for listed models; policy overlap check |
| L3 | framework | `kairyu/entrypoints/server/{app,responses_service,messages_service,extra_routes}.py`, `kairyu/batch/worker.py`, `kairyu/async_requests/worker.py` | thread the set next to `legacy_chat_models` |
| Docs | framework | `docs/design/m9-truthful-api.md` | D2 amendment |
| Config | example | `examples/{glm-5.3-flash-6gpu,deepseek-v4.1-flash-6gpu}/kairyu.yaml`, `README.md` | switch to `upstream_chat_models`; correct the comments |
| Docs | example | both examples' `MEASUREMENTS.md` | GPU evidence |
| Docs | - | `PROGRESS.md` | amendment and progress entries (old entries archived first) |

## CPU tests

1. Route: a multi-turn conversation with an assistant tool call and a tool
   result, sent to a listed model on a fake vLLM upstream, arrives upstream as
   the same messages, with `tools` and `reasoning_effort`.
2. Startup: listing a model whose replica's upstream lacks the capability
   (`llamacpp`) fails the build.
3. Usage fallback (review): a stream from an upstream that reports no usage
   counts the conversation in the response usage and the usage ledger.

Local: `uv run ruff check .` and the touched test files with
`CUDA_VISIBLE_DEVICES=`; the full suite runs in GitHub CI.

## GPU verification (per example, on the PR head; GLM, then DeepSeek)

1. Start with every other example stopped and GPUs 0-5 idle; plain
   `./run.sh up`.
2. All nine example gates (`l1`, `serving`, `completed`, `tool-calling`,
   `vision`, `reasoning`, `cancellation`, `long-context`, `restart`) against
   the existing `MEASUREMENTS.md` pass criteria: every text request changes
   shape, so none is omitted. About 1.5 h each.
3. Pre-benchmark check (about 15 min): the stopped GLM run's second-turn
   request (5 messages, tool result included) sent 5 times through Kairyu
   returns 5 structured tool calls; one captured Kairyu-to-vLLM body carries
   the 5 messages natively and `reasoning_effort: max`; vLLM `/tokenize` of
   that body renders the max marker (GLM `Reasoning Effort: Max`, DeepSeek
   max = 100).
4. The DeepSWE max-effort runs and the wiki report are tracked in issue #649
   (owner decision 2026-10-12), not in this PR.

A passed step is reported and the work continues. A failed gate does not stop
the remaining gates; the failures are analyzed and reported together, and
nothing is changed or merged without the owner. The PR merges when every gate
and the pre-benchmark check pass on both examples and CI is green.
