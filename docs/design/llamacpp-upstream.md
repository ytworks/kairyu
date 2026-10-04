# llama.cpp Upstream (GGUF models as Kairyu L1 workers)

Status: **Accepted 2026-10-04 (PR #620)**. CPU contract gate passes against
stock llama.cpp `b11391` (`46847e6`). Winnow-12B example GPU gates are pending.
Plan: `docs/superpowers/plans/2026-10-04-llamacpp-gguf-l1-upstream.md`.
Gate: `l1.correctness.llamacpp_upstream_contract`
(`verification/l1/correctness/llamacpp_upstream_contract.py`).

## Goal

Serve GGUF checkpoints under Kairyu. A `llama-server` process is the L1 engine.
Kairyu L2 (ReplicaPool, Router/profiles judge, Conductor, MoA, checklist answer
roles) and L3 (Chat, Responses, Messages, Chat UI) consume it unchanged. All
framework code lives in `kairyu/engine/`. `kairyu/orchestration/`,
`kairyu/dsl/`, `kairyu/deploy/` and `kairyu/entrypoints/` do not change.

## Decisions

### LCP-D1. External `llama-server` behind the `openai` backend

A llama.cpp server is attached exactly like a vLLM server:

```yaml
backend: openai
options:
  upstream: llamacpp
```

plus an explicit `health_url` (see LCP-D5). Its lifecycle belongs to
compose/Helm, as for vLLM. No code outside `kairyu/engine/` branches on the
`upstream` string, so a new immutable `OpenAIRequestCapabilities` profile is
the whole seam.

Rejected alternatives:

| Alternative | Why not |
|---|---|
| In-process `llama-cpp-python` | Its `Llama` generates one sequence at a time. Its bundled server serializes requests and lets a new request interrupt the one streaming. Releases lag llama.cpp by weeks. |
| Native GGUF loading in Kairyu's engine | Needs GGML k-quant/IQ kernels and per-architecture parity. Transformers' GGUF path dequantizes. A separate roadmap item. |
| Kairyu-managed `llama-server` subprocess | A second lifecycle stack, which fails framework admission (compose/Helm already own L1 lifecycle). |
| `upstream: generic` + overrides | Cannot rename `repetition_penalty`, rewrite a named `tool_choice`, lift `top_logprobs: 0`, enable assistant prefill, count tokens, or keep client images from becoming HTTP 500. |

### LCP-D2. Profile contents

llama-server silently ignores unknown JSON keys. Its `json_value` helper also
falls back to the default on a type mismatch, logging only a server warning.
The profile therefore forwards only fields llama-server executes; everything
else fails closed before dispatch.

**Accepted (`sampling_fields`)**
- The OpenAI core: `frequency_penalty`, `logprobs`, `max_tokens`, `n`,
  `presence_penalty`, `response_format`, `seed`, `stop`, `temperature`, `top_p`.
- `top_k`, `min_p`, `repetition_penalty`, `ignore_eos`.

All five generation-config fields are included because the L2 profile judge
sends `temperature=0.0` with nothing omitted.

**Rejected before dispatch**
- `min_tokens`, `stop_token_ids`, `skip_special_tokens=false`,
  `forced_token_ids`, `best_of`, `prompt_logprobs`, `priority`.
- `strict: true` tools: llama.cpp never reads `strict`.

**Other settings**
- `n`: llama-server hard-limits it to `1..n_parallel` and answers HTTP 400
  above that (a client error).
- Upstream 400s (`n` above the slot count, prompt overflow) do not count
  against replica health. Kairyu's L3 reports them as 502 `backend_error`
  without upstream text; that is the existing policy for every upstream,
  vLLM included.
- `parallel_tool_calls` is forwarded.
- `chat_template_kwargs` are allowlisted per deployment with
  `allow_chat_template_kwargs`.
- Multimodal input is opt-in with `allow_prompt_kinds: [multimodal]` plus an
  `image_input_policy`.

### LCP-D3. Wire adaptations (in `OpenAICompatBackend`)

| Kairyu intent | llama-server wire | Why |
|---|---|---|
| `repetition_penalty` | `repeat_penalty` (capability `repetition_penalty_wire_name`) | Only `repeat_penalty` is parsed. |
| `top_k = -1` | `top_k: 0` | llama.cpp disables top-k with 0. |
| `tool_choice` naming function X | `tools: [X]`, `tool_choice: "required"` | Only string `tool_choice` values parse; an object silently becomes `"auto"`. |
| `logprobs = 0` | `top_logprobs: 1`; returned alternatives trimmed to 0 | `top_logprobs: 0` sets `n_probs = 0`, which disables probabilities. |
| `assistant_prefill` | trailing assistant message, `continue_final_message: true`, `add_generation_prompt: false` (capability `assistant_prefill`, shared with `vllm`) | Native llama.cpp continuation. |
| `/v1/messages/count_tokens` | `POST /tokenize {"content", "add_special": true}` → `len(tokens)` | Matches vLLM `/tokenize` defaults. |
| WebP image | re-encoded as lossless PNG from the validated raster | stb_image has no WebP. Without ffmpeg the decode fails as HTTP 500. |
| `TemplatedPrompt` passthrough | rejected at configuration | llama-server would template the rendered text again. |

Already compatible without adaptation:

- `reasoning_content` deltas;
- native `tool_calls` (normalized to Kairyu's `<tool_call>` text);
- `usage.prompt_tokens_details.cached_tokens`;
- the usage chunk sent for `stream_options.include_usage`;
- `[DONE]`;
- `exceed_context_size_error` as HTTP 400.

`response_format` (`json_schema`) is forwarded unchanged (owner decision
2026-10-04). llama.cpp converts the schema to a grammar. A regex `pattern` it
cannot express is relaxed to "any string" with only a server-log warning;
invalid schemas return HTTP 400. Deployments that need exact `pattern`
enforcement must not rely on llama.cpp's converter.

Returned logprobs are pre-sampling softmax probabilities, the llama.cpp
default and the same as vLLM's raw default.

### LCP-D4. No client-caused HTTP 5xx

`ReplicaPool` counts every non-4xx failure toward ejection. llama-server
reports some client-input failures as HTTP 500, so each known path is closed
before dispatch:

- **Undecodable images:** Kairyu decodes every image with Pillow first and
  re-encodes WebP (LCP-D3).
- **Images without `--mmproj`:** `multimodal` is declared only when
  configured; the deployment attests `/props.modalities.vision`.
- **Tools without Jinja:** the deployment runs `--jinja` and attests
  `/props.chat_template_caps.supports_tool_calls`.
- **Unified-KV exhaustion:** this fails every processing slot. Prevented by
  never oversubscribing KV (LCP-D5).

### LCP-D5. Deployment invariants (operator/example-owned, attested from `/props`)

**Pinning**
- Pin a digest-locked image.
- Attest the commit suffix of `/props.build_info` (`b<N>-<commit>`). The build
  number depends on clone depth, so compare the commit, not `N`.

**Model and template**
- `-m <first shard>.gguf`.
- `--alias <served name>` equal to the Kairyu `options.model`.
- `--jinja`.
- Reasoning either off or `--reasoning-format deepseek`, so reasoning arrives
  as `reasoning_content`.

**Context and slots**
- `-np N` explicit. Auto means 4 slots sharing a unified KV pool.
- KV is never oversubscribed. With an explicit `-np`, the KV cache is split per
  slot (`n_ctx / N`). The alternative is `--kv-unified-per-slot L` with `-c`
  unset.
- Kairyu `max_model_len` = per-slot context, attested from
  `/props.default_generation_settings.n_ctx` and `total_slots`.
- `-fit off` with explicit `-c`/`-ngl`.
- Context shift stays off (the default).

**Sampling defaults**
- llama-server fills omitted sampling fields from CLI flag > GGUF
  `general.sampling.*` > built-ins.
- The built-ins are temperature 0.8, top_k 40, top_p 0.95, min_p 0.05; they
  are not neutral.
- Set the model card's values explicitly and attest them from
  `/props.default_generation_settings.params`.

**Kairyu options**
- `health_url: http://<host>:<port>/health`. The default `/readyz` does not
  exist on llama-server, and it returns 503 while loading.
- `server.max_concurrency` sized to the slots: llama-server queues without limit.

Kairyu never sends `X-Conversation-Id`, so a client disconnect cancels the
llama.cpp task (verified by the gate) and orchestration cancellations free slots.

### LCP-D6. Unsupported (fail closed)

- **Batch/AsyncRequest:** tenant `batch_priority` ≥ 1 versus no upstream
  priority; owner-accepted limitation, shared with `generic`/`openai`.
- **SLO defer:** the request is shed instead.
- **`discovery:` pools:** these hardcode `upstream="kairyu"`; use static
  `replicas:`.
- **Not used:** `/completion` token passthrough, router mode, LoRA.

Text chats of `legacy_chat_models` reach every `openai` upstream as one
rendered user message. This is an inherited L3 behavior, the same as for vLLM.

## Evidence

The gate ran on the CPU against a stock `b11391` build with a synthetic Gemma 4
vocabulary model. All rows pass. The raw negative controls confirm each
adaptation is needed:

- `repetition_penalty: 3.0` output equals the baseline, while `repeat_penalty`
  changes it.
- A raw named-`weather` object calls `lookup`.
- Raw `top_logprobs: 0` returns `logprobs: null`.
- The server logs `Wrong type supplied for parameter 'tool_choice', using
  default value`.

The recorded request/reply pairs are the unit-test fixtures in
`tests/fixtures/llamacpp/`.

**End to end on the CPU (2026-10-04).** Two such llama-servers ran as a static
ReplicaPool behind `kairyu serve`, with no L2/L3 code change:

- The prober marked both ready through their `/health` URLs.
- Chat, the Responses API, `/v1/messages/count_tokens` (llama.cpp `/tokenize`)
  and `logprobs` with `top_logprobs: 0` (sampled-token logprobs, no
  alternatives) answered 200.
- `min_tokens` failed with 400 before dispatch.
- A concurrent burst spread over both replicas.
- Killing replica 0 failed one request (502), ejected the replica and moved
  traffic to replica 1. After a restart the prober restored it and traffic
  spread again.

The random model's tool calls run out of tokens with arbitrary arguments, and
L3 correctly refused them as `tool_choice_not_satisfied`. The tool path is
therefore covered by the replayed fixtures and the examples' GPU
`tool-calling` gate.

That run also showed that llama.cpp's Gemma 4 tool-call grammar (b11391)
does not constrain arguments to the schema: an `enum` with
`additionalProperties: false` still admitted arbitrary text. This is one more
reason `strict_tools` stays off for this profile.

## Winnow-12B examples

`examples/winnow-12b-q8-1gpu` and `examples/winnow-12b-q8-dp8-8gpu` serve
EldanRing/Winnow-12B Q8_0, a merged Gemma 4 12B fine-tune for typed decisions.

**Runtime.** `winnow-server` from EldanRing/winnow-inference: llama.cpp
`911f6cd` (`b11036`) plus four patches. Every LCP-D2/D3/D4 behavior was
re-read at that commit and is unchanged. The patched runtime is example-owned.

**System One.** Its `/v1/systemone` speaks Jev's typed-decision wire shape.
The examples publish it through Kairyu's existing `systemone:` forwarder
(m11 D8) with configuration only.
