# llama.cpp as a Kairyu L1 worker: serve GGUF models with zero L2 change

Status: owner-approved 2026-10-04 (decisions under "Owner decisions");
amended by the PR #620 review: frequency/presence penalties are rejected,
`repetition_penalty` gets `repeat_last_n` = `max_model_len` (now required),
`/v1/messages/count_tokens` is declined, and `n` is limited to 1 (llama.cpp
reports usage per candidate). `docs/design/llamacpp-upstream.md` is
authoritative where this plan differs. Base:
`main` at `a8d242b`. llama.cpp reference: `ggml-org/llama.cpp` master
`46847e6` (tag `b11391`, 2026-10-04); every llama.cpp claim below was read in
that source (paths are relative to its repo root). The example runtime's base
`911f6cd` (`b11036`, 2026-09-18) was re-checked for every row of LCP-D2/D3/D4
and behaves the same.

## Goal

Run GGUF checkpoints under Kairyu. A `llama-server` process is the L1 engine.
Everything above L1 consumes it unchanged:

- L2: ReplicaPool, Router/profiles judge, Conductor, MoA, checklist answer roles.
- L3: Chat, Responses, Messages, the Chat UI.

Hard constraint: no code change in `kairyu/orchestration/`, `kairyu/dsl/` or
`kairyu/deploy/`, and none in the L3 server. All framework code changes stay
inside `kairyu/engine/`.

Non-goals (rationale in LCP-D1):

- native GGUF loading in Kairyu's own engine;
- in-process llama.cpp bindings;
- Kairyu-managed `llama-server` subprocesses;
- llama.cpp router (multi-model) mode;
- `discovery:` replica pools;
- llama.cpp embeddings and rerank.

## Decisions (proposed, to be recorded as LCP-D1..D6 in `docs/design/llamacpp-upstream.md`)

### LCP-D1. Integration shape: external `llama-server` behind the existing `openai` backend

Kairyu already composes external L1 workers this way. Every vLLM example runs
`vllm serve` as a compose service and attaches it with:

```yaml
backend: openai
options:
  upstream: vllm
```

Upstream differences live in one immutable `OpenAIRequestCapabilities`
profile (`kairyu/engine/openai_capabilities.py`, `_PROFILES`). The L2/L3
survey (below) found no code outside `kairyu/engine/` that branches on the
`upstream` string. A new named profile `upstream: llamacpp` is therefore the
whole integration seam.

Rejected alternatives:

| Alternative | Why not |
|---|---|
| In-process `llama-cpp-python` backend | Its `Llama` class generates one sequence at a time and is synchronous. Its bundled server serializes requests behind a global lock, and with `interrupt_requests=True` a new request interrupts the one currently streaming. Releases vendor llama.cpp up to ~6 weeks behind, while llama.cpp tags ~20 builds a day. Kairyu would have to rebuild slots, continuous batching, prompt cache and cancellation that `llama-server` already provides, and would gain a GIL and crash coupling. |
| Native GGUF loading in Kairyu's engine | Needs GGML k-quant/IQ/MXFP4 kernels, a GGUF tokenizer path and parity gates per architecture. transformers' GGUF loader dequantizes to dense weights for almost every architecture, which defeats GGUF. This would be a separate roadmap item, not an adapter. |
| Kairyu-managed `llama-server` subprocess (`backend: llamacpp`) | A second lifecycle stack. compose/Helm already own L1 process lifecycle for vLLM. It fails the framework admission gate (an existing extension point satisfies the need), so it stays out of this plan. |
| `upstream: generic` + capability overrides | Works for some requests, but cannot be made truthful: see the gaps in LCP-D2/D3 (wire renames, silent `tool_choice` fallback, `top_logprobs: 0`, assistant prefill, client-caused HTTP 500s). |

### LCP-D2. The `llamacpp` capability profile (what Kairyu forwards vs. rejects before dispatch)

llama-server silently ignores unknown JSON keys. Its `json_value` helper also
silently falls back to the default on a type mismatch
(`tools/server/server-common.h:43-55`). Every field Kairyu forwards must
therefore be one llama-server provably executes. Every other field fails
closed before dispatch, which the profile mechanism already does.

Accepted (`sampling_fields`):

- OpenAI core: `frequency_penalty`, `logprobs`, `max_tokens`, `n`,
  `presence_penalty`, `response_format`, `seed`, `stop`, `temperature`,
  `top_p`.
- `top_k`, `min_p`, `repetition_penalty`, `ignore_eos`.

All five `GENERATION_CONFIG_SAMPLING_FIELDS` are included. This is
required: the L2 profile judge sends `temperature=0.0` with nothing marked
omitted (`kairyu/orchestration/orchestrator.py:902-905`), so `top_p`, `top_k`,
`min_p` and `repetition_penalty` all count as active. Conductor role overrides
set `top_k`, `min_p` and `repetition_penalty` (`kairyu/orchestration/conductor.py:812-823`).

Rejected before dispatch, because llama-server does not execute them:

| Rejected | Reason |
|---|---|
| `min_tokens`, `stop_token_ids`, `skip_special_tokens=false`, `forced_token_ids` | Not executed by llama-server |
| `best_of` | `/v1/completions` raises `Unsupported param` as a `runtime_error`, so HTTP 500 (`server-common.cpp:1056-1059`) |
| `prompt_logprobs` | No prompt logprobs in llama-server |
| `priority` | llama-server has no priority; its deferred queue is FIFO |
| `strict: true` tools | `common/chat.cpp` never reads `strict` |

Other settings:

- `max_n`: none. llama-server hard-limits `n` to `1..n_parallel` and returns
  400 above that (`tools/server/server-schema.cpp:62-65`), so the deployment's
  slot count is the real ceiling and violations arrive as client errors.
- `parallel_tool_calls`: true.
- `strict_tools`: false.
- `priority`: false.
- `prompt_kinds`: `{text}`. A deployment adds `multimodal` with
  `allow_prompt_kinds` plus an `image_input_policy` when `--mmproj` is loaded.
- `chat_template_kwargs`: none built in. The deployment allowlists keys such as
  `enable_thinking` with `allow_chat_template_kwargs`. llama.cpp merges them
  into the template context.
- `max_tokens_wire_name`: `max_tokens` (an alias of `n_predict`).

### LCP-D3. Wire adaptations inside `OpenAICompatBackend` (L1 only)

Each row closes a gap where a `generic` forward would be wrong. Where a
reusable concept exists, it becomes a capability field (precedent:
`max_tokens_wire_name`) instead of another `upstream ==` branch.

| Kairyu intent | llama-server wire | Evidence / reason |
|---|---|---|
| `repetition_penalty` | `repeat_penalty` | Only `repeat_penalty` is parsed (`server-schema.cpp:130`); `repetition_penalty` would be silently ignored. New field `OpenAIRequestCapabilities.repetition_penalty_wire_name` (default `repetition_penalty`), mirroring `max_tokens_wire_name`. |
| `top_k = -1` (disabled) | `top_k: 0` | llama.cpp's "disabled" is 0. Its soft limit would clamp -1 to 0 (`server-schema.cpp` top_k `set_limits(0, …)`), but we map explicitly instead of relying on a clamp. |
| `tool_choice: {"type":"function","function":{"name":X}}` | `tools: [X]`, `tool_choice: "required"` | Only the strings `auto`/`none`/`required` parse (`common/chat.cpp:345-355`). An object hits the `json_value` type fallback and silently becomes `"auto"` (`server-common.cpp:1234`). Restricting `tools` to the named function and requiring a call is the exact OpenAI meaning. |
| `logprobs = k` | `logprobs: true`, `top_logprobs: max(k, 1)`; trim returned `top_logprobs` to `k` | `top_logprobs: 0` sets `n_probs = 0`, which disables probabilities entirely (`server-common.cpp:1429-1437`; `server-context.cpp` gates on `n_probs > 0`). The sampled token's logprob does not depend on `k`. |
| `assistant_prefill` | trailing assistant message + `continue_final_message: true` + `add_generation_prompt: false` | Supported, vLLM-compatible (`server-common.cpp:1322-1342`). Today the gate is `upstream != "vllm"` (`kairyu/engine/openai_backend.py:855-863`). Replace it with a capability flag `assistant_prefill` (true for `vllm` and `llamacpp`; vLLM behavior unchanged). Enables DTO-D15 `reasoning_continuation: chat` on llama.cpp final workers. |
| `/v1/messages/count_tokens` probe | `POST /tokenize {"content": prompt, "add_special": true}` → `len(tokens)` | Today it is vLLM-only (`openai_backend.py:984`). `add_special: true` mirrors vLLM `/tokenize` defaults. |
| Images (data URLs) | PNG/JPEG data URLs only; WebP re-encoded to PNG from the already-decoded Pillow raster | llama.cpp decodes images with stb_image, which lacks WebP. WebP decodes only in builds with ffmpeg on PATH (`tools/mtmd/mtmd-helper.cpp:404-419`). A failed decode raises `runtime_error` (`server-common.cpp:942`), which `ex_wrapper` maps to **HTTP 500** (`tools/server/server.cpp`). See LCP-D4. Pixels are unchanged (lossless re-encode). |
| `TemplatedPrompt` passthrough | rejected at config validation (`allow_templated_chat_passthrough` with `llamacpp`) | llama-server would template the Kairyu-rendered text a second time through its Jinja chat template. A `/completion` route is deferred (LCP-D6). |

These need no change, because existing parsing already matches
llama-server's output:

- `reasoning_content` in `message`/`delta`: read by `_upstream_reasoning`.
- native `tool_calls`: converted to Kairyu's `<tool_call>` text by `_message_text` / `_stream_tool_calls_text`.
- `usage.prompt_tokens_details.cached_tokens` (`server-task.cpp:370`): read by `_usage_from`.
- `stream_options.include_usage`: final empty-`choices` chunk (`server-schema.cpp:27`).
- `data: [DONE]` terminator.
- OpenAI-shaped errors.
- `exceed_context_size_error`: returned as 400, so it maps to `UpstreamClientError` and does not count as a replica failure.

### LCP-D4. No client-caused HTTP 5xx

`ReplicaPool` counts every non-`UpstreamClientError` toward ejection
(`kairyu/orchestration/replica.py:1553-1561`). It does not count 4xx, so that
one malformed client cannot eject the fleet (O1).

llama-server reports some client-input failures as 500, which breaks that
assumption. Each known path is closed before dispatch, not by parsing error
text:

- **Undecodable images.** Kairyu already decodes every image with Pillow
  before dispatch, and the L1 path re-encodes non-PNG/JPEG as PNG (LCP-D3).
  llama-server never receives an image it cannot decode.
- **Images to a server without `--mmproj`** ("image input is not supported",
  a `runtime_error`, so 500; `server-common.cpp:1156`). Prevented by
  configuration: Kairyu only declares `multimodal` when the deployment
  configures it. The deployment attests `/props.modalities.vision` (LCP-D5).
- **Tools without Jinja** (`tools param requires --jinja flag`, a
  `runtime_error`, so 500). Prevented by the deployment invariant `--jinja`
  (LCP-D5) and attested from `/props.chat_template_caps`.
- Requests combining `logprobs`, tools and streaming already return 400
  (`invalid_argument`, `server-common.cpp:1433`). That is a client error, not
  a replica failure.
- **Unified-KV exhaustion** ("Context size has been exceeded.", 500 to *every*
  processing slot). Prevented by the deployment invariant "no KV
  oversubscription" (LCP-D5).

### LCP-D5. Deployment invariants (example/operator-owned, attested at startup)

The llama-server HTTP API has no stability guarantee. llama.cpp's semver
(v0.1.0+, 2026-08) covers only the C API. Every llama-server deployment
therefore pins and attests the following.

**Build**
- Pin a digest-locked image `ghcr.io/ggml-org/llama.cpp:server-<variant>-b<N>@sha256:…`.
  Use a CUDA variant that supports the target GPU. SM120 must be verified
  against the `server-cuda` vs `server-cuda13` toolchains.
- Check `/props.build_info == "b<N>-<commit>"`.

**Model**
- `-m <first shard>.gguf`.
- `--alias <served name>` equal to Kairyu `options.model`.

**Template and reasoning**
- `--jinja` (the default since b7170; pass it explicitly).
- `--reasoning-format deepseek` explicitly, so reasoning always arrives in
  `reasoning_content`.

**Context and slots**
- `-np N` explicit, never `auto` (auto means 4 slots with unified KV).
- KV is never oversubscribed. Use either:
  - `--no-kv-unified` with `-c = N × L`, giving per-slot context `L`; or
  - `--kv-unified-per-slot L` with `-c` unset, so the pool is sized to N × L.
- Kairyu `max_model_len = L`. Attest it from `/props.default_generation_settings.n_ctx` and `total_slots`.
- `-c` and `-ngl` explicit and `-fit off`, so llama.cpp's automatic fitting
  cannot silently lower context or offload.
- Context shift stays off (the default since b6205).

**Sampling defaults**
- When Kairyu omits a generation-config field, llama-server applies:
  CLI flag > GGUF `general.sampling.*` > built-ins.
- The built-ins (temperature 0.8, top_k 40, top_p 0.95, min_p 0.05) are not neutral.
- The deployment sets `--temp/--top-k/--top-p/--min-p/--repeat-penalty` to
  the model card's values. This is the llama.cpp equivalent of vLLM's
  `--generation-config` choice. Attest them from `/props.default_generation_settings.params`.

**Kairyu engine options**
- `health_url: http://<host>:<port>/health`. Required: the default is `/readyz`
  (`kairyu/deploy/spec.py:71-79`), which llama-server does not serve.
- `api_key_env: null` on a private network, or `--api-key` with a matching env.
- `timeout_s` ≤ llama-server `-to`.
- `server.max_concurrency` sized to the slots. llama-server queues without
  limit (FIFO deferred tasks).

**Other llama-server flags**
- `--no-webui`; `--metrics` for `llamacpp:*` Prometheus series.
- Vision: `--mmproj` + `allow_prompt_kinds: [multimodal]` + `image_input_policy`, attested from `/props.modalities.vision`.

Kairyu never sends `X-Conversation-Id`, so a client disconnect cancels the
llama.cpp task for both streaming and non-streaming requests. Orchestration
cancellations (MoA losers, head/audit aborts) therefore free slots.

### LCP-D6. Explicitly unsupported or deferred

Each item fails closed today and stays so:

- **Batch/AsyncRequest on a llamacpp engine.** Tenants' `batch_priority`
  defaults to 1 (`kairyu/entrypoints/server/tenancy.py:43`) and the profile
  rejects non-zero `priority`. This limitation is already shared by the
  `generic`/`openai` upstreams.
- **SLO defer.** The request is shed instead of deferred; existing behavior.
- **`discovery:` pools.** `openai_replica_factory` hardcodes `upstream="kairyu"`
  and `/readyz` (`kairyu/deploy/registry.py:430-450`). Use static `replicas:`.
- **`TemplatedPrompt` passthrough via native `/completion`.** It needs
  BOS/special-token handling. The `prompt` string path adds BOS when the model's
  `add_bos_token` is set; token-ID prompts avoid it.
- **Router mode, LoRA, speculative decoding, embeddings/rerank.** Router mode
  would serve many models behind one upstream; one engine entry stays one model.
  Speculative decoding is a server flag, so it is example policy and needs no
  Kairyu change.

Inherited, not changed: for every `openai` engine listed in
`legacy_chat_models`, L3 flattens a text chat into one rendered user message
(`kairyu/entrypoints/server/chat_service.py:624-642`). The vLLM examples have
the same behavior. Changing it is an L3 change and out of scope.

## Framework admission (`.claude/rules/framework-boundary.md`)

1. **Missing shared contract.** Kairyu has no truthful OpenAI-compatible
   upstream contract for `llama-server`, the standard GGUF server. Code path:
   `OpenAIRequestCapabilities` / `_PROFILES` → `_sampling_payload` /
   `_payload` / `count_prompt_tokens_async` / image preparation in
   `kairyu/engine/openai_backend.py`.
2. **Why existing extension points fail.** `upstream: generic` plus overrides
   can only allow or deny a fixed field set. It cannot:
   - rename `repetition_penalty`;
   - rewrite a named `tool_choice` that llama-server silently turns into `auto`;
   - lift `top_logprobs: 0`, which silently drops logprobs;
   - enable `assistant_prefill` (vLLM-gated);
   - count tokens (vLLM-gated);
   - prevent image inputs that return HTTP 500 and eject replicas.
3. **Concrete use and observable regressions, independent of any example.**
   Any GGUF model on CPU, Apple, or consumer and datacenter GPUs. Each of the
   following is observable through `generic` today:
   - a request with `repetition_penalty: 1.1` returns output generated without
     the penalty;
   - a named `tool_choice` lets the model answer without the function;
   - `logprobs: 0` returns no logprobs;
   - one WebP request ejects a healthy replica.
4. **Smallest shared mechanism.** One profile, two capability fields
   (`repetition_penalty_wire_name`, `assistant_prefill`), three
   `llamacpp`-gated adapter branches (`tool_choice` rewrite, `top_logprobs`
   floor, `/tokenize`), and image re-encoding for that profile. The
   following stay in examples and deployments: the model, quantization, GPU
   layout, slots, context, sampling defaults, reasoning budget, template
   kwargs, mmproj, and flag choices.

## L2/L3 connectivity: zero change, with what L1 must provide

The survey covered every backend attribute read outside `kairyu/engine/`. It
found no `upstream` branching. The only name special cases are
`type(engine).__name__` labels, which already map `OpenAICompatBackend` to
`openai`.

| L2/L3 feature | Needs from L1 | Status with this plan |
|---|---|---|
| Engine served as a public model (`engines:` + `legacy_chat_models`) | Chat completions, usage, reasoning, tools | Works (LCP-D2/D3) |
| ReplicaPool over N llama-servers (static `replicas:`) | Same validation key on each replica, health URL, 4xx/5xx split | Works. The capability tuple hashes into `request_validation_key`/`admission_upper_bound_key`. `health_url` must be explicit; LCP-D4 keeps 5xx meaningful. |
| Prefix-aware placement | Nothing (pool-local text fingerprint) | Works. Inside each server, `cache_prompt` (default on) plus slot similarity reuse KV. |
| Router / profiles judge | All five generation-config sampling fields | Works (LCP-D2) |
| Conductor roles: tools, `reasoning_effort`, allowlisted `chat_template_kwargs` | `tool_choice` semantics, effort forwarded to the template | Works (LCP-D3 `tool_choice` rewrite). llama.cpp puts `reasoning_effort` into template kwargs (`server-common.cpp:1372-1380`), the same as vLLM. |
| DTO-D15 public-output floor, `reasoning_continuation: chat` | `assistant_prefill` + `reasoning_content` | Works (LCP-D3); `prefix` mode already works |
| MoA proposers and aggregator | `seed`; `n` up to the slot count | Works |
| Multimodal roles | Exact processed usage on multimodal requests | Works with `include_usage` (already requested) and LCP-D3 images |
| Checklist answer roles | Same as Conductor | Works (verifiers run on System One, not on this engine) |
| `/v1/messages/count_tokens` | `count_prompt_tokens_async` | Works (LCP-D3); returns 404 without it |
| `/backends`, `/readyz` | Declared metadata | Works from options: `model_revision`, `quantization_format` (e.g. `gguf:Q4_K_M`), `max_model_len`, `container_image_digest` |
| SLO defer, Batch/AsyncRequest, `discovery:` pools | `priority`, Kairyu upstream | Fail closed (LCP-D6) |

## Work plan

### Phase 0: contract gate against the real binary (CPU, no GPU)

New verification gate `l1.correctness.llamacpp_upstream_contract`. Files:
`verification/l1/correctness/llamacpp_upstream_contract.py` and a
`verification/registry.toml` entry (`scope = "l1"`, `kind = "correctness"`).

The gate:

- Starts the pinned CPU image `ghcr.io/ggml-org/llama.cpp:server-b<N>` with a
  small public chat GGUF that has a reasoning- and tool-capable template (for
  example a ~0.6B Qwen3-family Q8_0).
- Drives it through `OpenAICompatBackend(upstream="llamacpp")` and asserts
  every row of LCP-D2/D3/D4:
  - each accepted field changes `/props`-visible slot params or output as
    specified;
  - each rejected field fails before dispatch;
  - named `tool_choice` produces a call to that function;
  - `logprobs: 0` returns sampled-token logprobs;
  - assistant prefill continues;
  - WebP is accepted;
  - prompt overflow returns 400;
  - disconnect cancels the task, seen in `/slots` `is_processing`;
  - usage and `cached_tokens` are present on a repeated prefix;
  - the stream ends with `[DONE]`.
- Records the raw SSE/JSON of a reasoning + tool-call exchange as fixtures
  for the Phase 1 unit tests.

Re-run on every pin bump. It is the regression guard for llama.cpp's
unversioned HTTP API.

Exit: every row passes on the pinned build, or the corresponding plan row is
amended before Phase 1.

### Phase 1: L1 implementation (`kairyu/engine/` only)

1. `kairyu/engine/openai_capabilities.py`:
   - add the `llamacpp` profile (LCP-D2);
   - add the fields `repetition_penalty_wire_name` and `assistant_prefill`,
     appended to keep positional ABI;
   - set `assistant_prefill=True` on `vllm`.
2. `kairyu/engine/openai_backend.py`:
   - `_sampling_payload`: use the wire name and map `top_k -1` to `0` for `llamacpp`;
   - `_payload`: named `tool_choice` rewrite; `top_logprobs` floor and trim
     (non-stream and stream);
   - `_validate_request_structure`: the `assistant_prefill` capability replaces
     the vLLM check;
   - `count_prompt_tokens_async`: llama.cpp `/tokenize`;
   - image preparation: re-encode non-PNG/JPEG to PNG for `llamacpp`.
3. `kairyu/engine/config_validation.py`:
   - reject `allow_templated_chat_passthrough: true` and
     `completion_reasoning_end_tag` with `upstream: llamacpp`;
   - validation of `assistant_prefill` follows the capability.

No change to `kairyu/orchestration/`, `kairyu/dsl/`, `kairyu/deploy/`, or
`kairyu/entrypoints/`. The plan's acceptance check is
`git diff main --stat -- kairyu/orchestration kairyu/dsl kairyu/deploy kairyu/entrypoints`
showing nothing.

### Phase 2: design and documentation

- `docs/design/llamacpp-upstream.md` with LCP-D1..D6, plus a one-paragraph
  "llama.cpp upstream amendment" under m1 D1 pointing to it.
- `docs/deployment.md`:
  - a `llamacpp` row in the upstream profile table (`:567-573`);
  - a GGUF recipe covering the LCP-D5 invariants for Docker CUDA and for a
    bare-metal Metal/CPU `llama-server` (same `kairyu.yaml`).
- `PROGRESS.md`: one `[design]` Change Log entry and a "What works today"
  line, in the same commit as the design doc.

### Phase 3: GPU-verified examples (example-owned policy)

Model (owner, Q1): [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B)
`gguf/Winnow-12B-Q8_0.gguf` (12,669,646,592 bytes, sha256 `b710efc4…18ea`)
with `gguf/mmproj-F16.gguf` at model revision `b2b14213`. It is a merged
Gemma 4 12B IT fine-tune for typed decisions that also chats and takes images.

Its runtime is not stock llama.cpp. [EldanRing/winnow-inference](https://github.com/EldanRing/winnow-inference)
(`77d1458`) pins llama.cpp `911f6cd` (`b11036`) plus four patches:

- Gemma 4 tied embedding;
- bounded SWA fork;
- classifier head;
- a server patch that adds `/v1/systemone`, Jev's typed-decision wire shape.

Its `winnow-server` keeps llama-server's `/v1/chat/completions`, `/health`
and `/props` unchanged. The example therefore builds that image (`CUDA_ARCH=120`,
digest recorded in `example.json`) as its L1. The patched runtime is
example-owned, like the vLLM SM120 overlays.

Because Kairyu's existing System One forwarder is wire-transparent
(`kairyu/engine/systemone.py`; `usage.input_tokens`/`output_tokens`), each
example also publishes Winnow's typed decisions through a `systemone:` entry,
with no framework change.

Two examples (Q4):

- `examples/winnow-12b-q8-1gpu/`: one `winnow-server` on one RTX PRO 6000.
- `examples/winnow-12b-q8-dp8-8gpu/`: eight `winnow-server`s, one per card,
  behind one ReplicaPool for chat and one multi-replica `systemone:` entry.

They follow the vLLM 1-GPU and DP8 examples' structure:

- `compose.yaml` with the pinned llama-server, Kairyu and Open WebUI;
- `kairyu.yaml` with `upstream: llamacpp` and an explicit `health_url`;
- `example.json` with the model repo, revision, GGUF file SHA-256, image
  digest, slots, context and sampling defaults;
- `run.sh`;
- `verify.sh` with `serving`, `tool-calling`, `vision` (if mmproj) and
  `attest` gates;
- `MEASUREMENTS.md`.

`attest` compares `/props` (build, slots, per-slot `n_ctx`, sampling
defaults, `chat_template_caps`, modalities) with `example.json` and
`kairyu.yaml`. The `examples/README.md` title and table mention the
llama.cpp L1.

### Phase 4 (optional; needs separate authorization)

Each item below needs its own admission case:

- an L1-side `/props` attestation inside the backend, catching
  `max_model_len` drift without the example;
- `/completion` token-ID passthrough for Kairyu-owned templates;
- a priority story for Batch/AsyncRequest on non-priority upstreams.

## Tests (CLAUDE.md test policy)

Base collection: `uv run pytest --collect-only -q` → **6317 collected
(369 deselected)** at `a8d242b`; **6337** at the PR #620 head after the
second review round, with the same command. This is additive work, so the
count rises. Each new test protects a concrete llama.cpp-specific regression
that an existing test cannot see.

| Test area (file) | Kind | Regression it protects |
|---|---|---|
| Wire body for `upstream="llamacpp"` (`tests/unit/test_openai_backend.py`, one parametrized test) | new | `repeat_penalty` rename, `top_k` 0, named `tool_choice` → restricted tools + `required`, `top_logprobs` floor and trim. Each case is a silent-ignore in llama.cpp. |
| Recorded llama-server stream/unary fixture → `GenerationResult` | new | Wire drift in `reasoning_content`, `tool_calls`, `usage.cached_tokens` and `[DONE]` from the pinned build (fixture from Phase 0) |
| WebP → PNG before dispatch | new | Client-caused 500 ejects a replica (LCP-D4) |
| Existing assistant-prefill test parametrized over `vllm` and `llamacpp` | extended (not duplicated) | The prefill gate moved to a capability; it must stay on for vLLM |
| Existing pre-dispatch rejection parametrization gains `llamacpp` rows | extended | `min_tokens`, `stop_token_ids` and `skip_special_tokens` (ignored upstream), frequency/presence penalties (applied to prompt tokens upstream) and `n > 1` (usage reported per candidate) must never reach llama-server |
| Winnow examples (`tests/unit/test_winnow_llamacpp_examples.py`, each over both examples) | new | Slot geometry, admission and health URL drift from what winnow-server runs; `attest` passing without sampling defaults; a full disk blocking `run.sh down`; the streaming tool-call gate passing a text-only reply |
| `known_openai_upstreams()` exact tuple, deployment-spec upstream parametrization | updated in place | Existing tests; no new static-list tests are added |

No tests are added for profile constants beyond these behavior tests.
Construction-time validation stays in product code. The real-binary
contract lives in the Phase 0 verification gate, not in pytest. The final
report states the head count and this rationale.

## Risks

- **HTTP API drift.** Pin the build, attest `build_info`, and re-run the Phase 0
  gate on every bump.
- **`response_format` fidelity.** llama.cpp turns an unsupported regex
  `pattern` into "any string" with only a server-log warning
  (`common/json-schema-to-grammar.cpp:397, 975-981`). Invalid schemas do
  return 400. See Q2.
- **Sampling-default surprise.** Covered by LCP-D5 explicit flags and attestation.
- **Latency/throughput.** llama.cpp is not the A6 performance path. GGUF serving
  is a capability, not a performance claim. `MEASUREMENTS.md` records numbers
  without gates beyond serving SLOs the owner sets.
- **Logprob semantics.** These are pre-sampling softmax probabilities (the
  llama.cpp default, the same as vLLM's raw default). Documented, not altered.

## Owner decisions (2026-10-04)

- **Q1 (example model).** Winnow-12B Q8_0 (see Phase 3).
- **Q2 (`response_format` with `json_schema`).** Forward it unchanged and
  document llama.cpp's relaxation of unsupported regex `pattern`.
- **Q3 (Batch/AsyncRequest).** Keep the existing fail-closed limitation
  (LCP-D6).
- **Q4 (replicas).** Include a DP example: several llama-servers behind one
  ReplicaPool.
