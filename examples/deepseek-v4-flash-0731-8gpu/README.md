# DeepSeek-V4-Flash-0731 on 8 x RTX PRO 6000 Blackwell

This is the single-model eight-GPU example (see [`../README.md`](../README.md)
for all three environments). It starts the complete local stack:

```text
Open WebUI -> Kairyu L3 (:8002) -> vLLM L1 (TP8 + EP8, all 8 GPUs)
```

The default is tuned for the exact hardware and checkpoint: mixed FP4/FP8
weights, FP8 KV cache, native 1,048,576-token context, prefix caching,
full/piecewise CUDA Graphs, and five-token DSpark speculation.
Five draft tokens match the checkpoint's DSpark block size. vLLM owns the
prompt, with the dp2 example's L1 settings (FN-D8 amendment, 2026-10-17): it
renders every request with the checkpoint's own encoder (`--tokenizer-mode
deepseek_v4`), separates reasoning, and parses DSML tool calls into OpenAI tool
calls (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`). Kairyu
sends each request to vLLM in the API its caller used (`kairyu.yaml` lists the
model in `upstream_chat_models`, `upstream_responses_models` and
`upstream_messages_models`): a chat as the caller's own messages, a Responses
request to vLLM's `/v1/responses`, an Anthropic Messages request to vLLM's
`/v1/messages`. Ordinary requests default to direct chat mode
(`--default-chat-template-kwargs` turns thinking off); an explicit reasoning
effort (`low`, `high`, `max`) selects thinking mode. On the Responses wire,
whose effort vocabulary ends at `xhigh`, Kairyu sends `max` as `xhigh`; the
pinned tokenizer would fold `xhigh` to the high tier, so this example builds a
one-line overlay (`vllm-effort.Dockerfile`, `patch_effort.py`) that renders it
as the max tier. Open WebUI defaults model output to 32,768 tokens so normal
answers are not cut into manual **Continue Response** steps.
The official MegaMoE recommendation is not used: the pinned implementation
rejects SM120 as SM100-only, so this example explicitly disables DeepGEMM and
uses vLLM's supported SM120 MoE path.
The official FP4 indexer cache is also disabled because the pinned vLLM
implementation supports it on SM100 datacenter Blackwell, not SM120.

## Start

```sh
./run.sh
```

The command verifies exactly eight matching GPUs, reuses or builds the pinned
SM120 vLLM source revision and this example's effort overlay if its image is
absent, downloads and hashes the
exact model revision unless its content attestation is already present, builds
Kairyu, waits for readiness, and prints:

```text
OpenAI API: http://127.0.0.1:8002/v1
Chat UI:    http://127.0.0.1:3000
```

The first build/download is large. Later starts use the content attestation.
Set `VERIFY_MODEL=1` to rehash every checkpoint file. Lifecycle commands are
`./run.sh up`, `./run.sh status`, `./run.sh logs`, and `./run.sh down`.

## Chat UI

Open WebUI talks to Kairyu over the OpenAI Responses API (m11 D7 amendment,
2026-10-17): `compose.yaml` sets the connection's `api_type` to `responses`.
Open WebUI then forwards some chat-only fields unchanged (Chat Controls'
reasoning effort, `seed`, `top_k`) and replays saved answers with its own
bookkeeping fields, which Kairyu's strict `/v1/responses` rejects, so the UI
needs the global filter in `webui-responses-filter.py`: it moves the effort to
`reasoning.effort`, drops fields the Responses API lacks, keeps only Responses
fields on saved items, and sends `store: false`. Sign-in is enabled here, so
`./run.sh up` installs the filter only when `WEBUI_ADMIN_EMAIL` and
`WEBUI_ADMIN_PASSWORD` are set (the administrator Open WebUI creates on a fresh
volume) and switches the saved Kairyu connection to `api_type=responses` when
an existing Open WebUI volume still has the chat connection it saved earlier;
without them it prints a warning, and an administrator installs the file under
Admin Panel > Functions as a global filter and sets Admin Panel > Settings >
Connections > Kairyu > API Type to Responses.

## Verification

```sh
./verify.sh list
./verify.sh serving
./verify.sh api-equivalence
```

`serving` records TTFT, TPOT, requests/s, and output tokens/s on Kairyu's
`/v1/responses` for an approximately 8K-token input and up to 256 generated
tokens at concurrency 1, 8, 16, and 32. The Responses API has no
`ignore_eos`/`min_tokens`, so each request ends naturally within
`max_output_tokens`; a row passes when every request returns 1 to 256 output
tokens. These rows are not comparable with the fixed-length chat rows recorded
before 2026-10-17. `api-equivalence` sends the same Responses requests (text,
stream, each effort, a tool round trip, a length cap, and Open WebUI's body) to
vLLM directly and through Kairyu, requires the same Responses structure from
both, and reads vLLM's request counters to confirm that Kairyu reached it on
`/v1/responses`. Artifacts go to
`verification/results/examples/deepseek-v4-flash-0731-8gpu/<UTC-run-id>/`.

GPU verification of the 2026-10-17 configuration (Responses connection, vLLM
rendering, effort overlay) is pending. The locked results in
[MEASUREMENTS.md](MEASUREMENTS.md) were measured on the previous configuration
(Kairyu-rendered `/completions`, fixed 256-token chat rows): the warm run
reached 168.01 output tokens/s with 220.10 ms median TTFT at concurrency 1 and
972.93 output tokens/s at concurrency 32. Model and product evaluations are
invoked explicitly through `python -m evals`.

## Public performance context

No public result with the exact combination (0731 checkpoint, current vLLM
SM120 patch, and eight RTX PRO 6000 Blackwell cards) was found, so the committed
local report is the authoritative number for this example. Useful but
non-comparable public context is:

- The official [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  recommends FP8 KV, block size 256, FP4 indexer cache, expert parallelism,
  DeepGEMM Mega MoE, and DSpark.
- The official [vLLM recipe for this exact GPU class](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?features=tool_calling%2Creasoning&hardware=b300)
  selects TP8 plus expert parallelism on an eight-card PCIe node.
- An earlier two-card SM120 community run reported roughly 88 output tokens/s
  at concurrency 1 and 399 tokens/s at concurrency 32 without a draft head,
  and 159/633 tokens/s with the earlier draft path. Those are aggregate TP2
  preview-checkpoint figures, not an eight-card 0731 prediction; see the
  [NVIDIA forum measurement](https://forums.developer.nvidia.com/t/i-am-extremely-disappointed-with-the-current-state-of-dgx-spark/365572?page=5).
- A separate four-card SGLang patch reports 37.6 output tokens/s and 11.3-second
  TTFT at 8K input, again a different engine and patch; see its
  [published table](https://github.com/0xSero/deepseek-v4-flash-sm120/blob/main/README.md).

## Reproducibility pins

- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM source: `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`
- vLLM base image digest:
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Open WebUI: `v0.12.3` plus the digest in `example.json`

The served vLLM image (`local/vllm-openai:deepseek-v4-0731-8gpu-effort`) is
rebuilt by this change and has no recorded image ID yet: `example.json`
`image_id` and `kairyu.yaml` `container_image_digest` stay null until the GPU
build records them, and until then `run.sh up` stops unless
`VLLM_ALLOW_UNPINNED_IMAGE=1` serves it as non-evidence.

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override images with
`VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM override must already
exist locally. Dotenv files are intentionally ignored and credentials are not
written into evidence.

For a deliberately public UI, bind only the UI and keep the unauthenticated
Kairyu API on loopback:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3000 \
ENABLE_SIGNUP=false WEBUI_ADMIN_EMAIL=<email> \
WEBUI_ADMIN_PASSWORD=<strong-secret> ./run.sh up
```

On a fresh data volume, Open WebUI creates that administrator and automatically
disables signup. Never commit the password. Public production use should put
TLS and a firewall/reverse proxy in front of port 3000.
