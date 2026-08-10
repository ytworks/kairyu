# DeepSeek-V4-Flash-0731 on 8 x RTX PRO 6000 Blackwell

This is the repository's only example. It starts the complete local stack:

```text
Open WebUI -> Kairyu L3 (:8002) -> vLLM L1 (TP8 + EP8, all 8 GPUs)
```

The default is tuned for the exact hardware and checkpoint: mixed FP4/FP8
weights, FP8 KV cache, native 1,048,576-token context, prefix caching,
full/piecewise CUDA Graphs, and five-token DSpark speculation.
Five draft tokens match the checkpoint's DSpark block size. Kairyu owns the
official DeepSeek-V4 text prompt and sends it through vLLM's identity template.
`tools` metadata emitted by OpenAI-compatible chat clients is ignored by this
text-only checkpoint template, so Open WebUI's built-in tool declarations do
not block ordinary questions. Model-side function-tool execution is not
provided by this example. Ordinary UI requests default to direct chat mode;
an explicit `reasoning_effort` selects thinking mode for accuracy runs. Open
WebUI defaults model output to 32,768 tokens so normal answers are not cut into
manual **Continue Response** steps.
The official MegaMoE recommendation is not used: the pinned implementation
rejects SM120 as SM100-only, so this example explicitly disables DeepGEMM and
uses vLLM's supported SM120 MoE path.
The official FP4 indexer cache is also disabled because the pinned vLLM
implementation supports it on SM100 datacenter Blackwell, not SM120.

## Start

```sh
./run.sh
```

The command verifies exactly eight matching GPUs, builds the pinned vLLM source
revision if its image is absent, downloads and hashes the exact model revision
if its model volume is absent, builds Kairyu, waits for readiness, and prints:

```text
OpenAI API: http://127.0.0.1:8002/v1
Chat UI:    http://127.0.0.1:3000
```

The first build/download is large. Later starts use the content attestation.
Set `VERIFY_MODEL=1` to rehash every checkpoint file. Lifecycle commands are
`./run.sh up`, `./run.sh status`, `./run.sh logs`, and `./run.sh down`.

## Benchmarks

```sh
./bench.sh list
./bench.sh serving
./bench.sh livecodebench
./bench.sh all
```

`serving` records measured TTFT, TPOT, requests/s, and output tokens/s for a
fixed approximately 8K-token input and exactly 256 generated tokens at
concurrency 1, 8, 16, and 32. `livecodebench` runs the pinned complete
`release_v6`: all 1,055 problems, pass@1, with generated code isolated in a
content-addressed Docker executor. It uses DeepSeek's agentic recommendation
(`temperature=1.0`, `top_p=0.95`, `reasoning_effort=max`) and has no limit or
smoke flag. `all` runs both and continues far enough to finalize evidence after
an individual failure.

Artifacts go to
`bench/results/examples/deepseek-v4-flash-0731-8gpu/<UTC-run-id>/`.

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
- vLLM SM120 source: `aa0d51302747ea80f282e26949708b3253409fe2`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override prebuilt images with
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
