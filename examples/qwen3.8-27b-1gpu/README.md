# Qwen3.8-27B on 1 x RTX PRO 6000 Blackwell

This example starts the complete local stack with one command:

```text
Open WebUI -> Kairyu L3 (:8001) -> vLLM L1 (one selected TP1 GPU)
```

The serving checkpoint is the official `Qwen/Qwen3.8-27B-FP8` revision. The
configuration keeps its native vision encoder and 262,144-token context, with
FP8 weights and KV cache, FP16 Gated-DeltaNet state, prefix caching, chunked
prefill, FlashInfer autotuning, and piecewise CUDA Graphs. Kairyu validates
one inline PNG/JPEG/WebP image up to 8 MiB and 2,097,152 pixels, then preserves
the OpenAI content parts for the checkpoint-owned processor and the example's
adapted Qwen chat template (`chat_template.jinja`, mounted into vLLM); ordinary
requests default to direct answers (vLLM's `--default-chat-template-kwargs
'{"enable_thinking": false}'` tells its `qwen3` reasoning parser the same, so a
direct answer or tool call is not filed as reasoning), and an explicit
reasoning effort enables Qwen's thinking mode at a fixed low effort level (the template clamps `high`
and `max`, and `xhigh` — `max` on the Responses wire — to `low`). Kairyu sends
each request to vLLM in the API its caller used (`kairyu.yaml` lists the model
in `upstream_chat_models`, `upstream_responses_models` and
`upstream_messages_models`, m9 D2 amendment 2026-10-17): a chat as the caller's
own messages, a Responses request to vLLM's `/v1/responses`, an Anthropic
Messages request to vLLM's `/v1/messages`. The committed batching, cache-state, graph, and
speculative-decoding values are selected by the local Qwen3.8 tuning run in
[MEASUREMENTS.md](MEASUREMENTS.md). Open WebUI always talks to Kairyu L3 rather
than directly to L1.

The selected L1 envelope is `max_num_batched_tokens=32768`,
`max_num_seqs=32`, `kv_cache_dtype=fp8`, `cudagraph_mode=PIECEWISE`, and MTP
disabled. It reached 867.58 output tok/s at concurrency 32 in the fixed 8K/256
matrix (fixed-length chat rows measured before 2026-10-17); 64K failed
startup and MTP-3 reduced saturated throughput.

## Start

```sh
./run.sh
```

The command validates the selected GPU (`GPU_ID=0` by default), pins it to its
local NUMA CPUs, prepares bind-backed storage below `/mnt/nvme/kairyu`, pulls
the digest-pinned official vLLM release if needed, downloads and hashes the
exact model revision, builds Kairyu, waits for all three services, and prints:

```text
OpenAI API: http://127.0.0.1:8001/v1
Chat UI:    http://127.0.0.1:3000
```

Model files and Open WebUI state share one isolated directory below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/` and never use Docker's
default volume area. The same directory holds vLLM's compilation cache, so
configuration comparisons and later starts can reuse compiled graphs. Set
`NVME_STORAGE_ROOT` only to another absolute path below `/mnt/nvme` if the
default must change. The first model download is approximately 31 GB. Later
starts use its content attestation; `VERIFY_MODEL=1 ./run.sh` rehashes it.

Lifecycle commands are `./run.sh up`, `./run.sh status`, `./run.sh logs`, and
`./run.sh down`. Select another matching card with `GPU_ID=<index>`.

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

Saved model settings: Open WebUI applies a model's saved Advanced Params
(Admin Panel or Workspace > Models) once more after the filter, just before it
sends the request, so the filter cannot normalize them. Save only Responses
fields there (temperature, top_p, max tokens; Open WebUI itself drops stop and
the penalties). A default effort goes in as the Custom Parameter `reasoning`
with the value `{"effort": "high"}`; it applies to requests that carry no
effort of their own. A saved `seed`, `top_k`, `min_p` or `reasoning_effort`
makes Kairyu answer 400 `Unknown parameter`, as OpenAI's Responses API does
(m11 D7 amendment 2026-10-18).

## Verification

```sh
./verify.sh list
./verify.sh serving
./verify.sh api-equivalence
```

`serving` records TTFT, TPOT, requests/s, and output tokens/s on Kairyu's
`/v1/responses` for approximately 8K-token inputs and up to 256 generated tokens
at concurrency 1, 8, 16, and 32. The Responses API has no
`ignore_eos`/`min_tokens`, so each request ends naturally within
`max_output_tokens`; a row passes when every request returns 1 to 256 output
tokens. These rows are not comparable with the fixed-length chat rows recorded
before 2026-10-17. Its deterministic prompts do not share a first prefix block,
so prefix-cache reuse cannot inflate the matrix. `api-equivalence` sends the
same Responses requests (text, stream, each effort, a tool round trip, an image,
a length cap, and Open WebUI's body) to vLLM directly and through Kairyu,
requires the same Responses structure from both, and reads vLLM's request
counters to confirm that Kairyu reached it on `/v1/responses`. Artifacts go to
the environment's NVMe storage,
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). GPU
verification of the 2026-10-17 configuration is pending.

Model and product evaluations are invoked explicitly through `python -m evals`;
see the repository benchmark documentation for the retained evaluation suites.

## Public performance context

The local result in [MEASUREMENTS.md](MEASUREMENTS.md) is the only directly
comparable number for this complete configuration. Public measurements use
different quantizations, contexts, prompts, or software revisions:

- The [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B) specifies a
  native 262,144-token context and a trained MTP layer. The
  [official FP8 repository](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) is
  pinned directly by revision and attested file-tree digest in this example.
- A public [bare-metal RTX PRO 6000 Blackwell report](https://github.com/lastloop-ai/vllm-blackwell-guide)
  measured 117 output tok/s for one stream and 377 tok/s across four streams;
  its tuned MTP test reached about 125 tok/s. It uses a different quantization
  and vLLM revision, so it is a useful range rather than a baseline guarantee.
- A predecessor-model [RTX 6000 Ada FP8 serving report](https://blog.hexgrid.cloud/qwen3-6-27b-fp8-on-one-rtx-6000-ada-fast-ttft-668-tok-s-peak-throughput-benchmark)
  measured 161.5 average output tok/s and 668.5 peak total tok/s across
  concurrency 8–16 at only 8K maximum context. Ada, workload, and context make
  those aggregate figures non-comparable to this Blackwell example.
- NVIDIA specifies [96 GB GDDR7 and 1.8 TB/s memory bandwidth](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro6000-x-nvidia-us-3519208-web.pdf)
  for the Workstation Edition. The
  [vLLM tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/)
  recommends chunked prefill and more than 8,192 batched tokens for throughput
  on smaller models and large GPUs. The committed budget and MTP choice are
  determined by the current local Qwen3.8 comparison in `MEASUREMENTS.md`.

## Reproducibility pins

- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256: `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM release/source: `v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- vLLM image digest: `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Open WebUI: `v0.12.3` plus the digest in `example.json`

Override API/UI ports with `API_PORT` and `CHAT_UI_PORT`. Override prebuilt
images with `VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM override
must already exist locally. Dotenv files are ignored and credentials are not
written into evidence.

To expose the UI deliberately, keep the unauthenticated Kairyu API on loopback:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3000 ENABLE_SIGNUP=false \
WEBUI_ADMIN_EMAIL=<email> WEBUI_ADMIN_PASSWORD=<strong-secret> ./run.sh up
```

Use TLS and a firewall or reverse proxy before exposing port 3000 publicly.
