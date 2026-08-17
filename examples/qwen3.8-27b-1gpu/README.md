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
requests default to direct answers, and an explicit `reasoning_effort` enables
Qwen's thinking mode. The committed batching, cache-state, graph, and
speculative-decoding values are selected by the local Qwen3.8 tuning run in
[MEASUREMENTS.md](MEASUREMENTS.md). Open WebUI always talks to Kairyu L3 rather
than directly to L1.

The selected L1 envelope is `max_num_batched_tokens=32768`,
`max_num_seqs=32`, `kv_cache_dtype=fp8`, `cudagraph_mode=PIECEWISE`, and MTP
disabled. It reached 867.58 output tok/s at concurrency 32 in the fixed 8K/256
matrix; 64K failed startup and MTP-3 reduced saturated throughput.

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

## Serving verification

```sh
./verify.sh list
./verify.sh serving
```

`serving` records TTFT, TPOT, requests/s, and output tokens/s for fixed
approximately 8K-token inputs and exactly 256 generated tokens at concurrency
1, 8, 16, and 32. Its deterministic prompts do not share a first prefix block,
so prefix-cache reuse cannot inflate the matrix. Artifacts go to
`verification/results/examples/qwen3.8-27b-1gpu/<UTC-run-id>/`.

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
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

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
