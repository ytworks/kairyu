# Winnow-12B Q8_0 (GGUF, llama.cpp) on 1 x RTX PRO 6000 Blackwell

This example starts the complete local stack with one command:

```text
Open WebUI -> Kairyu L3 (:8001) -> winnow-server L1 (llama.cpp, one selected GPU)
                       \-> /v1/systemone (typed decisions, same loaded model)
```

L1 serves a GGUF checkpoint through llama.cpp instead of vLLM. Kairyu attaches
it with the `openai` backend and `upstream: llamacpp`
(`docs/design/llamacpp-upstream.md`); Kairyu L2/L3 are unchanged.

## Model and runtime

**Model.** [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B)
revision `b2b14213`, `gguf/Winnow-12B-Q8_0.gguf` plus its vision projector
`gguf/mmproj-F16.gguf`. It is a merged Gemma 4 12B IT fine-tune for typed
decisions that also chats and reads images.

**Runtime.** Its runtime, `winnow-server`, comes from
[EldanRing/winnow-inference](https://github.com/EldanRing/winnow-inference)
`77d1458`: llama.cpp `911f6cd` (b11036) plus four small patches.

- The Gemma 4 tied-embedding and bounded-SWA patches change model execution.
- Winnow's patches add `/v1/systemone`, Jev's typed-decision request shape.
- The chat API stays llama-server's own.

The example builds that image for SM120 (`CUDA_ARCH=120`).

**L1 settings.** The committed values are provisional until the first GPU run
records [MEASUREMENTS.md](MEASUREMENTS.md):

- 8 chat slots of 65,536 tokens each (`--context 524288 --chat-parallel 8`);
  an explicit slot count gives every slot its own KV cache.
- q8_0 KV cache, full GPU residency.
- A 65,536-token decision context with 8 decision branches.
- `--memory auto`, so chat and decisions share the card.
- Gemma 4's recommended sampling as server defaults: temperature 1.0,
  top_k 64, top_p 0.95, min_p 0, no repeat penalty.
- `winnow-server`'s launcher always passes `--jinja`, `--reasoning off`,
  `--fit off` and `--no-context-shift`.

## Start

```sh
./run.sh
```

The command:

1. validates the selected GPU (`GPU_ID=0` by default) and pins it to its local
   NUMA CPUs;
2. builds `winnow-server` at the pinned revision if the image is absent;
3. downloads the GGUF files with Winnow's own resumable downloader, which
   checks size and SHA-256, and confirms they match `example.json`;
4. builds Kairyu and waits for all three services.

It then prints:

```text
OpenAI API: http://127.0.0.1:8001/v1  (model winnow-12b)
System One: http://127.0.0.1:8001/v1/systemone  (model winnow-12b-systemone)
Chat UI:    http://127.0.0.1:3000
```

The GGUF files (12.85 GB) live once below
`/mnt/nvme/kairyu/model-volumes/winnow-12b-q8/models/` and are shared with the
DP8 example. Open WebUI state is per environment. Lifecycle commands are
`./run.sh up|status|logs|down`.

## What Kairyu adds

- **Chat.** `/v1/chat/completions` and `/v1/responses` with OpenAI tool calls.
  For tool calls, llama.cpp parses Gemma 4's own call format and Kairyu
  normalizes the calls. Image requests (PNG/JPEG/WebP) are validated by Kairyu
  first. WebP is re-encoded as PNG because llama.cpp cannot decode it without
  ffmpeg.
- **Request fields.** Fields llama.cpp would silently ignore (`min_tokens`,
  `stop_token_ids`, `skip_special_tokens`, priority) are rejected with HTTP 400.
  So are frequency/presence penalties, which llama.cpp would also apply to
  prompt tokens. `repetition_penalty` covers the whole 65,536-token slot.
- **Admission.** Kairyu admits 8 chat requests at a time, the slot count.
- **System One.** `/v1/systemone` forwards Jev-shaped decision requests to
  Winnow. Kairyu queues bursts and answers 429 before Winnow's own 128-request
  queue fills. Decision reads never touch the chat engine's health.

## Verification

```sh
./verify.sh list
./verify.sh all          # attest, contract, tool-calling, vision, systemone, serving
```

| Gate | What it proves |
|---|---|
| `attest` | Checks `/props` against `example.json`: llama.cpp commit, 8 slots, 65,536-token slots, sampling defaults, tool-capable template, vision projector. |
| `contract` | `l1.correctness.llamacpp_upstream_contract` passes on the replica. |
| `tool-calling` | Through Kairyu: auto, named, tool-result turn and streaming tool calls. |
| `vision` | A PNG and a WebP image are both answered correctly. |
| `systemone` | Decisions through Kairyu match a direct read. |
| `serving` | Complete 1K-in/256-out rows at concurrency 1/4/8, with fixed length from `ignore_eos`. |

Results go to `verification/results/examples/winnow-12b-q8-1gpu/<run>/`.

## Limitations

- **Batch and async requests** fail closed: llama-server has no scheduling
  priority.
- **SLO defer:** a deferred request is shed instead.
- **`response_format` with a regex `pattern`:** llama.cpp cannot express some
  patterns and relaxes them to any string, warning only in its log.
