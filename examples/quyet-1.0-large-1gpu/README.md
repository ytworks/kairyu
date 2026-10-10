# Quyet-1.0-Large on one RTX PRO 6000 Blackwell (chat + System One)

[Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) is a calibrated
decision model: given a state and typed questions, it returns one probability
distribution per question. This example serves it from one GPU, in the same shape
as the other Jev-family examples: one loaded model answers chat and System One
(`/v1/systemone`, the Jev wire format), behind Kairyu, with Open WebUI and a
Jev-style playground.

```text
Open WebUI (:3014) -> Kairyu (:8014) -> vLLM v0.31.0: Quyet-1.0-Large bf16, one GPU        (chat)
Playground (:3015) -> Kairyu /v1/systemone -> quyet-systemone (CPU) -> the same vLLM      (System One)
```

## Model and runtime

**Model.** `chinhnc/Quyet-1.0-Large` revision `9a0f0511`: Gemma-4-31B-it with a
merged rank-16 LoRA, bf16, 62.6 GB in 15 files. Every file is checked against the
publisher's `MANIFEST.sha256` and the whole tree against `example.json`. English,
also tuned for Vietnamese. Apache-2.0; credit: **Quyet by Chinh Nguyen** (see the
checkpoint's `NOTICE`).

**How Quyet decides.** Quyet's own runtime, the `quyet` package (1.0.2), reads one
forward pass per question. The options are lettered A..J in a fixed chat prompt,
and the answer is the softmax of those letters' next-token logits at a calibrated
temperature per question type (choice 1.3007, score 1.3159, noul 1.4957). Nothing
is generated. A state is truncated at 6,000 tokens (a conversation keeps its
latest turns, anything else its beginning) with a `state_truncated` warning.
Text only, at most 10 options per question.

**Chat (L1 vLLM).** The stock `vllm/vllm-openai:v0.31.0` image (digest-pinned, no
overlay) serves the checkpoint as `quyet-1.0-large`:

- bf16, 65,536-token context, 64 sequences, 95 % of GPU memory, prefix caching;
- Gemma 4's tool-call and reasoning parsers, up to 4 images per request;
- the checkpoint's sampling defaults (temperature 1.0, top_k 64, top_p 0.95).

The model is trained for decisions. Chat, tool calls and images rely on the
Gemma 4 base; the gates show that they work, not how well this fine-tune chats.

**System One (L1 adapter).** The package has no server, and its transformers
forward pass would need a second copy of the weights beside vLLM, which one GPU
cannot hold. `quyet_systemone.py` therefore keeps the package's own code for the
prompt, truncation, calibration and answer shape, and replaces only the forward
pass: `QuyetOnVllm` subclasses `quyet.llm.runtime.LLMModel`, and vLLM's
`/v1/completions` returns the letters' logprobs (`logprob_token_ids`) for the
exact prompt token IDs the package built. A softmax over logprobs equals one over
logits, so the calibration applies unchanged. OpenJev serves JevK5 (the same kind
of model) this way. Kernel differences between vLLM and transformers move the
probabilities slightly; `verify.sh systemone` bounds that against the package's
own run.

The adapter answers like OpenJev's Jev-compatible server:

- 422 with a validation list for a body of the wrong shape;
- 400 for a question Quyet cannot ask (more than 10 options, for example), an
  unknown question type or model, and for `images`, `think`, `samples` > 1,
  `steps` > 1 or `sequential` (their neutral values are a plain read);
- 529 when 16 reads run and 16 wait; 503 when vLLM is unavailable;
- a `Server-Timing` header (`model`, `server`, `total`).

It runs on the CPU. Its image is the same vLLM image plus `quyet` (hash-pinned,
`--no-deps`, so vLLM's torch and transformers stay).

## Kairyu

- **Chat.** Pool `quyet-1.0-large`, one vLLM replica, legacy chat, multimodal
  admission. Kairyu admits 8 chats at a time, so the adapter's 32 reads in flight
  always fit vLLM's 64 sequences.
- **System One.** `quyet-1.0-large-systemone`, aliases `quyet-latest`,
  `jev-latest` and `jev-preview` (TypeSafe's SDK defaults). Kairyu forwards 16
  reads at a time and queues 64 for up to 30 s, then answers 429. The adapter
  accepts 32, so callers never see its 529. Reads never pass through the chat
  pool and cannot eject the chat replica.

`kairyu/` is not changed for this example.

```sh
curl http://127.0.0.1:8014/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "quyet-latest", "state": "I was charged twice this month.",
  "questions": {"is_billing": {"type": "noul", "instructions": "Is this a billing issue?"}}}'
```

## Start

```sh
./run.sh            # up: preflight, images, model download + hash check, readiness probes
./run.sh status
./run.sh logs
./run.sh down
```

`run.sh up`:

1. checks the selected GPU (`GPU_ID`, default 0) and refuses one that another
   workload uses; vLLM is pinned to the GPU's NUMA-local CPUs;
2. pulls the pinned vLLM image and builds the adapter image on it, then checks
   both image IDs against `example.json` (a rebuilt adapter needs its new ID
   recorded, or `QUYET_ALLOW_UNPINNED_IMAGE=1` for a run that is not evidence);
3. downloads the checkpoint below
   `/mnt/nvme/kairyu/model-volumes/quyet-1.0-large-1gpu/models` and attests it;
4. starts the stack and probes chat, a tool call, an image and System One under
   every name.

It then prints:

```text
OpenAI API: http://127.0.0.1:8014/v1  (model quyet-1.0-large)
System One: http://127.0.0.1:8014/v1/systemone  (model quyet-1.0-large-systemone, aliases ...)
Chat UI:    http://<public host>:3014 (no authentication)
Playground: http://<public host>:3015 (System One, no authentication)
```

The Chat UI and the playground listen on all interfaces and are printed with the
host's outward-facing address (`PUBLIC_HOST` to name it,
`CHAT_UI_BIND_ADDRESS=127.0.0.1` to keep them local). The API stays host-local.

## Playground

`http://<public host>:3015` (`playground/index.html`, served by nginx with
Kairyu's API on the same origin):

- a state (text, or JSON: an object or a conversation list) and yes/no, choice
  or score questions, with examples in English, Vietnamese, JSON and an agent turn;
- the left column shows each answer's distribution, confidence, input tokens,
  latency, `Server-Timing` and any truncation warning;
- the right column asks `quyet-1.0-large` the same questions as a chat.

## Verification

```sh
./verify.sh list
./verify.sh all      # in order; stops at the first failure
```

| Gate | What it proves |
|---|---|
| `reference` | With the stack down, the `quyet` package's own CLI (transformers, bf16, the GPU) answers the 48 requests of `systemone-reference.jsonl`: English, Vietnamese, Japanese, JSON and conversation states, 10-option choices, 5-level scores and four states truncated past 6,000 tokens. |
| `attest` | Running images, every checkpoint file re-hashed, vLLM settings and version, the checkpoint's sampling defaults in vLLM's log, vLLM's KV cache holding 8 full-context sequences, the adapter's `quyet` version and calibration, Kairyu's model lists. |
| `systemone` | Through Kairyu, every reference request has the official input token count and truncation flags, the official top option wherever the official top two differ by 0.05 or more, and probabilities within 0.005 (median) and 0.06 (max) of the official ones. Aliases work; refusals have Jev's shapes. |
| `tool-calling` | Auto, named, tool-result turn and streamed tool calls through Kairyu. |
| `vision` | A PNG and a WebP image are named correctly. |
| `serving` | Chat at 1K tokens in and 256 out (`ignore_eos`), c1/4/8, 32 requests each, all complete. |
| `systemone-serving` | Cache-busted 1K-token states with 3 questions at c1/16/32/64 (64 each) all answer; c1 p50 at most 1.0 s; the adapter directly at c1 for Kairyu's overhead. |
| `systemone-isolation` | 640 concurrent reads beside 8 chats: every read is 200 or Kairyu's 429, the chats answer `323`, the chat replica stays healthy. |

Results go to `verification/results/examples/quyet-1.0-large-1gpu/<run>/`, and
the measured numbers to [MEASUREMENTS.md](MEASUREMENTS.md).

## Limitations

- System One is text only and has no `think`, `samples`, `steps` or `sequential`.
- Probabilities differ slightly from the transformers run of the same package
  (bounded by the `systemone` gate).
- Batch and async requests are not configured.
