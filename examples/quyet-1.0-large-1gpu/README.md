# Quyet-1.0-Large on one RTX PRO 6000 Blackwell (System One + chat)

[Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) is a calibrated
decision model in the Jev family. Like TypeSafe's Jev, it is a System One model: code
sends a state and typed questions (choice, score, yes/no) and gets back typed answers
with calibrated probabilities. It does not chat, call tools or write text for the
decision; software branches, routes and gates on its answers. This example serves it
from one GPU in the shape of the other Jev-family examples: one loaded model answers
System One (`/v1/systemone`, the Jev wire format) and plain chat, behind Kairyu, with
a Jev-style playground and Open WebUI.

```text
Playground (:3015) -> Kairyu /v1/systemone -> quyet-systemone (CPU) -> vLLM v0.31.0: Quyet-1.0-Large bf16, one GPU
Open WebUI (:3014) -> Kairyu (:8014)  --------------------------------> the same vLLM (plain text chat)
```

## Model and runtime

**Model.** `chinhnc/Quyet-1.0-Large` revision `9a0f0511`: Gemma-4-31B-it with a
merged rank-16 LoRA, bf16, 62.6 GB in 15 files. Every file is checked against the
publisher's `MANIFEST.sha256` and the whole tree against `example.json`. English,
also tuned for Vietnamese. Apache-2.0; credit: **Quyet by Chinh Nguyen** (see the
checkpoint's `NOTICE`). On JevBench's open-weights board it ranks first (v1.6.1:
Capability 81.7, Intelligence 73.4, Calibration 90.0, measured with `quyet` 1.0.0 in
process on an H100).

**How Quyet decides.** Its runtime, the `quyet` package (1.0.2), reads one forward
pass per question. The options are lettered A..J in a fixed chat prompt, and the
answer is the softmax of those letters' next-token logits at a calibrated
temperature per question type (choice 1.3007, score 1.3159, noul 1.4957). Nothing
is generated. A state is truncated at 6,000 tokens (a conversation keeps its latest
turns, anything else its beginning) with a `state_truncated` warning. Text only, at
most 10 options per question.

**System One (L1 adapter).** The package has no server, and its transformers forward
pass would need a second copy of the weights beside vLLM, which one GPU cannot hold.
`quyet_systemone.py` keeps the package's own code for the prompt, truncation,
calibration and answer shape, and replaces only the forward pass: `QuyetOnVllm`
subclasses `quyet.llm.runtime.LLMModel`, and vLLM's `/v1/completions` returns the
letters' logprobs (`logprob_token_ids`) for the exact prompt token IDs the package
built. A softmax over logprobs equals one over logits, so the calibration applies
unchanged. OpenJev serves JevK5 (the same kind of model) this way. The questions of
one request are read in parallel; the state comes first in every prompt, so vLLM's
prefix cache reads it once.

The adapter answers like OpenJev's Jev-compatible server:

- 422 with a validation list for a body of the wrong shape;
- 400 for a question Quyet cannot ask (more than 10 options, for example), an
  unknown question type or model, and for `images`, `think`, `samples` > 1,
  `steps` > 1 or `sequential` (their neutral values are a plain read);
- 529 when 16 reads run and 16 wait; 503 when vLLM is unavailable;
- a `Server-Timing` header (`model`, `server`, `total`).

Quyet's answers are its package's: `confidence` is the top probability (TypeSafe
derives it differently from the distribution), noul answers carry one too, and a
truncated state adds `warnings` and `truncated`. TypeSafe's SDK reads them as is.

**Chat (L1 vLLM).** The stock `vllm/vllm-openai:v0.31.0` image (digest-pinned, no
overlay) serves the checkpoint as `quyet-1.0-large`, text only: bf16, 32,768-token
context (the plan's rule: 65,536 only if the KV cache holds 8 such sequences; it holds
104,535 tokens), 64 sequences, 95 % of GPU memory, prefix caching, the checkpoint's
sampling defaults (temperature 1.0, top_k 64, top_p 0.95). Chat is there for the
playground's side-by-side answer. The decision fine-tune does not write Gemma 4 tool
calls and its thinking mode breaks down, so tools and images are not offered.

## Kairyu

- **System One.** `quyet-1.0-large-systemone`, aliases `quyet-latest`, `jev-latest`
  and `jev-preview` (TypeSafe's SDK default is `jev-latest`). Kairyu forwards 16
  reads at a time and queues 64 for up to 30 s, then answers 429. The adapter accepts
  32, so callers never see its 529. Reads never pass through the chat pool.
- **Chat.** Pool `quyet-1.0-large`, one vLLM replica, legacy text chat. Kairyu admits
  8 chats at a time, so the adapter's 32 reads in flight always fit vLLM's 64
  sequences.

`kairyu/` is not changed for this example.

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient(api_key="unused", base_url="http://127.0.0.1:8014") as client:
    r = client.system_one(
        state={"message": "I was charged twice this month."},
        questions={
            "team": Choice(instructions="Which team should handle it?",
                           criteria={"billing": "charges", "technical": "bugs", "other": None}),
            "urgent": Noul(instructions="Does the customer need a reply within the hour?"),
        },
    )
print(r.choices["team"].choice, r.nouls["urgent"].noul)
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
2. pulls the pinned vLLM image and builds the adapter image on it, then checks both
   image IDs against `example.json` (a rebuilt adapter needs its new ID recorded, or
   `QUYET_ALLOW_UNPINNED_IMAGE=1` for a run that is not evidence);
3. downloads the checkpoint below
   `/mnt/nvme/kairyu/model-volumes/quyet-1.0-large-1gpu/models` and attests it;
4. starts the stack and probes a plain chat answer and System One under every name.

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

Storage: the checkpoint, every compile cache (vLLM, Triton, torch), logs and the
verification scratch (JevBench, the SDK environment, raw responses) live on NVMe.
Only the Docker images sit on the root disk.

## Playground

`http://<public host>:3015` (`playground/index.html`, served by nginx with Kairyu's
API on the same origin): a state (text, or JSON: an object or a conversation list)
and yes/no, choice or score questions, with examples in English, Vietnamese, JSON
and an agent turn. The left column shows each answer's distribution, confidence,
input tokens, latency, `Server-Timing` and any truncation warning; the right column
asks `quyet-1.0-large` the same questions as a chat.

## Verification

The gates follow how a System One model is used (TypeSafe's documentation): typed
answers your code can branch on, calibrated probabilities, many questions per call,
consistent answers, the official SDK, and throughput for bulk decisions.

```sh
./verify.sh list
./verify.sh all      # in order; stops at the first failure
```

| Gate | What it proves |
|---|---|
| `reference` | With the stack down, the `quyet` package's own CLI (transformers, bf16, the GPU) answers 279 requests: the 48 of `systemone-reference.jsonl` (English, Vietnamese, Japanese, JSON and conversation states, 10-option choices, four states truncated past 6,000 tokens) and JevBench's 231 public items, asked as JevBench asks them. |
| `attest` | Running images, every checkpoint file re-hashed, vLLM settings and version, the checkpoint's sampling defaults in vLLM's log, the context rule, the adapter's `quyet` version and calibration, Kairyu's model lists. |
| `systemone` | Through Kairyu, all 279 requests have the official input token count and truncation, the official top option wherever the official top two differ by 0.05 or more, and probabilities within 0.005 (median) and 0.06 (max) of the official ones. Aliases answer; refusals have the documented shapes. |
| `jevbench` | JevBench's own runner (pinned, `typesafe` adapter, one request at a time, as its board measures) on the 231 public items through Kairyu: every answer valid; per split, correct answers within one item of the official package's, Brier and ECE within 0.01; p50 latency at most 0.5 s. |
| `fanout` | 1, 8 and 32 questions about one 1K-token state in a single call: all answered, and 32 questions take at most 4 times as long as one. |
| `consistency` | The same request, ten times alone and ten times while other reads load vLLM, keeps its top answers, and its probabilities move by at most 0.01. |
| `sdk` | TypeSafe's Python SDK (0.7.4) against Kairyu: typed `choices` / `nouls` / `scores` under `jev-latest`, `quyet-latest` and the full name; an 11-option question raises `TypeSafeBadRequestError`. |
| `systemone-serving` | Cache-busted states of about 50, 2,000 and 6,000 tokens with 3 questions, at c1/16/32/64 (64 each): all answered; req/s and p50/p95 recorded. |
| `systemone-isolation` | 640 concurrent reads beside 8 chats: every read is 200 or Kairyu's 429, the chats answer `323`, the chat replica stays healthy. |

Results go to `verification/results/examples/quyet-1.0-large-1gpu/<run>/`, the
measured numbers to [MEASUREMENTS.md](MEASUREMENTS.md).

## Limitations

- System One is text only and has no `think`, `samples`, `steps` or `sequential`.
- Probabilities differ slightly from the transformers run of the same package
  (bounded by the `systemone` and `jevbench` gates).
- TypeSafe's SDK `models.list()` expects a `release_date` that Kairyu's Jev model list
  does not carry; `system_one` itself is unaffected.
- Chat is plain text; batch and async requests are not configured.
