# Quyet-1.0-Large System One on one RTX PRO 6000 Blackwell

[Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) is a calibrated
decision model in the Jev family. Like TypeSafe's Jev, it is a System One model: code
sends a state and typed questions (choice, score, yes/no) and gets back typed answers
with calibrated probabilities, then branches, routes and gates on them. It does not
chat, call tools or write text. This example serves it the way a Jev model is used:
`POST /v1/systemone` (the Jev wire format) through Kairyu, for applications and
TypeSafe's SDK, with a Jev-style playground.

```text
apps / TypeSafe SDK / playground (:3015) -> Kairyu (:8014) /v1/systemone -> quyet-systemone (CPU) -> vLLM: Quyet-1.0-Large bf16, one GPU (internal)
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
pass per question. The options are lettered A..J in a fixed prompt, and the answer is
the softmax of those letters' next-token logits at a calibrated temperature per
question type (choice 1.3007, score 1.3159, noul 1.4957). Nothing is generated. A
state is truncated at 6,000 tokens (a conversation keeps its latest turns, anything
else its beginning) with a `state_truncated` warning. Text only, at most 10 options
per question.

**System One adapter (L1).** The package has no server. `quyet_systemone.py` keeps
the package's own code for the prompt, truncation, calibration and answer shape, and
replaces only the forward pass: `QuyetOnVllm` subclasses `quyet.llm.runtime.LLMModel`,
and vLLM's `/v1/completions` returns the letters' logprobs (`logprob_token_ids`) for
the exact prompt token IDs the package built. A softmax over logprobs equals one over
logits, so the calibration applies unchanged. OpenJev serves JevK5 (the same kind of
model) this way. vLLM's kernels are not transformers', so the probabilities differ
slightly from the package's own run; the `systemone` and `jevbench` gates bound that.

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

**Model server (L1 vLLM, internal).** The stock `vllm/vllm-openai:v0.31.0` image
(registry digest pinned, no overlay) holds the checkpoint: bf16, an 8,192-token
context (Quyet's prompts stop at 8,000), 64 sequences, 95 % of GPU memory, prefix
caching, text only, and vLLM's batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`), so a
request gets exactly the same answer whatever else runs (measured: 20 repeats, alone
and under load, identical to the last digit; about 10 % slower than without). vLLM
batches the reads of all requests together, and the state opens every question's
prompt, so the questions of one request read it once.

## Kairyu

- **System One** is the only public model: `quyet-1.0-large-systemone`, aliases
  `quyet-latest`, `jev-latest` and `jev-preview` (TypeSafe's SDK default is
  `jev-latest`). Kairyu forwards 16 requests at a time and queues 64 for up to 30 s,
  then answers 429; the adapter accepts 32, so callers never see its 529.
- **The model server** is registered as a pool so that Kairyu reports ready only while
  it is healthy. It is not public (`public_models`): `/v1/models` lists no chat model
  and `/v1/chat/completions` answers 404.

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
2. pulls the vLLM image and checks its registry digest; builds the adapter image on it
   whenever its sources change (the image carries the hashes of its sources as labels,
   which is what is checked, not an image ID that differs between Docker's stores);
3. downloads the checkpoint below
   `/mnt/nvme/kairyu/model-volumes/quyet-1.0-large-1gpu/models` and attests it;
4. starts the stack and probes System One under every name.

It then prints:

```text
System One: http://127.0.0.1:8014/v1/systemone  (model quyet-1.0-large-systemone, aliases ...)
Playground: http://<public host>:3015 (no authentication)
```

The playground listens on all interfaces and is printed with the host's
outward-facing address (`PUBLIC_HOST` to name it, `PLAYGROUND_BIND_ADDRESS=127.0.0.1`
to keep it local). The API stays host-local.

Storage: the checkpoint, every compile cache (vLLM, Triton, torch), logs and the
verification scratch (JevBench, the SDK environment, raw responses) live on NVMe.
Only the Docker images sit on the root disk.

## Playground

`http://<public host>:3015` (`playground/index.html`, served by nginx with Kairyu's
API on the same origin): a state (text, or JSON: an object or a conversation list)
and yes/no, choice or score questions, with examples in English, Vietnamese, JSON and
an agent turn. Each answer shows its distribution, confidence and any truncation, with
the request's input tokens, latency and `Server-Timing`.

## Verification

The gates follow how a System One model is used (TypeSafe's documentation): typed
answers code can branch on, calibrated probabilities, many questions per call,
consistent answers, the official SDK, and throughput for bulk decisions.

```sh
./verify.sh list
./verify.sh all      # in order; stops at the first failure
```

| Gate | What it proves |
|---|---|
| `reference` | With the stack down, the `quyet` package's own CLI (transformers, bf16, the GPU) answers 279 requests: the 48 of `systemone-reference.jsonl` (English, Vietnamese, Japanese, JSON and conversation states, 10-option choices, four states truncated past 6,000 tokens) and JevBench's 231 public items, asked as JevBench asks them. A later gate reuses it only for the same request bodies, checkpoint and adapter sources. |
| `attest` | vLLM's registry digest and the adapter's source labels on the running containers, every checkpoint file re-hashed, vLLM settings and version, the adapter's `quyet` version and calibration, System One public and no chat model. |
| `systemone` | Through Kairyu, all 279 requests have the official input token count, truncation and answer shape (the same prompts); at least 99 % of the official answers that clear TypeSafe's 0.5 confidence floor keep their top option; the median probability difference is at most 0.005. Aliases answer; refusals have the documented shapes. |
| `jevbench` | JevBench's own runner (pinned, `typesafe` adapter, one request at a time, as its board measures) on the 231 public items through Kairyu: every answer valid; per split, correct answers within one item of the official package's, Brier and ECE within 0.01; p50 latency at most 0.5 s. |
| `fanout` | 1, 8 and 32 questions about one 1K-token state in a single call: all answered, and 32 questions take at most 4 times as long as one. |
| `consistency` | The same request, ten times alone and ten times while other reads load vLLM, keeps its top answers, and its probabilities move by at most 0.01. |
| `sdk` | TypeSafe's Python SDK (0.7.4) against Kairyu: typed `choices` / `nouls` / `scores` under `jev-latest`, `quyet-latest` and the full name; an 11-option question raises `TypeSafeBadRequestError`. |
| `systemone-serving` | Cache-busted states of about 50, 2,000 and 6,000 tokens with 3 questions, at c1/16/32/64 (64 each): all answered; req/s and p50/p95 recorded. |
| `systemone-isolation` | 640 concurrent reads: every read is 200 or Kairyu's 429, never 529; Kairyu is ready on a healthy model server and reads answer right after. |

Results go to `verification/results/examples/quyet-1.0-large-1gpu/<run>/`, the
measured numbers to [MEASUREMENTS.md](MEASUREMENTS.md).

## Limitations

- Text only, with no `think`, `samples`, `steps` or `sequential`.
- vLLM's kernels are not transformers', so individual probabilities differ from the
  package's own run (median 0.0001; up to about 0.2 on states of a few thousand tokens,
  where a confident answer can occasionally change). Decisions and JevBench quality are
  gated against the package's run (`systemone`, `jevbench`).
- Batch and async requests are not configured.
