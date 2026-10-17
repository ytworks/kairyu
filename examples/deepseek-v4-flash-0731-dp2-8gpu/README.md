# DeepSeek-V4-Flash-0731 as 2 x TP4+EP4 replicas on 8 x RTX PRO 6000 Blackwell

This example starts one public DeepSeek model backed by two identical
four-GPU vLLM replicas with one command:

```text
Open WebUI -> Kairyu L3 (:8006; model deepseek-v4-flash-0731)
                -> Kairyu L2 replica pool (placement only, no orchestration)
                    -> 2 x vLLM L1, DeepSeek-V4-Flash-0731 TP4+EP4
                       (replica 0 on GPU 0-3, replica 1 on GPU 4-7)
```

Clients see exactly one OpenAI-compatible model, `deepseek-v4-flash-0731`.
Kairyu L2 does nothing but choose a replica for each request, so the two
halves of the node behave like one server with twice the sequence capacity of
one TP4 replica — the topology the
[tiered example](../qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md) measured and
selected for DeepSeek ("Tier2 speculation, batch-budget, and CUDA Graph
selection"): mixed FP4/FP8 weights, FP8 KV cache, native 1,048,576-token
context, 256-token blocks, prefix caching, full/piecewise CUDA Graphs,
five-token DSpark speculation, a 16K batch-token budget, and 32 sequences per
replica. The SM100-only MegaMoE and FP4 indexer-cache paths stay disabled on
SM120, as in the [TP8 example](../deepseek-v4-flash-0731-8gpu/README.md).

vLLM renders every request with the checkpoint's own prompt encoder
(`--tokenizer-mode deepseek_v4`, the same `encoding_dsv4.py` that ships inside
the checkpoint), and Kairyu forwards each request to vLLM in the API its
caller used (m9 D2 amendment 2026-10-17): Chat Completions as the caller's own
messages, the Responses API as Responses, and Anthropic Messages as Messages
(`upstream_chat_models`, `upstream_responses_models` and
`upstream_messages_models` in [`kairyu.yaml`](kairyu.yaml); Kairyu refuses to
start if a replica does not serve one of them). **OpenAI function-tool calling
works end to end**: declared `tools` are rendered into the official DSML
prompt form, the model's tool calls are parsed by vLLM's `deepseek_v4` tool
parser, and tool results round-trip on the next turn. This is what
tool-driven agents (Codex CLI, SWE-bench Pro's mini-swe-agent) require.
Ordinary requests default to direct chat mode; an explicit effort
(`reasoning.effort` on Responses, `reasoning_effort` on chat) selects thinking
mode. Kairyu sends L3 `max` to vLLM's Responses API as `xhigh`, the OpenAI
vocabulary's top tier; the pinned DeepSeek-V4 tokenizer would render that as
`high`, so the served vLLM image adds one patch
([`patch_deepseek_v4_effort.py`](patch_deepseek_v4_effort.py),
[`vllm-effort.Dockerfile`](vllm-effort.Dockerfile)) that renders it as `max`.
Open WebUI defaults output to 32,768 tokens.

**Status (2026-10-17): the Responses connection, the native forwarding and the
effort patch are implemented; GPU verification is pending.** The locked
results in [MEASUREMENTS.md](MEASUREMENTS.md) were measured on the earlier
chat-only configuration.

## How requests are spread over the replicas

The pool in [`kairyu.yaml`](kairyu.yaml) uses Kairyu's built-in placement
policy with two knobs set for even, efficient spreading:

- `prefix_index: true` — a request whose prompt prefix is already warm in a
  replica's KV cache (a multi-turn chat continuing on the same replica) goes
  back to that replica **only while it is idle**, so a long cached context is
  reused instead of being prefilled again on the other replica.
- `queue_depth_threshold: 0` — the moment the preferred replica has one
  in-flight request, placement falls through to **least-outstanding**: the
  replica with fewer in-flight requests wins, ties going to replica 0.
  Concurrent traffic therefore alternates between the two replicas before
  either takes a second request; strictly serial traffic uses replica 0.

A replica is eligible only after its readiness probe succeeds and is ejected
after one failed request (`unhealthy_after: 1`) until it probes healthy
again. Every placement decision is appended to the pool's placement log
(`placement_log_path`), which `verify.sh` reads to prove the distribution.

## Start

```sh
./run.sh
```

The command verifies exactly eight matching GPUs (indices 0-7), pins each
replica to the union of its four GPUs' NUMA-local CPUs, reuses or builds the
pinned SM120 vLLM source revision if its image is absent, builds the served
image from it (the effort patch above; the layer cache makes this instant
after the first time), reuses the
NVMe-backed checkpoint volume shared with the TP8 and tiered examples (or
downloads and hashes the exact model revision), builds Kairyu, waits for
readiness, checks that Kairyu reports exactly one public model with two
healthy replicas, proves one live bash tool call over `/v1/responses` (the
environment is not "ready" without a `function_call`), installs the Chat UI
filter (below), and prints:

```text
OpenAI API: http://127.0.0.1:8006/v1
Chat UI:    http://127.0.0.1:3000
```

The first build/download is large; DeepSeek's first engine initialization is
also long (compilation and mHC warm-up), and the per-replica compilation
caches below `/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/`
make later starts much faster. Set `VERIFY_MODEL=1` to rehash every checkpoint
file. Lifecycle commands are `./run.sh up`, `./run.sh status`, `./run.sh logs`,
and `./run.sh down`.

## Chat UI

Open WebUI talks to Kairyu over `/v1/responses` (`OPENAI_API_CONFIGS`
`api_type: responses` in [`compose.yaml`](compose.yaml); m11 D7 amendment
2026-10-17). Open WebUI converts its chat body into a Responses body but keeps
a few fields the strict Responses API does not have, so `run.sh up` installs
[`webui-responses-filter.py`](webui-responses-filter.py) as a global filter:
Chat Controls' reasoning effort becomes `reasoning.effort`, sampling keys
outside the Responses API (`seed`, `top_k`, ...) are dropped, saved output
items keep only their Responses fields, and `store` is false because the UI
sends the whole conversation every turn. Installing it needs the admin
account, so it happens only when `WEBUI_ADMIN_EMAIL` and
`WEBUI_ADMIN_PASSWORD` are set (they also create that account on the first
start); otherwise `run.sh up` prints a warning and an admin installs the file
under Admin Panel > Functions and sets Admin Panel > Settings > Connections >
API Type to Responses. With the credentials set, `run.sh up` also switches the
saved Kairyu connection to the Responses API when an existing `webui-data`
volume still has the chat connection it saved earlier.

## Serving verification

```sh
./verify.sh list
./verify.sh serving
./verify.sh tool-calling
./verify.sh api-equivalence
```

Every gate calls Kairyu's public `/v1/responses`, the API the Chat UI uses.

`tool-calling` gates the OpenAI agent contract the serving claim depends on:
an auto-choice `bash` tool call (the SWE-bench Pro mini-swe-agent request
shape) fanned across both replicas, the follow-up turn with the tool result
(the task still needs a call after the listing: "List the files in the current
directory, then show the first line of README.md.", the PR #647 wording), the
streamed variant, thinking mode (an explicit effort) with tools, and the
non-thinking default. Artifacts land in the same results directory as
`serving`.

`api-equivalence` sends the same Responses requests (text, stream, each
effort, a tool round trip, a length cap, and Open WebUI's request shape) to
replica 0 directly and to Kairyu, requires the same Responses structure and
input-token count from both
([`upstream_api_equivalence.py`](../../verification/product/correctness/upstream_api_equivalence.py)),
and reads both replicas' own request counters to confirm Kairyu reached them on
`/v1/responses`, never `/v1/chat/completions`.

`serving` warms both replicas with one short request each, then records TTFT,
TPOT, requests/s, and output tokens/s for approximately 8K-token inputs and up
to 256 generated tokens at concurrency 1, 8, 16, 32, and 64 (64 requests per
row). The Responses API has no `ignore_eos`/`min_tokens`, so each request ends
naturally within 256 tokens and the row is valid when every request generated
1-256 tokens; these rows are not comparable with the locked fixed-length chat
rows. Each row's prompts carry a row-unique prefix first, so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix. For every row it also reads the placement log delta and writes
`placement.json` with the per-replica request counts; at concurrency >= 8 the
row **fails** unless both replicas received traffic and neither took more than
1.25x the even share (40 of 64 requests). Artifacts go to
`/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-dp2-8gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). The locked chat-era results are in
[MEASUREMENTS.md](MEASUREMENTS.md). Model and product evaluations are invoked
explicitly through `python -m evals`.

## Reproducibility pins

- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM source: `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`
- vLLM image digest (source build):
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`;
  the served image adds `vllm-effort.Dockerfile` on top
- Open WebUI: `v0.12.3` plus the digest in `example.json`

The served vLLM image (`local/vllm-openai:deepseek-v4-flash-0731-dp2-xhigh`)
is rebuilt by this change and has no recorded image ID yet: `example.json`
`image_id` and `kairyu.yaml` `container_image_digest` stay null until the GPU
build records them, and until then `run.sh up` stops unless
`DEEPSEEK_ALLOW_UNPINNED_IMAGE=1` serves it as non-evidence.

Override ports with `API_PORT` and `CHAT_UI_PORT`. Override images with
`DEEPSEEK_VLLM_IMAGE` (the source build the effort patch is applied to) or
`OPEN_WEBUI_IMAGE`; a non-default vLLM override must already exist locally. Dotenv files are intentionally ignored and credentials
are not written into evidence.

For a deliberately public UI, bind only the UI and keep the unauthenticated
Kairyu API on loopback:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3000 \
ENABLE_SIGNUP=false WEBUI_ADMIN_EMAIL=<email> \
WEBUI_ADMIN_PASSWORD=<strong-secret> ./run.sh up
```

Public production use should put TLS and a firewall/reverse proxy in front of
port 3000.
