# Qwen3.8-27B as 8 x TP1 replicas on 8 x RTX PRO 6000 Blackwell

This example starts one public Qwen model backed by eight identical single-GPU
vLLM replicas with one command:

```text
Open WebUI -> Kairyu L3 (:8004; model qwen3.8-27b)
                -> Kairyu L2 replica pool (placement only, no orchestration)
                    -> 8 x vLLM L1, Qwen3.8-27B-FP8 TP1, one per GPU (GPU 0-7)
```

Clients see exactly one OpenAI-compatible model, `qwen3.8-27b`. Kairyu L2 does
nothing but choose a replica for each request, so the eight GPUs behave like
one server with eight times the capacity of the
[single-GPU example](../qwen3.8-27b-1gpu/README.md), whose measured L1
envelope every replica carries unchanged (official FP8 checkpoint, 262,144-token
context, FP8 KV cache, FP16 Gated-DeltaNet state, prefix caching, chunked
prefill, FlashInfer autotuning, piecewise CUDA Graphs, MTP off, and the same
adapted chat template mounted from that example). Kairyu forwards each
request to vLLM in the API its caller used (m9 D2 amendment 2026-10-17): Chat
Completions as the caller's own messages, the Responses API as Responses, and
Anthropic Messages as Messages (`upstream_chat_models`,
`upstream_responses_models` and `upstream_messages_models` in
[`kairyu.yaml`](kairyu.yaml); Kairyu refuses to start if a replica does not
serve one of them). Ordinary requests default to direct answers; an explicit
effort (`reasoning.effort` on Responses, `reasoning_effort` on chat) enables
Qwen's thinking mode at the fixed low effort level, as in the single-GPU
example (the template treats every effort, including the `xhigh` Kairyu sends
for L3 `max` on the Responses wire, as low). The replicas also pass
`--default-chat-template-kwargs '{"enable_thinking": false}'` so vLLM's `qwen3`
reasoning parser shares the template's default: without it a direct answer
(which contains no `</think>`) is filed as reasoning and the answer text comes
back empty. An explicit effort still switches thinking on.

**Status (2026-10-17): the Responses connection and the native forwarding are
implemented; GPU verification is pending.** The locked results in
[MEASUREMENTS.md](MEASUREMENTS.md) were measured on the earlier chat-only
configuration.

## How requests are spread over the replicas

The pool in [`kairyu.yaml`](kairyu.yaml) uses Kairyu's built-in placement
policy with two knobs set for even, efficient spreading:

- `prefix_index: true` — a request whose prompt prefix is already warm in a
  replica's KV cache (a multi-turn chat continuing on the same replica) goes
  back to that replica **only while it is idle**, so the cached prefix is
  reused instead of being prefilled again elsewhere.
- `queue_depth_threshold: 0` — the moment the preferred replica has one
  in-flight request, placement falls through to **least-outstanding**: the
  replica with the fewest in-flight requests wins, ties going to the lowest
  replica id. Concurrent traffic therefore lands one-per-replica before any
  replica takes a second request; strictly serial traffic uses replica 0.

A replica is eligible only after its readiness probe succeeds and is ejected
after one failed request (`unhealthy_after: 1`) until it probes healthy
again. Every placement decision is appended to the pool's placement log
(`placement_log_path`), which `verify.sh` reads to prove the distribution.

## Start

```sh
./run.sh
```

The command validates eight matching GPUs (indices 0-7), pins each replica to
its GPU's NUMA-local CPUs, prepares bind-backed storage below
`/mnt/nvme/kairyu`, pulls the digest-pinned official vLLM release if needed,
reuses the single-GPU example's attested checkpoint download (or downloads and
hashes it), builds Kairyu, waits for all ten services, checks that Kairyu
reports exactly one public model with eight healthy replicas, proves one live
bash tool call over `/v1/responses` (the environment is not "ready" without a
`function_call`), installs the Chat UI filter (below), and prints:

```text
OpenAI API: http://127.0.0.1:8004/v1
Chat UI:    http://127.0.0.1:3000
```

Model files live in `/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/models`
(shared with the single-GPU example); Open WebUI state, one vLLM compilation
cache per replica, and the placement log live below
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-dp8-8gpu/`. Set
`NVME_STORAGE_ROOT` only to another absolute path below `/mnt/nvme`.
`VERIFY_MODEL=1 ./run.sh` rehashes the checkpoint. Lifecycle commands are
`./run.sh up`, `./run.sh status`, `./run.sh logs`, and `./run.sh down`.

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

`tool-calling` gates the OpenAI agent contract (auto-choice `bash` tool call
fanned across all eight replicas, the follow-up turn with the tool result —
the task still needs a call after the listing, "List the files in the current
directory, then show the first line of README.md.", the PR #647 wording — the
streamed variant, thinking mode with tools, and the non-thinking default);
vLLM's `qwen3_coder` tool parser produces the calls. `api-equivalence` sends
the same Responses requests (text, stream, each effort, a tool round trip, an
image, a length cap, and Open WebUI's request shape) to replica 0 directly and
to Kairyu, requires the same Responses structure and input-token count from
both
([`upstream_api_equivalence.py`](../../verification/product/correctness/upstream_api_equivalence.py)),
and reads all eight replicas' own request counters to confirm Kairyu reached
them on `/v1/responses`, never `/v1/chat/completions`. `serving` warms every
replica with one short request, then records TTFT, TPOT, requests/s, and
output tokens/s for approximately 8K-token inputs and up to 256 generated
tokens at concurrency 1, 8, 16, 32, and 64 (64 requests per row). The
Responses API has no `ignore_eos`/`min_tokens`, so each request ends naturally
within 256 tokens and the row is valid when every request generated 1-256
tokens; these rows are not comparable with the locked fixed-length chat rows.
Each row's prompts carry a row-unique prefix first, so
neither vLLM prefix caching nor Kairyu's prefix-aware placement can inflate the
matrix. For every row it also reads the placement log delta and writes
`placement.json` with the per-replica request counts; at concurrency >= 8 the
row **fails** unless every replica received traffic and no replica took more
than 1.25x the even share (10 of 64 requests). Artifacts go to
`/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-dp8-8gpu/verification-results/<UTC-run-id>/`
(override with `VERIFICATION_RESULTS_ROOT`). The locked chat-era results are in
[MEASUREMENTS.md](MEASUREMENTS.md).

Model and product evaluations are invoked explicitly through `python -m evals`;
see the repository benchmark documentation for the retained evaluation suites.

## Reproducibility pins

- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256: `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM release/source: `v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- vLLM image digest: `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Open WebUI: `v0.12.3` plus the digest in `example.json`

Override API/UI ports with `API_PORT` and `CHAT_UI_PORT`. Override prebuilt
images with `QWEN_VLLM_IMAGE` or `OPEN_WEBUI_IMAGE`; a non-default vLLM
override must already exist locally. Dotenv files are ignored and credentials
are not written into evidence.

To expose the UI deliberately, keep the unauthenticated Kairyu API on loopback:

```sh
CHAT_UI_BIND_ADDRESS=0.0.0.0 PUBLIC_HOST=<public-ip> \
WEBUI_URL=http://<public-ip>:3000 ENABLE_SIGNUP=false \
WEBUI_ADMIN_EMAIL=<email> WEBUI_ADMIN_PASSWORD=<strong-secret> ./run.sh up
```

Use TLS and a firewall or reverse proxy before exposing port 3000 publicly.
