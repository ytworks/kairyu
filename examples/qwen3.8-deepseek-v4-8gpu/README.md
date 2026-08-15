# Qwen3.8 + DeepSeek V4 tiered orchestration on 8 x RTX PRO 6000

This example starts one layered product path with one command:

```text
Open WebUI
    -> Kairyu L3 product API (:8003; model kairyu-auto-max)
        -> Kairyu L2 role DAG
            -> deployment-owned L1 pool: 4 x Qwen3.8-27B-FP8 TP1 (GPU 0-3)
            -> deployment-owned L1 pool: DeepSeek-V4-Flash-0731 TP4+EP4 (GPU 4-7)
        -> Kairyu L2 verified synthesis and bounded refinement
    -> Kairyu L3 final answer

Embedding clients
    -> Kairyu L3 embeddings API (:8003; model embed-small)
        -> pinned offline FastEmbed MiniLM bundle (384 dimensions)
```

Qwen fits one 96 GB card, so four independent TP1 replicas provide more
aggregate memory bandwidth and lower queueing TTFT than spreading one dense
model over PCIe with TP4. Each Qwen replica retains the checkpoint's vision
encoder. Kairyu validates one inline PNG/JPEG/WebP image up to 8 MiB and
2,097,152 pixels, passes it to every image-capable Qwen proposal role, and gives
the text-only DeepSeek roles the same role-tagged conversation with explicit
image placeholders. DeepSeek is sharded TP4+EP4 for capacity and retains the
measured eight-GPU example's FP8 KV, DSpark-5, SM120 fallbacks, prefix caching,
chunked batching, and full/piecewise CUDA Graphs.

The Qwen replicas carry the single-GPU winner unchanged:
`max_num_batched_tokens=32768`, `max_num_seqs=32`, FP8 KV, FP16
Gated-DeltaNet state, piecewise CUDA Graphs, and no MTP. Qwen runs on official
vLLM v0.23.0. DeepSeek intentionally stays on the measured
`aa0d513027` SM120 build because v0.23.0 does not support this checkpoint's
DSpark path and its generic MTP loader cannot load the 0731 MTP weights.

Kairyu exposes one public chat model, `kairyu-auto-max`, and one public
embedding model, `embed-small`. A chat request enters L3 once, then L2 borrows
the deployment-owned L1 pools through
`engine_ref`: DeepSeek planning, three parallel Qwen proposals, DeepSeek draft
synthesis, verification, and DeepSeek publishing. A failed verifier can repeat
synthesis and verification at most twice (`moa_samples: 0`,
`max_refine_depth: 2`, `max_steps: 11`); L2 never calls the public L3 endpoint
recursively.

In the same assistant response, completed L2/L1 stages are sent as
model-attributed `reasoning_content` and rendered by pinned Open WebUI in a
separate expandable internal-work item. The publisher's L3 final answer alone
is sent in `content`, so opening the item reveals each role, attempt, worker,
engine, and model without mixing intermediate work into the answer.

The composed L1 services still use pinned vLLM. This proves the L3/L2/L1 object
boundary and UI behavior, but does **not** close the native-Kairyu L1 production
gate; native full-checkpoint correctness, recovery, soak, and performance gates
remain open. See
[`docs/design/example-layered-orchestration.md`](../../docs/design/example-layered-orchestration.md).

## Start

```sh
./run.sh
```

The command validates the exact eight-card inventory and NUMA affinity, pulls
the pinned Qwen vLLM release, reuses or builds the pinned DeepSeek SM120 image,
verifies or downloads both exact model revisions, builds Kairyu with the
pinned offline MiniLM bundle, waits for all seven services, verifies
`/routing`, sends a two-input embedding smoke, and prints:

```text
OpenAI API: http://127.0.0.1:8003/v1
Chat UI:    http://<outward-facing-host>:3000 (no authentication)
Chat model:      kairyu-auto-max (the only Chat UI model)
Embedding model: embed-small
```

Open WebUI listens on all host interfaces, requires no login, calls only
Kairyu L3, and is explicitly limited to `kairyu-auto-max`. The public
`/v1/models` endpoint additionally returns `embed-small`; the L1 pools are not
public IDs or Chat UI choices. The launcher validates that exact public
inventory, the explicit seven-role DAG, and two ordered finite 384-dimensional
embedding vectors with positive usage before printing the URL.

The embedding model is the truthfully named
`sentence-transformers/all-MiniLM-L6-v2` FastEmbed deployment, not an alias for
OpenAI's `text-embedding-3-large`. Probe it directly with:

```sh
curl -sS http://127.0.0.1:8003/v1/embeddings \
  -H 'Content-Type: application/json' \
  --data '{"model":"embed-small","input":["first","second"],"encoding_format":"float"}'
```

Selecting this model from tau2's pinned `banking_knowledge/alltools` consumer
is tracked separately in `ytworks/kairyu-bench#5`; this deployment does not
mislabel MiniLM to satisfy tau2's historical OpenAI model default.

All persistent state is bind-backed below `/mnt/nvme`:

- Qwen weights reuse `/mnt/nvme/kairyu/model-volumes/qwen3.8-27b-1gpu/models`.
- DeepSeek's external Docker volume is verified to bind
  `/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-8gpu`.
- Four independent Qwen compilation caches, the DeepSeek compilation cache,
  and Open WebUI data live below
  `/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4-8gpu/`.

`NVME_STORAGE_ROOT` may select a different root only when it is still under
`/mnt/nvme`; non-NVMe roots fail closed. `VERIFY_MODEL=1 ./run.sh` rehashes both
checkpoint trees. Lifecycle commands are `./run.sh up`, `./run.sh status`,
`./run.sh logs`, and `./run.sh down`.

## Serving verification

```sh
./verify.sh list
./verify.sh serving-auto-max
```

`serving-auto-max` records the verifier-gated product DAG serving matrix.
Historical MoA performance results in `MEASUREMENTS.md` do not transfer to the
current DAG without a fresh run. ChatUI continues to call only Kairyu L3. Raw
artifacts go to the configured NVMe `verification-results/<UTC-run-id>/`
directory. Model and product evaluations are invoked explicitly through
`python -m evals`.

## Reproducibility pins

- Qwen revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Qwen tree SHA-256: `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- DeepSeek revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- DeepSeek tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- Qwen vLLM release/source: `v0.23.0` /
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- Qwen vLLM image digest:
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- DeepSeek vLLM source: `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`
- DeepSeek vLLM image digest:
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override API/UI/tokenizer-oracle ports with `API_PORT`, `CHAT_UI_PORT`, and
`DEEPSEEK_L1_PORT`. Both L3 endpoints bind all host interfaces by default, so
the API and UI remain reachable through both `127.0.0.1` and the outward-facing
host address. The launcher discovers that address for its printed URLs; set
`PUBLIC_HOST` when clients must use a DNS name, public NAT address, or reverse
proxy. Kairyu's L3 API and the UI are intentionally unauthenticated, so restrict
ports 8003 and 3000 at the firewall or place appropriate TLS/access controls in
front of them when exposure beyond a trusted network is not intended. Set an
explicit bind address when either endpoint must be restricted. Override the
two L1 images independently with `QWEN_VLLM_IMAGE` and
`DEEPSEEK_VLLM_IMAGE`; non-default overrides must already exist locally.

See [MEASUREMENTS.md](MEASUREMENTS.md) for the historical runtime-selection
and serving-performance analysis.
