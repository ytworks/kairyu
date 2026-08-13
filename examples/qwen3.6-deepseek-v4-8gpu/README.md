# Qwen3.6 + DeepSeek V4 tiered orchestration on 8 x RTX PRO 6000

This example starts one layered product path with one command:

```text
Open WebUI
    -> Kairyu L3 product API (:8003; model kairyu-auto-max)
        -> Kairyu L2 role DAG
            -> deployment-owned L1 pool: 4 x Qwen3.6-27B-FP8 TP1 (GPU 0-3)
            -> deployment-owned L1 pool: DeepSeek-V4-Flash-0731 TP4+EP4 (GPU 4-7)
        -> Kairyu L2 verified synthesis and bounded refinement
    -> Kairyu L3 final answer
```

Qwen fits one 96 GB card, so four independent TP1 replicas provide more
aggregate memory bandwidth and lower queueing TTFT than spreading one dense
model over PCIe with TP4. DeepSeek is sharded TP4+EP4 for capacity and retains
the measured eight-GPU example's FP8 KV, DSpark-5, SM120 fallbacks, prefix
caching, chunked batching, and full/piecewise CUDA Graphs.

Kairyu exposes exactly one public product model, `kairyu-auto-max`. Its request
enters L3 once, then L2 borrows the deployment-owned L1 pools through
`engine_ref`: DeepSeek planning, three parallel Qwen proposals, DeepSeek draft
synthesis and verification, then Qwen publishing. Every role uses its model's
bounded output contract so the next role receives a non-empty stage result:
Qwen and DeepSeek roles both use their direct-answer templates, and DeepSeek's
complete bounded generation is the explicit plan, draft, or verdict rather
than a private scratchpad that can consume the whole token budget. Image
requests first run one conditional Qwen
vision-grounding role; its grounded text is an explicit input to planning,
synthesis, verification, and publishing, while text-only requests skip that
role. A failed verifier can repeat synthesis and verification at most twice
(`moa_samples: 0`, `max_refine_depth: 2`, `max_steps: 12`); L2 never calls the
public L3 endpoint recursively.

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

The command validates the exact eight-card inventory and NUMA affinity, builds
the pinned vLLM source image if absent, verifies or downloads both exact model
revisions, builds Kairyu, waits for all seven services, verifies `/routing`, and
prints:

```text
OpenAI API: http://127.0.0.1:8003/v1
Chat UI:    http://<outward-facing-host>:3000 (no authentication)
```

Open WebUI listens on all host interfaces, requires no login, calls only
Kairyu L3, and defaults to `kairyu-auto-max`, the only model returned by the
product `/v1/models` endpoint. The L1 pools are not Chat UI choices. The
launcher validates that exact public inventory and the explicit eight-role DAG
before printing the URL.

All persistent state is bind-backed below `/mnt/nvme`:

- Qwen weights reuse `/mnt/nvme/kairyu/model-volumes/qwen3.6-27b-1gpu/models`.
- DeepSeek's external Docker volume is verified to bind
  `/mnt/nvme/kairyu/model-volumes/deepseek-v4-flash-0731-8gpu`.
- Four independent Qwen compilation caches, the DeepSeek compilation cache,
  and Open WebUI data live below
  `/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/`.

`NVME_STORAGE_ROOT` may select a different root only when it is still under
`/mnt/nvme`; non-NVMe roots fail closed. `VERIFY_MODEL=1 ./run.sh` rehashes both
checkpoint trees. Lifecycle commands are `./run.sh up`, `./run.sh status`,
`./run.sh logs`, and `./run.sh down`.

## Benchmarks

```sh
./bench.sh list
./bench.sh serving-auto-max
./bench.sh terminalbench-pilot
./bench.sh terminalbench
./bench.sh accuracy-pilot --run-id <fresh-run-id>
./bench.sh all
```

`serving-auto-max` records the verifier-gated product DAG's serving matrix.
`terminalbench-pilot` runs its four-task pilot, and `terminalbench` runs the
same product model over all 89 tasks with terminus-2 and the published 500-turn
budget. Historical MoA results in `MEASUREMENTS.md` do not transfer to this DAG
without a fresh run. ChatUI continues to call only Kairyu L3.

`accuracy-pilot` runs the unfiltered 12-slot Accuracy suite against the single
public `kairyu-auto-max` L3 endpoint with real data, `--limit 3 --attempts 1
--seed 0`, the same L3 model as judge/user simulator, and the content-addressed
Docker execution image. Its post-run gate requires every selected item (and
every scoreable sub-step in three selected SciCode problem chains) to complete
with a numeric score and no failed, skipped, or unjudged evidence. Accuracy
cache and results remain on NVMe; retain the pinned tau2-bench checkout under
the example's `bench-data/` directory.

The Harbor dataset is exported once to
`/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-data/terminal-bench-2-1`;
Harbor jobs use the adjacent `bench-tmp/` directory and are retained for raw
per-task evidence and `harbor job resume`. The agent config caps every task at
two effective hours (`max_timeout_sec=900` before the 8x multiplier). This
avoids Harbor's home-directory cache and eight-hour outliers for example-owned
runs.

`all` continues through every benchmark after an individual failure and always
finalizes `run.json`. Artifacts go to
`/mnt/nvme/kairyu/model-volumes/qwen3.6-deepseek-v4-8gpu/bench-results/<UTC-run-id>/`.

## Reproducibility pins

- Qwen revision: `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`
- Qwen tree SHA-256: `f108556571d80514a792b458de366221c9b910fe69cbd5d2525c207580cd51aa`
- DeepSeek revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- DeepSeek tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM SM120 source: `aa0d51302747ea80f282e26949708b3253409fe2`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override API/UI/tokenizer-oracle ports with `API_PORT`, `CHAT_UI_PORT`, and
`DEEPSEEK_L1_PORT`. The launcher discovers
the outward-facing IPv4 address used in the printed URL; set `PUBLIC_HOST` when
the browser must use a DNS name, public NAT address, or reverse proxy. Kairyu's
L3 API remains on loopback. The UI is intentionally unauthenticated, so restrict
port 3000 at the firewall or place appropriate TLS/access controls in front of
it when exposure beyond a trusted network is not intended.

See [MEASUREMENTS.md](MEASUREMENTS.md) for the historical runtime-selection
analysis and
[`terminalbench-result.json`](terminalbench-result.json) for the exact
zero-inclusive historical full-task-set score ledger.
