# Qwen3.6 + DeepSeek V4 tiered orchestration on 8 x RTX PRO 6000

This example starts the complete local stack with one command:

```text
Open WebUI -> Kairyu L3 + L2 (:8003)
                  |-> Tier1: 4 x Qwen3.6-27B-FP8 TP1 vLLM replicas (GPU 0-3)
                  `-> Tier2: DeepSeek-V4-Flash-0731 TP4 + EP4 vLLM (GPU 4-7)
```

Qwen fits one 96 GB card, so four independent TP1 replicas provide more
aggregate memory bandwidth and lower queueing TTFT than spreading one dense
model over PCIe with TP4. DeepSeek is sharded TP4+EP4 for capacity and retains
the measured eight-GPU example's FP8 KV, DSpark-5, SM120 fallbacks, prefix
caching, chunked batching, and full/piecewise CUDA Graphs.

Kairyu owns the public L3 API and both L2 policies. `kairyu-auto` routes short
requests to Qwen, difficult requests to DeepSeek, and large/multi-step requests
through two parallel Qwen proposals plus DeepSeek synthesis. `kairyu-auto-max`
always uses three Qwen proposals plus DeepSeek synthesis. The ordinary direct
DeepSeek model remains in chat mode; the internal Tier2 alias defaults to max
thinking so agentic requests that cannot forward a reasoning knob still use
the quality tier.

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
Chat UI:    http://127.0.0.1:3000
```

Open WebUI calls only Kairyu L3 and defaults to `kairyu-auto`. Direct
`qwen3.6-27b`, direct `deepseek-v4-flash-0731`, and `kairyu-auto-max` also
appear in the model inventory. During the selection gate,
`kairyu-auto-max-moa1` through `kairyu-auto-max-moa4` expose matched fixed
fan-out candidates; `kairyu-auto-max` remains the selected alias.

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
./bench.sh serving-qwen
./bench.sh serving-deepseek
./bench.sh serving-auto
./bench.sh serving-auto-max
./bench.sh serving-auto-max-moa1
./bench.sh serving-auto-max-moa2
./bench.sh serving-auto-max-moa3
./bench.sh serving-auto-max-moa4
./bench.sh orchestration
./bench.sh terminalbench
./bench.sh all
```

The `serving-*` commands independently record median/p99 TTFT, TPOT,
requests/s, and output tokens/s for fixed approximately 8K-token inputs and
exactly 256 generated tokens at concurrency 1, 8, 16, and 32. Prompts have
unique first blocks so prefix reuse cannot inflate the matrix. Auto requests
require a valid L3 trace for every sample; fixed-fanout candidates additionally
require the exact MoA proposal count and retain bounded internal input/output
token totals in their trace evidence.

`orchestration` runs Kairyu's fixed direct/auto/auto-max L2 latency and
LiveCodeBench-quality diagnostic, including internal calls, internal tokens,
route identity, and allocated GPU-seconds. `terminalbench` runs the complete
Harbor Hub `terminal-bench/terminal-bench-2-1` package with terminus-2 and the
published 500-turn budget. It deliberately passes no task limit and no sampling
knob that Harbor cannot forward. The checked-in one-trial result is therefore a
complete task-set measurement, not an official five-trial leaderboard entry.

`all` continues through every benchmark after an individual failure and always
finalizes `run.json`. Artifacts go to
`bench/results/examples/qwen3.6-deepseek-v4-8gpu/<UTC-run-id>/`.

## Reproducibility pins

- Qwen revision: `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`
- Qwen tree SHA-256: `f108556571d80514a792b458de366221c9b910fe69cbd5d2525c207580cd51aa`
- DeepSeek revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- DeepSeek tree SHA-256: `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM SM120 source: `aa0d51302747ea80f282e26949708b3253409fe2`
- Open WebUI: `v0.11.0-slim` plus the digest in `example.json`

Override API/UI ports with `API_PORT` and `CHAT_UI_PORT`. To expose the UI,
set `CHAT_UI_BIND_ADDRESS=0.0.0.0` and `PUBLIC_HOST`, but leave the unauthenticated
Kairyu API on loopback and put TLS/firewall controls in front of port 3000.

See [MEASUREMENTS.md](MEASUREMENTS.md) for selected-runtime evidence and the
completed Terminal-Bench result.
