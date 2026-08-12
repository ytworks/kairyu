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
the quality tier. Its reasoning is private L2 state: Kairyu fail-closed splits
the configured `</think>` boundary for both buffered and streamed vLLM
completions, and multi-stage responses expose only the final answer. The
role-tagged conversation JSON is context data, not a request to wrap normal UI
answers in a JSON `role`/`content` envelope. Raw completion `logprobs` are
rejected before dispatch on this private-reasoning alias because they cannot be
truthfully aligned after the hidden prefix is removed; ordinary direct models
retain their normal logprobs support.

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
Kairyu L3, and defaults to the quality-first `kairyu-auto-max`. Direct
`qwen3.6-27b`, direct `deepseek-v4-flash-0731`, fast `kairyu-auto`, and
`kairyu-auto-max` all appear in the model inventory. During the selection gate,
`kairyu-auto-max-chat` retains the diagnostic ordinary-DeepSeek MoA-3 policy.
Both its MoA-2 and MoA-3 variants scored below direct DeepSeek in the fixed
Terminal-Bench pilot. The selected quality policy is `kairyu-auto-max`: three
Qwen proposals followed by private-thinking DeepSeek synthesis. Its generic
private-work allowance is 2048 tokens so a long reasoning tail can reach the
configured `</think>` boundary and return a non-empty public answer. Natural
EOS keeps shorter requests unchanged, and the caller's final-answer budget
remains independent. This policy scored 3/4 in the clean selection pilot versus
2/4 for direct DeepSeek and completed its L3 c1/c8/c16/c32 matrix without an
empty public answer.
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
./bench.sh serving-auto-max-chat
./bench.sh serving-auto-max-moa1
./bench.sh serving-auto-max-moa2
./bench.sh serving-auto-max-moa3
./bench.sh serving-auto-max-moa4
./bench.sh orchestration
./bench.sh swebench-pro
./bench.sh terminalbench-pilot
./bench.sh terminalbench
./bench.sh all
```

The `serving-*` commands independently record median/p99 semantic TTFT and E2E,
TPOT, requests/s, and output tokens/s for fixed approximately 8K-token inputs
at concurrency 1, 8, 16, and 32. Direct Qwen/DeepSeek and fast `auto` use an
exact 256-token completion. Thinking `auto-max` instead uses natural EOS with
an approximately 256-token public-answer instruction and enough combined
reasoning/output budget to reach the private `</think>` boundary. Its artifact
separates cumulative orchestration tokens from exact public-answer tokens,
counted after (and outside) the timed interval by the pinned DeepSeek tokenizer
through loopback-only port 8005. Prompts have unique first blocks so prefix
reuse cannot inflate the matrix. Auto requests require a valid L3 trace for
every sample; fixed-fanout candidates additionally require the exact MoA
proposal count and retain bounded internal input/output token totals in their
trace evidence. ChatUI has no route to the loopback L1 port and continues to
call only Kairyu L3.

The thinking MoA-3 and ordinary chat-synthesis MoA-3 paths remain separately
measurable. The same performance and quality gates select the final
`kairyu-auto-max` policy.
`orchestration` runs Kairyu's fixed direct/auto/auto-max L2 latency and
LiveCodeBench-quality diagnostic, including internal calls, internal tokens,
route identity, and allocated GPU-seconds. `terminalbench-pilot` runs the same
four named Terminal-Bench 2.1 tasks on direct DeepSeek and the quality-first
thinking-MoA3 candidate. `swebench-pro` runs the selected `kairyu-auto-max` over
all 731 pinned public tasks with mini-swe-agent's bundled text-action scaffold,
a 1,000-step budget, 8 workers, and the pinned official ScaleAI evaluator. Its
dataset, evaluator, predictions, trajectories, Docker logs, and zero-inclusive
official verdicts remain below the NVMe benchmark cache and result directory.
Task images pulled by the run are evicted after use so the full suite does not
grow Docker storage in proportion to all 731 tasks; preloaded images are kept.
`terminalbench` runs the selected `kairyu-auto-max` over
all 89 tasks with terminus-2 and the published 500-turn budget. It deliberately
passes no unsupported sampling knob. The one-trial full run launched all 89
tasks and scored 60/89 (67.42%) with official task verifiers and Harbor's
zero-inclusive Mean; it is not a five-trial leaderboard entry. The two
image-pull errors and one operator-interrupted outlier remain zeros and are not
replaced by diagnostic retries.

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

See [MEASUREMENTS.md](MEASUREMENTS.md) for the selected-runtime analysis and
[`terminalbench-result.json`](terminalbench-result.json) for the exact
zero-inclusive full-task-set score ledger.
