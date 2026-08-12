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

Direct `qwen3.6-27b` requests support one image (up to 8 MiB and 2,097,152
pixels). Kairyu validates the image and preserves the structured content for
Qwen's native multimodal processor; the text-only L2 orchestration policies do
not synthesize image-bearing requests through DeepSeek.

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
thinking-MoA3 candidate. `terminalbench` runs the selected `kairyu-auto-max` over
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

### Full Accuracy runs (all 12 slots)

The commands below run each complete benchmark population. They intentionally
contain neither `--limit` nor `--smoke`. Start the stack first, run them from
the repository root, and use a fresh `STAMP` for a new measurement. Three
gated Hugging Face datasets require `HF_TOKEN`; inject it into the current
shell from the machine's secret environment. Do not put the token in this
README, a command argument, `.env`, or a run artifact.

```bash
./examples/qwen3.6-deepseek-v4-8gpu/run.sh up

: "${HF_TOKEN:?Set the accepted Hugging Face token in this shell environment}"

API=http://127.0.0.1:8003/v1
NVME_ROOT=${NVME_STORAGE_ROOT:-/mnt/nvme/kairyu}
ACCURACY_ROOT="$NVME_ROOT/model-volumes/qwen3.6-deepseek-v4-8gpu"
CACHE="$ACCURACY_ROOT/bench-data/accuracy-cache"
RESULTS="$ACCURACY_ROOT/bench-results/accuracy-full"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SERVED_SHA=641de6442eec69afdee1b657b36649382543f56e94dad7c16bb03e46e3abe082
SERVED_LABEL=rtx-pro-6000-blackwell-8x-qwen-tp1x4-deepseek-tp4-ep4

mkdir -p "$CACHE" "$RESULTS"

COMMON=(
  --suite accuracy
  --base-url "$API"
  --served-config-label "$SERVED_LABEL"
  --served-config-sha256 "$SERVED_SHA"
  --max-context-tokens 262144
  --request-timeout-s 86400
  --attempts 1
  --cache-dir "$CACHE"
  --results-dir "$RESULTS"
)
TEXT_CHAT=(
  "${COMMON[@]}"
  --model kairyu-auto-max-chat
  --no-vision
  --max-output-tokens 65536
  --recommended-sampling
  --concurrency 4
)
VISION_JUDGED=(
  "${COMMON[@]}"
  --model qwen3.6-27b
  --max-output-tokens 8192
  --recommended-sampling
  --judge-base-url "$API"
  --judge-model deepseek-v4-flash-0731
  --concurrency 4
)
```

Build and identify the immutable sandbox image before the three generated-code
slots. `EXEC_IMAGE` must be the `sha256:...` ID printed by Docker, not a mutable
tag.

```bash
docker build -f deploy/bench/Dockerfile.exec \
  -t kairyu-bench-exec:local deploy/bench
EXEC_IMAGE=$(docker image inspect kairyu-bench-exec:local --format '{{.Id}}')
SANDBOX=(--exec-runner docker --exec-image "$EXEC_IMAGE")
```

Run the eight non-agentic slots as follows:

```bash
# 1. LiveCodeBench — complete 1,055-problem release_v6 population
uv run kairyu bench run "${TEXT_CHAT[@]}" "${SANDBOX[@]}" \
  --only livecodebench --run-id "livecodebench-full-$STAMP"

# 2. LiveCodeBench Pro — complete pinned 167-row population (uses HF_TOKEN)
uv run kairyu bench run "${TEXT_CHAT[@]}" "${SANDBOX[@]}" \
  --only livecodebench-pro --run-id "livecodebench-pro-full-$STAMP"

# 3. Humanity's Last Exam — complete gated population, including images
uv run kairyu bench run "${VISION_JUDGED[@]}" \
  --only hle --run-id "hle-full-$STAMP"

# 4. CharXiv Reasoning — complete vision population
uv run kairyu bench run "${VISION_JUDGED[@]}" \
  --only charxiv-reasoning --run-id "charxiv-reasoning-full-$STAMP"

# 5. GPQA Diamond — complete gated Diamond split (uses HF_TOKEN)
uv run kairyu bench run "${COMMON[@]}" \
  --model kairyu-auto-max --no-vision --max-output-tokens 65536 \
  --recommended-sampling --concurrency 4 \
  --only gpqa-diamond --run-id "gpqa-diamond-full-$STAMP"

# 6. SciCode — all problems and their complete sequential sub-step chains
uv run kairyu bench run "${TEXT_CHAT[@]}" "${SANDBOX[@]}" \
  --only scicode --run-id "scicode-full-$STAMP"

# 7. Long Context Reasoning — complete pinned LongBench-v2 substitute
uv run kairyu bench run "${TEXT_CHAT[@]}" \
  --only long-context-reasoning \
  --run-id "long-context-reasoning-full-$STAMP"

# 8. MRCRv2 — complete selected 500-row, 8-needle, <=128K population
uv run kairyu bench run "${TEXT_CHAT[@]}" \
  --only mrcr-v2 --run-id "mrcr-v2-full-$STAMP"
```

The four agentic slots require the agentic extra and a working local Docker
daemon. The official tau-three release still installs and runs under the
`tau2` package/CLI name. Keep its task data on NVMe and point `TAU2_DATA_DIR`
at the pinned checkout; the tiered example uses the official offline
`bm25_grep` retrieval condition because it does not expose the embedding model
required by `alltools`.

```bash
uv sync --extra bench-agentic
TAU_CHECKOUT="$ACCURACY_ROOT/bench-data/tau2-bench"
test -d "$TAU_CHECKOUT/.git" || \
  git clone https://github.com/sierra-research/tau2-bench.git "$TAU_CHECKOUT"
git -C "$TAU_CHECKOUT" fetch origin fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C "$TAU_CHECKOUT" switch --detach fc0055dc4e0a316c3f83133267fbd6faaa770992
uv pip install "$TAU_CHECKOUT[knowledge]"

export TAU2_DATA_DIR="$TAU_CHECKOUT/data"
export KAIRYU_TAU_RETRIEVAL_CONFIG=bm25_grep
```

```bash
# 9. tau-three Banking — every official Banking task, one trial per task
uv run kairyu bench run "${COMMON[@]}" \
  --model kairyu-auto-max-chat --no-vision --max-output-tokens 65536 \
  --judge-base-url "$API" --judge-model deepseek-v4-flash-0731 \
  --judge-reasoning-effort low --concurrency 1 \
  --only tau-bench-banking --run-id "tau-bench-banking-full-$STAMP"

# 10. SWE-bench Pro — complete split, 1,000 mini-SWE-agent steps per task
# The path itself is recorded, but no credential is.
SWE_PRO_EVAL="$ACCURACY_ROOT/bench-data/SWE-bench_Pro-os"
test -d "$SWE_PRO_EVAL/.git" || \
  git clone --recurse-submodules \
    https://github.com/scaleapi/SWE-bench_Pro-os.git "$SWE_PRO_EVAL"
git -C "$SWE_PRO_EVAL" fetch origin ca10a60a5fcae51e6948ffe1485d4153d421e6c5
git -C "$SWE_PRO_EVAL" switch --detach ca10a60a5fcae51e6948ffe1485d4153d421e6c5
git -C "$SWE_PRO_EVAL" submodule update --init --recursive
export KAIRYU_SWEBENCH_PRO_EVAL_PATH="$SWE_PRO_EVAL"
uv run kairyu bench run "${COMMON[@]}" \
  --model kairyu-auto --no-vision --max-output-tokens 4096 \
  --reasoning-effort low --concurrency 4 \
  --only swe-bench-pro --run-id "swe-bench-pro-full-$STAMP"

# 11. SWE-bench Verified — all 500 tasks, 250 steps per task
uv run kairyu bench run "${COMMON[@]}" \
  --model kairyu-auto --no-vision --max-output-tokens 4096 \
  --reasoning-effort low --concurrency 4 \
  --only swe-bench-verified --run-id "swe-bench-verified-full-$STAMP"

# 12. Terminal-Bench 2.1 — all 89 tasks, terminus-2, 500 turns
# This example-owned wrapper exports the dataset to NVMe, retains Harbor jobs,
# runs the full population, and validates that all 89 results are present.
./examples/qwen3.6-deepseek-v4-8gpu/bench.sh terminalbench \
  --run-id "terminal-bench-full-$STAMP" --no-start
```

`--attempts 1` means one trial per source item. This is a full-population run,
but it is not Terminal-Bench's five-trial leaderboard condition or tau-three's
published pass@4 condition. Increase their trial count only when that separate,
much more expensive comparison is intended. Reusing the same `--run-id`
resumes matching evidence; use a new ID when any immutable input changes.

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
