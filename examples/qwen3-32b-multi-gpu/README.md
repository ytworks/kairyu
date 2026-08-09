# Qwen3-32B on all GPUs

Runs `Qwen/Qwen3-32B` with Kairyu using every NVIDIA GPU visible to the
container. At startup, the container detects the GPU count and uses it as
Kairyu's tensor-parallel size. The deployment opts into CUDA-graph decode for
batches up to 8 and page tables up to 512 pages (8,192 tokens); larger decode shapes
automatically use eager execution. The attention backend follows the hardware
profile, selecting the image's AOT FlashInfer kernels on supported GPUs; set
`KAIRYU_ATTENTION_BACKEND=torch` only when deliberately measuring the reference
backend.

Requirements:

- Docker Compose v2 and NVIDIA Container Toolkit
- `curl`
- 2, 4, or 8 visible NVIDIA GPUs; every visible GPU is used
- About 70 GB of free disk space for the model

From the repository root:

```console
./examples/qwen3-32b-multi-gpu/run.sh
```

If Hugging Face authentication is required:

```console
HF_TOKEN=hf_... ./examples/qwen3-32b-multi-gpu/run.sh
```

The OpenAI-compatible API is available at `http://127.0.0.1:8001/v1`.
The downloaded model is kept in the `kairyu-qwen3-32b_qwen3-32b` Docker volume.

Start the service, wait for port `8001`, run the benchmark, and generate the
report with one command:

```console
./examples/qwen3-32b-multi-gpu/run-benchmark.sh
```

While the command runs, the shell reports model readiness and benchmark
progress every five seconds. Completed request counts come from Kairyu's local
metrics endpoint; if that endpoint cannot be read, the shell continues to show
elapsed time. Progress reporting does not change the benchmark result.

To benchmark an already-running service separately:

```console
./examples/qwen3-32b-multi-gpu/benchmark.sh
```

Each run writes a timestamped JSON file under
`examples/qwen3-32b-multi-gpu/results/`. The script also regenerates
`results/report.md`, which summarizes every saved run.

The workload can be changed with environment variables:

```console
NUM_REQUESTS=256 CONCURRENCY=64 MAX_TOKENS=256 \
  ./examples/qwen3-32b-multi-gpu/benchmark.sh
```

Available variables are `NUM_REQUESTS`, `CONCURRENCY`, `MAX_TOKENS`,
`TTFT_SLO_S`, and `TIMEOUT_S`.

## AUTO direct-route TTFT gate

Issue P-B1 compares `kairyu-auto` with its exact underlying Qwen3-32B backend
through one gateway. Start the Qwen service above, then run the gateway and the
paired benchmark in separate shells:

```console
uv run kairyu serve examples/qwen3-32b-multi-gpu/auto-gateway.yaml
```

```console
uv run python bench/orchestration_stream_bench.py \
  --base-url http://127.0.0.1:8002 \
  --direct-model qwen3-32b \
  --auto-model kairyu-auto
```

During local development an existing CUDA dependency image can validate the
current checkout without rebuilding multi-gigabyte framework layers:

```console
docker compose \
  -f examples/qwen3-32b-multi-gpu/compose.yaml \
  -f examples/qwen3-32b-multi-gpu/compose.source-override.yaml \
  up -d --no-build kairyu auto-gateway
```

The override's gateway uses host networking so its checked-in
`127.0.0.1:8001` backend address still names the Qwen service. This also lets a
sandboxed development client run the benchmark inside that container:

```console
docker compose \
  -f examples/qwen3-32b-multi-gpu/compose.yaml \
  -f examples/qwen3-32b-multi-gpu/compose.source-override.yaml \
  exec auto-gateway python bench/orchestration_stream_bench.py
```

The workload alternates direct/AUTO order within each prompt pair. TTFT starts
at the first non-empty assistant content delta, so AUTO keep-alive comments do
not make the result look faster. The command exits nonzero unless both p50 and
p99 AUTO/direct TTFT ratios are at most 1.5, and writes the raw paired samples
plus the pull-through-versus-queue responsibility-boundary A/B under
`bench/results/`.

## AUTO tier latency/quality gate

The same production gateway also serves `kairyu-auto-max`. The standard tier
uses the Conductor role DAG; the max tier uses three parallel Qwen3-32B
proposals followed by Qwen3-32B synthesis. Run the P-B4 gate after the gateway
above is ready:

```console
.venv/bin/python bench/tiered_auto_bench.py \
  --base-url http://127.0.0.1:8002 \
  --result bench/results/tiered-auto-qwen3-32b-tp8.json
```

The quality set is a fixed seed-198 sample of eight LiveCodeBench release-v6
items whose canonical prompts route to `multi_agent`. Every answer is executed
against its public and private tests. The artifact includes `/v1/models`
discovery, paired direct/AUTO TTFT, per-tier scores, actual internal call and
token totals from the structured trace, and allocated GPU-seconds. This is a
fixed subset gate, not a claim about full-suite LiveCodeBench accuracy.

## Answer quality: the Accuracy suite

The commands above measure throughput. To measure *answers* — all eleven
benchmarks from Sakana's Fugu release table, then an accuracy report against
their published scores — start the service and run the suite with one command:

```console
export HF_TOKEN=hf_...
./examples/qwen3-32b-multi-gpu/run-accuracy-benchmark.sh
```

It starts Qwen3-32B on every visible GPU (reusing an already-running service),
waits for readiness, verifies that `qwen3-32b` is the model actually served, and
runs the suite. During DeploymentSpec preflight, Kairyu loads the exact
checkpoint-owned HF chat template directly from the local tokenizer directory
(`chat_template.jinja` / `additional_chat_templates/*.jinja` before
`tokenizer_config.json`) and injects its named special tokens. Compose neither
extracts nor materializes a temporary template file, and no stale template copy
is committed. Progress is reported per slot and per item while it runs.

Two artifacts land under `results/accuracy/<run_id>/`:

- `scoreboard.md` — the Accuracy-suite table of what was measured,
- `comparison.md` — each cell next to the published Fugu / Fugu Ultra /
  Opus 4.8 / Gemini 3.1 Pro / GPT 5.5 values, with the delta and every reason a
  delta may not mean parity.

The same script also reuses an already-running service. After accepting the
GPQA Diamond, HLE, and LiveCodeBench Pro dataset licenses, the only required
operator setting is `HF_TOKEN`:

```console
export HF_TOKEN=hf_...
./examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh
```

If `HF_TOKEN` is already exported in the machine environment, run only the
second command. The script deliberately does not read `.env`; it passes the
inherited value to both Docker Compose's model downloader and the host-side
benchmark process.

The script starts or reuses Qwen3-32B, asks `uv` to provision both benchmark
extras plus the commit-pinned official τ³ v1.0.1 harness, downloads missing
datasets and its task-data checkout, and builds or reuses the hash-pinned Docker
sandbox for generated code. `uv`, Git, Docker, the NVIDIA driver, sufficient
disk, and the accepted gated dataset licenses are host prerequisites; they are
not per-run configuration.

By default each slot is capped at 20 items, because a full run is tens of
thousands of judged items (HLE alone is 2,500) and takes hours. The cap is
announced on every run and recorded in the scoreboard's item counts.

```console
BENCH_LIMIT=0 ./examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh          # full suite
BENCH_ONLY=gpqa-diamond,mrcr-v2 ./examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh
ATTEMPTS=4 ./examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh
OFFLINE_FIXTURES=1 ./examples/qwen3-32b-multi-gpu/accuracy-benchmark.sh     # plumbing only
```

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_LIMIT` | `20` | Items per benchmark; `0` runs everything. Rejected unless a non-negative integer — a typo must not silently become a full run |
| `BENCH_ONLY` / `BENCH_EXCLUDE` | — | Comma-separated slot names |
| `ATTEMPTS` | `1` | Trials per task for the agentic slots (Fugu reports τ³ as pass@4) |
| `EXTRA_BODY` | Qwen thinking enabled | JSON merged into every target request |
| `JUDGE_EXTRA_BODY` | Qwen thinking disabled | JSON merged into judge / τ user-simulator requests |
| `MODEL` / `JUDGE_MODEL` | `qwen3-32b` | Served model ids |
| `BENCH_CONCURRENCY` | `8` | In-flight requests |
| `RESULTS_DIR` / `RUN_ID` | `results/accuracy`, timestamp | Where evidence lands; reuse a `RUN_ID` to resume |
| `OFFLINE_FIXTURES` | — | Synthetic stand-in data: checks the plumbing, scores are meaningless |
| `VISION` | — | Declare the target vision-capable (see below) |
| `PORT` | `8001` | Host port; reaches the Compose mapping, so a custom value really is where the service listens |
| `KAIRYU_BENCH_EXEC_IMAGE` | auto-built | Existing local image/digest for sandboxed code execution |

GPQA Diamond, HLE and LiveCodeBench Pro require the dataset licenses to be
accepted for the supplied `HF_TOKEN`. SWE-Bench Pro, Terminal-Bench and τ³ use
Docker and harnesses provisioned by the script. Any remaining unmet runtime
precondition is recorded as `skipped` rather than fabricated as a score.

The local Qwen endpoint does not implement OpenAI's provider-specific
`reasoning_effort` field. Instead, this example uses Qwen3's native chat-template
control: target requests set `enable_thinking=true`, while judge and τ user
simulator requests set it to `false`. Kairyu applies these variables while
rendering the auto-loaded HF template and rejects them when no template exists.

**The vision slots skip by design here.** `Qwen/Qwen3-32B` is a text-generation
causal LM — the vision family is the separate Qwen3-VL — so the target is
declared text-only (`--no-vision`). CharXiv is therefore precondition-skipped and
HLE's image rows are item-skipped, instead of being answered from a prompt whose
image part the text-only chat template drops and then recorded as a completed
score. Set `VISION=1` only for a genuinely multimodal deployment.

Subset (`BENCH_LIMIT`) and fixture (`OFFLINE_FIXTURES`) runs are marked inside
`scoreboard.md` and `comparison.md` themselves, which withhold every delta
against the published Fugu scores — a shell warning would not survive into the
file you open hours later.

Stop the service with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/qwen3-32b-multi-gpu/compose.yaml down
```
