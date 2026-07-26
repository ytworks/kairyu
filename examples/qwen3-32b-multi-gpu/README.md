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

## Answer quality: the Fugu suite

The commands above measure throughput. To measure *answers* — all eleven
benchmarks from Sakana's Fugu release table, then an accuracy report against
their published scores — start the service and run the suite with one command:

```console
./examples/qwen3-32b-multi-gpu/run-fugu-benchmark.sh
```

It starts Qwen3-32B on every visible GPU (reusing an already-running service),
waits for readiness, verifies that `qwen3-32b` is the model actually served, and
runs the suite. Progress is reported per slot and per item while it runs.

Two artifacts land under `results/fugu/<run_id>/`:

- `scoreboard.md` — the Fugu-layout table of what was measured,
- `comparison.md` — each cell next to the published Fugu / Fugu Ultra /
  Opus 4.8 / Gemini 3.1 Pro / GPT 5.5 values, with the delta and every reason a
  delta may not mean parity.

To benchmark an already-running service:

```console
./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh
```

**This needs [uv](https://docs.astral.sh/uv/) on the host.** The suite downloads
and normalizes datasets, which needs the `bench` extra; the serving image does
not carry it, so the suite runs from the repository while the model serves in
the container.

By default each slot is capped at 20 items, because a full run is tens of
thousands of judged items (HLE alone is 2,500) and takes hours. The cap is
announced on every run and recorded in the scoreboard's item counts.

```console
BENCH_LIMIT=0 ./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh          # full suite
BENCH_ONLY=gpqa-diamond,mrcr-v2 ./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh
REASONING_EFFORT=high ATTEMPTS=4 ./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh
OFFLINE_FIXTURES=1 ./examples/qwen3-32b-multi-gpu/fugu-benchmark.sh     # plumbing only
```

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_LIMIT` | `20` | Items per benchmark; `0` runs everything. Rejected unless a non-negative integer — a typo must not silently become a full run |
| `BENCH_ONLY` / `BENCH_EXCLUDE` | — | Comma-separated slot names |
| `ATTEMPTS` | `1` | Trials per task for the agentic slots (Fugu reports τ³ as pass@4) |
| `REASONING_EFFORT` | — | `reasoning_effort` sent with every request (Fugu reports max effort) |
| `JUDGE_REASONING_EFFORT` | — | Effort for the judge / τ user simulator (Fugu used `low`) |
| `EXTRA_BODY` | — | JSON merged into every request, e.g. `{"chat_template_kwargs":{"enable_thinking":true}}` |
| `MODEL` / `JUDGE_MODEL` | `qwen3-32b` | Served model ids |
| `BENCH_CONCURRENCY` | `8` | In-flight requests |
| `RESULTS_DIR` / `RUN_ID` | `results/fugu`, timestamp | Where evidence lands; reuse a `RUN_ID` to resume |
| `OFFLINE_FIXTURES` | — | Synthetic stand-in data: checks the plumbing, scores are meaningless |
| `VISION` | — | Declare the target vision-capable (see below) |
| `PORT` | `8001` | Host port; reaches the Compose mapping, so a custom value really is where the service listens |

Slots with unmet preconditions report `skipped` rather than failing the run:
GPQA Diamond, HLE and LiveCodeBench Pro need `HF_TOKEN` with the dataset
licenses accepted, and SWE-Bench Pro, Terminal-Bench and τ³ need docker plus
their harnesses (`uv sync --extra bench-agentic`). See `docs/benchmarks.md`.

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
