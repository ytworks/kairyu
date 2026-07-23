# Qwen3-32B on all GPUs

Runs `Qwen/Qwen3-32B` with Kairyu using every NVIDIA GPU visible to the
container. At startup, the container detects the GPU count and uses it as
Kairyu's tensor-parallel size.

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

Stop the service with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/qwen3-32b-multi-gpu/compose.yaml down
```
