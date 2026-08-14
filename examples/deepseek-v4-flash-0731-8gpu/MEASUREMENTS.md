# Measured performance

These are local measurements of the exact example configuration, not
manufacturer specifications or estimates. The serving run was warm and used
unique prompt prefixes to prevent prefix-cache reuse from inflating the matrix.

## Locked system

- Hardware: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB per
  GPU, compute capability 12.0. `nvidia-smi topo -m` reported PCIe/NUMA paths
  (`NODE`/`SYS`) and no NVLink paths.
- Model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Model tree SHA-256:
  `90bd164d6f778d798eeaecd3517d83b87d49d300756a9217ada14a2b15203754`
- vLLM revision: `aa0d51302747ea80f282e26949708b3253409fe2`
- vLLM image ID:
  `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Kairyu verification commit:
  `34397110888aa1b0e54f35ce491ebca5bba2d455` with a clean source tree
- Served-config SHA-256:
  `32cc81e53b5e66ed790910293ad0b7c153daa57c77e138ebdefab1bbd6322e05`
- L1: tensor parallel 8, expert parallel 8, FP8 KV, 256-token blocks,
  prefix caching, full/piecewise CUDA Graphs, and five-token DSpark
  speculation. The SM100-only MegaMoE and FP4 indexer-cache paths are disabled
  on SM120.

The topology follows the official [vLLM eight-card recipe for this GPU
class](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?features=tool_calling%2Creasoning&hardware=b300).
The remaining parameters follow the official [DeepSeek model
card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) where the
pinned SM120 implementation supports them. No public measurement with this
exact eight-card/checkpoint/engine combination was found, so the numbers below
are the authoritative performance evidence for this example.

## Warm serving result

Each operating point completed 32/32 requests with approximately 8K input
tokens and exactly 256 output tokens per request. Percentiles use the harness's
nearest-rank method.

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | Requests/s | Output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 220.10 | 286.79 | 5.097 | 0.66 | 168.01 |
| 8 | 369.52 | 607.92 | 12.620 | 2.04 | 522.21 |
| 16 | 390.29 | 1,913.50 | 17.539 | 2.66 | 681.37 |
| 32 | 1,354.54 | 1,372.12 | 22.419 | 3.80 | 972.93 |

Concurrency 1 is the measured minimum-TTFT operating point. Concurrency 32 is
the measured maximum-throughput point. There is no single concurrency that
simultaneously minimizes queueing latency and maximizes aggregate throughput.

Run ID: `tp8-ep8-dspark5-warm-20260811`.

## Raw evidence identity

Historical serving artifacts are under
`bench/results/examples/deepseek-v4-flash-0731-8gpu/` on the measurement host.
New runs are written below `verification/results/examples/`. Their identities are:

| Artifact | SHA-256 |
|---|---|
| Serving concurrency 1 | `544aa032d3fa7882233119ff26707a5b4432579309575f2ec18f8bddd0a27c27` |
| Serving concurrency 8 | `1e14c9092de71793a26f597d8ad97872d32dceebd76862263770ed7e3ee6a983` |
| Serving concurrency 16 | `dbf5693ea23d4785493820541a005d7b42e091c4c6e85e9e7aada647c7f7454a` |
| Serving concurrency 32 | `186610767fb68182e58e4e7fd98f3626b1ab7ef968f48f57cca657198c8c5387` |
