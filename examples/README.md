# Frontier model examples

This directory intentionally contains only three production-shaped environments and their shared controllers.

| Environment | GPU layout | Native context |
|---|---|---:|
| `qwen3.6-27b-1gpu` | selected TP1 GPU (default 0) | 262,144 |
| `deepseek-v4-flash-0731-8gpu` | two EP4 + Attention-DP replicas | 1,048,576 |
| `qwen3.6-deepseek-v4-8gpu` | DeepSeek EP4 on 0–3, Qwen TP1 ×4 on 4–7 | 262,144 / 1,048,576 |

Every directory exposes the same interface:

```text
./run.sh <vllm|kairyu> [up|down|status|logs]
./bench.sh <vllm|kairyu|compare> <benchmark-id|all>
./bench.sh list
```

`run.sh` checks SM120, full VRAM, disk and NUMA-local pairs and fails instead of reducing context. The first start downloads an exact checkpoint revision, hashes all files, builds content-addressed Kairyu images and then serves the same read-only model volume offline. vLLM base images are repository digests; MTP and DSpark remain disabled until their parity, accuracy and 5% goodput gates pass.

Benchmark artifacts are written under `bench/results/examples/<environment>/<backend>/<run-id>/`. Each benchmark finalizes JSON and Markdown immediately, `all` continues after failures, and `compare` runs the two backends sequentially to avoid GPU overlap. Credentials are never included in evidence.
