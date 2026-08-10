# Qwen3.6 + DeepSeek V4 / 8 GPU orchestration

GPUs 0–3 run one DeepSeek EP4 + Attention-DP tier. GPUs 4–7 run four independent Qwen TP1 replicas. `kairyu-auto` uses a hash-pinned quality-gated router; `kairyu-auto-max` runs three Qwen proposals in parallel and DeepSeek synthesis.

The checked-in router is the structurally safe all-Tier2 baseline (quality ratio 1 by construction). Benchmark calibration may replace it only with a fixed train/holdout artifact whose 95% CI lower bound is at least 0.99 and whose eligible threshold maximizes goodput.

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

Configuration is inherited only from the invoking process environment; dotenv
files are not read. Export `HF_TOKEN` when Hugging Face authentication is
required and optionally set an absolute `MODEL_STORAGE_ROOT` for bind-backed
model volumes. Quality runs that include Qwen preserve its documented 81,920
output-token thinking budget. The external benchmark client uses concurrency 16
to maximize aggregate GPU throughput. The four Qwen TP1 replicas and
`kairyu-auto-max` proposal fan-out remain additional internal parallelism.

Inputs above 1,048,576 tokens fail with HTTP 400; no tier truncates them. AUTO chat uses a deterministic role-preserving L2 JSON envelope, not tokenizer control tokens or the legacy renderer.
