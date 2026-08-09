# Qwen3.6 + DeepSeek V4 / 8 GPU orchestration

GPUs 0–3 run one DeepSeek EP4 + Attention-DP tier. GPUs 4–7 run four independent Qwen TP1 replicas. `kairyu-auto` uses a hash-pinned quality-gated router; `kairyu-auto-max` runs three Qwen proposals in parallel and DeepSeek synthesis.

The checked-in router is the structurally safe all-Tier2 baseline (quality ratio 1 by construction). Benchmark calibration may replace it only with a fixed train/holdout artifact whose 95% CI lower bound is at least 0.99 and whose eligible threshold maximizes goodput.

```sh
cp .env.example .env
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

Inputs above 1,048,576 tokens fail with HTTP 400; no tier truncates them. AUTO chat uses a deterministic role-preserving L2 JSON envelope, not tokenizer control tokens or the legacy renderer.
