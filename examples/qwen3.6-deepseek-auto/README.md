# Qwen3.6-27B + DeepSeek-V4-Flash AUTO orchestration on 8 GPUs

One command starts a heterogeneous two-tier Kairyu deployment: Qwen3.6-27B as
the latency tier, DeepSeek-V4-Flash-0731 as the quality tier, and the routed
`kairyu-auto` / `kairyu-auto-max` models in front of both. Kairyu has a
single-device text reference path for both architectures, but this deployment
needs optimized multi-GPU execution, so stock vLLM serves both pools and
Kairyu fronts them — the Qwen3-VL-32B pattern.

## GPU allocation

| GPUs | Model | Shape |
|---|---|---|
| 0-3 | DeepSeek-V4-Flash-0731 (284B MoE, FP8) | TP4 + expert parallel, FP8 KV, DSpark drafts |
| 4-7 | Qwen3.6-27B (dense) | DP4 — four TP1 replicas, MTP drafts |

The split is designed, not arbitrary:

- The FP8 DeepSeek checkpoint is ~167 GB of weights, so it cannot run on
  fewer than two 96 GB GPUs, and its 256 routed experts shard evenly only
  over 2/4/8-way groups. With tier1 sharing the node, TP4 (~42 GB weights per
  GPU) is the only workable shape and leaves ample FP8 KV headroom.
- Dense 27B Qwen3.6 fits one 96 GB GPU with a 32k context. On this PCIe
  hardware profile the repository's measured doctrine is DP-first
  (`docs/roadmap.md`): PCIe P2P at 30-37 GB/s makes per-layer tensor-parallel
  all-reduce a poor trade for a model that does not need sharding, so four
  independent replicas behind vLLM's data-parallel front end beat TP4 on both
  throughput and TTFT.
- The MoA max tier fires three tier1 proposals concurrently; DP4 absorbs all
  three in parallel — one per replica — while tier2 synthesizes.

## Served models

- `qwen3.6-27b`, `deepseek-v4-flash-0731` — direct pools
- `kairyu-auto` — RuleRouter: short/easy queries go to tier1, heavy signals
  (code fences, reasoning keywords, length) to tier2, and multi-step prompts
  through the Conductor role DAG (tier2 plans, tier1 executes, tier2 verifies
  with one refine round, tier2 synthesizes and streams)
- `kairyu-auto-max` — Mixture-of-Agents: three seeded Qwen3.6-27B proposals,
  one DeepSeek-V4-Flash synthesis pass

Requirements:

- Docker Compose v2 and NVIDIA Container Toolkit
- 8 visible NVIDIA GPUs with 96 GB each
- About 240 GB of free disk space for the two model caches

From the repository root:

```console
./examples/qwen3.6-deepseek-auto/run.sh
```

If Hugging Face authentication is required:

```console
HF_TOKEN=hf_... ./examples/qwen3.6-deepseek-auto/run.sh
```

The OpenAI-compatible API is available at `http://127.0.0.1:8001/v1`
(override with `PORT`). Model caches persist in the
`kairyu-qwen3-6-deepseek-auto_*-model-cache` Docker volumes.

```console
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "kairyu-auto", "messages": [{"role": "user", "content": "First analyze, then plan, then implement a rate limiter in Python."}]}'
```

Stop the stack with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/qwen3.6-deepseek-auto/compose.yaml down
```
