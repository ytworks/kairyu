# DeepSeek-V4-Flash-0731 on 8 GPUs

Serves `deepseek-ai/DeepSeek-V4-Flash-0731` (284B-parameter MoE, 256 routed
experts, 6 active per token) through a Kairyu gateway with one command. The
checkpoint is `DeepseekV4ForCausalLM`, an architecture outside Kairyu's native
engine zoo (which stops at DeepSeek-V3), so the model runs on stock vLLM with
`tensor_parallel_size=8` plus expert parallelism, and Kairyu fronts it as the
`deepseek-v4-flash-0731` pool — the same topology as the Qwen3-VL-32B
image-chat deployment (`deploy/compose/docker-compose.webui-vlm.yaml`).

The vLLM replica follows the official 0731 serving recipe: FP8 KV cache,
256-token blocks, the `deepseek_v4` tokenizer mode, and the bundled DSpark
draft module for speculative decode. The example serves a 32,768-token
context window.

Requirements:

- Docker Compose v2 and NVIDIA Container Toolkit
- 8 visible NVIDIA GPUs with 96 GB each (the FP8 checkpoint is ~167 GB of
  weights before KV cache)
- About 180 GB of free disk space for the model cache

From the repository root:

```console
./examples/deepseek-v4-flash-0731/run.sh
```

If Hugging Face authentication is required:

```console
HF_TOKEN=hf_... ./examples/deepseek-v4-flash-0731/run.sh
```

The OpenAI-compatible API is available at `http://127.0.0.1:8001/v1` (override
with `PORT`). The downloaded model is kept in the
`kairyu-deepseek-v4-flash-0731_deepseek-v4-flash-model-cache` Docker volume.

```console
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": "Hello"}]}'
```

Stop the service with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/deepseek-v4-flash-0731/compose.yaml down
```
