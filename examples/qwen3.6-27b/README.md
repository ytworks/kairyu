# Qwen3.6-27B on 8 GPUs

Serves `Qwen/Qwen3.6-27B` through a Kairyu gateway with one command. The
checkpoint is `Qwen3_5ForConditionalGeneration` (hybrid Gated DeltaNet /
Gated Attention), an architecture outside Kairyu's native engine zoo, so the
model runs on stock vLLM with `tensor_parallel_size=8` and Kairyu fronts it as
the `qwen3.6-27b` pool — the same topology as the Qwen3-VL-32B image-chat
deployment (`deploy/compose/docker-compose.webui-vlm.yaml`).

The vLLM replica enables the checkpoint's bundled MTP weights for speculative
decode and parses Qwen3-style reasoning content. The example serves a
32,768-token context window; the checkpoint natively supports 262,144 tokens
if the KV budget is raised.

Requirements:

- Docker Compose v2 and NVIDIA Container Toolkit
- 8 visible NVIDIA GPUs
- About 60 GB of free disk space for the model cache

From the repository root:

```console
./examples/qwen3.6-27b/run.sh
```

If Hugging Face authentication is required:

```console
HF_TOKEN=hf_... ./examples/qwen3.6-27b/run.sh
```

The OpenAI-compatible API is available at `http://127.0.0.1:8001/v1` (override
with `PORT`). The downloaded model is kept in the
`kairyu-qwen3-6-27b_qwen3-6-27b-model-cache` Docker volume.

```console
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Stop the service with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/qwen3.6-27b/compose.yaml down
```
