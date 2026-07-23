# Qwen3-32B on one GPU

Runs `Qwen/Qwen3-32B` with Kairyu on one NVIDIA GPU.

Requirements:

- Docker Compose v2 and NVIDIA Container Toolkit
- One GPU with at least 80 GB VRAM
- About 70 GB of free disk space for the model

From the repository root:

```console
./examples/qwen3-32b-single-gpu/run.sh
```

If Hugging Face authentication is required:

```console
HF_TOKEN=hf_... ./examples/qwen3-32b-single-gpu/run.sh
```

The OpenAI-compatible API is available at `http://127.0.0.1:8000/v1`.
The downloaded model is kept in the `kairyu-qwen3-32b_qwen3-32b` Docker volume.

Stop the service with `Ctrl-C`. Remove its containers with:

```console
docker compose -f examples/qwen3-32b-single-gpu/compose.yaml down
```
