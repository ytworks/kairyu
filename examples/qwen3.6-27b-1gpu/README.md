# Qwen3.6-27B / 1 GPU

One selected SM120 GPU serves the pinned BF16 checkpoint at its native 262,144-token context. Configuration comes only from the invoking process environment; the lifecycle does not read dotenv files. It inherits `HF_TOKEN` when Hugging Face requires authentication. For large local storage, export an absolute parent directory such as `MODEL_STORAGE_ROOT=/mnt/nvme/kairyu/model-volumes`; the example creates a bind-backed volume below it and checks free space there. Then run:

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

The first start verifies GPU/VRAM/model-storage space, builds content-addressed Kairyu images, downloads revision `6a9e13b…`, hashes every model file, and then reuses the read-only model volume offline. The token is forwarded to the downloader by environment-variable name and is never placed in a command argument. MTP remains off until the parity and 5% goodput gate selects it.
