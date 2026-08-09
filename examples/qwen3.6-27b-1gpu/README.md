# Qwen3.6-27B / 1 GPU

One selected SM120 GPU serves the pinned BF16 checkpoint at its native 262,144-token context. The lifecycle inherits `HF_TOKEN` from the invoking environment (required only when Hugging Face requires authentication). For large local storage, set `MODEL_STORAGE_ROOT` to an absolute parent directory; the example creates a bind-backed volume below it and checks free space there. Then run:

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

The first start verifies GPU/VRAM/model-storage space, builds content-addressed Kairyu images, downloads revision `6a9e13b…`, hashes every model file, and then reuses the read-only model volume offline. The token is forwarded to the downloader by environment-variable name and is never placed in a command argument. MTP remains off until the parity and 5% goodput gate selects it.
