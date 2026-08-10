# Qwen3.6-27B / 1 GPU

One selected SM120 GPU serves the pinned BF16 checkpoint at its native 262,144-token context. Configuration comes only from the invoking process environment; the lifecycle does not read dotenv files. It inherits `HF_TOKEN` when Hugging Face requires authentication. For large local storage, export an absolute parent directory such as `MODEL_STORAGE_ROOT=/mnt/nvme/kairyu/model-volumes`; the example creates a bind-backed volume below it and checks free space there. Then run:

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

`bench.sh` keeps the full-dataset default. For a fixed paired performance
diagnostic over the same deterministic 30 LiveCodeBench items on both backends,
run:

```sh
./bench-livecodebench-30.sh compare
```

That entrypoint fixes `limit=30`, item-selection `seed=0`, and the example's
16-request concurrency. It does not redefine the full accuracy run.

The first start verifies GPU/VRAM/model-storage space, builds content-addressed Kairyu images, downloads revision `6a9e13b…`, hashes every model file, and then reuses the read-only model volume offline. The token is forwarded to the downloader by environment-variable name and is never placed in a command argument. Quality runs preserve the checkpoint's documented thinking budget of 81,920 output tokens instead of the benchmark harness's generic 8,192-token default. The external benchmark client uses concurrency 16 so the single GPU engine can continuously batch requests for aggregate throughput; this is sixteen simultaneous in-flight requests, not merely a client queue depth. MTP remains off until the parity and 5% goodput gate selects it.
