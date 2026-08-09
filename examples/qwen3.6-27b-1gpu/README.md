# Qwen3.6-27B / 1 GPU

One selected SM120 GPU serves the pinned BF16 checkpoint at its native 262,144-token context. Copy `.env.example` to `.env`, provide `HF_TOKEN` for the first download, then run:

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

The first start verifies GPU/VRAM/disk, builds content-addressed Kairyu images, downloads revision `6a9e13b…`, hashes every model file, and then reuses the read-only model volume offline. MTP remains off until the parity and 5% goodput gate selects it.
