# FlashInfer SM120 (Blackwell) AOT packaging

Status: **ACTIVE packaging contract**. `Dockerfile.cuda` contains the scoped
AOT build. The 2026-07-31 unit-scale G4 E-KV bake failed its product-quality
gates, so public FP8-KV startup is disabled even if these candidate kernels are
present. A calibrated replacement must pass a new bake and the cold-runtime
verification below before any FP8-KV path is declared deployable.

Date: 2026-07-13; updated 2026-07-31 for E4M3 KV-cache support.

## Decision

- Build FlashInfer's `flashinfer-jit-cache` wheel in a CUDA 13.0.1 `-devel`
  stage, then install it into the CUDA 13.0.1 `-runtime` stage. The runtime
  intentionally has no `nvcc`, so every supported production attention shape
  must be present ahead of time.
- Pin FlashInfer to v0.6.14 and `FLASHINFER_CUDA_ARCH_LIST=12.0f`. This matches
  the locked Python package and limits compilation to SM120.
- Scope attention to FA2, head dimensions 64 and 128, no sliding window, and no
  logits soft cap. FA3 and head dimension 256 remain outside this image's
  supported matrix.
- Compile FP16 and BF16 query/output kernels with FP16, BF16, or E4M3 KV.
  E5M2 KV is intentionally not enabled.

The effective override in `Dockerfile.cuda` is:

```python
{
    "fa2_head_dim": [(64, 64), (128, 128)],
    "fa3_head_dim": [],
    "f16_dtype": [torch.float16, torch.bfloat16],
    "f8_dtype": [torch.float8_e4m3fn],
    "use_sliding_window": [False],
    "use_logits_soft_cap": [False],
}
```

## Why `f8_dtype` cannot be empty

In the pinned v0.6.14 source, `flashinfer/aot.py` constructs attention variants
from the Cartesian product of:

- `f16_dtype` for query and output; and
- `f16_dtype + f8_dtype` for K and V.

`gen_fa2` then emits single/batch prefill and single/batch decode modules for
each accepted combination. An empty `f8_dtype` therefore produces no BF16-query
+ E4M3-KV module. Host tests can hide that packaging gap by JIT-compiling, but
the production runtime cannot because it has no `nvcc`.

This contract is based on the exact pinned upstream source:

- [`build_backend.py` at v0.6.14](https://github.com/flashinfer-ai/flashinfer/blob/19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23/flashinfer-jit-cache/build_backend.py)
- [`aot.py` at v0.6.14](https://github.com/flashinfer-ai/flashinfer/blob/19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23/flashinfer/aot.py)

## Rebuild and runtime verification

The source-only change does not claim that an old image contains the new
kernel. When enough build disk is available:

1. Build `Dockerfile.cuda` and confirm its patched-config checks pass and the
   build summary contains SM120, heads 64/128, BF16, and
   `torch.float8_e4m3fn`.
2. In the resulting runtime image, confirm `nvcc` is absent.
3. Only after calibrated scales pass a new G4 E-KV bake and the public gate is
   deliberately re-enabled, start Kairyu with `kv_cache_dtype: fp8_e4m3` and a
   forced FlashInfer backend.
   Exercise both prefill and decode for supported head dimensions with an empty
   writable FlashInfer cache (or a read-only cache location).
4. Require successful responses, `/backends` reporting requested/resolved
   E4M3 KV plus FlashInfer, and no JIT compilation or newly generated cache
   artifact. Run the focused FlashInfer FP8 GPU tests inside that image.

If a new model needs another head dimension or attention option, extend the AOT
matrix and its static guard in the same change; otherwise it would work only in
a development environment capable of JIT compilation.
