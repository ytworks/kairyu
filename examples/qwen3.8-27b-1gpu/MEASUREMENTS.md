# Measured performance

## Qwen3.8 result (vLLM v0.23.0)

The selected single-GPU configuration uses the attested
`Qwen/Qwen3.8-27B-FP8` checkpoint, FP8 KV cache, FP16 Gated-DeltaNet state,
FlashInfer attention, piecewise CUDA Graphs, 32 maximum sequences, a
32,768-token chunked-prefill budget, and no speculative decoding.

Run `qwen38-nomtp-piecewise-fp16-b32k-20260815` completed 32/32 requests and
8,192/8,192 output tokens at every row through Kairyu L3. Inputs are
approximately 8K tokens and each response contains exactly 256 output tokens.

| Concurrency | TTFT p50 | TTFT p99 | Mean TPOT | Output throughput | Wall time |
|---:|---:|---:|---:|---:|---:|
| 1 | 941.06 ms | 5,333.25 ms | 21.814 ms/token | 38.55 tok/s | 212.511 s |
| 8 | 401.51 ms | 1,684.24 ms | 24.432 ms/token | 298.25 tok/s | 27.467 s |
| 16 | 837.43 ms | 976.12 ms | 27.040 ms/token | 532.98 tok/s | 15.370 s |
| 32 | 1,465.87 ms | 1,482.19 ms | 31.956 ms/token | **867.58 tok/s** | 9.442 s |

### Tuning decision

The same fixed dataset and generation contract were used for each candidate:

| Candidate | c1 output | c8 output | c16 output | c32 output | c32 TTFT p50 | Result |
|---|---:|---:|---:|---:|---:|---|
| no MTP, 16K batch | 38.49 | 299.15 | 535.94 | 846.57 | 1,557.72 ms | stable baseline |
| no MTP, 32K batch | 38.55 | 298.25 | 532.98 | **867.58** | **1,465.87 ms** | selected |
| MTP-3, 16K batch | 53.16 | 279.02 | 405.18 | 654.25 | 6,300.13 ms | rejected |

The 32K budget improves saturated c32 throughput by 2.5% and median TTFT by
5.9% over 16K, while c8/c16 throughput changes by less than 1%. MTP-3 improves
single-stream decode but loses 22.7% c32 throughput, greatly increases c32
TTFT, consumes more graph memory, and disables sampling features including
`min_p` and `logit_bias`; it is therefore not suitable for the general OpenAI
API example.

Two startup candidates were also rejected before serving measurement:

- `FULL_AND_PIECEWISE` plus `custom_ops=["all"]` fails Qwen3.8 GDN tracing in
  vLLM v0.23.0, so the selected graph mode is `PIECEWISE`.
- A 65,536-token budget reaches Triton autotuning but fails with a CUDA illegal
  memory access. The selected 32K budget is the largest fully validated value.

The 32K configuration reserves capacity for 1,791,840 KV-cache tokens, or
6.84 concurrent requests at the full 262,144-token context.

### Reproducibility identity

- Date: 2026-08-14 UTC (2026-08-15 JST)
- GPU: GPU 0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
  SM 12.0; driver 595.84, CUDA 13.2; CPU set `0-15,64-79`
- Model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Model tree SHA-256:
  `9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e`
- vLLM release/source: `v0.23.0` /
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- vLLM image digest:
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`

> Historical baseline: the measurements below are for the superseded
> Qwen3.6/vLLM source-build example. They are retained for comparison and must
> not be interpreted as Qwen3.8 evidence.

## Result

The selected configuration uses FP8 weights and KV cache, FP16
Gated-DeltaNet state, FlashInfer attention, a 16,384-token chunked-prefill
budget, 32 maximum sequences, full/piecewise CUDA Graphs, and no speculative
decoding. It completed the full serving matrix through Kairyu L3.

### Fixed 8K/256 serving matrix

Run `final-no-mtp-20260811` issued 32 unique-prefix requests per row. Inputs
are approximately 8K tokens and every valid response contains exactly 256
output tokens (`ignore_eos=true`). Times include the HTTP and Kairyu L3 path.

| Concurrency | TTFT p50 | TTFT p99 | Mean TPOT | Output throughput | Wall time |
|---:|---:|---:|---:|---:|---:|
| 1 | 190.82 ms | 209.67 ms | 23.490 ms/token | 41.43 tok/s | 197.720 s |
| 8 | 492.25 ms | 561.55 ms | 25.388 ms/token | 294.25 tok/s | 27.840 s |
| 16 | 674.05 ms | 835.17 ms | 28.121 ms/token | 521.60 tok/s | 15.705 s |
| 32 | 1,246.77 ms | 1,387.43 ms | 33.737 ms/token | **842.35 tok/s** | 9.725 s |

All four rows produced 32/32 responses and 8,192/8,192 output tokens. The
runner rejects partial streams, missing token counts, zero-token results, and
multiple result artifacts even when an underlying client exits zero.

### Tuning decision

The final no-MTP configuration and the complete MTP-3 baseline used the same
prompts, seed, model, and 16,384-token prefill budget:

| Concurrency | no-MTP TTFT p50 | MTP-3 TTFT p50 | no-MTP output | MTP-3 output |
|---:|---:|---:|---:|---:|
| 1 | **190.82 ms** | 1,307.31 ms | 41.43 tok/s | **56.76 tok/s** |
| 8 | **492.25 ms** | 848.00 ms | 294.25 tok/s | **333.47 tok/s** |
| 16 | **674.05 ms** | 1,777.45 ms | **521.60 tok/s** | 452.22 tok/s |
| 32 | **1,246.77 ms** | 7,035.82 ms | **842.35 tok/s** | 561.08 tok/s |

No-MTP wins TTFT at every tested concurrency and aggregate throughput at the
service-saturation c16/c32 rows. It gives up 27.0% c1 and 11.8% c8 output
throughput, while gaining 15.3% at c16 and 50.1% at c32. Because the requested
target is TTFT plus general serving throughput, the committed configuration
optimizes the low-latency/high-concurrency envelope rather than single-stream
decode speed.

Two larger candidates failed the stability gate:

- MTP-5 reached 57.73 tok/s at c1, but fell to 302.66 tok/s at c8 and its
  EngineCore died with a CUDA illegal-memory access during c16. MTP-3 was the
  only speculative candidate to finish c32.
- Raising `max_num_batched_tokens` from 16,384 to 32,768 with MTP-3 exhausted
  the 94.97 GiB device during the first c32 prefill (`torch.OutOfMemoryError`,
  another 632 MiB requested with only 503.56 MiB free). All 32 streams produced
  zero tokens. The 16,384-token budget is the largest locally validated value.

The MTP candidates were automatically reduced from requested
`FULL_AND_PIECEWISE` graphs to `PIECEWISE`, because this FlashInfer/speculative
combination does not support full graphs. The selected no-MTP configuration
retains the requested full/piecewise graph modes.

## Reproducibility identity

- Date: 2026-08-11 UTC; Ubuntu 24.04.4, Linux 6.8.0-134-generic
- Host: 2 x Intel Xeon Gold 6530, 128 logical CPUs, four NUMA nodes
- Selected GPU: GPU 0, NVIDIA RTX PRO 6000 Blackwell Server Edition,
  97,887 MiB, SM 12.0, PCI `00000000:16:00.0`, driver 595.84, CUDA 13.2;
  process CPU set `0-15,64-79`
- Model: `Qwen/Qwen3.6-27B-FP8` revision
  `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; attested tree
  `f108556571d80514a792b458de366221c9b910fe69cbd5d2525c207580cd51aa`
- vLLM: source `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2`;
  image `sha256:99756b54424a4697f69476b29aa02fb7f8112aaa74fa8203a7bf8a0bae4ca6f1`
- Kairyu verification commit: `83241cc1a6ccfa06b8da8bb68672a9c64015ecca`
- Clean source-tree SHA-256:
  `98ac53c04aa1411e2ed26a7b2106de05c3179acde00b31eba2b6e7f02067da13`
- Served configuration SHA-256:
  `14e7f6b4421945279b4eda7b203541d15004e600f354786c056f7e44d75cb85c`
- Persistent paths: model, Open WebUI state, and vLLM compilation cache are
  bind-mounted below `/mnt/nvme/kairyu/model-volumes/qwen3.6-27b-1gpu/`

The official checkpoint does not provide calibrated FP8 attention Q/prob
scales, so vLLM warns that unit scales may affect output quality. This example also
overrides the checkpoint's FP32 Gated-DeltaNet state request with FP16 for the
measured latency/capacity configuration. Both choices are explicit and are part
of the served-configuration hash.

## Public context

Public numbers are useful ranges, not direct comparisons. A bare-metal RTX PRO
6000 Blackwell community report measured 117 tok/s for one stream, 377 tok/s
across four streams, and about 125 tok/s with its tuned MTP case. A separate RTX
6000 Ada FP8 report measured 161.5 average output tok/s and 668.5 peak aggregate
tok/s at concurrency 8–16 with an 8K maximum context. Quantization, GPU variant,
prompt length, maximum context, and vLLM revision differ from this run; links
and exact caveats are maintained in the example README.
