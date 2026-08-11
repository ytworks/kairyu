# Measured performance and accuracy

## Result

The final candidate is FP8 weights and KV cache, FP16 Gated-DeltaNet state,
FlashInfer attention, a 16,384-token chunked-prefill budget, 32 maximum
sequences, and no speculative decoding. Its c32 screen reached 846.89 output
tok/s with 1,210.58 ms median TTFT, versus 561.08 tok/s and 7,035.82 ms for
MTP-3. The clean final matrix and LiveCodeBench rerun are in progress; the
tables below retain the completed MTP-3 baseline until that replacement is
finished.

### Fixed 8K/256 serving matrix — MTP-3 baseline

Run `tuning-mtp3-20260811` issued 32 unique-prefix requests per row through
Kairyu L3. Inputs are approximately 8K tokens and every valid response contains
exactly 256 output tokens (`ignore_eos=true`). Times include the HTTP and L3
path.

| Concurrency | TTFT p50 | TTFT p99 | Mean TPOT | Output throughput | Wall time |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,307.31 ms | 1,921.11 ms | 12.486 ms/token | 56.76 tok/s | 144.337 s |
| 8 | 848.00 ms | 3,881.68 ms | 17.436 ms/token | 333.47 tok/s | 24.566 s |
| 16 | 1,777.45 ms | 4,346.52 ms | 24.762 ms/token | 452.22 tok/s | 18.115 s |
| 32 | 7,035.82 ms | 8,087.26 ms | 28.262 ms/token | 561.08 tok/s | 14.600 s |

All four rows produced 32/32 responses and 8,192/8,192 output tokens. The
runner now rejects partial streams, missing token counts, zero-token results,
and multiple result artifacts even when an underlying client exits zero.

### Tuning decision

The same prompts and seed compared native MTP depths three and five. The
five-token candidate reached 57.73 tok/s at c1 versus 56.76 tok/s for MTP-3,
but its TTFT was slightly worse (1,319.65 versus 1,307.31 ms). At c8 it fell to
302.66 tok/s versus 333.47 tok/s and mean TPOT rose to 20.123 versus 17.436
ms/token. During c16, its EngineCore terminated with a CUDA illegal-memory
access; the asynchronous stack surfaced while FlashInfer attention metadata
was being built. The following c32 requests produced no tokens. MTP-5 is
therefore neither faster across the matrix nor stable enough for this image and
hardware. MTP-3 is the measured winner among the speculative candidates.

A second c32 screen raised `max_num_batched_tokens` from 16,384 to 32,768 while
keeping MTP-3 and every request parameter fixed. The engine exhausted the
94.97 GiB device during the first prefill (`torch.OutOfMemoryError`, another
632 MiB requested with only 503.56 MiB free), so all 32 streams produced zero
tokens. The complete 16,384-token run is retained as the largest locally
validated budget for this full-native-context configuration.

Finally, disabling MTP at the stable 16,384-token budget improved the fixed c32
screen to 846.89 output tok/s and 1,210.58 ms median TTFT. That is 50.94% more
aggregate output throughput and 82.79% lower median TTFT than MTP-3 on the same
32 prompts. Mean TPOT was higher (33.710 versus 28.262 ms/token), so the final
selection optimizes the requested aggregate throughput and TTFT rather than
single-request inter-token latency under saturation.

vLLM automatically changes the requested `FULL_AND_PIECEWISE` CUDA Graph mode
to `PIECEWISE` because this FlashInfer/speculative-decode combination does not
support full graphs. The effective serving mode is therefore piecewise graphs.

## LiveCodeBench-20 — MTP-3 baseline

Run `qwen36-fp8-mtp3-lcb20-20260811` completed the deterministic first 20
`release_v6` items selected with seed 0. It used pass@1, temperature 1.0,
top-p 0.95, `reasoning_effort=max`, a 32,768-token output ceiling, concurrency
16, and the content-addressed networkless Docker executor.

| Metric | Result |
|---|---:|
| Correct / scored | 14 / 20 |
| Simplified pass@1 | **70.0%** |
| Wilson 95% confidence interval | 48.10%–85.45% |
| Request / scoring errors | 0 / 0 |
| Retries / unmeasured requests | 0 / 0 |
| TTFT p50 / p95 | 4,926.02 / 4,929.01 ms |
| Per-request generation TPS p50 | 61.85 tok/s |
| Completion tokens | 479,332 |
| End-to-end run window | 767 s |
| Aggregate output throughput | 624.94 tok/s |
| Request latency p50 / p95 | 445.267 / 538.186 s |
| Median output length | 26,468 tokens |
| Requests reaching 32,768-token limit | 7 / 20 |

The 70% figure is a quick 20-question signal, not a full benchmark score. The
scoreboard marks it non-comparable with the complete 1,055-problem release_v6
run. Seven limit hits also show that this model's maximum-thinking mode is very
verbose; the latency and throughput values intentionally retain that workload
rather than shortening it after the fact.

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
- Kairyu benchmark commit: `dd65716c7e90703d1631bf79df230a10c46b59c1`
- Served configuration SHA-256:
  `7167e456b76ef62dcd0be11f88c1309ff77998614bc2f3a74363ee2964fc4443`
- LiveCodeBench dataset revision:
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- Execution image:
  `sha256:9c1efcecac25ac2e1ce1cc284687633616110eede6412d31b1367e79c3f5f7d1`
- Persistent paths: model, Open WebUI state, and vLLM compilation cache are
  bind-mounted below `/mnt/nvme/kairyu/model-volumes/qwen3.6-27b-1gpu/`

The official checkpoint does not provide calibrated FP8 attention Q/prob
scales, so vLLM warns that unit scales may affect accuracy. This example also
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
