# V4.1 six-plus-two measurements

Evidence for this new example only. The original V4 example remains unchanged.

## V41C six-GPU preflight: numerical gate FAIL (2026-09-13)

These results belong to draft PR #601, starting from main `99c5eadb`.
They use the main V4.1 runtime and checkpoint assets, not the closed
ensemble branch. The existing deployment was preserved for restoration;
its successful restart does not validate this candidate.

| Identity / candidate setting | Value |
|---|---|
| Image ID | `sha256:027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359` |
| vLLM source | `179dd0fa9` from the pinned image |
| Checkpoint revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Checkpoint config SHA-256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Candidate topology | GPU 0–5, TP2 × attention DP3, EP6, one frontend; Qwen remains on 6/7 |
| Runtime candidate | CPU Engram offload, DSpark 5, NCCL, Marlin, 16,384 batch tokens, 32 sequences per DP engine, utilization 0.90 |
| Native input | `deepseek_v41` tokenizer/reasoning/tool parsers, context 1,048,576; default template `thinking:false` |
| Cache configuration | 64-token blocks, BLHNC, FP8 MLA KV, MXFP4 indexer |

The checkpoint has 64 attention heads and 8 output groups, so TP6 is not
a supported partition. Pinned source supports TP2/DP3 with EP6. Its ordinary
MoE mapping distributes 384 target experts evenly and 128 draft experts
as 22/22/21/21/21/21; EPLB stays disabled. Uneven draft placement is therefore
not a source-level reason to remove DSpark, but startup correctness is open.
Header-only weight accounting found 475.241 GiB total, including 188.833 GiB
Engram; offload and runtime memory overhead still require actual startup
measurement. This table describes an unselected candidate.

The numerical probe runs on one SM120 GPU with the candidate's **32 local
attention heads**. It is not a six-GPU collective or full-model test.
It uses the existing main
`examples/deepseek-v4.1-flash-8gpu/check_sm120_pages.py` with only the Q-head
and sink dimensions changed from 8 to 32. Seed 4175, case order, packed
cache construction and the independent reference remain unchanged.

At 13:49 UTC, eight cases passed before the first secondary-page-64,
one-token, top-k-128, unmasked case failed `atol=0.05, rtol=0.05`.
One of 16,384 elements failed: index `[0, 26, 425]`, reference
`-0.5795126557350159`, kernel `-0.486328125`, absolute error
`0.09318453073501587`. The original PyTorch exception reports the greatest
error among failing elements; the global maximum absolute error is
`0.16298317909240723` at an element that passes its relative tolerance.

The reference unpacks the **actual** FP8 KV bytes and BF16 RoPE values,
then uses original BF16 Q promoted to FP32 for QK/softmax/PV. Pinned kernel
source additionally quantizes Q NoPE and softmax-weight/value-scale
products to FP8 and rounds split/output values to BF16. These differences
are relevant to further diagnosis; they do not establish which operation
caused this outlier or justify relaxing the gate.

At 14:04 UTC, a controlled diagnostic retained the failing fixture and
passed its heads 24–31, corresponding sinks, identical packed KV bytes
and indices through the eight-head kernel. Its 4,096 outputs were
**bit-exact** to the corresponding 32-head outputs. Both fail the same
reference element. This fixture's failure is not specific to 32 heads;
the original numerical gate remains **FAIL**.

The first probe stopped the idle old gateway/DeepSeek and retained their
original containers. They were restored healthy by 13:55 UTC; gateway
`/readyz` returned `ready`, and both Qwen workers and UI were healthy.
The follow-up ran without stopping services, with a 2% PyTorch GPU allocator
cap, 8 GiB container memory and four CPUs. A read-only Triton cache setup
error was recorded and corrected before numerical execution. Peak PyTorch
allocation was 315,577,856 bytes; all original services remained healthy
and diagnostic GPU memory was released at 14:06 UTC.

Selected evidence is committed under
[`measurements/20260913T134938Z-v41-preflight/`](measurements/20260913T134938Z-v41-preflight/):
the original failure, its log (`.txt`, trailing whitespace removed), same-fixture result
and artifact manifest. Full scripts, synthetic tensors, source excerpts,
candidate compose and setup logs are retained at
`/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4-8gpu/verification-results/20260913T134938Z/v41-sixgpu-preflight/`.
All manifest hashes refer to the raw archived files and were verified after
archival. The normalized repository log has 1,931 bytes and SHA-256
`9c92806ce6bc5ed5c516f850f4eabce64bb88875845c2cdace01f624a038721b`;
the raw log remains unchanged in the archive. In particular the
744,468-byte fixture SHA-256 is
`5e0d936bc47f1afbf8862f580714611d66fc6fb6b24d16f05cb8a84cf8618979`.

No full-model six-GPU startup, native API gates, AUTO deployment,
natural/forced-primary performance matrix or paired baseline was attempted.
The remaining numerical discrepancy must be resolved before promoting
this runtime candidate; previous TP8/V4/closed-branch results do not close it.

### Native non-thinking configuration: CPU rendering only

The immutable image's `ChatCompletionRequest.build_chat_params` and exact
checkpoint tokenizer were exercised with default template kwargs
`{"thinking":false}`. Omitted effort closes thinking; explicit effort enables
it with the checkpoint's native budgets. No framework or tokenizer patch
is needed for this setting. This applies to the candidate ensemble L1;
the standalone example's default remains unchanged.

| Top-level request effort | Thinking enabled | Native budget | Final marker |
|---|---|---:|---|
| omitted | false | none | `</think>` |
| low | true | 50 | `<think>` |
| high | true | 75 | `<think>` |
| max | true | 100 | `<think>` |

[`native-thinking-render.json`](measurements/20260913T134938Z-v41-preflight/native-thinking-render.json)
records image/checkpoint/source identities and rendered-prompt hashes;
SHA-256 `bd743b479f98b2470fa857c7a2f56e35589d9abddfb43e71fa07a29c7f70b83c`.
These are CPU renderer results, not live generated-response gates. The
Requirement high floor with explicit max inheritance remains separate work.

## V41C arithmetic diagnosis: 16/16 PASS, floating fidelity FAIL retained (2026-09-13)

A bounded follow-up completed the entire original 16-case matrix at **32
heads** on the same immutable main image. No service was stopped for this
follow-up. The independent CPU arithmetic reference passes every case at
unchanged `atol=0.05, rtol=0.05`. The original BF16-Q floating-reference
comparison passes 14/16 cases and remains **FAIL**: secondary-page-64,
one-token, unmasked top-k-128 and top-k-192 each have one failing element.
Every main/secondary-page, decode/prefill and masking combination remains
an individually reported case; no failed case or fixture was removed.

The independent reference decodes actual stored FP8 KV bytes, UE8M0 footer
scales and BF16 RoPE, then computes with PyTorch CPU contractions. Its
formats follow the actual source-defined operation:

- Decode rounds each 64-dimensional Q NoPE group to FP8 E4M3 with a
  power-of-two scale, while Q RoPE remains BF16. Each 64-candidate tile
  rounds softmax weight × value-group scale to FP8 for NoPE PV; RoPE PV
  uses BF16 weights. Split outputs round to BF16 before the FP32
  LSE-weighted merge, sink normalization and final BF16 store.
- Both full-tile and masked dual-cache prefill dispatch to **BF16 QK**,
  including the dequantized-KV-to-BF16 conversion. They use the same FP8
  weight/value arithmetic with a single FP32 online accumulator and a
  final BF16 store. Applying decode's Q quantization to prefill would be
  the wrong arithmetic reference.

The original failing fixture is replayed with exactly identical Q, indices
and kernel output; its 744,468 bytes and SHA-256 above are unchanged.
A CPU ablation isolates the saved `[0, 26, 425]` coordinate:

| Computation | Value |
|---|---:|
| Original floating reference | -0.5795126557350159 |
| Independent reference with Q quantization only | -0.489356130361557 |
| Independent pinned arithmetic with split rounding | -0.486328125 |
| Actual kernel | -0.486328125 |

The arithmetic reference itself fails the original floating-fidelity
comparison at this coordinate. Together with the earlier bit-exact
8-head/32-head slice comparison, this attributes this fixture's outlier to
the intended quantized operation, principally Q quantization. It is not
evidence of a 32-head indexing defect. It also does not prove full-model
answer quality or close the retained floating-fidelity gate.

Across all 16 cases, maximum absolute kernel/arithmetic residual is
`0.0078125`. All eight decode cases have the expected actual active split
count for one chunk per split, and their per-split output comparisons pass.
Maximum split LSE residual is `1.9073486328125e-6` in log2 units. The oracle
models formats and mathematical operations independently, without copying
CUDA thread/indexing logic; it does not reproduce CUDA exp2 or FP32 MMA
reduction rounding bit-for-bit.

The same arithmetic is integrated into the existing standalone probe;
no new example Python file was added. Its default floating-reference gate
is unchanged. Reproduce the separate arithmetic gate inside the pinned
image with one available SM120 GPU and the existing example mounted:

```sh
python3 check_sm120_pages.py --heads 32 --arithmetic --gpu-memory-fraction 0.02
```

The integrated entry point checks actual planner selection, rejects a
calibrated split policy outside this oracle's envelope, and completed all
16 cases with `gate: quantized_arithmetic`, `passed: true`,
`float_reference_passed: false`. The separate scratch-inspection replay
peaked at **315,594,752 bytes** of PyTorch CUDA allocation under the 2%
allocator cap. Original gateway/DeepSeek stayed ready and idle; diagnostic
GPU memory was released. Full-model startup remains unattempted.

Selected JSON and the integrated log are committed alongside the original
preflight evidence:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| [Arithmetic ablation](measurements/20260913T134938Z-v41-preflight/quantized-arithmetic-result.json) | 3,113 | `741d14bd0486c80219dc9291861fd0a3db2c7cf76e735f57115e47ce28f1e24c` |
| [Full matrix and actual split checks](measurements/20260913T134938Z-v41-preflight/quantized-matrix-result.json) | 12,616 | `07b039d04293e3e99f92f41fc10f270d8da7a091b2c65ab3802b534015bd7cd2` |
| [Integrated entry-point log](measurements/20260913T134938Z-v41-preflight/quantized-main-oracle.txt) | 5,834 | `574c62172a124c2ece22b88d2030927e30ae59ad47ca77a13b34625e1d1c4f34` |
| Executed `check_sm120_pages.py` | 13,091 | `d2ab3aeb10d189b6fd6b52510a8603d23c049a8f6a4615b9a70c500d57c17a1e` |

The [arithmetic manifest](measurements/20260913T134938Z-v41-preflight/arithmetic-artifact-manifest.json)
records every source and evidence hash. In particular, pinned FlashInfer
`common/fp8_quant.cuh` is SHA-256
`4de8f104e3ccdfa6c5b82eb340929a010a35d1ed73a87b62c85709cc2e7af745`,
`decode_dsv4_kernel.cuh` is
`e9d939bd6cc23cdfb1c8a9be0811e0d8ee3307a930a67ff02c30c4450299ab7d`,
and `prefill_mg_kernel.cuh` is
`0ec1603dca10cdbdec9795d82736a9e3662a91dfb212220e381ee1696e12a179`.
The complete 23-file, **1,516,420-byte** archive, including synthetic tensors,
source snapshots, diagnostic scripts and two corrected diagnostic-instrumentation
errors, is under the existing NVMe preflight path's `arithmetic/` directory.
Every archived byte count and SHA-256 was checked after copying. Neither
error changed the fixture, mathematical reference, tolerance or original
failure.
