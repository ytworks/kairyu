# V4.1 ensemble evidence

Status: **GPU validation in progress.**

The user released all eight GPUs on 2026-09-13 (JST). GPU trials are recorded
below. The original CPU-only implementation and the sibling TP8 results do not
establish GPU correctness for this example.

## What is established

Local validation on 2026-09-12: **202 tests passed** in one invocation:

```sh
python -m pytest tests/unit/test_v41_ensemble_*.py \
  tests/unit/test_tiered_frontier_examplectl.py \
  tests/unit/test_deepseek_v41_example.py \
  tests/unit/test_replica_examplectl.py --no-cov -q
ruff check .
python scripts/check_progress_size.py
```

After the GPU-driven implementation changes, the expanded selected suite passes
**261 tests** on 2026-09-13, including the new native/L2 probe, capacity
attestation, kernel patch and parent-runtime build tests. Whole-repository Ruff,
progress-size and whitespace checks pass. Review reproduced and fixed a capacity
false pass where an altered context limit retained the original config-hash
environment variable; capacity now checks the actual entire DeepSeek command.

Docker Compose `config --quiet` passed using the lifecycle-generated environment
without starting containers. Configuration digest agreement and verification
command listing passed. An independent review's stale-deployment provenance
finding was fixed and re-reviewed; no actionable P1/P2 remained in that review.
The full repository test suite is left to CI; the tests above are the selected
local regression set. These are CPU/static checks only.

- The fixed existing V4.1 image and checkpoint source were inspected read-only.
  Exact image, source paths/hashes and constraints are in [L1-NOTES.md](L1-NOTES.md)
  and `example.json`. This is provenance and a static feasibility assessment.
- CPU contracts exercise deployment loading and Compose rendering; the real
  Conductor's effort/dependency/image/repair behavior with scripted engines;
  native OpenAI-compatible tool/image forwarding with an HTTP mock transport;
  role-specific mode/schema/budget middleware; lifecycle isolation and runtime
  attestation; stage coverage and measurement provenance.
- Requirement is fixed DeepSeek high, while Qwen retains its existing effort
  mapping. The removed image-description stage is not part of the new DAG.

## GPU startup investigation

Hardware: 8 × RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, PCIe;
1 TiB host RAM. Runtime image remains `027bf47b2bd6` (full ID in example.json).
The previous DeepSWE run completed before its TP8 service was stopped.

- `20260912-gpu-startup`: initial TP2/DP3/EP6, GPU-resident Engram, no DSpark,
  16K batching, 1M context and 0.90 memory utilization. Model loads at
  84.09 GiB/GPU, but sparse-indexer profiling fails allocating 512 MiB with
  only 377 MiB free. Full failure retained in `initial-worker.log`.
- `20260912-gpu-offload-startup`: only Engram CPU offload changes. GPU model
  memory becomes 52.62 GiB; two 15.74 GiB tables per rank consume about
  189 GiB pinned host RAM. Graph profiling completes and reports 21.42 GiB
  available KV/GPU and about 8.53M cache tokens per internal DP engine.
  API and correctness gates are still in progress; these capacity reports
  alone are not successful long-context requests.
- The offloaded service starts, but repeated short native requests expose NaN
  log probabilities, corrupt text and a tool grammar error. A first correct
  arithmetic answer was not sufficient evidence. Eager execution
  (`20260912-gpu-eager-startup`) and disabling custom all-reduce
  (`20260912-gpu-nccl-startup`) each reproduce the failure. Eager concurrent
  requests targeting all three DP ranks fail 10 of 12 cases.
- `20260912-gpu-finite-diagnostic`: a separate temporary image adds finite
  checks around Engram, Attention and MoE. The first nonfinite tensor is layer
  0's Attention output on both TP ranks of the second request. This diagnostic
  image is not the measured serving candidate.
- An independent GPU-6 sparse-attention oracle poisons unused KV rows with
  NaNs: all 12 cases containing masked `-1` indices produce NaNs, while all
  36 control cases remain finite. The independent reference remains finite
  in all 48 cases. Invalid indices are clamped to slot zero by the original
  gather; a zero attention weight cannot eliminate `0 * NaN` contamination
  in the value product. Raw diagnostics and failed builds are retained in the
  persistent kernel-investigation directory below.

## Masked-KV correction

Selected child image:
`sha256:8b8bcb200f3a77f289f2483526903df34c7f7b15f58691d8ab27c0ccb15e2eaf`.
The parent image and sibling example remain unchanged. Invalid gathers use an
aligned zero device row; valid addresses, transfer sizes and barriers are
unchanged. Dedicated decode, common gather/footer and prefill pointer paths are
covered. CUDA compilation exposed missing declaration visibility and a larger
DOTS3 row ABI in the first two builds; both failed builds are retained. They
were never accepted for serving verification.

- Paired oracle: **48/48 pass**, zero nonfinite outputs, **48/48 bitwise equal**
  to the original runtime's corresponding finite-cache outputs. The old runtime
  passes 36 controls and fails all 12 poisoned masked cases on the same inputs.
- Dimensions cover 8/32 heads (TP8/TP2), 1/22/48 query tokens, extra page size
  32/64, masked/unmasked and zero/NaN unused cache. Independent dequantization and
  attention use unchanged `atol=0.05, rtol=0.05`; maximum absolute error is
  0.0110161. These paired cases scale source amplitude by 0.25 to isolate the
  masked-cache failure from unrelated random FP8 tolerance outliers.
- The sibling's unchanged full-amplitude **16/16 numerical cases also pass**.
- SP communication diagnostic: 168 all-gathers and 168 reduce-scatters per rank
  match exactly. On this PCIe host these SP operations already use NCCL fallback;
  the test does not claim to exercise the custom communication implementation.

Raw kernel evidence is under `20260913-kernel-investigation/` relative to the
evidence root below (57 files, 11,891,234 bytes). Copies preserve all original
artifact hashes. This establishes the isolated kernel fix; full-model native,
composed L2, context and performance gates remain separate.

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `2a9c66eafd5bfba3e6127ebd19a7eaad5272c77ecb0750eca1e18ba87ad32d21` |
| `v41-masked-kv-fix-20260913/final-summary.json` | `843233132c42c2781e887d059b6aa18a65b61023526b6d44c8b6231730254c82` |
| `v41-masked-kv-fix-20260913/patched-v3-results.json` | `99f65990f6a7bb100e3ae34ed296a5ddd5139f59c34e820bdc8a3c5215cc7ad8` |
| `v41-masked-kv-fix-20260913/baseline-results.json` | `f7922e899d869059f8ff48a7545c8113be2c61013b279bcbf1250ce158948438` |

Raw trial directories are under
`/mnt/nvme/kairyu/model-volumes/qwen3.8-deepseek-v4.1-8gpu/verification-results/`.
Each retains git state, hardware inventory, candidate spec/config hash and
complete worker logs.

## What remains unmeasured

TP2×DP3/EP6 initialization, memory fit, collective/kernel correctness, actual
native tokenizer/schema/thinking-budget enforcement, image understanding, tool
behavior on generated model outputs, restart/cancellation/co-residency, long
context, concurrency, latency and throughput all remain open.

The sibling TP8 V4.1 result and PR #595's Qwen Requirement measurements are **not**
measurements of this example. No old outputs, throughput numbers or baseline
fallbacks were copied here. The 1M DeepSeek and 256K Qwen settings are candidate
limits, not verified usable capacity for this combined deployment.

## Next evidence record

Follow [README.md](README.md#gpu-validation-execution-order). Record
checkout SHA, exact runtime image IDs, checkpoint attestation, configuration
hash, request/response/trace artifacts, first failures and explicit skipped
gates. Preserve protocol results separately from model/task quality and human
semantic review. Mark only completed gates as passed; a completed implementation
or a healthy process alone is insufficient to close the remaining gates.
