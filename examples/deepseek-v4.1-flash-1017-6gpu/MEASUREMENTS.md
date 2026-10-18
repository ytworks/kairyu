# deepseek-v4.1-flash-1017-6gpu evidence

Status: **L1 selected (2026-10-19); final gates in progress.**

Hardware: 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB,
SM120, PCIe) out of an 8-GPU host; 1 TiB host memory, 4 NUMA nodes with
GPU pairs (0,1)(2,3)(4,5)(6,7) each on one node.
Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
Artifacts live under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-6gpu/`.

## Checkpoint against 0909

Compared with `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a`
(the 0909 examples' pin): `config.json`, `tokenizer_config.json`,
`inference/config.json`, the 96,085 tensor names of
`model.safetensors.index.json` and its `total_size` (510,286,023,000 bytes)
are identical; the 48 weight shards and `encoding/` changed, and the
technical report PDF, `assets/` and `evaluation/` are no longer shipped.

`encoding/encoding.py` changes (the model card's "minor update"):

- `find_last_user_index` skips a user message whose content blocks hold
  only tool results: without a tools declaration, reasoning before the
  latest *user request* is dropped, and the current turn keeps its
  reasoning across tool calls (0909 dropped it at the first tool result).
  With a tools declaration nothing is dropped, as before.
- tool messages may carry content blocks (images in tool results), user
  content may be a list, and images are collected after tool results are
  merged into prompt order.

The download (97 min, plain `snapshot_download` at the pinned revision on
NVMe) hashes to tree SHA-256
`e25bddd997c2e7b959f444bb70813e210b813b8c528bf25e7a0e25b1d89bb807` (83 files,
`model-manifest.json`); every size and all 49 LFS SHA-256 values equal
Hugging Face's. Identical to 0909: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors.index.json`, the encoder test
fixtures and most of `inference/`.

The reference `inference/model.py` changed in two places: only an index-key
owner layer (`kv_source_layers`) publishes the shared index keys, and the
attention output `o` is FP8 quantized and dequantized before the `wo_a`
einsum (the production FP8 `wo_a` GEMM's numerics); `tilelang` moves to
0.1.12. vLLM keeps its own implementation of both; nothing is ported.

vLLM's encoder copy (`vllm/tokenizers/deepseek_v41_encoding.py`) is the same
file in the 2026-09-30 nightly (`ac68c3087`) and v0.32.0 and keeps the 0909
rule. vLLM's tokenizer wrapper flattens list content to text with image
placeholders before encoding, so only the turn rule changes what vLLM
renders; `patch_sm120.py` ports it. On the host: both vLLM copies fail the
checkpoint's own 1017 test case before the edit ("the current turn's
reasoning is dropped at its tool result") and pass it after, as the 1017
reference encoder does.

## Runtime images

| Image | vLLM / FlashInfer / torch | Overlay build | Kernel gate (`check_sm120_kernels.py`) |
|---|---|---|---|
| `vllm/vllm-openai:v0.32.0@sha256:befccf0e…` + overlay → `local/vllm-openai:deepseek-v41-1017-6gpu` `sha256:3dd34fc8c9ff103a3d9f0104176da77bbb2345a0643becb441fc6363f9986687` | 0.32.0 / 0.7.0.post1 / 2.13.0+cu130 | all nine anchors patched; encoder checks pass (efforts 50/75/100, `xhigh` 100, default 75, 1017 tool-turn case) | PASS 28/28 (`kernel-checks/kernels-v032.log`) |
| `vllm/vllm-openai:nightly@sha256:b5406fca…` + overlay → `local/vllm-openai:deepseek-v41-1017-6gpu-nightly` `sha256:7a1fb67d1cb7d24983eff1ca299d66e5bde9270b29a04bf495fc2e35f1320fc3` | 0.30.1rc1.dev396+gac68c3087 / 0.7.0.post1 / 2.13.0+cu130 | same | PASS 28/28 (`kernel-checks/kernels-nightly.log`) |

The anchors of every overlay edit are identical in the two vLLM versions
(checked against the `ac68c3087` and `v0.32.0` sources before the builds).
The SM120 edits are still missing upstream in v0.32.0: SWA pages are
hard-coded to 32, MLA/indexer pages exclude SM120, the MXFP4 indexer is
SM100-only, the split top-p cutoff has no guard, and FlashInfer 0.7.0.post1
is unchanged.

## L1 selection

Rule (the six-GPU amendment's): one change at a time; adopted only with
≥ 5 % output throughput at c1, c8 or c32 and no > 5 % loss elsewhere.
Rows (`tune.py`): 32 requests of ≈ 8K unique input tokens through Kairyu's
`/v1/responses`, capped at 256 output tokens, thinking high (almost every
request reaches the cap inside its reasoning). Cells: output tok/s / model
TTFT p50 ms. Single trials.

### `l1` on the v0.32.0 image

Run `20261018T181734Z-l1`: **PASS 17/17.** Rendered efforts default 75,
low 50, high 75, max 100, `xhigh` 100, chat closes thinking; vLLM's prompt
tokens equal the checkpoint encoder's for the 1017 tool-turn case (no tools
declared: the current turn's `r2` reasoning is kept across its tool result)
and the two golden chat fixtures; 12 concurrent `17 * 19` probes per
variant (default/low/high/max/chat) all exact with finite log-probabilities,
answered by all six DP engines; L1 default sampling `{'temperature': 1.0,
'top_p': 1.0}`; tool call and image pass. Weights 59.02–59.07 GiB per GPU,
14.88 GiB KV, 8,202,752–8,213,527 KV tokens per DP engine (7.8 × a
1,048,576-token request). The first run (`20261018T181644Z-l1`) failed only
`encoder_golden_1_tools`: the checkpoint fixture's tool calls carry no `id`,
which vLLM's chat API requires; the gate now gives each call the id of the
tool result that answers it, for vLLM and the reference encoder alike.

### S1: runtime image

Run `20261018T181801Z-tuning` (committed DP6 / EP6 command on both images).

| Candidate | c1 | c8 | c32 | KV tokens per engine |
|---|---|---|---|---|
| v0.32.0 (running baseline) | 101.8 / 713 | 284.8 / 1,495 | 665.6 / 5,060 | 8,213,527 |
| v0.32.0, restarted | 102.4 / 709 | 276.4 / 1,507 | 678.6 / 5,077 | 8,202,752 |
| 2026-09-30 nightly | 100.7 / 685 | 289.4 / 1,336 | 624.4 / 5,240 | 8,283,516 |

Against the mean of the two v0.32.0 rows (c1 102.1, c8 280.6, c32 672.1),
the nightly is −1.4 % / +3.1 % / −7.1 %: no gain of 5 % anywhere, so
**v0.32.0 is adopted**. The two v0.32.0 rows differ by 3.0 % at c8 and 1.9 %
at c32, which sizes single-trial noise. The nightly candidate passed
`tune.py`'s readiness probes (exact answers from every DP rank, tool call,
image); its full `l1` was not run because it is not adopted.

`use_thp` (recipe B200 TP2 arm) is not a candidate: v0.32.0's EngramConfig
requires `dp_shared_memory=false` for it, i.e. one private pinned copy of
the 183 GiB tables per DP rank (6 × 183 GiB exceeds the host's 1 TiB).

### S2: recipe levers on v0.32.0

Run `20261018T185202Z-tuning`, one change each against the committed
command; percentages against the mean of the two v0.32.0 rows of S1.

| Candidate | c1 | c8 | c32 | KV tokens per engine | Result |
|---|---|---|---|---|---|
| adaptive verification (NVIDIA DSpark default) | — | — | — | — | **FAIL** at startup: `Adaptive verification trims verification requests on device, which the DeepseekV41IndexerBackend attention backend does not support` (as on the nightly) |
| `--prefill-schedule-interval 64` | 102.6 / 703 (+0.5 %) | 270.9 / 2,045 (−3.5 %) | 541.6 / 5,072 (−19.4 %) | 8,202,752 | rejected |
| `--max-cudagraph-capture-size 1024` | 100.8 / 704 (−1.3 %) | 267.0 / 1,411 (−4.8 %) | 656.3 / 4,949 (−2.4 %) | 6,212,106 | rejected (and 24 % less KV) |
| batch 8K | 104.7 / 639 (+2.5 %) | 292.8 / 1,488 (+4.3 %) | 668.8 / 5,676 (−0.5 %) | 4,926,403 | rejected (no ≥ 5 % gain; 40 % less KV) |

No lever reaches the 5 % line, so nothing is combined (S3 not needed).

## Selected L1

vLLM v0.32.0 + this overlay (image `sha256:3dd34fc8…`, the one every row
above and the final gates ran on), DP6 / EP6 (TP1), Engram tables in shared
pinned host memory, `FLASHINFER_MLA_SPARSE_DSV41`, FP8 KV, MXFP4 indexer,
64-token pages, Marlin MoE, DSpark 5 (full verification),
`--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.92`,
`--max-num-seqs 128`, prefix caching, temperature 1.0 / top_p 1.0,
thinking high by default. The overlay keeps accepting the nightly so S1 can
be repeated (`tune.py image-nightly`).
