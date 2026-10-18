# deepseek-v4.1-flash-1017-8gpu evidence

Status: **L1 selection in progress (2026-10-19).** Only the rows below with
a run or log name are measured.

Hardware: 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB,
SM120, PCIe); 1 TiB host memory, 4 NUMA nodes with GPU pairs
(0,1)(2,3)(4,5)(6,7) each on one node.
Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
Artifacts live under
`/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-1017-8gpu/`.

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

The checkpoint (the six-GPU 1017 example's download, hard-linked) hashes to tree SHA-256
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

## Runtime image

| Image | vLLM / FlashInfer / torch | Overlay build | Kernel gate (`check_sm120_kernels.py`) |
|---|---|---|---|
| `vllm/vllm-openai:v0.32.0@sha256:befccf0e…` + this example's overlay → `local/vllm-openai:deepseek-v41-1017-8gpu` `sha256:7237421efe0a2eb901560ff550600c02e90310fda100caded27ff8475bf19b0c` | 0.32.0 / 0.7.0.post1 / 2.13.0+cu130 | all nine anchors patched; encoder checks pass (efforts 50/75/100, `xhigh` 100, default 75, 1017 tool-turn case) | PASS 28/28 (`kernel-checks/kernels-v032-r2.log`; the overlay accepts only v0.32.0, the six-GPU 1017 example's S1 selection) |

The image follows the six-GPU 1017 example's S1 selection: v0.32.0 over the
2026-09-30 nightly with the same overlay (c32 +7.6 %, c1/c8 within noise;
`deepseek-v4.1-flash-1017-6gpu/MEASUREMENTS.md`).

## L1 selection

Rule (the six-GPU amendment's): one change at a time; adopted only with
≥ 5 % output throughput at c1, c8 or c32 and no > 5 % loss elsewhere; the
replica shape is chosen by c8 throughput (the DeepSWE run's concurrency),
then c32. Rows (`tune.py`): 32 requests of ≈ 8K unique input tokens through
Kairyu's `/v1/responses`, capped at 256 output tokens, thinking high.
Cells: output tok/s / model TTFT p50 ms. Single trials.

### `l1` on DP8 / EP8

Run `20261018T204321Z-l1`: **PASS 17/17.** Rendered efforts default 75,
low 50, high 75, max 100, `xhigh` 100, chat closes thinking; vLLM's prompt
tokens equal the checkpoint encoder's for the 1017 tool-turn case and the two
golden chat fixtures; 12 concurrent `17 * 19` probes per variant all exact
with finite log-probabilities, answered by all eight DP engines; L1 default
sampling `{'temperature': 1.0, 'top_p': 1.0}`; tool call and image pass.
Weights 47.55 GiB per GPU (48 of 384 experts per rank), 25.03 GiB KV,
13,799,928 KV tokens per DP engine; idle GPUs at 87,633 MiB.

### S1: replica shape

Run `20261018T204356Z-tuning` on v0.32.0, starting from DP8 / EP8 (the
committed command at the time, with the six-GPU 1017 levers).

| Candidate | c1 | c8 | c32 | KV per GPU / tokens per engine | Result |
|---|---|---|---|---|---|
| DP8 / EP8, Engram in shared host memory (running) | 101.0 / 726 | 291.7 / 1,216 | 620.2 / 4,319 | 25.03 GiB / 13,799,928 | PASS |
| DP8 / EP8, restarted | 98.7 / 719 | 288.9 / 1,378 | 655.8 / 4,331 | 25.03 GiB / 13,799,928 | PASS |
| DP8 / EP8 + `embedding_across_dp` (Engram sharded in HBM) | — | — | — | — | **FAIL** at start: 1.51–1.84 GiB KV per GPU, below the 1.9 GiB one 1,048,576-token request needs |
| **TP8 / EP8**, Engram TP-sharded in HBM, NCCL (`--disable-custom-all-reduce`) | **152.2 / 470** | **376.2 / 1,515** | **458.2 / 7,017** | 15.70 GiB / 8,613,322 | PASS |

Against the mean of the two DP8 rows (c1 99.9, c8 290.3, c32 638.0), TP8
is +52 % at c1, +30 % at c8 and −28 % at c32. By the plan's rule (c8, the
concurrency of the DeepSWE runs, decides; c32 second) **TP8 / EP8 is
adopted**; DP8 keeps the higher aggregate rate at c32. Weights are 61.16 GiB
per GPU on TP8 (47.55 GiB on DP8 with the tables in host memory). For
comparison, `deepseek-v4.1-flash-8gpu` (0909 weights, TP8 on its 0909 image,
chat rows of exactly 256 tokens) measured 126.8 / 275.4 / 326.8 tok/s at
c1/c8/c32.

`use_thp` (recipe B200 TP2 arm) is not a candidate: v0.32.0's EngramConfig
requires `dp_shared_memory=false` for it, i.e. one private pinned copy of
the 183 GiB tables per DP rank (8 × 183 GiB exceeds the host's 1 TiB).
