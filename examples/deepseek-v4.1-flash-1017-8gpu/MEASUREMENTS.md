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
| `vllm/vllm-openai:v0.32.0@sha256:befccf0e…` + this example's overlay → `local/vllm-openai:deepseek-v41-1017-8gpu` `sha256:995a89fcb6c9cf245b39143712bb3730837808e9bfb2b670d3f8abbfcbe65fbd` | 0.32.0 / 0.7.0.post1 / 2.13.0+cu130 | all nine anchors patched; encoder checks pass (efforts 50/75/100, `xhigh` 100, default 75, 1017 tool-turn case) | PASS 28/28 (`kernel-checks/kernels-v032.log`) |

The image follows the six-GPU 1017 example's S1 image selection (v0.32.0 or
the 2026-09-30 nightly, both with the same overlay).

## L1 selection

Pending on GPU (plan S1–S2): `verify.sh l1`, then `tune.py baseline
embedding-across-dp tp8` and the recipe levers one at a time on the selected
shape. Selection by c8 output throughput (the DeepSWE run's concurrency),
then c32.

`use_thp` (recipe B200 TP2 arm) is not a candidate: v0.32.0's EngramConfig
requires `dp_shared_memory=false` for it, i.e. one private pinned copy of
the 183 GiB tables per DP rank (8 × 183 GiB exceeds the host's 1 TiB).
