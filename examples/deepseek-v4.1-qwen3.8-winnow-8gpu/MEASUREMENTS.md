# deepseek-v4.1-qwen3.8-winnow-8gpu evidence

Host: 8 x RTX PRO 6000 Blackwell Server Edition (SM120), PCIe. DeepSeek-V4.1
DP6/EP6 on GPUs 0-5 (image `sha256:119afb09…`, the six-GPU example's SM120
overlay), Qwen3.8-27B FP8 on GPU 6 (vLLM v0.23.0), Winnow-12B Q8_0 on GPU 7
(winnow-server `77d1458` + f072b10).
Raw evidence: `/mnt/nvme/kairyu/model-volumes/deepseek-v4.1-qwen3.8-winnow-8gpu/results/`.

No GPU gate has run on this configuration yet (VCO-D18, 2026-10-06).

The evidence of the checklist-verified configuration this example replaced
(OpenJev judge, checklist DAG) is in this file's history before VCO-D18, as
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md` at `df109a6b`.
