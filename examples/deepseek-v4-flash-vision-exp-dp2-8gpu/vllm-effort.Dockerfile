# syntax=docker/dockerfile:1.7
# Served vLLM image: the pinned DeepSeek-V4 build (example.json "vllm") plus
# one prompt-rendering edit. Kairyu sends an L3 `max` request to vLLM's
# /v1/responses as `reasoning.effort: "xhigh"` (m11 D4 amendment 2026-10-17);
# the pinned DeepSeek-V4 tokenizer renders it as high, so
# patch_deepseek_v4_effort.py renders xhigh as max. The build fails when the
# patched source anchor is missing or repeated.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}
COPY patch_deepseek_v4_effort.py /opt/kairyu/patch_deepseek_v4_effort.py
RUN python3 /opt/kairyu/patch_deepseek_v4_effort.py
