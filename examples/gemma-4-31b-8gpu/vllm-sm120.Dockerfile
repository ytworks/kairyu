# syntax=docker/dockerfile:1.7
# SM120 overlay for EmbeddingGemma 2 on RTX PRO 6000 GPUs: one edit to vLLM's
# Triton encoder prefill tiles (patch_sm120.py) on the pinned upstream release.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}
COPY patch_sm120.py /opt/kairyu/patch_sm120.py
RUN python3 /opt/kairyu/patch_sm120.py
