# syntax=docker/dockerfile:1.7
# Effort overlay for DeepSeek-V4-Flash-0731 on the pinned SM120 vLLM build.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

# Kairyu sends L3 max to vLLM's /v1/responses as reasoning.effort "xhigh"
# (m11 D4 amendment 2026-10-17); patch_effort.py makes the encoder render it
# as the max tier.
COPY patch_effort.py /opt/kairyu/patch_effort.py
RUN python3 /opt/kairyu/patch_effort.py
