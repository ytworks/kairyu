# syntax=docker/dockerfile:1.7
# Tokenizer overlay for DeepSeek-V4-Flash-0731 in the Qwen + DeepSeek ensemble.
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

# The Python frontend renders chats with the tokenizer patched below.
ENV VLLM_USE_RUST_FRONTEND=0
COPY patch_tokenizer.py /opt/kairyu/patch_tokenizer.py
RUN python3 /opt/kairyu/patch_tokenizer.py
