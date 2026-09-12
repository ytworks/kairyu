# syntax=docker/dockerfile:1.7
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}
COPY patch_masked_kv.py /opt/kairyu/patch_masked_kv.py
RUN python3 /opt/kairyu/patch_masked_kv.py
# Isolate generated binaries from inherited/bind-mounted unpatched JIT caches.
# Reusing /root/.cache would allow a same-version stale kernel to shadow the fix.
ENV FLASHINFER_WORKSPACE_BASE=/opt/kairyu/flashinfer-masked-kv-v1
