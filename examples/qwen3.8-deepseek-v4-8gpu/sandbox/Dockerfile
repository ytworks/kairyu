# Sandbox execution runner (ECO-D1). Build has network access; the running
# container has no network namespace interfaces and communicates only through
# a shared Unix socket, with a read-only rootfs, noexec tmpfs workdirs, and no
# capabilities (compose.yaml).
FROM python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

RUN pip install --no-cache-dir pytest==9.1.1 \
    && install -d -o 65534 -g 65534 /run/kairyu-executor

COPY --chmod=0644 runner.py /opt/runner.py

USER 65534:65534

CMD ["python", "/opt/runner.py"]
