# M19 Design: Deploy-Ready Packaging

Status: **Implemented** (2026-07-03; final milestone of the local-complete
plan — self-verified via the dry-run pin suite, no external review panel:
packaging follows the reviewed m7/m10a/m16-m18 decisions and the runbook).
Milestone: M19
Date: 2026-07-03

## Decisions

- **D1 `Dockerfile.cuda`**: nvidia/cuda 12.4 runtime + uv sync with
  `--extra gpu --extra hf --extra fleet`; same one-image-per-role model as
  the CPU Dockerfile (the mounted DeploymentSpec decides gateway/replica).
- **D2 GPU deploy configs**: `deploy/compose/docker-compose.gpu.yaml`
  (gateway + GPU replica, nvidia device reservation, model volume),
  `deploy/compose/gpu-replica.yaml`, `deploy/helm/kairyu/values-gpu.yaml`
  (nvidia.com/gpu limits, runtimeClass, per-profile nodeSelector for the
  pcie-gddr / nvlink-hbm pools from the roadmap hardware matrix).
- **D3 `scripts/gpu_gates/`**: runbook §0/§1/§2/§3/§6/§7/§9 + G4/G5 gates
  as executable scripts. EVERY script supports `--dry-run`;
  `tests/unit/test_gpu_gates_scripts.py` pins that (a) dry-run emits the
  command plan, (b) every referenced tests/scripts/bench/deploy path EXISTS
  today — deploy day cannot discover missing files.
- **D4 `[gpu]` extra**: flashinfer-python/triton/nixl with
  `sys_platform == 'linux'` markers — macOS `uv sync` ignores them.

## Acceptance (plan §final)

(a) macOS `uv sync` clean; (b) default suite green; (c) all gate scripts
dry-run valid plans; (d) Dockerfile.cuda builds (deploy day — needs the
CUDA base image); (e) PROGRESS.md Current Status reflects deploy-ready.

## Amendments

### 2026-07-13 — D2/D3: GPU Helm rendering is a mandatory CI gate

- **D2 amendment:** the chart's CPU defaults and checked-in GPU overlay are both
  schema-linted and template-rendered before the kind cluster is created. The GPU
  render validates placement, NVIDIA RuntimeClass/resource limits, read-only model
  storage, and the real `kairyu` backend without requiring a GPU device in CI.
- **D3 amendment:** `scripts/kind_smoke.sh` is the single source of truth for the
  four fail-fast Helm commands. Its default path runs them before the existing
  CPU kind install/HTTP drill, while `--helm-check` runs only the same lint/render
  gate for an explicit CI step; the workflow does not duplicate Helm semantics.
- **Verification boundary:** ordinary CI renders and schema-validates the GPU pod
  but does not schedule or execute it. GPU execution remains a hardware gate.
