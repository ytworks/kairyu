# Kairyu Helm chart

This chart deploys one Kairyu gateway/replica service. The default values are a CPU-safe
smoke configuration backed by the mock engine:

```console
helm install kairyu deploy/helm/kairyu
```

The checked-in GPU overlay requests one NVIDIA GPU, selects the `pcie-gddr` node profile,
uses the `nvidia` RuntimeClass, mounts model files read-only, and starts the real `kairyu`
engine:

```console
helm install kairyu deploy/helm/kairyu \
  -f deploy/helm/kairyu/values-gpu.yaml
```

## GPU prerequisites

The cluster must have NVIDIA drivers and the NVIDIA GPU Operator (or equivalent device
plugin) installed. It must expose `nvidia.com/gpu`, provide a RuntimeClass named `nvidia`,
and label the target nodes with `kairyu.dev/gpu-profile=pcie-gddr` (or the selector must be
overridden for the cluster).

Ordinary CI only lints and renders the GPU manifest. It has no GPU node and does not run
the resulting pod.

## Model storage

Model storage is disabled by default. When enabled, configure exactly one source:

- `hostPath`: an absolute path already present on every eligible GPU node; or
- `pvcName`: the name of an existing PersistentVolumeClaim in the release namespace.

The GPU overlay uses a host path and expects the checkpoint directory to exist at
`/models/checkpoint` on the selected node. It mounts `/models` read-only at `/models` in
the container:

```yaml
modelStorage:
  enabled: true
  pvcName: ""
  hostPath: /models
  mountPath: /models
```

To use an existing PVC instead, keep the same directory layout inside the volume and
override the source without editing the Deployment template:

```console
helm install kairyu deploy/helm/kairyu \
  -f deploy/helm/kairyu/values-gpu.yaml \
  --set-string modelStorage.pvcName=kairyu-models \
  --set-string modelStorage.hostPath=
```

In both cases, the mounted storage must contain `/models/checkpoint` as seen from the
container because the GPU DeploymentSpec sets `model_path: /models/checkpoint`. Both
`hostPath` and `mountPath` must be absolute. The values schema rejects enabled storage
with no source, both sources at once, relative paths, and unknown storage fields.
