#!/usr/bin/env bash
# m10a D5 kind smoke: image -> cluster -> helm -> ready -> completion -> teardown
#
# Usage: kind_smoke.sh [--helm-check | --setup-only | --no-setup]
#   --helm-check  run only the Helm schema and GPU template gate
#   (none)        setup, then the cluster smoke
#   --setup-only  build the image into an archive and pull the node image (the
#                 networked setup; CI retries it)
#   --no-setup    run the cluster smoke from that archive and node image; never
#                 build or pull
# The image is loaded from its archive, so the load does not depend on the
# runner's Docker image store. Waits observe states or events; their bound
# only detects a hang.
set -euo pipefail

helm_gate() {
  helm lint deploy/helm/kairyu
  helm lint deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml
  helm template kairyu deploy/helm/kairyu >/dev/null
  helm template kairyu deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml >/dev/null
}

MODE=${1:-}
case "$MODE" in
  "" | --helm-check | --setup-only | --no-setup) ;;
  *) echo "usage: $0 [--helm-check | --setup-only | --no-setup]" >&2; exit 2 ;;
esac

if [[ "$MODE" != --setup-only ]]; then
  helm_gate
fi
if [[ "$MODE" == --helm-check ]]; then
  exit 0
fi

CLUSTER=${CLUSTER:-kairyu-smoke}
IMAGE=${IMAGE:-kairyu:dev}
# The default node image of kind v0.31.0 (KIND_VERSION in ci.yml).
NODE_IMAGE=kindest/node:v1.35.0@sha256:452d707d4862f52530247495d180205e029056831160e22870e37e3f6c1ac31f
WORK_DIR=${KIND_SMOKE_WORK_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/kairyu-kind-smoke}
IMAGE_ARCHIVE="$WORK_DIR/image.tar"
# Far above every expected wait; reaching it means a hang.
HANG_BOUND_S=600

if [[ "$MODE" != --no-setup ]]; then
  mkdir -p "$WORK_DIR"
  docker build --provenance=false -t "$IMAGE" .
  docker image save --output "$IMAGE_ARCHIVE" "$IMAGE"
  docker pull "$NODE_IMAGE"
fi
if [[ "$MODE" == --setup-only ]]; then
  exit 0
fi

PF=
PF_LOG=
cleanup() {
  if [[ -n "$PF" ]]; then
    kill "$PF" 2>/dev/null || true
  fi
  if [[ -n "$PF_LOG" ]]; then
    rm -f "$PF_LOG"
  fi
  if [[ -z "$MODE" ]]; then
    rm -f "$IMAGE_ARCHIVE" # this run's own setup; a split setup keeps it
  fi
  kind delete cluster --name "$CLUSTER"
}

kind create cluster --name "$CLUSTER" --image "$NODE_IMAGE" --wait "${HANG_BOUND_S}s"
trap cleanup EXIT
kind load image-archive "$IMAGE_ARCHIVE" --name "$CLUSTER"

helm install kairyu deploy/helm/kairyu \
  --set image.repository="${IMAGE%%:*}" --set image.tag="${IMAGE##*:}"
kubectl rollout status deployment/kairyu --timeout="${HANG_BOUND_S}s" || {
  echo "kind smoke HANG: deployment/kairyu did not finish its rollout within ${HANG_BOUND_S}s (a hang bound)" >&2
  exit 1
}

# kubectl names the chosen local port once the forward listens.
PF_LOG=$(mktemp "${TMPDIR:-/tmp}/kairyu-kind-smoke-port-forward.XXXXXX")
kubectl port-forward svc/kairyu :8000 >"$PF_LOG" 2>&1 &
PF=$!
port=
for _ in $(seq 1 "$HANG_BOUND_S"); do
  port=$(sed -nE 's/^Forwarding from 127\.0\.0\.1:([0-9]+) -> 8000$/\1/p' "$PF_LOG" | head -1)
  [[ -z "$port" ]] || break
  kill -0 "$PF" 2>/dev/null || { cat "$PF_LOG" >&2; echo "kind smoke: port-forward exited" >&2; exit 1; }
  sleep 1
done
[[ -n "$port" ]] || {
  echo "kind smoke HANG: port-forward did not listen within ${HANG_BOUND_S}s (a hang bound)" >&2
  exit 1
}
curl -sf "http://127.0.0.1:${port}/v1/models" | grep -q '"data"'
curl -sf "http://127.0.0.1:${port}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"hello","max_tokens":4}' | grep -q '"choices"'
echo "kind smoke OK"
