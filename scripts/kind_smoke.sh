#!/usr/bin/env bash
# m10a D5 kind smoke: cluster -> image -> helm -> ready -> completion -> teardown
set -euo pipefail

helm_gate() {
  helm lint deploy/helm/kairyu
  helm lint deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml
  helm template kairyu deploy/helm/kairyu >/dev/null
  helm template kairyu deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml >/dev/null
}

helm_gate
if [[ "${1:-}" == "--helm-check" ]]; then
  exit 0
fi

CLUSTER=${CLUSTER:-kairyu-smoke}
IMAGE=${IMAGE:-kairyu:dev}

kind create cluster --name "$CLUSTER" --wait 120s
trap 'kind delete cluster --name "$CLUSTER"' EXIT

docker build -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER"

helm install kairyu deploy/helm/kairyu \
  --set image.repository="${IMAGE%%:*}" --set image.tag="${IMAGE##*:}"
kubectl rollout status deployment/kairyu --timeout=180s

kubectl port-forward svc/kairyu 18080:8000 &
PF=$!
sleep 3
curl -sf http://127.0.0.1:18080/v1/models | grep -q '"data"'
curl -sf http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"hello","max_tokens":4}' | grep -q '"choices"'
kill $PF
echo "kind smoke OK"
