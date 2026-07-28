#!/usr/bin/env bash
# G5 F1a: build -> one-node kind -> 200-replica open-loop churn gate.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/kind_churn_gate.sh [--formal|--smoke] [--keep-cluster]

Environment overrides:
  KIND, KUBECTL, DOCKER, CURL, UV, GIT, TAR, TIMEOUT
  F1A_CLUSTER_NAME, F1A_KIND_CONFIG, F1A_MANIFEST_DIR
  F1A_RESULTS_DIR, F1A_GATEWAY_PORT

The formal profile is frozen in bench/fleet_churn_bench.py.  --smoke only
shortens the same protocol to a four-replica kustomize overlay and two epochs.
EOF
}

PROFILE=formal
KEEP_CLUSTER=0
while (($#)); do
  case "$1" in
    --formal)
      PROFILE=formal
      ;;
    --smoke)
      PROFILE=smoke
      ;;
    --keep-cluster)
      KEEP_CLUSTER=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "$REPO_ROOT"

KIND=${KIND:-kind}
KUBECTL=${KUBECTL:-kubectl}
DOCKER=${DOCKER:-docker}
CURL=${CURL:-curl}
UV=${UV:-uv}
GIT=${GIT:-git}
TAR=${TAR:-tar}
TIMEOUT=${TIMEOUT:-timeout}

run_bounded() {
  local duration=$1
  shift
  "$TIMEOUT" --signal=TERM --kill-after=5s "$duration" "$@"
}

CLUSTER_NAME=${F1A_CLUSTER_NAME:-kairyu-f1a}
GATEWAY_IMAGE=kairyu:dev
MOCK_IMAGE=kairyu-f1a-mock:dev
KIND_CONFIG=${F1A_KIND_CONFIG:-deploy/kind/f1a/kind-config.yaml}
MANIFEST_DIR=${F1A_MANIFEST_DIR:-deploy/kind/f1a}
RESULTS_DIR=${F1A_RESULTS_DIR:-bench/results/f1a-fleet-churn}
GATEWAY_PORT=${F1A_GATEWAY_PORT:-18080}
GATEWAY_URL="http://127.0.0.1:${GATEWAY_PORT}"
OUTPUT="${RESULTS_DIR}/manifest.json"
VERIFIER_OUTPUT="${RESULTS_DIR}/verifier.json"
PORT_FORWARD_LOG="${RESULTS_DIR}/port-forward.log"
REPLICAS=200
APPLY_DIR=$MANIFEST_DIR
if [[ "$PROFILE" == "smoke" ]]; then
  REPLICAS=4
  APPLY_DIR="${MANIFEST_DIR}/overlays/smoke"
fi

mkdir -p "$RESULTS_DIR"
known_results=(
  manifest.json
  verifier.json
  requests.jsonl
  placements.jsonl
  churn.jsonl
  endpointslices.jsonl
  pods.jsonl
  resources.jsonl
  port-forward.log
  kubernetes-final.txt
  events.txt
  gateway.log
  gateway-placements.jsonl
  gateway-placements-copy-error.txt
  rendered-manifest.yaml
  live-applied.yaml
  pods-final.json
  nodes-describe.txt
  runtime-images.txt
  gateway-membership-before-traffic.prom
)
for result_name in "${known_results[@]}"; do
  rm -f -- "${RESULTS_DIR}/${result_name}"
done

PORT_FORWARD_PID=
CLUSTER_CREATED=0
CLUSTER_MAY_EXIST=0
SOURCE_ARCHIVE_DIR=

collect_evidence() {
  if ((CLUSTER_CREATED == 0)); then
    return
  fi
  run_bounded 20s "$KUBECTL" -n kairyu-f1a \
    get pods,endpointslices -o wide \
    >"${RESULTS_DIR}/kubernetes-final.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1a \
    get events --sort-by=.lastTimestamp \
    >"${RESULTS_DIR}/events.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1a get \
    serviceaccounts,roles,rolebindings,configmaps,services,statefulsets,deployments \
    -o yaml >"${RESULTS_DIR}/live-applied.yaml" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1a get pods -o json \
    >"${RESULTS_DIR}/pods-final.json" 2>&1 || true
  run_bounded 20s "$KUBECTL" describe nodes \
    >"${RESULTS_DIR}/nodes-describe.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1a get pods -o jsonpath='
{range .items[*]}{.metadata.name}{"\t"}{range .spec.containers[*]}{.name}{"="}{.image}{"\t"}{end}{range .status.containerStatuses[*]}{.name}{"="}{.imageID}{"\t"}{end}{"\n"}{end}' \
    >"${RESULTS_DIR}/runtime-images.txt" 2>&1 || true
  run_bounded 20s "$KUBECTL" -n kairyu-f1a \
    logs deployment/f1a-gateway \
    >"${RESULTS_DIR}/gateway.log" 2>&1 || true
  local gateway_pod
  gateway_pod=$(
    run_bounded 20s "$KUBECTL" -n kairyu-f1a get pods \
      -l app.kubernetes.io/name=f1a-gateway \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
  ) || true
  if [[ -n "$gateway_pod" ]]; then
    run_bounded 20s "$KUBECTL" -n kairyu-f1a \
      exec "$gateway_pod" -- \
      python -c \
      'from pathlib import Path; import sys; sys.stdout.write(Path(sys.argv[1]).read_text())' \
      /evidence/placements.jsonl \
      >"${RESULTS_DIR}/gateway-placements.jsonl" \
      2>"${RESULTS_DIR}/gateway-placements-copy-error.txt" || true
  fi
}

cleanup() {
  local status=$?
  local cleanup_failed=0
  set +e
  if [[ -n "$PORT_FORWARD_PID" ]]; then
    kill "$PORT_FORWARD_PID" 2>/dev/null
    for _ in $(seq 1 50); do
      if ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
      kill -KILL "$PORT_FORWARD_PID" 2>/dev/null
    fi
    for _ in $(seq 1 50); do
      if ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
      wait "$PORT_FORWARD_PID" 2>/dev/null
    else
      echo "port-forward did not exit after SIGKILL; skipping wait" >&2
      cleanup_failed=1
    fi
  fi
  collect_evidence
  if ((CLUSTER_MAY_EXIST == 1 && (CLUSTER_CREATED == 0 || KEEP_CLUSTER == 0))); then
    if ! run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"; then
      echo "failed to delete kind cluster ${CLUSTER_NAME}" >&2
      cleanup_failed=1
    fi
  fi
  if [[ -n "$SOURCE_ARCHIVE_DIR" && -d "$SOURCE_ARCHIVE_DIR" ]]; then
    if ! rm -rf -- "$SOURCE_ARCHIVE_DIR"; then
      echo "failed to remove source archive ${SOURCE_ARCHIVE_DIR}" >&2
      cleanup_failed=1
    fi
  fi
  if ((status == 0 && cleanup_failed != 0)); then
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

for tool in \
  "$KIND" "$KUBECTL" "$DOCKER" "$CURL" "$UV" "$GIT" "$TAR" "$TIMEOUT"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool is unavailable: $tool" >&2
    exit 1
  fi
done
if [[ ! -f "$KIND_CONFIG" || ! -f "${APPLY_DIR}/kustomization.yaml" ]]; then
  echo "F1a kind config or kustomization is missing" >&2
  exit 1
fi

SOURCE_COMMIT=$("$GIT" rev-parse HEAD)
source_inputs=(
  .dockerignore
  Dockerfile
  README.md
  pyproject.toml
  uv.lock
  kairyu
  bench/fleet_churn_bench.py
  scripts/kind_churn_gate.sh
  deploy/kind/f1a
  .github/workflows/f1a-churn.yml
)
assert_clean_source() {
  local current_commit tracked_status scoped_status
  current_commit=$("$GIT" rev-parse HEAD)
  tracked_status=$("$GIT" status --porcelain --untracked-files=no)
  scoped_status=$(
    "$GIT" status --porcelain --untracked-files=all -- "${source_inputs[@]}"
  )
  if [[ "$current_commit" != "$SOURCE_COMMIT" ||
        -n "$tracked_status" ||
        -n "$scoped_status" ]]; then
    echo "F1a gate requires one unchanged clean source commit" >&2
    printf '%s\n' "$tracked_status" "$scoped_status" >&2
    exit 1
  fi
}
assert_clean_source

# Build and apply only the committed tree. Concurrent worktree edits therefore
# cannot change the gateway image or manifests after the clean-source check.
SOURCE_ARCHIVE_DIR=$(mktemp -d /tmp/kairyu-f1a-source.XXXXXX)
"$GIT" archive --format=tar "$SOURCE_COMMIT" \
  | "$TAR" -xf - -C "$SOURCE_ARCHIVE_DIR"
ARCHIVE_KIND_CONFIG="${SOURCE_ARCHIVE_DIR}/${KIND_CONFIG}"
ARCHIVE_MANIFEST_DIR="${SOURCE_ARCHIVE_DIR}/${MANIFEST_DIR}"
ARCHIVE_APPLY_DIR="${SOURCE_ARCHIVE_DIR}/${APPLY_DIR}"
if [[ ! -f "$ARCHIVE_KIND_CONFIG" ||
      ! -f "${ARCHIVE_APPLY_DIR}/kustomization.yaml" ]]; then
  echo "committed F1a kind config or kustomization is missing" >&2
  exit 1
fi
"$KUBECTL" kustomize "$ARCHIVE_APPLY_DIR" \
  >"${RESULTS_DIR}/rendered-manifest.yaml"

# Docker 29 starts build containers with nofile=1024 on the target host.  uv's
# bytecode compilation can exceed that, so both builds pin a sufficient limit.
"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$GATEWAY_IMAGE" "$SOURCE_ARCHIVE_DIR"
"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$MOCK_IMAGE" "${ARCHIVE_MANIFEST_DIR}/mock"
assert_clean_source

mapfile -t existing_clusters < <("$KIND" get clusters)
for existing in "${existing_clusters[@]}"; do
  if [[ "$existing" == "$CLUSTER_NAME" ]]; then
    run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"
    break
  fi
done

CLUSTER_MAY_EXIST=1
"$KIND" create cluster --name "$CLUSTER_NAME" \
  --config "$ARCHIVE_KIND_CONFIG" --wait 180s
CLUSTER_CREATED=1
"$KIND" load docker-image "$GATEWAY_IMAGE" "$MOCK_IMAGE" \
  --name "$CLUSTER_NAME"
"$KUBECTL" apply -k "$ARCHIVE_APPLY_DIR"

required_capacity=$((REPLICAS + 10))
node_capacity=$(
  "$KUBECTL" get nodes -o jsonpath='{.items[0].status.capacity.pods}'
)
if ((node_capacity < required_capacity)); then
  echo "kind node pod capacity ${node_capacity} < ${required_capacity}" >&2
  exit 1
fi

replica_set_ready=0
observed_replica_count=0
observed_ready_count=0
for _ in $(seq 1 600); do
  mapfile -t replica_states < <(
    "$KUBECTL" -n kairyu-f1a get pods \
      -l app.kubernetes.io/name=f1a-replica \
      -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
  )
  observed_replica_count=${#replica_states[@]}
  observed_ready_count=0
  for replica_state in "${replica_states[@]}"; do
    if [[ "$replica_state" == *=True ]]; then
      observed_ready_count=$((observed_ready_count + 1))
    fi
  done
  if ((observed_replica_count == REPLICAS && observed_ready_count == REPLICAS)); then
    replica_set_ready=1
    break
  fi
  sleep 1
done
if ((replica_set_ready == 0)); then
  echo "expected exactly ${REPLICAS} Ready replica pods; observed " \
    "${observed_ready_count}/${observed_replica_count}" >&2
  exit 1
fi
# The StatefulSet deliberately uses OnDelete, so readiness is the exact
# selected pod set above; revision equality is not part of this gate.
"$KUBECTL" -n kairyu-f1a rollout status deployment/f1a-gateway \
  --timeout=5m

"$KUBECTL" -n kairyu-f1a port-forward service/f1a-gateway \
  "${GATEWAY_PORT}:8000" >"$PORT_FORWARD_LOG" 2>&1 &
PORT_FORWARD_PID=$!
ready=0
for _ in $(seq 1 120); do
  if "$CURL" -sf "${GATEWAY_URL}/readyz" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if ((ready == 0)); then
  echo "gateway port-forward did not become ready" >&2
  exit 1
fi

membership_value() {
  local state=$1
  awk -v key="kairyu_pool_replicas{pool=\"f1a\",state=\"${state}\"}" '
    $1 == key {
      printf "%.0f\n", $2
      found = 1
    }
    END {
      if (!found) {
        exit 1
      }
    }
  '
}

membership_ready=0
configured=
healthy=
draining=
for _ in $(seq 1 300); do
  if membership_metrics=$("$CURL" -sf "${GATEWAY_URL}/metrics"); then
    configured=$(
      printf '%s\n' "$membership_metrics" | membership_value configured
    ) || configured=
    healthy=$(
      printf '%s\n' "$membership_metrics" | membership_value healthy
    ) || healthy=
    draining=$(
      printf '%s\n' "$membership_metrics" | membership_value draining
    ) || draining=
    if [[ "$configured" == "$REPLICAS" &&
          "$healthy" == "$REPLICAS" &&
          "$draining" == "0" ]]; then
      printf '%s\n' "$membership_metrics" \
        >"${RESULTS_DIR}/gateway-membership-before-traffic.prom"
      membership_ready=1
      break
    fi
  fi
  if ! kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if ((membership_ready == 0)); then
  echo "gateway membership did not converge: configured=${configured:-missing} " \
    "healthy=${healthy:-missing} draining=${draining:-missing}; " \
    "expected ${REPLICAS}/${REPLICAS}/0" >&2
  exit 1
fi

GATEWAY_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$GATEWAY_IMAGE"
)
MOCK_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$MOCK_IMAGE"
)
for image in "$GATEWAY_IMAGE" "$MOCK_IMAGE"; do
  image_revision=$(
    "$DOCKER" image inspect \
      --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
      "$image"
  )
  if [[ "$image_revision" != "$SOURCE_COMMIT" ]]; then
    echo "image ${image} revision ${image_revision} != ${SOURCE_COMMIT}" >&2
    exit 1
  fi
done
mapfile -t kind_nodes < <("$KIND" get nodes --name "$CLUSTER_NAME")
CONTROL_PLANE=${kind_nodes[0]}
KIND_NODE_IMAGE=$(
  "$DOCKER" inspect --format '{{.Config.Image}}' "$CONTROL_PLANE"
)
KIND_VERSION=$("$KIND" version)
KUBECTL_VERSION=$("$KUBECTL" version --client=true --output=yaml)
DOCKER_VERSION=$("$DOCKER" version --format '{{.Server.Version}}')

frozen_args=()
while IFS= read -r input; do
  frozen_args+=(--frozen-input "$input")
done < <("$GIT" ls-files -- "$MANIFEST_DIR" "$KIND_CONFIG")
if ((${#frozen_args[@]} == 0)); then
  echo "no tracked F1a manifests found for provenance" >&2
  exit 1
fi

ulimit -n 65536 2>/dev/null || true
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/kairyu-f1a-uv-cache}
"$UV" run python bench/fleet_churn_bench.py \
  --profile "$PROFILE" \
  --gateway-url "$GATEWAY_URL" \
  --kubectl "$KUBECTL" \
  --gateway-image-digest "$GATEWAY_IMAGE_DIGEST" \
  --mock-image-digest "$MOCK_IMAGE_DIGEST" \
  --kind-node-image "$KIND_NODE_IMAGE" \
  --kind-version "$KIND_VERSION" \
  --kubectl-version "$KUBECTL_VERSION" \
  --docker-version "$DOCKER_VERSION" \
  --expected-git-commit "$SOURCE_COMMIT" \
  --driver-path scripts/kind_churn_gate.sh \
  "${frozen_args[@]}" \
  --output "$OUTPUT" \
  --assert-gate

"$UV" run python bench/fleet_churn_bench.py \
  --verify-artifact "$OUTPUT" \
  --assert-gate >"$VERIFIER_OUTPUT"
echo "F1a ${PROFILE} churn gate passed: ${OUTPUT}"
