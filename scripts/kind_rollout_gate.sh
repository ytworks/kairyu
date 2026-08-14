#!/usr/bin/env bash
# G5 F1b: clean-head build -> kind -> drain-first partitioned rolling rollout.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/kind_rollout_gate.sh [--formal|--smoke] [--keep-cluster]

Environment overrides:
  KIND, KUBECTL, DOCKER, CURL, UV, GIT, TAR, TIMEOUT, GREP, SED, JQ
  F1B_CLUSTER_NAME, F1B_KIND_CONFIG, F1B_MANIFEST_DIR
  F1B_RESULTS_DIR, F1B_GATEWAY_PORT

The formal profile rolls 100 replicas from partition 100. The smoke profile
uses the same protocol with four replicas from partition 4.
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
GREP=${GREP:-grep}
SED=${SED:-sed}
JQ=${JQ:-jq}

run_bounded() {
  local duration=$1
  shift
  "$TIMEOUT" --signal=TERM --kill-after=5s "$duration" "$@"
}

CLUSTER_NAME=${F1B_CLUSTER_NAME:-kairyu-f1b}
KIND_CONFIG=${F1B_KIND_CONFIG:-deploy/kind/f1a/kind-config.yaml}
MANIFEST_DIR=${F1B_MANIFEST_DIR:-deploy/kind/f1b}
RESULTS_DIR=${F1B_RESULTS_DIR:-bench/results/f1b-zero-failure-rollout}
GATEWAY_PORT=${F1B_GATEWAY_PORT:-18081}
APPLY_DIR="${MANIFEST_DIR}/overlays/${PROFILE}"
REPLICAS=100
if [[ "$PROFILE" == "smoke" ]]; then
  REPLICAS=4
fi

GATEWAY_IMAGE=kairyu:dev
MOCK_IMAGE=kairyu-f1a-mock:dev
GATEWAY_CONTAINERD_IMAGE=docker.io/library/kairyu:dev
MOCK_CONTAINERD_IMAGE=docker.io/library/kairyu-f1a-mock:dev
GATEWAY_CRI_IMAGE_METADATA="${RESULTS_DIR}/gateway-cri-image.json"
MOCK_CRI_IMAGE_METADATA="${RESULTS_DIR}/mock-cri-image.json"
GATEWAY_CONTAINERD_MANIFEST="${RESULTS_DIR}/gateway-containerd-manifest.json"
MOCK_CONTAINERD_MANIFEST="${RESULTS_DIR}/mock-containerd-manifest.json"
RENDERED_KIND_CONFIG="${RESULTS_DIR}/rendered-kind-config.yaml"
RENDERED_MANIFEST="${RESULTS_DIR}/rendered-manifest.yaml"
OUTPUT="${RESULTS_DIR}/manifest.json"
VERIFIER_OUTPUT="${RESULTS_DIR}/verifier.json"

for tool in \
  "$KIND" "$KUBECTL" "$DOCKER" "$CURL" "$UV" "$GIT" "$TAR" "$TIMEOUT" \
  "$GREP" "$SED" "$JQ"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool is unavailable: $tool" >&2
    exit 1
  fi
done
if [[ ! -f "$KIND_CONFIG" || ! -f "${APPLY_DIR}/kustomization.yaml" ]]; then
  echo "F1b kind config or ${PROFILE} overlay is missing" >&2
  exit 1
fi
if [[ ! "$CLUSTER_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "F1B_CLUSTER_NAME must be a valid DNS label" >&2
  exit 1
fi
if [[ ! "$GATEWAY_PORT" =~ ^[0-9]+$ || ${#GATEWAY_PORT} -gt 5 ]]; then
  echo "F1B_GATEWAY_PORT must be an integer from 1 through 65535" >&2
  exit 1
fi
GATEWAY_PORT=$((10#$GATEWAY_PORT))
if ((GATEWAY_PORT < 1 || GATEWAY_PORT > 65535)); then
  echo "F1B_GATEWAY_PORT must be an integer from 1 through 65535" >&2
  exit 1
fi
GATEWAY_URL="http://127.0.0.1:${GATEWAY_PORT}"

mkdir -p "$RESULTS_DIR"
known_results=(
  manifest.json
  verifier.json
  requests.jsonl
  placements.jsonl
  rollout.jsonl
  endpointslices.jsonl
  pods.jsonl
  gateway.log
  gateway-placements.jsonl
  gateway-placements-copy-error.txt
  kubernetes-final.txt
  events.txt
  live-applied.yaml
  pods-final.json
  nodes-describe.txt
  runtime-images.txt
  gateway-membership-before-traffic.prom
  gateway-cri-image.json
  mock-cri-image.json
  gateway-containerd-manifest.json
  mock-containerd-manifest.json
  rendered-kind-config.yaml
  rendered-manifest.yaml
)
for result_name in "${known_results[@]}"; do
  rm -f -- "${RESULTS_DIR}/${result_name}"
done

CLUSTER_CREATED=0
CLUSTER_MAY_EXIST=0
SOURCE_ARCHIVE_DIR=

collect_evidence() {
  if ((CLUSTER_CREATED == 0)); then
    return
  fi
  run_bounded 20s "$KUBECTL" -n kairyu-f1a \
    get pods,endpointslices,statefulsets -o wide \
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
  collect_evidence
  if ((CLUSTER_MAY_EXIST == 1 && (CLUSTER_CREATED == 0 || KEEP_CLUSTER == 0))); then
    if ! run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"; then
      echo "failed to delete kind cluster ${CLUSTER_NAME}" >&2
      cleanup_failed=1
    fi
  fi
  if [[ -n "$SOURCE_ARCHIVE_DIR" && -d "$SOURCE_ARCHIVE_DIR" ]]; then
    case "$SOURCE_ARCHIVE_DIR" in
      /tmp/kairyu-f1b-source.*)
        if ! rm -rf -- "$SOURCE_ARCHIVE_DIR"; then
          echo "failed to remove source archive ${SOURCE_ARCHIVE_DIR}" >&2
          cleanup_failed=1
        fi
        ;;
      *)
        echo "refusing to remove unexpected source path ${SOURCE_ARCHIVE_DIR}" >&2
        cleanup_failed=1
        ;;
    esac
  fi
  if ((status == 0 && cleanup_failed != 0)); then
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

SOURCE_COMMIT=$("$GIT" rev-parse HEAD)
source_inputs=(
  .dockerignore
  Dockerfile
  README.md
  pyproject.toml
  uv.lock
  kairyu
  verification/fleet/resilience/fleet_churn_bench.py
  verification/fleet/resilience/fleet_rollout_bench.py
  scripts/kind_rollout_gate.sh
  deploy/kind/f1a
  deploy/kind/f1b
  .github/workflows/f1b-rollout.yml
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
    echo "F1b gate requires one unchanged clean source commit" >&2
    printf '%s\n' "$tracked_status" "$scoped_status" >&2
    exit 1
  fi
}
assert_clean_source

SOURCE_ARCHIVE_DIR=$(mktemp -d /tmp/kairyu-f1b-source.XXXXXX)
"$GIT" archive --format=tar "$SOURCE_COMMIT" |
  "$TAR" -xf - -C "$SOURCE_ARCHIVE_DIR"
ARCHIVE_KIND_CONFIG="${SOURCE_ARCHIVE_DIR}/${KIND_CONFIG}"
ARCHIVE_APPLY_DIR="${SOURCE_ARCHIVE_DIR}/${APPLY_DIR}"
ARCHIVE_MOCK_DIR="${SOURCE_ARCHIVE_DIR}/deploy/kind/f1a/mock"
GATEWAY_IMAGE_ARCHIVE="${SOURCE_ARCHIVE_DIR}/gateway-image.tar"
MOCK_IMAGE_ARCHIVE="${SOURCE_ARCHIVE_DIR}/mock-image.tar"
if [[ ! -f "$ARCHIVE_KIND_CONFIG" ||
      ! -f "${ARCHIVE_APPLY_DIR}/kustomization.yaml" ||
      ! -f "${ARCHIVE_MOCK_DIR}/Dockerfile" ]]; then
  echo "committed F1b kind inputs are incomplete" >&2
  exit 1
fi

host_port_count=$(
  "$GREP" -cE '^[[:space:]]+hostPort:[[:space:]]+' \
    "$ARCHIVE_KIND_CONFIG" || true
)
pinned_host_port_count=$(
  "$GREP" -cE '^[[:space:]]+hostPort:[[:space:]]+18080[[:space:]]*$' \
    "$ARCHIVE_KIND_CONFIG" || true
)
pinned_name_count=$(
  "$GREP" -cE '^name:[[:space:]]+kairyu-f1a[[:space:]]*$' \
    "$ARCHIVE_KIND_CONFIG" || true
)
if [[ "$host_port_count" != "1" ||
      "$pinned_host_port_count" != "1" ||
      "$pinned_name_count" != "1" ]]; then
  echo "committed kind config must pin one cluster name and hostPort 18080" >&2
  exit 1
fi
"$SED" -E \
  -e "s/^name:[[:space:]]+kairyu-f1a[[:space:]]*$/name: ${CLUSTER_NAME}/" \
  -e "s/^([[:space:]]+)hostPort:[[:space:]]+18080[[:space:]]*$/\\1hostPort: ${GATEWAY_PORT}/" \
  "$ARCHIVE_KIND_CONFIG" >"$RENDERED_KIND_CONFIG"
if [[ $("$GREP" -cE "^name:[[:space:]]+${CLUSTER_NAME}[[:space:]]*$" \
        "$RENDERED_KIND_CONFIG" || true) != "1" ||
      $("$GREP" -cE \
        "^[[:space:]]+hostPort:[[:space:]]+${GATEWAY_PORT}[[:space:]]*$" \
        "$RENDERED_KIND_CONFIG" || true) != "1" ]]; then
  echo "failed to render the F1b kind cluster name and gateway port" >&2
  exit 1
fi
"$KUBECTL" kustomize "$ARCHIVE_APPLY_DIR" >"$RENDERED_MANIFEST"

"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$GATEWAY_IMAGE" "$SOURCE_ARCHIVE_DIR"
# Freeze each tag before kind startup; ``load docker-image`` would resolve and
# export the mutable host tag only after the cluster has been created.
"$DOCKER" image save --output "$GATEWAY_IMAGE_ARCHIVE" "$GATEWAY_IMAGE"
"$DOCKER" build --ulimit nofile=65536:65536 \
  --provenance=false \
  --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" \
  -t "$MOCK_IMAGE" "$ARCHIVE_MOCK_DIR"
"$DOCKER" image save --output "$MOCK_IMAGE_ARCHIVE" "$MOCK_IMAGE"
assert_clean_source

GATEWAY_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$GATEWAY_IMAGE"
)
MOCK_IMAGE_DIGEST=$(
  "$DOCKER" image inspect --format '{{.Id}}' "$MOCK_IMAGE"
)
docker_image_config_digest() {
  local image_reference=$1
  "$DOCKER" image inspect "$image_reference" |
    "$JQ" -er '.[0] | .Descriptor.annotations["config.digest"] // .Id'
}
GATEWAY_IMAGE_CONFIG_DIGEST=$(docker_image_config_digest "$GATEWAY_IMAGE")
MOCK_IMAGE_CONFIG_DIGEST=$(docker_image_config_digest "$MOCK_IMAGE")
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

mapfile -t existing_clusters < <("$KIND" get clusters)
for existing in "${existing_clusters[@]}"; do
  if [[ "$existing" == "$CLUSTER_NAME" ]]; then
    run_bounded 120s "$KIND" delete cluster --name "$CLUSTER_NAME"
    break
  fi
done

CLUSTER_MAY_EXIST=1
"$KIND" create cluster --name "$CLUSTER_NAME" \
  --config "$RENDERED_KIND_CONFIG" --wait 180s
CLUSTER_CREATED=1
mapfile -t kind_nodes < <("$KIND" get nodes --name "$CLUSTER_NAME")
if ((${#kind_nodes[@]} != 1)); then
  echo "expected one kind node, found ${#kind_nodes[@]}" >&2
  exit 1
fi
CONTROL_PLANE=${kind_nodes[0]}
if ! run_bounded 120s \
  "$KIND" load image-archive "$GATEWAY_IMAGE_ARCHIVE" "$MOCK_IMAGE_ARCHIVE" \
  --name "$CLUSTER_NAME"; then
  echo "retrying kind image archive import" >&2
  run_bounded 120s \
    "$KIND" load image-archive "$GATEWAY_IMAGE_ARCHIVE" "$MOCK_IMAGE_ARCHIVE" \
    --name "$CLUSTER_NAME"
fi

run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  crictl inspecti --output json "$GATEWAY_IMAGE" \
  >"$GATEWAY_CRI_IMAGE_METADATA"
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  crictl inspecti --output json "$MOCK_IMAGE" \
  >"$MOCK_CRI_IMAGE_METADATA"
containerd_target_digest() {
  local image_reference=$1
  run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" ctr -n k8s.io images ls |
    awk -v ref="$image_reference" '
      $1 == ref {
        print $3
        found++
      }
      END {
        if (found != 1) {
          exit 1
        }
      }
    '
}
GATEWAY_CONTAINERD_IMAGE_DIGEST=$(
  containerd_target_digest "$GATEWAY_CONTAINERD_IMAGE"
)
MOCK_CONTAINERD_IMAGE_DIGEST=$(
  containerd_target_digest "$MOCK_CONTAINERD_IMAGE"
)
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  ctr -n k8s.io content get "$GATEWAY_CONTAINERD_IMAGE_DIGEST" \
  >"$GATEWAY_CONTAINERD_MANIFEST"
run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
  ctr -n k8s.io content get "$MOCK_CONTAINERD_IMAGE_DIGEST" \
  >"$MOCK_CONTAINERD_MANIFEST"
containerd_config_digest() {
  local target_digest=$1
  run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" sh -ec \
    'ctr -n k8s.io content get "$1" | jq -er ".config.digest"' \
    sh "$target_digest"
}
GATEWAY_CONTAINERD_CONFIG_DIGEST=$(
  containerd_config_digest "$GATEWAY_CONTAINERD_IMAGE_DIGEST"
)
MOCK_CONTAINERD_CONFIG_DIGEST=$(
  containerd_config_digest "$MOCK_CONTAINERD_IMAGE_DIGEST"
)
cri_status_digest() {
  local image_reference=$1
  run_bounded 20s "$DOCKER" exec "$CONTROL_PLANE" \
    crictl inspecti --output go-template --template '{{.status.id}}' \
    "$image_reference"
}
GATEWAY_CRI_STATUS_DIGEST=$(cri_status_digest "$GATEWAY_IMAGE")
MOCK_CRI_STATUS_DIGEST=$(cri_status_digest "$MOCK_IMAGE")
is_sha256_digest() {
  [[ "$1" =~ ^sha256:[0-9a-f]{64}$ ]]
}
for digest in \
  "$GATEWAY_IMAGE_DIGEST" "$MOCK_IMAGE_DIGEST" \
  "$GATEWAY_IMAGE_CONFIG_DIGEST" "$MOCK_IMAGE_CONFIG_DIGEST" \
  "$GATEWAY_CONTAINERD_IMAGE_DIGEST" "$MOCK_CONTAINERD_IMAGE_DIGEST" \
  "$GATEWAY_CONTAINERD_CONFIG_DIGEST" "$MOCK_CONTAINERD_CONFIG_DIGEST" \
  "$GATEWAY_CRI_STATUS_DIGEST" "$MOCK_CRI_STATUS_DIGEST"; do
  if ! is_sha256_digest "$digest"; then
    echo "invalid image provenance digest: ${digest}" >&2
    exit 1
  fi
done
if [[ "$GATEWAY_IMAGE_CONFIG_DIGEST" != \
        "$GATEWAY_CONTAINERD_CONFIG_DIGEST" ||
      "$GATEWAY_IMAGE_CONFIG_DIGEST" != "$GATEWAY_CRI_STATUS_DIGEST" ||
      "$MOCK_IMAGE_CONFIG_DIGEST" != \
        "$MOCK_CONTAINERD_CONFIG_DIGEST" ||
      "$MOCK_IMAGE_CONFIG_DIGEST" != "$MOCK_CRI_STATUS_DIGEST" ]]; then
  echo "kind image config does not match built Docker image and CRI" >&2
  exit 1
fi
if [[ "$GATEWAY_IMAGE_DIGEST" != "$GATEWAY_IMAGE_CONFIG_DIGEST" &&
      "$GATEWAY_IMAGE_DIGEST" != "$GATEWAY_CONTAINERD_IMAGE_DIGEST" ]] ||
   [[ "$MOCK_IMAGE_DIGEST" != "$MOCK_IMAGE_CONFIG_DIGEST" &&
      "$MOCK_IMAGE_DIGEST" != "$MOCK_CONTAINERD_IMAGE_DIGEST" ]]; then
  echo "Docker image ID is neither the config nor containerd target" >&2
  exit 1
fi

"$KUBECTL" apply -k "$ARCHIVE_APPLY_DIR"
live_replicas=$(
  "$KUBECTL" -n kairyu-f1a get statefulset f1a-replica \
    -o jsonpath='{.spec.replicas}'
)
live_strategy=$(
  "$KUBECTL" -n kairyu-f1a get statefulset f1a-replica \
    -o jsonpath='{.spec.updateStrategy.type}'
)
live_partition=$(
  "$KUBECTL" -n kairyu-f1a get statefulset f1a-replica \
    -o jsonpath='{.spec.updateStrategy.rollingUpdate.partition}'
)
if [[ "$live_replicas" != "$REPLICAS" ||
      "$live_strategy" != "RollingUpdate" ||
      "$live_partition" != "$REPLICAS" ]]; then
  echo "unexpected F1b StatefulSet contract: replicas=${live_replicas}, " \
    "strategy=${live_strategy}, partition=${live_partition}" >&2
  exit 1
fi

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
"$KUBECTL" -n kairyu-f1a rollout status deployment/f1a-gateway \
  --timeout=5m

ready=0
for _ in $(seq 1 120); do
  if "$CURL" -sf "${GATEWAY_URL}/readyz" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if ((ready == 0)); then
  echo "gateway NodePort did not become ready at ${GATEWAY_URL}" >&2
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
  sleep 1
done
if ((membership_ready == 0)); then
  echo "gateway membership did not converge: configured=${configured:-missing} " \
    "healthy=${healthy:-missing} draining=${draining:-missing}; " \
    "expected ${REPLICAS}/${REPLICAS}/0" >&2
  exit 1
fi

KIND_NODE_IMAGE=$(
  "$DOCKER" inspect --format '{{.Config.Image}}' "$CONTROL_PLANE"
)
KIND_VERSION=$("$KIND" version)
KUBECTL_VERSION=$("$KUBECTL" version --client=true --output=yaml)
DOCKER_VERSION=$("$DOCKER" version --format '{{.Server.Version}}')

frozen_args=()
while IFS= read -r input; do
  frozen_args+=(--frozen-input "$input")
done < <(
  "$GIT" ls-files -- \
    verification/fleet/resilience/fleet_churn_bench.py \
    verification/fleet/resilience/fleet_rollout_bench.py \
    scripts/kind_rollout_gate.sh \
    .github/workflows/f1b-rollout.yml \
    deploy/kind/f1a \
    "$MANIFEST_DIR"
)
if ((${#frozen_args[@]} == 0)); then
  echo "no tracked F1b manifest inputs found for provenance" >&2
  exit 1
fi

ulimit -n 65536 2>/dev/null || true
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/kairyu-f1b-uv-cache}
"$UV" run --frozen python verification/fleet/resilience/fleet_rollout_bench.py \
  --profile "$PROFILE" \
  --gateway-url "$GATEWAY_URL" \
  --kubectl "$KUBECTL" \
  --gateway-image-digest "$GATEWAY_IMAGE_DIGEST" \
  --mock-image-digest "$MOCK_IMAGE_DIGEST" \
  --gateway-containerd-image-digest "$GATEWAY_CONTAINERD_IMAGE_DIGEST" \
  --mock-containerd-image-digest "$MOCK_CONTAINERD_IMAGE_DIGEST" \
  --gateway-containerd-config-digest "$GATEWAY_CONTAINERD_CONFIG_DIGEST" \
  --mock-containerd-config-digest "$MOCK_CONTAINERD_CONFIG_DIGEST" \
  --gateway-containerd-image-metadata "$GATEWAY_CONTAINERD_MANIFEST" \
  --mock-containerd-image-metadata "$MOCK_CONTAINERD_MANIFEST" \
  --gateway-cri-image-metadata "$GATEWAY_CRI_IMAGE_METADATA" \
  --mock-cri-image-metadata "$MOCK_CRI_IMAGE_METADATA" \
  --kind-node-image "$KIND_NODE_IMAGE" \
  --kind-version "$KIND_VERSION" \
  --kubectl-version "$KUBECTL_VERSION" \
  --docker-version "$DOCKER_VERSION" \
  --expected-git-commit "$SOURCE_COMMIT" \
  --driver-path scripts/kind_rollout_gate.sh \
  "${frozen_args[@]}" \
  --frozen-sidecar "$RENDERED_KIND_CONFIG" \
  --frozen-sidecar "$RENDERED_MANIFEST" \
  --gateway-image-config-digest "$GATEWAY_IMAGE_CONFIG_DIGEST" \
  --mock-image-config-digest "$MOCK_IMAGE_CONFIG_DIGEST" \
  --output "$OUTPUT" \
  --assert-gate

"$UV" run --frozen python verification/fleet/resilience/fleet_rollout_bench.py \
  --verify-artifact "$OUTPUT" \
  --assert-gate >"$VERIFIER_OUTPUT"
echo "F1b ${PROFILE} zero-failure rollout gate passed: ${OUTPUT}"
