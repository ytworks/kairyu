#!/usr/bin/env bash
# Configuration/lifecycle wrapper only; inference and orchestration stay in Compose/Kairyu.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
spec="$here/example.json"
action=${1:-status}
if (($#)); then shift; fi
die() { echo "$*" >&2; exit 1; }
for executable in docker jq curl sha256sum nvidia-smi; do
  command -v "$executable" >/dev/null || die "Required executable: $executable"
done

export NVME_STORAGE_ROOT=${NVME_STORAGE_ROOT:-$(jq -r '.storage.root' "$spec")}
NVME_STORAGE_ROOT=$(realpath -m -- "$NVME_STORAGE_ROOT")
case "$NVME_STORAGE_ROOT" in /mnt/nvme|/mnt/nvme/*) ;; *) die 'Storage must remain below /mnt/nvme';; esac
environment=$(jq -r '.environment' "$spec")
storage="$NVME_STORAGE_ROOT/model-volumes/$environment"
export COMPOSE_DISABLE_ENV_FILE=1
export COMPOSE_PROJECT_NAME=${environment//./-}
unset COMPOSE_FILE COMPOSE_PROFILES
export QWEN_MODEL_STORAGE_PATH="$NVME_STORAGE_ROOT/model-volumes/qwen3.8-27b-1gpu/models"
export DEEPSEEK_MODEL_STORAGE_PATH="$NVME_STORAGE_ROOT/model-volumes/$(jq -r '.storage.deepseek_model_environment' "$spec")/models"
export QWEN_CACHE_0_PATH="$storage/compile-cache/qwen-0"
export QWEN_CACHE_1_PATH="$storage/compile-cache/qwen-1"
export DEEPSEEK_CACHE_PATH="$storage/compile-cache/deepseek"
export WEBUI_STORAGE_PATH="$storage/webui-data"
export KAIRYU_ORCHESTRATOR_SPEC_PATH=${KAIRYU_ORCHESTRATOR_SPEC_PATH:-$here/auto-max.yaml}
export QWEN_VLLM_IMAGE=$(jq -r '.vllm.qwen.image' "$spec")
export DEEPSEEK_VLLM_IMAGE=$(jq -r '.vllm.deepseek.image' "$spec")
export OPEN_WEBUI_IMAGE=$(jq -r '.webui.image' "$spec")
export KAIRYU_IMAGE=${KAIRYU_IMAGE:-local/kairyu:v41-critical-ensemble}
export API_PORT=${API_PORT:-$(jq -r '.api_port' "$spec")}
export DEEPSEEK_L1_PORT=${DEEPSEEK_L1_PORT:-$(jq -r '.deepseek_l1_loopback_port' "$spec")}
export CHAT_UI_PORT=${CHAT_UI_PORT:-$(jq -r '.webui.port' "$spec")}
export API_BIND_ADDRESS=${API_BIND_ADDRESS:-0.0.0.0}
export CHAT_UI_BIND_ADDRESS=${CHAT_UI_BIND_ADDRESS:-0.0.0.0}
export WEBUI_URL=${WEBUI_URL:-http://localhost:$CHAT_UI_PORT}

# Derive affinity from the actual PCI devices, including for verification restarts.
declare -A gpu_cpus
gpu_count=0
while IFS=, read -r gpu name memory capability bus; do
  gpu=${gpu// /}; memory=${memory// /}; capability=${capability// /}; bus=${bus// /}
  name=$(printf '%s' "$name" | sed 's/^ *//;s/ *$//')
  [[ "$name" == "$(jq -r '.hardware.product' "$spec")" ]] || die "GPU $gpu has an unexpected model"
  jq -en --argjson m "$memory" --argjson c "$capability" --slurpfile s "$spec" \
    '$m >= $s[0].hardware.minimum_vram_mib and $c >= $s[0].hardware.minimum_compute_capability' >/dev/null \
    || die "GPU $gpu does not meet the hardware minimum"
  bus=${bus,,}; bus=${bus/#00000000:/0000:}
  read -r node < "/sys/bus/pci/devices/$bus/numa_node"
  ((node >= 0)) || die "GPU $gpu has no NUMA node"
  read -r cpus < "/sys/devices/system/node/node$node/cpulist"
  gpu_cpus[$gpu]=$cpus
  ((gpu_count+=1))
done < <(nvidia-smi --query-gpu=index,name,memory.total,compute_cap,pci.bus_id --format=csv,noheader,nounits)
[[ $gpu_count == "$(jq -r '.hardware.gpu_count' "$spec")" ]] || die 'Expected eight GPUs'
export QWEN_0_CPUSET=${gpu_cpus[$(jq -r '.allocation.tier1.gpu_ids[0]' "$spec")]}
export QWEN_1_CPUSET=${gpu_cpus[$(jq -r '.allocation.tier1.gpu_ids[1]' "$spec")]}
DEEPSEEK_CPUSET=$(while read -r gpu; do printf '%s\n' "${gpu_cpus[$gpu]}"; done \
  < <(jq -r '.allocation.tier2.gpu_ids[]' "$spec") | sort -u | paste -sd,)
export DEEPSEEK_CPUSET

compose() { docker compose --project-directory "$here" --file "$here/compose.yaml" "$@"; }
check_models() {
  local tier base slug manifest path size expected actual
  for tier in tier1 tier2; do
    base=$QWEN_MODEL_STORAGE_PATH
    [[ $tier != tier2 ]] || base=$DEEPSEEK_MODEL_STORAGE_PATH
    slug=$(jq -r --arg t "$tier" '.models[$t].slug' "$spec")
    manifest="$base/$slug/.kairyu-model-attestation.json"
    [[ -f "$manifest" ]] || die "Missing existing pinned checkpoint attestation: $manifest"
    jq -e --arg t "$tier" --slurpfile s "$spec" \
      '(.repo == $s[0].models[$t].repo) and (.revision == $s[0].models[$t].revision)
       and (.tree_sha256 == $s[0].models[$t].tree_sha256)' "$manifest" >/dev/null \
      || die "$tier checkpoint pin differs"
    actual=$(printf '%s' "$(jq -cS '.files' "$manifest")" | sha256sum | cut -d' ' -f1)
    [[ $actual == "$(jq -r '.tree_sha256' "$manifest")" ]] || die "$tier manifest tree differs"
    while IFS=$'\t' read -r path size expected; do
      [[ $path != /* && $path != ../* && $path != */../* ]] || die 'Invalid checkpoint manifest path'
      [[ $(stat -c %s -- "$base/$slug/$path") == "$size" ]] || die "$tier checkpoint file size differs"
      if [[ $path != *.safetensors || ${VERIFY_MODEL:-0} == 1 ]]; then
        actual=$(sha256sum < "$base/$slug/$path"); actual=${actual%% *}
        [[ $actual == "$expected" ]] || die "$tier checkpoint file digest differs"
      fi
    done < <(jq -r '.files[] | [.path,.size,.sha256] | @tsv' "$manifest")
  done
}
provision_ui() {
  local tmp token state id=reasoning_effort
  tmp=$(mktemp -d); chmod 700 "$tmp"
  trap 'rm -rf -- "$tmp"' RETURN
  token=$(curl -fsS --json '{"email":"","password":""}' \
    "http://127.0.0.1:$CHAT_UI_PORT/api/v1/auths/signin" | jq -er 'select(.role=="admin") | .token')
  printf 'Authorization: Bearer %s\n' "$token" > "$tmp/headers"
  chmod 600 "$tmp/headers"; unset token
  jq -n --rawfile content "$here/../qwen3.8-deepseek-v4-8gpu/webui-reasoning-effort-filter.py" \
    --arg id "$id" '{id:$id,name:"Reasoning Effort",content:$content,meta:{description:"Select reasoning effort"}}' \
    > "$tmp/filter.json"
  if curl -fsS -H @"$tmp/headers" "http://127.0.0.1:$CHAT_UI_PORT/api/v1/functions/" \
      | jq -e --arg id "$id" 'any(.[]; .id==$id)' >/dev/null; then
    endpoint="id/$id/update"
  else
    endpoint=create
  fi
  curl -fsS -H @"$tmp/headers" -H 'Content-Type: application/json' --data-binary @"$tmp/filter.json" \
    "http://127.0.0.1:$CHAT_UI_PORT/api/v1/functions/$endpoint" > "$tmp/state.json"
  for state in active global; do
    if ! jq -e --arg field "is_$state" '.[$field] == true' "$tmp/state.json" >/dev/null; then
      endpoint="id/$id/toggle"; [[ $state != global ]] || endpoint+="/global"
      curl -fsS -X POST -H @"$tmp/headers" \
        "http://127.0.0.1:$CHAT_UI_PORT/api/v1/functions/$endpoint" > "$tmp/state.json"
    fi
  done
  rm -rf -- "$tmp"; trap - RETURN
}

case "$action" in
  up|build|provision-ui) : "${KAIRYU_RESPONSES_COMPACTION_SECRET:?Export a stable compaction secret before deployment}" ;;
  compose)
    case " $* " in *' up '*|*' start '*|*' restart '*)
      : "${KAIRYU_RESPONSES_COMPACTION_SECRET:?Export the existing stable compaction secret}" ;;
    esac ;;
esac
# Compose requires interpolation even for read-only config/status operations.
export KAIRYU_RESPONSES_COMPACTION_SECRET=${KAIRYU_RESPONSES_COMPACTION_SECRET:-render-only-placeholder}
case "$action" in
  check-models) check_models ;;
  build|up)
    check_models
    mkdir -p "$QWEN_CACHE_0_PATH" "$QWEN_CACHE_1_PATH" "$DEEPSEEK_CACHE_PATH" "$WEBUI_STORAGE_PATH"
    free=$(df -B1 --output=avail "$NVME_STORAGE_ROOT" | tail -1)
    ((free >= $(jq -r '.storage.minimum_free_gib' "$spec") * 1024**3)) || die 'Insufficient NVMe free space'
    docker image inspect --format '{{.Id}}' "$QWEN_VLLM_IMAGE" >/dev/null 2>&1 || docker pull "$QWEN_VLLM_IMAGE"
    if ! docker image inspect --format '{{.Id}}' "$DEEPSEEK_VLLM_IMAGE" >/dev/null 2>&1; then
      docker build --file "$here/$(jq -r '.vllm.deepseek.dockerfile' "$spec")" \
        --tag "$DEEPSEEK_VLLM_IMAGE" "$here"
    fi
    actual=$(docker image inspect --format '{{.Id}}' "$DEEPSEEK_VLLM_IMAGE")
    [[ $actual == "$(jq -r '.vllm.deepseek.image_id' "$spec")" ]] || die 'DeepSeek image ID differs; attest this build before deployment'
    compose build executor kairyu
    [[ $action != up ]] || { compose up --detach --wait --wait-timeout 1200; provision_ui; }
    ;;
  provision-ui) provision_ui ;;
  status) compose ps ;;
  down) compose down ;;
  logs) compose logs --follow --tail 200 "$@" ;;
  compose) compose "$@" ;;
  *) die 'Usage: run.sh {status|check-models|build|up|down|logs|provision-ui|compose ...}' ;;
esac
