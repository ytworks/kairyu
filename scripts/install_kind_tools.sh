#!/usr/bin/env bash
set -euo pipefail

: "${KIND_VERSION:?KIND_VERSION is required}"
: "${KUBECTL_VERSION:?KUBECTL_VERSION is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${GITHUB_PATH:?GITHUB_PATH is required}"

CURL=${CURL:-curl}
SHA256SUM=${SHA256SUM:-sha256sum}
TOOLS_DIR="${RUNNER_TEMP}/kairyu-tools"
mkdir -p "$TOOLS_DIR"

download() {
  local url=$1
  local output=$2
  local partial="${output}.download"
  local attempt

  for attempt in 1 2 3; do
    if "$CURL" \
      --fail --show-error --silent --location \
      --proto '=https' --proto-redir '=https' --tlsv1.2 \
      --connect-timeout 15 --max-time 120 \
      --retry 4 --retry-delay 2 --retry-max-time 90 --retry-all-errors \
      "$url" --output "$partial"; then
      mv -- "$partial" "$output"
      return 0
    fi
    echo "download attempt ${attempt}/3 failed: ${url}" >&2
  done
  return 1
}

verify() {
  local checksum_file=$1
  local binary=$2
  local checksum

  checksum=$(awk 'NF {print $1; exit}' "$checksum_file")
  if [[ ! "$checksum" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "invalid SHA-256 checksum in ${checksum_file}" >&2
    return 1
  fi
  printf '%s  %s\n' "$checksum" "$binary" | "$SHA256SUM" --check
}

KIND_BASE_URL="https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}"
KUBECTL_BASE_URL="https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64"

download "${KIND_BASE_URL}/kind-linux-amd64.sha256sum" "${TOOLS_DIR}/kind.sha256"
download "${KIND_BASE_URL}/kind-linux-amd64" "${TOOLS_DIR}/kind"
verify "${TOOLS_DIR}/kind.sha256" "${TOOLS_DIR}/kind"

download "${KUBECTL_BASE_URL}/kubectl.sha256" "${TOOLS_DIR}/kubectl.sha256"
download "${KUBECTL_BASE_URL}/kubectl" "${TOOLS_DIR}/kubectl"
verify "${TOOLS_DIR}/kubectl.sha256" "${TOOLS_DIR}/kubectl"

chmod +x "${TOOLS_DIR}/kind" "${TOOLS_DIR}/kubectl"
kind_version_output=$("${TOOLS_DIR}/kind" version)
kubectl_version_output=$("${TOOLS_DIR}/kubectl" version --client=true)
if [[ "$kind_version_output" != "kind ${KIND_VERSION} "* ]]; then
  echo "unexpected kind version: ${kind_version_output}" >&2
  exit 1
fi
if [[ "$kubectl_version_output" != *"Client Version: ${KUBECTL_VERSION}"* ]]; then
  echo "unexpected kubectl version: ${kubectl_version_output}" >&2
  exit 1
fi
printf '%s\n' "$kind_version_output" "$kubectl_version_output"
printf '%s\n' "$TOOLS_DIR" >> "$GITHUB_PATH"
