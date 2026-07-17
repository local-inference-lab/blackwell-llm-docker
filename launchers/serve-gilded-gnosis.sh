#!/usr/bin/env bash
set -euo pipefail

# The unified image targets a single PCIe host. Multi-node deployments can
# override both interfaces explicitly.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
printf 'Process-group interfaces: GLOO_SOCKET_IFNAME=%s NCCL_SOCKET_IFNAME=%s\n' \
  "${GLOO_SOCKET_IFNAME}" "${NCCL_SOCKET_IFNAME}"

case "${MODEL_FAMILY:-}" in
  glm52|glm5.2|glm)
    exec /usr/local/bin/serve-glm52-v18.sh "$@"
    ;;
  glm52-hybrid|nf3)
    exec /usr/local/bin/serve-glm52-hybrid-v18.sh "$@"
    ;;
  ds4|ds4-flash|dspark)
    exec /usr/local/bin/serve-ds4-flash.sh "$@"
    ;;
  *)
    echo "ERROR: MODEL_FAMILY must be glm52, glm52-hybrid, or ds4" >&2
    exit 2
    ;;
esac
