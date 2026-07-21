#!/usr/bin/env bash
set -euo pipefail

# SparkInfer was formerly named B12X. Keep the legacy Docker build arguments
# internal so existing cache layers and downstream recipes remain compatible.
export B12X_REPO="${SPARKINFER_REPO:-${B12X_REPO:-https://github.com/local-inference-lab/sparkinfer.git}}"
export B12X_REF="${SPARKINFER_REF:-${B12X_REF:-master}}"
export B12X_COMMIT="${SPARKINFER_COMMIT:-${B12X_COMMIT:-}}"

exec "$(dirname "$0")/build-vllm-b12x-cu132.sh" "$@"
