#!/usr/bin/env bash
set -euo pipefail

case "${MODEL_FAMILY:-}" in
  glm52|glm5.2|glm)
    exec /usr/local/bin/serve-glm52-v16.sh "$@"
    ;;
  ds4|ds4-flash|dspark)
    exec /usr/local/bin/serve-ds4-flash.sh "$@"
    ;;
  *)
    echo "ERROR: MODEL_FAMILY must be glm52 or ds4" >&2
    exit 2
    ;;
esac
