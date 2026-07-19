#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

TP="${TP:-8}"
DCP="${DCP:-1}"
DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-${VLLM_DCP_QUERY_SPLIT:-auto}}"
DCP_CKV_GATHER="${DCP_CKV_GATHER:-${VLLM_B12X_MLA_CKV_GATHER:-auto}}"

case "${DCP_QUERY_SPLIT}" in
  auto|0|1) ;;
  *) die "DCP_QUERY_SPLIT must be auto, 0, or 1" ;;
esac

case "${DCP_CKV_GATHER}" in
  auto|0|1) ;;
  *) die "DCP_CKV_GATHER must be auto, 0, or 1" ;;
esac

# The full-CKV/query-split stack is faster when each rank has a naturally
# 8-head-aligned local shard. TP4 and TP8 meet that contract. Virtual TP6 has
# 11 local heads; measured 64k prefill regressed by 27% with full-CKV and 32%
# with query split, so its faster borrowed-workspace path remains the default.
case "${TP}:${DCP}" in
  4:2|4:4|8:2|8:4|8:8)
    [[ "${DCP_QUERY_SPLIT}" == "auto" ]] && DCP_QUERY_SPLIT=1
    [[ "${DCP_CKV_GATHER}" == "auto" ]] && DCP_CKV_GATHER=1
    ;;
  *)
    [[ "${DCP_QUERY_SPLIT}" == "auto" ]] && DCP_QUERY_SPLIT=0
    [[ "${DCP_CKV_GATHER}" == "auto" ]] && DCP_CKV_GATHER=0
    ;;
esac

export VLLM_DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT}"
export VLLM_B12X_MLA_CKV_GATHER="${DCP_CKV_GATHER}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'VLLM_DCP_QUERY_SPLIT=%q\n' "${VLLM_DCP_QUERY_SPLIT}"
  printf 'VLLM_B12X_MLA_CKV_GATHER=%q\n' "${VLLM_B12X_MLA_CKV_GATHER}"
fi

exec /usr/local/bin/serve-glm52-v16.sh "$@"
