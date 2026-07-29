#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

TP="${TP:-8}"
DCP="${DCP:-1}"
# Keep calibration and serving on the same ordered devices. Compose commonly
# leaves GPUS empty while supplying an intentional CUDA_VISIBLE_DEVICES order.
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export GPUS
DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-${VLLM_DCP_QUERY_SPLIT:-auto}}"
DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-${VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-auto}}"
DCP_CKV_GATHER="${DCP_CKV_GATHER:-${VLLM_B12X_MLA_CKV_GATHER:-auto}}"
DCP_CKV_GATHER_MAX_TOKENS="${DCP_CKV_GATHER_MAX_TOKENS:-\
${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:-140000}}"
DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE:-${VLLM_DCP_TOPK_OWNER_MERGE:-auto}}"
DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS:-${VLLM_DCP_INDEXER_SHARDS:-auto}}"
DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH:-auto}}"
DCP_CKV_PREFETCH_WORKSPACE_MIB="${DCP_CKV_PREFETCH_WORKSPACE_MIB:-${VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB:-1024}}"
DCP_CKV_PREFETCH_TOPOLOGY="${DCP_CKV_PREFETCH_TOPOLOGY:-auto}"
F8_DMA="${F8_DMA:-0}"
B12X_PCIE_DMA="${B12X_PCIE_DMA:-1}"
PCIE_CALIBRATION="${PCIE_CALIBRATION:-auto}"
PCIE_CALIBRATION_ONLY="${PCIE_CALIBRATION_ONLY:-0}"
PCIE_CALIBRATION_TIMEOUT="${PCIE_CALIBRATION_TIMEOUT:-600}"
PCIE_CALIBRATION_CACHE_DIR="${PCIE_CALIBRATION_CACHE_DIR:-${XDG_CACHE_HOME:-/cache}/pcie-calibration}"
PCIE_DMA_MIN_BYTES="${PCIE_DMA_MIN_BYTES:-${VLLM_PCIE_DMA_MIN_BYTES:-auto}}"
launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PCIE_CALIBRATOR="${PCIE_CALIBRATOR:-${launcher_dir}/glm52-pcie-calibration.py}"
# shellcheck source=glm52-dcp-prefill-policy.sh
source "${launcher_dir}/glm52-dcp-prefill-policy.sh"
# shellcheck source=glm52-pcie-runtime-env.sh
source "${launcher_dir}/glm52-pcie-runtime-env.sh"

case "${DCP_QUERY_SPLIT}" in
  auto|0|1) ;;
  *) die "DCP_QUERY_SPLIT must be auto, 0, or 1" ;;
esac
[[ "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}" == "auto" || \
   "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}" =~ ^[0-9]+$ ]] || \
  die "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS must be auto or a non-negative integer"

case "${DCP_CKV_GATHER}" in
  auto|0|1) ;;
  *) die "DCP_CKV_GATHER must be auto, 0, or 1" ;;
esac
[[ "${DCP_CKV_GATHER_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]] || \
  die "DCP_CKV_GATHER_MAX_TOKENS must be a positive integer"

case "${DCP_TOPK_OWNER_MERGE}" in
  auto|0|1) ;;
  *) die "DCP_TOPK_OWNER_MERGE must be auto, 0, or 1" ;;
esac

[[ "${DCP_INDEXER_SHARDS}" == "auto" || "${DCP_INDEXER_SHARDS}" =~ ^[0-9]+$ ]] || \
  die "DCP_INDEXER_SHARDS must be auto or a non-negative integer"
[[ "${DCP_CKV_PREFETCH_DEPTH}" == "auto" || "${DCP_CKV_PREFETCH_DEPTH}" =~ ^[0-9]+$ ]] || \
  die "DCP_CKV_PREFETCH_DEPTH must be auto or a non-negative integer"
[[ "${DCP_CKV_PREFETCH_WORKSPACE_MIB}" =~ ^[0-9]+$ ]] || \
  die "DCP_CKV_PREFETCH_WORKSPACE_MIB must be a non-negative integer"
case "${DCP_CKV_PREFETCH_TOPOLOGY}" in
  auto|safe|unsafe) ;;
  *) die "DCP_CKV_PREFETCH_TOPOLOGY must be auto, safe, or unsafe" ;;
esac
case "${PCIE_CALIBRATION}" in
  auto|off|force) ;;
  *) die "PCIE_CALIBRATION must be auto, off, or force" ;;
esac
case "${PCIE_CALIBRATION_ONLY}" in
  0|1) ;;
  *) die "PCIE_CALIBRATION_ONLY must be 0 or 1" ;;
esac
case "${F8_DMA}" in
  0|ag|ring|a2a|i8|i8_ring|i8_a2a|mx|mx_ring|mx_a2a) ;;
  *) die "F8_DMA is not a supported DMA wire mode: ${F8_DMA}" ;;
esac
case "${B12X_PCIE_DMA}" in
  0|1) ;;
  *) die "B12X_PCIE_DMA must be 0 or 1" ;;
esac
[[ "${PCIE_CALIBRATION_TIMEOUT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  die "PCIE_CALIBRATION_TIMEOUT must be a positive number"
[[ ! "${PCIE_CALIBRATION_TIMEOUT}" =~ ^0+([.]0+)?$ ]] || \
  die "PCIE_CALIBRATION_TIMEOUT must be a positive number"
if [[ "${PCIE_DMA_MIN_BYTES}" != "auto" && \
      "${PCIE_DMA_MIN_BYTES}" != "off" && \
      ! "${PCIE_DMA_MIN_BYTES}" =~ ^[0-9]+(K|KB|M|MB)?$ ]]; then
  die "PCIE_DMA_MIN_BYTES must be auto, off, bytes, or a K/KB/M/MB value"
fi
[[ "${PCIE_DMA_MIN_BYTES}" == "0" ]] && PCIE_DMA_MIN_BYTES=off

configure_glm52_pcie_runtime_env "${B12X_PCIE_DMA}" "${F8_DMA}"

# Resolve the indexer shard count needed by the standalone calibration. Zero
# means that vLLM uses all configured DCP shards.
read -r \
  calibration_default_query_split \
  calibration_effective_ckv_gather \
  _ \
  calibration_indexer_shards \
  _ < <(
  resolve_glm52_dcp_prefill_policy \
    "${TP}" "${DCP}" "${DCP_QUERY_SPLIT}" "${DCP_CKV_GATHER}" \
    auto "${DCP_INDEXER_SHARDS}" auto 1
)
if [[ "${calibration_indexer_shards}" == "0" ]]; then
  calibration_indexer_shards="${DCP}"
fi

calibration_status="not-requested"
calibration_cache=""
calibration_prefetch=""
calibration_query_split=""
calibration_query_split_min_context_tokens=""
calibration_dma_min_bytes=""
calibration_needed=0
if [[ "${DCP_CKV_PREFETCH_DEPTH}" == "auto" && \
      "${calibration_effective_ckv_gather}" == "1" ]]; then
  calibration_needed=1
fi
[[ "${DCP_QUERY_SPLIT}" == "auto" ]] && calibration_needed=1
if [[ "${DCP_QUERY_SPLIT}" != "0" && \
      "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}" == "auto" ]]; then
  calibration_needed=1
fi
[[ "${PCIE_DMA_MIN_BYTES}" == "auto" ]] && calibration_needed=1
calibration_supported=0
case "${TP}:${DCP}" in
  4:1|4:2|4:4|8:1|8:2|8:4|8:8) calibration_supported=1 ;;
esac

if [[ "${calibration_supported}" != "1" ]]; then
  calibration_status="skipped:unsupported-tp-dcp"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
  calibration_status="skipped:dry-run"
elif [[ "${PCIE_CALIBRATION}" == "off" ]]; then
  calibration_status="skipped:disabled"
elif [[ "${calibration_needed}" == "0" ]]; then
  calibration_status="skipped:all-explicit"
elif [[ "${B12X_PCIE_DMA}" != "1" ]]; then
  calibration_status="skipped:dma-disabled"
elif [[ "${F8_DMA}" != "0" ]]; then
  # Compressed wire modes are explicit quality/performance choices. A BF16
  # calibration must never make automatic decisions for those paths.
  calibration_status="skipped:explicit-compressed-dma"
else
  force_arg=()
  [[ "${PCIE_CALIBRATION}" == "force" ]] && force_arg=(--force)

  # Calibrator exit contract: 0 measured/cache-hit; 3 clean preflight skip
  # (busy GPUs, nvidia-smi unavailable); 4 probe failure with the complete
  # untruncated output parked in <cache-dir>/<digest>.failure.log. For 3
  # and 4 the calibrator prints exactly one stderr line; no traceback or
  # torchrun error wall ever reaches the boot log.
  calibration_line=""
  calibration_exit=0
  calibration_line="$(
    "${PCIE_CALIBRATOR}" \
      --tp-size "${TP}" \
      --dcp-size "${DCP}" \
      --indexer-shards "${calibration_indexer_shards}" \
      --gpus "${GPUS}" \
      --cache-dir "${PCIE_CALIBRATION_CACHE_DIR}" \
      --timeout "${PCIE_CALIBRATION_TIMEOUT}" \
      "${force_arg[@]}"
  )" || calibration_exit=$?
  if [[ "${calibration_exit}" -eq 0 ]]; then
    calibration_line="$(awk 'NF { line=$0 } END { print line }' \
      <<<"${calibration_line}")"
    IFS=$'\t' read -r \
      calibration_status \
      calibration_prefetch \
      calibration_query_split \
      calibration_query_split_min_context_tokens \
      calibration_dma_min_bytes \
      calibration_cache <<<"${calibration_line}"
    [[ "${calibration_prefetch}" =~ ^[01]$ ]] || \
      die "calibrator returned an invalid prefetch decision"
    [[ "${calibration_query_split}" =~ ^[01]$ ]] || \
      die "calibrator returned an invalid query-split decision"
    [[ "${calibration_query_split_min_context_tokens}" =~ ^[0-9]+$ ]] || \
      die "calibrator returned an invalid query-split context crossover"
    [[ "${calibration_dma_min_bytes}" == "off" || \
       "${calibration_dma_min_bytes}" =~ ^[1-9][0-9]*$ ]] || \
      die "calibrator returned an invalid DMA crossover"

    if [[ "${DCP_CKV_PREFETCH_DEPTH}" == "auto" && \
          "${calibration_effective_ckv_gather}" == "1" && \
          "${DCP_CKV_PREFETCH_TOPOLOGY}" == "auto" ]]; then
      DCP_CKV_PREFETCH_DEPTH="${calibration_prefetch}"
    fi
    if [[ "${DCP_QUERY_SPLIT}" == "auto" && \
          "${calibration_default_query_split}" == "1" ]]; then
      DCP_QUERY_SPLIT="${calibration_query_split}"
    fi
    if [[ "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}" == "auto" ]]; then
      if [[ "${DCP_QUERY_SPLIT}" == "1" ]]; then
        DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${calibration_query_split_min_context_tokens}"
      else
        DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0
      fi
    fi
    if [[ "${PCIE_DMA_MIN_BYTES}" == "auto" ]]; then
      PCIE_DMA_MIN_BYTES="${calibration_dma_min_bytes}"
    fi
  elif [[ "${calibration_exit}" -eq 3 ]]; then
    # Expected condition (for example GPUs still hold memory from a
    # previous instance); the calibrator already printed its one-line
    # reason. The conservative topology policy applies.
    calibration_status="skipped:preflight"
  else
    calibration_status="failed:fallback-to-topology"
    printf 'NOTICE: PCIe calibration unavailable (calibrator exit %s); using conservative topology policy.\n' \
      "${calibration_exit}" >&2
  fi
fi

# Preserve the previous vLLM crossover whenever measured calibration is not
# applicable. Explicit values still pass through unchanged.
[[ "${PCIE_DMA_MIN_BYTES}" == "auto" ]] && PCIE_DMA_MIN_BYTES=6MB
[[ "${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}" == "auto" ]] && \
  DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0

prefetch_overlap_safe=1
if [[ -n "${calibration_prefetch}" && \
      "${DCP_CKV_PREFETCH_TOPOLOGY}" == "auto" ]]; then
  prefetch_topology_decision="measured:${calibration_status}"
else
  prefetch_topology_decision="not-applicable"
fi
if [[ "${DCP_CKV_PREFETCH_DEPTH}" == "auto" && "${DCP}" != "1" ]]; then
  case "${DCP_CKV_PREFETCH_TOPOLOGY}" in
    safe)
      prefetch_overlap_safe=1
      prefetch_topology_decision="safe:explicit-override"
      ;;
    unsafe)
      prefetch_overlap_safe=0
      prefetch_topology_decision="unsafe:explicit-override"
      ;;
    auto)
      topology_output=""
      if command -v nvidia-smi >/dev/null 2>&1; then
        topology_output="$(nvidia-smi topo -m 2>/dev/null || true)"
      fi
      if [[ -n "${topology_output}" ]]; then
        prefetch_topology_decision="$({
          printf '%s\n' "${topology_output}"
        } | classify_glm52_ckv_prefetch_topology "${TP}" "${DCP}" "${GPUS}")"
      else
        prefetch_topology_decision="unsafe:topology-unavailable"
      fi
      if [[ "${prefetch_topology_decision}" == safe:* ]]; then
        prefetch_overlap_safe=1
      else
        prefetch_overlap_safe=0
      fi
      ;;
  esac
fi

read -r \
  DCP_QUERY_SPLIT \
  DCP_CKV_GATHER \
  DCP_TOPK_OWNER_MERGE \
  DCP_INDEXER_SHARDS \
  DCP_CKV_PREFETCH_DEPTH < <(
    resolve_glm52_dcp_prefill_policy \
      "${TP}" \
      "${DCP}" \
      "${DCP_QUERY_SPLIT}" \
      "${DCP_CKV_GATHER}" \
      "${DCP_TOPK_OWNER_MERGE}" \
      "${DCP_INDEXER_SHARDS}" \
      "${DCP_CKV_PREFETCH_DEPTH}" \
      "${prefetch_overlap_safe}"
  )

printf 'GLM-5.2 DCP CKV prefetch topology: %s\n' \
  "${prefetch_topology_decision}" >&2

export VLLM_DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT}"
if [[ "${DCP_QUERY_SPLIT}" == "0" ]]; then
  DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0
fi
export VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}"
export VLLM_B12X_MLA_CKV_GATHER="${DCP_CKV_GATHER}"
export VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS="${DCP_CKV_GATHER_MAX_TOKENS}"
export VLLM_DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE}"
export VLLM_DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS}"
export VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH}"
export VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB="${DCP_CKV_PREFETCH_WORKSPACE_MIB}"
export VLLM_PCIE_DMA_MIN_BYTES="${PCIE_DMA_MIN_BYTES}"

printf 'GLM-5.2 PCIe calibration: %s cache=%s DMA-min=%s\n' \
  "${calibration_status}" "${calibration_cache:-none}" \
  "${VLLM_PCIE_DMA_MIN_BYTES}" >&2
printf 'GLM-5.2 full CKV gather capacity: %s tokens\n' \
  "${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS}" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'VLLM_DCP_QUERY_SPLIT=%q\n' "${VLLM_DCP_QUERY_SPLIT}"
  printf 'VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=%q\n' \
    "${VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}"
  printf 'VLLM_B12X_MLA_CKV_GATHER=%q\n' "${VLLM_B12X_MLA_CKV_GATHER}"
  printf 'VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=%q\n' \
    "${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS}"
  printf 'VLLM_DCP_TOPK_OWNER_MERGE=%q\n' "${VLLM_DCP_TOPK_OWNER_MERGE}"
  printf 'VLLM_DCP_INDEXER_SHARDS=%q\n' "${VLLM_DCP_INDEXER_SHARDS}"
  printf 'VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=%q\n' "${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH}"
  printf 'VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB=%q\n' "${VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB}"
  printf 'VLLM_PCIE_DMA_MIN_BYTES=%q\n' "${VLLM_PCIE_DMA_MIN_BYTES}"
  printf 'DCP_CKV_PREFETCH_TOPOLOGY_DECISION=%q\n' "${prefetch_topology_decision}"
  printf 'PCIE_CALIBRATION_STATUS=%q\n' "${calibration_status}"
fi

if [[ "${PCIE_CALIBRATION_ONLY}" == "1" ]]; then
  printf 'VLLM_DCP_QUERY_SPLIT=%q\n' "${VLLM_DCP_QUERY_SPLIT}"
  printf 'VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=%q\n' \
    "${VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS}"
  printf 'VLLM_B12X_MLA_CKV_GATHER=%q\n' "${VLLM_B12X_MLA_CKV_GATHER}"
  printf 'VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=%q\n' \
    "${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS}"
  printf 'VLLM_DCP_TOPK_OWNER_MERGE=%q\n' "${VLLM_DCP_TOPK_OWNER_MERGE}"
  printf 'VLLM_DCP_INDEXER_SHARDS=%q\n' "${VLLM_DCP_INDEXER_SHARDS}"
  printf 'VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=%q\n' \
    "${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH}"
  printf 'VLLM_PCIE_DMA_MIN_BYTES=%q\n' "${VLLM_PCIE_DMA_MIN_BYTES}"
  printf 'PCIE_CALIBRATION_STATUS=%q\n' "${calibration_status}"
  printf 'PCIE_CALIBRATION_CACHE=%q\n' "${calibration_cache}"
  exit 0
fi

exec /usr/local/bin/serve-glm52-v16.sh "$@"
