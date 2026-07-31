#!/usr/bin/env bash

# Keep the PCIe/NCCL environment identical between the pre-model calibration
# and the eventual vLLM process. Callers validate both arguments first.
configure_glm52_pcie_runtime_env() {
  local pcie_dma_enabled="$1"
  local dma_wire_mode="$2"
  local nccl_path="${NCCL_LOCAL_INFERENCE_PATH:-/opt/libnccl-local-inference.so.2.30.4}"
  local current_preload="${LD_PRELOAD:-}"
  local preload_entries

  export VLLM_ENABLE_PCIE_ALLREDUCE=1
  export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
  export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
  export VLLM_USE_B12X_PCIE_DMA="${pcie_dma_enabled}"
  export VLLM_PCIE_DMA_FP8="${dma_wire_mode}"
  export B12X_PCIE_DMA_FP8="${dma_wire_mode}"
  export SPARKINFER_PCIE_DMA_FP8="${dma_wire_mode}"

  export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
  export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  if [[ -f "${nccl_path}" ]]; then
    preload_entries=" ${current_preload//:/ } "
    case "${preload_entries}" in
      *" ${nccl_path} "*) ;;
      *) current_preload="${nccl_path}${current_preload:+:${current_preload}}" ;;
    esac
    export LD_PRELOAD="${current_preload}"
    export VLLM_NCCL_SO_PATH="${nccl_path}"
  fi
}
