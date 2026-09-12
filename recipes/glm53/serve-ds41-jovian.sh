#!/usr/bin/env bash
set -euo pipefail

# Use the source-locked native DS4.1 policy, without inherited GLM kernel tuning.
for name in ${!VLLM_@}; do
    case "$name" in
        VLLM_SOURCE_DIR|VLLM_NCCL_SO_PATH|VLLM_CACHE_ROOT|VLLM_CACHE_DIR) ;;
        *) unset "$name" ;;
    esac
done
for name in ${!B12X_@}; do
    case "$name" in
        B12X_CUTE_COMPILE_CACHE_DIR|B12X_COMPILE_CACHE_DIR) ;;
        *) unset "$name" ;;
    esac
done
unset NCCL_GRAPH_FILE

export PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
export MODEL_PATH=${MODEL_PATH:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8000}
export TP_SIZE=${TP_SIZE:-4}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
export LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
export ENGRAM_TABLE_MEMORY=${ENGRAM_TABLE_MEMORY:-disk}
export VLLM_HOST_IP=127.0.0.1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-16}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
case "$ENGRAM_TABLE_MEMORY" in
    ram|disk) ;;
    *) echo 'ENGRAM_TABLE_MEMORY must be ram or disk.' >&2; exit 2 ;;
esac

generation_args=(--override-generation-config '{"temperature":1.0,"top_p":0.95}')
for argument in "$@"; do
    case "${argument//_/-}" in
        --generation-config|--generation-config=*|--override-generation-config|\
        --override-generation-config=*|--override-generation-config.*|--config|--config=*)
            generation_args=() ;;
    esac
done
exec "${VLLM_SOURCE_DIR:-/opt/glm53-flash/vllm}/serve-ds41-flash.sh" \
    --decode-context-parallel-size "${DCP_SIZE:-1}" \
    --kv-cache-dtype fp8 --enable-chunked-prefill \
    --attention-backend B12X \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    "${generation_args[@]}" "$@"
