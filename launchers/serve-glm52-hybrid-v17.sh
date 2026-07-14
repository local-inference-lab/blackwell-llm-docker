#!/usr/bin/env bash
set -euo pipefail

# TP4 preset for madeby561's mixed NVFP4/NF3 checkpoint. Override only the
# deployment envelope (GPU set, DCP, MTP, context, batching, and port).
export MODEL="${MODEL:-madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
export MODEL_REVISION="${MODEL_REVISION:-68babde27a97a4c980c2494e830dd424975cd5a3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
export GPUS="${GPUS:-0,1,2,3}"
export TP="${TP:-4}"
export DCP="${DCP:-1}"
export DCP_BACKEND="${DCP_BACKEND:-a2a}"
export DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-16}"
export DCP_A2A_LARGE_BACKEND="${DCP_A2A_LARGE_BACKEND:-ag_rs}"
export MTP="${MTP:-0}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export GRAPH="${GRAPH:-64}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
export MOE_MODE="${MOE_MODE:-a16}"
export QUANTIZATION="${QUANTIZATION:-nvfp4_nf3_hybrid}"
export ONLINE_QUANT="${ONLINE_QUANT:-mxfp8}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
export LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"

exec /usr/local/bin/serve-glm52-v16.sh "$@"
