#!/usr/bin/env bash
set -euo pipefail

# GPU-local GLM Spark profile qualified on two 96 GiB RTX PRO 6000 GPUs.
# The scheduler launcher owns vLLM argument rendering and CLI precedence.
export MODEL=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4-Spark}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4-Spark}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8000}
export TP=${TP:-2} DCP=${DCP:-2} SPECULATOR=${SPECULATOR:-mtp}
export NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-3072}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4} MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
export CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
export CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 12 16'}
export LOAD_FORMAT=${LOAD_FORMAT:-safetensors}
export MTP_MOE_BACKEND=${MTP_MOE_BACKEND:-b12x}
export MOE_BACKEND=${MOE_BACKEND:-b12x} LINEAR_BACKEND=${LINEAR_BACKEND:-b12x}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_ds_mla}
export VLLM_LM_HEAD_A16=${VLLM_LM_HEAD_A16:-1}
export VLLM_GLM53_MTP_DRAFT_HEAD=${VLLM_GLM53_MTP_DRAFT_HEAD:-nvfp4}
export VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:-65536}
export GLM53_KDA_PREFILL_BACKEND=${GLM53_KDA_PREFILL_BACKEND:-b12x}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-2}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,large_segment_size_mb:12}

if [[ ${TP} != 2 || ${DCP} != 2 || ${SPECULATOR} != mtp ]]; then
  echo 'This experimental entrypoint requires TP=2, DCP=2 and SPECULATOR=mtp. Other profiles are not qualified.' >&2
  exit 2
fi
if [[ ${CACHE_MODE:-vram} != vram ]]; then
  echo 'This TP2 profile qualifies GPU-local cache only, not LMCache.' >&2
  exit 2
fi
if [[ $# == 1 && ($1 == --help || $1 == -h) ]]; then
  printf '%s\n' 'GLM Spark TP2/DCP2 experimental profile: MTP3, FP8 KV, vision, 1M context, batch 3072, four request slots.
MODEL accepts a Hugging Face repository or a mounted checkpoint. PORT defaults to 8000; HOST defaults to 0.0.0.0.
KV_CACHE_MEMORY_BYTES defaults to 4294967296 per GPU. Explicit vLLM CLI options take precedence.
DRY_RUN=1 prints the complete command without loading weights. VRAM overclocking is never applied by this launcher.'
  exit 0
fi
has_option() {
  local option=$1 argument
  shift
  for argument in "$@"; do
    [[ $argument == "$option" || $argument == "$option="* ]] && return 0
  done
  return 1
}
args=("$@")
has_option --kv-cache-memory-bytes "$@" || args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-4294967296}")
has_option --limit-mm-per-prompt "$@" || args+=(--limit-mm-per-prompt '{"image":1,"video":0}')
has_option --recurrent-checkpoint-policy "$@" || args+=(--recurrent-checkpoint-policy request_boundaries)
exec /usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh "${args[@]}"
