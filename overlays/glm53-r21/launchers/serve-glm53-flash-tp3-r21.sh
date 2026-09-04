#!/usr/bin/env bash
set -euo pipefail

# Strict GLM-5.3-Flash TP3 runtime policy (R21 launcher chain).
#
# This launcher mirrors the semantic intent of the TP3 runtime work done on
# older bases (PR #30 / the R17-era TP3 child) but is built on the R21
# cache-complete chain. It is fail-closed: every value that shapes the TP3
# runtime is locked, and caller overrides are rejected before startup.
# Hardware qualification covers ordinary decoding, MTP3, and DFlash2 K7 at
# the locked one-million-token envelope below.

readonly capture_launcher=/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[1-9][0-9]*$ ]] ||
    fail "${name} must be a positive integer; got ${value}"
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${name} must be a non-negative integer; got ${value}"
}

# TP=3 is the entire point of this launcher; reject anything else up front.
case "${TP:-}" in
  3) ;;
  *) fail "R21 TP3 policy requires TP=3; got ${TP:-unset}" ;;
esac

case ${CACHE_MODE:-vram} in
  vram)
    if [[ ${LMCACHE_ENABLED:-0} != 0 ]]; then
      fail 'R21 TP3 dense-cache policy does not support LMCache'
    fi
    ;;
  native | lmcache)
    fail "R21 TP3 policy supports dense GPU cache only; got CACHE_MODE=${CACHE_MODE}"
    ;;
  *)
    fail "CACHE_MODE must be vram, native, or lmcache; got ${CACHE_MODE}"
    ;;
esac

if [[ ${LMCACHE_ENABLED:-0} != 0 ]]; then
  fail 'R21 TP3 dense-cache policy does not support LMCache'
fi
if [[ ${DCP:-1} != 1 ]]; then
  fail "R21 TP3 policy requires DCP=1; got ${DCP:-unset}"
fi
if [[ ${MM_ENCODER_TP_MODE:-weights} != weights ]]; then
  fail "R21 TP3 requires MM_ENCODER_TP_MODE=weights; got ${MM_ENCODER_TP_MODE:-unset}"
fi
if (($# > 0)) && [[ $1 != -* ]]; then
  fail 'R21 TP3 target model is locked; positional model overrides are forbidden'
fi
if [[ -v VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE ||
      -v VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE ]]; then
  fail 'R21 TP3 dense cache rejects split-page block-size variables'
fi

# The public dispatcher canonicalizes these aliases. Repeat the normalization
# here so direct invocation of this strict launcher cannot widen the qualified
# speculative envelope.
tp3_speculator=${SPECULATOR:-mtp}
case "${tp3_speculator}" in
  mtp)
    tp3_speculative_tokens=${MTP_DEPTH:-${MTP:-${NUM_SPECULATIVE_TOKENS:-0}}}
    require_nonnegative_integer MTP_DEPTH "${tp3_speculative_tokens}"
    case "${tp3_speculative_tokens}" in
      0 | 3) ;;
      *) fail "R21 TP3 MTP depth must be 0 or 3; got ${tp3_speculative_tokens}" ;;
    esac
    ;;
  dflash | dflash2)
    tp3_speculator=dflash2
    tp3_speculative_tokens=${DFLASH_DEPTH:-${NUM_SPECULATIVE_TOKENS:-7}}
    require_positive_integer DFLASH_DEPTH "${tp3_speculative_tokens}"
    [[ ${tp3_speculative_tokens} == 7 ]] ||
      fail "R21 TP3 DFlash2 depth must be 7; got ${tp3_speculative_tokens}"
    ;;
  *) fail "R21 TP3 SPECULATOR must be mtp or dflash2; got ${tp3_speculator}" ;;
esac
export SPECULATOR=${tp3_speculator}
export NUM_SPECULATIVE_TOKENS=${tp3_speculative_tokens}

# The qualified TP3 profile accepts no caller CLI arguments. Both wrapper
# layers append mandatory engine arguments, so even an end-of-options marker
# could hide policy values from vLLM. Supported mode selection is environment
# based and normalized by the public dispatcher before this point.
if (($# > 0)); then
  fail "R21 TP3 policy rejects caller option ${1%%=*}"
fi

lock_env() {
  local name=$1 expected=$2
  if [[ -v ${name} && ${!name} != "${expected}" ]]; then
    fail "R21 TP3 ${name} is locked to ${expected}; got ${!name}"
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}

lock_env_from_parent() {
  local name=$1 expected=$2 parent_default=$3
  if [[ -v ${name} &&
        ${!name} != "${expected}" &&
        ${!name} != "${parent_default}" ]]; then
    fail "R21 TP3 ${name} is locked to ${expected}; got ${!name}"
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}

require_unset_env() {
  local name=$1
  if [[ -v ${name} ]]; then
    fail "R21 TP3 ${name} must be unset; got ${!name}"
  fi
}

readonly locked_model=local-inference-lab/GLM-5.3-Flash-NVFP4
readonly locked_dflash_model=local-inference-lab/GLM-5.3-Flash-DFlash2
readonly locked_model_revision=378ca54585c46542bad1f3cb3ed0d73ae51cdb62
readonly locked_dflash_revision=aea0ac8a05624512ca9e106c09c16087da998426
lock_env TP 3
lock_env DCP 1
require_unset_env VLLM_DP_SIZE
require_unset_env VLLM_DP_RANK
require_unset_env VLLM_DP_RANK_LOCAL
require_unset_env VLLM_DP_MASTER_IP
require_unset_env VLLM_DP_MASTER_PORT
lock_env MM_ENCODER_TP_MODE weights
lock_env MODEL "${locked_model}"
lock_env DFLASH_MODEL "${locked_dflash_model}"
lock_env MODEL_REVISION "${locked_model_revision}"
lock_env DFLASH_MODEL_REVISION "${locked_dflash_revision}"
lock_env LMCACHE_ENABLED 0
lock_env GLM53_CACHE_LAYOUT dense
lock_env CP_KV_CACHE_INTERLEAVE_SIZE 4
lock_env DCP_CKV_GATHER 0
lock_env MAX_MODEL_LEN 1048576
lock_env_from_parent MAX_NUM_SEQS 8 32
lock_env_from_parent MAX_NUM_BATCHED_TOKENS 8192 4096
lock_env_from_parent PREFILL_SCHEDULE_INTERVAL 8 1
lock_env MAX_NUM_PREFILL_TOKENS_PER_STEP 0
lock_env MAX_NUM_PARTIAL_PREFILLS 0
lock_env DECODE_PREFILL_MIN_DECODE_STEPS 0
lock_env DECODE_PREFILL_MAX_WAIT_MS 0
lock_env GPU_MEMORY_UTILIZATION 0.91
lock_env_from_parent MAX_CUDAGRAPH_CAPTURE_SIZE 16 256
lock_env_from_parent CUDAGRAPH_CAPTURE_SIZES '1 2 4 8 16' \
  '1 2 4 8 16 32 40 48 64 96 128 192 256'
lock_env_from_parent FAIRNESS_ENGINE none compute_share
lock_env_from_parent PREFILL_COMPUTE_SHARE none 0.4
lock_env KV_CACHE_DTYPE fp8
lock_env LOAD_FORMAT instanttensor
require_unset_env KV_CACHE_QUANT
require_unset_env VLLM_KV_CACHE_LAYOUT
require_unset_env VLLM_SSM_CONV_STATE_LAYOUT
require_unset_env GLM53_TARGET_BLOCK_SIZE
lock_env NCCL_IB_DISABLE 1
lock_env NCCL_P2P_LEVEL SYS
lock_env NCCL_P2P_DISABLE 0
lock_env NCCL_CUMEM_ENABLE 0
lock_env NCCL_PROTO LL,LL128,Simple
lock_env NCCL_MIN_NCHANNELS 16
lock_env NCCL_MAX_NCHANNELS 16
lock_env NCCL_BUFFSIZE 2097152
lock_env NCCL_NET_PLUGIN none
lock_env NCCL_TUNER_PLUGIN none
require_unset_env NCCL_ALGO
require_unset_env NCCL_COLLNET_ENABLE
require_unset_env NCCL_NVLS_ENABLE
require_unset_env NCCL_SHM_DISABLE
require_unset_env NCCL_PXN_DISABLE
require_unset_env NCCL_P2P_DIRECT_DISABLE
require_unset_env GLM53_MAMBA_BLOCK_SIZE
lock_env ATTENTION_BACKEND B12X
lock_env VLLM_DISABLE_PYNCCL 0
lock_env VLLM_ALLREDUCE_USE_SYMM_MEM 1
lock_env MOE_BACKEND auto
lock_env LINEAR_BACKEND b12x
lock_env MTP_ATTENTION_BACKEND B12X
lock_env MTP_MOE_BACKEND humming
lock_env DFLASH_ATTENTION_BACKEND FLASH_ATTN
lock_env DFLASH_KV_CACHE_DTYPE auto
lock_env B12X_POLICY_MODE auto
lock_env VLLM_B12X_MOE_FP4_FORCE_A16 0
lock_env B12X_PCIE_ALLREDUCE 1
lock_env VLLM_ENABLE_PCIE_ALLREDUCE 1
lock_env VLLM_PCIE_ALLREDUCE_BACKEND b12x
lock_env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE 84KB
lock_env VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE 84KB
lock_env VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE 768KB
lock_env VLLM_PCIE_DMA_MIN_BYTES 6MB
lock_env B12X_PCIE_ONESHOT_THREADS 512
lock_env B12X_PCIE_ONESHOT_BLOCK_LIMIT 4
lock_env B12X_PCIE_ONESHOT_PUSH 0
lock_env B12X_PCIE_FUSED_THREADS 256
lock_env B12X_PCIE_FUSED_CTAS_PER_ROW 0
lock_env B12X_PCIE_DMA_PIECES 0
lock_env B12X_PCIE_ONESHOT_PDL 1
lock_env B12X_MHC_PDL 1
lock_env VLLM_CPP_AR_1STAGE_NCCL_CUTOFF 56KB
lock_env VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS 0
require_unset_env VLLM_PCIE_DMA_FP8
require_unset_env B12X_PCIE_DMA_FP8
lock_env B12X_PCIE_ALLREDUCE_ALGORITHM auto
lock_env GLM53_KDA_DECODE_BACKEND b12x
lock_env GLM53_KDA_PREFILL_BACKEND flashkda
lock_env CUDAGRAPH_MODE FULL
lock_env ENABLE_PREFIX_CACHING 1
lock_env GLM53_R17_REQUIRE_RUNTIME_PROOF 1

require_positive_integer CP_KV_CACHE_INTERLEAVE_SIZE \
  "${CP_KV_CACHE_INTERLEAVE_SIZE}"
require_positive_integer MAX_NUM_SEQS "${MAX_NUM_SEQS}"
require_positive_integer MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS}"
require_positive_integer PREFILL_SCHEDULE_INTERVAL \
  "${PREFILL_SCHEDULE_INTERVAL}"
case "${GPU_MEMORY_UTILIZATION}" in
  0.91) ;;
  *) fail "R21 TP3 GPU_MEMORY_UTILIZATION is locked to 0.91; got ${GPU_MEMORY_UTILIZATION}" ;;
esac

# Dense (GPU-only) KV cache: dense target and auto recurrent pages, exactly
# the shape the base launcher resolves on its own for DCP=1. No external cache
# connector and no split-page geometry overrides.
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto
export LMCACHE_VLLM_KV_CACHE_DTYPE=fp8
export LMCACHE_KV_CACHE_DTYPE=fp8_ds_mla

# The dense-cache fingerprint separates the TP3 compiled artifacts from the
# qualified TP4/TP8 caches so a TP3 warmup never invalidates the R21 layout.
readonly fingerprint=cu133-torch213-glm53-r21-tp3-vllme96b18db-b12x6d47b10e-dense-ctx1m-seq8-bt8192
readonly cache_root=/cache/jit/${fingerprint}
export LOCAL_INFERENCE_CACHE_FINGERPRINT=${fingerprint}
export XDG_CACHE_HOME=${cache_root}
export VLLM_CACHE_ROOT=${cache_root}/vllm
export VLLM_CACHE_DIR=${cache_root}/vllm
export TRITON_CACHE_DIR=${cache_root}/triton
export TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions
export TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${cache_root}/flashinfer-autotune
export FLASHINFER_WORKSPACE_BASE=${cache_root}/flashinfer
export TVM_FFI_CACHE_DIR=${cache_root}/tvm-ffi
export TVM_CACHE_DIR=${cache_root}/tvm
export TILELANG_CACHE_DIR=${cache_root}/tilelang
export CUTE_DSL_CACHE_DIR=${cache_root}/cute-dsl
export B12X_CUTE_COMPILE_CACHE_DIR=${cache_root}/b12x/cute
export B12X_COMPILE_CACHE_DIR=${cache_root}/b12x/compile
export SPARKINFER_COMPILE_CACHE_DIR=${cache_root}/b12x/compile
export DG_JIT_CACHE_DIR=${cache_root}/deep-gemm
export MM_SPARSE_ATTN_AOT_CACHE=${cache_root}/minfer/mm-sparse-attn
export MINFER_FMHA_CACHE_DIR=${cache_root}/minfer/fmha-sm120
export NUMBA_CACHE_DIR=${cache_root}/numba
export CUDA_CACHE_PATH=${cache_root}/cuda
export CUPY_CACHE_DIR=${cache_root}/cupy

exec "${capture_launcher}" "$@" \
  --enable-expert-parallel \
  --mm-encoder-tp-mode weights
